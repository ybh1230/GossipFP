import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class GossipPrototypeBank(nn.Module):
    """Reliability-aware class gossip memory.

    Each semantic class is a node in a dynamic graph. Nodes update their own
    prototypes, estimate teacher reliability, build normalized class-relation
    edges, and exchange information through lazy gossip diffusion.
    """

    def __init__(
        self,
        num_classes,
        feat_dim,
        momentum=0.99,
        confusion_momentum=0.95,
        topk=3,
        temperature=0.2,
        similarity_weight=1.0,
        confusion_weight=0.5,
        min_pixels=16,
        margin=0.2,
        ignore_label=255,
        diffusion_rounds=2,
        self_loop=0.55,
        min_reliability=0.35,
        reliability_warmup=256.0,
        epsilon_min_ratio=0.35,
        risk_gamma=1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.confusion_momentum = float(confusion_momentum)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.similarity_weight = float(similarity_weight)
        self.confusion_weight = float(confusion_weight)
        self.min_pixels = int(min_pixels)
        self.margin = float(margin)
        self.ignore_label = int(ignore_label)
        self.diffusion_rounds = int(diffusion_rounds)
        self.self_loop = float(self_loop)
        self.min_reliability = float(min_reliability)
        self.reliability_warmup = float(reliability_warmup)
        self.epsilon_min_ratio = float(epsilon_min_ratio)
        self.risk_gamma = float(risk_gamma)

        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer("counts", torch.zeros(self.num_classes))
        self.register_buffer("seen", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("confidence", torch.zeros(self.num_classes))
        self.register_buffer("reliability", torch.zeros(self.num_classes))
        self.register_buffer("confusion", torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer("last_risk", torch.zeros(self.num_classes))
        self.register_buffer("last_adjacency", torch.zeros(self.num_classes, self.num_classes))

    @torch.no_grad()
    def update(self, features, labels, logits=None, ignore_mask=None, label_confidence=None):
        """Update class nodes from teacher features and labels.

        `label_confidence` can be supplied to distinguish ground-truth labels
        from pseudo labels. Ground-truth pixels should use confidence 1.
        """
        features = features.detach().float()
        labels = self._resize_labels(labels, features.shape[-2:])
        labels = self._apply_ignore(labels, ignore_mask, features.shape[-2:])
        label_confidence = self._prepare_confidence(
            label_confidence, logits, labels, features.shape[-2:]
        )

        flat_features = F.normalize(
            features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]),
            dim=1,
        )
        flat_labels = labels.reshape(-1)
        flat_confidence = label_confidence.reshape(-1)
        valid = self._valid_label_mask(flat_labels)

        sums = features.new_zeros(self.num_classes, features.shape[1])
        counts = features.new_zeros(self.num_classes)
        confidence_sums = features.new_zeros(self.num_classes)

        if valid.any():
            valid_features = flat_features[valid]
            valid_labels = flat_labels[valid]
            valid_confidence = flat_confidence[valid]
            sums.index_add_(0, valid_labels, valid_features)
            counts.index_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=features.dtype))
            confidence_sums.index_add_(0, valid_labels, valid_confidence.to(features.dtype))

        self._all_reduce_(sums)
        self._all_reduce_(counts)
        self._all_reduce_(confidence_sums)

        active = counts >= self.min_pixels
        old_seen_active = self.seen[active].clone() if active.any() else None
        if active.any():
            batch_proto = sums[active] / counts[active].clamp_min(1.0).unsqueeze(1)
            batch_proto = F.normalize(batch_proto, dim=1)
            old_proto = self.prototypes[active]
            old_seen = old_seen_active.unsqueeze(1)
            mixed_proto = F.normalize(
                self.momentum * old_proto + (1.0 - self.momentum) * batch_proto,
                dim=1,
            )
            self.prototypes[active] = torch.where(old_seen, mixed_proto, batch_proto)
            self.seen[active] = True

            batch_confidence = confidence_sums[active] / counts[active].clamp_min(1.0)
            mixed_confidence = (
                self.momentum * self.confidence[active]
                + (1.0 - self.momentum) * batch_confidence
            )
            self.confidence[active] = torch.where(old_seen_active, mixed_confidence, batch_confidence)

        self.counts.mul_(self.momentum)
        if active.any():
            mixed_counts = self.counts[active] + (1.0 - self.momentum) * counts[active]
            self.counts[active] = torch.where(old_seen_active, mixed_counts, counts[active])
        coverage = 1.0 - torch.exp(-self.counts / max(self.reliability_warmup, 1.0))
        self.reliability.copy_(torch.clamp(self.confidence * coverage, 0.0, 1.0))

        if logits is not None:
            self._update_confusion(logits.detach().float(), labels, counts)

        self._refresh_graph_cache()

        reliable = self._ready_nodes()
        return {
            "active_classes": int(active.sum().item()),
            "seen_classes": int(self.seen.sum().item()),
            "reliable_classes": int(reliable.sum().item()),
            "mean_reliability": float(self.reliability[reliable].mean().item()) if reliable.any() else 0.0,
        }

    def attack_loss(self, features, labels, ignore_mask=None):
        """Loss maximized to create gossip-guided adversarial perturbations."""
        feat, target, peer, valid, risk = self._prepare_pairs(features, labels, ignore_mask)
        if not valid.any():
            return features.sum() * 0.0

        feat = feat[valid]
        target = target[valid]
        risk = risk[valid]
        own = self.prototypes.detach()[target]
        peer = peer.detach()[target]
        loss = (feat * peer).sum(1) - (feat * own).sum(1)
        return (loss * risk).sum() / risk.sum().clamp_min(1e-6)

    def separation_loss(self, features, labels, ignore_mask=None):
        """Keep clean features closer to their own node than to gossiped peers."""
        feat, target, peer, valid, risk = self._prepare_pairs(features, labels, ignore_mask)
        if not valid.any():
            return features.sum() * 0.0

        feat = feat[valid]
        target = target[valid]
        risk = risk[valid]
        own = self.prototypes.detach()[target]
        peer = peer.detach()[target]
        own_sim = (feat * own).sum(1)
        peer_sim = (feat * peer).sum(1)
        loss = F.softplus((peer_sim - own_sim + self.margin) / self.temperature) * self.temperature
        return (loss * risk).sum() / risk.sum().clamp_min(1e-6)

    @torch.no_grad()
    def perturbation_scale(self, labels, size):
        """Return a per-pixel epsilon scale in [epsilon_min_ratio, 1]."""
        labels = self._resize_labels(labels, size)
        flat_labels = labels.reshape(-1)
        valid = self._valid_label_mask(flat_labels)
        safe_labels = flat_labels.clamp(0, self.num_classes - 1)
        class_risk = self.last_risk.to(labels.device)
        risk = class_risk[safe_labels]
        risk = torch.where(valid, risk, torch.zeros_like(risk))
        risk = risk.clamp(0.0, 1.0).pow(max(self.risk_gamma, 1e-6))
        scale = self.epsilon_min_ratio + (1.0 - self.epsilon_min_ratio) * risk
        scale = torch.where(valid, scale, torch.zeros_like(scale))
        return scale.view(labels.shape[0], 1, labels.shape[1], labels.shape[2])

    @torch.no_grad()
    def _update_confusion(self, logits, labels, counts):
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=True)
        probs = F.softmax(logits, dim=1)
        flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
        flat_labels = labels.reshape(-1)
        valid = self._valid_label_mask(flat_labels)

        confusion_sum = logits.new_zeros(self.num_classes, self.num_classes)
        if valid.any():
            confusion_sum.index_add_(0, flat_labels[valid], flat_probs[valid])

        self._all_reduce_(confusion_sum)
        row_counts = counts.clamp_min(1.0).unsqueeze(1)
        batch_confusion = confusion_sum / row_counts
        diag = torch.arange(self.num_classes, device=batch_confusion.device)
        batch_confusion[diag, diag] = 0.0

        active = counts >= self.min_pixels
        if active.any():
            self.confusion[active] = (
                self.confusion_momentum * self.confusion[active]
                + (1.0 - self.confusion_momentum) * batch_confusion[active]
            )

    def _prepare_pairs(self, features, labels, ignore_mask=None):
        features_float = features.float()
        labels = self._resize_labels(labels, features_float.shape[-2:])
        labels = self._apply_ignore(labels, ignore_mask, features_float.shape[-2:])

        feat = F.normalize(
            features_float.permute(0, 2, 3, 1).reshape(-1, features_float.shape[1]),
            dim=1,
        )
        target = labels.reshape(-1)
        valid = self._valid_label_mask(target)

        peer, peer_valid, class_risk = self._gossip_peers(features_float.device)
        safe_target = target.clamp(0, self.num_classes - 1)
        class_ready = self.seen.to(target.device)[safe_target] & peer_valid.to(target.device)[safe_target]
        risk = class_risk.to(target.device)[safe_target]
        valid = valid & class_ready & (risk > 0)
        return feat, target, peer, valid, risk

    @torch.no_grad()
    def _refresh_graph_cache(self):
        peer, peer_valid, risk = self._gossip_peers(self.prototypes.device)
        self.last_risk.copy_(torch.where(peer_valid, risk, torch.zeros_like(risk)))

    @torch.no_grad()
    def _gossip_peers(self, device):
        peer = self.prototypes.new_zeros(self.prototypes.shape)
        peer_valid = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        class_risk = torch.zeros(self.num_classes, device=device)

        ready = self._ready_nodes().to(device)
        if int(ready.sum().item()) < 2:
            self.last_adjacency.zero_()
            return peer, peer_valid, class_risk

        proto = F.normalize(self.prototypes.float(), dim=1)
        scores, ready_pair = self._relation_scores(proto, ready)
        k = max(1, min(self.topk, int(ready.sum().item()) - 1))
        vals, idx = torch.topk(scores, k=k, dim=1)
        valid = (vals[:, 0] > -1e3) & ready

        adjacency = scores.new_zeros(scores.shape)
        weights = F.softmax(vals / max(self.temperature, 1e-6), dim=1)
        adjacency.scatter_(1, idx, weights)
        adjacency = adjacency * valid.unsqueeze(1).float()
        adjacency = adjacency.masked_fill(~ready_pair, 0.0)
        row_sum = adjacency.sum(1, keepdim=True).clamp_min(1e-6)
        adjacency = adjacency / row_sum
        self.last_adjacency.copy_(adjacency)

        state = proto
        rounds = max(1, self.diffusion_rounds)
        for _ in range(rounds):
            neighbor_state = torch.matmul(adjacency, state)
            state = F.normalize(self.self_loop * state + (1.0 - self.self_loop) * neighbor_state, dim=1)

        risk = torch.sigmoid(vals[:, 0]) * self.reliability.to(scores.device)
        risk = torch.where(valid, risk, torch.zeros_like(risk))
        peer = torch.where(valid.unsqueeze(1), state, torch.zeros_like(state))
        peer_valid.copy_(valid)
        class_risk.copy_(risk)
        return peer.to(device), peer_valid, class_risk.to(device)

    def _relation_scores(self, proto, ready):
        ready_pair = ready.unsqueeze(0) & ready.unsqueeze(1)
        diag = torch.eye(self.num_classes, dtype=torch.bool, device=proto.device)
        ready_pair = ready_pair & ~diag

        cosine = torch.matmul(proto, proto.t())
        cosine = self._masked_row_zscore(cosine, ready_pair)
        confusion = self._masked_row_zscore(self.confusion.float().to(proto.device), ready_pair)
        scores = self.similarity_weight * cosine + self.confusion_weight * confusion
        scores = scores.masked_fill(~ready_pair, -1e4)
        return scores, ready_pair

    def _masked_row_zscore(self, matrix, valid):
        valid_float = valid.float()
        count = valid_float.sum(1, keepdim=True).clamp_min(1.0)
        safe = torch.where(valid, matrix, torch.zeros_like(matrix))
        mean = safe.sum(1, keepdim=True) / count
        var = (((safe - mean) * valid_float) ** 2).sum(1, keepdim=True) / count
        z = (matrix - mean) / torch.sqrt(var + 1e-6)
        return z.masked_fill(~valid, -1e4)

    def _ready_nodes(self):
        return self.seen & (self.reliability >= self.min_reliability)

    def _prepare_confidence(self, label_confidence, logits, labels, size):
        if label_confidence is not None:
            if label_confidence.dim() == 4:
                label_confidence = label_confidence.squeeze(1)
            if label_confidence.shape[-2:] != size:
                label_confidence = F.interpolate(
                    label_confidence.unsqueeze(1).float(),
                    size=size,
                    mode="nearest",
                ).squeeze(1)
            return label_confidence.float().clamp(0.0, 1.0)

        if logits is None:
            return torch.ones_like(labels, dtype=torch.float32)

        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=True)
        probs = F.softmax(logits.float(), dim=1)
        safe_labels = labels.clamp(0, self.num_classes - 1)
        confidence = probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
        return torch.where(self._valid_label_mask(labels), confidence, torch.zeros_like(confidence))

    def _valid_label_mask(self, labels):
        return (
            (labels != self.ignore_label)
            & (labels >= 0)
            & (labels < self.num_classes)
        )

    def _resize_labels(self, labels, size):
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        labels = labels.detach()
        if labels.shape[-2:] != size:
            labels = F.interpolate(labels.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1)
        return labels.long()

    def _apply_ignore(self, labels, ignore_mask, size):
        if ignore_mask is None:
            return labels
        mask = self._resize_labels(ignore_mask, size)
        labels = labels.clone()
        labels[mask == self.ignore_label] = self.ignore_label
        return labels

    @staticmethod
    def _all_reduce_(tensor):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor)
