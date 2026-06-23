import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class GossipPrototypeBank(nn.Module):
    """Class-level gossip memory for residual feature perturbation.

    Each semantic class is a node. Nodes maintain online prototypes, build a
    normalized relation graph from prototype similarity and teacher confusion,
    and exchange information through a small number of lazy gossip rounds.
    """

    def __init__(
        self,
        num_classes,
        feat_dim,
        momentum=0.99,
        confusion_momentum=0.95,
        topk=3,
        temperature=0.2,
        confusion_weight=0.5,
        min_pixels=16,
        ignore_label=255,
        diffusion_rounds=2,
        perturbation_mode="residual",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.confusion_momentum = float(confusion_momentum)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.confusion_weight = float(confusion_weight)
        self.min_pixels = int(min_pixels)
        self.ignore_label = int(ignore_label)
        self.diffusion_rounds = int(diffusion_rounds)
        self.separation_margin = 0.2
        self.perturbation_mode = str(perturbation_mode).lower()
        if self.perturbation_mode not in {"residual", "consensus"}:
            raise ValueError("perturbation_mode must be either 'residual' or 'consensus'")

        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer("counts", torch.zeros(self.num_classes))
        self.register_buffer("seen", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("confusion", torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer("last_residual_norm", torch.zeros(self.num_classes))
        self.register_buffer("last_adjacency", torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer("last_disagreement", torch.zeros(self.num_classes, self.feat_dim))

    @torch.no_grad()
    def update(self, features, labels, logits=None, ignore_mask=None, label_confidence=None):
        """Update class prototypes and teacher confusion statistics."""
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
        valid = self._valid_label_mask(flat_labels)

        sums = features.new_zeros(self.num_classes, features.shape[1])
        counts = features.new_zeros(self.num_classes)

        if valid.any():
            valid_features = flat_features[valid]
            valid_labels = flat_labels[valid]
            sums.index_add_(0, valid_labels, valid_features)
            counts.index_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=features.dtype))

        self._all_reduce_(sums)
        self._all_reduce_(counts)

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

        self.counts.mul_(self.momentum)
        if active.any():
            mixed_counts = self.counts[active] + (1.0 - self.momentum) * counts[active]
            self.counts[active] = torch.where(old_seen_active, mixed_counts, counts[active])

        if logits is not None:
            self._update_confusion(logits.detach().float(), labels, label_confidence)

        self._refresh_graph_cache()

        ready = self._ready_nodes()
        return {
            "active_classes": int(active.sum().item()),
            "seen_classes": int(self.seen.sum().item()),
            "ready_classes": int(ready.sum().item()),
            "mean_residual": float(self.last_residual_norm[ready].mean().item()) if ready.any() else 0.0,
        }

    def attack_loss(self, features, labels, ignore_mask=None):
        """Loss maximized to create gossip-guided feature perturbations."""
        feat, target, peer, valid = self._prepare_pairs(features, labels, ignore_mask)
        if not valid.any():
            return features.sum() * 0.0

        feat = feat[valid]
        target = target[valid]
        own = self.prototypes.detach()[target]
        peer = peer.detach()[target]

        if self.perturbation_mode == "consensus":
            direction = F.normalize(peer, dim=1, eps=1e-6)
            direction_valid = torch.ones(feat.shape[0], dtype=torch.bool, device=feat.device)
        else:
            residual = peer - own
            residual_norm = residual.norm(dim=1)
            direction_valid = residual_norm > 1e-6
            direction = residual / residual_norm.clamp_min(1e-6).unsqueeze(1)

        if not direction_valid.any():
            return features.sum() * 0.0

        feat = feat[direction_valid]
        direction = direction[direction_valid]
        return (feat * direction).sum(1).mean()

    def separation_loss(self, features, labels, ignore_mask=None):
        """Optional auxiliary loss; disabled by default in the provided configs."""
        feat, target, peer, valid = self._prepare_pairs(features, labels, ignore_mask)
        if not valid.any():
            return features.sum() * 0.0

        feat = feat[valid]
        target = target[valid]
        own = self.prototypes.detach()[target]
        peer = peer.detach()[target]
        own_sim = (feat * own).sum(1)
        peer_sim = (feat * peer).sum(1)
        loss = F.softplus((peer_sim - own_sim + self.separation_margin) / self.temperature)
        return (loss * self.temperature).mean()

    @torch.no_grad()
    def _update_confusion(self, logits, labels, label_confidence):
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=True)
        probs = F.softmax(logits, dim=1)
        flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
        flat_labels = labels.reshape(-1)
        flat_confidence = label_confidence.reshape(-1).to(flat_probs.dtype)
        valid = self._valid_label_mask(flat_labels)

        confusion_sum = logits.new_zeros(self.num_classes, self.num_classes)
        confidence_sum = logits.new_zeros(self.num_classes)
        count_sum = logits.new_zeros(self.num_classes)
        if valid.any():
            weighted_probs = flat_probs[valid] * flat_confidence[valid].unsqueeze(1)
            confusion_sum.index_add_(0, flat_labels[valid], weighted_probs)
            confidence_sum.index_add_(0, flat_labels[valid], flat_confidence[valid])
            count_sum.index_add_(0, flat_labels[valid], torch.ones_like(flat_confidence[valid]))

        self._all_reduce_(confusion_sum)
        self._all_reduce_(confidence_sum)
        self._all_reduce_(count_sum)
        batch_confusion = confusion_sum / confidence_sum.clamp_min(1.0).unsqueeze(1)
        diag = torch.arange(self.num_classes, device=batch_confusion.device)
        batch_confusion[diag, diag] = 0.0

        active = (count_sum >= self.min_pixels) & (confidence_sum > 0)
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

        peer, peer_valid = self._gossip_peers(features_float.device)
        safe_target = target.clamp(0, self.num_classes - 1)
        class_ready = self.seen.to(target.device)[safe_target] & peer_valid.to(target.device)[safe_target]
        valid = valid & class_ready
        return feat, target, peer, valid

    @torch.no_grad()
    def _refresh_graph_cache(self):
        self._gossip_peers(self.prototypes.device)

    @torch.no_grad()
    def _gossip_peers(self, device):
        peer = self.prototypes.new_zeros(self.prototypes.shape)
        peer_valid = torch.zeros(self.num_classes, dtype=torch.bool, device=device)

        ready = self._ready_nodes().to(device)
        if int(ready.sum().item()) < 2:
            self.last_adjacency.zero_()
            self.last_disagreement.zero_()
            self.last_residual_norm.zero_()
            return peer, peer_valid

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
            state = F.normalize(
                0.5 * state + 0.5 * neighbor_state,
                dim=1,
            )

        disagreement = state - proto
        residual_norm = disagreement.norm(dim=1).clamp(0.0, 2.0) / 2.0
        self.last_disagreement.copy_(
            torch.where(valid.unsqueeze(1), disagreement, torch.zeros_like(disagreement))
        )
        self.last_residual_norm.copy_(torch.where(valid, residual_norm, torch.zeros_like(residual_norm)))

        peer = torch.where(valid.unsqueeze(1), state, torch.zeros_like(state))
        peer_valid.copy_(valid)
        return peer.to(device), peer_valid

    def _relation_scores(self, proto, ready):
        ready_pair = ready.unsqueeze(0) & ready.unsqueeze(1)
        diag = torch.eye(self.num_classes, dtype=torch.bool, device=proto.device)
        ready_pair = ready_pair & ~diag

        cosine = self._masked_row_zscore(torch.matmul(proto, proto.t()), ready_pair)
        confusion = self._masked_row_zscore(self.confusion.float().to(proto.device), ready_pair)
        confusion_weight = min(max(self.confusion_weight, 0.0), 1.0)
        scores = (1.0 - confusion_weight) * cosine + confusion_weight * confusion
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
        return self.seen & (self.counts >= self.min_pixels)

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
