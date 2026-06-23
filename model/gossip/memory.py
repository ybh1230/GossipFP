import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class GossipPrototypeBank(nn.Module):
    """Class-level gossip memory for feature perturbation.

    Each semantic class is treated as a gossip node. The node keeps an online
    prototype, listens to confusing or nearby nodes, and exposes losses that
    can move features toward class boundaries without a density estimator.
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
        margin=0.2,
        ignore_label=255,
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
        self.margin = float(margin)
        self.ignore_label = int(ignore_label)

        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer("counts", torch.zeros(self.num_classes))
        self.register_buffer("seen", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("confusion", torch.zeros(self.num_classes, self.num_classes))

    @torch.no_grad()
    def update(self, features, labels, logits=None, ignore_mask=None):
        """Update class nodes from teacher features and pseudo labels."""
        features = features.detach().float()
        labels = self._resize_labels(labels, features.shape[-2:])
        labels = self._apply_ignore(labels, ignore_mask, features.shape[-2:])

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
        if active.any():
            batch_proto = sums[active] / counts[active].clamp_min(1.0).unsqueeze(1)
            batch_proto = F.normalize(batch_proto, dim=1)
            old_proto = self.prototypes[active]
            old_seen = self.seen[active].unsqueeze(1)
            mixed_proto = F.normalize(
                self.momentum * old_proto + (1.0 - self.momentum) * batch_proto,
                dim=1,
            )
            self.prototypes[active] = torch.where(old_seen, mixed_proto, batch_proto)
            self.seen[active] = True

        self.counts.mul_(self.momentum).add_(counts * (1.0 - self.momentum))

        if logits is not None:
            self._update_confusion(logits.detach().float(), labels, counts)

        return {
            "active_classes": int(active.sum().item()),
            "seen_classes": int(self.seen.sum().item()),
        }

    def attack_loss(self, features, labels, ignore_mask=None):
        """Loss maximized to create gossip-guided adversarial perturbations."""
        feat, target, peer, valid = self._prepare_pairs(features, labels, ignore_mask)
        if not valid.any():
            return features.sum() * 0.0

        feat = feat[valid]
        target = target[valid]
        own = self.prototypes.detach()[target]
        peer = peer.detach()[target]
        return ((feat * peer).sum(1) - (feat * own).sum(1)).mean()

    def separation_loss(self, features, labels, ignore_mask=None):
        """Train-time regularizer that keeps clean features class-consistent."""
        feat, target, peer, valid = self._prepare_pairs(features, labels, ignore_mask)
        if not valid.any():
            return features.sum() * 0.0

        feat = feat[valid]
        target = target[valid]
        own = self.prototypes.detach()[target]
        peer = peer.detach()[target]
        own_sim = (feat * own).sum(1)
        peer_sim = (feat * peer).sum(1)
        return F.softplus((peer_sim - own_sim + self.margin) / self.temperature).mean() * self.temperature

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

        peer, peer_valid = self._gossip_peers(features_float.device)
        if valid.any():
            safe_target = target.clamp(0, self.num_classes - 1)
            class_ready = self.seen.to(target.device)[safe_target] & peer_valid.to(target.device)[safe_target]
            valid = valid & class_ready
        return feat, target, peer, valid

    @torch.no_grad()
    def _gossip_peers(self, device):
        peer = self.prototypes.new_zeros(self.prototypes.shape)
        peer_valid = torch.zeros(self.num_classes, dtype=torch.bool, device=device)

        if int(self.seen.sum().item()) < 2:
            return peer, peer_valid

        proto = F.normalize(self.prototypes.float(), dim=1)
        scores = torch.matmul(proto, proto.t()) + self.confusion_weight * self.confusion.float()
        seen = self.seen.to(scores.device)
        scores = scores.masked_fill(~seen.unsqueeze(0), -1e4)
        scores = scores.masked_fill(~seen.unsqueeze(1), -1e4)
        scores.fill_diagonal_(-1e4)

        k = max(1, min(self.topk, self.num_classes - 1))
        vals, idx = torch.topk(scores, k=k, dim=1)
        valid = (vals[:, 0] > -1e3) & seen
        weights = F.softmax(vals / max(self.temperature, 1e-6), dim=1)
        gathered = proto[idx]
        mixed = (weights.unsqueeze(-1) * gathered).sum(1)
        peer = F.normalize(mixed, dim=1)
        peer = torch.where(valid.unsqueeze(1), peer, torch.zeros_like(peer))
        peer_valid.copy_(valid)
        return peer.to(device), peer_valid

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
