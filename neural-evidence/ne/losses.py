"""Loss functions for E3: cross-entropy, class-weighted cross-entropy, focal loss."""
import torch
import torch.nn.functional as F


def class_weights(labels, num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency weights normalised so a balanced set gives all ones."""
    counts = torch.bincount(torch.as_tensor(labels), minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp(min=1))


def focal_loss(logits, targets, gamma: float = 2.0, weight=None):
    """Lin et al. (2017): mean of -(1 - p_t)^gamma * log p_t (optionally class-weighted)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    loss = -((1 - logp_t.exp()) ** gamma) * logp_t
    if weight is not None:
        loss = loss * weight.to(logits.device)[targets]
    return loss.mean()


def make_loss(kind: str, train_labels=None, gamma: float = 2.0):
    if kind == "ce":
        return lambda logits, y: F.cross_entropy(logits.float(), y)
    if kind == "weighted_ce":
        w = class_weights(train_labels)
        print("class weights:", [round(x, 3) for x in w.tolist()])
        return lambda logits, y: F.cross_entropy(logits.float(), y, weight=w.to(logits.device))
    if kind == "focal":
        return lambda logits, y: focal_loss(logits, y, gamma=gamma)
    raise ValueError(f"unknown loss {kind!r}")
