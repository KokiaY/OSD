from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

__all__ = ["SoftCrossEntropyLoss"]


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: float = 0.0,
        ignore_index: Optional[int] = -100,
        dim=1,
        class_weights: Optional[Tensor] = None,
    ):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.smooth_factor,
        )
