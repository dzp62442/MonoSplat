from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        l1_loss: bool = False,
        clamp_large_error: float = 0.0,
        valid_depth_mask: Tensor | None = None,
    ) -> Float[Tensor, ""]:
        delta = prediction.color - batch["target"]["image"]
        if (
            valid_depth_mask is not None
            and valid_depth_mask.max() > 0.5
            and valid_depth_mask.min() < 0.5
        ):
            delta = delta[~valid_depth_mask]
        else:
            delta = delta.reshape(-1)

        if clamp_large_error is not None and clamp_large_error > 0:
            valid = (delta**2) < clamp_large_error
            delta = delta[valid]

        if delta.numel() == 0:
            return torch.zeros((), dtype=prediction.color.dtype, device=prediction.color.device)

        if l1_loss:
            return self.cfg.weight * delta.abs().mean()
        return self.cfg.weight * (delta**2).mean()
