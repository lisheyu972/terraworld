"""Differentiable PSNR objectives used by paired-image training."""

import torch
import torch.nn.functional as F


def negative_psnr_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    eps: float = 1e-6,
):
    """Return ``(-PSNR, PSNR, MSE)`` using one stable batch MSE.

    Minimizing negative PSNR is equivalent to maximizing PSNR.  Clamping the
    MSE prevents the logarithm and its gradient from becoming singular once
    the requested numerical precision has been reached.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "PSNR prediction/target shapes differ: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
        )
    data_range = float(data_range)
    eps = float(eps)
    if data_range <= 0:
        raise ValueError("PSNR data_range must be positive.")
    if eps <= 0:
        raise ValueError("PSNR eps must be positive.")

    mse = F.mse_loss(prediction.float(), target.float(), reduction="mean")
    safe_mse = mse.clamp_min(eps)
    data_range_tensor = safe_mse.new_tensor(data_range)
    psnr = 10.0 * torch.log10(data_range_tensor.square() / safe_mse)
    return -psnr, psnr, mse
