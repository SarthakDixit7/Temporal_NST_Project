"""Temporal motion utilities for video style transfer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def load_frozen_raft() -> torch.nn.Module:
    """Load a frozen pre-trained RAFT-Large model."""
    try:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights)
    except Exception:
        model = raft_large(weights=None)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def warp_image(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp an image or feature map with a dense optical flow field.

    The input flow is assumed to be in pixel units with channel order (dx, dy).
    It is converted to normalized coordinates in [-1, 1] before sampling with
    grid_sample(..., align_corners=True).
    """
    if image.ndim != 4 or flow.ndim != 4:
        raise ValueError("image and flow must both have shape [B, C, H, W]")

    batch_size, _, height, width = image.shape
    if flow.shape[0] != batch_size or flow.shape[2:] != (height, width):
        raise ValueError("image and flow must share batch size, height, and width")

    device = image.device
    flow = flow.to(device=device)

    base_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    base_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    flow_x = 2.0 * flow[:, 0].float() / max(width - 1, 1)
    flow_y = 2.0 * flow[:, 1].float() / max(height - 1, 1)
    normalized_flow = torch.stack((flow_x, flow_y), dim=-1)

    sampling_grid = base_grid + normalized_flow
    sampling_grid = sampling_grid.to(dtype=image.dtype)

    return F.grid_sample(
        image,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def compute_occlusion_mask(
    flow_forward: torch.Tensor,
    flow_backward: torch.Tensor,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Compute a forward-backward consistency mask.

    Pixels are kept when ||f + warp(b, f)||_2^2 < epsilon.
    """
    warped_backward = warp_image(flow_backward, flow_forward)
    consistency = flow_forward + warped_backward
    squared_distance = torch.sum(consistency * consistency, dim=1, keepdim=True)
    return (squared_distance < epsilon).to(dtype=flow_forward.dtype)