import argparse
from pathlib import Path

import lpips
import torch
import torch.nn as nn

from data.dataset import VideoDataset
from models.temporal import compute_occlusion_mask, load_frozen_raft, warp_image
from models.transform_net import TransformNetwork


def load_transform_network(checkpoint_path: str, device: torch.device) -> TransformNetwork:
    model = TransformNetwork().to(device)

    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    model.load_state_dict(state_dict)
    model.eval()
    return model


def to_lpips_range(image_batch: torch.Tensor) -> torch.Tensor:
    return (image_batch * 2.0 - 1.0).clamp(-1.0, 1.0)


@torch.no_grad()
def evaluate(checkpoint_path: str, video_dir: str, device: torch.device) -> tuple[float, float]:
    dataset = VideoDataset(video_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No consecutive frame pairs found under: {video_dir}")

    transform_net = load_transform_network(checkpoint_path, device)
    raft = load_frozen_raft().to(device)
    raft.eval()

    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    mse_loss = nn.MSELoss(reduction="mean")
    warping_error_sum = 0.0
    lpips_sum = 0.0
    pair_count = 0

    for frame_t_minus_1, frame_t in dataset:
        frame_t_minus_1 = frame_t_minus_1.unsqueeze(0).to(device)
        frame_t = frame_t.unsqueeze(0).to(device)

        stylized_t_minus_1 = transform_net(frame_t_minus_1)
        stylized_t = transform_net(frame_t)

        flow_forward = raft(frame_t_minus_1 * 255.0, frame_t * 255.0)[-1]
        flow_backward = raft(frame_t * 255.0, frame_t_minus_1 * 255.0)[-1]
        occlusion_mask = compute_occlusion_mask(flow_forward, flow_backward)

        warped_stylized_t_minus_1 = warp_image(stylized_t_minus_1, flow_forward)
        warping_error = mse_loss(
            occlusion_mask.float() * stylized_t.float(),
            occlusion_mask.float() * warped_stylized_t_minus_1.float(),
        )

        lpips_score = loss_fn(
            stylized_t.float(),
            to_lpips_range(frame_t.float()),
        ).mean()

        warping_error_sum += float(warping_error.item())
        lpips_sum += float(lpips_score.item())
        pair_count += 1

    return warping_error_sum / pair_count, lpips_sum / pair_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal warping error and LPIPS for a trained style transfer model."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the .pth checkpoint.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing sequential test frames.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (e.g. cpu, cuda, cuda:0).",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    warping_error, lpips_score = evaluate(str(checkpoint_path), str(video_dir), device)

    print(f"Average Warping Error: {warping_error:.6f}")
    print(f"Average LPIPS: {lpips_score:.6f}")


if __name__ == "__main__":
    main()