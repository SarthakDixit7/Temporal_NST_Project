import argparse
from pathlib import Path

import lpips
import torch
import torch.nn as nn

from config import VIDEO_ROOT
from data.dataset import VideoDataset
from models.temporal import compute_occlusion_mask, compute_inbounds_mask, load_frozen_raft, warp_image
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


def resolve_video_dir(video_dir: str) -> Path:
    candidate = Path(video_dir)
    if candidate.exists():
        return candidate

    candidate_parts = candidate.parts
    if "sintel" in candidate_parts:
        sintel_index = candidate_parts.index("sintel")
        relative_path = Path(*candidate_parts[sintel_index + 1 :])
    else:
        relative_path = Path(candidate.name)

    search_roots = [Path(VIDEO_ROOT).parent.parent]
    fallback_root = Path(__file__).resolve().parent / "data" / "sintel"
    if fallback_root not in search_roots:
        search_roots.append(fallback_root)

    checked_paths = [candidate]
    for root in search_roots:
        resolved = root / relative_path
        checked_paths.append(resolved)
        if resolved.exists():
            return resolved

    checked = "\n".join(f"- {path}" for path in checked_paths)
    raise FileNotFoundError(
        f"Video directory not found. Checked:\n{checked}"
    )


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    video_dir: str,
    device: torch.device,
    require_pretrained_raft: bool = True,
) -> tuple[float, float]:
    dataset = VideoDataset(str(resolve_video_dir(video_dir)))
    if len(dataset) == 0:
        raise RuntimeError(f"No consecutive frame pairs found under: {video_dir}")

    transform_net = load_transform_network(checkpoint_path, device)
    raft = load_frozen_raft(require_pretrained=require_pretrained_raft).to(device)
    raft.eval()

    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    sequence_stats = {}

    for frame_t_minus_1, frame_t, seq_name in dataset:
        if seq_name not in sequence_stats:
            sequence_stats[seq_name] = {"warping_error": 0.0, "lpips": 0.0, "count": 0}

        frame_t_minus_1 = frame_t_minus_1.unsqueeze(0).to(device)
        frame_t = frame_t.unsqueeze(0).to(device)

        stylized_t_minus_1 = transform_net(frame_t_minus_1)
        stylized_t = transform_net(frame_t)

        flow_forward = raft(frame_t_minus_1 * 255.0, frame_t * 255.0)[-1]
        flow_backward = raft(frame_t * 255.0, frame_t_minus_1 * 255.0)[-1]
        occlusion_mask = compute_occlusion_mask(flow_forward, flow_backward)
        inbounds_mask = compute_inbounds_mask(flow_backward)

        warped_stylized_t_minus_1 = warp_image(stylized_t_minus_1, flow_backward)
        diff = stylized_t.float() - warped_stylized_t_minus_1.float()
        valid_mask = occlusion_mask.float() * inbounds_mask.float()
        squared_error = (diff ** 2) * valid_mask
        
        valid_pixels = valid_mask.sum()
        if valid_pixels > 0:
            warping_error = squared_error.sum() / (valid_pixels * stylized_t.shape[1])
        else:
            warping_error = torch.tensor(0.0, device=device)

        lpips_score = loss_fn(
            stylized_t.float(),
            to_lpips_range(frame_t.float()),
        ).mean()

        sequence_stats[seq_name]["warping_error"] += float(warping_error.item())
        sequence_stats[seq_name]["lpips"] += float(lpips_score.item())
        sequence_stats[seq_name]["count"] += 1

    print("\n--- Per-Sequence Results ---")
    seq_warping_errors = []
    seq_lpips_scores = []
    
    for seq_name, stats in sorted(sequence_stats.items()):
        count = stats["count"]
        mean_we = stats["warping_error"] / count
        mean_lpips = stats["lpips"] / count
        seq_warping_errors.append(mean_we)
        seq_lpips_scores.append(mean_lpips)
        print(f"Sequence: {seq_name:<20} | Pairs: {count:<4} | Warping Error: {mean_we:.6f} | LPIPS: {mean_lpips:.6f}")

    print("----------------------------\n")
    global_mean_we = sum(seq_warping_errors) / len(seq_warping_errors)
    global_mean_lpips = sum(seq_lpips_scores) / len(seq_lpips_scores)

    return global_mean_we, global_mean_lpips


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
    parser.add_argument(
        "--allow_raft_fallback",
        action="store_false",
        dest="require_pretrained_raft",
        default=True,
        help="Allow fallback to random RAFT weights if pretrained cannot be loaded.",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    video_dir = resolve_video_dir(args.video_dir)

    warping_error, lpips_score = evaluate(
        str(checkpoint_path),
        str(video_dir),
        device,
        require_pretrained_raft=args.require_pretrained_raft,
    )

    print(f"Global Sequence-Averaged Warping Error: {warping_error:.6f}")
    print(f"Global Sequence-Averaged LPIPS: {lpips_score:.6f}")


if __name__ == "__main__":
    main()

