import argparse
import csv
from pathlib import Path

import lpips
import torch
import torch.nn as nn
from torchvision.models.optical_flow import Raft_Large_Weights

from evaluate import load_transform_network, resolve_video_dir, to_lpips_range
from models.temporal import compute_occlusion_mask, load_frozen_raft, warp_image
from data.dataset import VideoDataset


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc standard-preprocessing evaluation of temporal and baseline models."
    )
    parser.add_argument("--baseline_checkpoint", type=str, required=True, help="Path to baseline .pth")
    parser.add_argument("--temporal_checkpoint", type=str, required=True, help="Path to temporal .pth")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to Sintel dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Resolve video dir and initialize dataset
    video_dir = resolve_video_dir(args.video_dir)
    dataset = VideoDataset(str(video_dir))

    # Sanity check total pair count
    pair_count = len(dataset)
    print(f"Sanity Check: Total pairs found = {pair_count}")
    if pair_count != 1041:
        print("Warning: Expected 1041 pairs for MPI Sintel training set, but found a different number. Continuing anyway.")

    # 2. Load models
    print("Loading Baseline model...")
    baseline_net = load_transform_network(args.baseline_checkpoint, device)
    
    print("Loading Temporal model...")
    temporal_net = load_transform_network(args.temporal_checkpoint, device)

    print("Loading RAFT-Large with DEFAULT weights...")
    raft = load_frozen_raft(require_pretrained=True, log_status=True).to(device)
    if not getattr(raft, "pretrained_loaded", False):
         raise RuntimeError("Pretrained RAFT weights failed to load. Aborting.")
    
    raft.eval()
    
    weights = Raft_Large_Weights.DEFAULT
    raft_transforms = weights.transforms()
    print("Sanity Check: Using official RAFT transforms:", raft_transforms)

    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    # We will keep raw results here
    pair_results = []
    
    # 3. Evaluate pairs
    print(f"Evaluating {pair_count} pairs...")
    
    import math

    mask_fractions = []

    for i in range(len(dataset)):
        # Extract sequence info from paths
        prev_path, curr_path = dataset.pairs[i]
        seq_name = Path(prev_path).parent.name
        frame_idx = Path(curr_path).stem  # e.g., 'frame_0002'

        frame_t_minus_1, frame_t = dataset[i]
        frame_t_minus_1 = frame_t_minus_1.unsqueeze(0).to(device)
        frame_t = frame_t.unsqueeze(0).to(device)

        # Baseline Stylisation
        base_stylized_t_minus_1 = baseline_net(frame_t_minus_1)
        base_stylized_t = baseline_net(frame_t)

        # Temporal Stylisation
        temp_stylized_t_minus_1 = temporal_net(frame_t_minus_1)
        temp_stylized_t = temporal_net(frame_t)

        # Post-hoc standard RAFT preprocessing and Flow Estimation (using ORIGINAL unstylised frames)
        # Note: dataset outputs [0,1] floats. raft_transforms scales them properly.
        raft_input_t_m_1, raft_input_t = raft_transforms(frame_t_minus_1, frame_t)
        flow_forward = raft(raft_input_t_m_1, raft_input_t)[-1]
        
        raft_input_t_for_bw, raft_input_t_m_1_for_bw = raft_transforms(frame_t, frame_t_minus_1)
        flow_backward = raft(raft_input_t_for_bw, raft_input_t_m_1_for_bw)[-1]

        # Occlusion Mask
        occlusion_mask = compute_occlusion_mask(flow_forward, flow_backward)
        valid_pixels = occlusion_mask.sum().item()
        total_pixels = occlusion_mask.numel()
        mask_fraction = valid_pixels / max(total_pixels, 1)
        mask_fractions.append(mask_fraction)

        # Helper to compute warping error
        def calc_warp_error(stylized_t, stylized_t_m_1, flow_fw, mask):
            warped = warp_image(stylized_t_m_1, flow_fw)
            diff = stylized_t.float() - warped.float()
            squared_error = (diff.square() * mask.float()).sum()
            # normalise by valid pixels * channels, clamp_min to avoid div by zero
            channels = stylized_t.shape[1]
            denominator = max(mask.sum().item() * channels, 1.0)
            return (squared_error / denominator).item()

        base_warp_err = calc_warp_error(base_stylized_t, base_stylized_t_minus_1, flow_forward, occlusion_mask)
        temp_warp_err = calc_warp_error(temp_stylized_t, temp_stylized_t_minus_1, flow_forward, occlusion_mask)

        # LPIPS
        # compare stylised current frame against unstylised current frame (both in LPIPS range)
        orig_t_lpips = to_lpips_range(frame_t.float())
        base_lpips = loss_fn(base_stylized_t.float(), orig_t_lpips).mean().item()
        temp_lpips = loss_fn(temp_stylized_t.float(), orig_t_lpips).mean().item()

        # Check NaN/Inf
        if math.isnan(base_warp_err) or math.isinf(base_warp_err):
            print(f"Warning: NaN/Inf detected in Baseline Warping Error for pair {i}")
        if math.isnan(temp_warp_err) or math.isinf(temp_warp_err):
            print(f"Warning: NaN/Inf detected in Temporal Warping Error for pair {i}")

        pair_results.append({
            "sequence": seq_name,
            "frame_index": frame_idx,
            "baseline_warping_error": base_warp_err,
            "temporal_warping_error": temp_warp_err,
            "baseline_lpips": base_lpips,
            "temporal_lpips": temp_lpips,
            "valid_mask_fraction": mask_fraction
        })

    # Output Pair Results
    pair_csv_path = "posthoc_eval_pair_results.csv"
    with open(pair_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sequence", "frame_index", "baseline_warping_error", "temporal_warping_error",
            "baseline_lpips", "temporal_lpips", "valid_mask_fraction"
        ])
        writer.writeheader()
        writer.writerows(pair_results)
    
    print(f"Saved pair results to {pair_csv_path}")

    # Sequence Summaries
    sequences = {}
    for r in pair_results:
        seq = r["sequence"]
        if seq not in sequences:
            sequences[seq] = []
        sequences[seq].append(r)
    
    seq_summaries = []
    for seq, rows in sequences.items():
        count = len(rows)
        seq_summaries.append({
            "sequence": seq,
            "baseline_warping_error": sum(r["baseline_warping_error"] for r in rows) / count,
            "temporal_warping_error": sum(r["temporal_warping_error"] for r in rows) / count,
            "baseline_lpips": sum(r["baseline_lpips"] for r in rows) / count,
            "temporal_lpips": sum(r["temporal_lpips"] for r in rows) / count,
            "valid_mask_fraction": sum(r["valid_mask_fraction"] for r in rows) / count,
        })
    
    seq_csv_path = "posthoc_eval_sequence_summary.csv"
    with open(seq_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sequence", "baseline_warping_error", "temporal_warping_error",
            "baseline_lpips", "temporal_lpips", "valid_mask_fraction"
        ])
        writer.writeheader()
        writer.writerows(seq_summaries)
    
    print(f"Saved sequence summaries to {seq_csv_path}")

    # Aggregation
    # A. Global pair-level mean
    total_pairs = len(pair_results)
    global_base_warp = sum(r["baseline_warping_error"] for r in pair_results) / total_pairs
    global_temp_warp = sum(r["temporal_warping_error"] for r in pair_results) / total_pairs
    global_base_lpips = sum(r["baseline_lpips"] for r in pair_results) / total_pairs
    global_temp_lpips = sum(r["temporal_lpips"] for r in pair_results) / total_pairs

    # B. Sequence macro-average
    num_seqs = len(seq_summaries)
    macro_base_warp = sum(s["baseline_warping_error"] for s in seq_summaries) / num_seqs
    macro_temp_warp = sum(s["temporal_warping_error"] for s in seq_summaries) / num_seqs
    macro_base_lpips = sum(s["baseline_lpips"] for s in seq_summaries) / num_seqs
    macro_temp_lpips = sum(s["temporal_lpips"] for s in seq_summaries) / num_seqs

    # Reductions
    warp_red_factor = global_base_warp / global_temp_warp if global_temp_warp > 0 else 0
    warp_red_pct = (global_base_warp - global_temp_warp) / global_base_warp * 100 if global_base_warp > 0 else 0
    lpips_delta = global_temp_lpips - global_base_lpips

    print("\n--- Final Terminal Summary ---")
    print(f"Number of sequences: {num_seqs}")
    print(f"Total pair count: {total_pairs}")
    print(f"Valid mask fraction: Min = {min(mask_fractions):.4f}, Mean = {sum(mask_fractions)/len(mask_fractions):.4f}, Max = {max(mask_fractions):.4f}")
    
    print("\n[Global Pair-Level Means (over all 1041 pairs)]")
    print(f"Baseline Warping Error: {global_base_warp:.6f}")
    print(f"Temporal Warping Error: {global_temp_warp:.6f}")
    print(f"Warping Error Reduction Factor: {warp_red_factor:.2f}x")
    print(f"Warping Error % Reduction: {warp_red_pct:.2f}%")
    print(f"Baseline LPIPS:         {global_base_lpips:.6f}")
    print(f"Temporal LPIPS:         {global_temp_lpips:.6f}")
    print(f"LPIPS Delta (Temp-Base):{lpips_delta:.6f}")

    print("\n[Sequence Macro-Averages (mean of 23 sequence means)]")
    print(f"Baseline Warping Error: {macro_base_warp:.6f}")
    print(f"Temporal Warping Error: {macro_temp_warp:.6f}")
    print(f"Baseline LPIPS:         {macro_base_lpips:.6f}")
    print(f"Temporal LPIPS:         {macro_temp_lpips:.6f}")
    print("------------------------------\n")


if __name__ == "__main__":
    main()
