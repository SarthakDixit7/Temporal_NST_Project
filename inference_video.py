import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from models.transform_net import TransformNetwork

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_REALESRGAN = True
except ImportError:
    HAS_REALESRGAN = False
    print("Warning: Real-ESRGAN not found. Upscaling will use simple bicubic resize.")

def calculate_scaled_dimensions(width, height, max_size=1024):
    """Calculates the new dimensions keeping the longest edge at max_size."""
    longest = max(width, height)
    if longest <= max_size:
        return width, height
    scale = max_size / longest
    new_width = int(width * scale)
    new_height = int(height * scale)
    return new_width, new_height


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("transform_net_iter_*.pth"))
    if not checkpoints:
        return None

    def parse_step(path):
        match = re.search(r"transform_net_iter_(\d+)\.pth$", path.name)
        return int(match.group(1)) if match else -1

    latest = max(checkpoints, key=parse_step)
    return latest if parse_step(latest) >= 0 else None

@torch.no_grad()
def stylize_high_res_video(checkpoint_path, input_video_path, output_video_path, max_size=1024, device="cuda"):
    """
    Reads an MP4 video, downscales frames to enlarge brushstrokes, stylizes,
    upscales back to original resolution, and saves as a new MP4.
    """
    print(f"[*] Loading model from {checkpoint_path}...")
    
    model = TransformNetwork().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Support both checkpoint formats:
    # 1) legacy: raw model state_dict
    # 2) new: dict with model_state_dict/optimizer_state_dict/global_step
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Initialize Real-ESRGAN
    upsampler = None
    if HAS_REALESRGAN:
        print("[*] Setting up Real-ESRGAN for upscaling...")
        rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=rrdb_model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=(device == "cuda")
        )

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {input_video_path}")

    # Get original video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the downscaled sizes for the TransformNetwork
    scaled_width, scaled_height = calculate_scaled_dimensions(orig_width, orig_height, max_size)

    print(f"[*] Original Video: {orig_width}x{orig_height} @ {fps}fps | Total Frames: {total_frames}")
    print(f"[*] Neural Network Processing Size: {scaled_width}x{scaled_height} (Longest edge: {max_size})")
    
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # CRITICAL: The output video MUST match the original 4K/1080p dimensions!
    out = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 30.0, (orig_width, orig_height))
    if not out.isOpened():
        raise RuntimeError(
            f"Error opening video writer for {output_path}. "
            "Try a different codec or verify that OpenCV has FFMPEG support."
        )

    frame_count = 0
    start_time = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. DOWNSCALE (Process Medium) to make brushstrokes thick
            small_frame = cv2.resize(frame_rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            
            # 3. STYLIZE
            # Convert to tensor, scale to [0, 1], add batch dimension, and move to GPU
            frame_tensor = to_tensor(small_frame).unsqueeze(0).to(device)

            # Pass through network (Outputs [-1.0, 1.0])
            stylized_tensor = model(frame_tensor)

            # Shift back to [0.0, 1.0] and clamp
            stylized_tensor = (stylized_tensor + 1.0) / 2.0
            stylized_tensor = stylized_tensor.clamp(0, 1)

            # Convert back to Numpy array and HWC format
            stylized_frame = stylized_tensor.squeeze(0).cpu().numpy()
            stylized_frame = stylized_frame.transpose(1, 2, 0)
            
            # Convert [0, 1] float back to [0, 255] uint8
            stylized_frame_uint8 = (stylized_frame * 255.0).astype("uint8")

            # 4. UPSCALE HIGH (Back to original dimensions)
            if upsampler is not None:
                # RealESRGAN expects BGR
                stylized_bgr = cv2.cvtColor(stylized_frame_uint8, cv2.COLOR_RGB2BGR)
                upscaled_bgr, _ = upsampler.enhance(stylized_bgr, outscale=4)
                # Resize back to exact original dimensions
                final_frame_bgr = cv2.resize(upscaled_bgr, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
            else:
                upscaled_frame = cv2.resize(stylized_frame_uint8, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
                final_frame_bgr = cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR)

            out.write(final_frame_bgr)

            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({fps_processing:.2f} fps)")
    finally:
        cap.release()
        out.release()

    if frame_count == 0:
        raise RuntimeError("No frames were processed. Check the input video and codec support.")

    print(f"[*] Finished! High-Res stylized video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Res Video Inference Pipeline (Downscale -> Stylize -> Upscale)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path. If omitted, latest in --checkpoint_dir is used.")
    parser.add_argument("--checkpoint_dir", type=str, default="/mnt/fast0/sd2885/checkpoints", help="Directory to search for latest checkpoint when --checkpoint is omitted.")
    parser.add_argument("--input", type=str, required=True, help="/homes/sd2885/Temporal_NST_Project/Family_guy.mp4")
    parser.add_argument("--output", type=str, default="/mnt/fast0/sd2885/output/Family_guy_1024.mp4", help="Path to save output .mp4 video")
    parser.add_argument("--max_size", type=int, default=1024, help="Max size for longest edge during Neural Net processing")
    
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(args.checkpoint_dir)
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"No checkpoint found. checkpoint={args.checkpoint} checkpoint_dir={args.checkpoint_dir}"
        )
    print(f"[*] Using checkpoint: {checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stylize_high_res_video(str(checkpoint_path), args.input, args.output, args.max_size, device)
