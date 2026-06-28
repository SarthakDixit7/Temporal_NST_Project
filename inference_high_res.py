import os
os.environ["TORCH_HOME"] = "/mnt/fast0/sd2885/cache/"

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from models.transform_net import TransformNetwork

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    raise ImportError("Please install realesrgan: pip install realesrgan")


def preserve_content_color(content_image: Image.Image, stylized_image: Image.Image) -> Image.Image:
    """Keep original photo chrominance (Cb/Cr) and stylized luminance (Y)."""
    if stylized_image.size != content_image.size:
        stylized_image = stylized_image.resize(content_image.size, Image.Resampling.BICUBIC)

    content_ycbcr = content_image.convert("YCbCr")
    stylized_ycbcr = stylized_image.convert("YCbCr")

    y, _, _ = stylized_ycbcr.split()
    _, cb, cr = content_ycbcr.split()

    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def resize_to_longest_edge(img: Image.Image, target_longest: int) -> Image.Image:
    """
    Downscale image so that the longest edge is exactly target_longest pixels,
    maintaining aspect ratio. Uses high-quality Lanczos resampling.
    """
    w, h = img.size
    if w > h:
        new_w = target_longest
        new_h = int(h * (target_longest / w))
    else:
        new_h = target_longest
        new_w = int(w * (target_longest / h))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def infer_high_res(input_img, checkpoint_path, output_img, medium_size=1024, preserve_color=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Pre-Processing (Process Medium)
    with Image.open(input_img) as img:
        original_size = img.size  # (width, height)
        img_rgb = img.convert("RGB")
    
    # Downscale so that the longest edge is exactly medium_size (1024px)
    # This intentionally bloats the relative size of the brushstrokes
    medium_img = resize_to_longest_edge(img_rgb, medium_size)
    
    # Step 2: Neural Style Transfer (Stylize)
    model = TransformNetwork().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    medium_tensor = transform(medium_img).unsqueeze(0).to(device)
    
    autocast_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        with autocast_context:
            stylized_tensor = model(medium_tensor)
    
    stylized_tensor = (stylized_tensor + 1.0) / 2.0
    stylized_tensor = stylized_tensor.clamp(0.0, 1.0)
    stylized_medium_img = to_pil_image(stylized_tensor.squeeze(0).cpu())
    
    # Step 3: Post-Processing (Upscale High with Real-ESRGAN)
    # 1. Define the model architecture
    rrdbnet = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # 2. Initialize Real-ESRGAN
    upscaler = RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=rrdbnet,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True if device.type == "cuda" else False,
        device=device.type,
    )
    
    # Upscale
    stylized_np = np.array(stylized_medium_img)
    output_np, _ = upscaler.enhance(stylized_np, outscale=None)
    high_res_img = Image.fromarray(output_np)
    
    # Ensure final output matches original dimensions
    if high_res_img.size != original_size:
        high_res_img = high_res_img.resize(original_size, Image.Resampling.LANCZOS)
    
    # Apply color preservation if needed
    if preserve_color:
        high_res_img = preserve_content_color(img_rgb, high_res_img)
    
    # Save the output
    output_path = Path(output_img)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    high_res_img.save(output_path)
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-resolution neural style transfer using Downscale -> Stylize -> AI Upscale pipeline."
    )
    parser.add_argument(
        "--input_img",
        type=str,
        default="/mnt/fast0/sd2885/data/train2014/train2014/COCO_train2014_000000000092.jpg",
        help="Path to the input content image (may be up to 4K resolution)."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/mnt/fast0/sd2885/checkpoints/transform_net_iter_31119.pth",
        help="Path to the trained TransformNetwork checkpoint (.pth)."
    )
    parser.add_argument(
        "--output_img",
        type=str,
        default="/mnt/fast0/sd2885/output/testoutput_high_res.png",
        help="Path to save the final high-resolution stylized image."
    )
    parser.add_argument(
        "--medium_size",
        type=int,
        default=512,
        help="Size of the longest edge for the medium-resolution stylization step (default: 1024)."
    )
    parser.add_argument(
        "--no_color_preservation",
        action="store_true",
        help="Disable YCbCr color preservation (save raw stylized output)."
    )
    
    args = parser.parse_args()
    infer_high_res(
        input_img=args.input_img,
        checkpoint_path=args.checkpoint_path,
        output_img=args.output_img,
        medium_size=args.medium_size,
        preserve_color=not args.no_color_preservation
    )