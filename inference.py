import os

os.environ["TORCH_HOME"] = "/mnt/fast0/sd2885/cache/"

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from models.transform_net import TransformNetwork


def build_transform(resolution: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(resolution),
            transforms.ToTensor(),
        ]
    )


def preserve_content_color(content_image: Image.Image, stylized_image: Image.Image) -> Image.Image:
    """Keep original photo chrominance (Cb/Cr) and stylized luminance (Y)."""
    if stylized_image.size != content_image.size:
        stylized_image = stylized_image.resize(content_image.size, Image.BICUBIC)

    content_ycbcr = content_image.convert("YCbCr")
    stylized_ycbcr = stylized_image.convert("YCbCr")

    y, _, _ = stylized_ycbcr.split()
    _, cb, cr = content_ycbcr.split()

    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def infer(content_path, checkpoint_path, output_path, resolution, preserve_color=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformNetwork().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    with Image.open(content_path) as content_image:
        content_rgb = content_image.convert("RGB")

    transform = build_transform(resolution)
    content_tensor = transform(content_rgb).unsqueeze(0).to(device)

    autocast_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        with autocast_context:
            generated = model(content_tensor)

    generated = (generated + 1.0) / 2.0
    generated = generated.clamp(0.0, 1.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stylized_pil = to_pil_image(generated.squeeze(0).cpu())
    if preserve_color:
        output_rgb = preserve_content_color(content_rgb, stylized_pil)
        output_rgb.save(output_path)
    else:
        save_image(generated, output_path)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained Transform Network.")
    parser.add_argument(
        "--content_path",
        type=str,
        default="/mnt/fast0/sd2885/data/train2014/train2014/COCO_train2014_000000002150.jpg",
        help="Path to the input content image.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/mnt/fast0/sd2885/checkpoints/transform_net_iter_10373.pth",
        help="Path to a .pth checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/fast0/sd2885/output/testoutput.png",
        help="Path to save the stylized image.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1080,
        help="Shortest-edge resolution used for preprocessing.",
    )
    parser.add_argument(
        "--no_color_preservation",
        action="store_true",
        help="Disable YCbCr color preservation (save raw stylized output).",
    )

    args = parser.parse_args()
    infer(
        content_path=args.content_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        resolution=args.resolution,
        preserve_color=not args.no_color_preservation,
    )
