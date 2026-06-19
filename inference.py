import os

os.environ["TORCH_HOME"] = "/mnt/fast0/sd2885/cache/"

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
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


def infer(content_path, checkpoint_path, output_path, resolution):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformNetwork().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    transform = build_transform(resolution)
    with Image.open(content_path) as image:
        content_tensor = transform(image).unsqueeze(0).to(device)

    autocast_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        with autocast_context:
            generated = model(content_tensor)

    generated = generated.clamp(0.0, 1.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(generated, output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained Transform Network.")
    parser.add_argument(
        "--content_path",
        type=str,
        default="/mnt/fast0/sd2885/data/train2014/train2014/COCO_train2014_000000000839.jpg",
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
        default="/mnt/fast0/sd2885/output/giraffe.png",
        help="Path to save the stylized image.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1080,
        help="Shortest-edge resolution used for preprocessing.",
    )

    args = parser.parse_args()
    infer(
        content_path=args.content_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        resolution=args.resolution,
    )
