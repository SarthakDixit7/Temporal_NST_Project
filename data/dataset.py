import os
from pathlib import Path
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _list_images(folder: str) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend([str(x) for x in p.rglob(e)])
    files = sorted(files)
    return files


def _resize_longest_side(img: Image.Image, max_longest_side: int = 512) -> Image.Image:
    """Scale image so its longest side is at most `max_longest_side` pixels."""
    width, height = img.size
    longest = max(width, height)
    if longest <= max_longest_side:
        return img
    scale = max_longest_side / longest
    new_size = (int(width * scale), int(height * scale))
    return img.resize(new_size, Image.BICUBIC)


class CocoDataset(Dataset):
    """Simple Dataset for MS-COCO images stored in a directory.

    Expects the images to be extracted under `root_dir` (e.g. `data/train2014`).
    Each image is resized to 256x256 and converted to a tensor.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.files = _list_images(root_dir)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        return img


class StyleDataset(Dataset):
    """Dataset for style reference images.

    CRITICAL: Style images must be downscaled so their longest side is ~512 px
    before the final 256x256 training resize. Feeding 4K style images directly
    prevents the VGG receptive field from capturing macro brushstrokes (e.g.
    Van Gogh swirls). Place your target style (e.g. Starry Night) in the
    style directory at reasonable resolution; this loader enforces the cap.
    """

    STYLE_MAX_LONGEST_SIDE = 512

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.files = _list_images(root_dir)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Lambda(
                    lambda img: _resize_longest_side(img, self.STYLE_MAX_LONGEST_SIDE)
                ),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        return img
