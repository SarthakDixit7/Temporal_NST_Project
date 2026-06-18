import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

import sys
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config import DATA_ROOT


def download_file(url, dest_path, chunk_size=1024 * 1024, retries=5):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    file_size = 0
    if dest_path.exists():
        file_size = dest_path.stat().st_size
        headers["Range"] = f"bytes={file_size}-"

    for attempt in range(retries):
        try:
            resp = requests.get(url, stream=True, headers=headers, timeout=30)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            if "Content-Range" in resp.headers:
                total = int(resp.headers["Content-Range"].split("/")[1])

            mode = "ab" if file_size > 0 else "wb"
            with open(dest_path, mode) as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
                initial=file_size,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            # Verify download completion
            if dest_path.stat().st_size == total:
                return dest_path
            else:
                print(f"Download incomplete. Retrying...")
                file_size = dest_path.stat().st_size
                headers["Range"] = f"bytes={file_size}-"

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Download error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
                file_size = dest_path.stat().st_size
                headers["Range"] = f"bytes={file_size}-"
            else:
                raise
    
    raise IOError("Failed to download file after multiple retries.")


def download_and_extract(url, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name
    tmp_archive = out_dir / filename

    print(f"Downloading {url} -> {tmp_archive}")
    download_file(url, tmp_archive)
    print(f"Extracting {tmp_archive} to {out_dir}")

    if ".zip" in filename:
        with zipfile.ZipFile(tmp_archive, "r") as z:
            z.extractall(out_dir)
    elif ".tar" in filename:
        with tarfile.open(tmp_archive, "r") as t:
            t.extractall(out_dir)
    else:
        print(f"Unsupported archive format for {filename}")
        return

    tmp_archive.unlink()
    print(f"Deleted {tmp_archive}")


def download_coco_train2014(target_root):
    """Downloads and extracts MS-COCO 2014 training images.

    The official COCO train2014 archive is large (~13GB). This helper
    downloads the official zip and extracts it into `target_root`.
    """
    url = "http://images.cocodataset.org/zips/train2014.zip"
    download_and_extract(url, target_root)


def download_wikiart_subset(target_root, urls_file=None):
    """Downloads a small subset of style images.

    By default this function looks for a newline-separated list of image
    URLs at `urls_file` (e.g., `data/wikiart_urls.txt`). If none is found,
    it will download a small fallback set of public artwork images from
    Wikimedia Commons so you have some style images to work with.
    """
    out_dir = Path(target_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = []
    if urls_file:
        p = Path(urls_file)
        if p.exists():
            urls = [line.strip() for line in p.read_text().splitlines() if line.strip()]

    if not urls:
        # Fallback small set (public domain via Wikimedia Commons)
        urls = [
            "https://upload.wikimedia.org/wikipedia/commons/0/0a/Claude_Monet_-_Water_Lilies%2C_1916.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/4/4c/Vincent_van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        ]

    for i, url in enumerate(urls, start=1):
        try:
            ext = os.path.splitext(url)[1].split("?")[0] or ".jpg"
            out_path = out_dir / f"style_{i}{ext}"
            if out_path.exists():
                print(f"Exists, skipping: {out_path}")
                continue
            print(f"Downloading style image {i}: {url}")
            download_file(url, out_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")


def download_wikiart_hf(target_root, num_samples=200, split="train"):
    """Download a subset of the Hugging Face `huggan/wikiart` dataset.

    Saves up to `num_samples` images into `target_root`.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Please install the `datasets` package to use Hugging Face downloads") from e

    out_dir = Path(target_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading Hugging Face dataset 'huggan/wikiart' (split={split})")
    ds = load_dataset("huggan/wikiart", split=split)
    from PIL import Image
    from io import BytesIO

    saved = 0
    for i, example in enumerate(ds):
        if num_samples and saved >= num_samples:
            break
        img = example.get("image") or example.get("img") or example.get("image_url")
        if img is None:
            continue
        out_path = out_dir / f"hf_style_{saved+1}.jpg"
        try:
            if hasattr(img, "save"):
                img.save(out_path)
            elif isinstance(img, dict) and "bytes" in img:
                Image.open(BytesIO(img["bytes"])).convert("RGB").save(out_path)
            elif isinstance(img, (bytes, bytearray)):
                Image.open(BytesIO(img)).convert("RGB").save(out_path)
            elif isinstance(img, str) and img.startswith("http"):
                # fallback: download URL
                download_file(img, out_path)
            else:
                # attempt to convert via PIL
                Image.open(img).convert("RGB").save(out_path)
            saved += 1
            if saved % 50 == 0:
                print(f"Saved {saved} images")
        except Exception as e:
            print(f"Failed to save example {i}: {e}")

    print(f"Downloaded {saved} images to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download COCO and WikiArt subsets")
    parser.add_argument(
        "--target_dir",
        type=str,
        default=DATA_ROOT,
        help="Base output directory for datasets.",
    )
    parser.add_argument("--coco", action="store_true", help="Download COCO train2014 (large)")
    parser.add_argument("--wikiart", action="store_true", help="Download WikiArt subset (small)")
    parser.add_argument("--wikiart-urls", type=str, help="Path to newline-separated URLs to download")
    parser.add_argument("--wikiart-hf", action="store_true", help="Download WikiArt from Hugging Face (huggan/wikiart)")
    parser.add_argument("--hf-num", type=int, default=200, help="Number of HF images to download (default 200)")
    parser.add_argument("--hf-split", type=str, default="train", help="HF dataset split to use (default 'train')")
    args = parser.parse_args()

    base = Path(args.target_dir)
    base.mkdir(parents=True, exist_ok=True)

    if args.coco:
        download_coco_train2014(base / "train2014")
    if args.wikiart:
        download_wikiart_subset(base / "wikiart_subset", urls_file=args.wikiart_urls)
    if args.wikiart_hf:
        download_wikiart_hf(base / "wikiart_hf", num_samples=args.hf_num, split=args.hf_split)

