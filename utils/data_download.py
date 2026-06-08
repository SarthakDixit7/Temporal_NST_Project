import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url, dest_path, chunk_size=1024 * 1024):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest_path.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            pbar.update(len(chunk))
    return dest_path


def download_and_extract_zip(url, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_dir / "download_temp.zip"
    print(f"Downloading {url} -> {tmp_zip}")
    download_file(url, tmp_zip)
    print(f"Extracting {tmp_zip} to {out_dir}")
    with zipfile.ZipFile(tmp_zip, "r") as z:
        z.extractall(out_dir)
    tmp_zip.unlink()


def download_coco_train2014(target_root="../data/train2014"):
    """Downloads and extracts MS-COCO 2014 training images.

    The official COCO train2014 archive is large (~13GB). This helper
    downloads the official zip and extracts it into `target_root`.
    """
    url = "http://images.cocodataset.org/zips/train2014.zip"
    download_and_extract_zip(url, target_root)


def download_wikiart_subset(target_root="../data/wikiart_subset", urls_file=None):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download COCO and WikiArt subsets")
    parser.add_argument("--coco", action="store_true", help="Download COCO train2014 (large)")
    parser.add_argument("--wikiart", action="store_true", help="Download WikiArt subset (small)")
    parser.add_argument("--wikiart-urls", type=str, help="Path to newline-separated URLs to download")
    parser.add_argument("--out", type=str, default="data", help="Base output directory")
    args = parser.parse_args()

    base = Path(args.out)
    if args.coco:
        download_coco_train2014(base / "train2014")
    if args.wikiart:
        download_wikiart_subset(base / "wikiart_subset", urls_file=args.wikiart_urls)
