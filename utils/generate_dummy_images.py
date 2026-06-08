from pathlib import Path
from PIL import Image
import random


def make_images(out_dir, count=8, size=(512, 512)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        img = Image.new("RGB", size, (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        img.save(out_dir / f"img_{i}.jpg", quality=85)


if __name__ == "__main__":
    make_images("data/wikiart_subset", count=4, size=(512,512))
    make_images("data/train2014", count=8, size=(640,480))
    print("Dummy images created: data/wikiart_subset (4), data/train2014 (8)")
