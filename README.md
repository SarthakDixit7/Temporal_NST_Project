# Temporal_NST_Project

Quick setup and data instructions.

Requirements
```
pip install -r requirements.txt
```

Download datasets (examples):
```
python -m utils.data_download --wikiart --out data --wikiart-urls data/wikiart_urls.txt
# COCO is large; only download if you have space/bandwidth:
# python -m utils.data_download --coco --out data
```

Style image requirement:
- Place your target style image (e.g. Starry Night) in your `--style_dir`.
- The loader automatically caps the style image longest side at **512 pixels** before the 256x256 training resize.
- Do **not** feed 4K style images directly; the VGG receptive field will fail to capture macro brushstrokes.

Test dataloaders:
```
python train.py
```
