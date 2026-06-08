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

Test dataloaders:
```
python train.py
```
