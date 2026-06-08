from torch.utils.data import DataLoader

from data.dataset import CocoDataset, StyleDataset


def make_dataloaders(
	coco_root="data/train2014", wikiart_root="data/wikiart_subset", batch_size=16, num_workers=4
):
	coco_ds = CocoDataset(coco_root)
	style_ds = StyleDataset(wikiart_root)

	coco_loader = DataLoader(coco_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	style_loader = DataLoader(style_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	return coco_loader, style_loader


if __name__ == "__main__":
	coco_loader, style_loader = make_dataloaders()
	print("COCO samples:", len(coco_loader.dataset))
	print("Style samples:", len(style_loader.dataset))
	# Print a single batch shape (if available)
	try:
		cb = next(iter(coco_loader))
		sb = next(iter(style_loader))
		print("COCO batch shape:", cb.shape)
		print("Style batch shape:", sb.shape)
	except Exception as e:
		print("Could not fetch a batch (maybe datasets are empty):", e)
