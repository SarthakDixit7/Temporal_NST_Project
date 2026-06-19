"""Environment-aware paths for local and scratch-backed runs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRATCH_ROOT = Path("/mnt/fast0/sd2885")

if SCRATCH_ROOT.exists():
	DATA_ROOT = str(SCRATCH_ROOT / "data")
	STYLE_ROOT = str(SCRATCH_ROOT / "data" / "wikiart_hf")
	CHECKPOINT_ROOT = str(SCRATCH_ROOT / "checkpoints")
else:
	DATA_ROOT = str(PROJECT_ROOT / "data")
	STYLE_ROOT = str(PROJECT_ROOT / "data" / "wikiart_hf")
	CHECKPOINT_ROOT = str(PROJECT_ROOT / "checkpoints")
