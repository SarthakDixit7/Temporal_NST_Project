"""Environment-aware paths for local and scratch-backed runs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRATCH_ROOT = Path("/mnt/fast0/sd2885")

if SCRATCH_ROOT.exists():
	DATA_ROOT = str(SCRATCH_ROOT / "data")
	VIDEO_ROOT = str(SCRATCH_ROOT / "data" / "sintel" / "training" / "clean")
	STYLE_ROOT = str(SCRATCH_ROOT / "data" / "my_style")
	CHECKPOINT_ROOT = str(SCRATCH_ROOT / "checkpoints")
else:
	DATA_ROOT = str(PROJECT_ROOT / "data")
	VIDEO_ROOT = str(PROJECT_ROOT / "data" / "sintel" / "training" / "clean")
	STYLE_ROOT = str(PROJECT_ROOT / "data" / "my_style")
	CHECKPOINT_ROOT = str(PROJECT_ROOT / "checkpoints")

DEFAULT_CONTENT_WEIGHT = 5000.0
DEFAULT_STYLE_WEIGHT = 1e10
DEFAULT_TEMPORAL_WEIGHT = 1e5
DEFAULT_TV_WEIGHT = 1e-3
