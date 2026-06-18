import os

# Check if running on the Hex server environment
IS_HEX = os.path.exists("/mnt/fast0/")

if IS_HEX:
    # Paths for the Hex scratch storage
    BASE_DIR = "/mnt/fast0/sd2885/"
    DATA_ROOT = os.path.join(BASE_DIR, "data")
    STYLE_ROOT = os.path.join(BASE_DIR, "data")  # Assuming styles are also in the data directory
    CHECKPOINT_ROOT = os.path.join(BASE_DIR, "checkpoints")
else:
    # Local paths
    BASE_DIR = "."
    DATA_ROOT = os.path.join(BASE_DIR, "data")
    STYLE_ROOT = os.path.join(BASE_DIR, "data")
    CHECKPOINT_ROOT = os.path.join(BASE_DIR, "checkpoints")
