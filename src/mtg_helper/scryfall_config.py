"""Shared constants for interacting with and locally caching Scryfall data."""

from pathlib import Path

# required to ensure cache files are always written to the same
# location regardless of current cwd at runtime
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "scryfall_cache"

# CacheBuilder's exclusive domain - raw data fetched straight from Scryfall
RAW_CACHE_DIR = CACHE_DIR / "raw"

# IndexBuilder's exclusive domain - locally-computed, runtime-queryable indexes
INDEX_CACHE_DIR = CACHE_DIR / "indexes"

# the image_uris variants we download and cache per card
IMAGE_VARIANTS = ("normal", "border_crop")
