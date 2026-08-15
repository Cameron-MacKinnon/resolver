"""Shared constants for interacting with and locally caching reference data."""

from ..paths import PROJECT_ROOT

# identifies this project to Scryfall/WotC's servers, per their API etiquette
USER_AGENT = "resolver/0.1 (+https://github.com/Cameron-MacKinnon/resolver)"

CACHE_DIR = PROJECT_ROOT / "reference_cache"

# CacheBuilder's exclusive domain: raw data fetched straight from source (Scryfall/WoTC)
RAW_CACHE_DIR = CACHE_DIR / "raw"

# IndexBuilder's exclusive domain: locally-computed, runtime-queryable indexes
INDEX_CACHE_DIR = CACHE_DIR / "indexes"

# the image_uris variants we download and cache per card
IMAGE_VARIANTS = ("normal", "border_crop")
