import json
from enum import Enum
from pathlib import Path

from .scryfall_client import BulkTypes, ScryfallClient

# required to ensure cache files are always written to the same
# location regardless of current cwd at runtime
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cache_dir = PROJECT_ROOT / "scryfall_cache"

# the image_uris variants we download and cache per card
IMAGE_VARIANTS = ("normal", "border_crop")


class CacheTypes(Enum):
    CARD_DATA = "card_data"
    IMAGES = "image_data"
    RULINGS_DATA = "rulings"


class CacheBuilderError(Exception):
    """base cache builder exception"""


class InvalidCacheTypeError(CacheBuilderError):
    """raised when the user passes an invalid cache type"""


class NoCardDataCacheError(CacheBuilderError):
    """raised when a user tries to build the card image cache before the card data cache exists"""


class CardCacheEmptyError(CacheBuilderError):
    """raised when a user tries to build the card image cache but the card data cache is empty"""


class CacheBuilder:
    """Fetches data from Scryfall and persists it locally as JSONL/image caches - pure I/O, no processing."""

    def __init__(self) -> None:
        self.scryfall_client = ScryfallClient()

    def _build_data_cache(self, cache_type: CacheTypes, bulk_type: BulkTypes) -> None:
        """Fetch a Scryfall bulk-data type and write it to disk as JSONL, verbatim."""
        # fetch bulk data from scryfall API
        card_data = self.scryfall_client.fetch_bulk_data(bulk_type)

        # write to JSONL (JSON Lines) file
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = cache_dir / f"{cache_type.value}.jsonl"
        with open(filename, "w") as file:
            for record in card_data:
                file.write(json.dumps(record) + "\n")

    def build_image_cache(self) -> None:
        """Download and cache desired image variants for every card in the card-data cache.

        Requires build_card_cache() to have already run as this function reads
        its output from disk rather than depending on in-memory state, this is
        intentional so this can be run independently without re-fetching from Scryfall.
        """
        # first, check that a card data cache exists, we need it to fetch image URIs
        card_cache_path = cache_dir / "card_data.jsonl"
        if not card_cache_path.exists():
            raise NoCardDataCacheError(
                f'card data cache does not exist, expected at "{card_cache_path.name}"'
            )
        if card_cache_path.stat().st_size == 0:
            raise CardCacheEmptyError(
                f'card cache file exists at "{card_cache_path.name}", but contains no data'
            )

        # load the card data cache to memory
        with open(card_cache_path, "r") as file:
            card_data = [json.loads(line) for line in file]

        # fetch and write desired image variants for every cached card
        images_dir = cache_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for record in card_data:
            for variant in IMAGE_VARIANTS:
                image_bytes = self.scryfall_client.fetch_image(
                    record["image_uris"][variant]
                )
                image_path = images_dir / f"{record['id']}_{variant}.jpg"
                image_path.write_bytes(image_bytes)

    def build_index(self, cache_type: CacheTypes) -> None:
        """Build the requested cache (card data, rulings, or images) by type."""
        match cache_type:
            case CacheTypes.CARD_DATA:
                self._build_data_cache(
                    cache_type=CacheTypes.CARD_DATA, bulk_type=BulkTypes.UNIQUE_ARTWORK
                )
            case CacheTypes.RULINGS_DATA:
                self._build_data_cache(
                    cache_type=CacheTypes.RULINGS_DATA, bulk_type=BulkTypes.RULINGS
                )
            case CacheTypes.IMAGES:
                self.build_image_cache()
            case _:
                raise InvalidCacheTypeError(
                    f'invalid cache type argument "{cache_type}"'
                )
