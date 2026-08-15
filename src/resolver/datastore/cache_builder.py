import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import requests

from .cache_config import IMAGE_VARIANTS, RAW_CACHE_DIR
from .scryfall_client import BulkType, ScryfallClient
from .wotc_client import WotcClient

# How many concurrent image downloads we want to perform. *.scryfall.io has
# no stated rate limit, but we try not to go overboard
MAX_IMAGE_FETCH_WORKERS = 40

# print a progress update every N completed image downloads
PROGRESS_LOG_INTERVAL = 1000


class CacheType(Enum):
    CARD_DATA = "card_data"
    IMAGES = "images"
    RULINGS = "rulings"
    GLOSSARY = "glossary"


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
        # scryfall client
        self.scryfall_client = ScryfallClient()

        # wizards of the coast client (comprehensive rules / glossary)
        self.wotc_client = WotcClient()

        # shared progress state for build_image_cache's worker threads
        self._progress_lock = threading.Lock()
        self._completed = 0
        self._total = 0

    def _build_data_cache(self, cache_type: CacheType, bulk_type: BulkType) -> None:
        """Fetch a Scryfall bulk-data type and write it to disk as JSONL, verbatim."""
        start_time = time.perf_counter()

        # fetch bulk data from scryfall API
        card_data = self.scryfall_client.fetch_bulk_data(bulk_type)

        # write to JSONL (JSON Lines) file
        RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filename = RAW_CACHE_DIR / f"{cache_type.value}.jsonl"
        with open(filename, "w") as file:
            for record in card_data:
                file.write(json.dumps(record) + "\n")

        elapsed = time.perf_counter() - start_time
        print(
            f"{cache_type.value} cache built in {elapsed:.1f}s ({len(card_data)} records)"
        )

    def build_glossary_cache(self) -> None:
        """Fetch the official Comprehensive Rules text and write it to disk as-is"""
        start_time = time.perf_counter()

        # get the rules text txt file from WoTC server
        rules_text = self.wotc_client.fetch_comprehensive_rules()

        # dump the txt to disk
        RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        glossary_path = RAW_CACHE_DIR / "rules.txt"
        glossary_path.write_text(rules_text)

        elapsed = time.perf_counter() - start_time
        print(f"glossary cache built in {elapsed:.1f}s")

    def build_image_cache(self) -> None:
        """Download and cache desired image variants for every card in the card-data cache.

        Requires build_cache(CacheType.CARD_DATA) to have already run, as this
        function reads its output from disk rather than depending on in-memory
        state - intentional, so this can be run independently without
        re-fetching card data from Scryfall.
        """
        start_time = time.perf_counter()

        # first, check that a card data cache exists, we need it to fetch image URIs
        card_cache_path = RAW_CACHE_DIR / "card_data.jsonl"
        if not card_cache_path.exists():
            raise NoCardDataCacheError(
                f'card data cache does not exist, expected at "{card_cache_path.name}"'
            )
        if card_cache_path.stat().st_size == 0:
            raise CardCacheEmptyError(
                f'card cache file exists at "{card_cache_path.name}", but contains no data'
            )

        # load the cached card data to memory
        with open(card_cache_path, "r") as file:
            card_data = [json.loads(line) for line in file]

        # build the list of (url, destination) pairs to fetch, skipping
        # anything already downloaded so interrupted runs can resume cheaply
        images_dir = RAW_CACHE_DIR / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        urls: list[str] = []
        paths: list[Path] = []
        for record in card_data:
            for face_index, image_uris in enumerate(self._get_face_image_uris(record)):
                for variant in IMAGE_VARIANTS:
                    image_path = (
                        images_dir / f"{record['id']}_{face_index}_{variant}.jpg"
                    )
                    if image_path.exists():
                        continue
                    urls.append(image_uris[variant])
                    paths.append(image_path)

        # now that we've extracted all URLs to hit, begin concurrent download
        # process. A thread pool is a good fit here since this is I/O-bound
        # (waiting on network), not CPU-bound
        self._completed = 0
        self._total = len(urls)
        print(f"fetching {self._total} images ({len(card_data)} cards)...")
        with ThreadPoolExecutor(max_workers=MAX_IMAGE_FETCH_WORKERS) as executor:
            list(executor.map(self._fetch_and_write_image, urls, paths))

        elapsed = time.perf_counter() - start_time
        print(
            f"done - {self._completed}/{self._total} images fetched in {elapsed:.1f}s"
        )

    def _fetch_and_write_image(self, url: str, image_path: Path) -> None:
        """Fetch one image and write it to disk, logging on failure.

        Runs as a worker task under a thread pool, we log failed downloads instead
        of raising an error so one bad download doesn't derail this (rather long)
        process. Future runs will skip already downloaded cards.
        """
        try:
            image_bytes = self.scryfall_client.fetch_image(url)
            image_path.write_bytes(image_bytes)
        except (requests.RequestException, OSError) as error:
            print(f"failed to fetch {url}: {error}")
        finally:
            self._report_progress()

    def _report_progress(self) -> None:
        """Thread-safely bump the completed-download count and log every PROGRESS_LOG_INTERVAL."""
        with self._progress_lock:
            self._completed += 1
            if self._completed % PROGRESS_LOG_INTERVAL == 0:
                print(f"progress: {self._completed}/{self._total}")

    def _get_face_image_uris(self, record: dict) -> list[dict]:
        """Return one image_uris dict per face of the card.

        Most cards have a single top-level image_uris, however, double-faced/transform/
        art-series-style cards have none at the top level; instead, each entry
        in card_faces carries its own image_uris, one per physical face, so an extra
        layer of unpacking is required.
        """
        if "image_uris" in record:
            return [record["image_uris"]]
        return [face["image_uris"] for face in record["card_faces"]]

    def build_cache(self, cache_type: CacheType) -> None:
        """Build the requested cache (card data, rulings, images, or glossary) by type."""
        match cache_type:
            case CacheType.CARD_DATA:
                self._build_data_cache(
                    cache_type=CacheType.CARD_DATA, bulk_type=BulkType.UNIQUE_ARTWORK
                )
            case CacheType.RULINGS:
                self._build_data_cache(
                    cache_type=CacheType.RULINGS, bulk_type=BulkType.RULINGS
                )
            case CacheType.IMAGES:
                self.build_image_cache()
            case CacheType.GLOSSARY:
                self.build_glossary_cache()
            case _:
                raise InvalidCacheTypeError(
                    f'invalid cache type argument "{cache_type}"'
                )
