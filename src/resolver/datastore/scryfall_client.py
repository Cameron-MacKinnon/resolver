import gzip
import json
from enum import Enum

import requests
from requests.adapters import HTTPAdapter

from .cache_config import USER_AGENT

# CacheBuilder currently supports 40 concurrent workers, this connection
# pool is sized to comfortably accommodate that
CONNECTION_POOL_SIZE = 45


class BulkType(Enum):
    """Types allowed by the Scryfall bulk API endpoint"""

    ORACLE_CARDS = "oracle_cards"
    UNIQUE_ARTWORK = "unique_artwork"
    DEFAULT_CARDS = "default_cards"
    ALL_CARDS = "all_cards"
    RULINGS = "rulings"
    ART_TAGS = "art_tags"
    ORACLE_TAGS = "oracle_tags"


class ScryfallClient:
    """Handles all contact with the Scryfall API, bulk data and card images."""

    def __init__(self) -> None:
        self.bulk_url: str = "https://api.scryfall.com/bulk-data"
        self.headers: dict[str, str] = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }

        # we reuse TCP/TLS connections across requests instead of creating a
        # fresh one for every call and paying the resultant handshake cost
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_maxsize=CONNECTION_POOL_SIZE, pool_connections=CONNECTION_POOL_SIZE
        )
        self.session.mount("https://", adapter)

    def fetch_bulk_data(self, bulk_type: BulkType) -> list[dict]:
        """Fetch a Scryfall bulk-data file (e.g. card data or rulings) as a list of dicts."""
        # bulk data api returns a metadata object that points to the actual
        # file download, so grab that first
        metadata_url = f"{self.bulk_url}/{bulk_type.value}"
        metadata_response = self.session.get(url=metadata_url, headers=self.headers)
        metadata_response.raise_for_status()
        download_uri = metadata_response.json()["jsonl_download_uri"]

        # hit the actual download location to fetch bulk data, returns
        # g-zipped JSONL (JSON Lines) archive, build a list of dicts and
        # return to caller
        with (
            self.session.get(
                url=download_uri, headers=self.headers, stream=True
            ) as response,
            gzip.GzipFile(fileobj=response.raw) as decompressed,
        ):
            response.raise_for_status()
            data = [json.loads(line) for line in decompressed]

        return data

    def fetch_image(self, url: str) -> bytes:
        """Download a single card image from a Scryfall-provided URL."""
        # fetch the image and return to caller
        response = self.session.get(url=url, headers=self.headers)
        response.raise_for_status()
        return response.content
