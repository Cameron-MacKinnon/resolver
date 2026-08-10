import json
from pathlib import Path

from .cache_config import INDEX_CACHE_DIR


class IndexLoadError(Exception):
    """Base index loading error exception"""

    def __init__(self, message: str) -> None:
        suffix = ", maybe you forgot to run the cache/index generation pipeline?"
        super().__init__(f"{message}{suffix}")


class MissingCardDataIndexError(IndexLoadError):
    """Raised when the card data index file is missing."""


class EmptyCardDataIndexError(IndexLoadError):
    """Raised when the card data index file exists but is empty."""


class MissingNameIndexError(IndexLoadError):
    """Raised when the card name index file is missing."""


class EmptyNameIndexError(IndexLoadError):
    """Raised when the card name index file exists but is empty."""


class MissingRulingsIndexError(IndexLoadError):
    """Raised when the rulings index file is missing."""


class EmptyRulingsIndexError(IndexLoadError):
    """Raised when the rulings index file exists but is empty."""


class MissingKeywordIndexError(IndexLoadError):
    """Raised when the keyword index file is missing."""


class EmptyKeywordIndexError(IndexLoadError):
    """Raised when the keyword index file exists but is empty."""


class MissingHashIndexError(IndexLoadError):
    """Raised when the image hash index file is missing."""


class EmptyHashIndexError(IndexLoadError):
    """Raised when the image hash index file exists but is empty."""


class MissingOracleIdIndexError(IndexLoadError):
    """Raised when the oracle_id index file is missing."""


class EmptyOracleIdIndexError(IndexLoadError):
    """Raised when the oracle_id index file exists but is empty."""


class IndexStore:
    """Loads and holds every runtime-queryable index in memory once so
    they can be accessed by multiple consumers via Dependency Injection"""

    def __init__(self) -> None:
        self.card_data_index: dict[str, dict] = self._load_json_index(
            INDEX_CACHE_DIR / "card_data_index.json",
            MissingCardDataIndexError,
            EmptyCardDataIndexError,
        )
        self.name_index: dict[str, str] = self._load_json_index(
            INDEX_CACHE_DIR / "name_index.json",
            MissingNameIndexError,
            EmptyNameIndexError,
        )
        self.oracle_id_index: dict[str, str] = self._load_json_index(
            INDEX_CACHE_DIR / "oracle_id_index.json",
            MissingOracleIdIndexError,
            EmptyOracleIdIndexError,
        )
        self.rulings_index: dict[str, list[dict]] = self._load_json_index(
            INDEX_CACHE_DIR / "rulings_index.json",
            MissingRulingsIndexError,
            EmptyRulingsIndexError,
        )
        self.keyword_index: dict[str, str] = self._load_json_index(
            INDEX_CACHE_DIR / "keyword_index.json",
            MissingKeywordIndexError,
            EmptyKeywordIndexError,
        )
        self.hash_index: list[dict] = self._load_jsonl_index(
            INDEX_CACHE_DIR / "hash_index.jsonl",
            MissingHashIndexError,
            EmptyHashIndexError,
        )

    def _check_file(
        self,
        path: Path,
        missing_exc: type[IndexLoadError],
        empty_exc: type[IndexLoadError],
    ) -> None:
        """Make sure a given index file exists and isn't empty"""
        if not path.is_file():
            raise missing_exc(f'missing index file, expected at "{path.name}"')
        if path.stat().st_size == 0:
            raise empty_exc(f'empty index file at "{path.name}"')

    def _load_json_index(
        self,
        path: Path,
        missing_exc: type[IndexLoadError],
        empty_exc: type[IndexLoadError],
    ) -> dict:
        """Load a standard JSON file to dict"""
        self._check_file(path, missing_exc, empty_exc)
        with open(path, "r") as file:
            return json.load(file)

    def _load_jsonl_index(
        self,
        path: Path,
        missing_exc: type[IndexLoadError],
        empty_exc: type[IndexLoadError],
    ) -> list[dict]:
        """Load a JSONL file to a list of dicts"""
        self._check_file(path, missing_exc, empty_exc)
        with open(path, "r") as file:
            return [json.loads(line) for line in file]
