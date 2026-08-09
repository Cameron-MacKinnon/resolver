import json
from pathlib import Path

import cv2
from cv2.typing import MatLike

from .cache_config import INDEX_CACHE_DIR, PROJECT_ROOT
from .card_detector import CardDetector
from .comparator import Comparator, MatchResult
from .frame_selector import FrameSelector
from .hasher import Hasher
from .scanner import Scanner


class IndexLoadError(Exception):
    """Base index loading error exception"""

    def __init__(self, message):
        suffix = ", maybe you forgot to run the cache/index generation pipeline?"
        modified_message = f"{message}{suffix}"
        super().__init__(modified_message)


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


class RecognitionPipeline:
    def __init__(self) -> None:
        # declare indexes then load them
        self.card_data_index: dict[str, dict] | None = None
        self.name_index: dict[str, list[str]] | None = None
        self.rulings_index: dict[str, list[dict]] | None = None
        self.keyword_index: dict[str, str] | None = None
        self.hash_index: list[dict] | None = None
        self._load_indexes()

        # init scanner
        self.scanner: Scanner = Scanner(camera_input=0)

        # init comparator (performs hash index preprocessing to cast
        # hex hashes back to ImageHash), we only want to do this once
        assert self.hash_index is not None
        self.comparator: Comparator = Comparator(self.hash_index)

        # create test output folder if not exists (only used for local auditing)
        Path(PROJECT_ROOT / "test_output").mkdir(exist_ok=True)

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

    def _load_indexes(self) -> None:
        """Load the indexes files used by the image recognition pipeline to memory,
        these will be injected as dependencies to callers in this class where required."""
        # index filepaths
        card_data_index_path = INDEX_CACHE_DIR / "card_data_index.json"
        name_index_path = INDEX_CACHE_DIR / "name_index.json"
        rulings_index_path = INDEX_CACHE_DIR / "rulings_index.json"
        keyword_index_path = INDEX_CACHE_DIR / "keyword_index.json"
        hash_index_path = INDEX_CACHE_DIR / "hash_index.jsonl"

        # check / load card data index
        self._check_file(
            card_data_index_path, MissingCardDataIndexError, EmptyCardDataIndexError
        )
        with open(card_data_index_path, "r") as file:
            self.card_data_index = json.load(file)

        # check / load name index
        self._check_file(name_index_path, MissingNameIndexError, EmptyNameIndexError)
        with open(name_index_path, "r") as file:
            self.name_index = json.load(file)

        # load rulings index
        self._check_file(
            rulings_index_path, MissingRulingsIndexError, EmptyRulingsIndexError
        )
        with open(rulings_index_path, "r") as file:
            self.rulings_index = json.load(file)

        # load keyword index
        self._check_file(
            keyword_index_path, MissingKeywordIndexError, EmptyKeywordIndexError
        )
        with open(keyword_index_path, "r") as file:
            self.keyword_index = json.load(file)

        # load image hash index
        self._check_file(hash_index_path, MissingHashIndexError, EmptyHashIndexError)
        with open(hash_index_path, "r") as file:
            self.hash_index = [json.loads(line) for line in file]

    def _hash_and_search(self, card: MatLike) -> MatchResult | None:
        # generate phash for this card
        hash = Hasher(card).compute_hash()

        # perform a comparison against the image hash, reusing the comparator
        # built once in __init__ rather than constructing a new one per card
        result = self.comparator.find_best_match(hash)

        return result

    def _get_oracle_text(self, card_data: dict) -> str:
        """Return the card's oracle text.

        Most cards have a usable top-level oracle_text. Multi-faced cards
        (double-faced/transform/adventure/split/etc) have no top-level
        oracle_text at all,  each entry in card_faces carries its own
        instead, so those get joined together.
        """
        if "card_faces" in card_data:
            return "\n\n".join(face["oracle_text"] for face in card_data["card_faces"])
        return card_data["oracle_text"]

    def _get_linked_card_details(
        self, match: MatchResult
    ) -> dict[str, dict | list | int]:
        """Uses the image recognition result to lookup the card details from the indexes
        and form a single return payload"""
        # reassure mypy that the indexes are not None
        assert self.card_data_index is not None
        assert self.rulings_index is not None
        assert self.keyword_index is not None

        # get card details and unpack keys to use here
        card_data = self.card_data_index[match.id]
        oracle_id = card_data["oracle_id"]
        keywords = card_data["keywords"]

        # get rulings data (if any)
        rulings = self.rulings_index.get(oracle_id, [])

        # get keyword definitions (if any), skipping any keyword
        # names that don't have an exact match in the glossary
        keyword_definitions: dict[str, str] = {}
        for keyword in keywords:
            definition = self.keyword_index.get(keyword)
            if definition is not None:
                keyword_definitions[keyword] = definition

        # build final payload
        return {
            "card_details": {
                "name": card_data["name"],
                "color_identity": card_data["color_identity"],
                "mana_cost": card_data.get("mana_cost"),
                "type_line": card_data["type_line"],
                "oracle_text": self._get_oracle_text(card_data),
            },
            "keyword_definitions": keyword_definitions,
            "rulings": rulings,
            "ids": {
                "id": card_data["id"],
                "oracle_id": card_data["oracle_id"],
            },
        }

    def run(self) -> dict[str, dict | list | int] | None:
        """Run the full image capture -> card recognition pipeline."""
        # capture image burst
        image_burst = self.scanner.capture_burst()

        # select sharpest image from the burst (write to disk for auditing)
        sharpest_image = FrameSelector(image_burst).select_sharpest_image()
        cv2.imwrite(
            Path(PROJECT_ROOT / "test_output" / "sharpest_image.jpg"), sharpest_image
        )

        # detect all cardlike objects in the image (write to disk for auditing)
        card_candidates = CardDetector(sharpest_image).detect_cards()
        for index, card in enumerate(card_candidates):
            cv2.imwrite(
                Path(PROJECT_ROOT / "test_output" / f"detected_card_{index}.jpg"), card
            )

        # search for matches and gather reults
        results: list[MatchResult | None] = []
        for card in card_candidates:
            results.append(self._hash_and_search(card))

        # drop any unmatched candidates and filter down to the single most
        # confident match; since we only expect a single card in the frame,
        # this also handles situations where the card gets detected twice
        # (e.g., the absolute border and croppoed border of the card are
        # identified as separate entities during card_detector's run)
        matches = [entry for entry in results if entry is not None]
        if not matches:
            return None
        best_match = max(matches, key=lambda entry: entry.score)

        # lookup this cards details and return payload
        return self._get_linked_card_details(best_match)
