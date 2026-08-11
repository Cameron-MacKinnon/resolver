from pathlib import Path

import cv2
from cv2.typing import MatLike

from .cache_config import PROJECT_ROOT
from .card_detector import CardDetector
from .comparator import Comparator, MatchResult
from .frame_selector import FrameSelector
from .hasher import Hasher
from .index_store import IndexStore
from .scanner import Scanner


class RecognitionPipeline:
    def __init__(self, index_store: IndexStore) -> None:
        self.index_store = index_store

        # init scanner
        self.scanner: Scanner = Scanner(camera_input=0)

        # init comparator (performs hash index preprocessing to cast
        # hex hashes back to ImageHash), we only want to do this once
        self.comparator: Comparator = Comparator(index_store.hash_index)

        # create test output folder if not exists (only used for local auditing)
        Path(PROJECT_ROOT / "test_output").mkdir(exist_ok=True)

    def _hash_and_search(self, card: MatLike) -> MatchResult | None:
        """Generate a phash for the given card and return the match"""
        # generate phash for this card
        hash = Hasher(card).compute_hash()

        # perform a comparison against the image hash
        result = self.comparator.find_best_match(hash)

        return result

    def _get_linked_card_details(
        self, match: MatchResult
    ) -> dict[str, dict | list | int]:
        """Uses the image recognition result to lookup the card details from the indexes
        and form a single return payload"""
        # get card details and unpack keys to use here
        card_data = self.index_store.card_data_index[match.id]
        oracle_id = card_data["oracle_id"]
        keywords = card_data["keywords"]

        # get rulings data (if any)
        rulings = self.index_store.rulings_index.get(oracle_id, [])

        # get keyword definitions (if any), skipping any keyword
        # names that don't have an exact match in the glossary
        keyword_definitions: dict[str, str] = {}
        for keyword in keywords:
            definition = self.index_store.keyword_index.get(keyword)
            if definition is not None:
                keyword_definitions[keyword] = definition

        # build final payload
        return {
            "card_details": {
                "name": card_data["name"],
                "color_identity": card_data["color_identity"],
                "mana_cost": card_data["mana_cost"],
                "type_line": card_data["type_line"],
                "oracle_text": card_data["oracle_text"],
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
