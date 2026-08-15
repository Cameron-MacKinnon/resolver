from pathlib import Path

import cv2
from cv2.typing import MatLike

from ..datastore.index_lookup import IndexLookup
from ..paths import PROJECT_ROOT
from .card_detector import CardDetector
from .comparator import Comparator, MatchResult
from .frame_selector import FrameSelector
from .hasher import Hasher
from .scanner import Scanner


class RecognitionPipeline:
    def __init__(self, index_lookup: IndexLookup) -> None:
        self.index_lookup = index_lookup

        # init scanner
        self.scanner: Scanner = Scanner(camera_input=0)

        # init comparator (performs hash index preprocessing to cast
        # hex hashes back to ImageHash), we only want to do this once
        self.comparator: Comparator = Comparator(index_lookup.index_store.hash_index)

        # create test output folder if not exists (only used for local auditing)
        Path(PROJECT_ROOT / "test_output").mkdir(exist_ok=True)

    def release(self) -> None:
        """Release the underlying camera device."""
        self.scanner.release()

    def _hash_and_search(self, card: MatLike) -> MatchResult | None:
        """Generate a phash for the given card and return the match"""
        # generate phash for this card
        hash = Hasher(card).compute_hash()

        # perform a comparison against the image hash
        result = self.comparator.find_best_match(hash)

        return result

    def run(self) -> dict[str, dict | list] | None:
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

        # search for matches and gather results
        results: list[MatchResult | None] = []
        for card in card_candidates:
            results.append(self._hash_and_search(card))

        # drop any unmatched candidates and filter down to the single most
        # confident match; since we only expect a single card in the frame,
        # this also handles situations where the card gets detected twice
        # (e.g., the absolute border and cropped border of the card are
        # identified as separate entities during card_detector's run)
        matches = [entry for entry in results if entry is not None]
        if not matches:
            return None
        best_match = max(matches, key=lambda entry: entry.score)

        # lookup this card's full context (details, rulings, keyword
        # definitions) via the same path every other caller uses
        return self.index_lookup.get_card_context(best_match.id)
