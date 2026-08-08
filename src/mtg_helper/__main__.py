import json
from pathlib import Path

import cv2
from cv2.typing import MatLike

from .cache_builder import CacheBuilder, CacheType
from .cache_config import INDEX_CACHE_DIR
from .card_detector import CardDetector
from .comparator import Comparator
from .frame_selector import FrameSelector
from .hasher import Hasher
from .index_builder import IndexBuilder, IndexType
from .scanner import Scanner

# local test outputs go here
output_dir = Path("test_output")

MENU = """
1) run card identification loop
2) test a locally saved image
3) run cache generation
4) run index generation
q) quit
"""


def run_cache_generation() -> None:
    """Fetch and persist every raw cache from Scryfall/WotC."""
    builder = CacheBuilder()
    builder.build_cache(CacheType.CARD_DATA)
    builder.build_cache(CacheType.RULINGS)
    builder.build_cache(CacheType.IMAGES)
    builder.build_cache(CacheType.GLOSSARY)


def run_index_generation() -> None:
    """Build every runtime-queryable index from the raw caches."""
    builder = IndexBuilder()
    builder.build_index(IndexType.CARD_DATA)
    builder.build_index(IndexType.NAME)
    builder.build_index(IndexType.RULINGS)
    builder.build_index(IndexType.KEYWORD)
    builder.build_index(IndexType.PHASH)


def load_card_data_index() -> dict:
    """Load the card data index so match ids can be resolved to card names."""
    card_data_index_path = INDEX_CACHE_DIR / "card_data_index.json"
    with open(card_data_index_path, "r") as file:
        return json.load(file)


def report_match(
    card: MatLike, label: str, comparator: Comparator, card_data_index: dict
) -> None:
    """Hash a single card image, match it against the index, and print the result."""
    search_hash = Hasher(card).compute_hash()
    match = comparator.best_match(search_hash)

    if match is None:
        print(f"  {label}: no confident match")
        return

    name = card_data_index[match["id"]]["name"]
    print(
        f"  {label}: {name} (id={match['id']}, "
        f"distance={match['distance']}, score={match['score']:.2f})"
    )


def run_identification_loop(comparator: Comparator, card_data_index: dict) -> None:
    """Repeatedly capture from the camera and match detected cards, until 'q'."""
    scanner = Scanner(camera_input=0)

    print("ready - press enter to scan, or 'q' to return to the menu")
    while True:
        command = input("> ").strip().lower()
        if command == "q":
            return

        # capture a burst and select the sharpest frame
        image_burst = scanner.scan()
        frame_selector = FrameSelector(image_burst)
        sharpest_image = frame_selector.select_sharpest_image()
        cv2.imwrite(str(output_dir / "sharpest_image.jpg"), sharpest_image)

        # search for card objects in the image
        detected_cards = CardDetector(sharpest_image).detect_cards()
        print(f"detected {len(detected_cards)} card-like candidate(s)")

        for index, card in enumerate(detected_cards):
            cv2.imwrite(str(output_dir / f"detected_card_{index}.jpg"), card)
            report_match(card, f"card {index}", comparator, card_data_index)


def run_saved_image_test(comparator: Comparator, card_data_index: dict) -> None:
    """Match a single already-cropped card image from disk, no camera required."""
    path = Path(input("image path: ").strip())
    if not path.exists():
        print(f"no file found at '{path}'")
        return

    image = cv2.imread(str(path))
    if image is None:
        print(f"could not read image at '{path}'")
        return

    report_match(image, str(path), comparator, card_data_index)


def main() -> None:
    output_dir.mkdir(exist_ok=True)

    # comparator/card_data_index are only needed for options 1/2, and
    # loading them requires the indexes to already exist - so load them
    # lazily on first use rather than eagerly, and reuse them after that
    comparator = Comparator()
    card_data_index = load_card_data_index()

    while True:
        print(MENU)
        choice = input("> ").strip().lower()

        if choice in ("1", "2"):
            card_data_index = load_card_data_index()
        if choice == "1":
            run_identification_loop(comparator, card_data_index)
        elif choice == "2":
            run_saved_image_test(comparator, card_data_index)
        elif choice == "3":
            run_cache_generation()
        elif choice == "4":
            run_index_generation()
        elif choice == "q":
            break
        else:
            print(f"unrecognised option: '{choice}'")


if __name__ == "__main__":
    main()
