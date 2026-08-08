import json
from pathlib import Path

import cv2

from .cache_config import INDEX_CACHE_DIR
from .card_detector import CardDetector
from .comparator import Comparator
from .frame_selector import FrameSelector
from .hasher import Hasher
from .index_builder import IndexBuilder
from .scanner import Scanner

# ------------------------------ SETUP RUN ------------------------------ #
# local test outputs go here
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

# pay the hash-index load cost once, up front, then reuse it across every
# scan for the rest of this session
comparator = Comparator()

# load the card data index
card_data_index_path = INDEX_CACHE_DIR / "card_data_index.json"
with open(card_data_index_path, "r") as file:
    card_data_index = json.load(file)

# init camera
scanner = Scanner(camera_input=0)

# ------------------------------ MAIN LOOP ------------------------------ #
print("ready - press enter to scan, or 'q' to quit")
while True:
    command = input("> ").strip().lower()
    if command == "q":
        break

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

        search_hash = Hasher(card).compute_hash()
        match = comparator.best_match(search_hash)

        if match is None:
            print(f"  card {index}: no confident match")
            continue

        name = card_data_index[match["id"]]["name"]
        print(
            f"  card {index}: {name} (id={match['id']}, "
            f"distance={match['distance']}, score={match['score']:.2f})"
        )
