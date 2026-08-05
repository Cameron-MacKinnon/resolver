import time
from pathlib import Path

import cv2

from .card_detector import CardDetector
from .frame_selector import FrameSelector
from .scanner import Scanner

# local test outputs go here, not the repo root
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

# init scanner and capture burst
scanner = Scanner(camera_input=1)
image_burst = scanner.scan()

# select sharpest frame
frame_selector = FrameSelector(image_burst)
sharpest_image = frame_selector.select_sharpest_image()
cv2.imwrite(str(output_dir / "sharpest_image.jpg"), sharpest_image)

# search for card objects in the image
detected_cards = CardDetector(sharpest_image).detect_cards()
print(f"Detected {len(detected_cards)} card-like candidate(s)")

for index, card in enumerate(detected_cards):
    cv2.imwrite(str(output_dir / f"detected_card_{index}.jpg"), card)
