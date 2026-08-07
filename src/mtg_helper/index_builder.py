import json
import time
from pathlib import Path

import cv2

from .hasher import Hasher
from .scryfall_config import IMAGE_VARIANTS, INDEX_CACHE_DIR, RAW_CACHE_DIR


class IndexBuilder:
    """Reads locally-cached Scryfall data and builds runtime-queryable indexes from it"""

    def _parse_image_filename(self, image_path: Path) -> tuple[str, int, str]:
        """Split a cached image filename back into (id, face_index, variant).

        We match against the names of our chosen card image variants.
        """
        for variant in IMAGE_VARIANTS:
            suffix = f"_{variant}"
            if image_path.stem.endswith(suffix):
                remainder = image_path.stem[: -len(suffix)]
                id_, face_index = remainder.rsplit("_", 1)
                return id_, int(face_index), variant
        raise ValueError(f"unrecognised image filename: {image_path.name}")

    def build_hash_index(self) -> None:
        """Compute a phash (perceptual hash) for every cached image and persist
        as a JSONL hash index.

        This deisgned to be resumable if the process is interrupted for some
        reason, previosuly hashed cards are skipped on subsequent runs.
        """
        # start performance timer
        start_time = time.perf_counter()

        # set the directories to be used
        images_dir = RAW_CACHE_DIR / "images"
        hash_index_path = INDEX_CACHE_DIR / "hash_index.jsonl"
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # load already-computed entries to a lookup set so we can skip them
        already_hashed: set[tuple[str, int, str]] = set()
        if hash_index_path.exists():
            with open(hash_index_path, "r") as file:
                for line in file:
                    entry = json.loads(line)
                    already_hashed.add(
                        (entry["id"], entry["face_index"], entry["variant"])
                    )

        # hash whatever images exist on disk (so any failed downloads will naturally
        # be skipped to prevent exceptions / interruptions)
        image_paths = [p for p in images_dir.iterdir() if p.suffix == ".jpg"]
        completed = 0
        skipped = 0
        with open(hash_index_path, "a") as file:
            # check if this image should be skipped
            for image_path in image_paths:
                id_, face_index, variant = self._parse_image_filename(image_path)
                if (id_, face_index, variant) in already_hashed:
                    skipped += 1
                    continue

                # load the image, skip it if it's unreadable for some reason
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"skipping unreadable image: {image_path.name}")
                    continue

                # compute the hash, write this card's entry to the index file
                image_hash = Hasher(image).compute_hash()
                entry = {
                    "id": id_,
                    "face_index": face_index,
                    "variant": variant,
                    "hash": str(image_hash),
                }
                file.write(json.dumps(entry) + "\n")
                completed += 1

        # finish timer and print conclusion stats
        elapsed = time.perf_counter() - start_time
        print(
            f"hash index built in {elapsed:.1f}s ({completed} computed, {skipped} skipped)"
        )
