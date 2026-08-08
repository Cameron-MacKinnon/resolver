import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import cv2

from .cache_config import IMAGE_VARIANTS, INDEX_CACHE_DIR, RAW_CACHE_DIR
from .hasher import Hasher


class IndexType(Enum):
    PHASH = "phash"
    RULINGS = "rulings"
    NAME = "name"
    CARD_DATA = "card_data"
    KEYWORD = "keyword"


@dataclass
class Ruling:
    oracle_id: str
    source: str
    published_at: str
    comment: str


@dataclass
class HashEntry:
    id: str
    face_index: int
    variant: str
    hash: str


class IndexBuilderError(Exception):
    """base index builder exception"""


class NoRulingsCacheError(IndexBuilderError):
    """raised when a user tries to build the rulings index before the rulings cache exists"""


class RulingsCacheEmptyError(IndexBuilderError):
    """raised when a user tries to build the rulings index but the rulings cache is empty"""


class CardDataCacheMissingError(IndexBuilderError):
    """raised when a user tries to build the names index before the card data cache exists"""


class CardDataCacheEmptyError(IndexBuilderError):
    """raised when a user tries to build the names index but the card data cache is empty"""


class RulesCacheMissingError(IndexBuilderError):
    """raised when a user tries to build the keyword index before the rules cache exists"""


class RulesCacheEmptyError(IndexBuilderError):
    """raised when a user tries to build the keyword index but the rules cache is empty"""


class InvalidIndexTypeError(IndexBuilderError):
    """raised when the user passes an invalid index type"""


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

    def _get_face_card_names(self, record: dict) -> list[str]:
        """Return one card name per face of the card.

        Most cards have a single, useable top-level name. Multi-faced cards
        (double-faced/transform/art-series/adventure/split/etc) have a
        top-level name too, but it's a combined "Face A // Face B" string,
        which is not ideal for fuzzy-matching against directly. Each entry
        in card_faces carries its own individual name instead.
        """
        if "card_faces" in record:
            return [face["name"] for face in record["card_faces"]]
        return [record["name"]]

    def build_phash_index(self) -> None:
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
                    entry = HashEntry(**json.loads(line))
                    already_hashed.add((entry.id, entry.face_index, entry.variant))

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
                entry = HashEntry(
                    id=id_, face_index=face_index, variant=variant, hash=str(image_hash)
                )
                file.write(json.dumps(asdict(entry)) + "\n")
                completed += 1

        # finish timer and print conclusion stats
        elapsed = time.perf_counter() - start_time
        print(
            f"hash index built in {elapsed:.1f}s ({completed} computed, {skipped} skipped)"
        )

    def build_rulings_index(self) -> None:
        """Group cached rulings by oracle_id and persist as a single JSON file."""
        start_time = time.perf_counter()

        # files / dirs used
        rulings_cache_path = RAW_CACHE_DIR / "rulings.jsonl"
        rulings_index_path = INDEX_CACHE_DIR / "rulings_index.json"
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # first, check that a rulings cache exists to build the index from
        if not rulings_cache_path.exists():
            raise NoRulingsCacheError(
                f'rulings cache does not exist, expected at "{rulings_cache_path.name}"'
            )
        if rulings_cache_path.stat().st_size == 0:
            raise RulingsCacheEmptyError(
                f'rulings cache file exists at "{rulings_cache_path.name}", but contains no data'
            )

        # loop over jsonl rulings cache and group rulings by oracle_id
        rulings_index: dict[str, list[dict]] = {}
        with open(rulings_cache_path, "r") as cache_file:
            for line in cache_file:
                data = json.loads(line)
                oracle_id = data["oracle_id"]

                ruling = Ruling(
                    oracle_id=oracle_id,
                    source=data["source"],
                    published_at=data["published_at"],
                    comment=data["comment"],
                )
                rulings_index.setdefault(oracle_id, []).append(asdict(ruling))

        # write the built index to disk
        with open(rulings_index_path, "w") as file:
            json.dump(rulings_index, file, indent=2, ensure_ascii=False)

        elapsed = time.perf_counter() - start_time
        print(
            f"rulings index built in {elapsed:.1f}s ({len(rulings_index)} oracle_ids)"
        )

    def build_name_index(self) -> None:
        """Group cached card ids by name and persist as a single JSON file."""
        start_time = time.perf_counter()

        # files / dirs used
        card_data_cache_path = RAW_CACHE_DIR / "card_data.jsonl"
        name_index_path = INDEX_CACHE_DIR / "name_index.json"
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # first, check that a card data cache exists to build the index from
        if not card_data_cache_path.exists():
            raise CardDataCacheMissingError(
                f'card data cache does not exist, expected at "{card_data_cache_path.name}"'
            )
        if card_data_cache_path.stat().st_size == 0:
            raise CardDataCacheEmptyError(
                f'card data cache file exists at "{card_data_cache_path.name}", but contains no data'
            )

        # loop over jsonl card data cache and group ids by name
        name_index: dict[str, list[str]] = {}
        with open(card_data_cache_path, "r") as cache_file:
            for line in cache_file:
                data = json.loads(line)
                card_id = data["id"]
                for name in self._get_face_card_names(data):
                    ids = name_index.setdefault(name, [])
                    if card_id not in ids:
                        ids.append(card_id)

        # write the built index to disk
        with open(name_index_path, "w") as file:
            json.dump(name_index, file, indent=2, ensure_ascii=False)

        elapsed = time.perf_counter() - start_time
        print(f"name index built in {elapsed:.1f}s ({len(name_index)} unique names)")

    def build_card_data_index(self) -> None:
        """Load the card data cache into a dict keyed by id and persist as JSON"""
        start_time = time.perf_counter()

        card_data_cache_path = RAW_CACHE_DIR / "card_data.jsonl"
        card_data_index_path = INDEX_CACHE_DIR / "card_data_index.json"
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # first, check that a card data cache exists to build the index from
        if not card_data_cache_path.exists():
            raise CardDataCacheMissingError(
                f'card data cache does not exist, expected at "{card_data_cache_path.name}"'
            )
        if card_data_cache_path.stat().st_size == 0:
            raise CardDataCacheEmptyError(
                f'card data cache file exists at "{card_data_cache_path.name}", but contains no data'
            )

        card_data_index: dict[str, dict] = {}
        with open(card_data_cache_path, "r") as cache_file:
            for line in cache_file:
                record = json.loads(line)
                card_data_index[record["id"]] = record

        with open(card_data_index_path, "w") as file:
            json.dump(card_data_index, file, indent=2, ensure_ascii=False)

        elapsed = time.perf_counter() - start_time
        print(f"card data index built in {elapsed:.1f}s ({len(card_data_index)} cards)")

    def build_keyword_index(self) -> None:
        """Parse the offical rules text to extract keyword definitons and convert to a
        dict keyed by keyword, persist as JSON"""
        start_time = time.perf_counter()

        rules_text_path = RAW_CACHE_DIR / "rules.txt"
        keyword_index_path = INDEX_CACHE_DIR / "keyword_index.json"
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # first, check that the rules file exists to build the index from
        if not rules_text_path.exists():
            raise RulesCacheMissingError(
                f'rules file does not exist, expected at "{rules_text_path.name}"'
            )
        if rules_text_path.stat().st_size == 0:
            raise RulesCacheEmptyError(
                f'rules file exists at "{rules_text_path.name}", but contains no data'
            )

        # Find the beginning and end of the 'Glossary' section, as this
        # contains straight definitions of all the words we'd want to look up
        lines = rules_text_path.read_text().split("\n")
        glossary_start = max(i for i, line in enumerate(lines) if line == "Glossary")
        glossary_end = max(i for i, line in enumerate(lines) if line == "Credits")
        block = lines[glossary_start + 1 : glossary_end]

        # group lines into entries for the index dictionary
        keyword_index: dict[str, str] = {}
        current: list[str] = []
        for line in block:
            # if the line is blank, this is the end of the current entry,
            # if we've written lines to current then it's good to go;
            # the first line is the keyword, so use this as the key then
            # join the rest into a single sentence to form the value
            if line.strip() == "":
                if current:
                    keyword_index[current[0]] = " ".join(current[1:])
                    current = []
                continue
            # the source text uses U+2028 (line separator) to force a manual
            # line break within a definition, not to mark a new entry. We
            # do not need this and it muddies up the index files. Treat
            # it as a space so it doesn't leak into the stored definition
            current.append(line.replace(" ", " "))

        # polish off the final entry once we've exited the loop
        if current:
            keyword_index[current[0]] = " ".join(current[1:])

        # write to disk
        with open(keyword_index_path, "w") as file:
            json.dump(keyword_index, file, indent=2, ensure_ascii=False)

        elapsed = time.perf_counter() - start_time
        print(f"keyword index built in {elapsed:.1f}s ({len(keyword_index)} keywords)")

    def build_index(self, index_type: IndexType) -> None:
        """Build the requested index (phash, rulings, name, card data, or keyword) by type."""
        match index_type:
            case IndexType.PHASH:
                self.build_phash_index()
            case IndexType.RULINGS:
                self.build_rulings_index()
            case IndexType.NAME:
                self.build_name_index()
            case IndexType.CARD_DATA:
                self.build_card_data_index()
            case IndexType.KEYWORD:
                self.build_keyword_index()
            case _:
                raise InvalidIndexTypeError(
                    f'invalid index type argument "{index_type}"'
                )
