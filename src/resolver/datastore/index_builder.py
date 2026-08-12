import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import cv2

from ..recognition.hasher import Hasher
from .cache_config import IMAGE_VARIANTS, INDEX_CACHE_DIR, RAW_CACHE_DIR


class IndexType(Enum):
    PHASH = "phash"
    RULINGS = "rulings"
    NAME = "name"
    CARD_DATA = "card_data"
    KEYWORD = "keyword"
    ORACLE_ID = "oracle_id"


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


@dataclass
class CardRecord:
    id: str
    oracle_id: str
    name: str
    oracle_text: str
    type_line: str
    mana_cost: str
    color_identity: list[str]
    keywords: list[str]


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

    def _get_names(self, record: dict) -> list[str]:
        """Return one card name per face of the card.

        Most cards have a single, useable top-level name. Multi-faced cards
        have a top-level name too, but it's a combined "Face A // Face B"
        string, not ideal for fuzzy-matching against directly - each face's
        own name is returned separately instead.
        """
        if "card_faces" in record:
            return [face["name"] for face in record["card_faces"]]
        return [record["name"]]

    def _get_oracle_id(self, record: dict) -> str | None:
        """Return the record's oracle_id, or None if it has none.

        Unlike name/oracle_text, most multi-faced layouts keep a valid
        oracle_id at the top level (it identifies the whole card, not a
        face) - only reversible_card doesn't, so we only fall back to the
        first face's oracle_id when the top-level one is genuinely missing.
        """
        oracle_id = record.get("oracle_id")
        if oracle_id is not None:
            return oracle_id
        if "card_faces" in record:
            return record["card_faces"][0].get("oracle_id")
        return None

    def _get_oracle_text(self, record: dict) -> str:
        """Return the record's oracle text, or an empty string if it has none.

        Joins all faces with a blank-line separator to keep each face's text
        visually distinct. Some non-gameplay objects sharing a name with a real
        card (e.g. art series cards) have no real text anywhere, the empty-string
        result can be used to filter those out.
        """
        if "card_faces" in record:
            return "\n\n".join(
                face.get("oracle_text") or "" for face in record["card_faces"]
            )
        return record.get("oracle_text") or ""

    def _get_type_line(self, record: dict) -> str:
        """Return the record's type line, or an empty string if it has none.

        Reversible_card is the only layout missing a top-level type_line,
        each face carries its own instead. Most reversible_card printings are
        the same card on both sides (just a different border/treatment), so
        identical faces are collapsed down to one value; a handful (Omen
        cards) genuinely have two different faces, and those get joined with
        " // " to represent both.
        """
        type_line = record.get("type_line")
        if type_line is not None:
            return type_line
        if "card_faces" not in record:
            return ""
        face_type_lines = [face.get("type_line") or "" for face in record["card_faces"]]
        unique_type_lines = dict.fromkeys(face_type_lines)
        return " // ".join(unique_type_lines)

    def _get_mana_cost(self, record: dict) -> str:
        """Return the record's mana cost, or an empty string if it has none.

        Several layouts (transform/modal_dfc/reversible_card/double_faced_token/
        art_series) have no top-level mana_cost, each face carries its own
        instead. The castable cost, when a card has one, is always on the
        first face - other faces (a transform's back, a token, art) are empty.
        """
        mana_cost = record.get("mana_cost")
        if mana_cost is not None:
            return mana_cost
        if "card_faces" not in record:
            return ""
        return record["card_faces"][0].get("mana_cost") or ""

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
        """Map each card name to its oracle_id and persist as a single JSON file.

        WoTC enforces name uniqueness for real gameplay cards, so a name
        maps to exactly one oracle_id in practice, and every reprint of a
        card shares the same oracle_i. The one exception is non-gameplay
        objects that share a name with a real card (e.g. art series cards,
        which have their own distinct oracle_id and no real oracle text);
        when that happens we keep whichever record actually has usable oracle
        text, since a name lookup should always resolve to the real playable card.
        """
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

        # loop over jsonl card data cache, keeping one oracle_id per name;
        # prefer whichever record has usable oracle text if there's a clash
        name_index: dict[str, str] = {}
        name_index_records: dict[str, dict] = {}
        with open(card_data_cache_path, "r") as cache_file:
            for line in cache_file:
                data = json.loads(line)
                oracle_id = self._get_oracle_id(data)
                if oracle_id is None:
                    continue

                for name in self._get_names(data):
                    existing = name_index_records.get(name)
                    if existing is None or (
                        not self._get_oracle_text(existing)
                        and self._get_oracle_text(data)
                    ):
                        name_index_records[name] = data
                        name_index[name] = oracle_id

        # write the built index to disk
        with open(name_index_path, "w") as file:
            json.dump(name_index, file, indent=2, ensure_ascii=False)

        elapsed = time.perf_counter() - start_time
        print(f"name index built in {elapsed:.1f}s ({len(name_index)} unique names)")

    def build_card_data_index(self) -> None:
        """Load the card data cache into a dict keyed by id and persist as JSON.

        Each record is normalized into a CardRecord rather than stored
        verbatim - multi-faced cards are inconsistent about which fields
        live at the top level versus nested per face, and it varies field
        by field, so every printing gets resolved once here rather than
        leaving every downstream reader to handle it defensively.
        """
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

                oracle_id = self._get_oracle_id(record)
                assert oracle_id is not None, (
                    f"card {record['id']!r} has no resolvable oracle_id"
                )

                card_record = CardRecord(
                    id=record["id"],
                    oracle_id=oracle_id,
                    name=record["name"],
                    oracle_text=self._get_oracle_text(record),
                    type_line=self._get_type_line(record),
                    mana_cost=self._get_mana_cost(record),
                    color_identity=record["color_identity"],
                    keywords=record["keywords"],
                )
                card_data_index[card_record.id] = asdict(card_record)

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

    def build_oracle_id_index(self) -> None:
        """Map each oracle_id to a single representative id and persist as a JSON file.

        card_data_index stays keyed by id so every printing's data is kept;
        this is the bridge that lets a lookup starting from an oracle_id
        (e.g. via name_index) resolve to one usable id."""
        start_time = time.perf_counter()

        card_data_cache_path = RAW_CACHE_DIR / "card_data.jsonl"
        oracle_id_index_path = INDEX_CACHE_DIR / "oracle_id_index.json"
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

        # iterate over the card data and filter down to a single representative id
        # for each oracle_id (chosen id must have valid oracle text otherwise it's useless)
        oracle_id_index: dict[str, str] = {}
        oracle_id_records: dict[str, dict] = {}
        with open(card_data_cache_path, "r") as cache_file:
            for line in cache_file:
                data = json.loads(line)
                oracle_id = self._get_oracle_id(data)
                if oracle_id is None:
                    continue

                existing = oracle_id_records.get(oracle_id)
                if existing is None or (
                    not self._get_oracle_text(existing) and self._get_oracle_text(data)
                ):
                    oracle_id_records[oracle_id] = data
                    oracle_id_index[oracle_id] = data["id"]

        # write the built index to disk
        with open(oracle_id_index_path, "w") as file:
            json.dump(oracle_id_index, file, indent=2, ensure_ascii=False)

        elapsed = time.perf_counter() - start_time
        print(
            f"oracle_id index built in {elapsed:.1f}s ({len(oracle_id_index)} oracle_ids)"
        )

    def build_index(self, index_type: IndexType) -> None:
        """Build the requested index (phash, rulings, name, card data, keyword, or oracle_id) by type."""
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
            case IndexType.ORACLE_ID:
                self.build_oracle_id_index()
            case _:
                raise InvalidIndexTypeError(
                    f'invalid index type argument "{index_type}"'
                )
