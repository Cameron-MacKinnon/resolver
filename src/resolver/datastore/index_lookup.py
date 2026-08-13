from rapidfuzz import fuzz, process, utils

from .index_store import IndexStore


class IndexLookup:
    def __init__(self, index_store: IndexStore) -> None:
        self.index_store = index_store

    def _fuzzy_match_card_name(self, card_name: str) -> str | None:
        """Given a string, returns the closest matching name from the index."""
        result = process.extractOne(
            card_name,
            self.index_store.name_index.keys(),
            scorer=fuzz.WRatio,
            processor=utils.default_process,  # lowercase + strip punctuation
            score_cutoff=90,
        )
        if result is None:
            return None
        matched_name, _score, _index = result
        return self.index_store.name_index[matched_name]

    def get_card_data_by_id(self, scryfall_id: str) -> dict | None:
        """Retrieve a specific card's data from its globally unique scryfall id"""
        return self.index_store.card_data_index.get(scryfall_id)

    def get_card_data_by_oracle_id(self, oracle_id: str) -> dict | None:
        """Retrieve a card's data using its card-specific (shared across all versions) ID"""
        scryfall_id = self.index_store.oracle_id_index.get(oracle_id)
        if scryfall_id is None:
            return None

        # oracle_id_index and card_data_index are built from the same source at
        # the same time, so a scryfall_id it returns is guaranteed to exist in
        # card_data_index. None here means the indexes are somehow out of
        # sync with each other, not that this card doesn't exist
        card_data = self.get_card_data_by_id(scryfall_id)
        assert card_data is not None, (
            f"oracle_id_index points to unknown id {scryfall_id!r} "
            f"for oracle_id {oracle_id!r} - indexes are out of sync"
        )
        return card_data

    def get_card_data_by_name(self, card_name: str) -> dict | None:
        """Retrieve a card's data using its name, fallback to fuzzy match once
        if exact match is non-existent"""
        oracle_id = self.index_store.name_index.get(card_name)
        if oracle_id is None:
            oracle_id = self._fuzzy_match_card_name(card_name)
        if oracle_id is None:
            return None
        return self.get_card_data_by_oracle_id(oracle_id)

    def get_keyword_definition(self, keyword: str) -> str | None:
        """For a given keyword, return the definition as per the official rules"""
        return self.index_store.keyword_index.get(keyword)

    def get_rule(self, rule_number: str) -> dict | None:
        """For a given rule number (e.g. "702.4a"), return its entry from the
        official rules: text, plus the section it belongs to."""
        return self.index_store.rule_index.get(rule_number)

    def get_rule_children(self, rule_number: str) -> list[dict]:
        """Return the direct subrules of a given rule number (e.g. "702.4" ->
        "702.4a", "702.4b", ...), letting a caller walk the rule tree downward.

        Matches on rule_number immediately followed by a letter, not a digit,
        so "100.1" doesn't pick up the unrelated rule "100.10".
        """
        return [
            entry
            for number, entry in self.index_store.rule_index.items()
            if number.startswith(rule_number) and number[len(rule_number) :].isalpha()
        ]

    def get_rule_tree(self, rule_number: str) -> list[dict]:
        """Return a rule's own entry (if any) plus every rule/subrule beneath
        it at any depth, in document order - e.g. "510" returns 510 itself
        (if indexed), 510.1, 510.1a, 510.1b, 510.2, etc. Lets a caller pull
        a whole section or rule's full picture in one call instead of
        walking level by level with get_rule/get_rule_children themselves.

        A number is a descendant if it extends rule_number with ".<digits>"
        (one level down to a rule) and/or "<letters>" (one level down to a
        subrule) - checking only the first extension step is enough to
        catch descendants at any depth, since that's the full extent of the
        section/rule/subrule grammar.
        """
        descendants = []
        for number, entry in self.index_store.rule_index.items():
            if not number.startswith(rule_number) or number == rule_number:
                continue
            remainder = number[len(rule_number) :]
            if (remainder[0] == "." and remainder[1:2].isdigit()) or remainder[
                0
            ].isalpha():
                descendants.append(entry)

        self_entry = self.index_store.rule_index.get(rule_number)
        return ([self_entry] if self_entry is not None else []) + descendants

    def get_card_rulings(self, oracle_id: str) -> list[dict]:
        """For a given oracle_id, return all associated official rulings assosciated with this card"""
        return self.index_store.rulings_index.get(oracle_id, [])

    def get_card_context(self, scryfall_id: str) -> dict[str, dict | list] | None:
        """Build a single payload of everything useful about a card: its
        details, rulings, and definitions for any keywords it has.

        card_data_index stores already-normalized records (see
        IndexBuilder.build_card_data_index), so every field here is a
        plain, safe lookup - no multi-faced-card handling needed at
        this layer.
        """
        card_data = self.get_card_data_by_id(scryfall_id)
        if card_data is None:
            return None

        # get keyword definitions (if any), skipping any keyword names
        # that don't have an exact match in the glossary
        keyword_definitions: dict[str, str] = {}
        for keyword in card_data["keywords"]:
            definition = self.get_keyword_definition(keyword)
            if definition is not None:
                keyword_definitions[keyword] = definition

        return {
            "card_details": {
                "name": card_data["name"],
                "color_identity": card_data["color_identity"],
                "mana_cost": card_data["mana_cost"],
                "type_line": card_data["type_line"],
                "oracle_text": card_data["oracle_text"],
            },
            "keyword_definitions": keyword_definitions,
            "rulings": self.get_card_rulings(card_data["oracle_id"]),
            "ids": {
                "id": card_data["id"],
                "oracle_id": card_data["oracle_id"],
            },
        }
