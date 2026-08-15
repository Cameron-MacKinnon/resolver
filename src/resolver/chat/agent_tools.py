import json
from typing import Any, Callable, TypedDict

from openrouter.components import ChatFunctionToolFunctionTypedDict

from ..datastore.index_lookup import IndexLookup


class _ToolEntry(TypedDict):
    """The full definition of a single agent-callable tool: the function to
    run, the description/parameters schema shown to the model, and a plain-
    English label for our own display purposes (e.g. logging tool usage).
    Kept as one entry (rather than parallel dicts) so a tool can't exist in
    dispatch() without also existing in schemas(), or vice versa."""

    callable: Callable[..., Any]
    description: str
    parameters: dict[str, Any]
    label: str


class AgentTools:
    """Exposes a subset of IndexLookup's methods to the LLM as callable
    tools. Advertises them via schemas() and executes them via dispatch()."""

    def __init__(self, index_lookup: IndexLookup) -> None:
        # define functions callable to an LLM agent
        self._tools: dict[str, _ToolEntry] = {
            "get_card_data_by_name": {
                "callable": index_lookup.get_card_data_by_name,
                "label": "Looking up card information",
                "description": (
                    "Look up a card's basic details (name, mana cost, type, "
                    "oracle text, id) by its exact name. This alone is not "
                    "enough to properly explain a card the user is asking "
                    "about for the first time - follow up with "
                    "get_card_context using the returned id to also get its "
                    "rulings and keyword definitions before answering."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"card_name": {"type": "string"}},
                    "required": ["card_name"],
                },
            },
            "get_card_context": {
                "callable": index_lookup.get_card_context,
                "label": "Gathering full card context",
                "description": (
                    "Get full context for a card - details, rulings, and "
                    "keyword definitions - by its scryfall id. Use this "
                    "whenever you're actually explaining a card to the user, "
                    "not just its basic data. Requires the card's scryfall "
                    "id (the 'id' field) - call get_card_data_by_name first "
                    "if you don't already have it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"scryfall_id": {"type": "string"}},
                    "required": ["scryfall_id"],
                },
            },
            "get_keyword_definition": {
                "callable": index_lookup.get_keyword_definition,
                "label": "Looking up keyword definition",
                "description": (
                    "Get the official rules definition of a keyword ability "
                    "(e.g. Flying, Trample, Vigilance). Always call this "
                    "when the user asks what a keyword means or how it "
                    "works, even a common one you already know - don't "
                    "answer from memory, since exact wording and edge cases "
                    "matter and this returns the authoritative text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"keyword": {"type": "string"}},
                    "required": ["keyword"],
                },
            },
            "get_rule_tree": {
                "callable": index_lookup.get_rule_tree,
                "label": "Checking the official game rules",
                "description": (
                    "Get the official text of a numbered rule (e.g. '510', "
                    "'702.4a'), plus every rule and subrule beneath it in "
                    "one call - e.g. '510' returns 510 itself (if it has "
                    "text), 510.1, 510.1a, 510.1b, 510.2, and so on. Card "
                    "text, keyword definitions, and rulings often reference "
                    "a specific rule number ('see rule 510, \"Combat Damage "
                    "Step\"') - call this to resolve exactly what it (and "
                    "everything under it) says rather than guessing from "
                    "memory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"rule_number": {"type": "string"}},
                    "required": ["rule_number"],
                },
            },
            "get_card_rulings": {
                "callable": index_lookup.get_card_rulings,
                "label": "Checking for card rulings",
                "description": (
                    "Get official rulings for a card. Requires the card's "
                    "oracle_id - call get_card_data_by_name first if you "
                    "don't already have it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"oracle_id": {"type": "string"}},
                    "required": ["oracle_id"],
                },
            },
        }

    def schemas(self) -> list[ChatFunctionToolFunctionTypedDict]:
        """Build the `tools` payload for send(), one entry per registered
        tool, in the JSON-schema shape the model expects to see."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for name, tool in self._tools.items()
        ]

    def label_for(self, name: str) -> str:
        """Return the plain-English label for a registered tool, for display
        purposes (e.g. logging which tool is being called) rather than
        showing the raw function name to the user."""
        return self._tools[name]["label"]

    def dispatch(self, name: str, arguments_json: str) -> str:
        """Run the named tool with the model-supplied arguments (a JSON string
        of kwargs) and return a JSON string result, ready to feed straight
        back into conversation memory as a "tool" role message."""
        kwargs = json.loads(arguments_json)
        result = self._tools[name]["callable"](**kwargs)
        if result is None:
            return "no result found for the given arguments"
        return json.dumps(result)
