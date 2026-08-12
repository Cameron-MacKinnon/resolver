import json
from typing import Any, Callable, TypedDict

from openrouter.components import ChatFunctionToolFunctionTypedDict

from ..datastore.index_lookup import IndexLookup


class _ToolEntry(TypedDict):
    """The full definition of a single agent-callable tool: the function to
    run, plus the description/parameters schema shown to the model. Kept as
    one entry (rather than parallel dicts) so a tool can't exist in dispatch()
    without also existing in schemas(), or vice versa."""

    callable: Callable[..., Any]
    description: str
    parameters: dict[str, Any]


class AgentTools:
    """Exposes a subset of IndexLookup's methods to the LLM as callable
    tools. Advertises them via schemas() and executes them via dispatch()."""

    def __init__(self, index_lookup: IndexLookup) -> None:
        # define functions callable to an LLM agent
        self._tools: dict[str, _ToolEntry] = {
            "get_card_data_by_name": {
                "callable": index_lookup.get_card_data_by_name,
                "description": "Look up a card's details by its exact name",
                "parameters": {
                    "type": "object",
                    "properties": {"card_name": {"type": "string"}},
                    "required": ["card_name"],
                },
            },
            "get_card_context": {
                "callable": index_lookup.get_card_context,
                "description": "Get full context for a card - details, rulings, and keyword definitions - by its scryfall id",
                "parameters": {
                    "type": "object",
                    "properties": {"scryfall_id": {"type": "string"}},
                    "required": ["scryfall_id"],
                },
            },
        }

    def schemas(self) -> list[ChatFunctionToolFunctionTypedDict]:
        """Build the `tools` payload for send() - one entry per registered
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

    def dispatch(self, name: str, arguments_json: str) -> str:
        """Run the named tool with the model-supplied arguments (a JSON string
        of kwargs) and return a JSON string result, ready to feed straight
        back into conversation memory as a "tool" role message."""
        kwargs = json.loads(arguments_json)
        result = self._tools[name]["callable"](**kwargs)
        if result is None:
            return "no result found for the given arguments"
        return json.dumps(result)
