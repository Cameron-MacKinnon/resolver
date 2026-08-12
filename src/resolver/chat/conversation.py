from typing import Literal, cast

from openrouter.components import (
    ChatAssistantMessageTypedDict,
    ChatSystemMessageTypedDict,
    ChatToolMessageTypedDict,
    ChatUserMessageTypedDict,
)

Role = Literal["system", "user", "assistant", "tool"]
ChatMessage = (
    ChatSystemMessageTypedDict
    | ChatUserMessageTypedDict
    | ChatAssistantMessageTypedDict
    | ChatToolMessageTypedDict
)

SYSTEM_PROMPT = (
    "You are a Magic: The Gathering expert. Given the provided card context "
    "JSON - explain the card to the user in natural language. The user is "
    "holding the card infront of them so can already see the card's colours "
    "and mana cost, but is looking for guidance on how the card works and how "
    "they should use it.\n\n"
    "Before answering the user, consider if any of the card's keywords or rulings "
    "are particularly pertinent, obscure, or difficult to understand, and factor "
    "this into the following:\n\n"
    "First, provide a concise summary designed to introduce the card and make it "
    "sound as simple and digestible as possible. Following that, your reply should "
    "be sorted into plain-text sections using short, ALL-CAPS labels (e.g. STRATEGY, "
    "KEYWORDS, RULINGS) each followed by a blank line, so it's easy to skim straight "
    "to the part someone's interested in - no markdown formatting (no #, **, or - "
    "bullets), since this is displayed in a raw terminal.\n\n"
)


class Conversation:
    def __init__(self) -> None:
        self.memory: list[ChatMessage] = [
            cast(ChatMessage, {"role": "system", "content": SYSTEM_PROMPT})
        ]
        self.first_message = True

    def add_to_memory(
        self,
        role: Role,
        content: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict] | None = None,
    ) -> None:
        message: dict = {"role": role}
        if content is not None:
            message["content"] = content
        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        self.memory.append(cast(ChatMessage, message))
