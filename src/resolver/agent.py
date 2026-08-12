import json
import os
from typing import Literal, cast

from dotenv import load_dotenv
from openrouter import OpenRouter
from openrouter.components import (
    ChatAssistantMessageTypedDict,
    ChatSystemMessageTypedDict,
    ChatUserMessageTypedDict,
)
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .cache_config import PROJECT_ROOT
from .index_store import IndexStore

Role = Literal["system", "user", "assistant"]
ChatMessage = ChatSystemMessageTypedDict | ChatUserMessageTypedDict | ChatAssistantMessageTypedDict


class OpenrouterAgentError(Exception):
    """Base exception for OpenrouterAgent"""


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


class OpenRouterAgent:
    def __init__(
        self, index_store: IndexStore, model: str = "anthropic/claude-haiku-4.5"
    ) -> None:
        self.model = model
        self.index_store = index_store
        self.start_of_conversation = True
        self.console = Console()

        # set / validate API key
        load_dotenv(PROJECT_ROOT / ".env")
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if self.api_key is None:
            raise OpenrouterAgentError(
                'API key not set -add it to ".env" at project root'
            )

        # init chat memory
        self.memory: list[ChatMessage] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def add_to_memory(self, role: Role, content: str) -> None:
        """Append a new message to the agent's local memory state"""
        entry = cast(ChatMessage, {"role": role, "content": content})
        self.memory.append(entry)

    def _send(self, client: OpenRouter) -> str:
        """Send the current memory to the model (with a spinner while waiting)
        and return the reply text."""
        with self.console.status("[bold cyan]Thinking...[/]"):
            response = client.chat.send(model=self.model, messages=self.memory)
        if not response.choices:
            self.console.print(json.dumps(response.model_dump(), indent=2))
            raise OpenrouterAgentError("no response content returned by model")
        return str(response.choices[0].message.content)

    def _render_reply(self, reply: str) -> None:
        """Print an assistant reply in a bordered panel, bolding any ALL-CAPS
        section labels (SUMMARY, KEYWORDS, RULINGS, ...) the system prompt
        asks the model to structure its answer with."""
        body = Text()
        for line in reply.splitlines():
            stripped = line.strip()
            if stripped and stripped.isupper() and len(stripped) <= 30:
                body.append(stripped, style="bold cyan")
            else:
                body.append(line)
            body.append("\n")

        self.console.print(
            Panel(
                body,
                title="[bold magenta]AGENT[/]",
                title_align="left",
                border_style="magenta",
            )
        )

    def launch_chat_session(self, payload: dict[str, str | int]) -> None:
        """Begin an agentic chat session, continue until user terminates or context
        reaches limit."""
        # print model context message
        self.console.print(
            f'[dim]Chat session initiated with model "{self.model}", type "exit" '
            'or "quit" to end session[/]'
        )

        # enter loop
        with OpenRouter(api_key=self.api_key) as client:
            while True:
                try:
                    # if this is the first iteration, feed in the card payload
                    # and get the agents initial summary message
                    if self.start_of_conversation:
                        self.add_to_memory(role="user", content=json.dumps(payload))
                        reply = self._send(client)
                        self._render_reply(reply)
                        self.add_to_memory(role="assistant", content=reply)
                        self.start_of_conversation = False

                    # capture user input, check for exit condition, add to memory
                    self.console.print(Rule(style="dim"))
                    user_input = self.console.input("[bold cyan]YOU[/] > ")
                    if user_input.lower().strip() in ("q", "quit", "exit"):
                        break
                    if not user_input:
                        continue
                    self.add_to_memory(role="user", content=user_input)

                    # invoke LLM and capture reply (configurable to return multiple
                    # response options, we're not doing so, so just grab 1st)
                    reply = self._send(client)
                    self._render_reply(reply)
                    self.add_to_memory(role="assistant", content=reply)
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Session interrupted by user. Exiting...[/]")
                    break
                except Exception as e:
                    self.console.print(f"\n[bold red]An error occurred:[/] {e}\n")
                    break
