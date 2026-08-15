import json

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from .chat.agent_tools import AgentTools
from .chat.chat_session import ChatSession
from .chat.conversation import Conversation
from .chat.terminal_chat_view import TerminalChatView
from .datastore.cache_builder import CacheBuilder, CacheType
from .datastore.index_builder import IndexBuilder, IndexType
from .datastore.index_lookup import IndexLookup
from .datastore.index_store import IndexStore
from .recognition.recognition_pipeline import RecognitionPipeline

# Model of choice for chat
MODEL = "anthropic/claude-sonnet-5"

# Displayed on app startup
STARTUP_BANNER = Panel(
    Group(
        Align.center(Text("R E S O L V E R", style="bold bright_magenta")),
        Align.center(
            Text(
                "MTG card recognition & rules assistant",
                style="dim",
            )
        ),
    ),
    border_style="bright_magenta",
    box=box.DOUBLE,
    padding=(1, 2),
)

# Displayed during main menu loop
MENU = (
    "\n"
    "[bold cyan]1[/]) Identify card\n"
    "[bold cyan]2[/]) Identify and chat\n"
    "[bold cyan]3[/]) Chat to resolver\n"
    "[bold cyan]4[/]) Run cache generation\n"
    "[bold cyan]5[/]) Run index generation\n"
    "[bold cyan]q[/]) Quit\n"
)

# Init rich console
console = Console()


def run_cache_generation() -> None:
    """Fetch and persist every raw cache from Scryfall/WotC."""
    builder = CacheBuilder()
    builder.build_cache(CacheType.CARD_DATA)
    builder.build_cache(CacheType.RULINGS)
    builder.build_cache(CacheType.IMAGES)
    builder.build_cache(CacheType.GLOSSARY)


def run_index_generation() -> None:
    """Build every runtime-queryable index from the raw caches."""
    builder = IndexBuilder()
    builder.build_index(IndexType.CARD_DATA)
    builder.build_index(IndexType.NAME)
    builder.build_index(IndexType.ORACLE_ID)
    builder.build_index(IndexType.RULINGS)
    builder.build_index(IndexType.KEYWORD)
    builder.build_index(IndexType.RULE)
    builder.build_index(IndexType.PHASH)


def ask_llm_about_card(
    index_lookup: IndexLookup, recognition_pipeline: RecognitionPipeline
) -> None:
    """Identify a card and use it's data to seed a chat with the reoslver LLM."""
    # identify card and get card data
    payload = recognition_pipeline.run()
    if payload is None:
        print("No confident match, check conditions and try again")
        return

    # start chat session
    session = ChatSession(
        conversation=Conversation(),
        view=TerminalChatView(),
        tools=AgentTools(index_lookup),
        model=MODEL,
    )
    session.launch_chat_session(payload)


def start_chat(index_lookup: IndexLookup) -> None:
    """Launch a chat session with the resolver LLM."""
    session = ChatSession(
        conversation=Conversation(),
        view=TerminalChatView(),
        tools=AgentTools(index_lookup),
        model=MODEL,
    )
    session.launch_chat_session()


def main() -> None:
    # display startup banner and announce app start
    console.print(STARTUP_BANNER)
    console.print("[dim bold magenta]Initialising...[/]")

    # load every index once, then share it across whatever needs it
    index_store = IndexStore()
    index_lookup = IndexLookup(index_store)

    # init recognition pipeline
    recognition_pipeline = RecognitionPipeline(index_store)

    # display init complete message
    console.print("[dim bold magenta]Initialisation complete[/]")

    while True:
        console.print(MENU)
        choice = console.input("[bold cyan]>[/] ").strip().lower()
        if choice == "1":
            print(json.dumps(recognition_pipeline.run(), indent=2, ensure_ascii=False))
        elif choice == "2":
            ask_llm_about_card(index_lookup, recognition_pipeline)
        elif choice == "3":
            start_chat(index_lookup)
        elif choice == "4":
            run_cache_generation()
        elif choice == "5":
            run_index_generation()
        elif choice == "q":
            break
        else:
            console.print(f"[yellow]unrecognised option:[/] '{choice}'")


if __name__ == "__main__":
    main()
