import json
from pathlib import Path

from .chat.agent_tools import AgentTools
from .chat.chat_session import ChatSession
from .chat.conversation import Conversation
from .chat.terminal_chat_view import TerminalChatView
from .datastore.cache_builder import CacheBuilder, CacheType
from .datastore.index_builder import IndexBuilder, IndexType
from .datastore.index_lookup import IndexLookup
from .datastore.index_store import IndexStore
from .recognition.recognition_pipeline import RecognitionPipeline

MODEL = "anthropic/claude-haiku-4.5"

# local test outputs go here
output_dir = Path("test_output")

MENU = """
1) identify card
2) identify card and ask an LLM about it
3) run cache generation
4) run index generation
q) quit
"""


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
    builder.build_index(IndexType.PHASH)


def ask_llm_about_card(
    index_lookup: IndexLookup, recognition_pipeline: RecognitionPipeline
) -> None:
    """Identify a card, then send the resulting payload to an LLM as a smoke test."""
    payload = recognition_pipeline.run()
    if payload is None:
        print("no confident match - nothing to send")
        return

    session = ChatSession(
        conversation=Conversation(),
        view=TerminalChatView(),
        tools=AgentTools(index_lookup),
        model=MODEL,
    )
    session.launch_chat_session(payload)


def main() -> None:
    # load every index once, then share it across whatever needs it
    index_store = IndexStore()
    index_lookup = IndexLookup(index_store)

    # init recognition pipeline
    recognition_pipeline = RecognitionPipeline(index_store)

    while True:
        print(MENU)
        choice = input("> ").strip().lower()
        if choice == "1":
            print(json.dumps(recognition_pipeline.run(), indent=2, ensure_ascii=False))
        elif choice == "2":
            ask_llm_about_card(index_lookup, recognition_pipeline)
        elif choice == "3":
            run_cache_generation()
        elif choice == "4":
            run_index_generation()
        elif choice == "q":
            break
        else:
            print(f"unrecognised option: '{choice}'")


if __name__ == "__main__":
    main()
