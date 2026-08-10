import json
from pathlib import Path

from .cache_builder import CacheBuilder, CacheType
from .index_builder import IndexBuilder, IndexType
from .index_store import IndexStore
from .openrouter_client import OpenrouterClient
from .recognition_pipeline import RecognitionPipeline

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
    builder.build_index(IndexType.RULINGS)
    builder.build_index(IndexType.KEYWORD)
    builder.build_index(IndexType.PHASH)


def ask_llm_about_card(recognition_pipeline: RecognitionPipeline) -> None:
    """Identify a card, then send the resulting payload to an LLM as a smoke test."""
    payload = recognition_pipeline.run()
    if payload is None:
        print("no confident match - nothing to send")
        return

    model = "anthropic/claude-haiku-4.5"
    prompt = (
        "You are given JSON data for a Magic: The Gathering card someone is holding "
        "in their hand right now - they can already see its name, mana cost, type "
        "line, and oracle text, so don't repeat those back. Structure your reply "
        "into plain-text sections using short, ALL-CAPS labels (e.g. SUMMARY, "
        "KEYWORDS, RULINGS) each followed by a blank line, so it's easy to skim "
        "straight to the part someone's interested in - no markdown formatting "
        "(no #, **, or - bullets), since this is displayed in a raw terminal.\n\n"
        "SUMMARY: In 1-2 sentences, explain in plain English what the card "
        "actually does and how it plays, especially anything non-obvious.\n"
        "KEYWORDS: If the card has keyword abilities, briefly explain each one "
        "in plain English. Omit this section if there are none.\n"
        "RULINGS: If any rulings reveal a genuinely non-obvious interaction or "
        "common misplay, briefly mention it. Omit this section if nothing "
        "stands out.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    llm_client = OpenrouterClient()
    print(llm_client.send_prompt(prompt, model))


def main() -> None:
    # load every index once, then share it across whatever needs it
    index_store = IndexStore()

    # init recognition pipeline
    recognition_pipeline = RecognitionPipeline(index_store)

    while True:
        print(MENU)
        choice = input("> ").strip().lower()
        if choice == "1":
            print(json.dumps(recognition_pipeline.run(), indent=2, ensure_ascii=False))
        elif choice == "2":
            ask_llm_about_card(recognition_pipeline)
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
