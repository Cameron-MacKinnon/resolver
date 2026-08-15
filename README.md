# Resolver

This project was devised out of a desire to streamline the deeply 
challening experience of figuring out how Magic: The Gathering (MTG) works.
To that end, I built this project - I call it Resolver (if you're reading
this, chances are you get what I'm going for with that).

Resolver is, at its core, a locally-built knowledge base of Magic: The
Gathering, it builds detailed local indexes of every card, ruling, keyword,
and numbered rule in the Comprehensive Rules, and pairs them with a custom
agentic harness that lets an LLM answer questions grounded exclusively in
that real data, rather than relying on its own internal knowledge (which is
inherently terrible and wholly unreliable when it comes to something as
complex as MTG). Show it a physical card with your webcam and it will
identify that card and explain it in natural language, or skip the camera
entirely and just chat with it directly about the rules.

Card recognition itself (perceptual hashing + statistical matching) is
adapted from existing prior art, not something I invented - see
[Acknowledgements](#acknowledgements) below. If I have achieved anything
there, it's by standing on the shoulders of giants. The indexing pipeline
and the LLM tool-harness are the real selling point of this project.

## By the numbers

A full local build indexes essentially the entire game, and makes _all_ of
this data available offline and queryable in milliseconds:

| | |
|---|---|
| Unique cards indexed | 54,122 |
| Unique card names | 35,857 |
| Official rulings | 77,998 (across 19,770 cards) |
| Comprehensive Rules entries | 3,162 (every numbered rule and subrule) |
| Keyword ability definitions | 739 |
| Card images hashed for recognition | 115,128 |
| Local cache size | ~11 GB |

## Project structure

```
resolver/
├── src/resolver/
│   ├── __main__.py               # CLI entry point / menu
│   ├── paths.py                  # project-root path resolution
│   ├── datastore/                # fetching, caching, and indexing MTG reference data
│   │   ├── scryfall_client.py        # Scryfall API client (bulk data + images)
│   │   ├── wotc_client.py            # scrapes the official Comprehensive Rules .txt
│   │   ├── cache_builder.py          # raw JSONL/image cache builder
│   │   ├── cache_config.py           # shared cache paths/constants
│   │   ├── index_builder.py          # builds runtime-queryable indexes from the raw cache
│   │   ├── index_store.py            # loads every index into memory
│   │   └── index_lookup.py           # query interface used by recognition + chat
│   ├── recognition/               # webcam -> matched card
│   │   ├── scanner.py                # camera capture
│   │   ├── frame_selector.py         # picks the sharpest frame from a burst
│   │   ├── card_detector.py          # finds and warps a card-shaped rectangle
│   │   ├── card_geometry.py          # physical card dimension constants
│   │   ├── hasher.py                 # perceptual hashing (phash)
│   │   ├── comparator.py             # statistical outlier hash matching
│   │   └── recognition_pipeline.py   # orchestrates the above
│   └── chat/                      # LLM chat agent
│       ├── conversation.py           # message history + system prompt
│       ├── agent_tools.py            # LLM-callable tool definitions
│       ├── chat_session.py           # send/tool-call loop orchestration
│       └── terminal_chat_view.py     # terminal rendering (rich)
├── docs/
│   └── ARCHITECTURE.md           # how it all actually works in detail
└── reference_cache/              # generated, gitignored, raw caches + indexes
```

## How it works

1. **Recognition** - a webcam frame is scanned for a card-shaped rectangle,
   flattened to a straight-on image, and matched against a local database of
   perceptual image hashes for every card Scryfall knows about.
2. **Lookup** - the matched card's data, official rulings, and any keyword
   ability definitions are pulled from local JSON indexes built ahead of time
   from Scryfall and Wizards of the Coast data - no network calls at
   recognition time.
3. **Chat** - that payload seeds a conversation with an LLM (via
   [OpenRouter](https://openrouter.ai)), which can call tools to look up
   further keywords, specific numbered rules, or rulings on demand rather
   than answering from its own memory.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the recognition
pipeline, local caching/indexing, and chat agent actually work.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Create a `.env` file in the project root with an
[OpenRouter](https://openrouter.ai) API key (only needed for the chat
features):

```
OPENROUTER_API_KEY=your-key-here
```

Before first use, build the local cache and indexes (fetches Scryfall/WotC
data, downloads card images, computes perceptual hashes - this takes a
while and needs about 11 GB of disk space):

```
uv run python -m resolver
```

...then choose **4) Run cache generation** followed by **5) Run index
generation** from the menu.

## Usage

```
uv run python -m resolver
```

- **1) Identify card** - scan a card via webcam and print its recognised
  data as JSON.
- **2) Identify and chat** - scan a card, then start a chat session seeded
  with that card's context.
- **3) Chat to resolver** - start a chat session with no card, e.g. to ask
  general rules questions.
- **4/5) Run cache/index generation** - re-run the data pipeline (see
  Setup above).

Recognition requires a connected webcam (device index `0`).

## Acknowledgements

The recognition pipeline - card detection, perceptual hashing, and matching
via statistical outlier scoring rather than just nearest-neighbor - is built
on the concept and approach laid out in
[tmikonen's magic-card-detector](https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/).
Go and read it, it's excellent, and this project wouldn't exist in its current
form without it, so [tmikonen](https://github.com/tmikonen), if you ever happen to
see this, thank you!

## Legal

This is an unofficial fan project, not affiliated with or endorsed by
Wizards of the Coast. It's shared as-is for personal/portfolio purposes - no
license is granted for reuse, modification, or distribution. Card data,
images, and rulings are sourced from [Scryfall](https://scryfall.com); the
Comprehensive Rules text is sourced from Wizards of the Coast. Magic: The
Gathering is a trademark of Wizards of the Coast LLC.
