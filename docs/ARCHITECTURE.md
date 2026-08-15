# Architecture

Three layers, each usable independently: a **datastore** that fetches and
indexes MTG reference data locally, a **recognition** pipeline that
identifies a physical card from a webcam frame, and a **chat** layer that
lets an LLM discuss a card (or the rules in general) grounded in that local
data via tool calls.

## Local caching and indexing (`src/resolver/datastore/`)

Everything the app knows comes from two one-time (well, periodically
re-run) pipelines, kept deliberately separate from runtime:

**`CacheBuilder`** is pure I/O - it fetches Scryfall's bulk card data and
rulings, downloads card images, and scrapes the current Comprehensive
Rules `.txt` from Wizards of the Coast, writing everything to
`reference_cache/raw/` verbatim (JSONL for data, `.jpg` for images, `.txt`
for rules). No parsing or normalization happens here. `ScryfallClient`
reuses a single `requests.Session` with a sized connection pool since
`build_image_cache` fetches tens of thousands of images concurrently via a
`ThreadPoolExecutor`; failed downloads are logged and skipped rather than
aborting the whole run, since a single bad URL out of tens of thousands
shouldn't kill an hours-long fetch.

**`IndexBuilder`** turns those raw caches into the actual runtime-queryable
indexes under `reference_cache/indexes/`, each a flat JSON(L) file:

| Index | Keyed by | Built from |
|---|---|---|
| `card_data_index` | scryfall id | `card_data.jsonl` |
| `name_index` | card name | `card_data.jsonl` |
| `oracle_id_index` | oracle_id | `card_data.jsonl` |
| `rulings_index` | oracle_id | `rulings.jsonl` |
| `keyword_index` | keyword name | `rules.txt` (Glossary section) |
| `rule_index` | rule number | `rules.txt` (numbered rules body) |
| `hash_index` | (id, face, variant) | `images/*.jpg` |

A few things worth knowing about specific indexes:

- **`card_data_index`** normalizes every printing into a flat `CardRecord`
  up front. Scryfall's card objects are inconsistent about which fields
  live at the top level vs. nested per-face for multi-faced cards (split
  cards, transforms, etc.) - `IndexBuilder` resolves that once here so
  every downstream reader gets a plain, safe record instead of handling
  every layout defensively.
- **`rule_index`** is a flat dict keyed by rule number (`"702.4a"`), each
  entry carrying the section it belongs to (`"702"`, `"Keyword Abilities"`). 
  There's no separate tree structure - a rule number's lineage is already 
  encoded in the string itself (`"100.1a"` descends from `"100.1"`, which 
  descends from section `"100"`), so `IndexLookup.get_rule_tree` walks it 
  by prefix-matching against the index's own keys rather than maintaining 
  parent/child pointers.
- **`hash_index`** is JSONL, not JSON, and append-only/resumable -
  `build_phash_index` skips any (id, face, variant) already hashed, so an
  interrupted run just picks up where it left off.

**`IndexStore`** loads every index into memory once at startup (raising a
clear error if a required index file is missing or empty - almost always
meaning the pipeline above hasn't been run yet). **`IndexLookup`** is the
actual query interface everything else uses - card lookups by id/oracle_id
/name (with a fuzzy-match fallback via `rapidfuzz`), keyword definitions,
rule lookups, rulings, and `get_card_context`, which bundles a card's
details + rulings + keyword definitions into the single payload both the
recognition pipeline and chat layer consume.

## Recognition pipeline (`src/resolver/recognition/`)

`RecognitionPipeline.run()` chains five stages, each its own class:

1. **`Scanner.capture_burst()`** - opens the webcam, discards a few stale
   buffered frames (`grab()`, cheap - reads without decoding), then reads
   and discards ~15 more frames as a warm-up so autofocus/auto-exposure can
   settle before capturing a burst of 10 real frames. A burst rather than a
   single shot exists because any one frame - including the last warm-up
   read - might still catch motion blur or a focus hunt.
2. **`FrameSelector.select_sharpest_image()`** - scores each frame's
   Laplacian variance (a standard blur-detection heuristic: sharp images
   have many strong, varied edges, so the variance of their Laplacian is
   higher than a blurry image's) and keeps the highest-scoring frame.
3. **`CardDetector.detect_cards()`** - resizes the frame, applies CLAHE
   contrast normalization (LAB color space, lightness channel only), then
   runs the standard find-a-rectangle pipeline: grayscale, blur, adaptive
   threshold, `findContours`, filter down to convex, high-solidity
   quadrilaterals matching a card's aspect ratio, then perspective-warp
   each candidate to a flat `630x880` upright image. Multiple candidates
   can come back from one frame (e.g. a card's outer border and inner
   printed border sometimes get detected as two separate contours).
4. **`Hasher.compute_hash()`** - computes a perceptual hash (phash) of each
   warped candidate at `hash_size=16` (256 bits) rather than imagehash's
   default 8 (64 bits) - needed to keep tens of thousands of cards
   distinguishable from each other.
5. **`Comparator.find_best_match()`** - compares that hash's Hamming
   distance against every cached hash, deduped to one (closest) distance
   per card id. A match is only trusted if the closest candidate is a
   genuine statistical outlier - more than `RECOGNITION_SIGMA_THRESHOLD`
   (4) standard deviations below the mean distance of every other
   candidate - not just marginally closer than a cluster of
   similarly-plausible guesses. This rejects low-confidence frames instead
   of always returning *some* answer.

Whichever candidate wins (if any) gets resolved through
`IndexLookup.get_card_context` and returned - the same payload shape a
chat tool call would get.

Note the whole card face is hashed, not just the illustration - a
different-language printing of the same card has a different text box and
therefore a genuinely different hash. This works fine for the currently
English-only cached image set; see the repo's notes if extending to other
languages, since it means every language you want recognized needs its own
cached image and hash entry, not just a translation layer.

## Agentic chatbot (`src/resolver/chat/`)

**`Conversation`** owns the message history and a system prompt that
frames the assistant's voice (a knowledgeable player explaining a card at
the table, not a rules bot dumping text) and, importantly, tells it to
ground any claim about a keyword, numbered rule, or ruling in text it's
actually seen - either already in the conversation (e.g. a
`get_card_context` payload) or freshly fetched via a tool call - rather
than answering from the model's own training-data memory.

**`AgentTools`** exposes a curated subset of `IndexLookup`'s methods as
LLM-callable tools (`get_card_data_by_name`, `get_card_context`,
`get_keyword_definition`, `get_rule_tree`, `get_card_rulings`), each with a
hand-written description telling the model *when* to call it, plus a
plain-English `label` used only for our own UI logging (never sent to the
model). `get_rule_tree` in particular returns a rule's own text *and*
every rule/subrule beneath it in one call, so the model can resolve a "see
rule 510" reference - including its lettered subrules - without several
round trips.

**`ChatSession`** drives the actual request loop: send the conversation to
the model, and if it comes back with tool calls instead of a plain answer,
dispatch each one, append the results to conversation memory, and send
again - repeating until a plain-text answer comes back. The first turn is
special-cased: if a card was recognized, its `get_card_context` payload
seeds the conversation and the model opens with a summary; the
no-card-scanned entry point just skips straight to waiting on the user's
own first message.

**`TerminalChatView`** is the only piece that knows about the terminal
(rich `Console`/`Panel`/`Status`) - kept separate from `ChatSession` so the
orchestration logic doesn't care how (or whether) output gets rendered.
