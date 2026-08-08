import json
import statistics
import time

import imagehash
from imagehash import ImageHash

from .cache_config import INDEX_CACHE_DIR

# per https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/
# the closest candidate is only trusted once it's this many standard deviations
# below the mean of every other candidate's distance (I do not pretend to have
# thought of this myself...)
RECOGNITION_SIGMA_THRESHOLD = 4


class Comparator:
    def __init__(self) -> None:
        self.hash_index = self._load_index()

    def _load_index(self) -> list[dict]:
        """Open the index file and read it into a list of dicts, convert hex
        hashes back into ImageHash objects."""
        # start performance timer
        start_time = time.perf_counter()
        print("loading image hash index...")

        # load JSONL cache to dict
        hash_index_path = INDEX_CACHE_DIR / "hash_index.jsonl"
        with open(hash_index_path, "r") as file:
            hash_index: list[dict] = [json.loads(line) for line in file]

        # convert hexadecimal representations of hashes back into ImageHash objs
        for entry in hash_index:
            hash_obj = imagehash.hex_to_hash(entry["hash"])
            entry["hash"] = hash_obj

        # finish timer and print conclusion stats
        elapsed = time.perf_counter() - start_time
        print(f"image hash index loaded in {elapsed:.1f}s")

        return hash_index

    def _deduped_distances(self, search_hash: ImageHash) -> dict[str, int]:
        """Compute the hash distance from search_hash to every cached entry,
        collapsed down to the single closest distance per unique card id.

        A card can have several cached hash entries (one per image variant,
        one per face for multi-faced cards), so without deduping, cards with
        more entries would be over-represented in any statistics computed
        over this population.
        """
        distances: dict[str, int] = {}
        for entry in self.hash_index:
            # ImageHash subtraction returns numpy.int64, which the stdlib
            # statistics module can't handle, so we cast to a plain int up front
            distance = int(search_hash - entry["hash"])
            id_ = entry["id"]
            if id_ not in distances or distance < distances[id_]:
                distances[id_] = distance
        return distances

    def best_match(self, search_hash: ImageHash) -> dict | None:
        """Find the single best-matching card via statistical outlier scoring,
        or None if no candidate is confidently recognized.

        Compares the closest candidate's distance to the mean/stddev of every
        other candidate's distance. A candidate is only trusted if it's more
        than RECOGNITION_SIGMA_THRESHOLD standard deviations below the mean,
        i.e., a genuine statistical outlier, not just marginally closer than a
        cluster of similarly-plausible candidates. The returned score is that
        separation normalized by RECOGNITION_SIGMA_THRESHOLD standard
        deviations, so a score >= 1.0 means the candidate cleared the bar.
        """
        # get deduped list of all distance
        distances = self._deduped_distances(search_hash)

        # need at least two distinct candidates to compute a meaningful mean/stddev
        if len(distances) < 2:
            return None

        # get the entry with the lowest hamming distance (best match)
        ranked = sorted(distances.items(), key=lambda item: item[1])
        best_id, best_distance = ranked[0]
        rest = [distance for _, distance in ranked[1:]]

        # calculate mean, standad deviation of all candidates other than
        # the closest match
        mean = statistics.mean(rest)
        stdev = statistics.stdev(rest)
        if stdev == 0:
            return None

        # determine if the best fit is a genuine statistical outlier, return None
        # if this was not convincing enough to be anything other than a guess
        score = (mean - best_distance) / (RECOGNITION_SIGMA_THRESHOLD * stdev)
        if score < 1.0:
            return None

        return {"id": best_id, "distance": best_distance, "score": score}
