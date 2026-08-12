import json
import statistics
import time
from dataclasses import dataclass

import imagehash
from imagehash import ImageHash

# per https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/
# the closest candidate is only trusted once it's this many standard deviations
# below the mean of every other candidate's distance (I do not pretend to have
# thought of this myself...)
RECOGNITION_SIGMA_THRESHOLD = 4


@dataclass
class MatchResult:
    id: str
    distance: int
    score: float


class Comparator:
    def __init__(self, encoded_hash_index: list[dict]) -> None:
        self.hash_index = self._preprocess_hash_index(encoded_hash_index)

    def _preprocess_hash_index(self, hash_index: list[dict]) -> list[dict]:
        """Convert hex encoded hashes back into ImageHash objects.

        Builds a fresh list of dicts rather than mutating hash_index in place,
        since it's caller-owned data (injected via the constructor) that may
        be shared/reused elsewhere.
        """
        return [
            {**entry, "hash": imagehash.hex_to_hash(entry["hash"])}
            for entry in hash_index
        ]

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

    def find_best_match(self, search_hash: ImageHash) -> MatchResult | None:
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

        return MatchResult(id=best_id, distance=best_distance, score=score)
