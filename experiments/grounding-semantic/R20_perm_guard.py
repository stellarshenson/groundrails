"""Derive the banked permutation-fingerprint set from disk instead of by hand.

The collision guard exists so a new draw cannot silently reuse an earlier draw's
data ordering; two draws sharing a permutation are not independent, and a mean
built from them overstates its confidence.

It was a hardcoded set widened by hand at each launch, and it was found
under-covering three times in round 20 - most sharply when R20-H174's own
wrapper omitted that arm's draws 1 and 2, so draw 3 could have collided with its
siblings unseen. Deriving the set from disk removes the manual step.

Two sources, unioned, because neither alone is complete:

* banked result JSONs - every finished draw records ``perm_fingerprint``
* campaign logs - a draw prints its fingerprint at launch, so in-flight draws
  that have not yet written a result are covered too. This is the case the
  hand-maintained list kept missing.

A fingerprint shared across DIFFERENT arms can be deliberate: R19-H168 and
R19-H169 reuse R18-H150 draw 1's ordering on purpose, holding the data order
fixed to isolate a trunk swap. Collision checking is therefore the caller's
decision about its own draws - this module only reports what is already on disk,
with provenance so the caller can tell intent from accident.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent.parent / "logs"

_LOG_PERM_RE = re.compile(r"perm fingerprint ([0-9a-f]{16})")


def derive_banked_perm_fps(extra=()):
    """Return {fingerprint: [provenance, ...]} for every draw visible on disk."""
    found = {}

    for path in sorted(HERE.glob("*_result.json")):
        try:
            doc = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(doc, dict):
            continue
        fp = doc.get("perm_fingerprint")
        if fp:
            found.setdefault(fp, []).append(path.name)

    for path in sorted(LOGS.glob("*.log")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for fp in set(_LOG_PERM_RE.findall(text)):
            found.setdefault(fp, []).append(path.name)

    for fp in extra:
        found.setdefault(fp, []).append("caller-supplied")

    return found


def assert_distinct(perm_fp, extra=(), label="this draw"):
    """Abort if `perm_fp` already exists on disk. Returns the derived set."""
    banked = derive_banked_perm_fps(extra)
    if perm_fp in banked:
        raise SystemExit(
            f"PERMUTATION COLLISION ABORT: {label} draws ordering {perm_fp}, "
            f"already banked by {banked[perm_fp]}. Two draws sharing a "
            f"permutation are not independent draws."
        )
    print(
        f"perm guard: {perm_fp} distinct from {len(banked)} fingerprints "
        f"derived from disk",
        flush=True,
    )
    return banked


if __name__ == "__main__":
    banked = derive_banked_perm_fps()
    print(f"{len(banked)} distinct permutation fingerprints on disk")
    for fp, where in sorted(banked.items()):
        shared = " SHARED" if len({w for w in where if w.endswith('.json')}) > 1 else ""
        print(f"  {fp}  {len(where)} source(s){shared}  {where[:3]}")
