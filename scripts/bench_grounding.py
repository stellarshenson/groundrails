"""Grounding throughput benchmark - the artefact behind the DEF-7/DEF-8 numbers.

Measures ``ground()`` per-claim latency cold (caches cleared per claim - the
CLI single-claim path) and warm (caches shared across the batch - the
``ground_batch`` path), across batch sizes, with repeats and spread. The
speedup is batch-amortized: a cold single claim pays the corpus/index/yaml
cost in full, so the honest number depends on N.

Run: uv run --no-sync python scripts/bench_grounding.py
"""

from __future__ import annotations

import statistics
import time

from groundrails import ground, settings

REPEATS = 5

_PARA = (
    "Grandparent involvement moderates adolescent adjustment after parental "
    "separation, with contact frequency declining as the child ages. The pooled "
    "estimate attributes significant variance to maternal gatekeeping and "
    "co-residence patterns across three-generation households. "
)
SOURCE = "\n\n".join(
    _PARA + f"Cohort {i} replicated the effect with n={i * 37 + 100} families." for i in range(60)
)
CLAIMS = [
    f"Cohort {i} replicated the moderation effect with {i * 37 + 100} families in the panel."
    for i in range(60)
]


def _clear_caches() -> None:
    import groundrails.config as config
    import groundrails.grounding as grounding
    import groundrails.lexical as lexical

    grounding._BM25_CACHE.clear()
    lexical._CHUNK_CACHE.clear()
    lexical._CORPUS_CACHE.clear()
    config._RAW_YAML_CACHE.clear()


def _run(claims: list[str], cold: bool) -> float:
    """Wall-clock seconds for grounding ``claims``; cold clears caches per claim."""
    if not cold:
        _clear_caches()  # warm run still starts cold - first claim builds the caches
    t = time.perf_counter()
    for c in claims:
        if cold:
            _clear_caches()
        ground(c, [SOURCE])
    return time.perf_counter() - t


def main() -> None:
    settings.mark_ready()
    ground(CLAIMS[0], [SOURCE])  # one-time lazy imports (models, wordfreq, lingua)

    print(f"source ~{len(SOURCE) // 1024}KB, repeats={REPEATS}\n")
    print(f"{'mode':<18}{'batch':>6}{'ms/claim mean':>15}{'ms/claim sd':>13}")
    for cold, label in ((True, "cold (per-claim)"), (False, "warm (batch)")):
        for n in (1, 10, 60):
            claims = CLAIMS[:n]
            per_claim = [(_run(claims, cold) / n) * 1000 for _ in range(REPEATS)]
            mean = statistics.mean(per_claim)
            sd = statistics.stdev(per_claim) if len(per_claim) > 1 else 0.0
            print(f"{label:<18}{n:>6}{mean:>15.2f}{sd:>13.2f}")


if __name__ == "__main__":
    main()
