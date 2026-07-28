"""Re-ground the labelled prose set on the current tree and compare to the recorded runs.

Baselines from the collected forensics (same 77 claims, same 25 sources):
    run 7  lexical only : 18 fuzzy,  7 bm25, 3 contra - 25/77 (32.5%)
    run 6  semantic     :  4 fuzzy,  5 bm25, 19 semantic, 3 contra - 28/77 (36.4%)

Reports the same counts now, the number of cascade escalations avoided by the
DEF-20 out-of-scope skip, and the scope-aware score (out-of-scope claims held
out of the denominator, which is what the class exists to make possible).
"""

from collections import Counter
import json
from pathlib import Path
import time

from groundrails import semantic_ov, settings
from groundrails.extract import out_of_scope
from groundrails.grounding import ground_batch

settings.mark_ready()

HERE = Path(__file__).parent / "private-prose-forensics"
claims = [c["claim"] for c in json.loads((HERE / "claims-current.json").read_text())]
# (path, text) TUPLES, not a dict: a dict iterates into its KEYS, so
# `_unpack_sources` would silently ground every claim against the FILENAMES and
# discard the evidence entirely (see DEF-21) - that is what produced a 0/77
# lexical run on the first pass of this script.
sources = [(p.name, p.read_text(encoding="utf-8")) for p in sorted((HERE / "sources").iterdir())]
print(f"claims {len(claims)}  sources {len(sources)}", flush=True)

# Count real cascade entries so the skip is measured, not assumed.
escalations = Counter()
_real_score = semantic_ov.SemanticCascade.score


def _counting_score(self, claim, chunks):
    escalations["n"] += 1
    return _real_score(self, claim, chunks)


semantic_ov.SemanticCascade.score = _counting_score


def summarise(label, matches, elapsed):
    kinds = Counter(m.match_type for m in matches)
    grounded = sum(m.grounded for m in matches)
    oos = [m for m in matches if m.out_of_scope_reason]
    in_scope = len(matches) - len(oos)
    oos_grounded = sum(m.grounded for m in oos)
    print(f"\n=== {label}  ({elapsed:.1f}s) ===", flush=True)
    print(
        f"  exact {kinds['exact']}  fuzzy {kinds['fuzzy']}  bm25 {kinds['bm25']}  "
        f"semantic {kinds['semantic']}  nli {kinds['nli']}  "
        f"contradicted {kinds['contradicted']}  none {kinds['none']}"
    )
    print(f"  score (as reported)  : {grounded}/{len(matches)} ({grounded / len(matches):.1%})")
    print(
        f"  score (scope-aware)  : {grounded - oos_grounded}/{in_scope} "
        f"({(grounded - oos_grounded) / in_scope:.1%})   "
        f"[{len(oos)} out-of-scope held out]"
    )
    print(f"  cascade escalations  : {escalations['n']}")
    return matches


escalations.clear()
t = time.time()
lex = summarise("LEXICAL ONLY (baseline run 7: 25/77, 32.5%)", ground_batch(claims, sources), time.time() - t)

escalations.clear()
t = time.time()
sem = summarise(
    "SEMANTIC (baseline run 6: 28/77, 36.4%)",
    ground_batch(claims, sources, semantic=True),
    time.time() - t,
)

# What the skip bought: escalations that would have run without DEF-20.
would_have = sum(
    1
    for c, m in zip(claims, sem, strict=True)
    if m.out_of_scope_reason and m.match_type not in ("exact", "contradicted")
)
print(f"\ncascade calls avoided by the out-of-scope skip: {would_have}")
print("out-of-scope by reason:", dict(Counter(r for r in (out_of_scope(c) for c in claims) if r)))

# Agreement with the human labels on the claims that carry one.
labels = [json.loads(x) for x in (HERE / "labels.jsonl").read_text().splitlines()]
by_text = {r["claim"]: r["human_label"] for r in labels if "claim" in r}
flagged = [(c, m) for c, m in zip(claims, sem, strict=True) if m.out_of_scope_reason]
checked = [(c, by_text[c]) for c, _ in flagged if c in by_text]
print(
    f"\nflagged claims carrying a human label: {len(checked)}  "
    f"agree={sum(1 for _, v in checked if v == 'not_groundable')}  "
    f"disagree={sum(1 for _, v in checked if v in ('supported', 'not_supported'))}"
)
