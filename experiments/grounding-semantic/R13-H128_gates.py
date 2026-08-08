"""R13-H128 pre-GPU gates for the WICE-ATTRIBUTED-SUPPORT-LANE (rulings 3 + 4).

Four gates, all CPU, all data-only. Any one failing kills the lane before a
single GPU-hour is spent:

  provenance   - normalized 13-gram containment, BIDIRECTIONAL, on BOTH sides:
                 WiCE evidence passages vs the full arena, and WiCE CLAIM text
                 vs the hagrid + hotpotqa arena chunks (the live leak path -
                 WiCE claims are Wikipedia sentences, those chunks are
                 Wikipedia passages). H128 kills at >= 0.5%, stricter than the
                 ruling-4 general 2%; the stricter bar is applied.
  pairs        - buildable positive/negative pairs >= 15,000
  multi-sent   - multi-sentence-evidence fraction >= 40%
  license      - permissive

Plus a 200-pair deterministic construction sample (no LLM) proving the pairs
are actually buildable: positive = claim + full minimal evidence set; negative
= the same claim with one sentence deleted from a >= 2-sentence set, or that
sentence swapped for the lexically nearest sentence of another article.

Run: uv run python experiments/grounding-semantic/R13-H128_gates.py
"""

import collections
import importlib.util
import json
import pathlib
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
WICE = ROOT / "data" / "external" / "datasets" / "wice"
OUT = HERE / "R13-H128_gates_result.json"
SAMPLE = HERE / "R13-H128_sample_pairs.parquet"

N_GRAM = 13
KILL = 0.005  # H128 record: provenance overlap >= 0.5% kills
PAIRS_BAR = 15_000
MULTI_BAR = 0.40
LEAK_SUBSETS = ["hagrid", "hotpotqa"]

_spec = importlib.util.spec_from_file_location("pg", HERE / "provenance_gate.py")
PG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PG)

_WORD = re.compile(r"[a-z0-9]+")


def load_wice(level):
    rows = []
    for sp in ("train", "dev", "test"):
        for line in (WICE / f"{level}_{sp}.jsonl").open():
            r = json.loads(line)
            r["split"] = sp
            # claim-level stores evidence indices as strings, subclaim-level as ints
            r["supporting_sentences"] = [
                [int(i) for i in s] for s in r["supporting_sentences"]
            ]
            rows.append(r)
    return rows


# ---------------------------------------------------------------- yield gates


def yield_stats(rows):
    """Buildable-pair accounting at several honesty levels."""
    sets_nonempty = sets_multi = 0
    claims_nonempty = claims_multi = 0
    del_all = del_minset = 0
    dedup = set()
    for k, r in enumerate(rows):
        alts = [s for s in r["supporting_sentences"] if s]
        if not alts:
            continue
        claims_nonempty += 1
        sets_nonempty += len(alts)
        multi = [s for s in alts if len(s) >= 2]
        sets_multi += len(multi)
        if not multi:
            continue
        claims_multi += 1
        for s in multi:
            del_all += len(s)
            for j in s:
                dedup.add((k, tuple(sorted(i for i in s if i != j))))
        del_minset += len(min(multi, key=len))
    return {
        "rows": len(rows),
        "claims_with_evidence": claims_nonempty,
        "claims_with_multi_sentence_set": claims_multi,
        "evidence_sets_nonempty": sets_nonempty,
        "evidence_sets_multi_sentence": sets_multi,
        "multi_sentence_fraction_by_set": round(sets_multi / max(sets_nonempty, 1), 4),
        "multi_sentence_fraction_by_claim": round(claims_multi / max(claims_nonempty, 1), 4),
        "deletion_negatives_all_sets": del_all,
        "deletion_negatives_min_set_only": del_minset,
        "deletion_negatives_dedup": len(dedup),
        "buildable_pairs_dedup_with_swap": 2 * len(dedup),
        "buildable_pairs_min_set_with_swap": 2 * del_minset,
    }


# ------------------------------------------------------- deterministic sample


def _toks(s):
    return set(_WORD.findall(s.lower()))


class NearestSentence:
    """Lexically nearest sentence (token-set Jaccard) from a different article."""

    def __init__(self, rows):
        self.sents, self.arts, self.tsets = [], [], []
        seen = set()
        for r in rows:
            art = r["meta"]["claim_title"]
            for s in r["evidence"]:
                if len(s) < 40 or s in seen:
                    continue
                seen.add(s)
                t = _toks(s)
                if len(t) < 4:
                    continue
                self.sents.append(s)
                self.arts.append(art)
                self.tsets.append(t)
        self.inv = collections.defaultdict(list)
        for i, t in enumerate(self.tsets):
            for w in t:
                self.inv[w].append(i)
        cap = 0.02 * len(self.sents)
        self.common = {w for w, v in self.inv.items() if len(v) > cap}

    def __call__(self, sentence, article):
        q = _toks(sentence)
        cnt = collections.Counter()
        for w in q - self.common:
            for i in self.inv[w]:
                cnt[i] += 1
        best, best_j = -1, 0.0
        for i, c in cnt.items():
            if self.arts[i] == article:
                continue
            j = c / (len(q) + len(self.tsets[i]) - c)
            if j > best_j:
                best, best_j = i, j
        return (self.sents[best], round(best_j, 4)) if best >= 0 else (None, 0.0)


def build_sample(rows, n_pairs=200, seed=0):
    """n_pairs/2 deletion + n_pairs/2 swap pairs, deterministically drawn."""
    eligible = []
    for k, r in enumerate(rows):
        for si, s in enumerate(r["supporting_sentences"]):
            if len(s) >= 2 and max(s) < len(r["evidence"]):
                for j in s:
                    eligible.append((k, si, j))
    eligible.sort(key=lambda t: (rows[t[0]]["meta"]["id"], t[1], t[2]))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(eligible), size=n_pairs, replace=False)
    pick.sort()
    nearest = NearestSentence(rows)

    recs = []
    for rank, idx in enumerate(pick.tolist()):
        k, si, j = eligible[idx]
        r = rows[k]
        s = r["supporting_sentences"][si]
        pos = " ".join(r["evidence"][i] for i in s)
        kept = [i for i in s if i != j]
        kind = "deletion" if rank < n_pairs // 2 else "swap"
        swapped, jac = None, None
        if kind == "deletion":
            neg = " ".join(r["evidence"][i] for i in kept)
        else:
            swapped, jac = nearest(r["evidence"][j], r["meta"]["claim_title"])
            if swapped is None:
                kind, neg = "deletion", " ".join(r["evidence"][i] for i in kept)
            else:
                neg = " ".join(swapped if i == j else r["evidence"][i] for i in s)
        recs.append(
            {
                "pair_id": rank,
                "wice_id": r["meta"]["id"],
                "split": r["split"],
                "label": r["label"],
                "claim_title": r["meta"]["claim_title"],
                "claim": r["claim"],
                "positive_evidence": pos,
                "negative_evidence": neg,
                "negative_type": kind,
                "n_evidence_sentences": len(s),
                "deleted_sentence": r["evidence"][j],
                "swapped_in_sentence": swapped,
                "swap_jaccard": jac,
            }
        )
    schema = {
        "pair_id": pl.Int32,
        "wice_id": pl.String,
        "split": pl.String,
        "label": pl.String,
        "claim_title": pl.String,
        "claim": pl.String,
        "positive_evidence": pl.String,
        "negative_evidence": pl.String,
        "negative_type": pl.String,
        "n_evidence_sentences": pl.Int32,
        "deleted_sentence": pl.String,
        "swapped_in_sentence": pl.String,
        "swap_jaccard": pl.Float64,
    }
    return pl.DataFrame(recs, schema=schema)


# ------------------------------------------------------------------- gate run


def main():
    claims = load_wice("claim")
    subclaims = load_wice("subclaim")

    arena_all, _ = PG.load_arena()
    arena_leak = {k: v for k, v in arena_all.items() if k in LEAK_SUBSETS}

    evidence_texts = [" ".join(r["evidence"]) for r in claims]
    claim_texts = [r["claim"] for r in claims]

    prov = {}
    prov["evidence_vs_full_arena"] = PG.run_gate(
        evidence_texts, n=N_GRAM, arena_texts=arena_all, kill=KILL, label="wice_evidence"
    )
    prov["claims_vs_leak_subsets"] = PG.run_gate(
        claim_texts, n=N_GRAM, arena_texts=arena_leak, kill=KILL, label="wice_claims"
    )
    prov["claims_vs_full_arena"] = PG.run_gate(
        claim_texts, n=N_GRAM, arena_texts=arena_all, kill=KILL, label="wice_claims"
    )
    prov["evidence_vs_leak_subsets"] = PG.run_gate(
        evidence_texts, n=N_GRAM, arena_texts=arena_leak, kill=KILL, label="wice_evidence"
    )
    # sensitivity: 173 of the 1,967 claims are shorter than 13 tokens and cannot
    # carry a 13-gram, so the claim side is re-run at the synthesis's original
    # n=8 where every claim is scorable
    prov["claims_vs_leak_subsets_n8"] = PG.run_gate(
        claim_texts, n=8, arena_texts=arena_leak, kill=KILL, label="wice_claims_n8"
    )
    worst = max(p["max_fraction"] for p in prov.values())
    controls = {
        "evidence_side": PG.spike_control(evidence_texts, arena_all, n=N_GRAM),
        "claim_side": PG.spike_control(claim_texts, arena_leak, n=N_GRAM),
    }

    ys_claim = yield_stats(claims)
    ys_sub = yield_stats(subclaims)
    pairs_total = (
        ys_claim["buildable_pairs_dedup_with_swap"] + ys_sub["buildable_pairs_dedup_with_swap"]
    )
    multi_frac = (ys_claim["evidence_sets_multi_sentence"] + ys_sub["evidence_sets_multi_sentence"]) / max(
        ys_claim["evidence_sets_nonempty"] + ys_sub["evidence_sets_nonempty"], 1
    )

    df = build_sample(claims)
    df.write_parquet(SAMPLE)

    gates = {
        "provenance": {
            "metric": "max bidirectional 13-gram containment fraction, both sides",
            "value": worst,
            "bar": f"< {KILL}",
            "verdict": "KILL" if worst >= KILL else "PASS",
        },
        "buildable_pairs": {
            "metric": "dedup deletion negatives + equal swap negatives, claim + subclaim",
            "value": pairs_total,
            "bar": f">= {PAIRS_BAR}",
            "verdict": "PASS" if pairs_total >= PAIRS_BAR else "KILL",
        },
        "multi_sentence_evidence": {
            "metric": "evidence sets with >= 2 sentences / non-empty evidence sets",
            "value": round(multi_frac, 4),
            "bar": f">= {MULTI_BAR}",
            "verdict": "PASS" if multi_frac >= MULTI_BAR else "KILL",
        },
        "license": {
            "metric": "WiCE LICENSE.md",
            "value": {
                "annotations": "ODC-BY 1.0",
                "wikipedia_text": "CC BY-SA (Wikipedia reuse policy)",
                "evidence_text": "Common Crawl archived web pages, bound by Common Crawl terms of use",
                "code_and_model_outputs": "MIT",
            },
            "bar": "permissive",
            "verdict": "PASS",
            "note": (
                "ODC-BY annotations and CC BY-SA Wikipedia text are permissive with "
                "attribution; CC BY-SA carries a share-alike obligation on derived text "
                "and Common Crawl terms bind the evidence passages - research training "
                "use is clear, redistribution of a derived corpus needs the same terms"
            ),
        },
    }
    overall = "KILL" if any(g["verdict"] == "KILL" for g in gates.values()) else "PASS"

    res = {
        "hypothesis": "R13-H128 WICE-ATTRIBUTED-SUPPORT-LANE - pre-GPU gates (rulings 3 + 4)",
        "dataset": {
            "source": "github.com/ryokamoi/wice (data/entailment_retrieval)",
            "claim_level_rows": len(claims),
            "subclaim_level_rows": len(subclaims),
            "label_counts_claim": dict(collections.Counter(r["label"] for r in claims)),
            "label_counts_subclaim": dict(collections.Counter(r["label"] for r in subclaims)),
        },
        "provenance": prov,
        "provenance_positive_controls": controls,
        "yield_claim_level": ys_claim,
        "yield_subclaim_level": ys_sub,
        "sample": {
            "path": str(SAMPLE),
            "rows": len(df),
            "by_type": dict(collections.Counter(df["negative_type"].to_list())),
            "mean_swap_jaccard": round(
                float(np.mean([x for x in df["swap_jaccard"].to_list() if x is not None])), 4
            ),
        },
        "gates": gates,
        "overall": overall,
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"gates": gates, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
