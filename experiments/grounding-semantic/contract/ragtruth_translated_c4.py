"""Contract C4 - contamination census for `ragtruth_translated`. CPU ONLY.

INSTRUMENT, reused not reinvented: `provenance_gate.py` in the R14-H136 ruling-2
form the banked lane censuses run - 8-gram, Jaccard >= 0.3, BIDIRECTIONAL, WARN
0.5%, KILL 2% of the candidate side, against ALL TEN walled arena corpora, with
per-subset attribution. Thresholds are imported from `R19_supply_gates.py`, never
restated.

UNITS: contamination is a document-overlap property, so each language is gated on
its deduplicated EVIDENCE prompts and, separately, on its deduplicated CLAIMS.

CONTROLS, both required by the clause:
  spike  10 arena units injected into the candidate side; all must be detected
         with 0 baseline hits.
  live   a genuinely near-duplicate-BY-CONSTRUCTION side: 300 of the language's
         own prompts with a contiguous 20% span deleted, gated against the
         unperturbed prompts. This fires only if the instrument can actually see
         a near-duplicate IN THIS CORPUS'S register and script - the open
         question for Chinese, where the gate's whitespace tokenizer yields
         clause tokens rather than word tokens.

Also measured: the split-axis near-duplicate read C3 needs - each language's
train prompts gated against its own TEST prompts. Exact-string identity is
useless there because machine translation is not row-deterministic.

Run:  uv run python experiments/grounding-semantic/contract/ragtruth_translated_c4.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import importlib.util as _ilu
import io
import json
import random
import time
import zipfile
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
SEM = HERE.parent
DATA = SEM.parent.parent / "data" / "external" / "datasets"
ARCHIVE = DATA / "dataset-ragtruth-translated.zip"
OUT = HERE / "ragtruth_translated_c4.json"

LANGS = ("de", "fr", "es", "it", "pl", "hu", "cn")
LIVE_N = 300
SEED = 0


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = _mod("provgate", SEM / "provenance_gate.py")
_src = (SEM / "R19_supply_gates.py").read_text()
GATE_N = int(_src.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_src.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_src.split("GATE_KILL = ")[1].split("\n")[0])


def coverage(texts):
    h = G._TokenHasher()
    sizes = [G.ngram_hashes(t, GATE_N, h).size for t in texts]
    return {
        "units": len(texts),
        "units_scorable_by_8gram": sum(1 for s in sizes if s),
        "units_too_short_for_8gram": sum(1 for s in sizes if not s),
        "mean_8grams_per_unit": round(sum(sizes) / max(len(sizes), 1), 1),
        "short_unit_cover": "units below the 8-gram floor carry no n-gram evidence "
        "and are covered by the exact-string test in the C2 artifact",
    }


def perturb(text, rng):
    """Delete a contiguous 20% span - a near-duplicate by construction, not a copy."""
    n = len(text)
    cut = max(1, n // 5)
    s = rng.randrange(0, max(1, n - cut))
    return text[:s] + text[s + cut :]


def main():
    z = zipfile.ZipFile(ARCHIVE)
    arena, _ = G.load_arena()
    print(
        f"arena: {sum(len(v) for v in arena.values())} units over {len(arena)} subsets",
        flush=True,
    )
    rng = random.Random(SEED)

    res = {
        "member": "ragtruth_translated",
        "instrument": f"provenance_gate.py (R14-H136 ruling 2: {GATE_N}-gram, "
        f"Jaccard >= {GATE_JACCARD}, bidirectional, KILL {GATE_KILL})",
        "arena_subsets": {k: len(v) for k, v in arena.items()},
        "per_language": {},
    }

    for lg in LANGS:
        t0 = time.time()
        tr = pl.read_parquet(
            io.BytesIO(
                z.read(
                    next(
                        x
                        for x in z.namelist()
                        if f"ragtruth-{lg}-" in x and x.endswith("__train.parquet")
                    )
                )
            )
        )
        te = pl.read_parquet(
            io.BytesIO(
                z.read(
                    next(
                        x
                        for x in z.namelist()
                        if f"ragtruth-{lg}-" in x and x.endswith("__test.parquet")
                    )
                )
            )
        )
        prompts = sorted({p for p in tr["prompt"].to_list() if p.strip()})
        claims = sorted({a for a in tr["answer"].to_list() if a.strip()})
        te_prompts = sorted({p for p in te["prompt"].to_list() if p.strip()})

        block = {
            "evidence_coverage": coverage(prompts),
            "claim_coverage": coverage(claims),
        }

        spike = G.spike_control(
            prompts[:2000], arena, n=GATE_N, jaccard=GATE_JACCARD, k=10,
            label=f"{lg}_spike",
        )
        block["spike_control"] = spike

        for unit, texts in (("evidence_prompts", prompts), ("claims", claims)):
            g = G.run_gate(
                texts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                label=f"ragtruth_{lg}_{unit}", arena_texts=arena,
            )
            block[unit] = {
                "verdict": g["verdict"],
                "max_fraction": g["max_fraction"],
                "candidate_vs_arena": g["candidate_vs_arena"],
                "arena_vs_candidate": {
                    k: v for k, v in g["arena_vs_candidate"].items() if k != "per_arena_subset"
                },
                "hit_examples": g["hit_examples"],
            }
            print(
                f"  {lg} {unit}: {g['verdict']} max_fraction {g['max_fraction']} "
                f"best-J {g['candidate_vs_arena'].get('best_jaccard', {}).get('max')}",
                flush=True,
            )

        # LIVE positive control - near-duplicate by construction, not identical
        sample = rng.sample(prompts, min(LIVE_N, len(prompts)))
        live_side = {"member_prompts": sample}
        live = G.run_gate(
            [perturb(p, rng) for p in sample],
            n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
            label=f"{lg}_live_control", arena_texts=live_side,
        )
        block["live_positive_control"] = {
            "construction": "300 of this language's own prompts with a contiguous "
            "20% character span deleted, gated against the unperturbed prompts",
            "detected_fraction": live["candidate_vs_arena"]["fraction"],
            "units_with_hit": live["candidate_vs_arena"]["units_with_hit"],
            "n_units": live["candidate_vs_arena"]["n_units"],
            "best_jaccard": live["candidate_vs_arena"].get("best_jaccard"),
            "fires": live["candidate_vs_arena"]["fraction"] >= 0.95,
        }
        print(
            f"  {lg} live control: {block['live_positive_control']['detected_fraction']} "
            f"detected, fires={block['live_positive_control']['fires']}",
            flush=True,
        )

        # C3 support - near-duplicate read of the corpus's own train/test cut
        cut = G.run_gate(
            te_prompts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
            label=f"{lg}_test_vs_train", arena_texts={"train_prompts": prompts},
        )
        block["split_axis_near_duplicate"] = {
            "note": "the language's TEST prompts gated against its own TRAIN "
            "prompts - the near-duplicate read exact-string matching cannot give "
            "on a machine-translated corpus",
            "test_units": cut["candidate_vs_arena"]["n_units"],
            "test_units_with_train_hit": cut["candidate_vs_arena"]["units_with_hit"],
            "fraction": cut["candidate_vs_arena"]["fraction"],
            "best_jaccard": cut["candidate_vs_arena"].get("best_jaccard"),
        }
        print(
            f"  {lg} split near-dup: {block['split_axis_near_duplicate']['fraction']} "
            f"of test prompts hit train (best-J "
            f"{block['split_axis_near_duplicate']['best_jaccard']})",
            flush=True,
        )

        block["seconds"] = round(time.time() - t0, 1)
        res["per_language"][lg] = block
        OUT.write_text(json.dumps(res, indent=2))  # checkpoint per language

    res["verdict"] = {
        "all_languages_pass": all(
            b["evidence_prompts"]["verdict"] != "KILL"
            and b["claims"]["verdict"] != "KILL"
            for b in res["per_language"].values()
        ),
        "spike_all_pass": all(
            b["spike_control"]["passes"] for b in res["per_language"].values()
        ),
        "live_control_all_fire": all(
            b["live_positive_control"]["fires"] for b in res["per_language"].values()
        ),
        "worst_max_fraction": max(
            max(b["evidence_prompts"]["max_fraction"], b["claims"]["max_fraction"])
            for b in res["per_language"].values()
        ),
        "kill_threshold": GATE_KILL,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["verdict"], indent=2), flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
