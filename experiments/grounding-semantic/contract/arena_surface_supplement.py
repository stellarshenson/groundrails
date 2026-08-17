"""Supplement to `arena_surface_report.json` - characterises the collisions the
main pass counted, so a non-zero count can be read for what it is.

Two things the count alone does not say:

  1  the one arena DOCUMENT that matches a mix CLAIM in all eight string forms -
     what shape of unit it is;
  2  the RAGBench RESPONSES (across every split of all ten subsets) that match
     mix claims - how many, how long, and why the arena's own frozen filter
     leaves them out.

No evaluation-surface text is emitted: shapes, lengths, token counts and
blake2b-64 digests only.

Run AFTER the main pass has finished, or it will overwrite a partial report.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
          experiments/grounding-semantic/contract/arena_surface_supplement.py \
          2>&1 | tee -a logs/contract-arena_surface.log
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import hashlib
import importlib.util as _ilu
import io
import json
import zipfile
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
OUT = HERE / "arena_surface_report.json"
NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."
SUBSETS = ("covidqa", "delucionqa", "emanual", "expertqa", "finqa", "hagrid",
           "hotpotqa", "pubmedqa", "tatqa", "techqa")


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def shape(s):
    return {"chars": len(s), "tokens": len(s.split()),
            "blake2b_64": hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(),
            "is_punctuation_only": not any(c.isalnum() for c in s)}


def main():
    ARM = _mod("g1arm", EXP / "R16-H142_G1_arm.py")
    H174 = _mod("h174", EXP / "R20-H174_arm_run.py")

    with ARM.untruncated_evidence():
        claims, chunks, y, tags = ARM.H108.public_train()
    claims, tags = list(claims), list(tags)
    for fname, group, *_ in H174.LANES:
        d = pl.read_parquet(EXP / fname)
        claims += d["claim"].to_list()
        tags += [group] * len(d)
    claim_groups = collections.defaultdict(set)
    for c, t in zip(claims, tags, strict=True):
        claim_groups[c].add(t)
    print(f"mix claims indexed: {len(claim_groups)} distinct", flush=True)

    z = zipfile.ZipFile(DATA / "dataset-ragbench.zip")

    # ---- (1) the arena document that is also a mix claim -------------------- #
    doc_hits = []
    for sub in SUBSETS:
        df = pl.read_parquet(io.BytesIO(z.read(f"galileo-ai__ragbench__{sub}__test.parquet")))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0))
        df = df.sample(min(250, len(df)), seed=0)
        docs = [c for d in df["documents"].to_list() for c in d[:8]]
        for u in sorted(set(docs)):
            if u in claim_groups:
                doc_hits.append({"arena_subset": sub, **shape(u),
                                 "occurrences_in_subset_document_slots": docs.count(u),
                                 "mix_groups": sorted(claim_groups[u])})
    print(f"arena documents that are also mix claims: {len(doc_hits)}", flush=True)

    # ---- (2) RAGBench responses that are also mix claims, every split ------- #
    resp_hits = collections.defaultdict(list)
    for sub in SUBSETS:
        for split in ("train", "validation", "test"):
            nm = f"galileo-ai__ragbench__{sub}__{split}.parquet"
            if nm not in z.namelist():
                continue
            d = pl.read_parquet(io.BytesIO(z.read(nm)))
            for r in sorted({x for x in d["response"].to_list() if x}):
                if r in claim_groups:
                    resp_hits[f"{sub}/{split}"].append(
                        {**shape(r), "mix_groups": sorted(claim_groups[r])})
    flat = [h for v in resp_hits.values() for h in v]
    over20 = [h for h in flat if h["chars"] > 20]
    print(f"RAGBench responses that are also mix claims: {len(flat)} "
          f"over all splits, {len(over20)} longer than the arena's 20-char filter",
          flush=True)

    res = json.loads(OUT.read_text())
    res["collision_characterisation"] = {
        "arena_document_x_mix_claim": {
            "n": len(doc_hits),
            "units": doc_hits,
            "reading": ("the whole exact-match C2 signal on the arena is these "
                        "units; their shape is what says whether a count of 1 is "
                        "a leak or a degenerate string"),
        },
        "ragbench_response_x_mix_claim_all_splits": {
            "n_total": len(flat),
            "n_longer_than_the_arena_20_char_filter": len(over20),
            "per_subset_split": {k: len(v) for k, v in sorted(resp_hits.items())},
            "max_chars": max((h["chars"] for h in flat), default=0),
            "max_tokens": max((h["tokens"] for h in flat), default=0),
            "mix_groups_involved": sorted({g for h in flat for g in h["mix_groups"]}),
            "units": {k: v for k, v in sorted(resp_hits.items())},
            "reading": ("RAGBench short answers colliding with short training "
                        "claims. The arena's own frozen filter keeps only "
                        "responses longer than 20 characters, which is why the "
                        "arena's response channel reads 0 while the full corpus "
                        "does not"),
        },
        "note": NOTE,
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"supplement banked -> {OUT}", flush=True)
    print("=== ARENA SURFACE SUPPLEMENT COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
