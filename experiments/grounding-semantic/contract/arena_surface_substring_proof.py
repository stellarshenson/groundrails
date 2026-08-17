"""Third addendum to `arena_surface_report.json` - the VERBATIM SUBSTRING test.

The census reports overlap fractions; this reports the strongest possible form of
the same fact, which needs no threshold at all: is an arena document present, byte
for byte, INSIDE a training chunk?  Exact-equality matching cannot see this - a
document concatenated with a neighbouring paragraph is a different string - and it
is exactly the shape the containment reading found.

Candidates are the arena documents whose 8-gram containment against a single mix
chunk reaches 0.25; a verbatim substring cannot exist below containment 1.0, so
the candidate set is a strict superset of the possible hits.  Both string forms
are tested: raw, and whitespace-collapsed case-folded.

Two controls: a POSITIVE one (a real mix chunk's own interior slice, which must be
found) and a NEGATIVE one (the same slice with one character changed, which must
not be).

Reads the checkpointed census in `tmp/arena_surface/`.  Run AFTER the main pass.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
          experiments/grounding-semantic/contract/arena_surface_substring_proof.py \
          2>&1 | tee -a logs/contract-arena_surface.log
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import hashlib
import importlib.util as _ilu
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "arena_surface_report.json"
NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."
CANDIDATE_CONTAINMENT = 0.25


def main():
    spec = _ilu.spec_from_file_location("asv", HERE / "arena_surface_verify.py")
    M = _ilu.module_from_spec(spec)
    spec.loader.exec_module(M)
    norm = M.norm

    cen = M.load_census()
    if cen is None:
        raise SystemExit("ABORT: no checkpointed census in tmp/arena_surface")
    subsets = M.load_arena()
    arena_ch, _ = M.arena_units(subsets)
    claims, chunks, labels, tags, cut, groups = M.assemble()

    uniq = sorted(set(arena_ch["documents"]))
    best_c = cen["best_c"]["documents"]
    owner = cen["q_owner"]["documents"]
    cand = [i for i in range(len(uniq)) if best_c[i] >= CANDIDATE_CONTAINMENT]
    print(f"candidates at containment >= {CANDIDATE_CONTAINMENT}: {len(cand)}", flush=True)

    mix = list(dict.fromkeys(zip(chunks, tags)))
    mix_raw = [c for c, _t in mix]
    mix_tag = [t for _c, t in mix]
    mix_norm = [norm(c) for c in mix_raw]
    print(f"mix chunks indexed for substring search: {len(mix_raw)}", flush=True)

    t0 = time.time()
    hits = []
    for k, i in enumerate(cand):
        d = uniq[i]
        dn = norm(d)
        if len(d.strip()) < 40:
            continue
        raw_hit = norm_hit = None
        for j, c in enumerate(mix_raw):
            if d in c:
                raw_hit = j
                break
        for j, c in enumerate(mix_norm):
            if dn and dn in c:
                norm_hit = j
                break
        if raw_hit is not None or norm_hit is not None:
            j = raw_hit if raw_hit is not None else norm_hit
            hits.append({
                "arena_subset": owner[i],
                "doc_blake2b_64": hashlib.blake2b(d.encode("utf-8"),
                                                  digest_size=8).hexdigest(),
                "chars": len(d), "tokens": len(d.split()),
                "verbatim_substring_raw": raw_hit is not None,
                "substring_after_normalisation": norm_hit is not None,
                "mix_group": mix_tag[j],
                "mix_chunk_chars": len(mix_raw[j]),
                "arena_document_share_of_mix_chunk": round(len(d) / len(mix_raw[j]), 4),
                "ngram_containment": round(float(best_c[i]), 4),
            })
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(cand)} scanned, {len(hits)} substring hits "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"verbatim substring hits: {len(hits)} of {len(cand)} candidates "
          f"({time.time() - t0:.0f}s)", flush=True)

    hit_docs = set()
    for i in cand:
        d = uniq[i]
        h = hashlib.blake2b(d.encode("utf-8"), digest_size=8).hexdigest()
        if any(x["doc_blake2b_64"] == h for x in hits):
            hit_docs.add(d)
    per_sub = {}
    tot = 0
    for sub, v in subsets.items():
        n, labs = 0, []
        for docs, lab in zip(v["documents"], v["labels"], strict=True):
            if any(c in hit_docs for c in docs):
                n += 1
                labs.append(int(lab))
        tot += n
        per_sub[sub] = {
            "responses": len(v["responses"]),
            "responses_retrieving_a_verbatim_document": n,
            "fraction_of_subset_responses": round(n / len(v["responses"]), 6),
            "grounded_rate_all": round(float(np.mean(v["labels"])), 4),
            "grounded_rate_touched": round(float(np.mean(labs)), 4) if labs else None,
        }

    # controls
    probe = next(c for c in mix_raw if len(c) > 800)
    pos = probe[200:600]
    neg = pos[:200] + ("Z" if pos[200] != "Z" else "Q") + pos[201:]
    ctrl = {
        "positive": {"kind": "a real mix chunk's own interior 400-char slice",
                     "found": any(pos in c for c in mix_raw[:50000])},
        "negative": {"kind": "the same slice with one character changed",
                     "found": any(neg in c for c in mix_raw[:50000])},
    }
    print(f"controls: positive found={ctrl['positive']['found']} "
          f"negative found={ctrl['negative']['found']}", flush=True)

    res = json.loads(OUT.read_text())
    res["verbatim_substring_proof"] = {
        "test": ("is the arena document present byte for byte INSIDE a training "
                 "chunk - the form exact-equality matching structurally cannot see"),
        "candidate_rule": (f"arena documents at 8-gram containment >= "
                           f"{CANDIDATE_CONTAINMENT} against a single mix chunk; a "
                           f"verbatim substring implies containment 1.0, so this "
                           f"candidate set is a strict superset"),
        "candidates": len(cand),
        "arena_documents_verbatim_inside_a_training_chunk": len(hits),
        "of_distinct_arena_documents": len(uniq),
        "fraction_of_distinct_arena_documents": round(len(hits) / len(uniq), 6),
        "arena_responses_retrieving_one": tot,
        "fraction_of_arena_responses": round(tot / M.EXPECT_RESPONSES, 6),
        "per_mix_group": dict(collections.Counter(h["mix_group"] for h in hits)),
        "per_arena_subset": dict(collections.Counter(h["arena_subset"] for h in hits)),
        "per_subset_response_exposure": per_sub,
        "documents": hits,
        "controls": ctrl,
        "note": NOTE,
    }
    s = res["summary"]
    s["verbatim_arena_documents_inside_training_chunks"] = len(hits)
    s["verbatim_arena_responses_exposed"] = tot
    s["verbatim_per_arena_subset"] = res["verbatim_substring_proof"]["per_arena_subset"]
    s["verbatim_per_mix_group"] = res["verbatim_substring_proof"]["per_mix_group"]
    res["summary"] = s
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"substring proof banked -> {OUT}", flush=True)
    print("=== ARENA SURFACE SUBSTRING PROOF COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
