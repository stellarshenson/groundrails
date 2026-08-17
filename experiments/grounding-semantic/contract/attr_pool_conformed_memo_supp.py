"""attr_pool conformed - C6 supplement: what the mix actually associates with a
POOL DOCUMENT, as opposed to whether the document was merely seen.

The memo stage's executor-added document-keyed probe scored a pool by how many of
its passages appear in the mix "at label 1" and read within-pair 0.5991.  That
feature is only a memorisation channel if the mix attaches a LABEL SIGNAL to the
document.  The `frame_reject` lane - the only mix member built from MiniCheck
documents - puts BOTH its claims over the SAME chunk, so each of its documents
appears once at label 1 and once at label 0 and carries mean label 0.5, i.e. no
signal at all.  Measured here rather than assumed:

  * the distribution of the MIX MEAN LABEL over the lane's pool documents
  * the same probe rebuilt on mean label instead of "seen at label 1"
  * truth documents vs swap documents, to say whether the 0.5991 is a label
    association or a selection artifact of which documents can be a truth

CPU only.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import collections
import hashlib
import importlib.util as _ilu
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EXP = HERE.parent
SEP = "\n\n"


def _mod(name, fname, folder=EXP):
    spec = _ilu.spec_from_file_location(name, folder / fname)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("h174common", "R20-H174_lane_common.py")
B = _mod("apbuild", "attr_pool_conformed_build.py", folder=HERE)


def main():
    df = pl.read_parquet(EXP / "R20-H174_lane_L2_conformed.parquet")
    print("mix ...", flush=True)
    mclaims, mchunks, my, mtags = B.load_mix()

    lab_sum, lab_n = collections.Counter(), collections.Counter()
    tag_of = collections.defaultdict(set)
    for k, l, t in zip(mchunks, my.tolist(), mtags):
        h = hashlib.sha1(k.encode()).hexdigest()
        lab_sum[h] += float(l)
        lab_n[h] += 1
        tag_of[h].add(t)
        if t in ("quant_misbind", "quant_scale_unit", "frame_reject", "path_bind"):
            for p in k.split(SEP):
                if p != k:
                    hp = hashlib.sha1(p.encode()).hexdigest()
                    lab_sum[hp] += float(l)
                    lab_n[hp] += 1
                    tag_of[hp].add(t)
    del mclaims, mchunks

    def meanlab(text):
        h = hashlib.sha1(text.encode()).hexdigest()
        return (lab_sum[h] / lab_n[h], lab_n[h], sorted(tag_of[h])) if lab_n[h] else None

    # every distinct pool passage of the conformed lane
    seen, dist = {}, collections.Counter()
    for k in df["chunk"].to_list():
        for p in k.split(SEP):
            if p in seen:
                continue
            seen[p] = meanlab(p)
    inmix = {p: v for p, v in seen.items() if v}
    for v in inmix.values():
        dist[round(v[0], 3)] += 1

    # per-row features on the MEAN label the mix attaches to the pool documents
    smax, smean = np.zeros(df.height), np.zeros(df.height)
    for i, k in enumerate(df["chunk"].to_list()):
        vals = [seen[p][0] for p in k.split(SEP) if seen[p]]
        smax[i] = max(vals) if vals else 0.5
        smean[i] = float(np.mean(vals)) if vals else 0.5
    y = df["label"].to_numpy()
    probes = {}
    for name, s in (("max_mix_mean_label_over_pool", smax),
                    ("mean_mix_mean_label_over_pool", smean)):
        probes[name] = {
            "auroc_row_level": round(C.auroc(y, s), 4),
            "within_pair": C.within_pair_accuracy(df, s, by="neg_family"),
        }

    # truth vs swap documents: is being in the mix a property of the ROLE?
    posrows = df.filter(pl.col("label") == 1)
    negrows = df.filter(pl.col("label") == 0)
    truth_docs = set(posrows["doc_id"].to_list())
    swap_docs = set(x for x in negrows["swap_doc_id"].to_list() if x is not None)
    # map doc_id -> text via the pool: the truth text of a positive row is the
    # passage whose position matches doc_id in pool_doc_ids
    doc_text = {}
    for r in df.iter_rows(named=True):
        parts = r["chunk"].split(SEP)
        for d, t in zip(r["pool_doc_ids"], parts):
            doc_text.setdefault(d, t)
    def share_in_mix(ids):
        ids = [d for d in ids if d in doc_text]
        if not ids:
            return None
        return round(float(np.mean([1.0 if seen.get(doc_text[d]) else 0.0 for d in ids])), 4)

    out = {
        "C6_document_keyed_supplement": {
            "why": "a document-keyed feature is a MEMORISATION channel only if the "
            "mix attaches a label signal to the document. This measures the signal "
            "rather than the sighting",
            "distinct_pool_passages": len(seen),
            "pool_passages_found_in_the_mix": len(inmix),
            "share_found": round(len(inmix) / len(seen), 4),
            "mix_mean_label_distribution_over_those_passages": {
                str(k): v for k, v in sorted(dist.items())},
            "mix_members_they_come_from": sorted(
                {t for v in inmix.values() for t in v[2]}),
            "probes_on_the_mix_mean_label": probes,
            "documents_in_the_mix_by_role": {
                "truth_documents": len(truth_docs),
                "truth_documents_share_in_mix": share_in_mix(truth_docs),
                "swap_documents": len(swap_docs),
                "swap_documents_share_in_mix": share_in_mix(swap_docs),
            },
            "chance": 0.5,
        }
    }
    (HERE / "attr_pool_conformed_memo_supp.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
