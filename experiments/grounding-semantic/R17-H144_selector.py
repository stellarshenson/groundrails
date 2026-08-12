"""R17-H144 amendment A2 - checkpoint selection on a held-out control family.

Cycle 2's two clauses split across epochs of one run: epoch 2 met the conjunction
on the banked bar set, the lowest-validation-loss epoch 3 did not. Picking epoch 2
for that reason would be selection on the test set. A2 fixes the selector BEFORE
any of its scores exist:

    checkpoint = argmax control-validation ACCURACY over ep1/ep2/ep3
    tie-break  = lower SFT validation loss
    no other signal may enter

The validation split (`R17-H144_ctrlval.parquet`) comes from the same generator as
the SFT corpus's own lookup family - the same code at a different seed - over
tables content-disjoint from BOTH the SFT corpus and the banked eval set.

Scoring is the eval harness verbatim, imported from `R17-H144_eval.py` so the
prompt, the closed think block, the forced answer position and the verdict parser
are identical objects, not copies. The selector reads the VERDICT (accuracy); the
margin is recorded but never consulted.

Run (detached):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
  nohup setsid python experiments/grounding-semantic/R17-H144_selector.py \\
      >> logs/R17-H144_selector.log 2>&1 &
"""

import importlib.util
import json
import os
import pathlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.pop("H144_CYCLE2", None)          # path tags are set explicitly below

import polars as pl  # noqa: E402

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
CKPTS = ROOT / "models" / "R17-H144-student-c2"
VALSET = HERE / "R17-H144_ctrlval.parquet"
SFT_STATS = HERE / "R17-H144_sft_c2_stats.json"
RESULT = HERE / "R17-H144_result.json"
OUT = HERE / "R17-H144_selector.json"
EPOCHS = (1, 2, 3)


def _eval_module():
    spec = importlib.util.spec_from_file_location("h144eval", HERE / "R17-H144_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EV = _eval_module()
SA = EV.SA
SA.SCORES = HERE / "R17-H144_selector_scores.parquet"
SA.GATE_LOG = HERE / "R17-H144_selector_gate.json"
SA.SAMPLES = HERE / "R17-H144_selector_samples.json"


def disjointness() -> dict:
    """0 shared documents and 0 shared chunks against the eval set and the corpus."""
    val = pl.read_parquet(VALSET)
    ev = pl.read_parquet(SA.EVALSET)
    out = dict(val_rows=val.height, val_documents=val["doc_id"].n_unique(),
               shared_chunks_with_evalset=len(
                   set(val["chunk"]) & set(ev["chunk"])))
    ids, chunks = set(), set()
    for src in ("R17-H144_pairs.parquet", "R17-H144_lookup.parquet"):
        d = pl.read_parquet(HERE / src, columns=["chunk", "doc_id"])
        ids |= set(d["doc_id"].to_list())
        chunks |= set(d["chunk"].to_list())
    out["shared_documents_with_sft_corpus"] = len(set(val["doc_id"]) & ids)
    out["shared_chunks_with_sft_corpus"] = len(set(val["chunk"]) & chunks)
    out["sft_corpus_documents"] = len(ids)
    return out


def score(epoch: int, rows: pl.DataFrame) -> dict:
    name = str(CKPTS / f"epoch{epoch}")
    key = pl.col("pair_id") * 2 + pl.col("label")
    done = SA.already(name)
    todo = rows.filter(~key.is_in(list(done))) if done else rows
    print(f"[selector] epoch{epoch}: {todo.height} of {rows.height} to score", flush=True)
    samples = SA.load_samples()
    for s in range(0, todo.height, 300):
        SA.checkpoint(SA.run_transformers(name, EV.CFG, todo.slice(s, 300), samples))
        SA.SAMPLES.write_text(json.dumps(samples, indent=2))
        print(f"[selector] epoch{epoch} {min(s + 300, todo.height)}/{todo.height}",
              flush=True)

    d = pl.read_parquet(SA.SCORES).filter(pl.col("model") == name)
    want = pl.when(pl.col("label") == 1).then(pl.lit("GROUNDED")).otherwise(pl.lit("UNGROUNDED"))
    d = d.with_columns((pl.col("verdict") == want).alias("correct"))
    per_fam = {r["neg_family"]: round(r["correct"], 4) for r in
               d.group_by("neg_family").agg(pl.col("correct").mean()).sort("neg_family").to_dicts()}
    return dict(epoch=epoch, n=d.height,
                control_val_accuracy=round(float(d["correct"].mean()), 6),
                per_family_accuracy=per_fam,
                positive_accuracy=round(float(d.filter(pl.col("label") == 1)["correct"].mean()), 4),
                negative_accuracy=round(float(d.filter(pl.col("label") == 0)["correct"].mean()), 4),
                parse_fail_rate=round(float(d["parse_fail"].mean()), 4))


def main() -> None:
    val = pl.read_parquet(VALSET).with_columns(
        pl.lit(False).alias("control"), pl.lit("lookup").alias("claim_form"))
    dj = disjointness()
    print(f"[selector] disjointness {dj}", flush=True)
    assert dj["shared_documents_with_sft_corpus"] == 0, "validation split touches SFT documents"
    assert dj["shared_chunks_with_sft_corpus"] == 0, "validation split touches SFT chunks"
    assert dj["shared_chunks_with_evalset"] == 0, "validation split touches eval chunks"

    hist = {h["epoch"]: h for h in json.loads(SFT_STATS.read_text())["history"]}
    reads = [score(e, val) for e in EPOCHS]
    for r in reads:
        r["sft_val_loss"] = hist[r["epoch"]]["val_loss"]

    # the registered rule, applied verbatim: argmax accuracy, tie-break lower loss
    best = max(reads, key=lambda r: (r["control_val_accuracy"], -r["sft_val_loss"]))
    top = max(r["control_val_accuracy"] for r in reads)
    tied = [r["epoch"] for r in reads if r["control_val_accuracy"] == top]

    res = json.loads(RESULT.read_text())
    banked = {c["epoch"]: c for c in res["cycle2"]["student_eval"]["per_checkpoint"]}
    sel = banked.get(best["epoch"])
    out = dict(
        amendment="A2 - held-out control-family checkpoint selection",
        val_split=dict(examples=int(val.height), documents=dj["val_documents"],
                       disjointness_verification=dj,
                       manifest=json.loads((HERE / "R17-H144_ctrlval_manifest.json").read_text())),
        selector_rule="argmax control-validation accuracy; tie-break lower SFT val loss",
        selector_scores={f"ep{r['epoch']}": r["control_val_accuracy"] for r in reads},
        per_checkpoint=reads, tied_epochs=tied, selected_epoch=best["epoch"],
        banked_eval_of_selected=(
            None if sel is None else
            dict(pooled_auroc=sel["pooled_auroc"], control_auroc=sel["control_auroc"],
                 pooled_auroc_literal05=sel["pooled_auroc_literal05"],
                 parse_fail_rate=sel["parse_fail_rate"], bars=sel["bars"])),
        banked_eval_available=sel is not None,
    )
    if sel is None:
        out["verdict"] = ("SELECTED CHECKPOINT NOT YET SCORED on the banked eval - "
                          f"epoch {best['epoch']} was never read on the bar set")
    else:
        b = sel["bars"]
        out["verdict"] = ("VIABLE" if b["viable_070"] else
                          "KILL" if b["kill_060"] else "NOT-VIABLE-AT-BAR")
        out["verdict_basis"] = dict(
            pooled_ge_070=b["pooled_ge_070"], control_ge_090=b["control_ge_090"],
            kill_060=b["kill_060"])
    OUT.write_text(json.dumps(out, indent=2))
    res["cycle2"]["selector_A2"] = out
    RESULT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("selector_scores", "selected_epoch", "banked_eval_of_selected",
                       "verdict")}, indent=2), flush=True)
    print("=== R17-H144 SELECTOR DONE ===", flush=True)


if __name__ == "__main__":
    main()
