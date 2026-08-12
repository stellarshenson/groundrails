"""R18-H151a VARIANCE ANATOMY - per-window score dumps for both twin checkpoints.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R18-H151 SEED-VARIANCE ATTACK - registered (2026-08-12)": dump per-window
scores for the twin pair (seed 1142 draw 1, seed 2142 draw 2) on the four
high-variance plus two stable arena subsets (tatqa, techqa, pubmedqa,
hotpotqa / covidqa, emanual) and on gold_full, so each subset's seed swing
can be decomposed into argmax-window flip rate vs score-level drift (H151a)
and so the H151b pooling variants can be selected gold-side and adjudicated
blind on exactly these dumps (H151c). Measurement only - no training, no
arena tuning.

Reader lineage is the banked windowed decomposed-min read
(R16-H142_G1_reads.py through R16-H142_G1_arm.load_run): evidence sliced into
1,500-char windows at stride 750, one score per (claim-sentence, window)
through the adapter-aware forward path. score_pairs below replicates
ARM.score_sets pair-for-pair - same flat pair order, batch 64, fp32, the same
200k-row logit chunks - but KEEPS the per-pair logits instead of
max-reducing per sentence. The sanity gate therefore applies: max over
windows per sentence then min over sentences per item must reproduce the
banked windowed AUROC of every arena subset within 1e-4, recomputed from the
dumped rows themselves; on any miss the run stops and no parquet is written.

Rows: subset, item_id, sentence_id, window_id, score, label,
n_windows_in_sentence, checkpoint_seed. One parquet per checkpoint,
R18-H151_scores_<seed>.parquet. Per-(seed, subset) shards plus marker
sidecars under R18-H151_shards/ make a kill resumable; a finished checkpoint
is skipped whole.

GPU0 only (GPU1 trains H150, GPU2 foreign). Run detached:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
    nohup setsid uv run python \
        experiments/grounding-semantic/R18-H151_score_dump.py \
        >> logs/R18-H151_score_dump.log 2>&1 &
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU0 only - hard rule for this stage
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl
import torch

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
SHARDS = HERE / "R18-H151_shards"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARM = _mod("g1arm", "R16-H142_G1_arm.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA
M59 = ARENA.M59
H108 = ARM.H108

SUBSETS = ["tatqa", "techqa", "pubmedqa", "hotpotqa", "covidqa", "emanual"]
CHECKPOINTS = {1142: "R16-H142-G1-twin", 2142: "R16-H142-T-draw2"}

# The banked windowed decomposed-min reads (R16-H142_G1_twin_windowed_result.json,
# R16-H142_T_draw2_windowed_result.json) - the sanity gate targets.
BANKED = {
    1142: {"covidqa": 0.7645, "emanual": 0.6683, "hotpotqa": 0.6728,
           "pubmedqa": 0.6725, "tatqa": 0.7948, "techqa": 0.7745},
    2142: {"covidqa": 0.7661, "emanual": 0.6949, "hotpotqa": 0.6377,
           "pubmedqa": 0.6273, "tatqa": 0.7188, "techqa": 0.7026},
}
GATE_TOL = 1e-4


@torch.inference_mode()
def score_pairs(model, tok, flat_s, flat_w, set_index, n_sets, batch=64, tag=""):
    """ARM.score_sets replicated pair-for-pair, but the per-pair logits are
    returned instead of max-reduced per set. Identical numerics: same batch
    size, same fp32 encode, same ctx pooling, same logit chunking."""
    n = len(flat_s)
    cls_all = torch.zeros(n, model.trunk.config.hidden_size, dtype=torch.float32)
    t0 = time.time()
    for i in range(0, n, batch):
        enc = tok(flat_s[i : i + batch], flat_w[i : i + batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=ARM.MAX_LEN)
        enc = {k: v.cuda() for k, v in enc.items()}
        cls_all[i : i + batch] = model.encode(enc).float().cpu()
        if (i // batch) % 400 == 0 and i:
            print(f"    {tag} {i}/{n} ({i / max(time.time() - t0, 1e-9):.0f} pairs/s)",
                  flush=True)
    si = torch.as_tensor(set_index, dtype=torch.long).cuda()
    ctx = model.pool_ctx(cls_all.cuda(), si, n_sets)
    out = np.empty(n, dtype=np.float32)
    step = 200_000
    for a in range(0, n, step):
        b = min(a + step, n)
        lg = model.pair_logits(cls_all[a:b].cuda(), ctx[si[a:b]])
        out[a:b] = lg.float().cpu().numpy()
    return out


def score_subset(model, tok, seed, sub, claims, chunks, y):
    """Score every (sentence, window) pair of one subset; write the shard and
    its marker sidecar. Skipped whole when both are already on disk."""
    shard = SHARDS / f"{seed}_{sub}.parquet"
    sidecar = SHARDS / f"{seed}_{sub}.json"
    if shard.exists() and sidecar.exists():
        print(f"  {sub:10s} seed {seed}: shard on disk, skipped", flush=True)
        return
    flat_s, flat_w, set_index = [], [], []
    item_id, sent_id, window_id, nwin = [], [], [], []
    n_sent = 0
    for i, (c, ks) in enumerate(zip(claims, chunks, strict=True)):
        wlist = [w for k in ks for w in ARM.windows(k)]
        for s_id, s in enumerate(H92.sentences(c)):
            sid = n_sent
            n_sent += 1
            for w_id, w in enumerate(wlist):
                flat_s.append(s)
                flat_w.append(w)
                set_index.append(sid)
                item_id.append(i)
                sent_id.append(s_id)
                window_id.append(w_id)
                nwin.append(len(wlist))
    scores = score_pairs(model, tok, flat_s, flat_w, set_index, n_sent,
                         tag=f"{seed}/{sub}")
    df = pl.DataFrame({
        "subset": [sub] * len(scores),
        "item_id": np.array(item_id, dtype=np.int32),
        "sentence_id": np.array(sent_id, dtype=np.int32),
        "window_id": np.array(window_id, dtype=np.int32),
        "score": scores,
        "label": np.array(y, dtype=np.int8)[np.array(item_id)],
        "n_windows_in_sentence": np.array(nwin, dtype=np.int32),
        "checkpoint_seed": np.full(len(scores), seed, dtype=np.int32),
    })
    df.write_parquet(shard)
    sidecar.write_text(json.dumps({
        "seed": seed, "subset": sub, "n": len(y), "n_sent": n_sent,
        "n_pairs": len(scores),
    }))
    print(f"  {sub:10s} seed {seed}: n={len(y)} sents={n_sent} "
          f"pairs={len(scores)} -> {shard.name}", flush=True)


def gate_auc(frame):
    """Item-level AUROC recomputed from the dumped rows alone: max over
    windows per (item, sentence), then min over sentences per item."""
    item = (frame.group_by(["item_id", "sentence_id"], maintain_order=True)
                 .agg(pl.col("score").max(), pl.col("label").first())
                 .group_by("item_id", maintain_order=True)
                 .agg(pl.col("score").min(), pl.col("label").first())
                 .sort("item_id"))
    auc, _, _ = M59.auc_and_f1(item["label"].to_numpy(), item["score"].to_numpy())
    return auc


def run_checkpoint(seed, subs, gold):
    out = HERE / f"R18-H151_scores_{seed}.parquet"
    if out.exists():
        print(f"seed {seed}: {out.name} on disk, checkpoint skipped", flush=True)
        return
    ckpt = ROOT / "models" / CHECKPOINTS[seed]
    print(f"\n=== seed {seed} - {CHECKPOINTS[seed]}  {time.strftime('%F %T')} ===",
          flush=True)
    model, tok = ARM.load_run(ckpt)
    for sub in SUBSETS:
        score_subset(model, tok, seed, sub, *subs[sub])
    score_subset(model, tok, seed, "gold_full", *gold)
    del model
    torch.cuda.empty_cache()

    print(f"\nseed {seed} sanity gate (from dumped rows, tol {GATE_TOL}):", flush=True)
    frames, bad = {}, []
    for sub in SUBSETS + ["gold_full"]:
        frames[sub] = pl.read_parquet(SHARDS / f"{seed}_{sub}.parquet")
    for sub in SUBSETS:
        got = gate_auc(frames[sub])
        want = BANKED[seed][sub]
        ok = abs(got - want) <= GATE_TOL
        print(f"  {sub:10s} dumped {got:.4f}  banked {want:.4f}  "
              f"|d| {abs(got - want):.2e}  {'ok' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            bad.append((sub, round(got, 6), want))
    gf_auc = gate_auc(frames["gold_full"])
    print(f"  gold_full  dumped {gf_auc:.4f} (windowed read - no banked target, "
          "recorded for H151b)", flush=True)
    if bad:
        raise SystemExit(
            f"SANITY GATE FAILED for seed {seed}: {bad} - the dump is wrong, "
            "no parquet written")
    df = pl.concat([frames[s] for s in SUBSETS + ["gold_full"]])
    df.write_parquet(out)
    print(f"seed {seed}: gate PASSED, {len(df)} rows -> {out}", flush=True)


def main():
    print(f"=== R18-H151a per-window score dump  {time.strftime('%F %T')} ===",
          flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})", flush=True)
    SHARDS.mkdir(exist_ok=True)
    subs = {s: v for s, v in ARENA.load_subsets().items() if s in SUBSETS}
    missing = [s for s in SUBSETS if s not in subs]
    if missing:
        raise SystemExit(f"arena subsets missing from load_subsets(): {missing}")
    gold = H108.gold_full()
    print(f"arena: {len(subs)} subsets; gold_full: {len(gold[2])} items", flush=True)
    for seed in (1142, 2142):
        run_checkpoint(seed, subs, gold)
    print(f"\n=== R18-H151a SCORE DUMP DONE  {time.strftime('%F %T')} ===", flush=True)


if __name__ == "__main__":
    main()
