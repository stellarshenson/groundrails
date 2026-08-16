"""R19-H168 GATE B - is the EuroBERT TRUNK usable, measured against mmBERT-base?

WHY THIS AND NOT THE CLOZE TEST
-------------------------------
Gate A's masked-token control came back 1/3 for EuroBERT against 3/3 for
mmBERT-base. That is uninterpretable on its own: it cannot separate "EuroBERT-210m
is a weaker cloze model" (plausible - 210M against 307M, 12 layers against 22,
and its paper optimises retrieval and classification rather than generative
cloze) from "the load shim left the model degraded". Worse, it measures the
wrong thing: fine-tuning DISCARDS the masked-LM head entirely and keeps only the
encoder, so cloze quality is not the quantity the arm depends on.

This gate measures the quantity that matters - how linearly separable grounding
already is in the FROZEN trunk - and it measures it COMPARATIVELY, mmBERT-base
against EuroBERT-210m under an identical protocol. Both are stock base
checkpoints; neither has seen this task. The absolute numbers will be modest.
The COMPARISON is the read.

DISCIPLINE
----------
Rows come from the public TRAINING mix. Not the arena. Not `gold_full` (the
serving-selection surface). Nothing here is tuned, no threshold is chosen, and
no arena statistic is consulted. This is an instrument check on a trunk.

VERDICT RULE, stated before the read
------------------------------------
  DEAD        EuroBERT probe AUROC < 0.55 while mmBERT clears 0.60 - the trunk
              carries no usable grounding signal and the shim cannot be trusted;
              the arm is killed before any GPU-hour of training
  DEGRADED    EuroBERT trails mmBERT by more than 0.10 AUROC - report, and the
              author decides whether to spend the 2 draws anyway
  PROCEED     otherwise - the trunk is functional and the arm trains

FREE RIDER - the tokenizer confound census
------------------------------------------
The two trunks do not share a tokenizer (mmBERT 256,000 Gemma-2; EuroBERT
128,256 Llama-3). The SAME 1,500-character window therefore yields DIFFERENT
token counts, so a fixed MAX_LEN of 512 truncates the two models' inputs
differently - most sharply on Chinese. That is a genuine confound in a
"same recipe" comparison and it must be measured, not assumed away. Recorded per
corpus group here so the arm's registration can state it.

Run: CUDA_VISIBLE_DEVICES=1 uv run python R19-H168_trunk_gate_b.py
"""

import importlib.util
import json
import os
import pathlib
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

HERE = pathlib.Path(__file__).parent

TRUNKS = [
    ("mmBERT-base", "jhu-clsp/mmBERT-base", False),
    ("EuroBERT-210m", "EuroBERT/EuroBERT-210m", True),
]
N_PER_CLASS = 3_000
MAX_LEN = 512          # the campaign's serving length, held identical
BATCH = 32
SEED = 1168
BARS = {"dead_euro_below": 0.55, "dead_mm_above": 0.60, "degraded_gap": 0.10}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


COMPAT = _mod("h168compat", "R19-H168_eurobert_compat.py")
ARM = _mod("g1arm", "R16-H142_G1_arm.py")


def sample_rows():
    """A class-balanced, group-stratified sample of the PUBLIC TRAINING mix."""
    with ARM.untruncated_evidence():
        claims, chunks, y, tags = ARM.H108.public_train()
    y = np.asarray(y)
    rng = np.random.default_rng(SEED)
    picked = []
    for label in (0.0, 1.0):
        idx = np.flatnonzero(y == label)
        # stratify by group so no single corpus dominates the probe
        by_g = {}
        for i in idx:
            by_g.setdefault(tags[i], []).append(i)
        per_g = max(1, N_PER_CLASS // max(len(by_g), 1))
        for g, ids in sorted(by_g.items()):
            ids = np.asarray(ids)
            take = min(per_g, ids.size)
            picked.extend(rng.choice(ids, size=take, replace=False).tolist())
    picked = sorted(set(picked))
    rng.shuffle(picked)
    # one window per row, the first - identical row set and text for both trunks
    rows = [(claims[i], ARM.windows(chunks[i])[0], float(y[i]), tags[i]) for i in picked]
    return rows


def token_census(tok, rows):
    """Tokens consumed by the identical (claim, window) text, per corpus group."""
    per_g = {}
    for c, w, _lab, g in rows:
        per_g.setdefault(g, []).append((c, w))
    out = {}
    for g, pairs in sorted(per_g.items()):
        cs = [p[0] for p in pairs][:400]
        ws = [p[1] for p in pairs][:400]
        enc = tok(cs, ws, truncation=False)["input_ids"]
        n = np.array([len(e) for e in enc])
        wc = np.array([len(w) for w in ws])
        out[g] = {"n_rows": int(n.size), "mean_tokens": round(float(n.mean()), 1),
                  "p95_tokens": round(float(np.percentile(n, 95)), 1),
                  "frac_over_maxlen": round(float((n > MAX_LEN).mean()), 4),
                  "chars_per_token": round(float(wc.mean() / max(n.mean(), 1e-9)), 3)}
    return out


def embed(repo, shim, rows):
    if shim:
        COMPAT.install()
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    if hasattr(model.config, "reference_compile"):
        model.config.reference_compile = False
    model = model.to("cuda").eval()

    census = token_census(tok, rows)
    vecs = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        enc = tok([r[0] for r in chunk], [r[1] for r in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_LEN).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state[:, 0]
        vecs.append(h.float().cpu().numpy())
        if i % (BATCH * 40) == 0:
            print(f"    {repo} {i}/{len(rows)}", flush=True)
    X = np.concatenate(vecs)
    del model
    torch.cuda.empty_cache()
    return X, census, round(time.time() - t0, 1)


def probe(X, y):
    """5-fold stratified CV logistic probe, scaler fitted INSIDE each fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    from sklearn.metrics import roc_auc_score
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=4000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], lr.predict_proba(sc.transform(X[te]))[:, 1]))
    return {"fold_auc": [round(a, 5) for a in aucs],
            "mean": round(float(np.mean(aucs)), 5),
            "sd": round(float(np.std(aucs)), 5)}


def main():
    out = HERE / "R19-H168_trunk_gate_b.json"
    print(f"=== R19-H168 GATE B  {time.strftime('%F %T')} ===", flush=True)
    print(f"  device {torch.cuda.get_device_name(0)}", flush=True)

    rows = sample_rows()
    y = np.array([r[2] for r in rows])
    print(f"  probe set: {len(rows)} rows, {y.mean():.3f} positive, "
          f"{len(set(r[3] for r in rows))} groups", flush=True)

    res = {"arm": "R19-H168 EuroBERT-210m trunk swap", "gate": "B - frozen-trunk probe",
           "surface": "PUBLIC TRAINING MIX - not the arena, not gold_full",
           "n_rows": len(rows), "positive_rate": round(float(y.mean()), 4),
           "max_len": MAX_LEN, "seed": SEED, "bars": BARS, "trunks": {}}

    for name, repo, shim in TRUNKS:
        print(f"\n  --- {name} ---", flush=True)
        X, census, secs = embed(repo, shim, rows)
        pr = probe(X, y)
        res["trunks"][name] = {"repo": repo, "shim": shim, "probe": pr,
                               "seconds": secs, "dim": int(X.shape[1]),
                               "token_census": census}
        print(f"  {name}: probe AUROC {pr['mean']:.5f} (sd {pr['sd']:.5f})", flush=True)

    mm = res["trunks"]["mmBERT-base"]["probe"]["mean"]
    eu = res["trunks"]["EuroBERT-210m"]["probe"]["mean"]
    gap = round(mm - eu, 5)
    if eu < BARS["dead_euro_below"] and mm > BARS["dead_mm_above"]:
        verdict = "DEAD"
    elif gap > BARS["degraded_gap"]:
        verdict = "DEGRADED"
    else:
        verdict = "PROCEED"
    res["verdict"] = {"verdict": verdict, "mmbert": mm, "eurobert": eu, "gap": gap,
                      "rule": "DEAD if euro<0.55 while mm>0.60; DEGRADED if gap>0.10; "
                              "else PROCEED"}
    out.write_text(json.dumps(res, indent=1))
    print(f"\n  mmBERT-base {mm:.5f} | EuroBERT-210m {eu:.5f} | gap {gap:+.5f}",
          flush=True)
    print(f"  -> {out.name}", flush=True)
    print(f"=== H168 GATE B VERDICT: {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
