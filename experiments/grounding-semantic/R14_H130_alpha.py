"""R14-H130 (register block R14-A1) - the legal-data alpha fit.

Executes section 2 of `R14_H130_frozen_form.md`, which was written and closed
BEFORE this script was run (binding amendment i). Nothing here may deviate from
that document.

Fit: within-item OLS slope of `max over K windows of score(claim, window)` on
`ln K`, with K varied by stride at FIXED window width and FIXED document
content, on a fixed-seed stratified sample of the clean public mix. Frozen H105
draw-1 checkpoint. No RAGBench, no gold, no arena quantity of any kind.

KILL: slope <= 0 or slope > 0.15.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R14_H130_alpha.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
RESULT = HERE / "R14_gate_H130_alpha.json"

MODEL = str(ROOT / "models" / "R9-H105-mmbert-dann-clean")
WIN = 1500
STRIDES = (1500, 1000, 750, 500, 375, 250)
SEED = 20260809
N_ITEMS = 1000
BATCH = 64
MAX_LEN = 512  # the shipped read's tokenizer length; the fit is of the shipped read


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def starts(n, stride):
    """Shipped window geometry at an arbitrary stride: 1,500-char windows,
    final window flush to the end. Returns start offsets."""
    if n <= WIN:
        return [0]
    s = list(range(0, n - WIN + 1, stride))
    if s[-1] + WIN < n:
        s.append(n - WIN)
    return s


def main():
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    H105 = _mod("h105", "R9-H105_clean_mix.py")
    # Amendment A1-a (frozen_form.md section 2): the mix truncates every chunk to
    # 1,500 chars, so K == 1 everywhere and the order statistic does not exist.
    # Same rows, same filters, same tags - full-length evidence text only.
    H105.M59.CFG.chunk_max_chars = 10**9
    claims, chunks, y, tags = H105.public_train()
    print(f"public mix: {len(claims)} rows", flush=True)

    eligible = [i for i in range(len(claims)) if len(chunks[i]) > WIN]
    print(f"eligible (chunk > {WIN} chars): {len(eligible)}", flush=True)

    # stratified over DANN group tags, fixed seed, round-robin allocation
    rng = np.random.default_rng(SEED)
    by_tag = {}
    for i in eligible:
        by_tag.setdefault(tags[i], []).append(i)
    for t in by_tag:
        by_tag[t] = list(rng.permutation(by_tag[t]))
    order = sorted(by_tag)
    picked, cursor = [], {t: 0 for t in order}
    while len(picked) < N_ITEMS and any(cursor[t] < len(by_tag[t]) for t in order):
        for t in order:
            if len(picked) >= N_ITEMS:
                break
            if cursor[t] < len(by_tag[t]):
                picked.append(int(by_tag[t][cursor[t]]))
                cursor[t] += 1
    tag_counts = {t: sum(1 for i in picked if tags[i] == t) for t in order}
    print(f"sample: {len(picked)} items  {tag_counts}", flush=True)

    # per item: the union of window start offsets over the whole stride grid
    items = []
    flat_c, flat_w = [], []
    for i in picked:
        n = len(chunks[i])
        per_stride = {t: starts(n, t) for t in STRIDES}
        uniq = sorted({o for v in per_stride.values() for o in v})
        base = len(flat_c)
        pos = {o: base + j for j, o in enumerate(uniq)}
        for o in uniq:
            flat_c.append(claims[i])
            flat_w.append(chunks[i][o : o + WIN])
        items.append({"row": i, "tag": tags[i], "doc_len": n,
                      "per_stride": {t: [pos[o] for o in per_stride[t]] for t in STRIDES}})
    print(f"scoring {len(flat_c)} unique (claim, window) pairs", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    state = torch.load(
        pathlib.Path(MODEL) / "dann_student.pt", map_location="cpu", weights_only=False
    )
    trunk = AutoModel.from_pretrained(str(pathlib.Path(MODEL) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    head = nn.Linear(trunk.config.hidden_size, 1)
    head.load_state_dict(state["task_head"])
    head = head.cuda().eval()

    s = np.zeros(len(flat_c), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for j in range(0, len(flat_c), BATCH):
            enc = tok(flat_c[j : j + BATCH], flat_w[j : j + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            s[j : j + BATCH] = torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
            if j % (BATCH * 50) == 0:
                print(f"  {j}/{len(flat_c)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"scored in {time.time() - t0:.0f}s", flush=True)

    # within-item OLS of M on ln K
    num = den = 0.0
    per_item_resid, per_item_id = [], []
    kept, dropped = 0, 0
    grid_rows = []
    for it in items:
        L, M = [], []
        for t in STRIDES:
            idx = it["per_stride"][t]
            L.append(np.log(len(idx)))
            M.append(float(s[idx].max()))
        L, M = np.array(L), np.array(M)
        if L.std() == 0:
            dropped += 1
            continue
        kept += 1
        dL, dM = L - L.mean(), M - M.mean()
        num += float((dL * dM).sum())
        den += float((dL * dL).sum())
        grid_rows.append({"tag": it["tag"], "doc_len": it["doc_len"],
                          "K": [len(it["per_stride"][t]) for t in STRIDES],
                          "M": [round(v, 5) for v in M.tolist()]})
        per_item_id.append(it["tag"])
        per_item_resid.append((dL, dM))

    alpha_hat = num / den
    # clustered-by-item standard error
    meat = 0.0
    for dL, dM in per_item_resid:
        r = dM - alpha_hat * dL
        meat += float((dL * r).sum()) ** 2
    se = float(np.sqrt(meat) / den) if den > 0 else float("nan")

    alpha = round(float(alpha_hat), 3)
    kill = (alpha_hat <= 0.0) or (alpha_hat > 0.15)
    verdict = "KILL (slope out of the pre-registered band)" if kill else "LICENSED"

    # descriptive: mean M by K bucket, pooled
    kmap = {}
    for it in items:
        for t in STRIDES:
            k = len(it["per_stride"][t])
            kmap.setdefault(k, []).append(float(s[it["per_stride"][t]].max()))
    by_k = {int(k): {"n": len(v), "mean_max": round(float(np.mean(v)), 5)}
            for k, v in sorted(kmap.items()) if len(v) >= 20}

    res = {
        "gate": "R14-H130 (R14-A1) legal-data alpha fit",
        "frozen_form_doc": "R14_H130_frozen_form.md",
        "model": MODEL,
        "data": "R9-H105_clean_mix.public_train(); RAGBench and gold never touched",
        "window": WIN, "strides": list(STRIDES), "seed": SEED,
        "n_items_requested": N_ITEMS, "n_items_sampled": len(items),
        "n_items_in_fit": kept, "n_items_dropped_constant_K": dropped,
        "n_unique_pairs_scored": len(flat_c),
        "tag_counts": tag_counts,
        "alpha_hat": round(float(alpha_hat), 6),
        "alpha_clustered_se": round(se, 6),
        "alpha_frozen": alpha,
        "mean_max_by_K": by_k,
        "kill_clause": "KILL if alpha_hat <= 0 or alpha_hat > 0.15",
        "verdict": verdict,
    }
    RESULT.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 92)
    print("R14-H130 ALPHA FIT")
    print("=" * 92)
    print(f"  items in fit {kept} (dropped {dropped}), unique pairs {len(flat_c)}")
    print(f"  alpha_hat = {alpha_hat:.6f}  (clustered se {se:.6f})   frozen alpha = {alpha}")
    for k, v in by_k.items():
        print(f"    K={k:>3}  n={v['n']:>5}  mean max {v['mean_max']:.5f}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {RESULT}")


if __name__ == "__main__":
    main()
