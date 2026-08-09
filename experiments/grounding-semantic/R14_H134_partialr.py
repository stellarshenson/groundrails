"""R14-H134 (R14-A5 kill-gate, clause 2) - partial correlation between the H105
draw-1 task logit and claim digit fraction, controlling for the label, in
domain.

  KILL if |partial r| < 0.05 - the claim-side numeric prior did not survive into
  the deployed function in-domain and a training-time decorrelation term is
  unlicensed.

Instrument: the campaign's in-domain RAGTruth English gate,
`R7-H60_multilingual_parallel.load_english()` - the RAGTruth **test** split
(seed-0 sample), which no draw has trained on. The repo's RAGTruth archive ships
train and test only; there is no separate dev split, and this is the held-out
in-domain set every lineage gate in the campaign uses. Recorded explicitly.

Read: the shipped in-domain read - claim against `M59.top_chunks(context, k)`,
score = max over chunks. The quantity correlated is the PRE-SIGMOID task logit
of the deciding (argmax) chunk, which is monotone-equivalent to the read's score.

Claim digit fraction = digit characters / total characters of the claim (E1's
definition: "finqa claims are ~10% digits by character").

Partial correlation controlling for the binary label = the correlation of the
residuals of both variables after regression on the label, i.e. the
label-pooled within-group correlation.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R14_H134_partialr.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
RESULT = HERE / "R14_gate_H134_partialr.json"

MODEL = str(ROOT / "models" / "R9-H105-mmbert-dann-clean")
MAX_LEN = 512
BATCH = 64


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
M60 = _mod("m60", "R7-H60_multilingual_parallel.py")


def digit_fraction(s):
    return sum(c.isdigit() for c in s) / max(len(s), 1)


def main():
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    claims, ctx, y = M60.load_english()
    chunk_lists = [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
    print(f"RAGTruth EN held-out: {len(claims)} rows, grounded rate {y.mean():.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    state = torch.load(
        pathlib.Path(MODEL) / "dann_student.pt", map_location="cpu", weights_only=False
    )
    trunk = AutoModel.from_pretrained(str(pathlib.Path(MODEL) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    head = nn.Linear(trunk.config.hidden_size, 1)
    head.load_state_dict(state["task_head"])
    head = head.cuda().eval()

    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k[: M59.CFG.chunk_max_chars])
            owner.append(i)
    logits = np.zeros(len(flat_c), dtype=np.float32)
    with torch.inference_mode():
        for j in range(0, len(flat_c), BATCH):
            enc = tok(flat_c[j : j + BATCH], flat_k[j : j + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            logits[j : j + BATCH] = head(cls).float().squeeze(-1).cpu().numpy()
    owner = np.array(owner)
    z = np.array([logits[owner == i].max() for i in range(len(claims))], dtype=np.float64)
    del trunk, head
    torch.cuda.empty_cache()

    d = np.array([digit_fraction(c) for c in claims], dtype=np.float64)
    lab = y.astype(np.float64)

    # residualize both on the binary label (equivalently: pool within-label deviations)
    rz = z.copy()
    rd = d.copy()
    for v in (0.0, 1.0):
        m = lab == v
        if m.sum() > 1:
            rz[m] -= rz[m].mean()
            rd[m] -= rd[m].mean()
    denom = float(np.sqrt((rz**2).sum() * (rd**2).sum()))
    pr = float((rz * rd).sum() / denom) if denom > 0 else float("nan")
    raw = float(np.corrcoef(z, d)[0, 1])

    per_label = {}
    for v in (0, 1):
        m = lab == v
        per_label[f"label_{v}"] = {
            "n": int(m.sum()),
            "pearson_r": round(float(np.corrcoef(z[m], d[m])[0, 1]), 5),
            "mean_digit_fraction": round(float(d[m].mean()), 5),
        }

    auc, _, _ = M59.auc_and_f1(y, np.array([1 / (1 + np.exp(-v)) for v in z]))

    kill = abs(pr) < 0.05
    verdict = "KILL (|partial r| < 0.05)" if kill else "PASS (|partial r| >= 0.05)"

    res = {
        "gate": "R14-H134 (R14-A5 kill-gate clause 2) label-conditional partial correlation",
        "model": MODEL,
        "data": "R7-H60_multilingual_parallel.load_english() - RAGTruth EN TEST split, seed-0 "
                "sample; the campaign's in-domain held-out gate. The repo archive ships train "
                "and test only - there is no separate RAGTruth dev split.",
        "n": int(len(claims)),
        "read": "max over M59.top_chunks(context, semantic_top_k); quantity = pre-sigmoid task "
                "logit of the argmax chunk",
        "feature": "claim digit fraction = digit chars / total chars",
        "partial_r_controlling_for_label": round(pr, 5),
        "raw_r": round(raw, 5),
        "per_label": per_label,
        "in_domain_auc_sanity": round(float(auc), 4),
        "kill_clause": "KILL if |partial r| < 0.05",
        "verdict": verdict,
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 92)
    print("R14-H134 PARTIAL CORRELATION")
    print("=" * 92)
    print(f"  n={len(claims)}  partial r = {pr:+.5f}  (raw r {raw:+.5f})  bar |r| >= 0.05")
    print(f"  per label: {per_label}")
    print(f"\n  VERDICT: {verdict}\n  -> {RESULT}")


if __name__ == "__main__":
    main()
