"""R15-B3 killgate (amends R14-H134 / R14-A5) - does the BINARY ABSENT-NUMERAL
indicator survive into the deployed function in domain?

A5 decorrelates the task logit against claim DIGIT FRACTION. B3 substitutes the
binary absent-numeral indicator, the feature the P3/L2 census actually measures
(the two share mean r2 = 0.0973 across 24 (group, label) cells).

R14_H134_partialr.py re-run UNMODIFIED except for the feature, on the identical
600-row RAGTruth EN test sample and the identical read (max over
M59.top_chunks, pre-sigmoid task logit of the argmax chunk).

  KILL if |partial r| < 0.05 for the binary feature.

Both features are reported side by side so the substitution is auditable.

Frozen H105 draw 1, in-domain RAGTruth EN test, zero arena, zero gold.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python <this>
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
RESULT = HERE / "R15_gate_B3_result.json"

CKPT = "R9-H105-mmbert-dann-clean"
MAX_LEN = 512
BATCH = 64
BAR = 0.05


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def partial_r(z, x, lab):
    """Correlation of the residuals of both variables after regression on the label."""
    rz, rx = z.copy(), x.copy()
    for v in (0.0, 1.0):
        m = lab == v
        if m.sum() > 1:
            rz[m] -= rz[m].mean()
            rx[m] -= rx[m].mean()
    denom = float(np.sqrt((rz**2).sum() * (rx**2).sum()))
    return float((rz * rx).sum() / denom) if denom > 0 else float("nan")


def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
    M60 = _mod("m60", "R7-H60_multilingual_parallel.py")

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    claims, ctx, y = M60.load_english()
    chunk_lists = [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
    print(f"RAGTruth EN held-out: {len(claims)} rows, grounded rate {y.mean():.3f}", flush=True)

    tok, trunk, head = C.load_ckpt(CKPT)
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k[: M59.CFG.chunk_max_chars])
            owner.append(i)
    logits = np.zeros(len(flat_c), dtype=np.float32)
    with torch.inference_mode():
        for j in range(0, len(flat_c), BATCH):
            enc = tok(flat_c[j:j + BATCH], flat_k[j:j + BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            logits[j:j + BATCH] = head(cls).float().squeeze(-1).cpu().numpy()
    del trunk, head
    torch.cuda.empty_cache()

    owner = np.array(owner)
    arg = [int(np.flatnonzero(owner == i)[logits[owner == i].argmax()]) for i in range(len(claims))]
    z = logits[arg].astype(np.float64)
    lab = y.astype(np.float64)

    # the three evidence surfaces the absence flag can be defined against
    dec_chunk = [flat_k[a] for a in arg]
    all_chunks = [" ".join(k[: M59.CFG.chunk_max_chars] for k in ks) for ks in chunk_lists]
    feats = {
        "absent_binary_vs_deciding_chunk": np.array(
            [float(C.asserts_absent(c, e)) for c, e in zip(claims, dec_chunk)]),
        "absent_binary_vs_all_retrieved_chunks": np.array(
            [float(C.asserts_absent(c, e)) for c, e in zip(claims, all_chunks)]),
        "absent_binary_vs_full_context": np.array(
            [float(C.asserts_absent(c, e)) for c, e in zip(claims, ctx)]),
        "claim_digit_fraction_char": np.array(
            [sum(ch.isdigit() for ch in c) / max(len(c), 1) for c in claims]),
        "claim_digit_fraction_token": np.array(
            [sum(t.strip("#Ġ▁ ").isdigit() for t in tok.tokenize(c)) / max(len(tok.tokenize(c)), 1)
             for c in claims]),
    }

    per_feature = {}
    for name, x in feats.items():
        if x.std() < 1e-12:
            per_feature[name] = {"note": "degenerate feature - zero variance"}
            continue
        per_label = {}
        for v in (0, 1):
            m = lab == v
            per_label[f"label_{v}"] = {
                "n": int(m.sum()),
                "pearson_r": (round(float(np.corrcoef(z[m], x[m])[0, 1]), 5)
                              if x[m].std() > 1e-12 else None),
                "mean_feature": round(float(x[m].mean()), 5),
            }
        per_feature[name] = {
            "partial_r_controlling_for_label": round(partial_r(z, x, lab), 5),
            "raw_r": round(float(np.corrcoef(z, x)[0, 1]), 5),
            "prevalence_or_mean": round(float(x.mean()), 5),
            "per_label": per_label,
        }
        print(name, json.dumps(per_feature[name]["partial_r_controlling_for_label"]), flush=True)

    primary = "absent_binary_vs_deciding_chunk"
    pr = per_feature[primary]["partial_r_controlling_for_label"]
    verdict = "KILL" if abs(pr) < BAR else "PASS"
    auc = C.auroc(1 / (1 + np.exp(-z[lab == 1])), 1 / (1 + np.exp(-z[lab == 0])))

    res = {
        "gate": "R15-B3 killgate (amends R14-H134 / R14-A5) - binary absent-numeral partial "
                "correlation, substituted for claim digit fraction",
        "model": str(C.MODELS / CKPT),
        "data": "R7-H60_multilingual_parallel.load_english() - RAGTruth EN TEST split, seed-0 "
                "sample; the identical 600-row in-domain gate R14_H134_partialr.py used",
        "read": "max over M59.top_chunks(context, semantic_top_k); quantity = pre-sigmoid task "
                "logit of the argmax chunk - unmodified from R14_H134_partialr.py",
        "implementation_choices": [
            f"PRIMARY feature is '{primary}': the absence flag is computed against the DECIDING "
            "(argmax) chunk, because the correlated logit belongs to that (claim, chunk) pair and "
            "that pair is the same object the P3/L2 census flags. The all-retrieved-chunks and "
            "full-context variants are reported so the choice is auditable.",
            "Absence detector byte-identical to tmp/R15_L2_weights.py / P3 (canonical numeral "
            "forms of the claim minus those of the evidence).",
            "Token digit fraction (P2-E) is reported alongside the character form R14-A5 "
            "registered; neither is the gate's primary.",
        ],
        "n": int(len(claims)),
        "in_domain_auc_sanity": round(auc, 4),
        "banked_reference_digit_fraction_char": {
            "partial_r": 0.07307, "per_label_r": {"label_0": 0.27716, "label_1": -0.0364}},
        "primary_feature": primary,
        "features": per_feature,
        "bar": f"KILL if |partial r| < {BAR} for the binary absent indicator",
        "verdict": verdict,
        "gates_downstream": "the B3 substitution amendment to R14-H134 / A5 (which feature the "
                            "registered lambda_dec = 1.0 term regresses against)",
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 88)
    print(f"  binary absent (deciding chunk) partial r = {pr:+.5f}   bar |r| >= {BAR}")
    print(f"  digit fraction (char)          partial r = "
          f"{per_feature['claim_digit_fraction_char']['partial_r_controlling_for_label']:+.5f}")
    print(f"\n  VERDICT: {verdict}\n  -> {RESULT}", flush=True)


if __name__ == "__main__":
    main()
