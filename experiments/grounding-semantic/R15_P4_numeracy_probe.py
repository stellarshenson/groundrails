"""R15-P4 numeracy dossier probe - frozen weights, held-out TabFact, zero arena, zero gold.

Two instruments, both on the shipped R9-H105 clean checkpoint:

  (A) DERIVATION-FAMILY EXTENSION of R14-H133. H133 measured AUROC(correct vs
      wrong) = 0.4924 on ONE family (two-operand sum). This extends the same
      construction to five families - verbatim, comparison, sum, difference,
      ratio - so the dossier can say which derivation types the deployed
      function already discriminates and which it does not.

  (B) REPRESENTATION PROBE (Wallace et al. 2019 protocol, adapted). Linear
      probe on the frozen trunk's [CLS] for (b1) log-magnitude regression and
      (b2) pairwise comparison, with an interpolation split and an
      extrapolation split, to separate "the representation does not carry it"
      from "the task head does not use it".

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python <this>
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import io, json, pathlib, re, zipfile
import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
MODEL = str(ROOT / "models" / "R9-H105-mmbert-dann-clean")
RESULT = HERE / "R15_gate_P4_numeracy.json"
CHUNK_MAX, MAX_LEN, BATCH, SEED = 1500, 512, 64, 20260810
N_PER_FAMILY = 1200

NUM = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")


def parse(tbl):
    rows = [r.split("#") for r in tbl.replace("\r\n", "\n").strip().split("\n") if r.strip()]
    rows = [[c.strip() for c in r] for r in rows]
    if len(rows) < 4:
        return None, None
    w = len(rows[0])
    return rows[0], [r for r in rows[1:] if len(r) == w]


def as_num(s):
    s = s.strip()
    if not NUM.match(s):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def fmt(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


def held_tables(rng):
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    train_ids = set(pl.read_parquet(io.BytesIO(z.read(
        next(x for x in z.namelist() if x.endswith("__train.parquet")))))["table_id"].to_list())
    held = pl.concat([pl.read_parquet(io.BytesIO(z.read(n))) for n in z.namelist()
                      if n.endswith("__test.parquet") or n.endswith("__validation.parquet")]
                     ).unique(subset=["table_id"], keep="first")
    held = held.filter(~pl.col("table_id").is_in(list(train_ids)))
    return held, len(train_ids)


def build(rng):
    held, n_train = held_tables(rng)
    order = rng.permutation(len(held))
    caps, tbls, tids = (held["table_caption"].to_list(), held["table_text"].to_list(),
                        held["table_id"].to_list())
    fams = {k: [] for k in ("verbatim", "comparison", "sum", "difference", "ratio")}
    per_table = {}
    for oi in [int(o) for _ in range(6) for o in order]:
        if all(len(v) >= N_PER_FAMILY for v in fams.values()):
            break
        if per_table.get(oi, 0) >= 4:
            continue
        hdr, body = parse(tbls[oi])
        if hdr is None:
            continue
        ev = f"{caps[oi]}\n{tbls[oi]}".replace("\r\n", "\n").replace("#", " | ")[:CHUNK_MAX]
        cand = []
        for ci in range(1, len(hdr)):
            vals = [(ri, as_num(r[ci])) for ri, r in enumerate(body)]
            vals = [(ri, v) for ri, v in vals if v is not None]
            if len(vals) >= 4 and len({v for _, v in vals}) >= 4:
                cand.append((ci, vals))
        if not cand:
            continue
        ci, vals = cand[int(rng.integers(len(cand)))]
        col = hdr[ci] or f"column {ci}"
        if any((not body[ri][0]) or as_num(body[ri][0]) is not None for ri, _ in vals):
            continue
        pick = [int(p) for p in rng.permutation(len(vals))[:4]]
        (i, j, k, l) = pick
        (ri_i, vi), (ri_j, vj), (_, vk), (_, vl) = vals[i], vals[j], vals[k], vals[l]
        ka, kb = body[ri_i][0].strip(), body[ri_j][0].strip()
        base = {"table_id": tids[oi], "evidence": ev}
        used = False

        # verbatim - control. correct = the cell value; wrong = another row's value
        if vi != vj and len(fams["verbatim"]) < N_PER_FAMILY:
            fams["verbatim"].append({**base, "family": "verbatim",
                "claim_ok": f"The {col} of {ka} is {body[ri_i][ci].strip()}.",
                "claim_bad": f"The {col} of {ka} is {body[ri_j][ci].strip()}.",
                "v_ok": body[ri_i][ci].strip(), "v_bad": body[ri_j][ci].strip()}); used = True

        # comparison - NO new value asserted; both operands verbatim in the table
        if vi != vj and len(fams["comparison"]) < N_PER_FAMILY:
            hi, lo = (ka, kb) if vi > vj else (kb, ka)
            fams["comparison"].append({**base, "family": "comparison",
                "claim_ok": f"The {col} of {hi} is greater than the {col} of {lo}.",
                "claim_bad": f"The {col} of {lo} is greater than the {col} of {hi}.",
                "v_ok": "", "v_bad": ""}); used = True

        # sum - H133's family, replicated here under the same seed family
        s_ij, s_kl = vi + vj, vk + vl
        if abs(s_ij - s_kl) > 1e-9 and len(fams["sum"]) < N_PER_FAMILY:
            b, c = fmt(s_ij), fmt(s_kl)
            if b not in ev and c not in ev:
                fams["sum"].append({**base, "family": "sum",
                    "claim_ok": f"The combined {col} of {ka} and {kb} is {b}.",
                    "claim_bad": f"The combined {col} of {ka} and {kb} is {c}.",
                    "v_ok": b, "v_bad": c}); used = True

        # difference
        d_ij, d_kl = abs(vi - vj), abs(vk - vl)
        if abs(d_ij - d_kl) > 1e-9 and d_ij > 0 and len(fams["difference"]) < N_PER_FAMILY:
            b, c = fmt(d_ij), fmt(d_kl)
            if b not in ev and c not in ev:
                fams["difference"].append({**base, "family": "difference",
                    "claim_ok": f"The {col} of {ka} exceeds the {col} of {kb} by {b}.",
                    "claim_bad": f"The {col} of {ka} exceeds the {col} of {kb} by {c}.",
                    "v_ok": b, "v_bad": c}); used = True

        # ratio - percentage of one cell relative to another
        if vj not in (0,) and vl not in (0,) and len(fams["ratio"]) < N_PER_FAMILY:
            r_ij, r_kl = 100.0 * vi / vj, 100.0 * vk / vl
            if abs(r_ij - r_kl) > 0.5 and 0 < r_ij < 100000:
                b, c = f"{r_ij:.1f}", f"{r_kl:.1f}"
                if b not in ev and c not in ev:
                    fams["ratio"].append({**base, "family": "ratio",
                        "claim_ok": f"The {col} of {ka} is {b} percent of the {col} of {kb}.",
                        "claim_bad": f"The {col} of {ka} is {c} percent of the {col} of {kb}.",
                        "v_ok": b, "v_bad": c}); used = True
        if used:
            per_table[oi] = per_table.get(oi, 0) + 1
    return fams, n_train


def load_model():
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    state = torch.load(pathlib.Path(MODEL) / "dann_student.pt", map_location="cpu",
                       weights_only=False)
    trunk = AutoModel.from_pretrained(str(pathlib.Path(MODEL) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    head = nn.Linear(trunk.config.hidden_size, 1)
    head.load_state_dict(state["task_head"])
    return tok, trunk, head.cuda().eval()


def cls_and_score(tok, trunk, head, claims, evs):
    import torch
    C = np.zeros((len(claims), trunk.config.hidden_size), dtype=np.float32)
    S = np.zeros(len(claims), dtype=np.float32)
    with torch.inference_mode():
        for j in range(0, len(claims), BATCH):
            enc = tok(claims[j:j+BATCH], evs[j:j+BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            cls = trunk(**enc).last_hidden_state[:, 0]
            C[j:j+BATCH] = cls.float().cpu().numpy()
            S[j:j+BATCH] = torch.sigmoid(head(cls).float().squeeze(-1)).cpu().numpy()
    return C, S


def auroc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


def main():
    import torch
    rng = np.random.default_rng(SEED)
    fams, n_train = build(rng)
    print({k: len(v) for k, v in fams.items()}, flush=True)
    tok, trunk, head = load_model()

    # ---- (A) derivation families -------------------------------------------
    famres, rows = {}, []
    for name, items in fams.items():
        if not items:
            continue
        claims = [t["claim_ok"] for t in items] + [t["claim_bad"] for t in items]
        evs = [t["evidence"] for t in items] * 2
        _, s = cls_and_score(tok, trunk, head, claims, evs)
        n = len(items)
        ok, bad = s[:n], s[n:]
        famres[name] = {"n": n, "mean_correct": float(ok.mean()), "mean_wrong": float(bad.mean()),
                        "auroc_correct_vs_wrong": auroc(ok, bad),
                        "frac_correct_scored_higher": float((ok > bad).mean())}
        print(name, famres[name], flush=True)
        for i, t in enumerate(items):
            rows.append({k: v for k, v in t.items() if k != "evidence"} |
                        {"score_ok": float(ok[i]), "score_bad": float(bad[i])})
    pl.DataFrame(rows).write_parquet(HERE / "R15_P4_family_scores.parquet")

    # ---- (B) representation probe ------------------------------------------
    # magnitude: "The value is X." over a neutral evidence sentence
    def mk(vals):
        return ([f"The value is {v}." for v in vals],
                ["The table records a single measured quantity." for _ in vals])
    lo = rng.integers(1, 1000, 3000)              # interpolation domain
    hi = rng.integers(10_000, 100_000, 1500)      # extrapolation domain
    Cl, _ = cls_and_score(tok, trunk, head, *mk(lo))
    Ch, _ = cls_and_score(tok, trunk, head, *mk(hi))
    yl, yh = np.log10(lo.astype(float)), np.log10(hi.astype(float))

    def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
        Xtr = np.c_[Xtr, np.ones(len(Xtr))]; Xte = np.c_[Xte, np.ones(len(Xte))]
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ ytr)
        p = Xte @ w
        return float(1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())

    cut = 2000
    r2_interp = ridge_r2(Cl[:cut], yl[:cut], Cl[cut:], yl[cut:])
    r2_extrap = ridge_r2(Cl, yl, Ch, yh)

    # comparison: linear probe on CLS of a two-value string
    a = rng.integers(1, 1000, 4000); b = rng.integers(1, 1000, 4000)
    keep = a != b
    a, b = a[keep], b[keep]
    cc = [f"Alpha is {x} and Beta is {y}." for x, y in zip(a, b)]
    ee = ["The table lists two quantities." for _ in a]
    Cc, _ = cls_and_score(tok, trunk, head, cc, ee)
    ycmp = (a > b).astype(float)
    n_tr = int(0.7 * len(a))
    Xtr = np.c_[Cc[:n_tr], np.ones(n_tr)]; Xte = np.c_[Cc[n_tr:], np.ones(len(a) - n_tr)]
    A = Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ycmp[:n_tr])
    pred = Xte @ w
    acc_cmp = float(((pred > 0.5).astype(float) == ycmp[n_tr:]).mean())
    auc_cmp = auroc(pred[ycmp[n_tr:] == 1], pred[ycmp[n_tr:] == 0])

    res = {
        "probe": "R15-P4 numeracy dossier probe",
        "model": MODEL, "seed": SEED,
        "data": "TabFact test+validation, table_id-disjoint from train; zero arena, zero gold",
        "train_table_ids_excluded": n_train,
        "A_derivation_families": famres,
        "A_note": "R14-H133 measured the sum family alone at AUROC 0.4924 on 2,000 triples",
        "B_representation_probe": {
            "magnitude_ridge_r2_interpolation_1_to_999": r2_interp,
            "magnitude_ridge_r2_extrapolation_train_1_999_test_10k_100k": r2_extrap,
            "comparison_linear_probe_heldout_accuracy": acc_cmp,
            "comparison_linear_probe_heldout_auroc": auc_cmp,
            "note": "linear probe on the frozen fine-tuned trunk CLS; separates representation "
                    "content from task-head usage",
        },
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["B_representation_probe"], indent=2), flush=True)
    print("wrote", RESULT, flush=True)


if __name__ == "__main__":
    main()
