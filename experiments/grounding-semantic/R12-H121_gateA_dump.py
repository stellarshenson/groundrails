"""R12-H121 Gate A - extended per-window attribution dump on the blind arena.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 12);
full record in experiments/grounding-semantic/R12_synthesis_full_field.md
(R12-H121 DISTRACTOR-WINDOW).

The frozen read takes a max over 3-22.5 windows per sentence, of which 29-80%
belong to documents the annotator never utilized, yet no training row has ever
paired a true claim with a support-free window of its own document. Gate A asks
where the max actually lands. Identical windowing (1,500 chars, stride 750) and
identical scorer to R8-H101 / R9_PC_windowed_dump; the only change is that the
FULL per-(sentence, window) score vector, the window's document index, its index
within that document and its character offset are retained instead of collapsed
by max.

Each window additionally carries a `lex_support_free` flag from the torch-free
LOW-tier lexical manifold (groundrails.lexical, config-shipped weights): the
window is support-free for that sentence when the frozen lexical verdict does
not confirm support.

CONTAMINATION DISCIPLINE (author ruling 4, round 12): the arena's
`sentence_support_information` / `unsupported_response_sentence_keys`
annotations are read here for SENTENCE-LEVEL LABELS. This is ANALYSIS ONLY - no
quantity produced by this script may enter any lane's size, filter thresholds or
per-source mix.

Stages are idempotent; each writes its own artifact and is skipped when present.

  stage 1 (GPU)  R12-H121_gateA_scores.parquet   per-(sentence, window) scores
  stage 2 (CPU)  R12-H121_gateA_lex.parquet      + lexical support-free flags
  stage 3        R12-H121_gateA_result.json      statistics and verdict

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R12-H121_gateA_dump.py --stage all
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import importlib.util
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
ARCHIVE = ROOT / "data" / "external" / "datasets" / "dataset-ragbench.zip"

# The H105 draw-1 checkpoint, resolved from R9-H105_windowed_result.json "model".
MODEL = str(ROOT / json.loads((HERE / "R9-H105_windowed_result.json").read_text())["model"])

SCORES = HERE / "R12-H121_gateA_scores.parquet"
LEXOUT = HERE / "R12-H121_gateA_lex.parquet"
RESULT = HERE / "R12-H121_gateA_result.json"

WIN = 1500
STRIDE = 750
MAX_CHUNKS = 8
N_PER_SUBSET = 250

# Measured distractor fractions (recorded in the R12 field record), used only to
# order the subsets alongside the Gate A quantity - never fed into a build.
DISTRACTOR_FRACTION = {
    "techqa": 0.799, "tatqa": 0.737, "finqa": 0.628, "pubmedqa": 0.584,
    "hotpotqa": 0.550, "covidqa": 0.506, "delucionqa": 0.292,
}

# Subsets with 0.0% documents over WIN chars: a mid-document window cannot exist,
# so any mid-window argmax mass means the dump is misconfigured (registered check).
ZERO_MIDWINDOW_SUBSETS = ("covidqa", "pubmedqa")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def windows(chunk):
    """Sliding 1,500-char windows at stride 750; final window flush to the end.

    Byte-identical to R8-H101_windowed_read.windows; returns (text, offset) pairs
    so the dump can record where in the document each window starts.
    """
    n = len(chunk)
    if n <= WIN:
        return [(chunk, 0)]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [(chunk[s : s + WIN], s) for s in starts]


def load_annotated():
    """ARENA.load_subsets with the sentence-level annotation columns retained.

    Same filter, same seed-0 sample of N_PER_SUBSET, same MAX_CHUNKS truncation -
    the responses and documents are therefore the identical rows every arena read
    has scored; only extra columns come along.
    """
    z = zipfile.ZipFile(ARCHIVE)
    out = {}
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0)
        )
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        df = df.sample(min(N_PER_SUBSET, len(df)), seed=0)
        out[sub] = {
            "response": df["response"].to_list(),
            "documents": [d[:MAX_CHUNKS] for d in df["documents"].to_list()],
            "adherence": df["adherence_score"].cast(pl.Int8).to_numpy(),
            "response_sentences": df["response_sentences"].to_list(),
            "unsupported": df["unsupported_response_sentence_keys"].to_list(),
            "ssi": df["sentence_support_information"].to_list(),
        }
    return out


def _spans(response, texts):
    """Char spans of `texts` inside `response`, scanned left to right."""
    spans, cur = [], 0
    for t in texts:
        t = (t or "").strip()
        if not t:
            spans.append(None)
            continue
        j = response.find(t, cur)
        if j < 0:
            j = response.find(t)
        if j < 0:
            spans.append(None)
        else:
            spans.append((j, j + len(t)))
            cur = j + len(t)
    return spans


def sentence_labels(response, scored_sents, ann_sents, unsupported, ssi):
    """Sentence-level grounding label for each SCORED sentence.

    The scorer splits the response with H92.sentences (terminal punctuation, min
    25 chars, cap 12); the annotator split it its own way and keyed the pieces.
    Both are aligned by character span inside the response: an annotated sentence
    belongs to the scored sentence it overlaps most. A scored sentence is label 0
    when ANY annotated sentence it covers is unsupported, label 1 when it covers
    at least one and none are unsupported, and -1 (excluded from the gate) when
    no annotated sentence overlaps it.

    Unsupported set = `unsupported_response_sentence_keys` (populated on every
    subset) unioned with the explicit `fully_supported == False` entries of
    `sentence_support_information` (non-null on part of the field).
    """
    unsup = set(unsupported or [])
    for e in ssi or []:
        if e.get("fully_supported") is False:
            unsup.add(e.get("response_sentence_key"))

    ann_keys = [a[0] for a in (ann_sents or [])]
    ann_texts = [a[1] for a in (ann_sents or [])]
    ann_spans = _spans(response, ann_texts)
    sc_spans = _spans(response, scored_sents)

    labels = []
    for si, ss in enumerate(sc_spans):
        if ss is None:
            labels.append(-1)
            continue
        covered = []
        for ai, asp in enumerate(ann_spans):
            if asp is None:
                continue
            ov = min(ss[1], asp[1]) - max(ss[0], asp[0])
            if ov > 0 and ov >= 0.5 * (asp[1] - asp[0]):
                covered.append(ann_keys[ai])
        if not covered:
            labels.append(-1)
        elif any(k in unsup for k in covered):
            labels.append(0)
        else:
            labels.append(1)
    return labels


def stage1():
    """GPU: per-(sentence, window) scores on the frozen H105 draw-1 checkpoint."""
    if SCORES.exists():
        print(f"stage 1 skipped - {SCORES.name} exists", flush=True)
        return
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    H92 = _mod("h92", "R8-H92_decomposed_arena.py")

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"model: {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    state = torch.load(
        pathlib.Path(MODEL) / "dann_student.pt", map_location="cpu", weights_only=False
    )
    trunk = AutoModel.from_pretrained(str(pathlib.Path(MODEL) / "trunk")).cuda().eval()
    trunk.config.reference_compile = False
    task_head = nn.Linear(trunk.config.hidden_size, 1)
    task_head.load_state_dict(state["task_head"])
    task_head = task_head.cuda().eval()

    subs = load_annotated()
    print(f"RAGBench: {len(subs)} subsets, {sum(len(v['adherence']) for v in subs.values())} responses", flush=True)

    rows = {k: [] for k in (
        "subset", "resp_idx", "sent_idx", "label", "resp_label",
        "win_idx", "doc_idx", "win_in_doc", "n_win_in_doc", "char_offset", "doc_len", "score",
    )}
    sent_texts, win_texts = [], []  # kept for stage 2 (lexical) in a side parquet

    for sub, d in subs.items():
        t0 = time.time()
        flat_s, flat_w, meta = [], [], []
        for i, (resp, docs) in enumerate(zip(d["response"], d["documents"], strict=True)):
            sl = H92.sentences(resp)
            labs = sentence_labels(
                resp, sl, d["response_sentences"][i], d["unsupported"][i], d["ssi"][i]
            )
            wlist = []
            for di, k in enumerate(docs):
                ws = windows(k)
                for wi, (wtext, off) in enumerate(ws):
                    wlist.append((wtext, di, wi, len(ws), off, len(k)))
            for si, s in enumerate(sl):
                for gi, (wtext, di, wi, nw, off, dl) in enumerate(wlist):
                    flat_s.append(s)
                    flat_w.append(wtext)
                    meta.append((i, si, labs[si], int(d["adherence"][i]), gi, di, wi, nw, off, dl))

        s = np.zeros(len(flat_s), dtype=np.float32)
        with torch.inference_mode():
            for j in range(0, len(flat_s), 64):
                enc = tok(
                    flat_s[j : j + 64], flat_w[j : j + 64], return_tensors="pt",
                    padding=True, truncation=True, max_length=512,
                )
                enc = {k: v.cuda() for k, v in enc.items()}
                cls = trunk(**enc).last_hidden_state[:, 0]
                s[j : j + 64] = torch.sigmoid(task_head(cls).float().squeeze(-1)).cpu().numpy()

        for (i, si, lab, rl, gi, di, wi, nw, off, dl), sc in zip(meta, s, strict=True):
            rows["subset"].append(sub)
            rows["resp_idx"].append(i)
            rows["sent_idx"].append(si)
            rows["label"].append(lab)
            rows["resp_label"].append(rl)
            rows["win_idx"].append(gi)
            rows["doc_idx"].append(di)
            rows["win_in_doc"].append(wi)
            rows["n_win_in_doc"].append(nw)
            rows["char_offset"].append(off)
            rows["doc_len"].append(dl)
            rows["score"].append(float(sc))
        sent_texts += flat_s
        win_texts += flat_w
        print(f"  {sub:14s} pairs={len(flat_s):>6}  ({time.time() - t0:.0f}s)", flush=True)

    df = pl.DataFrame(rows).with_columns(
        pl.Series("sent_text", sent_texts), pl.Series("win_text", win_texts)
    )
    df.write_parquet(SCORES)
    print(f"stage 1 -> {SCORES}  ({len(df)} rows)", flush=True)
    del trunk, task_head
    torch.cuda.empty_cache()


def stage2(workers=8):
    """CPU: LOW-tier lexical support flag for every (sentence, window) pair."""
    if LEXOUT.exists():
        print(f"stage 2 skipped - {LEXOUT.name} exists", flush=True)
        return
    from concurrent.futures import ProcessPoolExecutor

    df = pl.read_parquet(SCORES)
    pairs = list(zip(df["sent_text"].to_list(), df["win_text"].to_list(), strict=True))
    print(f"stage 2: {len(pairs)} pairs, {workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_lex_init) as ex:
        flags = list(ex.map(_lex_one, pairs, chunksize=200))
    print(f"  lexical pass {time.time() - t0:.0f}s", flush=True)
    df = df.with_columns(pl.Series("lex_support_free", flags, dtype=pl.Int8)).drop(
        "sent_text", "win_text"
    )
    df.write_parquet(LEXOUT)
    print(f"stage 2 -> {LEXOUT}  ({len(df)} rows)", flush=True)


_LEX = {}


def _lex_init():
    import yaml
    from groundrails.lexical import LexicalVerdict

    cfg = yaml.safe_load((ROOT / "src" / "groundrails" / "config_document_processing.yaml").read_text())

    def find(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = find(v, key)
                if r is not None:
                    return r
        return None

    _LEX["v"] = LexicalVerdict.from_config({"lexical_manifolds": find(cfg, "lexical_manifolds")}, "low")


def _lex_one(pair):
    """1 when the frozen LOW-tier verdict does NOT confirm support in this window."""
    from groundrails.lexical import extract_lexical_features

    s, w = pair
    feat = extract_lexical_features(s, [w], effort="low")
    return 0 if _LEX["v"].confirmed(feat) else 1


def stage3():
    """Statistics, the registered misconfiguration check, and the verdict."""
    df = pl.read_parquet(LEXOUT)

    # argmax window per (subset, response, sentence) - the max the serving read takes
    top = (
        df.sort(["subset", "resp_idx", "sent_idx", "score"], descending=[False, False, False, True])
        .group_by(["subset", "resp_idx", "sent_idx"], maintain_order=True)
        .first()
    )

    # --- reproduction check: the dump must rebuild the recorded windowed read --
    # max over windows per sentence, MIN over sentences per response - the
    # primary decomposed-min read. Must reproduce R9-H105_windowed_result.json.
    M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
    ref = json.loads((HERE / "R9-H105_windowed_result.json").read_text())["per_subset"]
    subs_ann = load_annotated()
    repro = {}
    for sub in sorted(df["subset"].unique().to_list()):
        resp_scores = (
            top.filter(pl.col("subset") == sub)
            .group_by("resp_idx")
            .agg(pl.col("score").min().alias("s"))
            .sort("resp_idx")
        )
        y = subs_ann[sub]["adherence"]
        auc, _, _ = M59.auc_and_f1(y, resp_scores["s"].to_numpy())
        repro[sub] = {
            "auc": round(float(auc), 4),
            "recorded": ref[sub]["auc"],
            "n_window_pairs": int((df["subset"] == sub).sum()),
            "recorded_n_window_pairs": ref[sub]["n_window_pairs"],
        }
    repro_ok = all(
        abs(v["auc"] - v["recorded"]) < 5e-4 and v["n_window_pairs"] == v["recorded_n_window_pairs"]
        for v in repro.values()
    )

    # --- registered misconfiguration check -----------------------------------
    misconfig = {}
    for sub in ZERO_MIDWINDOW_SUBSETS:
        s = df.filter(pl.col("subset") == sub)
        t = top.filter(pl.col("subset") == sub)
        misconfig[sub] = {
            "windows_with_win_in_doc_gt0": int((s["win_in_doc"] > 0).sum()),
            "argmax_mid_window_count": int((t["win_in_doc"] > 0).sum()),
            "max_doc_len": int(s["doc_len"].max()),
        }
    void = any(v["argmax_mid_window_count"] > 0 for v in misconfig.values()) or not repro_ok

    labelled = top.filter(pl.col("label") >= 0)

    # --- (a) support-free share of the argmax for label-0 sentences -----------
    a_rows = {}
    for sub in sorted(df["subset"].unique().to_list()):
        t0 = labelled.filter((pl.col("subset") == sub) & (pl.col("label") == 0))
        n0 = len(t0)
        if n0:
            frac = float(t0["lex_support_free"].mean())
            wmass = float(
                (t0["score"] * t0["lex_support_free"]).sum() / max(t0["score"].sum(), 1e-9)
            )
        else:
            frac, wmass = float("nan"), float("nan")
        a_rows[sub] = {
            "n_label0_sent": n0,
            "argmax_support_free_frac": None if n0 == 0 else round(frac, 4),
            "argmax_support_free_score_weighted": None if n0 == 0 else round(wmass, 4),
            "distractor_fraction": DISTRACTOR_FRACTION.get(sub),
        }
    pooled0 = labelled.filter(pl.col("label") == 0)
    a_pooled = float(pooled0["lex_support_free"].mean())
    a_pooled_w = float(
        (pooled0["score"] * pooled0["lex_support_free"]).sum() / max(pooled0["score"].sum(), 1e-9)
    )

    # --- (b) mid-document-window share of the argmax, label 0 vs label 1 ------
    b_rows = {}
    for sub in sorted(df["subset"].unique().to_list()):
        r = {}
        for lab in (0, 1):
            t = labelled.filter((pl.col("subset") == sub) & (pl.col("label") == lab))
            r[f"mid_share_label{lab}"] = None if not len(t) else round(
                float((t["win_in_doc"] > 0).cast(pl.Float64).mean()), 4
            )
            r[f"n_label{lab}"] = len(t)
        if r["mid_share_label0"] is not None and r["mid_share_label1"] is not None:
            r["asymmetry_pp"] = round(100.0 * (r["mid_share_label0"] - r["mid_share_label1"]), 2)
        else:
            r["asymmetry_pp"] = None
        b_rows[sub] = r

    # The false-positive-inflation premise is DIRECTIONAL: it predicts label-0
    # sentences take their argmax on mid-document windows MORE often than label-1
    # ones, i.e. a positive signed asymmetry of at least 10pp. A negative
    # asymmetry of any size contradicts the premise rather than merely missing
    # the bar, so the signed value is the primary reading and |value| is recorded
    # beside it.
    asym = {s: b_rows[s]["asymmetry_pp"] for s in ("finqa", "techqa")}
    per_sub_premise = {
        s: (None if v is None else ("HOLDS" if v >= 10.0 else
            ("FALSIFIED-SIGN-INVERTED" if v <= -10.0 else "FALSIFIED-BELOW-BAR")))
        for s, v in asym.items()
    }
    fp_premise_falsified = all(
        (v is not None) and (v < 10.0) for v in asym.values()
    )

    # --- (c) ordering of (a) against the measured distractor fractions --------
    have = [
        (s, a_rows[s]["argmax_support_free_frac"], DISTRACTOR_FRACTION[s])
        for s in DISTRACTOR_FRACTION
        if a_rows.get(s, {}).get("argmax_support_free_frac") is not None
    ]
    order_a = [s for s, _, _ in sorted(have, key=lambda r: -r[1])]
    order_d = [s for s, _, _ in sorted(have, key=lambda r: -r[2])]
    if len(have) >= 3:
        from scipy.stats import spearmanr

        rho, pval = spearmanr([h[1] for h in have], [h[2] for h in have])
        rho, pval = round(float(rho), 4), round(float(pval), 4)
    else:
        rho, pval = None, None

    verdict = (
        "VOID (dump does not reproduce the recorded windowed read, or mid-document argmax "
        "mass on a 0%-over-1500-char subset)"
        if void
        else ("GATE-KILL" if a_pooled < 0.15 else "GATE-PASS")
    )

    res = {
        "gate": "R12-H121 Gate A - window attribution on the blind arena",
        "model": MODEL,
        "window": WIN, "stride": STRIDE,
        "n_pairs": len(df),
        "n_scored_sentences": len(top),
        "n_labelled_sentences": len(labelled),
        "n_unlabelled_sentences": len(top) - len(labelled),
        "reproduction_check": {"per_subset": repro, "ok": bool(repro_ok)},
        "misconfiguration_check": {"subsets": misconfig, "void": void},
        "a_argmax_support_free": {
            "pooled_frac": round(a_pooled, 4),
            "pooled_score_weighted": round(a_pooled_w, 4),
            "n_label0_sentences_pooled": len(pooled0),
            "per_subset": a_rows,
            "bar": "KILL if pooled < 0.15",
        },
        "b_mid_window_asymmetry": {
            "per_subset": b_rows,
            "finqa_techqa_asymmetry_pp": asym,
            "bar": "signed asymmetry (label0 - label1) >= +10pp holds the false-positive-"
                   "inflation premise; < +10pp falsifies it on that subset",
            "per_subset_premise": per_sub_premise,
            "fp_inflation_premise_falsified_on_both": bool(fp_premise_falsified),
            "note": "directional reading: a NEGATIVE asymmetry contradicts the premise, it "
                    "does not merely miss the bar",
        },
        "c_ordering_vs_distractor_fraction": {
            "order_by_support_free_frac": order_a,
            "order_by_distractor_fraction": order_d,
            "spearman_rho": rho, "spearman_p": pval,
        },
        "verdict": verdict,
        "contamination_note": (
            "arena sentence_support_information / unsupported_response_sentence_keys read for "
            "sentence labels; ANALYSIS ONLY per round-12 author ruling 4 - no quantity here "
            "may enter a lane's size, thresholds or mix"
        ),
    }
    RESULT.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 92)
    print("R12-H121 GATE A")
    print("=" * 92)
    print(f"  reproduction check: ok={repro_ok}")
    for s, v in repro.items():
        print(f"      {s:14s} dump {v['auc']:.4f} vs recorded {v['recorded']:.4f}  "
              f"pairs {v['n_window_pairs']} vs {v['recorded_n_window_pairs']}")
    print(f"  misconfiguration check: void={void}  {misconfig}")
    print(f"  (a) pooled argmax support-free share on label-0 sentences: {a_pooled:.4f} "
          f"(score-weighted {a_pooled_w:.4f})  bar 0.15")
    for s, r in a_rows.items():
        print(f"      {s:14s} n0={r['n_label0_sent']:>4}  frac={r['argmax_support_free_frac']}  "
              f"distractor={r['distractor_fraction']}")
    print(f"  (b) mid-window argmax asymmetry (label0 - label1), pp: {asym}")
    print(f"      per-subset premise: {per_sub_premise}   "
          f"falsified on both = {fp_premise_falsified}")
    for s, r in b_rows.items():
        print(f"      {s:14s} L0 {r['mid_share_label0']} (n={r['n_label0']})  "
              f"L1 {r['mid_share_label1']} (n={r['n_label1']})  asym {r['asymmetry_pp']}pp")
    print(f"  (c) spearman(a, distractor fraction) = {rho} (p={pval})")
    print(f"      order by (a):        {order_a}")
    print(f"      order by distractor: {order_d}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {RESULT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("1", "2", "3", "all"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.stage in ("1", "all"):
        stage1()
    if args.stage in ("2", "all"):
        stage2(args.workers)
    if args.stage in ("3", "all"):
        stage3()


if __name__ == "__main__":
    main()
