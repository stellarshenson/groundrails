"""R19-H161 lane A0 - the shared evidence substrate for the H159 regression. ANALYSIS ONLY.

R19-H159 (the arm that added five new public corpora to the training mix) lost
0.02607 blind mean against the R18-H150 flagship pair, and the loss concentrated
in finqa, tatqa and delucionqa. Four hypotheses compete to explain it. This lane
adjudicates none of them: it produces the one measurement substrate the three
analysis lanes read.

Nothing here trains a model, tunes a threshold or selects a serving formula. The
RAGBench arena is READ-ONLY evidence.

The read is the PRIMARY windowed decomposed-min convention, byte-identical to
the banked arena reads (`R16-H142_G1_reads.py`): the frozen per-subset gate
sample (`R8-H77.load_subsets`: adherence non-null, response > 20 chars,
documents non-empty, sample(min(250, n), seed=0), documents[:8]); each H92
sentence of the response against every 1,500-char window (stride 750) of every
retained document; MAX over windows (per sentence), then MIN over sentences.
Scoring goes through the banked `R16-H142_G1_arm.load_run` + `score_sets` encode
path; the only change is that per-PAIR logits are kept instead of the per-set
max, so each window's own contribution is recorded.

Three banked checkpoints, one parquet each:

    h150d1  models/R18-H150-arm-draw1   R18-H150_arm_draw1_windowed_result.json
    h150d2  models/R18-H150-arm-draw2   R18-H150_arm_draw2_windowed_result.json
    h159d1  models/R19-H159-arm-draw1   R19-H159_arm_draw1_windowed_result.json

POSITIVE CONTROL, run before any analysis-facing artifact is written: for every
checkpoint and every subset the item scores recomputed from this dump (MAX over
windows, then MIN over sentences) must reproduce the banked AUROC to <= 1e-3,
and the structural fingerprint (n items, n_sent sentences, n_pairs pairs) must
match the banked read EXACTLY. A miss means the scoring path diverged and the
dump is void - the checkpoint is aborted, never "corrected".

Run (detached, GPU0 only - cards 1 and 2 are running the H160 draws):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  nohup setsid uv run python experiments/grounding-semantic/R19-H161_dump.py \
    >> logs/R19-H161_dump.log 2>&1 &
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib
import re
import time

import numpy as np
import polars as pl
import torch

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
SCHEMA_JSON = HERE / "R19-H161_dump_schema.json"
FEATURES_PARQUET = HERE / "R19-H161_features.parquet"

CONTROL_TOL = 1e-3
ENCODE_BATCH = 64  # the banked read's batch, kept so the control is a like-for-like check

CHECKPOINTS = {
    "h150d1": {"dir": "R18-H150-arm-draw1", "banked": "R18-H150_arm_draw1_windowed_result.json"},
    "h150d2": {"dir": "R18-H150-arm-draw2", "banked": "R18-H150_arm_draw2_windowed_result.json"},
    "h159d1": {"dir": "R19-H159-arm-draw1", "banked": "R19-H159_arm_draw1_windowed_result.json"},
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARM = _mod("g1arm", "R16-H142_G1_arm.py")
H92 = _mod("h92", "R8-H92_decomposed_arena.py")
ARENA = H92.ARENA
M59 = ARENA.M59


# --- deterministic surface features (no tuning; definitions are frozen here) -------

TOKEN_RE = re.compile(r"[a-z0-9]+")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were "
    "be been being it its this that these those".split()
)


def raw_tokens(text):
    """Lowercased [a-z0-9]+ matches, stopwords INCLUDED - the verbatim-copy sequence."""
    return TOKEN_RE.findall(text.lower())


def content_set(toks):
    return frozenset(t for t in toks if t not in STOPWORDS)


def numerals(text):
    """Numeric surface forms, commas stripped."""
    return [m.group(0).replace(",", "") for m in NUM_RE.finditer(text)]


def position_index(toks):
    idx = {}
    for j, t in enumerate(toks):
        idx.setdefault(t, []).append(j)
    return idx


def max_common_ngram(a, b, b_index):
    """Length of the longest common contiguous token n-gram of a and b.

    Anchored extension off b's position index: every occurrence of a[i] in b is a
    candidate start, extended while the sequences agree. Exact, and linear in the
    number of matching anchors rather than in |a| x |b|.
    """
    best, la, lb = 0, len(a), len(b)
    for i, t in enumerate(a):
        if la - i <= best:
            break
        for j in b_index.get(t, ()):
            d = 0
            while i + d < la and j + d < lb and a[i + d] == b[j + d]:
                d += 1
            if d > best:
                best = d
    return best


def pair_features(s_toks, s_content, s_nums, b):
    """Surface-overlap features for one (sentence, window) pair. `b` is the cached
    window bundle: raw tokens, content set, numeral set, position index, char len."""
    inter = len(s_content & b["content"])
    union = len(s_content | b["content"])
    jac = inter / union if union else 0.0
    cont = inter / len(s_content) if s_content else 0.0
    if s_nums:
        num_cont = sum(1 for x in s_nums if x in b["nums"]) / len(s_nums)
    else:
        num_cont = None
    return jac, cont, num_cont, max_common_ngram(s_toks, b["toks"], b["index"])


# --- the windowed read's pair geometry ---------------------------------------------


def build_subset(sub, claims, chunk_lists, y):
    """Pair list plus the static (geometry + surface-feature) frame for one subset.

    Pair order is byte-identical to R16-H142_G1_reads.evidence_sets("windowed"):
    item -> H92 sentence -> document -> window within document.
    """
    flat_s, flat_w, set_index = [], [], []
    item_sent_slices = []
    cols = {k: [] for k in (
        "item_id", "label", "n_sent_item", "sent_idx", "n_win_sent", "doc_idx", "win_idx",
        "tok_jaccard", "tok_containment", "num_containment", "max_common_ngram",
        "n_num_sent", "char_len_sent", "char_len_win",
    )}

    next_sid = 0
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        wins = []
        for di, k in enumerate(ks):
            for w in ARM.windows(k):
                toks = raw_tokens(w)
                wins.append({
                    "text": w, "doc": di, "toks": toks, "content": content_set(toks),
                    "nums": frozenset(numerals(w)), "index": position_index(toks),
                    "chars": len(w),
                })
        sents = H92.sentences(c)
        lo = next_sid
        for si, s in enumerate(sents):
            sid = next_sid
            next_sid += 1
            s_toks = raw_tokens(s)
            s_content = content_set(s_toks)
            s_nums = numerals(s)
            for wi, b in enumerate(wins):
                flat_s.append(s)
                flat_w.append(b["text"])
                set_index.append(sid)
                jac, cont, num_cont, mcn = pair_features(s_toks, s_content, s_nums, b)
                cols["item_id"].append(i)
                cols["label"].append(int(y[i]))
                cols["n_sent_item"].append(len(sents))
                cols["sent_idx"].append(si)
                cols["n_win_sent"].append(len(wins))
                cols["doc_idx"].append(b["doc"])
                cols["win_idx"].append(wi)
                cols["tok_jaccard"].append(jac)
                cols["tok_containment"].append(cont)
                cols["num_containment"].append(num_cont)
                cols["max_common_ngram"].append(mcn)
                cols["n_num_sent"].append(len(s_nums))
                cols["char_len_sent"].append(len(s))
                cols["char_len_win"].append(b["chars"])
        item_sent_slices.append((lo, next_sid))

    static = pl.DataFrame(
        {"subset": [sub] * len(flat_s), **cols},
        schema={
            "subset": pl.Utf8, "item_id": pl.Int32, "label": pl.Int8,
            "n_sent_item": pl.Int16, "sent_idx": pl.Int16, "n_win_sent": pl.Int16,
            "doc_idx": pl.Int16, "win_idx": pl.Int16,
            "tok_jaccard": pl.Float32, "tok_containment": pl.Float32,
            "num_containment": pl.Float32, "max_common_ngram": pl.Int16,
            "n_num_sent": pl.Int16, "char_len_sent": pl.Int32, "char_len_win": pl.Int32,
        },
    )
    return {
        "flat_s": flat_s, "flat_w": flat_w,
        "set_index": np.asarray(set_index, dtype=np.int64),
        "item_sent_slices": item_sent_slices,
        "n_sets": next_sid,
        "static": static, "y": y,
    }


# --- scoring ------------------------------------------------------------------------


@torch.inference_mode()
def score_pairs(model, tok, flat_s, flat_w, set_index, n_sets, tag=""):
    """ARM.score_sets' exact encode path, but per-PAIR logits are returned instead
    of the per-set max. The window-ensemble context is still mean-pooled over the
    whole set before the adapter runs, so every pair logit is the one the banked
    read maxed over."""
    n = len(flat_s)
    t0 = time.time()
    cls_all = torch.zeros(n, model.trunk.config.hidden_size, dtype=torch.float32)
    for i in range(0, n, ENCODE_BATCH):
        enc = tok(flat_s[i : i + ENCODE_BATCH], flat_w[i : i + ENCODE_BATCH],
                  return_tensors="pt", padding=True, truncation=True, max_length=ARM.MAX_LEN)
        enc = {k: v.cuda() for k, v in enc.items()}
        cls_all[i : i + ENCODE_BATCH] = model.encode(enc).float().cpu()
    si = torch.as_tensor(set_index, dtype=torch.long).cuda()
    cls_gpu = cls_all.cuda()
    ctx = model.pool_ctx(cls_gpu, si, n_sets)
    out = np.empty(n, dtype=np.float32)
    step = 200_000
    for a in range(0, n, step):
        b = min(a + step, n)
        out[a:b] = model.pair_logits(cls_gpu[a:b], ctx[si[a:b]]).float().cpu().numpy()
    del cls_gpu, ctx, si
    torch.cuda.empty_cache()
    print(f"    {tag} {n} pairs in {time.time() - t0:.0f}s", flush=True)
    return out


def aggregate(logits, sub):
    """MAX over each sentence's windows, then MIN over the item's sentences - the
    banked windowed decomposed-min aggregation, on the logit."""
    set_index, n_sets = sub["set_index"], sub["n_sets"]
    starts = np.searchsorted(set_index, np.arange(n_sets), side="left")
    ends = np.searchsorted(set_index, np.arange(n_sets), side="right")
    sent_score = np.zeros(n_sets, dtype=np.float64)
    is_argmax = np.zeros(len(logits), dtype=bool)
    for sid in range(n_sets):
        a, b = starts[sid], ends[sid]
        if b <= a:
            continue
        j = a + int(np.argmax(logits[a:b]))
        sent_score[sid] = logits[j]
        is_argmax[j] = True
    item_score = np.zeros(len(sub["y"]), dtype=np.float64)
    is_sink_sent = np.zeros(n_sets, dtype=bool)
    for i, (a, b) in enumerate(sub["item_sent_slices"]):
        k = a + int(np.argmin(sent_score[a:b]))
        item_score[i] = sent_score[k]
        is_sink_sent[k] = True
    return sent_score, item_score, is_argmax, is_sink_sent


COLUMN_ORDER = [
    "subset", "item_id", "label", "n_sent_item", "sent_idx", "n_win_sent", "doc_idx",
    "win_idx", "logit", "is_argmax", "sent_score", "item_score", "is_sinking",
    "tok_jaccard", "tok_containment", "num_containment", "max_common_ngram",
    "n_num_sent", "char_len_sent", "char_len_win",
]


def pair_frame(sub, logits, sent_score, item_score, is_argmax, is_sink_sent):
    set_index = sub["set_index"]
    sent_of_item = np.repeat(
        np.arange(len(sub["y"])),
        [b - a for a, b in sub["item_sent_slices"]],
    )
    return sub["static"].with_columns(
        pl.Series("logit", logits, dtype=pl.Float32),
        pl.Series("is_argmax", is_argmax, dtype=pl.Boolean),
        pl.Series("sent_score", sent_score[set_index], dtype=pl.Float32),
        pl.Series("item_score", item_score[sent_of_item][set_index], dtype=pl.Float32),
        pl.Series("is_sinking", is_sink_sent[set_index], dtype=pl.Boolean),
    ).select(COLUMN_ORDER)


# --- banked reads ---------------------------------------------------------------------


def banked_per_subset(fname):
    """The banked windowed read's per-subset block. The three banked files were
    written by the same reader, which nests the block under `per_subset`; the
    top-level fallback keeps an older layout readable."""
    data = json.loads((HERE / fname).read_text())
    block = data.get("per_subset", data)
    return {k: v for k, v in block.items() if isinstance(v, dict) and "auc" in v}


def control_row(sub_name, sub, item_score, banked):
    auc, _, _ = M59.auc_and_f1(sub["y"], item_score)
    fp = {"n": len(sub["y"]), "n_sent": sub["n_sets"], "n_pairs": len(sub["flat_s"])}
    banked_fp = {k: int(banked[k]) for k in ("n", "n_sent", "n_pairs")}
    return {
        "subset": sub_name,
        "reproduced_auc": round(float(auc), 6),
        "banked_auc": float(banked["auc"]),
        "abs_delta": round(abs(float(auc) - float(banked["auc"])), 6),
        "auc_pass": bool(abs(float(auc) - float(banked["auc"])) <= CONTROL_TOL),
        "fingerprint": fp,
        "banked_fingerprint": banked_fp,
        "fingerprint_pass": fp == banked_fp,
    }


# --- driver ------------------------------------------------------------------------


def fail(reason):
    print(f"=== H161 DUMP FAILED: {reason} ===", flush=True)
    raise SystemExit(1)


def build_substrate():
    """The geometry + surface-feature frames for all ten subsets. Checkpoint-
    independent, so a restart reloads them from the cached parquet."""
    subs = ARENA.load_subsets()
    built = {}
    for name, (claims, chunks, y) in subs.items():
        t0 = time.time()
        built[name] = build_subset(name, claims, chunks, y)
        print(f"  {name:12s} n={len(y):>4} sentences={built[name]['n_sets']:>5} "
              f"pairs={len(built[name]['flat_s']):>6}  features {time.time() - t0:.0f}s",
              flush=True)
    pl.concat([b["static"] for b in built.values()], how="vertical").write_parquet(
        FEATURES_PARQUET)
    print(f"static feature frame -> {FEATURES_PARQUET}", flush=True)
    return built


def main():
    t_start = time.time()
    print(f"=== R19-H161 A0 substrate dump  {time.strftime('%F %T')} ===", flush=True)
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    if "RTX PRO 4000" not in dev:
        fail(f"wrong GPU: {dev} - R19-H161 is pinned to card 0 (RTX PRO 4000)")

    built = build_substrate()

    banked = {tag: banked_per_subset(spec["banked"]) for tag, spec in CHECKPOINTS.items()}
    missing = {tag: sorted(set(built) - set(b)) for tag, b in banked.items()}
    if any(missing.values()):
        fail(f"banked reads do not cover the arena: {missing}")

    control, rows_written = {}, {}
    for tag, spec in CHECKPOINTS.items():
        out_parquet = HERE / f"R19-H161_pairs_{tag}.parquet"
        print(f"\n--- {tag}  models/{spec['dir']} ---", flush=True)
        ctrl = []
        if out_parquet.exists():
            print(f"  {out_parquet.name} already on disk - re-verifying, not rescoring",
                  flush=True)
            df = pl.read_parquet(out_parquet)
            if df.height != sum(len(s["flat_s"]) for s in built.values()):
                fail(f"{out_parquet.name} has {df.height} rows, the read has "
                     f"{sum(len(s['flat_s']) for s in built.values())} pairs")
            for name, sub in built.items():
                ctrl.append(control_row(name, sub, item_scores(df, name),
                                        banked[tag][name]))
        else:
            model, tok = ARM.load_run(ROOT / "models" / spec["dir"])
            frames = []
            for name, sub in built.items():
                logits = score_pairs(model, tok, sub["flat_s"], sub["flat_w"],
                                     sub["set_index"], sub["n_sets"], tag=f"{tag}/{name}")
                sent_score, item_score, is_argmax, is_sink = aggregate(logits, sub)
                frames.append(
                    pair_frame(sub, logits, sent_score, item_score, is_argmax, is_sink))
                row = control_row(name, sub, item_score, banked[tag][name])
                ctrl.append(row)
                print(f"  CONTROL {tag}/{name:12s} read {row['reproduced_auc']:.4f}  "
                      f"banked {row['banked_auc']:.4f}  "
                      f"delta {row['reproduced_auc'] - row['banked_auc']:+.5f}  "
                      f"fp {'ok' if row['fingerprint_pass'] else row['fingerprint']}  "
                      f"{'PASS' if row['auc_pass'] and row['fingerprint_pass'] else 'FAIL'}",
                      flush=True)
            del model, tok
            torch.cuda.empty_cache()
            df = pl.concat(frames, how="vertical")

        bad = [r for r in ctrl if not (r["auc_pass"] and r["fingerprint_pass"])]
        control[tag] = {"per_subset": ctrl,
                        "pass": not bad,
                        "max_abs_auc_delta": round(max(r["abs_delta"] for r in ctrl), 6)}
        if bad:
            print(f"  MISMATCH TABLE {tag}:", flush=True)
            for r in bad:
                print(f"    {r['subset']:12s} read {r['reproduced_auc']:.4f} "
                      f"banked {r['banked_auc']:.4f} delta {r['abs_delta']:.5f} "
                      f"fp {r['fingerprint']} vs banked {r['banked_fingerprint']}", flush=True)
            print(f"=== H161 DUMP FAILED: {tag} positive control - "
                  f"{len(bad)} subsets diverged, dump void for this checkpoint ===",
                  flush=True)
            continue

        if not out_parquet.exists():
            df.write_parquet(out_parquet)
        rows_written[tag] = df.height
        print(f"  {tag} -> {out_parquet}  {df.shape}  "
              f"max |delta| {control[tag]['max_abs_auc_delta']:.5f}", flush=True)
        print(f"=== H161 DUMP COMPLETE {tag} ===", flush=True)

    if len(rows_written) != len(CHECKPOINTS):
        fail("not every checkpoint passed its positive control")

    schema = {
        "experiment": "R19-H161 lane A0 - shared evidence substrate",
        "licence": "ANALYSIS ONLY - nothing here trains, tunes or selects; the "
                   "RAGBench arena is read-only evidence",
        "read": "PRIMARY windowed decomposed-min (1500/750, MAX over windows on the "
                "logit, then MIN over sentences), banked R16-H142_G1_arm.load_run path",
        "grain": "one row per (subset, item, sentence, window)",
        "columns": [
            {"name": "subset", "dtype": "str", "meaning": "arena subset name"},
            {"name": "item_id", "dtype": "i32",
             "meaning": "0-based index into the frozen per-subset sample, in load order"},
            {"name": "label", "dtype": "i8",
             "meaning": "item-level gold adherence used by the banked read (1 = adherent)"},
            {"name": "n_sent_item", "dtype": "i16", "meaning": "sentences in this response"},
            {"name": "sent_idx", "dtype": "i16", "meaning": "0-based sentence index"},
            {"name": "n_win_sent", "dtype": "i16",
             "meaning": "windows scored for this sentence"},
            {"name": "doc_idx", "dtype": "i16",
             "meaning": "source document index of this window"},
            {"name": "win_idx", "dtype": "i16",
             "meaning": "0-based window index within the sentence's full window list"},
            {"name": "logit", "dtype": "f32",
             "meaning": "raw pre-sigmoid cross-encoder output for this (sentence, window)"},
            {"name": "is_argmax", "dtype": "bool",
             "meaning": "this window is the sentence's max (first max on a tie)"},
            {"name": "sent_score", "dtype": "f32",
             "meaning": "max over the sentence's windows, in logit space"},
            {"name": "item_score", "dtype": "f32",
             "meaning": "min over the item's sentence scores, in logit space"},
            {"name": "is_sinking", "dtype": "bool",
             "meaning": "this sentence is the item's min (first min on a tie)"},
            {"name": "tok_jaccard", "dtype": "f32",
             "meaning": "Jaccard of content-token sets, sentence vs window"},
            {"name": "tok_containment", "dtype": "f32",
             "meaning": "|sentence & window| / |sentence| over content tokens"},
            {"name": "num_containment", "dtype": "f32",
             "meaning": "fraction of the sentence's numerals present in the window; "
                        "null when the sentence has no numeral"},
            {"name": "max_common_ngram", "dtype": "i16",
             "meaning": "length of the longest common contiguous token n-gram"},
            {"name": "n_num_sent", "dtype": "i16",
             "meaning": "count of numerals in the sentence"},
            {"name": "char_len_sent", "dtype": "i32", "meaning": "characters in the sentence"},
            {"name": "char_len_win", "dtype": "i32", "meaning": "characters in the window"},
        ],
        "feature_definitions": {
            "content_tokens": "lowercased [a-z0-9]+ regex matches minus the fixed "
                              "stopword list",
            "stopwords": sorted(STOPWORDS),
            "numeral_regex": NUM_RE.pattern,
            "numeral_comparison": "commas stripped from both sides, then the sentence's "
                                  "numeral surface form is looked up in the set of the "
                                  "window's numeral surface forms",
            "max_common_ngram_tokens": "raw lowercased [a-z0-9]+ token sequence, "
                                       "stopwords INCLUDED - it measures verbatim copying",
            "tie_convention": "is_argmax and is_sinking mark the FIRST max / min in pair "
                              "order, so exactly one window per sentence and one sentence "
                              "per item carry the flag",
        },
        "checkpoints": {t: {"path": str(ROOT / "models" / s["dir"]),
                            "banked_read": s["banked"]} for t, s in CHECKPOINTS.items()},
        "positive_control": control,
        "max_auroc_deviation": round(
            max(c["max_abs_auc_delta"] for c in control.values()), 6),
        "rows": rows_written,
        "rows_per_subset": {
            t: {name: len(built[name]["flat_s"]) for name in built} for t in rows_written},
        "encode_batch": ENCODE_BATCH,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    SCHEMA_JSON.write_text(json.dumps(schema, indent=2))
    print(f"\nschema + control -> {SCHEMA_JSON}", flush=True)
    print(f"wall clock {(time.time() - t_start) / 60:.1f} min", flush=True)
    print("=== H161 DUMP ALL COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
