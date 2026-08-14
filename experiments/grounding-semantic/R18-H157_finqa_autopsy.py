"""R18-H157 - FINQA FAILURE-MODE AUTOPSY. ANALYSIS ONLY.

Registered in docs/experiments/semantic-grounding-experiments.md, block
"R18-H157 FINQA FAILURE-MODE AUTOPSY" (2026-08-13 ~23:15). finqa is the
flagship's sole arena loss to the incumbent (2-draw 0.6825 vs lettucedetect
0.7170). This arm measures and classifies; nothing here trains, tunes, or
selects anything on arena statistics (the H141 discipline).

Checkpoints (both banked, neither re-trained):

    draw1   models/R18-H150-arm-draw1   (the promoted flagship pair, draw 1)
    draw2   models/R18-H150-arm-draw2   (draw 2)

The read is the PRIMARY windowed decomposed-min convention, byte-identical to
the banked H150 arena reads: the frozen 250-item finqa gate sample
(R8-H77.load_subsets: adherence non-null, response > 20 chars, documents
non-empty, sample(min(250, n), seed=0), documents[:8]); each H92 sentence of
the response against every 1,500-char window (stride 750) of every retained
document; MAX over windows (per sentence), then MIN over sentences. Scoring
goes through the banked `R16-H142_G1_arm.load_run` + the `score_sets` encode
path; the only change is that per-PAIR logits are kept, so each sentence's
argmax window (the model's best shot at supporting it) is recorded, and each
item's sinking sentence (the min) is identified.

POSITIVE CONTROL, run before any analysis: each draw's reproduced finqa AUROC
must match its banked windowed value (draw1 0.6515, draw2 0.7135) to <= 1e-3,
and the structural fingerprint must match the banked read exactly (n 250,
sentences 563, pairs 2,918). The run aborts otherwise.

Measurements:

  1. per-item per-sentence scores with argmax-window provenance, both draws
       -> R18-H157_finqa_items.parquet (per-draw checkpoints written
          incrementally as R18-H157_finqa_items_draw{1,2}.parquet)
  2. error split at each draw's in-sample macro-F1-optimal threshold (the
     R17-H147 stated choice - nothing is tuned on it; the threshold-free
     rank-loss decomposition is reported alongside), false positives vs false
     negatives, with binomial standard errors; draw agreement/disagreement
  3. mechanism taxonomy per error item, rule-based pre-classification with an
     explicit override table: derivation arithmetic (the sinking sentence's
     number is absent verbatim from every window - the registered signature),
     table binding, scale/unit, entity/period confusion, window-boundary,
     other/ambiguous. Signals per item are dumped for audit
  4. probe cross-reference: the residual's class mass is read against the
     flagship's probe bank (bind_col ~0.95, bind_row ~0.99 installed;
     scale_unit flat vs control; relational compare 0.51 - derivation absent)
  5. lever mapping: each failure class to a legal lever (public-data lanes
     only; FinQA/TAT-QA are WALLED and never proposed)

Run (detached, GPU2 - shared beside the H156 training; ~3 GB needed):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
  nohup setsid uv run python experiments/grounding-semantic/R18-H157_finqa_autopsy.py \
    >> logs/R18-H157_autopsy.log 2>&1 &

Stages: `--stage score` (GPU only, writes per-draw parquets), `--stage analyze`
(CPU only, reads the banked parquets, writes the JSON). No argument runs both.
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import importlib.util
import json
import pathlib
import re
import time

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score
import torch

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT_JSON = HERE / "R18-H157_finqa_autopsy.json"
OUT_PARQUET = HERE / "R18-H157_finqa_items.parquet"
DRAW_PARQUETS = {d: HERE / f"R18-H157_finqa_items_{d}.parquet" for d in ("draw1", "draw2")}

SUBSET = "finqa"
CONTROL_TOL = 1e-3

DRAWS = {
    "draw1": {"dir": "R18-H150-arm-draw1", "banked_auc": 0.6515},
    "draw2": {"dir": "R18-H150-arm-draw2", "banked_auc": 0.7135},
}
# Structural fingerprint of the banked H150 windowed reads (both draws):
# R18-H150_arm_draw{1,2}_windowed_result.json, finqa block.
FINGERPRINT = {"n": 250, "n_sent": 563, "n_pairs": 2918}

# The flagship's probe bank (R18-H150_probes_draw1_result.json,
# R18-H150-d2_probes_draw2_result.json) - the story this autopsy cross-references.
PROBE_BANK = {
    "bind_col": {"draw1": 0.9603, "draw2": 0.948, "reading": "installed"},
    "bind_row": {"draw1": 0.9920, "draw2": 0.9881, "reading": "installed"},
    "scale_unit": {"draw1": 0.8587, "draw2": 0.8747,
                   "control": 0.8655, "reading": "flat vs control"},
    "relational_compare": {"value": 0.51, "reading": "chance - derivation absent"},
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


# --- GPU stage: per-pair scoring with argmax provenance ------------------------


def build_flat(claims, chunk_lists):
    """The windowed read's (sentence, window) pair list with full provenance.

    Pair order is byte-identical to R16-H142_G1_reads.evidence_sets("windowed"):
    item -> H92 sentence -> document -> window (ARM.windows order).
    """
    flat_s, flat_w, set_index = [], [], []
    owner, sent_idx, pair_doc, pair_win, win_span = [], [], [], [], []
    sent_texts, item_sent_slices, item_pair_slices = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        wlist = [(w, di, wi, s0, s0 + len(w)) for di, k in enumerate(ks)
                 for wi, (s0, w) in enumerate(_windows_spanned(k))]
        p0 = len(flat_s)
        s0i = len(sent_texts)
        for si, s in enumerate(H92.sentences(c)):
            sid = len(sent_texts)
            sent_texts.append(s)
            for w, di, wi, a, b in wlist:
                flat_s.append(s)
                flat_w.append(w)
                set_index.append(sid)
                owner.append(i)
                sent_idx.append(si)
                pair_doc.append(di)
                pair_win.append(wi)
                win_span.append((a, b))
        item_sent_slices.append((s0i, len(sent_texts)))
        item_pair_slices.append((p0, len(flat_s)))
    return {
        "flat_s": flat_s, "flat_w": flat_w,
        "set_index": np.array(set_index), "owner": np.array(owner),
        "sent_idx": np.array(sent_idx), "pair_doc": np.array(pair_doc),
        "pair_win": np.array(pair_win), "win_span": win_span,
        "sent_texts": sent_texts,
        "item_sent_slices": item_sent_slices, "item_pair_slices": item_pair_slices,
    }


def _windows_spanned(chunk):
    """ARM.windows with each window's char span in the parent chunk."""
    n = len(chunk)
    if n <= ARM.WIN:
        return [(0, chunk)]
    starts = list(range(0, n - ARM.WIN + 1, ARM.STRIDE))
    if starts[-1] + ARM.WIN < n:
        starts.append(n - ARM.WIN)
    return [(s, chunk[s : s + ARM.WIN]) for s in starts]


@torch.inference_mode()
def score_pairs(model, tok, flat_s, flat_w, set_index, n_sets, tag=""):
    """ARM.score_sets' exact encode path, but per-pair logits are returned
    instead of the per-set max, so each sentence's argmax window is recoverable.
    The aggregation downstream (max over the set on the LOGIT, then min over
    sentences) is the banked read's, unchanged."""
    n = len(flat_s)
    t0 = time.time()
    cls_all = torch.zeros(n, model.trunk.config.hidden_size, dtype=torch.float32)
    for i in range(0, n, 64):
        enc = tok(flat_s[i : i + 64], flat_w[i : i + 64], return_tensors="pt",
                  padding=True, truncation=True, max_length=ARM.MAX_LEN)
        enc = {k: v.cuda() for k, v in enc.items()}
        cls_all[i : i + 64] = model.encode(enc).float().cpu()
    si = torch.as_tensor(set_index, dtype=torch.long).cuda()
    ctx = model.pool_ctx(cls_all.cuda(), si, n_sets)
    lg = model.pair_logits(cls_all.cuda(), ctx[si])
    print(f"    {tag} {n} pairs in {time.time() - t0:.0f}s", flush=True)
    return lg.float().cpu().numpy()


def gpu_stage():
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    if "RTX 5000 Ada" not in dev:
        raise SystemExit(f"wrong GPU: {dev} - R18-H157 is pinned to GPU2 (RTX 5000 Ada)")

    subs = ARENA.load_subsets()
    claims, chunks, y = subs[SUBSET]
    flat = build_flat(claims, chunks)
    fp = {"n": len(y), "n_sent": len(flat["sent_texts"]), "n_pairs": len(flat["flat_s"])}
    print(f"finqa: n={fp['n']} sentences={fp['n_sent']} pairs={fp['n_pairs']}  "
          f"(banked {FINGERPRINT})", flush=True)
    if fp != FINGERPRINT:
        raise SystemExit(f"FINGERPRINT ABORT: {fp} != banked {FINGERPRINT} - "
                         "the frozen gate sample or the read geometry changed")

    n_sets = len(flat["sent_texts"])
    control = {}
    for tag, spec in DRAWS.items():
        model, tok = ARM.load_run(ROOT / "models" / spec["dir"])
        logits = score_pairs(model, tok, flat["flat_s"], flat["flat_w"],
                             flat["set_index"], n_sets, tag=f"h157/{tag}")
        del model
        torch.cuda.empty_cache()

        # per-sentence: max logit + argmax pair (the banked read aggregates on
        # the logit; sigmoid is recorded for readability - same ranking)
        sent_score = np.zeros(n_sets, dtype=np.float64)
        arg_pair = np.zeros(n_sets, dtype=np.int64)
        starts = np.searchsorted(flat["set_index"], np.arange(n_sets), side="left")
        ends = np.searchsorted(flat["set_index"], np.arange(n_sets), side="right")
        for sid in range(n_sets):
            a, b = starts[sid], ends[sid]
            j = a + int(np.argmax(logits[a:b]))
            sent_score[sid] = logits[j]
            arg_pair[sid] = j

        resp = np.array([sent_score[a:b].min() for a, b in flat["item_sent_slices"]])
        auc, _, _ = M59.auc_and_f1(y, resp)
        banked = spec["banked_auc"]
        control[tag] = {"reproduced": round(float(auc), 6), "banked": banked,
                        "abs_delta": round(abs(auc - banked), 6),
                        "pass": bool(abs(auc - banked) <= CONTROL_TOL)}
        print(f"  CONTROL {tag}: read {auc:.4f}  banked {banked:.4f}  "
              f"delta {auc - banked:+.5f}  "
              f"{'PASS' if abs(auc - banked) <= CONTROL_TOL else 'FAIL'}", flush=True)

        # sentence-level rows for this draw
        sent_prob = 1.0 / (1.0 + np.exp(-sent_score))
        rows = []
        for i in range(len(y)):
            sa, sb = flat["item_sent_slices"][i]
            sink_local = int(np.argmin(sent_score[sa:sb]))
            for k, sid in enumerate(range(sa, sb)):
                j = int(arg_pair[sid])
                a, b = flat["win_span"][j]
                rows.append({
                    "draw": tag, "item": i,
                    "label": int(y[i]),
                    "sent_idx": int(flat["sent_idx"][j]),
                    "sentence": flat["sent_texts"][sid],
                    "score": float(sent_prob[sid]),
                    "is_sinking": bool(k == sink_local),
                    "argmax_doc": int(flat["pair_doc"][j]),
                    "argmax_win": int(flat["pair_win"][j]),
                    "argmax_win_char0": a, "argmax_win_char1": b,
                    "argmax_window_text": flat["flat_w"][j],
                    "response_score": float(1.0 / (1.0 + np.exp(-resp[i]))),
                    "n_sent_item": sb - sa,
                })
        df = pl.DataFrame(rows)
        df.write_parquet(DRAW_PARQUETS[tag])
        print(f"  {tag} per-sentence scores -> {DRAW_PARQUETS[tag]}  {df.shape}",
              flush=True)

    return control


# --- number machinery (the derivation signature) --------------------------------

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_COMMA = re.compile(r"(?<=\d),(?=\d)")
_DERIV_WORDS = re.compile(
    r"\b(percent|ratio|increase|decrease|change|growth|grew|decline|margin|"
    r"difference|average|total|sum|share|rate|multiply|divid|calculat)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def extract_numbers(text):
    """Numeric surface forms in text: raw match, comma-stripped digits, value."""
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0)
        digits = raw.replace(",", "")
        try:
            val = float(digits)
        except ValueError:
            continue
        tail = text[m.end(): m.end() + 2]
        out.append({"raw": raw, "digits": digits, "value": val,
                    "is_percent": tail.lstrip().startswith("%")})
    return out


def norm_text(text):
    """Comma-stripped number rendering for verbatim presence checks."""
    return _COMMA.sub("", text)


def present_verbatim(digits, norm_texts):
    """The registered derivation signature's presence test: the digit string,
    guarded against digit/decimal neighbours, in any window."""
    pat = re.compile(r"(?<![\d.])" + re.escape(digits) + r"(?![\d.])")
    return any(pat.search(t) for t in norm_texts)


def math_candidates(x, evals, tol=0.01):
    """Scale relatives (single evidence number x 10^k) and derivation pairs
    (two evidence numbers under a finqa-register operation) for a number x
    absent verbatim. Candidates, not verdicts - the manual pass confirms."""
    if not evals:
        return {"scale": [], "derivation": []}
    e = np.array(sorted(set(evals)))
    scale, deriv = [], []
    for a in e:
        for k in (1, 2, 3, -1, -2, -3):
            t = a * 10.0**k
            if abs(t - x) / max(abs(x), 1e-9) <= tol:
                scale.append({"evidence": float(a), "factor": int(10**k)})
    if len(e) >= 2:
        A, B = np.meshgrid(e, e, indexing="ij")
        with np.errstate(divide="ignore", invalid="ignore"):
            ops = {
                "a/b*100": np.where(B != 0, A / B * 100.0, np.nan),
                "(a-b)/|b|*100": np.where(B != 0, (A - B) / np.abs(B) * 100.0, np.nan),
                "a-b": A - B,
                "a/b": np.where(B != 0, A / B, np.nan),
                "a+b": A + B,
            }
        for op, M in ops.items():
            hit = np.isfinite(M) & (np.abs(M - x) / max(abs(x), 1e-9) <= tol)
            for ai, bi in zip(*np.where(hit)):
                if ai == bi:
                    continue
                deriv.append({"a": float(e[ai]), "b": float(e[bi]), "op": op,
                              "result": float(M[ai, bi])})
    # keep the evidence readable: closest few per kind
    scale = sorted(scale, key=lambda h: abs(h["evidence"] - x))[:3]
    deriv = sorted(deriv, key=lambda h: abs(h["result"] - x))[:5]
    return {"scale": scale, "derivation": deriv}


def evidence_number_pool(windows_texts):
    """Unique evidence floats eligible as derivation operands: years and
    sub-2 values excluded (footnote indices, list markers, formula constants),
    rest deduped."""
    vals = []
    for t in windows_texts:
        for nm in extract_numbers(t):
            v = nm["value"]
            if 1900 <= v <= 2100 and v == int(v):
                continue
            if abs(v) < 2:
                continue
            vals.append(v)
    return vals


# Formula scaffolding constants ("x 100", "per cent") - absent verbatim almost
# everywhere and meaningless as claim content; excluded from the signature.
_FORMULA_CONSTANTS = frozenset({10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0})


# --- annotation access ------------------------------------------------------------


def item_annotation(r12, row):
    """Per-item annotation pack: H92-sentence -> annotated-sentence mapping,
    support keys, explanations, and the document-sentence locator map."""
    ssi = row["sentence_support_information"] or []
    unsupported = set(row["unsupported_response_sentence_keys"] or [])
    key_text = {p[0]: p[1] for p in (row["response_sentences"] or []) if len(p) >= 2}
    keys = [d["response_sentence_key"] for d in ssi] or list(key_text)
    ann_texts = [key_text.get(k, "") for k in keys]
    supp_keys = {d["response_sentence_key"]:
                 [k for k in (d["supporting_sentence_keys"] or []) if k] for d in ssi}
    expl = {d["response_sentence_key"]: d.get("explanation") or "" for d in ssi}
    smap = {}
    for di, ds in enumerate(row["documents_sentences"]):
        for pair in ds:
            if len(pair) >= 2:
                smap[pair[0]] = (di, pair[1])
    hs = r12.sentences(row["response"])
    mapping = r12.map_sentences(hs, ann_texts)
    return {"keys": keys, "ann_texts": ann_texts, "unsupported": unsupported,
            "supp_keys": supp_keys, "expl": expl, "smap": smap,
            "h92_sents": hs, "mapping": mapping}


def sentence_support_fit(r12, ann, h92_idx, docs_kept):
    """Does the annotated support of this H92 sentence fit ONE window? Returns
    (status, detail): 'covered' / 'split' / 'no_keys' / 'unmapped'. Strict
    single-doc single-window containment, R12 O4-strict semantics."""
    hits = ann["mapping"][h92_idx]
    if not hits:
        return "unmapped", {}
    sk = []
    for j in hits:
        sk.extend(ann["supp_keys"].get(ann["keys"][j], []))
    sk = [k for k in sk if k in ann["smap"]]
    if not sk:
        return "no_keys", {}
    by_doc = {}
    for k in sk:
        di, txt = ann["smap"][k]
        if di >= len(docs_kept):
            continue
        sp = r12.locate(txt, docs_kept[di])
        if sp is None:
            continue
        by_doc.setdefault(di, []).append(sp)
    if not by_doc:
        return "no_keys", {}
    if len(by_doc) > 1:
        return "split", {"reason": "support spans documents",
                         "docs": sorted(by_doc)}
    di, sps = next(iter(by_doc.items()))
    lo, hi = min(a for a, _ in sps), max(b for _, b in sps)
    spans = r12.win_spans(docs_kept[di])
    ok = any(ws <= lo and hi <= we for ws, we in spans)
    return ("covered" if ok else "split"), {
        "doc": di, "span": [lo, hi], "n_supporting": len(sps),
        "window_spans": [list(s) for s in spans]}


# --- rule-based taxonomy ----------------------------------------------------------

TAXONOMY = ("derivation_arithmetic", "table_binding", "scale_unit",
            "entity_confusion", "window_boundary", "other_ambiguous")

_EXPL_DERIV = re.compile(r"calculat|comput|arithmet|incorrect (?:result|value|total|"
                         r"calculation)|wrong (?:result|value|total)|mis-?calculat|"
                         r"should (?:be|have been)|sum|ratio|percentage", re.IGNORECASE)
_EXPL_SCALE = re.compile(r"million|billion|thousand|magnitude|decimal|unit|"
                         r"percentage point|fraction", re.IGNORECASE)
_EXPL_ENTITY = re.compile(r"wrong (?:year|period|company|entity|segment|fiscal)|"
                          r"\b20\d\d\b.*\b(?:not|instead|rather)\b|attributes? .* to|"
                          r"confus(?:es|ing|ed).*(?:year|period|compan|quarter)|"
                          r"\byear\b|fiscal|quarter", re.IGNORECASE)
_EXPL_BIND = re.compile(r"row|column|line.?item|wrong (?:line|item|figure|number)|"
                        r"confus(?:es|ing|ed)|misattribut|belongs to|refers? to", re.IGNORECASE)


def classify_fn(signals):
    """Supported item scored low: the model failed to VERIFY a true sentence."""
    if signals["support_fit"] == "split":
        return "window_boundary"
    if signals["derivation_candidates"]:
        return "derivation_arithmetic"
    if signals["scale_candidates"] and signals["n_absent_verbatim"] > 0:
        return "scale_unit"
    if signals["n_absent_verbatim"] > 0:
        if signals["derivation_register"]:
            return "derivation_arithmetic"
        return "other_ambiguous"
    if not signals["argmax_window_has_support"]:
        return "table_binding"
    return "other_ambiguous"


def classify_fp(signals):
    """Unsupported item scored high: the model failed to DETECT a false sentence.
    The annotation's explanation names the mechanism the model missed; the
    number analysis of the unsupported sentence corroborates."""
    expl = signals["unsupported_explanation"]
    if signals["unsupported_number_absent"]:
        if signals["scale_candidates"]:
            return "scale_unit"
        return "derivation_arithmetic"
    if _EXPL_DERIV.search(expl):
        return "derivation_arithmetic"
    if _EXPL_ENTITY.search(expl):
        return "entity_confusion"
    if _EXPL_SCALE.search(expl):
        return "scale_unit"
    if _EXPL_BIND.search(expl):
        return "table_binding"
    if signals["unsupported_numbers_all_present"]:
        # every claimed number surface-matches the evidence and the annotator
        # still calls the sentence unsupported - the failure is the binding
        return "table_binding"
    return "other_ambiguous"


# Manual-verification overrides, keyed (draw, item) -> final class. Every error
# item's evidence block (question, sinking sentence, argmax window, annotation
# explanation, number analysis) was read; the rule class stands unless listed
# here. Reasons, one line each:
#   36  annotator: correct calculation applied to the wrong year (2012 for 2011)
#   71  per-share fair value $78.29 claimed as a total in millions - unit error
#   85  false absence-claim ("passage does not mention X" when it does) - meta
#   200 "$ 5 2022 billion" misread as $5.2 billion - evidence literal corrupted
#   35  sinking sentence is a formula/method recital, no number content
#   100 comparison claim ("291 greater than 180") - relational-compare subtype
#       of the derivation family (the probe bank's compare leg)
#   132 formula/method recital, as 35
#   160 claim in billions against a table in millions (6.3 vs 6337) - scale
#   199 explicit difference (1,224 - 1,214 = 10) - the formula-constant filter
#       excluded the derived 10; manual pass restores the derivation class
#   31  wrong direction on a computed change (decrease vs actual increase 2,751)
#   242 lookup figure misstated (57,800 vs documented 57,100) - wrong literal
MANUAL_OVERRIDES = {
    ("draw1", 36): "entity_confusion", ("draw2", 36): "entity_confusion",
    ("draw1", 71): "scale_unit", ("draw2", 71): "scale_unit",
    ("draw1", 85): "other_ambiguous", ("draw2", 85): "other_ambiguous",
    ("draw1", 200): "table_binding", ("draw2", 200): "table_binding",
    ("draw1", 35): "other_ambiguous",
    ("draw1", 100): "derivation_arithmetic",
    ("draw1", 132): "other_ambiguous",
    ("draw1", 160): "scale_unit",
    ("draw1", 199): "derivation_arithmetic",
    ("draw2", 31): "derivation_arithmetic",
    ("draw2", 242): "table_binding",
}


# --- analysis helpers -------------------------------------------------------------


def op_threshold(y, s):
    """Macro-F1-optimal threshold, in-sample (the R17-H147 stated choice:
    it labels items correct/error; nothing is tuned on it)."""
    grid = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: f1_score(y, (s >= t).astype(int), average="macro")))


def rank_loss(y, s):
    """Per-item share of the AUROC's misordered pairs (ties count 0.5)."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    per = np.zeros(len(y))
    for i in pos:
        per[i] = np.sum(s[neg] > s[i]) + 0.5 * np.sum(s[neg] == s[i])
    for j in neg:
        per[j] = np.sum(s[pos] < s[j]) + 0.5 * np.sum(s[pos] == s[j])
    total = per[pos].sum()
    return per / max(total, 1e-9) / 2.0


def binom_se(k, n):
    p = k / n if n else 0.0
    return float(np.sqrt(p * (1 - p) / max(n, 1)))


def classify_errors(df, y, claims, chunks, r12, raw_rows, tag, thr):
    """Signals + rule class for every error item of one draw."""
    items = []
    norm_windows = []
    for ks in chunks:
        wts = [w for k in ks for w in ARM.windows(k)]
        norm_windows.append([norm_text(t) for t in wts])
    ev_pools = [evidence_number_pool([w for k in ks for w in ARM.windows(k)])
                for ks in chunks]

    sv = np.array([df.filter(pl.col("item") == i)["response_score"][0]
                   for i in range(len(y))])
    pred = (sv >= thr).astype(int)
    err = pred != y

    for i in np.where(err)[0]:
        row = raw_rows[i]
        ann = item_annotation(r12, row)
        sub = df.filter(pl.col("item") == i).sort("sent_idx")
        sink = sub.filter(pl.col("is_sinking")).row(0, named=True)
        s_txt = sink["sentence"]
        w_txt = sink["argmax_window_text"]
        s_nums = extract_numbers(s_txt)
        all_w = norm_windows[i]
        absent = [nm for nm in s_nums if not present_verbatim(nm["digits"], all_w)]
        absent_content = [nm for nm in absent
                          if nm["value"] not in _FORMULA_CONSTANTS]
        absent_in_argmax = [nm for nm in s_nums
                            if not present_verbatim(nm["digits"], [norm_text(w_txt)])]
        cands = {nm["digits"]: math_candidates(nm["value"], ev_pools[i])
                 for nm in absent_content}
        scale_c = [h for c in cands.values() for h in c["scale"]]
        deriv_c = [h for c in cands.values() for h in c["derivation"]]
        fit, fit_detail = sentence_support_fit(
            r12, ann, int(sink["sent_idx"]), row["documents"][:8])
        # does the argmax window carry the support? (locate the supporting
        # spans and test against the argmax window's char span)
        argmax_has_support = None
        if fit == "covered":
            d0 = fit_detail["doc"]
            lo, hi = fit_detail["span"]
            argmax_has_support = bool(
                d0 == sink["argmax_doc"]
                and sink["argmax_win_char0"] <= lo and hi <= sink["argmax_win_char1"])

        # FP-side evidence: the annotated unsupported sentence(s)
        unsp_expl, unsp_absent, unsp_all_present = "", False, None
        if y[i] == 0:
            u_texts = [ann["ann_texts"][j] for j, k in enumerate(ann["keys"])
                       if k in ann["unsupported"]]
            unsp_expl = " | ".join(ann["expl"].get(k, "") for k in ann["keys"]
                                   if k in ann["unsupported"])
            u_nums = [nm for t in u_texts for nm in extract_numbers(t)]
            u_absent = [nm for nm in u_nums
                        if not present_verbatim(nm["digits"], all_w)
                        and nm["value"] not in _FORMULA_CONSTANTS]
            unsp_absent = len(u_absent) > 0
            unsp_all_present = len(u_absent) == 0 and len(u_nums) > 0
            if unsp_absent:
                uc = math_candidates(u_absent[0]["value"], ev_pools[i])
                scale_c = scale_c or uc["scale"]
                deriv_c = deriv_c or uc["derivation"]
            # argmin localisation: did the sinking sentence map to an
            # annotated unsupported sentence?
            hits = ann["mapping"][int(sink["sent_idx"])]
            sink_is_unsupported = any(ann["keys"][j] in ann["unsupported"]
                                      for j in hits) if hits else False
        else:
            sink_is_unsupported = None

        signals = {
            "n_numbers_in_sinking": len(s_nums),
            "n_absent_verbatim": len(absent_content),
            "n_absent_in_argmax_window": len(absent_in_argmax),
            "absent_numbers": [nm["raw"] for nm in absent_content][:8],
            "derivation_register": bool(_DERIV_WORDS.search(s_txt)),
            "derivation_candidates": deriv_c[:3],
            "scale_candidates": scale_c[:3],
            "support_fit": fit,
            "argmax_window_has_support": argmax_has_support,
            "unsupported_explanation": unsp_expl,
            "unsupported_number_absent": unsp_absent,
            "unsupported_numbers_all_present": unsp_all_present,
            "sinking_is_annotated_unsupported": sink_is_unsupported,
        }
        cls = classify_fn(signals) if y[i] == 1 else classify_fp(signals)
        final = MANUAL_OVERRIDES.get((tag, int(i)), cls)
        items.append({
            "draw": tag, "item": int(i), "item_id": row["id"],
            "label": int(y[i]),
            "error_type": "fp" if y[i] == 0 else "fn",
            "score": round(float(sv[i]), 4), "threshold": round(thr, 4),
            "question": row["question"],
            "sinking_sentence": s_txt,
            "sinking_sent_idx": int(sink["sent_idx"]),
            "sinking_score": round(float(sink["score"]), 4),
            "argmax_doc": int(sink["argmax_doc"]),
            "argmax_win": int(sink["argmax_win"]),
            "argmax_window_text": w_txt,
            "rule_class": cls, "final_class": final,
            "signals": signals,
        })
    return items, sv, err


def analyze():
    keep = os.environ.get("CUDA_VISIBLE_DEVICES")
    R12 = _mod("r12", "R12_label_ceiling.py")
    if keep is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = keep

    subs = ARENA.load_subsets()
    claims, chunks, y = subs[SUBSET]
    raw = R12.load_rows()[SUBSET]
    y_raw = raw["adherence_score"].cast(pl.Int8).to_numpy()
    assert np.array_equal(y, y_raw), "arena and R12 row order disagree"
    assert all(a == b for a, b in zip(claims, raw["response"].to_list(), strict=True)), \
        "arena and R12 response order disagree"
    raw_rows = list(raw.iter_rows(named=True))

    n, n_pos, n_neg = len(y), int(y.sum()), int((y == 0).sum())

    # The positive control re-verified from the banked per-draw parquets, so an
    # analyze-only re-run carries the real numbers, not a placeholder. The
    # parquet's response_score is the sigmoid of the gpu-stage aggregation;
    # roc_auc_score is rank-based, so this is the same number to 1e-6.
    control = {}
    for tag, spec in DRAWS.items():
        df = pl.read_parquet(DRAW_PARQUETS[tag])
        sv = np.array([df.filter(pl.col("item") == i)["response_score"][0]
                       for i in range(n)])
        auc, _, _ = M59.auc_and_f1(y, sv)
        banked = spec["banked_auc"]
        control[tag] = {"reproduced": round(float(auc), 6), "banked": banked,
                        "abs_delta": round(abs(auc - banked), 6),
                        "pass": bool(abs(auc - banked) <= CONTROL_TOL),
                        "source": "re-verified from the banked per-draw parquet"}
    failed = [k for k, v in control.items() if not v["pass"]]
    if failed:
        raise SystemExit(f"positive control FAILED on {failed} from banked parquets")

    report = {"sample": {"n": n, "n_positive": n_pos, "n_negative": n_neg,
                         "note": "20 negatives in 250 -> the negative-side "
                                 "instrument SE is ~0.10; classes the instrument "
                                 "cannot resolve are labelled unresolvable"},
              "positive_control": control}

    per_draw, all_items, scores_by_draw = {}, [], {}
    for tag in DRAWS:
        df = pl.read_parquet(DRAW_PARQUETS[tag])
        sv = np.array([df.filter(pl.col("item") == i)["response_score"][0]
                       for i in range(n)])
        scores_by_draw[tag] = sv
        thr = op_threshold(y, sv)
        pred = (sv >= thr).astype(int)
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        items, _, _ = classify_errors(df, y, claims, chunks, R12, raw_rows, tag, thr)
        all_items.extend(items)
        rl = rank_loss(y, sv)
        per_draw[tag] = {
            "auc": round(float(roc_auc_score(y, sv)), 4),
            "operating_threshold": round(thr, 4),
            "n_errors": int(fp + fn),
            "false_positives": {"count": fp, "base": n_neg,
                                "rate": round(fp / n_neg, 4),
                                "rate_binomial_se": round(binom_se(fp, n_neg), 4)},
            "false_negatives": {"count": fn, "base": n_pos,
                                "rate": round(fn / n_pos, 4),
                                "rate_binomial_se": round(binom_se(fn, n_pos), 4)},
            "rank_loss_by_error_type": {
                "fp_share": round(float(rl[(y == 0)].sum()), 4),
                "fn_share": round(float(rl[(y == 1)].sum()), 4)},
        }
        per_draw[tag]["_err"] = np.where(pred != y)[0]
        per_draw[tag]["_rl"] = rl

    # draw agreement
    e1, e2 = per_draw["draw1"]["_err"], per_draw["draw2"]["_err"]
    both = sorted(set(e1) & set(e2))
    agreement = {
        "erred_by_both": len(both),
        "erred_only_draw1": len(set(e1) - set(e2)),
        "erred_only_draw2": len(set(e2) - set(e1)),
        "correct_in_both": n - len(set(e1) | set(e2)),
        "error_jaccard": round(len(both) / max(len(set(e1) | set(e2)), 1), 4),
        "both_items": [int(i) for i in both],
        "response_score_spearman": round(float(stats.spearmanr(
            scores_by_draw["draw1"], scores_by_draw["draw2"]).statistic), 4),
    }

    # taxonomy: per draw and consensus (erred by both), with binomial SEs and
    # the threshold-free rank-loss shares
    def tax_table(items_subset, n_err_base):
        counts = {c: 0 for c in TAXONOMY}
        for it in items_subset:
            counts[it["final_class"]] += 1
        out = {}
        for c, k in counts.items():
            se = binom_se(k, n_err_base)
            share = k / max(n_err_base, 1)
            out[c] = {"count": k, "share_of_errors": round(share, 4),
                      "share_binomial_se": round(se, 4),
                      "resolvable": bool(k == 0 or se < 0.5 * share)}
        return out

    tax = {"per_draw": {}}
    for tag in DRAWS:
        its = [it for it in all_items if it["draw"] == tag]
        tax["per_draw"][tag] = tax_table(its, len(its))
    cons_items = [it for it in all_items if it["item"] in set(both)]
    # consensus class = draw-1's class for the item (draw classes compared below)
    cons_d1 = {it["item"]: it["final_class"] for it in cons_items
               if it["draw"] == "draw1"}
    tax["consensus"] = tax_table(
        [{"final_class": cons_d1[i]} for i in cons_d1], len(both))
    tax["consensus_class_agreement_between_draws"] = round(float(np.mean([
        cons_d1[it["item"]] == it["final_class"] for it in cons_items
        if it["draw"] == "draw2"])) if both else 1.0, 4)

    # rank-loss mass per class (threshold-free): each error item's mean
    # rank-loss over the two draws, summed by final class (draw-1 class)
    rl_mean = (per_draw["draw1"]["_rl"] + per_draw["draw2"]["_rl"]) / 2.0
    rl_by_class = {c: 0.0 for c in TAXONOMY}
    item_class = {}
    for it in all_items:
        if it["draw"] == "draw1":
            item_class[it["item"]] = it["final_class"]
    for it in all_items:
        if it["draw"] == "draw2" and it["item"] not in item_class:
            item_class[it["item"]] = it["final_class"]
    for i, c in item_class.items():
        rl_by_class[c] += float(rl_mean[i])
    tax["rank_loss_share_by_class"] = {c: round(v, 4) for c, v in rl_by_class.items()}

    # probe cross-reference
    cons = tax["consensus"]
    deriv_mass = cons["derivation_arithmetic"]["count"]
    bind_mass = cons["table_binding"]["count"]
    scale_mass = cons["scale_unit"]["count"]
    rl_deriv = rl_by_class["derivation_arithmetic"]
    rl_bind = rl_by_class["table_binding"]
    verdict = ("confirm" if deriv_mass > bind_mass and rl_deriv > rl_bind
               else "contradict" if bind_mass > deriv_mass else "mixed")
    xref = {
        "probe_bank": PROBE_BANK,
        "residual_consensus_counts": {"derivation_arithmetic": deriv_mass,
                                      "table_binding": bind_mass,
                                      "scale_unit": scale_mass},
        "residual_rank_loss_mass": {"derivation_arithmetic": round(rl_deriv, 4),
                                    "table_binding": round(rl_bind, 4)},
        "verdict": verdict,
        "rationale": "the probe bank reads binding installed (bind_col ~0.95, "
                     "bind_row ~0.99) and derivation absent (relational compare "
                     "0.51, chance); a derivation-dominated residual CONFIRMS "
                     "the probe story, a binding-dominated residual CONTRADICTS it",
    }

    report.update({
        "per_draw": {t: {k: v for k, v in per_draw[t].items() if not k.startswith("_")}
                     for t in DRAWS},
        "draw_agreement": agreement,
        "taxonomy": tax,
        "probe_cross_reference": xref,
        "error_items": all_items,
        "meta": {
            "experiment": "R18-H157 FINQA FAILURE-MODE AUTOPSY",
            "licence": "ANALYSIS ONLY - no training, tuning, or selection on "
                       "arena statistics; levers are named, not built (the H141 "
                       "discipline)",
            "read": "PRIMARY windowed decomposed-min (1500/750, max over windows "
                    "on the logit, min over sentences), banked G1 load_run path",
            "checkpoints": {t: str(ROOT / "models" / s["dir"])
                            for t, s in DRAWS.items()},
            "n_manual_overrides": len(MANUAL_OVERRIDES),
        },
    })
    for t in DRAWS:
        del per_draw[t]["_err"], per_draw[t]["_rl"]
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("score", "analyze"), default=None)
    args = ap.parse_args()
    t0 = time.time()
    print(f"=== R18-H157 finqa failure-mode autopsy  {time.strftime('%F %T')} ===",
          flush=True)

    control = None
    if args.stage in (None, "score"):
        control = gpu_stage()
        failed = [k for k, v in control.items() if not v["pass"]]
        if failed:
            OUT_JSON.write_text(json.dumps(
                {"aborted": "positive control failed", "positive_control": control},
                indent=2))
            raise SystemExit(f"positive control FAILED on {failed} - read not trusted")
        print("positive control: both banked AUROCs reproduced to <= 1e-3\n",
              flush=True)
        # combine the per-draw checkpoints into the deliverable parquet
        both = pl.concat([pl.read_parquet(p) for p in DRAW_PARQUETS.values()],
                         how="vertical_relaxed")
        both.write_parquet(OUT_PARQUET)
        print(f"combined per-sentence scores -> {OUT_PARQUET}  {both.shape}", flush=True)

    if args.stage in (None, "analyze"):
        report = analyze()
        report["meta"]["runtime_seconds"] = round(time.time() - t0, 1)
        OUT_JSON.write_text(json.dumps(report, indent=2))
        print(f"\nautopsy -> {OUT_JSON}  ({time.time() - t0:.0f}s)", flush=True)

    print("=== R18-H157 AUTOPSY COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
