"""R19-H161 lane L1 - H1 OVERLAP-PRIOR SUPPRESSION. ANALYSIS ONLY. CPU ONLY.

Part of the R19-H161 fanout. The R19-H159 arm added five public corpora to the
training mix and LOST 0.02607 arena AUROC against the R18-H150 flagship pair
(0.68941 vs 0.71549), with the loss concentrated on finqa (-0.1429), tatqa
(-0.1328) and delucionqa (-0.1025) while hagrid GAINED (+0.0650).

H1 (this lane): the loss is OVERLAP-PRIOR SUPPRESSION. FAVA is the largest new
lane (7.93% of training pairs) and its lesson is "high lexical overlap does not
imply supported" - it marks swapped entities and altered relations inside text
that otherwise copies its source. On table-reading subsets that lesson removes
the only honest cue the cross-encoder has (a numeral's surface agreement with
its row and column labels), and on delucionqa (a car-manual QA subset with no
tables whose evidence is frequently near-verbatim in the answer) it removes the
near-copy cue. So the enriched checkpoint should show a SELECTIVELY reduced
sensitivity of its logit to lexical overlap on the three collapsed subsets, and
no such reduction on hagrid.

NOTHING HERE TRAINS. Nothing - no threshold, no cutoff, no stratum boundary -
is selected because it improves an arena number. Every constant below is fixed
in this docstring before the dump is read (the H141 discipline). The RAGBench
arena is read-only evidence.

INPUT - the shared per-pair dump written by lane A0 (not modified by this
script; this script never scores a model):

    R19-H161_pairs_h150d1.parquet   flagship draw 1
    R19-H161_pairs_h150d2.parquet   flagship draw 2
    R19-H161_pairs_h159d1.parquet   enriched (five new public corpora)

One row per (subset, item, sentence, window). Columns used here: subset,
item_id, label (item-level gold adherence, 1 = adherent), sent_idx, n_win_sent,
doc_idx, win_idx, logit (raw pre-sigmoid), is_argmax, tok_containment,
num_containment, max_common_ngram, n_num_sent.

PRE-REGISTERED CONSTANTS (fixed before reading the dump)

    NEAR_COPY_NGRAM      = 8     max_common_ngram >= 8 is "verbatim-looking"
    MIN_PAIRS            = 200   below this a per-cell correlation is reported
    MIN_ITEMS            = 30    but flagged too small to support a claim
    MIN_STRATUM_PAIRS    = 100   per-subset near-copy claim needs this many
    MIN_STRATUM_CLASS    = 20    ... and this many in EACH label class
    N_BOOT               = 500   item-clustered bootstrap, seed 0

NOISE FLOOR. The flagship's own two draws are the yardstick. Per cell:

    noise  = |h150d1 - h150d2|
    delta  = h159d1 - mean(h150d1, h150d2)
    clears = |delta| > noise

A delta that does not clear its own cell's noise floor is INDETERMINATE for
that cell and is reported as such. The bootstrap CI is reported alongside as
evidence about EVALUATION-SAMPLE noise; it does not replace the two-draw gap,
which is the seed noise the campaign actually cares about.

PRE-REGISTERED DECISION RULE, applied to the PRIMARY statistic (measurement 1
all-pairs Spearman of tok_containment against logit), in this order:

    1. n_fall < 2 of the three collapsed subsets showing a noise-clearing FALL
       -> NOT_SUPPORTED (the predicted selective suppression is absent)
    2. hagrid also falls clearing its noise floor, with |delta| >= the median
       |delta| of the three collapsed cells
       -> NOT_SUPPORTED (the suppression is global, so it cannot explain a
          loss concentrated on three subsets while hagrid gained)
    3. hagrid falls clearing its noise floor with a smaller magnitude
       -> INDETERMINATE (only partially selective)
    4. otherwise -> SUPPORTED (strong if all three collapsed cells fall,
       moderate if exactly two)

The same rule is also evaluated on the numeral-containment correlation, the
standardised slope and the argmax-only correlation. Those are CORROBORATION
and are reported next to the primary; they do not override it. If the primary
and the argmax-only reads disagree the disagreement is stated explicitly.

CALIBRATION GUARD. Checkpoints need not share a logit scale, so measurement 5
reports the near-copy mean logit BOTH raw and centred on the same checkpoint's
overall mean logit ("excess"). Only the centred figure is read as evidence of
overlap-prior suppression; a raw shift alone is a global bias change. Spearman
(the primary statistic) is invariant to any monotone recalibration.

SUPPLEMENTARY measurement 7, beyond the lane spec: the WITHIN-SENTENCE rank
correlation of tok_containment against logit, over sentences with at least 3
windows. The pooled per-subset correlation of measurement 1 mixes between-item
variation with the within-sentence window ranking, and it is the within-
sentence ranking that actually picks a sentence's argmax window and therefore
its score. Cheap, directly on-hypothesis, reported separately.

Run (detached, CPU only - the GPUs are taken by the R19-H160 draws):
  nohup setsid uv run python experiments/grounding-semantic/R19-H161_L1.py \
    >> logs/R19-H161_L1.log 2>&1 &

Writes experiments/grounding-semantic/R19-H161_L1_result.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

ROOT = Path("/home/lab/workspace/private/ai-assistants/groundrails")
EXP = ROOT / "experiments" / "grounding-semantic"
OUT_JSON = EXP / "R19-H161_L1_result.json"
SCHEMA_JSON = EXP / "R19-H161_dump_schema.json"

CKPTS = ("h150d1", "h150d2", "h159d1")
DUMPS = {c: EXP / f"R19-H161_pairs_{c}.parquet" for c in CKPTS}
FLAGSHIP = ("h150d1", "h150d2")
ENRICHED = "h159d1"

COLLAPSED = ("finqa", "tatqa", "delucionqa")
GAINER = "hagrid"

NEAR_COPY_NGRAM = 8
MIN_PAIRS = 200
MIN_ITEMS = 30
MIN_STRATUM_PAIRS = 100
MIN_STRATUM_CLASS = 20
N_BOOT = 500
SEED = 0

KEY_COLS = ("subset", "item_id", "sent_idx", "doc_idx", "win_idx")
NEEDED = (
    "subset",
    "item_id",
    "label",
    "sent_idx",
    "n_win_sent",
    "doc_idx",
    "win_idx",
    "logit",
    "is_argmax",
    "tok_containment",
    "num_containment",
    "max_common_ngram",
    "n_num_sent",
)


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------


def _f(v) -> float | None:
    """float() that turns nan/inf into None so the JSON stays valid."""
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Spearman rho and its p-value; (None, None) when undefined."""
    if x.size < 3:
        return None, None
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None, None
    res = spearmanr(x, y)
    return _f(res.statistic), _f(res.pvalue)


def slope_on_z(x: np.ndarray, y: np.ndarray) -> float | None:
    """OLS slope of y on the z-scored x, i.e. logit units per 1 SD of overlap."""
    if x.size < 3:
        return None
    sx = float(x.std(ddof=1))
    if not np.isfinite(sx) or sx == 0.0:
        return None
    z = (x - x.mean()) / sx
    var_z = float(z.var(ddof=1))
    if var_z == 0.0:
        return None
    return _f(float(np.cov(z, y, ddof=1)[0, 1]) / var_z)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUROC of scores against a binary label (1 = positive)."""
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int(scores.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(scores)
    return _f((float(r[pos].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Row indices bucketed by group value, for clustered resampling."""
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    _, starts = np.unique(g, return_index=True)
    ends = np.append(starts[1:], g.size)
    return order, [np.arange(s, e) for s, e in zip(starts, ends)]


def cluster_boot_spearman(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT
) -> tuple[float | None, float | None]:
    """Percentile CI for Spearman rho, resampling whole items with replacement."""
    if x.size < 3:
        return None, None
    order, buckets = _group_index(groups)
    xs, ys = x[order], y[order]
    n_g = len(buckets)
    if n_g < 3:
        return None, None
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_g, n_g)
        sel = np.concatenate([buckets[i] for i in pick])
        r, _p = spearman(xs[sel], ys[sel])
        if r is not None:
            draws.append(r)
    if len(draws) < n_boot // 2:
        return None, None
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return _f(lo), _f(hi)


# --------------------------------------------------------------------------
# load and verify
# --------------------------------------------------------------------------


def load_dumps() -> tuple[dict[str, pl.DataFrame], dict]:
    missing = [str(p) for p in DUMPS.values() if not p.exists()]
    if missing:
        raise SystemExit(f"L1 BLOCKED - dump artifacts absent: {missing}")

    frames: dict[str, pl.DataFrame] = {}
    for ck, path in DUMPS.items():
        df = pl.read_parquet(path)
        gap = [c for c in NEEDED if c not in df.columns]
        if gap:
            raise SystemExit(f"L1 BLOCKED - {path.name} missing columns {gap}")
        frames[ck] = df.select(NEEDED)

    # integrity: the three checkpoints must score the SAME pairs, otherwise the
    # data-side features (tok_containment, max_common_ngram) are not comparable
    ref = frames[CKPTS[0]].select(KEY_COLS).sort(KEY_COLS)
    integrity = {"n_rows": {}, "same_pair_set": {}, "same_features": {}}
    for ck in CKPTS:
        integrity["n_rows"][ck] = int(frames[ck].height)
        this = frames[ck].select(KEY_COLS).sort(KEY_COLS)
        integrity["same_pair_set"][ck] = bool(this.equals(ref))
    ref_feat = (
        frames[CKPTS[0]]
        .sort(KEY_COLS)
        .select("tok_containment", "max_common_ngram", "label")
    )
    for ck in CKPTS:
        this = (
            frames[ck].sort(KEY_COLS).select("tok_containment", "max_common_ngram", "label")
        )
        integrity["same_features"][ck] = bool(this.equals(ref_feat))

    if SCHEMA_JSON.exists():
        integrity["dump_schema_seen"] = True
    return frames, integrity


def subsets_of(frames: dict[str, pl.DataFrame]) -> list[str]:
    names: set[str] = set()
    for df in frames.values():
        names |= set(df["subset"].unique().to_list())
    return sorted(names)


# --------------------------------------------------------------------------
# per-cell measurements
# --------------------------------------------------------------------------


def cell_stats(
    df: pl.DataFrame, xcol: str, *, boot: bool
) -> dict:
    """Correlation, slope and census for one (checkpoint, subset, filter) cell."""
    sub = df.filter(pl.col(xcol).is_not_null() & pl.col("logit").is_not_null())
    n = sub.height
    out = {
        "n_pairs": int(n),
        "n_items": int(sub["item_id"].n_unique()) if n else 0,
        "rho": None,
        "p": None,
        "ci_lo": None,
        "ci_hi": None,
        "slope_z": None,
        "mean_logit": None,
        "sd_logit": None,
        "mean_x": None,
        "sd_x": None,
        "too_small": True,
    }
    if n == 0:
        return out
    x = sub[xcol].to_numpy().astype(np.float64)
    y = sub["logit"].to_numpy().astype(np.float64)
    items = sub["item_id"].to_numpy()
    rho, p = spearman(x, y)
    out["rho"], out["p"] = rho, p
    out["slope_z"] = slope_on_z(x, y)
    out["mean_logit"] = _f(y.mean())
    out["sd_logit"] = _f(y.std(ddof=1)) if n > 1 else None
    out["mean_x"] = _f(x.mean())
    out["sd_x"] = _f(x.std(ddof=1)) if n > 1 else None
    out["too_small"] = bool(n < MIN_PAIRS or out["n_items"] < MIN_ITEMS)
    if boot and not out["too_small"]:
        out["ci_lo"], out["ci_hi"] = cluster_boot_spearman(x, y, items)
    return out


def table(
    frames: dict[str, pl.DataFrame],
    subsets: list[str],
    xcol: str,
    *,
    row_filter=None,
    boot: bool = False,
) -> dict:
    """3 x 10 table of cell_stats, keyed subset -> checkpoint."""
    res: dict[str, dict] = {}
    for s in subsets:
        res[s] = {}
        for ck in CKPTS:
            df = frames[ck].filter(pl.col("subset") == s)
            if row_filter is not None:
                df = df.filter(row_filter)
            res[s][ck] = cell_stats(df, xcol, boot=boot)
    return res


def deltas(tab: dict, field: str = "rho") -> dict:
    """Enriched-vs-flagship delta per subset against that cell's own draw gap."""
    out: dict[str, dict] = {}
    for s, per_ck in tab.items():
        d1 = per_ck[FLAGSHIP[0]][field]
        d2 = per_ck[FLAGSHIP[1]][field]
        e = per_ck[ENRICHED][field]
        if d1 is None or d2 is None or e is None:
            out[s] = {
                "h150d1": d1,
                "h150d2": d2,
                "h159d1": e,
                "flag_mean": None,
                "noise": None,
                "delta": None,
                "clears": None,
                "direction": None,
                "too_small": True,
            }
            continue
        flag_mean = (d1 + d2) / 2.0
        noise = abs(d1 - d2)
        delta = e - flag_mean
        too_small = any(per_ck[c]["too_small"] for c in CKPTS)
        out[s] = {
            "h150d1": d1,
            "h150d2": d2,
            "h159d1": e,
            "flag_mean": _f(flag_mean),
            "noise": _f(noise),
            "delta": _f(delta),
            "clears": bool(abs(delta) > noise),
            "direction": "fall" if delta < 0 else ("rise" if delta > 0 else "flat"),
            "outside_draw_envelope": bool(e < min(d1, d2) or e > max(d1, d2)),
            "too_small": bool(too_small),
        }
    return out


def apply_rule(dlt: dict) -> dict:
    """The pre-registered decision rule. No inspection of the numbers first."""
    present = [s for s in COLLAPSED if dlt.get(s, {}).get("delta") is not None]
    fell = [
        s
        for s in present
        if dlt[s]["delta"] < 0 and dlt[s]["clears"] and not dlt[s]["too_small"]
    ]
    n_fall = len(fell)
    mags = [abs(dlt[s]["delta"]) for s in present]
    median_fall = float(np.median(mags)) if mags else None

    hag = dlt.get(GAINER, {})
    hag_delta = hag.get("delta")
    hag_falls = bool(
        hag_delta is not None
        and hag_delta < 0
        and hag.get("clears")
        and not hag.get("too_small")
    )

    if len(present) < len(COLLAPSED):
        verdict, reason, strength = (
            "INDETERMINATE",
            f"only {len(present)} of the three collapsed subsets resolved",
            None,
        )
    elif n_fall < 2:
        verdict, reason, strength = (
            "NOT_SUPPORTED",
            f"only {n_fall} of 3 collapsed subsets show a noise-clearing fall "
            f"(rule step 1; fell = {fell})",
            None,
        )
    elif hag_falls and median_fall is not None and abs(hag_delta) >= median_fall:
        verdict, reason, strength = (
            "NOT_SUPPORTED",
            f"hagrid also falls clearing its noise floor ({hag_delta:+.4f}) with "
            f"magnitude >= the collapsed median ({median_fall:.4f}); the "
            "suppression is global, not selective (rule step 2)",
            None,
        )
    elif hag_falls:
        verdict, reason, strength = (
            "INDETERMINATE",
            f"{n_fall} of 3 collapsed subsets fall, but hagrid falls too "
            f"({hag_delta:+.4f}, below the collapsed median {median_fall:.4f}) - "
            "only partially selective (rule step 3)",
            None,
        )
    else:
        strength = "strong" if n_fall == 3 else "moderate"
        verdict, reason = (
            "SUPPORTED",
            f"{n_fall} of 3 collapsed subsets fall clearing their own noise "
            f"floor and hagrid does not fall (rule step 4, {strength})",
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "strength": strength,
        "collapsed_fell": fell,
        "n_fall": n_fall,
        "collapsed_median_abs_delta": _f(median_fall),
        "hagrid_delta": hag_delta,
        "hagrid_falls_clearing_noise": hag_falls,
    }


# --------------------------------------------------------------------------
# measurement 5 - near-copy stratum
# --------------------------------------------------------------------------


def near_copy(frames: dict[str, pl.DataFrame], subsets: list[str]) -> dict:
    flt = pl.col("max_common_ngram") >= NEAR_COPY_NGRAM

    def one(df: pl.DataFrame) -> dict:
        strat = df.filter(flt)
        n = strat.height
        base_mean = _f(df["logit"].mean()) if df.height else None
        out = {
            "n_all_pairs": int(df.height),
            "n_stratum": int(n),
            "stratum_frac": _f(n / df.height) if df.height else None,
            "mean_logit_all_pairs": base_mean,
            "mean_logit_stratum": None,
            "excess_over_all_pairs": None,
            "mean_logit_adherent": None,
            "mean_logit_nonadherent": None,
            "n_adherent": 0,
            "n_nonadherent": 0,
            "separation": None,
            "auroc_in_stratum": None,
            "too_small": True,
        }
        if n == 0:
            return out
        y = strat["logit"].to_numpy().astype(np.float64)
        lab = strat["label"].to_numpy().astype(np.int64)
        m = float(y.mean())
        out["mean_logit_stratum"] = _f(m)
        out["excess_over_all_pairs"] = _f(m - base_mean) if base_mean is not None else None
        pos, neg = lab == 1, lab != 1
        out["n_adherent"], out["n_nonadherent"] = int(pos.sum()), int(neg.sum())
        if pos.any():
            out["mean_logit_adherent"] = _f(y[pos].mean())
        if neg.any():
            out["mean_logit_nonadherent"] = _f(y[neg].mean())
        if pos.any() and neg.any():
            out["separation"] = _f(y[pos].mean() - y[neg].mean())
            out["auroc_in_stratum"] = auroc(y, lab)
        out["too_small"] = bool(
            n < MIN_STRATUM_PAIRS
            or out["n_adherent"] < MIN_STRATUM_CLASS
            or out["n_nonadherent"] < MIN_STRATUM_CLASS
        )
        return out

    pooled = {ck: one(frames[ck]) for ck in CKPTS}
    per_subset = {
        s: {ck: one(frames[ck].filter(pl.col("subset") == s)) for ck in CKPTS}
        for s in subsets
    }

    def shift(block: dict, field: str) -> dict:
        d1, d2 = block[FLAGSHIP[0]][field], block[FLAGSHIP[1]][field]
        e = block[ENRICHED][field]
        if d1 is None or d2 is None or e is None:
            return {"flag_mean": None, "noise": None, "delta": None, "clears": None}
        fm, noise = (d1 + d2) / 2.0, abs(d1 - d2)
        return {
            "flag_mean": _f(fm),
            "noise": _f(noise),
            "delta": _f(e - fm),
            "clears": bool(abs(e - fm) > noise),
        }

    return {
        "stratum_definition": f"max_common_ngram >= {NEAR_COPY_NGRAM}",
        "pooled": pooled,
        "pooled_shift_raw_mean": shift(pooled, "mean_logit_stratum"),
        "pooled_shift_excess": shift(pooled, "excess_over_all_pairs"),
        "pooled_shift_separation": shift(pooled, "separation"),
        "pooled_shift_auroc": shift(pooled, "auroc_in_stratum"),
        "per_subset": per_subset,
        "per_subset_shift_excess": {
            s: shift(per_subset[s], "excess_over_all_pairs") for s in subsets
        },
        "per_subset_shift_auroc": {
            s: shift(per_subset[s], "auroc_in_stratum") for s in subsets
        },
    }


# --------------------------------------------------------------------------
# measurement 7 - within-sentence rank correlation (supplementary)
# --------------------------------------------------------------------------


def within_sentence(frames: dict[str, pl.DataFrame], subsets: list[str]) -> dict:
    res: dict[str, dict] = {}
    for s in subsets:
        res[s] = {}
        for ck in CKPTS:
            df = (
                frames[ck]
                .filter((pl.col("subset") == s) & (pl.col("n_win_sent") >= 3))
                .sort(["item_id", "sent_idx", "win_idx"])
            )
            if df.height == 0:
                res[s][ck] = {"n_sentences": 0, "mean_rho": None, "median_rho": None,
                              "too_small": True}
                continue
            key = (
                df["item_id"].to_numpy().astype(np.int64) * 10000
                + df["sent_idx"].to_numpy().astype(np.int64)
            )
            x = df["tok_containment"].to_numpy().astype(np.float64)
            y = df["logit"].to_numpy().astype(np.float64)
            order, buckets = _group_index(key)
            xs, ys = x[order], y[order]
            rhos = []
            for b in buckets:
                r, _p = spearman(xs[b], ys[b])
                if r is not None:
                    rhos.append(r)
            res[s][ck] = {
                "n_sentences": len(rhos),
                "mean_rho": _f(np.mean(rhos)) if rhos else None,
                "median_rho": _f(np.median(rhos)) if rhos else None,
                "too_small": bool(len(rhos) < MIN_ITEMS),
            }
    return res


def within_deltas(tab: dict) -> dict:
    out = {}
    for s, per_ck in tab.items():
        d1, d2 = per_ck[FLAGSHIP[0]]["mean_rho"], per_ck[FLAGSHIP[1]]["mean_rho"]
        e = per_ck[ENRICHED]["mean_rho"]
        if d1 is None or d2 is None or e is None:
            out[s] = {"delta": None, "noise": None, "clears": None}
            continue
        fm, noise = (d1 + d2) / 2.0, abs(d1 - d2)
        out[s] = {
            "h150d1": d1,
            "h150d2": d2,
            "h159d1": e,
            "noise": _f(noise),
            "delta": _f(e - fm),
            "clears": bool(abs(e - fm) > noise),
            "too_small": bool(any(per_ck[c]["too_small"] for c in CKPTS)),
        }
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def fmt(v, nd: int = 4, width: int = 9) -> str:
    return "n/a".rjust(width) if v is None else f"{v:+.{nd}f}".rjust(width)


def print_table(title: str, tab: dict, dlt: dict, field: str = "rho") -> None:
    print(f"\n--- {title} ---")
    print(
        f"{'subset':<14}{'h150d1':>10}{'h150d2':>10}{'h159d1':>10}"
        f"{'noise':>10}{'delta':>10}{'clears':>8}{'n_pairs':>9}  flag"
    )
    for s in sorted(tab):
        d = dlt[s]
        cell = tab[s][ENRICHED]
        clears = "-" if d["clears"] is None else ("YES" if d["clears"] else "no")
        flag = "SMALL" if d.get("too_small") else ""
        print(
            f"{s:<14}{fmt(d['h150d1'], width=10)}{fmt(d['h150d2'], width=10)}"
            f"{fmt(d['h159d1'], width=10)}{fmt(d['noise'], width=10)}"
            f"{fmt(d['delta'], width=10)}{clears:>8}{cell['n_pairs']:>9}  {flag}"
        )


def main() -> int:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[L1] start {started}")
    frames, integrity = load_dumps()
    subsets = subsets_of(frames)
    print(f"[L1] subsets ({len(subsets)}): {subsets}")
    print(f"[L1] integrity: {json.dumps(integrity)}")
    if not all(integrity["same_pair_set"].values()):
        print("[L1] WARNING - the three dumps do not cover an identical pair set")
    if not all(integrity["same_features"].values()):
        print("[L1] WARNING - data-side features differ across dumps")

    # measurement 1 - overlap sensitivity, all pairs
    m1_tok = table(frames, subsets, "tok_containment", boot=True)
    d1_tok = deltas(m1_tok)
    print_table("M1 Spearman(tok_containment, logit) - ALL PAIRS", m1_tok, d1_tok)

    num_filter = pl.col("n_num_sent") >= 1
    m1_num = table(frames, subsets, "num_containment", row_filter=num_filter)
    d1_num = deltas(m1_num)
    print_table(
        "M1 Spearman(num_containment, logit) - n_num_sent >= 1", m1_num, d1_num
    )

    # measurement 2 - standardised slope (logit units per SD of tok_containment)
    d2_slope = deltas(m1_tok, field="slope_z")
    print_table("M2 OLS slope of logit on z(tok_containment)", m1_tok, d2_slope,
                field="slope_z")

    # measurement 6 - argmax windows only
    m6_tok = table(
        frames, subsets, "tok_containment", row_filter=pl.col("is_argmax"), boot=True
    )
    d6_tok = deltas(m6_tok)
    print_table("M6 Spearman(tok_containment, logit) - ARGMAX WINDOWS", m6_tok, d6_tok)

    # measurement 5 - near-copy stratum
    m5 = near_copy(frames, subsets)
    print(f"\n--- M5 near-copy stratum ({m5['stratum_definition']}) ---")
    for ck in CKPTS:
        p = m5["pooled"][ck]
        print(
            f"{ck:<8} n={p['n_stratum']:>7} ({fmt(p['stratum_frac'], 3, 7)} of pairs) "
            f"mean={fmt(p['mean_logit_stratum'])} excess={fmt(p['excess_over_all_pairs'])} "
            f"adh={fmt(p['mean_logit_adherent'])} non={fmt(p['mean_logit_nonadherent'])} "
            f"sep={fmt(p['separation'])} auroc={fmt(p['auroc_in_stratum'])}"
        )
    for name in ("pooled_shift_raw_mean", "pooled_shift_excess",
                 "pooled_shift_separation", "pooled_shift_auroc"):
        sh = m5[name]
        print(
            f"{name:<26} delta={fmt(sh['delta'])} noise={fmt(sh['noise'])} "
            f"clears={sh['clears']}"
        )

    # measurement 7 - within-sentence (supplementary)
    m7 = within_sentence(frames, subsets)
    d7 = within_deltas(m7)
    print("\n--- M7 within-sentence mean Spearman (n_win_sent >= 3) ---")
    for s in sorted(m7):
        d = d7[s]
        clears = "-" if d.get("clears") is None else ("YES" if d["clears"] else "no")
        print(
            f"{s:<14}{fmt(d.get('h150d1'), width=10)}{fmt(d.get('h150d2'), width=10)}"
            f"{fmt(d.get('h159d1'), width=10)}{fmt(d.get('noise'), width=10)}"
            f"{fmt(d.get('delta'), width=10)}{clears:>8}"
            f"{m7[s][ENRICHED]['n_sentences']:>9}"
        )

    # verdicts - primary decides, the rest corroborate
    rule_primary = apply_rule(d1_tok)
    rule_num = apply_rule(d1_num)
    rule_slope = apply_rule(d2_slope)
    rule_argmax = apply_rule(d6_tok)
    rule_within = apply_rule(d7)

    agree_argmax = rule_argmax["verdict"] == rule_primary["verdict"]
    print("\n--- VERDICTS (pre-registered rule) ---")
    for name, r in (
        ("PRIMARY tok all-pairs", rule_primary),
        ("num_containment", rule_num),
        ("slope_z", rule_slope),
        ("argmax-only", rule_argmax),
        ("within-sentence", rule_within),
    ):
        print(f"{name:<24} {r['verdict']:<15} {r['reason']}")
    if not agree_argmax:
        print(
            "[L1] NOTE - the all-pairs and argmax-only reads DISAGREE; the "
            "argmax windows are the ones that set a sentence's score."
        )

    result = {
        "lane": "L1",
        "hypothesis": "H1 overlap-prior suppression",
        "generated_utc": started,
        "checkpoints": list(CKPTS),
        "subsets": subsets,
        "constants": {
            "near_copy_ngram": NEAR_COPY_NGRAM,
            "min_pairs": MIN_PAIRS,
            "min_items": MIN_ITEMS,
            "min_stratum_pairs": MIN_STRATUM_PAIRS,
            "min_stratum_class": MIN_STRATUM_CLASS,
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "integrity": integrity,
        "m1_spearman_tok_containment_all_pairs": m1_tok,
        "m1_spearman_tok_containment_deltas": d1_tok,
        "m1_spearman_num_containment": m1_num,
        "m1_spearman_num_containment_deltas": d1_num,
        "m2_slope_z_deltas": d2_slope,
        "m5_near_copy": m5,
        "m6_spearman_tok_containment_argmax": m6_tok,
        "m6_spearman_tok_containment_argmax_deltas": d6_tok,
        "m7_within_sentence": m7,
        "m7_within_sentence_deltas": d7,
        "rule": {
            "primary_statistic": "m1 all-pairs Spearman(tok_containment, logit)",
            "primary": rule_primary,
            "corroboration": {
                "num_containment": rule_num,
                "slope_z": rule_slope,
                "argmax_only": rule_argmax,
                "within_sentence": rule_within,
            },
            "argmax_agrees_with_primary": agree_argmax,
        },
        "verdict": rule_primary["verdict"],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\n[L1] wrote {OUT_JSON}")
    print(f"[L1] VERDICT {rule_primary['verdict']} - {rule_primary['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
