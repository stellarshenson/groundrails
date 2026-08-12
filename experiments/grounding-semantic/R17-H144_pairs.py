"""R17-H144 Stage-B PAIR CORPUS - build + verify only, CPU, no GPU.

Builds `R17-H144_pairs.parquet`: ~30,000 rows = ~15,000 BARE-claim minimal pairs
over public TabFact-train and FEVEROUS-train tables, at SEED 1144, over documents
DISJOINT from the R17-H143 eval set.

This is the distillation INPUT for R17-H144 (Stage B): a teacher labels these
pairs with verdict traces in a later stage, and a sub-400M student is SFT'd on
those traces.  Nothing here touches a GPU and no trace is attached to a claim.

  DERIVATION CORE (85% of rows)
    positive - a CORRECTLY DERIVED value, ABSENT from the serialized evidence
    negative - the byte-identical claim carrying a misbound / operator-swapped /
               numeral-corrupted / mis-rounded value, also ABSENT
    one negative per positive, so P(label 0 | absent) = 0.5 by construction

  RELATIONAL SUB-BLOCK (15% of rows)
    bind_col - value bound to the wrong COLUMN of the right row (both present)
    bind_row - value bound to the wrong ROW of the right column (both present)
    compare  - ordering claim against its reversal (both operands printed)

PROVENANCE.  The registered generator is the H133 lane's **v2** build (the one the
H143 eval set was sampled from).  The v2 source was overwritten in place by the v3
(trace-conditioned) rewrite and is not retrievable from git - only the v1 build is
committed, and the v1 lane was quarantined for a claim-only surface leak.  This
script therefore reconstructs v2 from three sources that ARE on disk:

  * the v3 script's shared infrastructure (serializers, `label_column`,
    `surface_ok` parity, explicit-place rounding templates, per-type direction
    balancing) - all of which the canonical doc records as v2 work carried forward
  * the v1 script's negative-family taxonomy (N1 band-targeted misbinds, N2
    operator swap, N7 numeral corruption, N_round, relational arms)
  * `R14-H133_lane.v2-SUPERSEDED.parquet`, from which the realized family mix and
    the parity/direction discipline were measured back and are reproduced here

Recorded deviations from v2, all strictly tightening or cosmetic:
  * evidence-side absence uses the punctuation-tolerant `present_numbers`
    (v3 fix); prose serializations put a full stop after every value and the
    banked `NUM_FREE` form cannot see those
  * `label_column` is the v3 version, which rejects rank-like and sentence-like
    row labels
  * relational rows carry `neg_family = "r:relational"` (v2 left it null); the
    H143 eval set uses the same string

Public data only.  Seeded, reproducible, Polars, no network, no torch.

Run:  uv run python experiments/grounding-semantic/R17-H144_pairs.py
"""

import collections
import io
import json
import math
import pathlib
import random
import re
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
FEVEROUS_SRC = ROOT / "tmp" / "R14_H133_feverous.parquet"

V2_LANE = HERE / "R14-H133_lane.v2-SUPERSEDED.parquet"   # family-mix reference
EVALSET = HERE / "R17-H143_evalset.parquet"              # documents to EXCLUDE
OUT = HERE / "R17-H144_pairs.parquet"
MANIFEST = HERE / "R17-H144_pairs_manifest.json"

SEED = 1144
CHUNK_MAX = 1500
BODY_ROWS = 6
N_PAIRS = 15_000          # 30,000 rows
SUB_FRAC = 0.15
BARE_FRAC = 0.60
DIGIT_LEN_MIN, DIGIT_LEN_MAX = 2, 7
DIGIT_LEN_CAP = 0.33
DOC_CAP_LADDER = (2, 3, 4, 5, 6, 8)   # max 8 pairs per document, as v2 did
N2_BAND = 0.06            # v2: operator swaps constrained to within 6% of correct

TAG_CORE = "quant_deriv"
TAG_REL = "quant_relational"
SINGLE = ("scale_unit", "rounding")
TWO_OP = ("sum", "difference", "mean", "ratio", "pct_change", "product")

FORM_WEIGHTS = {"row_prose": 30, "narrative": 30, "pipe": 15,
                "keyvalue": 10, "markdown": 10, "json_records": 5}

NUM_CELL = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")
NUM_FREE = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.,])")
NUM_EVID = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d,])")
WIKILINK = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]")
WIKIBARE = re.compile(r"\[\[([^|\]]+)\]\]")
WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def as_num(s):
    s = s.strip()
    if not NUM_CELL.match(s):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def fmt(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


def _canon(rx, s):
    p = set()
    for m in rx.findall(s or ""):
        v = m.replace(",", "")
        p.add(v)
        try:
            f = float(v)
        except (ValueError, OverflowError):
            continue
        if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
            continue
        p.add(str(int(round(f))) if abs(f - round(f)) < 1e-9
              else f"{f:.2f}".rstrip("0").rstrip("."))
    return p


def canon_set(s):
    """Canonical numeral forms of a CLAIM-side string (banked detector)."""
    return _canon(NUM_FREE, s)


def present_numbers(s):
    """Canonical numeral forms readable in EVIDENCE, tolerant of the trailing
    punctuation that prose serializations put after every value."""
    return _canon(NUM_EVID, s)


def digits(s):
    return sum(ch.isdigit() for ch in s)


def clean(x):
    return WS.sub(" ", (x or "")).strip()


def good_header(h):
    h = clean(h)
    return bool(h) and len(h) <= 40 and as_num(h) is None


def trailing_zeros(s):
    n = 0
    for ch in reversed(s):
        if ch == "0":
            n += 1
        else:
            break
    return n


def leading_digit(s):
    for ch in s:
        if ch.isdigit():
            return ch
    return ""


def surface_ok(p, n, allow_digit_shift=False):
    """Within-pair digit-surface parity (the v2 leak fix).

    The numeral is the only thing that differs between a positive and its
    negative, so it must match on every cheap surface statistic a character
    n-gram model reads: LEADING digit, digit count, trailing-zero count, decimal
    presence and sign.  Order-of-magnitude misbinds are allowed one digit of
    slack, because a 10x operand cannot hold the digit count fixed."""
    if p == n:
        return False
    if not (abs(digits(p) - digits(n)) <= 1 if allow_digit_shift
            else digits(p) == digits(n)):
        return False
    return (leading_digit(p) == leading_digit(n)
            and trailing_zeros(p) == trailing_zeros(n)
            and ("." in p) == ("." in n)
            and p.startswith("-") == n.startswith("-"))


def sigdigits(s):
    return len(s.replace("-", "").replace(".", "").replace(",", "").lstrip("0").rstrip("0")) or 1


# --------------------------------------------------------------------------- #
# serialization - one form per DOCUMENT
# --------------------------------------------------------------------------- #
def _attrs(hdr, row, lab_ci):
    return [(h, c) for i, (h, c) in enumerate(zip(hdr, row)) if i != lab_ci and c]


def _pipe(cap, hdr, body, lab_ci):
    return f"{cap}\n" + "\n".join(" | ".join(r) for r in [hdr] + body)


def _markdown(cap, hdr, body, lab_ci):
    sep = " | ".join("---" for _ in hdr)
    rows = ["| " + " | ".join(hdr) + " |", "| " + sep + " |"]
    rows += ["| " + " | ".join(r) + " |" for r in body]
    return f"{cap}\n" + "\n".join(rows)


def _keyvalue(cap, hdr, body, lab_ci):
    return "\n".join([cap] + ["; ".join(f"{h}: {c}" for h, c in zip(hdr, r)) for r in body])


def _json_records(cap, hdr, body, lab_ci):
    recs = [{h: c for h, c in zip(hdr, r)} for r in body]
    return cap + "\n" + json.dumps(recs, ensure_ascii=False)


def _narrative(cap, hdr, body, lab_ci):
    out = [f"{cap}."]
    for r in body:
        rest = ", ".join(f"a {h} of {c}" for h, c in _attrs(hdr, r, lab_ci))
        out.append(f"{r[lab_ci]} has {rest}." if rest else f"{r[lab_ci]} is listed.")
    return " ".join(out)


def _row_prose(cap, hdr, body, lab_ci, bind=()):
    """Every column the claim references gets the explicit `{column} of {row}`
    binding phrase (the v2 review caveat)."""
    out = [f"The following records are from {cap}."]
    want = set(bind)
    for r in body:
        cells = [(i, h, c) for i, (h, c) in enumerate(zip(hdr, r)) if i != lab_ci and c]
        if not cells:
            continue
        first = True
        for i, h, c in cells:
            if first or i in want:
                out.append(f"The {h} of {r[lab_ci]} is {c}.")
                first = False
            else:
                out.append(f"Its {h} is {c}.")
    return " ".join(out)


SERIALIZERS = {"pipe": _pipe, "markdown": _markdown, "keyvalue": _keyvalue,
               "json_records": _json_records, "narrative": _narrative,
               "row_prose": _row_prose}


def serialize(form, cap, hdr, body, lab_ci, bind=()):
    if form == "row_prose":
        return _row_prose(cap, hdr, body, lab_ci, bind).replace("\r\n", "\n")[:CHUNK_MAX]
    return SERIALIZERS[form](cap, hdr, body, lab_ci).replace("\r\n", "\n")[:CHUNK_MAX]


# --------------------------------------------------------------------------- #
# corpora - TabFact train + FEVEROUS train. PUBLIC ONLY.
# --------------------------------------------------------------------------- #
def tabfact_tables():
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    name = next(x for x in z.namelist() if x.endswith("__train.parquet"))
    d = pl.read_parquet(io.BytesIO(z.read(name))).unique(subset=["table_text"], keep="first", maintain_order=True)
    out = []
    for tid, cap, tbl in zip(d["table_id"].to_list(), d["table_caption"].to_list(),
                             d["table_text"].to_list()):
        rows = [r.split("#") for r in tbl.replace("\r\n", "\n").strip().split("\n") if r.strip()]
        rows = [[c.strip() for c in r] for r in rows]
        if len(rows) < 4:
            continue
        w = len(rows[0])
        body = [r for r in rows[1:] if len(r) == w]
        if len(body) < 4 or w < 2:
            continue
        hdr = [clean(c) for c in rows[0]]
        body = [[clean(c)[:80] for c in r] for r in body]
        out.append({"doc_id": f"tabfact:{tid}", "source": "tabfact",
                    "caption": clean(cap or "table")[:120] or "table",
                    "hdr": hdr, "body": body})
    return out


def feverous_tables():
    d = pl.read_parquet(FEVEROUS_SRC).unique(subset=["evidence"], keep="first", maintain_order=True)
    out, seen = [], set()
    for i, ev in enumerate(d["evidence"].to_list()):
        ev = WIKIBARE.sub(r"\1", WIKILINK.sub(r"\2", ev)).replace("[SEP]", "\n").strip()
        if "[ROW]" not in ev:
            continue
        segs = [s.strip() for s in ev.split("[ROW]") if s.strip()]
        rows = [[c.strip() for c in s.split("|")] for s in segs]
        widths = collections.Counter(len(r) for r in rows if len(r) >= 3)
        if not widths:
            continue
        w = widths.most_common(1)[0][0]
        keep = [r for r in rows if len(r) == w]
        if len(keep) < 5:
            continue
        hdr, body = keep[0], keep[1:]
        key = "\n".join("|".join(r) for r in keep)
        if key in seen:
            continue
        seen.add(key)
        cap = clean(segs[0].split("|")[0])[:120] or "table"
        hdr = [clean(c) for c in hdr]
        body = [[clean(c)[:80] for c in r] for r in body]
        out.append({"doc_id": f"feverous:{i}", "source": "feverous",
                    "caption": cap, "hdr": hdr, "body": body})
    return out


def label_column(hdr, body):
    for ci in range(min(3, len(hdr))):
        if not good_header(hdr[ci]):
            continue
        labs = [r[ci].strip() for r in body]
        if any((not x) or as_num(x.rstrip(". ")) is not None for x in labs):
            continue
        if any(";" in x or x.endswith(".") or len(x) > 60 for x in labs):
            continue
        if len(set(labs)) < max(3, int(0.8 * len(labs))):
            continue
        return ci
    return None


def numeric_columns(hdr, body, lab_ci):
    out = []
    for ci in range(len(hdr)):
        if ci == lab_ci or not good_header(hdr[ci]):
            continue
        vals = [(ri, as_num(r[ci])) for ri, r in enumerate(body)]
        vals = [(ri, v) for ri, v in vals if v is not None]
        if len(vals) < 4 or len({v for _, v in vals}) < 4:
            continue
        out.append((ci, vals))
    return out


# --------------------------------------------------------------------------- #
# the eight derivation types
# --------------------------------------------------------------------------- #
def template(dtype, col, ka, kb):
    if dtype == "sum":
        return f"The combined {col} of {ka} and {kb} is {{}}."
    if dtype == "difference":
        return f"The {col} of {ka} exceeds that of {kb} by {{}}."
    if dtype == "mean":
        return f"The average {col} of {ka} and {kb} is {{}}."
    if dtype == "ratio":
        return f"The ratio of the {col} of {ka} to that of {kb} is {{}}."
    if dtype == "pct_change":
        return f"From {ka} to {kb}, the {col} changed by {{}} percent."
    if dtype == "product":
        return f"The product of the {col} of {ka} and {kb} is {{}}."
    if dtype == "scale_unit":
        return f"Expressed in units of one thousandth, the {col} of {ka} is {{}}."
    raise ValueError(dtype)


ROUND_PLACES = [(10.0, "ten"), (100.0, "hundred"), (1000.0, "thousand")]


def rounding_template(col, ka, place):
    """The rounding place is STATED (v2 defect-1 fix): `approximately` left every
    coarser rounding defensible, so the wrong-place negatives were correct
    English labelled 0."""
    return f"Rounded to the nearest {place}, the {col} of {ka} is {{}}."


def rounding_steps(a):
    out = []
    for step, name in ROUND_PLACES:
        if abs(a) < 2 * step:
            continue
        if abs(round(a / step) * step - a) < 1e-9:
            continue          # already exact - the positive would be a table cell
        out.append((step, name))
    return out


def derive(dtype, a, b):
    if dtype == "sum":
        return a + b
    if dtype == "difference":
        return a - b
    if dtype == "mean":
        return (a + b) / 2
    if dtype == "ratio":
        return None if abs(b) < 1e-9 else a / b
    if dtype == "pct_change":
        return None if abs(a) < 1e-9 else (b - a) / a * 100
    if dtype == "product":
        return a * b
    if dtype == "scale_unit":
        return a * 1000
    raise ValueError(dtype)


def round_at(a, step):
    return round(a / step) * step


def band(correct, wrong):
    """Magnitude relation of a wrong-operand result to the correct one."""
    if abs(correct) < 1e-9 or abs(wrong) < 1e-9:
        return "arbitrary"
    lr = abs(math.log10(abs(wrong / correct)))
    if lr >= 1.0:
        return "oom"
    if abs(wrong - correct) / abs(correct) <= 0.15:
        return "near_miss"
    return "arbitrary"


N7_NGRAM_TOL = 0.5        # log-frequency slack allowed between a pair's numerals
_NGRAM_REF = None


def ngram_reference():
    """Character 2- and 3-gram counts over the v2 lane's derived POSITIVE values.

    A banked, independent sample of what a correctly derived numeral looks like -
    the naturalness yardstick the N7 corruption has to match."""
    global _NGRAM_REF
    if _NGRAM_REF is None:
        ref = pl.read_parquet(V2_LANE, columns=["block", "label", "v_pos"]).filter(
            (pl.col("block") == "core") & (pl.col("label") == 1.0))["v_pos"].to_list()
        c = collections.Counter()
        for s in ref:
            for n in (2, 3):
                for i in range(len(s) - n + 1):
                    g = s[i:i + n]
                    if all(ch.isdigit() or ch == "." for ch in g):
                        c[g] += 1
        _NGRAM_REF = c
    return _NGRAM_REF


def naturalness(s):
    c = ngram_reference()
    t = 0.0
    for n in (2, 3):
        for i in range(len(s) - n + 1):
            g = s[i:i + n]
            if all(ch.isdigit() or ch == "." for ch in g):
                t += math.log(c[g] + 1.0)
    return t


def transpose_digits(s, up):
    """N7 - ADJACENT digit transposition: the digit multiset is preserved, so the
    corruption cannot be read off digit statistics, and `surface_ok` rejects any
    swap that would move the leading digit or change the trailing-zero count.

    Two further constraints, both measured:

      * the swapped digits must be ADJACENT IN THE RENDERED STRING, never across
        the decimal point - a swap over the point moves the value by the whole
        integer part and lands far outside the distribution of derived results
      * of the swaps that survive, take the one whose character n-gram frequency
        is CLOSEST to the positive's, and reject the numeral outright when even
        the closest sits more than `N7_NGRAM_TOL` away.  Transposing a natural
        number leaves an unnatural digit order, which is precisely what a
        character n-gram classifier reads: unconstrained, the numeral alone
        settles the pair 0.62 of the time; under this constraint, 0.55"""
    base = naturalness(s)
    best = None
    for i in range(len(s) - 1):
        if not (s[i].isdigit() and s[i + 1].isdigit()) or s[i] == s[i + 1]:
            continue
        t = list(s)
        t[i], t[i + 1] = t[i + 1], t[i]
        t = "".join(t)
        if not surface_ok(s, t):
            continue
        try:
            if (float(t) > float(s)) != up:
                continue
        except ValueError:
            continue
        d = abs(naturalness(t) - base)
        if best is None or d < best[1]:
            best = (t, d)
    if best is None or best[1] > N7_NGRAM_TOL:
        return None
    return best[0]


def want_up(dir_count, key):
    """The minority error direction for this (family, derivation type).

    Keyed per TYPE: every family's negative being systematically larger than its
    positive leaves an ordinal cue at whatever digit position the pair first
    differs, which survives leading-digit parity.  Forcing the minority direction
    on every draw balances it to 50/50."""
    return dir_count[(key, "up")] <= dir_count[(key, "down")]


# --------------------------------------------------------------------------- #
# quotas - the v2 lane's REALIZED (dtype, family) mix, scaled to N_PAIRS
# --------------------------------------------------------------------------- #
def build_quotas():
    v2 = pl.read_parquet(V2_LANE, columns=["label", "block", "dtype", "neg_family", "arm"])
    neg = v2.filter(pl.col("label") == 0)
    core = neg.filter(pl.col("block") == "core")
    rel = neg.filter(pl.col("block") == "rel")

    core_pairs = int(round(N_PAIRS * (1 - SUB_FRAC)))
    sub_pairs = N_PAIRS - core_pairs

    src_core = {(d, f): n for d, f, n in
                core.group_by(["dtype", "neg_family"]).len().iter_rows()}
    src_rel = {a: n for a, n in rel.group_by("arm").len().iter_rows()}

    fam = collections.defaultdict(dict)
    tot = sum(src_core.values())
    alloc = 0
    items = sorted(src_core.items(), key=lambda kv: -kv[1])
    for (d, f), n in items[:-1]:
        k = int(round(core_pairs * n / tot))
        fam[d][f] = k
        alloc += k
    (d, f), _ = items[-1]
    fam[d][f] = core_pairs - alloc

    tot_r = sum(src_rel.values())
    sub, alloc = {}, 0
    arms = sorted(src_rel.items(), key=lambda kv: -kv[1])
    for a, n in arms[:-1]:
        sub[a] = int(round(sub_pairs * n / tot_r))
        alloc += sub[a]
    sub[arms[-1][0]] = sub_pairs - alloc

    tq = {d: sum(v.values()) for d, v in fam.items()}
    return tq, dict(fam), sub, core_pairs, sub_pairs, src_core, src_rel


# --------------------------------------------------------------------------- #
# core construction - BARE / SHOWN claims, no trace
# --------------------------------------------------------------------------- #
def try_core(tab, dtype, want_family, rng, dlen_count, dlen_cap, dir_count, attempts=5):
    for _ in range(attempts):
        r = _try_core_once(tab, dtype, want_family, rng, dlen_count, dlen_cap, dir_count)
        if r is not None:
            return r
    return None


def _try_core_once(tab, dtype, want_family, rng, dlen_count, dlen_cap, dir_count):
    lab_ci = tab["lab_ci"]
    cols = numeric_columns(tab["hdr"], tab["body"], lab_ci)
    if not cols:
        return None
    ci, vals = cols[rng.randrange(len(cols))]
    col = (tab["hdr"][ci] or f"column {ci}").strip() or f"column {ci}"
    single = dtype in SINGLE

    pool = rng.sample(vals, min(len(vals), BODY_ROWS))
    keep0 = {ri for ri, _ in pool}
    others = [r for r in range(len(tab["body"])) if r not in keep0]
    rng.shuffle(others)
    keep0 |= set(others[: BODY_ROWS - len(keep0)])
    in_keep = {ri: v for ri, v in vals if ri in keep0}

    cand_pairs = [(a, a) for a in pool] if single else \
        [(a, b) for a in pool for b in pool if a[0] != b[0]]
    rng.shuffle(cand_pairs)

    for (ri_i, vi), (ri_j, vj) in cand_pairs:
        if dtype in ("difference", "ratio"):
            if vi < vj:
                (ri_i, vi), (ri_j, vj) = (ri_j, vj), (ri_i, vi)
            if vj <= 0:
                continue
        if dtype == "product" and not (2 <= sigdigits(fmt(vi)) <= 3
                                       and 2 <= sigdigits(fmt(vj)) <= 3):
            continue
        ka = tab["body"][ri_i][lab_ci].strip()
        kb = tab["body"][ri_j][lab_ci].strip()
        if not ka or (not single and (not kb or ka == kb)):
            continue

        steps = rounding_steps(vi) if dtype == "rounding" else [(None, None)]
        rng.shuffle(steps)
        for step, place in steps:
            correct = round_at(vi, step) if dtype == "rounding" else derive(dtype, vi, vj)
            if correct is None:
                continue
            v_pos = fmt(correct)
            if not (DIGIT_LEN_MIN <= digits(v_pos) <= DIGIT_LEN_MAX):
                continue
            if dlen_count[digits(v_pos)] >= dlen_cap:
                continue

            fam = want_family
            up = want_up(dir_count, (fam, dtype))
            v_neg = None
            protect = {ri_i, ri_j}

            # ------------- N1 - band-targeted operand MISBIND ----------------
            if fam.startswith("N1_"):
                target = fam.split(":", 1)[0].split("_", 1)[1]
                slot = 0 if single else rng.randrange(2)
                true_v = vi if slot == 0 else vj
                row_of = ri_i if slot == 0 else ri_j
                cands = [(vv, rb) for rb, vv in in_keep.items() if rb != row_of]
                rng.shuffle(cands)
                for cv, src in cands:
                    if abs(cv - true_v) < 1e-9:
                        continue
                    ni, nj = (cv, vj) if slot == 0 else (vi, cv)
                    if dtype in ("difference", "ratio") and (nj <= 0 or ni < nj):
                        continue
                    w = round_at(ni, step) if dtype == "rounding" else derive(dtype, ni, nj)
                    if w is None or abs(w - correct) < 1e-9:
                        continue
                    if band(correct, w) != target:
                        continue
                    if (w > correct) != up:
                        continue
                    ws = fmt(w)
                    if not (DIGIT_LEN_MIN <= digits(ws) <= DIGIT_LEN_MAX):
                        continue
                    if not surface_ok(v_pos, ws, allow_digit_shift=(target == "oom")):
                        continue
                    v_neg = ws
                    protect.add(src)
                    break

            # ------------- N2 - operator swap, within 6% of correct ----------
            elif fam == "N2_operator":
                alts = [o for o in TWO_OP if o != dtype]
                rng.shuffle(alts)
                for other in alts:
                    w = derive(other, vi, vj)
                    if w is None or abs(w - correct) < 1e-9:
                        continue
                    if abs(w - correct) / max(abs(correct), 1e-9) > N2_BAND:
                        continue
                    if (w > correct) != up:
                        continue
                    ws = fmt(w)
                    if not (DIGIT_LEN_MIN <= digits(ws) <= DIGIT_LEN_MAX):
                        continue
                    if not surface_ok(v_pos, ws):
                        continue
                    v_neg = ws
                    break

            # ------------- N7 - digit transposition --------------------------
            elif fam == "N7_numeral":
                t = transpose_digits(v_pos, up)
                if t is not None and DIGIT_LEN_MIN <= digits(t) <= DIGIT_LEN_MAX:
                    v_neg = t

            # ------------- N_round - wrong direction / wrong multiple --------
            else:
                kind = fam.split(":", 1)[1]
                if kind == "wrong_direction":
                    # the OTHER adjacent multiple - the one on the far side of the
                    # true value, so the claim rounds the right cell the wrong way
                    cands = [correct - (1.0 if correct > vi else -1.0) * step]
                else:
                    # a different multiple of the STATED place, taken by rounding
                    # another cell of the same column.  Offsetting the correct
                    # multiple by a small k instead would draw the negative from a
                    # tighter band than the positives occupy, and the numeral alone
                    # then separates the pair (measured 0.600 within-pair against
                    # 0.503 for the misbound-rounding draw)
                    cands = [round_at(cv, step) for _, cv in vals]
                    rng.shuffle(cands)
                for w in cands:
                    if abs(w - correct) < 1e-9:
                        continue
                    if (w > correct) != up:
                        continue
                    ws = fmt(w)
                    if not (DIGIT_LEN_MIN <= digits(ws) <= DIGIT_LEN_MAX):
                        continue
                    if not surface_ok(v_pos, ws):
                        continue
                    v_neg = ws
                    break

            if v_neg is None or v_neg == v_pos:
                continue

            # ------------- claim form: bare vs shown, drawn independently ----
            tpl = rounding_template(col, ka, place) if dtype == "rounding" \
                else template(dtype, col, ka, kb)
            form = "bare" if rng.random() < BARE_FRAC else "shown"
            if form == "shown":
                li = tab["body"][ri_i][ci].strip()
                lj = tab["body"][ri_j][ci].strip()
                pre = f"The {col} of {ka} is {li}, so " if single else \
                    f"The {col} of {ka} is {li} and the {col} of {kb} is {lj}, so "
                tpl = pre + tpl[0].lower() + tpl[1:]

            # ------------- evidence, absence rule ----------------------------
            rows = sorted(keep0)
            body = [tab["body"][r] for r in rows]
            chunk = serialize(tab["form"], tab["caption"], tab["hdr"], body, lab_ci, (ci,))
            present = present_numbers(chunk)
            need = [fmt(vi)] if single else [fmt(vi), fmt(vj)]
            if any(not (canon_set(v) & present) for v in need):
                continue
            if not all(tab["body"][r][ci].strip() in chunk for r in protect):
                continue
            if canon_set(v_pos) & present or canon_set(v_neg) & present:
                continue

            dir_count[((fam, dtype), "up" if up else "down")] += 1
            return {
                "claim_pos": tpl.format(v_pos), "claim_neg": tpl.format(v_neg),
                "chunk": chunk, "claim_template": tpl,
                "dtype": dtype, "neg_family": fam, "claim_form": form,
                "arm": "", "serial_form": tab["form"], "doc_id": tab["doc_id"],
                "source": tab["source"], "column": col, "column_b": "",
                "key_a": ka, "key_b": kb if not single else ka,
                "operand_a": float(vi), "operand_b": float(vj),
                "round_step": float(step or 0.0), "round_place": place or "",
                "gap_stratum": "", "v_pos": v_pos, "v_neg": v_neg,
                "result_digits": digits(v_pos),
            }
    return None


# --------------------------------------------------------------------------- #
# relational sub-block - every asserted value PRESENT in the evidence
# --------------------------------------------------------------------------- #
def try_rel(tab, arm, rng, gap_count, gap_cap, attempts=4):
    for _ in range(attempts):
        r = _try_rel_once(tab, arm, rng, gap_count, gap_cap)
        if r is not None:
            return r
    return None


def _rel_chunk(tab, keep, ci, rng):
    rows = sorted(keep)
    body = [tab["body"][r] for r in rows]
    return serialize(tab["form"], tab["caption"], tab["hdr"], body, tab["lab_ci"], (ci,))


def _rel_base(tab, arm, col, col_b, ka, kb, v_pos, v_neg, gap):
    return {"arm": arm, "serial_form": tab["form"], "doc_id": tab["doc_id"],
            "source": tab["source"], "column": col, "column_b": col_b,
            "key_a": ka, "key_b": kb, "v_pos": v_pos, "v_neg": v_neg,
            "gap_stratum": gap, "dtype": "", "neg_family": "r:relational",
            "claim_form": "relational", "operand_a": None, "operand_b": None,
            "round_step": 0.0, "round_place": "", "result_digits": digits(v_pos)}


def _try_rel_once(tab, arm, rng, gap_count, gap_cap):
    lab_ci = tab["lab_ci"]
    cols = numeric_columns(tab["hdr"], tab["body"], lab_ci)
    if not cols:
        return None

    if arm == "bind_col":
        if len(cols) < 2:
            return None
        rng.shuffle(cols)
        ci, vals = cols[0]
        colx = (tab["hdr"][ci] or f"column {ci}").strip() or f"column {ci}"
        for cj, vals2 in cols[1:]:
            coly = (tab["hdr"][cj] or f"column {cj}").strip() or f"column {cj}"
            if coly == colx:
                continue
            m2 = dict(vals2)
            shared = [(ri, v) for ri, v in vals if ri in m2]
            rng.shuffle(shared)
            for ri, vx in shared:
                vy = m2[ri]
                lx, ly = fmt(vx), fmt(vy)
                if lx == ly:
                    continue
                ka = tab["body"][ri][lab_ci].strip()
                if not ka:
                    continue
                tpl = f"The {colx} of {ka} is {{}}."
                keep = {ri}
                others = [r for r in range(len(tab["body"])) if r != ri]
                rng.shuffle(others)
                keep |= set(others[: BODY_ROWS - 1])
                chunk = _rel_chunk(tab, keep, ci, rng)
                pres = present_numbers(chunk)
                if not (canon_set(lx) & pres) or not (canon_set(ly) & pres):
                    continue
                return {"claim_pos": tpl.format(lx), "claim_neg": tpl.format(ly),
                        "chunk": chunk, "claim_template": tpl,
                        **_rel_base(tab, arm, colx, coly, ka, ka, lx, ly, "")}
        return None

    ci, vals = cols[rng.randrange(len(cols))]
    col = (tab["hdr"][ci] or f"column {ci}").strip() or f"column {ci}"
    if len(vals) < 2:
        return None

    if arm == "bind_row":
        for _ in range(6):
            (ra, va), (rb, vb) = [vals[p] for p in rng.sample(range(len(vals)), 2)]
            la, lb = fmt(va), fmt(vb)
            ka, kb = tab["body"][ra][lab_ci].strip(), tab["body"][rb][lab_ci].strip()
            if la == lb or not ka or not kb or ka == kb:
                continue
            tpl = f"The {col} of {ka} is {{}}."
            keep = {ra, rb}
            others = [r for r in range(len(tab["body"])) if r not in keep]
            rng.shuffle(others)
            keep |= set(others[: BODY_ROWS - 2])
            chunk = _rel_chunk(tab, keep, ci, rng)
            pres = present_numbers(chunk)
            if not (canon_set(la) & pres) or not (canon_set(lb) & pres):
                continue
            return {"claim_pos": tpl.format(la), "claim_neg": tpl.format(lb),
                    "chunk": chunk, "claim_template": tpl,
                    **_rel_base(tab, arm, col, col, ka, kb, la, lb, "")}
        return None

    for _ in range(8):
        (ra, va), (rb, vb) = [vals[p] for p in rng.sample(range(len(vals)), 2)]
        if abs(va - vb) < 1e-9 or min(abs(va), abs(vb)) < 1e-9:
            continue
        if (va < 0) != (vb < 0):
            continue
        gap = abs(va - vb) / min(abs(va), abs(vb))
        stratum = "lt10pct" if gap < 0.10 else ("10to100pct" if gap <= 1.0 else "gt100pct")
        if gap_count[stratum] >= gap_cap:
            continue
        (rh, vh), (rl, vl) = ((ra, va), (rb, vb)) if va > vb else ((rb, vb), (ra, va))
        k_hi, k_lo = tab["body"][rh][lab_ci].strip(), tab["body"][rl][lab_ci].strip()
        if not k_hi or not k_lo or k_hi == k_lo:
            continue
        lh, ll = fmt(vh), fmt(vl)
        cp = f"The {col} of {k_hi} is greater than the {col} of {k_lo}."
        cn = f"The {col} of {k_lo} is greater than the {col} of {k_hi}."
        keep = {ra, rb}
        others = [r for r in range(len(tab["body"])) if r not in keep]
        rng.shuffle(others)
        keep |= set(others[: BODY_ROWS - 2])
        chunk = _rel_chunk(tab, keep, ci, rng)
        pres = present_numbers(chunk)
        if not (canon_set(lh) & pres) or not (canon_set(ll) & pres):
            continue
        gap_count[stratum] += 1
        return {"claim_pos": cp, "claim_neg": cn, "chunk": chunk,
                "claim_template": "", **_rel_base(tab, "compare", col, col,
                                                  k_hi, k_lo, lh, ll, stratum)}
    return None


# --------------------------------------------------------------------------- #
# excluded documents - the R17-H143 eval set's own tables
# --------------------------------------------------------------------------- #
def evalset_documents():
    """The eval set's documents, identified by CONTENT rather than by `doc_id`.

    The eval set carries no `doc_id` of its own; its rows join back to the v2 lane
    on (pair_id, claim, label), and on `chunk` for the 50 control pairs.  But the
    v2 lane's FEVEROUS ids are `feverous:{i}` over a `unique()` that was never
    order-stable, so `i` addresses a DIFFERENT table on every run - measured: 0 of
    95 sampled v2 FEVEROUS ids resolve to the table their claim came from.  Only
    the TabFact ids (which carry the corpus's own `table_id`) survive a rebuild.

    The disjointness bar is therefore enforced on content: an eval chunk is a
    serialization of its table's cells, so the canonical numerals readable in the
    chunk are a subset of the table's own.  A fresh table matching any eval chunk
    on that test is excluded whole."""
    ev = pl.read_parquet(EVALSET)
    v2 = pl.read_parquet(V2_LANE, columns=["pair_id", "claim", "label", "chunk", "doc_id"])
    graded = ev.filter(~pl.col("control")).join(
        v2.select(["pair_id", "claim", "label", "doc_id"]).with_columns(
            pl.col("label").cast(pl.Int8)),
        on=["pair_id", "claim", "label"], how="left")
    ctrl = ev.filter(pl.col("control")).join(
        v2.select(["chunk", "doc_id"]).unique(subset=["chunk"]), on="chunk", how="left")
    unmatched = int(graded["doc_id"].is_null().sum() + ctrl["doc_id"].is_null().sum())
    ids = set(graded["doc_id"].drop_nulls().to_list()) | set(ctrl["doc_id"].drop_nulls().to_list())
    chunks = set(graded["chunk"].to_list()) | set(ctrl["chunk"].to_list())
    prints = [(c.lower(), frozenset(present_numbers(c))) for c in chunks]
    prints = [(c, p) for c, p in prints if len(p) >= 4]
    return ids, prints, len(ev), unmatched


def long_strings(t):
    """The table's distinctive text - captions, headers and non-numeric cells long
    enough that a chance collision with an unrelated table is negligible."""
    out = {clean(t["caption"]).lower()}
    out |= {clean(h).lower() for h in t["hdr"]}
    out |= {clean(c).lower() for r in t["body"] for c in r}
    return {s for s in out if len(s) >= 6 and as_num(s) is None}


def excluded_tables(raw, prints):
    """Indices of `raw` tables that carry an eval-set document's content.

    Two signals must agree.  A chunk is a serialization of its table, so (i) at
    least three of the table's distinctive long strings are readable in the chunk -
    the discriminating test - and (ii) at least 60% of the chunk's canonical
    numerals sit inside the table's own.  The numeral test carried alone matched
    11,149 of 23,423 tables, because a chunk whose numerals are all small integers
    is a subset of half the corpus; carried as a strict subset it MISSED a real eval
    document, whose chunk showed three numerals the table does not hold (the
    1,500-character serving cap and the differing wiki-markup stripping of the v2
    build both leave numeral fragments behind).

    Candidates come from an inverted index on the chunk's three rarest INDEXED
    numerals, so the sweep is linear in matches rather than in tables x chunks.
    Indexed, because a numeral no table carries at all would otherwise be picked as
    the rarest and return an empty candidate list."""
    table_nums, table_strs = [], []
    index = collections.defaultdict(list)
    for ti, t in enumerate(raw):
        nums = present_numbers(" ".join([t["caption"]] + t["hdr"]
                                        + [c for r in t["body"] for c in r]))
        table_nums.append(nums)
        table_strs.append(long_strings(t))
        for v in nums:
            index[v].append(ti)
    out = set()
    for chunk, p in prints:
        indexed = sorted((v for v in p if index.get(v)), key=lambda v: len(index[v]))
        cands = set()
        for v in indexed[:3]:
            cands |= set(index[v])
        for ti in sorted(cands):
            if ti in out or len(p & table_nums[ti]) < 0.6 * len(p):
                continue
            strs = table_strs[ti]
            hit = sum(1 for s in strs if s in chunk)
            if hit >= min(3, len(strs)) and hit > 0:
                out.add(ti)
    return out


# --------------------------------------------------------------------------- #
# verify pass
# --------------------------------------------------------------------------- #
def auroc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def verify(df, rng):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = {}
    docs = sorted(df["doc_id"].unique().to_list())
    rng.shuffle(docs)
    cut = int(0.7 * len(docs))
    train_docs, test_docs = set(docs[:cut]), set(docs[cut:])
    tr = df.filter(pl.col("doc_id").is_in(train_docs))
    te = df.filter(pl.col("doc_id").is_in(test_docs))

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                          max_features=300_000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr["claim"].to_list())
    Xte = vec.transform(te["claim"].to_list())
    # liblinear at a tight tolerance, NOT the default lbfgs: minimal pairs make the
    # loss gradient at w = 0 cancel almost exactly, so lbfgs meets its default
    # convergence test on the first iteration and reports a degenerate all-zero fit
    # (every pair tied, every family exactly 0.5000) - an under-powered probe that
    # would pass any corpus.
    clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
    clf.fit(Xtr, tr["label"].to_list())
    ste = clf.decision_function(Xte)
    probe = auroc(te["label"].to_list(), ste)
    out["claim_only_tfidf_auroc"] = {"value": round(probe, 4), "bar": "< 0.55",
                                     "pass": bool(probe < 0.55),
                                     "train_docs": len(train_docs), "test_docs": len(test_docs),
                                     "test_rows": len(te)}

    # within-pair claim-only accuracy, per negative family (held-out documents)
    scored = te.select(["pair_id", "label", "neg_family"]).with_columns(
        pl.Series("score", ste))
    fam_acc, worst = {}, 0.0
    for fam, sub in scored.group_by("neg_family"):
        fam = fam[0]
        piv = sub.pivot(on="label", index="pair_id", values="score",
                        aggregate_function="first").drop_nulls()
        if len(piv) == 0:
            continue
        pos, neg = piv["1.0"].to_numpy(), piv["0.0"].to_numpy()
        acc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
        fam_acc[fam] = {"acc": round(acc, 4), "pairs": len(piv)}
        worst = max(worst, acc)
    out["within_pair_claim_only_accuracy"] = {
        "per_family": fam_acc, "worst": round(worst, 4), "bar": "< 0.60",
        "pass": bool(worst < 0.60)}

    # trailing-zero AUROC, per derivation type and pooled
    tz = df.with_columns(
        pl.col("asserted_value").map_elements(trailing_zeros, return_dtype=pl.Int64).alias("tz"))
    per_type, worst_dev = {}, 0.0
    for dt, sub in tz.filter(pl.col("block") == "core").group_by("dtype"):
        dt = dt[0]
        a = auroc(sub["label"].to_list(), sub["tz"].to_list())
        per_type[dt] = round(a, 4)
        worst_dev = max(worst_dev, abs(a - 0.5))
    pooled_tz = auroc(tz["label"].to_list(), tz["tz"].to_list())
    out["trailing_zero_auroc"] = {"pooled": round(pooled_tz, 4), "per_type": per_type,
                                  "bar": "in [0.45, 0.55]",
                                  "pass": bool(worst_dev <= 0.05 and abs(pooled_tz - 0.5) <= 0.05)}

    # arithmetic re-derivation on a 1,000-row sample of core positives
    core_pos = df.filter((pl.col("block") == "core") & (pl.col("label") == 1.0))
    samp = core_pos.sample(n=min(1000, len(core_pos)), seed=SEED)
    errs = []
    for r in samp.iter_rows(named=True):
        a, b, dt = r["operand_a"], r["operand_b"], r["dtype"]
        want = round_at(a, r["round_step"]) if dt == "rounding" else derive(dt, a, b)
        if want is None or fmt(want) != r["asserted_value"]:
            errs.append({"pair_id": r["pair_id"], "dtype": dt,
                         "expected": None if want is None else fmt(want),
                         "got": r["asserted_value"]})
    out["arithmetic_rederivation"] = {"sampled": len(samp), "errors": len(errs),
                                      "bar": "0 errors", "pass": len(errs) == 0,
                                      "examples": errs[:5]}

    # P(label 0 | asserted value ABSENT from the evidence)
    absent = [not (canon_set(v) & present_numbers(c)) for v, c in
              zip(df["asserted_value"].to_list(), df["chunk"].to_list())]
    lab = np.asarray(df["label"].to_list(), dtype=float)
    ab = np.asarray(absent, dtype=bool)
    p0 = float((lab[ab] == 0).mean()) if ab.any() else float("nan")
    out["p_label0_given_absent"] = {"value": round(p0, 5), "absent_rows": int(ab.sum()),
                                    "bar": "~= 0.5 (+/- 0.01)",
                                    "pass": bool(abs(p0 - 0.5) <= 0.01)}

    # report-only channels
    out["report_only"] = {
        "claim_char_length_auroc": round(
            auroc(df["label"].to_list(), [len(c) for c in df["claim"].to_list()]), 4),
        "digit_count_auroc": round(
            auroc(df["label"].to_list(), [digits(v) for v in df["asserted_value"].to_list()]), 4),
        "decimal_presence_auroc": round(
            auroc(df["label"].to_list(),
                  [1.0 if "." in v else 0.0 for v in df["asserted_value"].to_list()]), 4),
    }
    out["all_bars_pass"] = all(out[k]["pass"] for k in
                               ("claim_only_tfidf_auroc", "within_pair_claim_only_accuracy",
                                "trailing_zero_auroc", "arithmetic_rederivation",
                                "p_label0_given_absent"))
    return out


# --------------------------------------------------------------------------- #
def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    excluded_ids, prints, eval_rows, unmatched = evalset_documents()
    print(f"eval set: {eval_rows} rows -> {len(excluded_ids)} v2 doc_ids, "
          f"{len(prints)} content fingerprints ({unmatched} unmatched)", flush=True)

    print("loading corpora (public: TabFact train, FEVEROUS train)...", flush=True)
    raw = tabfact_tables() + feverous_tables()
    drop_idx = excluded_tables(raw, prints)
    matched_prints = len(drop_idx)
    print(f"  {len(raw)} candidate tables; {matched_prints} carry eval-set content",
          flush=True)
    tables, dropped_eval = [], 0
    for ti, t in enumerate(raw):
        if ti in drop_idx or t["doc_id"] in excluded_ids:
            dropped_eval += 1
            continue
        lab = label_column(t["hdr"], t["body"])
        if lab is None:
            continue
        if not numeric_columns(t["hdr"], t["body"], lab):
            continue
        t["lab_ci"] = lab
        tables.append(t)
    print(f"  {dropped_eval} dropped as eval-set documents; {len(tables)} admitted",
          flush=True)

    forms = list(FORM_WEIGHTS)
    wts = np.array([FORM_WEIGHTS[f] for f in forms], dtype=float)
    wts /= wts.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=wts)):
        t["form"] = forms[int(k)]

    tq, fam_q, sub_q, core_pairs, sub_pairs, src_core, src_rel = build_quotas()
    print(f"  core pairs {core_pairs}, sub-block pairs {sub_pairs}", flush=True)

    dlen_cap = int(DIGIT_LEN_CAP * core_pairs)
    dlen_count = collections.Counter()
    gap_cap = int(sub_q["compare"] / 3) + 1
    gap_count = collections.Counter({"lt10pct": 0, "10to100pct": 0, "gt100pct": 0})
    dir_count = collections.Counter()

    fam_left = {t: dict(f) for t, f in fam_q.items()}
    sub_left = dict(sub_q)
    per_doc = collections.Counter()
    rows, pair_id = [], 0
    rejections = 0

    def attempt(tab):
        nonlocal pair_id, rejections
        need_core = [t for t, f in fam_left.items() if sum(f.values()) > 0]
        need_sub = [a for a, n in sub_left.items() if n > 0]
        if not need_core and not need_sub:
            return None
        made = None
        if need_core and (not need_sub or rng.random() < 1 - SUB_FRAC):
            rng.shuffle(need_core)
            for dtype in need_core:
                fams = [f for f, n in fam_left[dtype].items() if n > 0]
                if not fams:
                    continue
                f = fams[rng.randrange(len(fams))]
                r = try_core(tab, dtype, f, rng, dlen_count, dlen_cap, dir_count)
                if r is None:
                    rejections += 1
                    continue
                fam_left[dtype][f] -= 1
                dlen_count[r["result_digits"]] += 1
                made = ("core", r)
                break
        elif need_sub:
            rng.shuffle(need_sub)
            for arm in need_sub:
                r = try_rel(tab, arm, rng, gap_count, gap_cap)
                if r is not None:
                    sub_left[arm] -= 1
                    made = ("rel", r)
                    break
        if made is None:
            return None
        kind, r = made
        tag = TAG_CORE if kind == "core" else TAG_REL
        base = {k: v for k, v in r.items() if k not in ("claim_pos", "claim_neg", "chunk")}
        for lab, cl, val in ((1.0, r["claim_pos"], r["v_pos"]),
                             (0.0, r["claim_neg"], r["v_neg"])):
            rows.append({"claim": cl, "chunk": r["chunk"], "label": lab, "tag": tag,
                         "pair_id": pair_id, "block": kind, "asserted_value": val, **base})
        pair_id += 1
        return kind

    # Surface parity makes some family/type combinations genuinely short of
    # supply; the unfillable remainder REFLOWS to a sibling family inside the
    # same type and every move is recorded.
    REFLOW = {"N1_oom:misbind": ["N1_arbitrary", "N1_near_miss"],
              "N1_arbitrary": ["N1_near_miss", "N1_oom:misbind"],
              "N1_near_miss": ["N1_arbitrary", "N1_oom:misbind"],
              "N2_operator": ["N1_near_miss", "N1_arbitrary"],
              "N7_numeral": ["N1_arbitrary", "N1_near_miss"],
              "N_round:wrong_direction": ["N_round:wrong_multiple"],
              "N_round:wrong_multiple": ["N_round:wrong_direction"]}
    reflowed = collections.Counter()

    def reflow():
        moved = 0
        for t, fams in fam_left.items():
            for f, n in list(fams.items()):
                if n <= 0:
                    continue
                for dest in REFLOW.get(f, []):
                    if dest in fams and dest != f:
                        fams[dest] += n
                        fams[f] = 0
                        reflowed[f"{t}:{f}->{dest}"] += n
                        moved += n
                        break
        return moved

    order = list(range(len(tables)))
    left = None
    for rnd in range(4):
        for cap in DOC_CAP_LADDER:
            rng.shuffle(order)
            for oi in order:
                while per_doc[oi] < cap:
                    if attempt(tables[oi]) is None:
                        break
                    per_doc[oi] += 1
            left = sum(sum(f.values()) for f in fam_left.values()) + sum(sub_left.values())
            print(f"  round={rnd} cap={cap}: pairs={pair_id} remaining_quota={left} "
                  f"docs_used={len(per_doc)}", flush=True)
            if left == 0:
                break
        if left == 0 or not reflow():
            break
        print(f"  reflowed {sum(reflowed.values())} unfillable family slots", flush=True)

    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("label").cast(pl.Float32))
    lead = ["pair_id", "claim", "chunk", "label", "neg_family", "doc_id", "source", "tag"]
    df = df.select(lead + [c for c in df.columns if c not in lead])
    df.write_parquet(OUT)
    print(f"wrote {OUT}  rows={len(df)} pairs={pair_id}", flush=True)

    # ---------------- disjointness proof -----------------------------------
    built_docs = set(df["doc_id"].unique().to_list())
    raw_index = {t["doc_id"]: ti for ti, t in enumerate(raw)}
    shared_ids = sorted(built_docs & excluded_ids)
    shared_content = sorted(d for d in built_docs if raw_index[d] in drop_idx)
    shared = sorted(set(shared_ids) | set(shared_content))
    # independent of the exclusion logic: no built evidence chunk may be an eval
    # chunk, and no built claim may be an eval claim
    ev = pl.read_parquet(EVALSET, columns=["claim", "chunk"])
    shared_chunks = len(set(df["chunk"].to_list()) & set(ev["chunk"].to_list()))
    shared_claims = len(set(df["claim"].to_list()) & set(ev["claim"].to_list()))
    print(f"disjointness: built docs {len(built_docs)}, eval documents "
          f"{matched_prints} (content) / {len(excluded_ids)} (v2 ids), "
          f"shared docs {len(shared)}, shared chunks {shared_chunks}, "
          f"shared claims {shared_claims}", flush=True)

    print("verify pass...", flush=True)
    v = verify(df, random.Random(SEED))
    print(json.dumps(v, indent=1), flush=True)

    per_doc_counts = df.filter(pl.col("label") == 1.0).group_by("doc_id").len()["len"]
    fam_shares = {f: n for f, n in
                  df.filter(pl.col("label") == 0.0).group_by("neg_family").len().iter_rows()}
    n_neg = sum(fam_shares.values())

    manifest = {
        "experiment": "R17-H144 Stage B - bare-claim derivation pair corpus",
        "generator": "v2 lane reconstruction (bare claims, no traces); see module docstring",
        "seed": SEED,
        "rows": len(df), "pairs": pair_id,
        "core_pairs_target": core_pairs, "sub_pairs_target": sub_pairs,
        "label_counts": {str(k): v for k, v in
                         df.group_by("label").len().iter_rows()},
        "block_rows": {str(k): v for k, v in df.group_by("block").len().iter_rows()},
        "tag_rows": {str(k): v for k, v in df.group_by("tag").len().iter_rows()},
        "claim_form_rows": {str(k): v for k, v in
                            df.group_by("claim_form").len().iter_rows()},
        "serial_form_rows": {str(k): v for k, v in
                             df.group_by("serial_form").len().iter_rows()},
        "source_rows": {str(k): v for k, v in df.group_by("source").len().iter_rows()},
        "negative_family_counts": fam_shares,
        "negative_family_shares": {f: round(n / n_neg, 5) for f, n in fam_shares.items()},
        "v2_reference_family_shares": {
            f"{d}:{f}": round(n / sum(src_core.values()), 5)
            for (d, f), n in sorted(src_core.items())},
        "v2_reference_relational_rows": src_rel,
        "type_quota": tq, "family_quota": fam_q, "sub_quota": sub_q,
        "family_unfilled": {t: {f: n for f, n in d.items() if n}
                            for t, d in fam_left.items()},
        "sub_unfilled": {a: n for a, n in sub_left.items() if n},
        "family_slots_reflowed": dict(reflowed),
        "tuples_rejected_at_construction": rejections,
        "diversity": {
            "documents": len(built_docs),
            "pairs_per_document_mean": round(float(per_doc_counts.mean()), 4),
            "pairs_per_document_max": int(per_doc_counts.max()),
            "pairs_per_document_p50": int(per_doc_counts.median()),
            "distinct_chunks": int(df["chunk"].n_unique()),
            "distinct_claims": int(df["claim"].n_unique()),
            "result_digit_length_counts": {str(k): v for k, v in sorted(dlen_count.items())},
            "compare_gap_strata": dict(gap_count),
        },
        "disjointness": {
            "method": ("content fingerprints (the canonical numerals of each eval "
                       "chunk must not sit inside a built table), because the v2 "
                       "lane's FEVEROUS doc_ids are not stable across rebuilds; "
                       "the v2 doc_id set is excluded as well"),
            "eval_rows": eval_rows,
            "eval_rows_unmatched_to_a_v2_document": unmatched,
            "eval_v2_doc_ids": len(excluded_ids),
            "eval_content_fingerprints": len(prints),
            "corpus_tables_matching_eval_content": matched_prints,
            "corpus_documents_dropped_as_eval": dropped_eval,
            "built_documents": len(built_docs),
            "shared_doc_ids": len(shared),
            "shared_by_v2_id": len(shared_ids),
            "shared_by_content": len(shared_content),
            "shared_examples": shared[:10],
            "shared_evidence_chunks_with_evalset": shared_chunks,
            "shared_claim_strings_with_evalset": shared_claims,
        },
        "negative_direction_balance": {
            str(k): v for k, v in sorted(dir_count.items(), key=lambda kv: str(kv[0]))},
        "verify": v,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MANIFEST}", flush=True)
    print(json.dumps({"rows": len(df), "pairs": pair_id,
                      "shared_doc_ids": len(shared),
                      "all_bars_pass": v["all_bars_pass"]}, indent=1), flush=True)
    if not v["all_bars_pass"] or shared or shared_chunks:
        raise SystemExit("ABORT: a verify bar failed or the document sets are not disjoint")


if __name__ == "__main__":
    main()
