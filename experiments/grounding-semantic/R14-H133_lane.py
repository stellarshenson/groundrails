"""R14-H133 (R14-A4) DERIVATION-PARITY LANE - build only, CPU, no GPU.

Builds `R14-H133_lane.parquet`: ~50,000 rows = ~25,000 minimal pairs over public
TabFact-train and FEVEROUS-train tables.

  DERIVATION CORE (85% of rows, R15-B1 schedule)
    positive - a CORRECTLY DERIVED value, ABSENT from the serialized evidence
    negative - the byte-identical claim with an operand-misbound / operator-swapped /
               mis-scaled / numeral-corrupted value, also ABSENT
    one negative per positive, so P(label 0 | absent) = 0.5000 by construction and
    the absent-number shortcut carries zero information inside the lane

  RELATIONAL SUB-BLOCK (15% of rows, R15-B4 / H138)
    bind_col - value bound to the wrong COLUMN of the right row (both present)
    compare  - ordering claim against its reversal (both operands printed)
    bind_row - 20% of the sub-block, the registered non-regression arm

DATA-ONLY (A4 binding amendment i): no hinge, no auxiliary head. The trainer and
the objective stay the clean recipe's; this script emits a parquet and nothing else.

Independent justification, carried verbatim per A4 amendment (ii):

    A grounding library that flags every arithmetically derivable quantity as
    unsupported false-alarms on the commonest shape of numeric RAG answer - this
    is a product requirement that stands with the arena deleted.

Public data only. Seeded (SEED = 1133), reproducible, Polars, no network.

Run:  uv run python experiments/grounding-semantic/R14-H133_lane.py
"""

import collections
import json
import math
import pathlib
import random
import re
import zipfile
import io

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
FEVEROUS_SRC = ROOT / "tmp" / "R14_H133_feverous.parquet"
OUT = HERE / "R14-H133_lane.parquet"
MANIFEST = HERE / "R14-H133_lane_manifest.json"

SEED = 1133
CHUNK_MAX = 1500          # the serving unit; the trainer truncates to this anyway
BODY_ROWS = 6             # R15-B1 clause 4 - a body-ROW budget, not a character prefix
N_PAIRS = 25_000          # 50,000 rows
SUB_FRAC = 0.15           # R15-B4 sub-block share of lane ROWS
BARE_FRAC = 0.60          # R15-B1 clause 2 - at least 50% bare-assertion form
DIGIT_LEN_MIN, DIGIT_LEN_MAX = 2, 7   # R15-B1 clause 6
DIGIT_LEN_CAP = 0.33      # per-result-digit-length cap (bar is 0.35; build to 0.33)

TAG_CORE = "quant_deriv"
TAG_REL = "quant_relational"

# R15-B1 schedule (shares as written; renormalised because they total 99)
TYPE_SHARES = {
    "difference": 16, "ratio": 16, "pct_change": 16, "sum": 16,
    "rounding": 12, "mean": 10, "scale_unit": 8, "product": 5,
}
# v3 negative families (author ruling). Every defect must be GROUNDABLE - found
# by matching the trace against the evidence or against the claim, never by
# doing arithmetic.
NEG_FAMILIES = {"a_trace_evidence": 0.70, "b_operation_word": 0.22, "c_conclusion": 0.08}
# `wrong_scale` is DROPPED in v3: a x10 operand cannot carry the true operand's
# trailing-zero count or decimal placement, so every surviving instance is
# separable on the claim text alone (measured within-pair 0.651)
A_KINDS = {"misbound_row": 0.70, "misbound_col": 0.30}

# characters per token, measured per serialization form in R15_gate_L1_serialtok.json;
# the claim carries prose and numerals and runs richer
CPT_FORM = {"pipe": 2.24, "markdown": 2.25, "keyvalue": 2.51,
            "json_records": 2.59, "narrative": 2.94, "row_prose": 2.92}
CPT_CLAIM = 3.4
TOKEN_TARGET = 500        # of MAX_LEN 512 - traces are never trimmed, evidence is

# R15-B1 clause 4 form weights, re-derived on token cost by L1
FORM_WEIGHTS = {"row_prose": 30, "narrative": 30, "pipe": 15,
                "keyvalue": 10, "markdown": 10, "json_records": 5}

NUM_CELL = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")
NUM_FREE = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.,])")
# The banked detector's trailing (?![\d.,]) makes it blind to a numeral followed
# by a full stop - i.e. to EVERY value in a prose serialization. The lane's whole
# absence rule runs on the evidence side, so evidence presence is detected with
# the punctuation-tolerant form below and the banked form is kept for reporting.
NUM_EVID = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d,])")
WIKILINK = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]")
WIKIBARE = re.compile(r"\[\[([^|\]]+)\]\]")


# --------------------------------------------------------------------------- #
# numbers - byte-identical to R14_H133_probe / R15_P1_typeprobe / R15_gate_common
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


def canon_set(s):
    """P3 / L2 absence detector - canonical numeral forms present in a string."""
    p = set()
    for m in NUM_FREE.findall(s or ""):
        v = m.replace(",", "")
        p.add(v)
        try:
            f = float(v)
        except (ValueError, OverflowError):
            continue
        if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
            continue
        p.add(str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.2f}".rstrip("0").rstrip("."))
    return p


def present_numbers(s):
    """Canonical numeral forms readable in an EVIDENCE string, tolerant of the
    trailing punctuation that prose serializations put after every value."""
    p = set()
    for m in NUM_EVID.findall(s or ""):
        v = m.replace(",", "")
        p.add(v)
        try:
            f = float(v)
        except (ValueError, OverflowError):
            continue
        if f != f or f in (float("inf"), float("-inf")) or abs(f) > 1e15:
            continue
        p.add(str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.2f}".rstrip("0").rstrip("."))
    return p


def digits(s):
    return sum(ch.isdigit() for ch in s)


WS = re.compile(r"\s+")


def clean(x):
    """One line, collapsed whitespace - FEVEROUS cells carry embedded prose."""
    return WS.sub(" ", (x or "")).strip()


def good_header(h):
    """A usable column name: short, non-empty, not itself a number. FEVEROUS's
    modal-width row is sometimes a data row or a prose sentence."""
    h = clean(h)
    return bool(h) and len(h) <= 40 and as_num(h) is None


def trailing_zeros(s):
    """Trailing '0' CHARACTERS of the rendered numeral - the surface feature a
    character-ngram classifier reads, not a mathematical property."""
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
    """Within-pair digit-surface parity (v2, defect 2).

    A claim-only classifier must find nothing to separate the pair on. The
    numeral is the only thing that differs between a positive and its negative,
    so it must match on every cheap surface statistic a character n-gram model
    reads: LEADING digit, digit count (R15-B1 clause 3), trailing-zero count,
    decimal presence and sign phrasing.

    The leading digit is the load-bearing one. Positives are natural derived
    values and follow Benford (leading 1 at 0.41); v2's first cut let the
    negatives flatten that to 0.29, which alone carried the residual leak."""
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
# serialization - one form per DOCUMENT, drawn independently of everything else
# --------------------------------------------------------------------------- #
def _attrs(hdr, row, lab_ci):
    """(header, cell) for every column except the row-label column."""
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
    """`Its {h} is {c}.` keeps the form short, but a column carried that way has
    no explicit "{column} of {row}" binding phrase. Every column the claim
    references is emitted in the explicit form (v2, review caveat)."""
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
# corpora - TabFact train + FEVEROUS train. PUBLIC ONLY. TabFact test/validation
# is RESERVED as the H133 re-read surface and is never touched here.
# --------------------------------------------------------------------------- #
def tabfact_tables():
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    name = next(x for x in z.namelist() if x.endswith("__train.parquet"))
    d = pl.read_parquet(io.BytesIO(z.read(name))).unique(subset=["table_text"], keep="first")
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
    d = pl.read_parquet(FEVEROUS_SRC).unique(subset=["evidence"], keep="first")
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
    """Index of the row-label column - the leftmost column whose cells are
    non-empty, non-numeric and mostly distinct. Column 0 wherever it qualifies
    (P1's rule); otherwise the next column that does, because TabFact's leftmost
    column is often a rank or a year."""
    for ci in range(min(3, len(hdr))):
        if not good_header(hdr[ci]):
            continue
        labs = [r[ci].strip() for r in body]
        # `42.` does not match NUM_CELL but is still a rank, not a row label -
        # it produced "The Passengers of 42. exceeds that of 54." in v2's draft
        if any((not x) or as_num(x.rstrip(". ")) is not None for x in labs):
            continue
        # a row label carrying a semicolon or a sentence break makes the trace
        # unreadable ("The Year of GSIST opening; Admission of 54 students. is 1995")
        if any(";" in x or x.endswith(".") or len(x) > 60 for x in labs):
            continue
        if len(set(labs)) < max(3, int(0.8 * len(labs))):
            continue
        return ci
    return None


def numeric_columns(hdr, body, lab_ci):
    """(col_index, [(row_index, value)]) for every usable numeric column.

    Usable = >= 4 numeric rows and >= 4 distinct values, in a column that is not
    the label column."""
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
# the eight derivation types - templates byte-identical to R15_P1_typeprobe.make
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
    """v2 (defect 1): the rounding place is STATED. `approximately` left every
    coarser rounding defensible, so 100% of the wrong-place negatives were
    correct English labelled 0."""
    return f"Rounded to the nearest {place}, the {col} of {ka} is {{}}."


def rounding_steps(a):
    """Places at which rounding `a` is meaningful and changes the value."""
    out = []
    for step, name in ROUND_PLACES:
        if abs(a) < 2 * step:
            continue
        if abs(round(a / step) * step - a) < 1e-9:
            continue          # already exact - the positive would be a table cell
        out.append((step, name))
    return out


def derive(dtype, a, b):
    """The CORRECT value of `dtype` over operands (a, b). None when undefined."""
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
    if dtype == "rounding":
        raise ValueError("rounding derives against an explicit place - see round_at")
    raise ValueError(dtype)


def round_at(a, step):
    return round(a / step) * step


BIND_FORMS_2 = [
    "The {col} of {ka} is {va} and the {col} of {kb} is {vb}",
    "According to the table, {ka} has a {col} of {va} and {kb} has a {col} of {vb}",
    "The table gives {ka} a {col} of {va} and {kb} a {col} of {vb}",
]
BIND_FORMS_1 = [
    "The {col} of {ka} is {va}",
    "According to the table, {ka} has a {col} of {va}",
    "The table gives {ka} a {col} of {va}",
]
CALC_FORMS = ["{expr} = {res}", "computing {expr} gives {res}"]

SINGLE = ("scale_unit", "rounding")


def calc_expr(dtype, va, vb, place=None):
    """The arithmetic, written out. The reader checks it by reading, not by
    computing: the operands are quoted from the evidence and the conclusion is
    quoted from this line."""
    if dtype == "sum":
        return f"{va} + {vb}"
    if dtype == "difference":
        return f"{va} - {vb}"
    if dtype == "mean":
        return f"({va} + {vb}) / 2"
    if dtype == "ratio":
        return f"{va} / {vb}"
    if dtype == "pct_change":
        return f"({vb} - {va}) / {va} * 100"
    if dtype == "product":
        return f"{va} * {vb}"
    if dtype == "scale_unit":
        return f"{va} * 1000"
    if dtype == "rounding":
        return f"{va} rounded to the nearest {place}"
    raise ValueError(dtype)


def conclusion(dtype, col, ka, kb, res, place=None):
    tpl = rounding_template(col, ka, place) if dtype == "rounding" \
        else template(dtype, col, ka, kb)
    t = tpl.format(res)
    return t[0].lower() + t[1:]


def assemble(bind, calc, concl):
    return f"{bind}; {calc}, so {concl}"


def trace_of(dtype, col, ka, kb, va, vb, res, bind_i, calc_i, place=None):
    """(bind clause, calc clause) for one derivation, at the chosen phrasing."""
    forms = BIND_FORMS_1 if dtype in SINGLE else BIND_FORMS_2
    bind = forms[bind_i].format(col=col, ka=ka, kb=kb, va=va, vb=vb)
    calc = CALC_FORMS[calc_i].format(expr=calc_expr(dtype, va, vb, place), res=res)
    return bind, calc


def est_tokens(form, chunk, claim):
    return len(chunk) / CPT_FORM[form] + len(claim) / CPT_CLAIM + 4


# --------------------------------------------------------------------------- #
# quotas
# --------------------------------------------------------------------------- #
def build_quotas():
    core_pairs = int(round(N_PAIRS * (1 - SUB_FRAC)))
    sub_pairs = N_PAIRS - core_pairs
    tot = sum(TYPE_SHARES.values())
    tq = {t: int(round(core_pairs * s / tot)) for t, s in TYPE_SHARES.items()}
    tq["sum"] += core_pairs - sum(tq.values())

    fam = {}
    for t, n in tq.items():
        f = {}
        n_a = int(round(NEG_FAMILIES["a_trace_evidence"] * n))
        n_c = int(round(NEG_FAMILIES["c_conclusion"] * n))
        n_b = n - n_a - n_c
        acc = 0
        kinds = list(A_KINDS)
        for k in kinds[:-1]:
            f[f"a:{k}"] = int(round(n_a * A_KINDS[k]))
            acc += f[f"a:{k}"]
        f[f"a:{kinds[-1]}"] = n_a - acc
        f["b:operation_word"] = n_b
        f["c:conclusion"] = n_c
        fam[t] = f
    # `scale_unit` and `rounding` can only swap conclusion words with each other,
    # so their (b) quotas must match or the pool cannot balance; the excess goes
    # back to family (a) inside its own type.
    lo = min(fam[t]["b:operation_word"] for t in SINGLE)
    for t in SINGLE:
        extra = fam[t]["b:operation_word"] - lo
        fam[t]["b:operation_word"] = lo
        fam[t]["a:misbound_row"] += extra
    sub = {"bind_col": int(round(sub_pairs * 0.40)),
           "compare": int(round(sub_pairs * 0.40))}
    sub["bind_row"] = sub_pairs - sub["bind_col"] - sub["compare"]
    return tq, fam, sub, core_pairs, sub_pairs


# --------------------------------------------------------------------------- #
# core construction
# --------------------------------------------------------------------------- #
def _spans_decade(vals):
    v = [abs(x) for _, x in vals if abs(x) > 1e-9]
    return bool(v) and max(v) / min(v) >= 10.0


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


def want_up(dir_count, key):
    """The minority error direction for this (family, derivation type).

    Keyed per TYPE, not per family: balancing `N2_operator` as a whole left the
    `difference` template almost always paired with a larger negative and the
    `sum` template with a smaller one, so the template's own words told a
    claim-only classifier which direction to expect (within-pair 0.627).

    Every family's negative was systematically LARGER than its positive (N2's
    swap adds, rounding's surviving multiples sit on one side), so whatever digit
    position the pair first differs at carried an ordinal cue - which is what
    held claim-only AUROC at 0.56 after leading-digit parity. Forcing the
    minority direction on every draw balances it to 50/50 per family."""
    return dir_count[(key, "up")] <= dir_count[(key, "down")]


def try_core(tab, dtype, want_family, rng, dlen_count, dlen_cap, dir_count, attempts=5):
    """One core minimal pair over one table, or None. Retried over independent
    column / operand-row draws, because most rejections are draw-specific."""
    loss = None
    for _ in range(attempts):
        r = _try_core_once(tab, dtype, want_family, rng, dlen_count, dlen_cap, dir_count)
        if r == "operand_row_loss":
            loss = r
            continue
        if r is not None:
            return r
    return loss


def _fit_evidence(tab, lab_ci, ci, keep, protect, claim_len, rng):
    """Trim EVIDENCE rows, never the trace, until the pair fits the token budget.

    Dropping a row can only remove numerals, so the absence rule survives; the
    operand rows and the misbind source row are protected, so grounding does."""
    rows = sorted(keep)
    while True:
        body = [tab["body"][r] for r in rows]
        chunk = serialize(tab["form"], tab["caption"], tab["hdr"], body, lab_ci, (ci,))
        if est_tokens(tab["form"], chunk, claim_len) <= TOKEN_TARGET:
            return rows, chunk
        droppable = [r for r in rows if r not in protect]
        if not droppable:
            return rows, chunk
        rows.remove(droppable[-1])


def _try_core_once(tab, dtype, want_family, rng, dlen_count, dlen_cap, dir_count):
    """One core minimal pair. v3: the claim carries a programmatic reasoning
    trace, and the negative's defect is always groundable - a trace that
    misquotes the evidence, a conclusion word that contradicts the trace, or a
    conclusion number that contradicts the trace. None of the three asks the
    model to compute anything."""
    lab_ci = tab["lab_ci"]
    cols = numeric_columns(tab["hdr"], tab["body"], lab_ci)
    if not cols:
        return None
    ci, vals = cols[rng.randrange(len(cols))]
    if len(vals) < 4:
        return None
    col = (tab["hdr"][ci] or f"column {ci}").strip() or f"column {ci}"
    single = dtype in SINGLE

    pool = rng.sample(vals, min(len(vals), BODY_ROWS))
    keep0 = {ri for ri, _ in pool}
    others = [r for r in range(len(tab["body"])) if r not in keep0]
    rng.shuffle(others)
    keep0 |= set(others[: BODY_ROWS - len(keep0)])

    bind_i = rng.randrange(len(BIND_FORMS_1))
    calc_i = rng.randrange(len(CALC_FORMS))

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
        # The trace QUOTES its operands, so a quoted operand that is a rounded
        # rendering makes the written arithmetic wrong on its face: `2.04 * 1000
        # = 2037` for a cell of 2.037. Only exactly-renderable operands qualify.
        if abs(float(fmt(vi)) - vi) > 1e-9 or (not single and abs(float(fmt(vj)) - vj) > 1e-9):
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

            va, vb = fmt(vi), fmt(vj)
            bind_p, calc_p = trace_of(dtype, col, ka, kb, va, vb, v_pos,
                                      bind_i, calc_i, place)
            claim_pos = assemble(bind_p, calc_p,
                                 conclusion(dtype, col, ka, kb, v_pos, place))

            protect = {ri_i, ri_j}
            fam = want_family
            detail = fam
            pending = []
            cited_true = cited_wrong = None
            neg_stated_op = ""

            # ---------------- (a) trace-evidence mismatch -------------------
            if fam.startswith("a:"):
                kind = fam.split(":", 1)[1]
                slot = 0 if single else rng.randrange(2)
                true_v = vi if slot == 0 else vj
                row_of = ri_i if slot == 0 else ri_j
                cands = []
                if kind == "misbound_row":
                    for (rb, vv) in vals:
                        if rb != row_of:
                            cands.append((vv, rb))
                else:   # misbound_col
                    for c2, vals2 in cols:
                        if c2 == ci:
                            continue
                        m2 = dict(vals2)
                        if row_of in m2:
                            cands.append((m2[row_of], row_of))
                rng.shuffle(cands)
                up = want_up(dir_count, (detail, dtype))
                got = None
                for cv, src in cands:
                    if abs(cv - true_v) < 1e-9 or abs(float(fmt(cv)) - cv) > 1e-9:
                        continue
                    cs = fmt(cv)
                    if not surface_ok(fmt(true_v), cs):
                        continue
                    ni, nj = (cv, vj) if slot == 0 else (vi, cv)
                    if dtype in ("difference", "ratio") and (nj <= 0 or ni < nj):
                        continue
                    w = round_at(ni, step) if dtype == "rounding" else derive(dtype, ni, nj)
                    if w is None or abs(w - correct) < 1e-9:
                        continue
                    if (w > correct) != up:
                        continue
                    ws = fmt(w)
                    if not (DIGIT_LEN_MIN <= digits(ws) <= DIGIT_LEN_MAX):
                        continue
                    if not surface_ok(v_pos, ws):
                        continue
                    got = (cs, ws, src, fmt(true_v))
                    break
                if got is None:
                    continue
                cited_wrong, v_neg, src, cited_true = got
                if src is not None:
                    protect.add(src)
                nva, nvb = (cited_wrong, vb) if slot == 0 else (va, cited_wrong)
                bind_n, calc_n = trace_of(dtype, col, ka, kb, nva, nvb, v_neg,
                                          bind_i, calc_i, place)
                claim_neg = assemble(bind_n, calc_n,
                                     conclusion(dtype, col, ka, kb, v_neg, place))
                pending += [((detail, dtype), "up" if up else "down")]

            # ---------------- (b) operation-word mismatch -------------------
            elif fam == "b:operation_word":
                # The wrong word is chosen to keep its marginal distribution
                # equal to the positives': drawn uniformly it read within-pair
                # 0.658, because a linear bag-of-ngrams needs only the word's
                # base rate, never its agreement with the trace.
                pool_ops = list(SINGLE) if single else \
                    ["sum", "difference", "mean", "ratio", "pct_change", "product"]
                alts = [o for o in pool_ops if o != dtype]
                # NOTE: `rounding` is deliberately NOT offered its own word with
                # a wrong place. The trace already says "rounded to the nearest
                # {place}", so a positive then carries that place word TWICE and a
                # negative once - a term-frequency tell that sublinear TF-IDF reads
                # directly. The single-arity pool is balanced by equalising the two
                # types' (b) quotas in build_quotas instead.
                rng.shuffle(alts)
                alts.sort(key=lambda o: dir_count[("word", o, 0)] - dir_count[("word", o, 1)])
                other = alts[0]
                if other == "rounding":
                    places = [n for _, n in ROUND_PLACES if n != place]
                    places.sort(key=lambda n: dir_count[("place", n, 0)]
                                - dir_count[("place", n, 1)])
                    alt_place = places[0]
                    pending += [("place", alt_place, 0)]
                    if place:
                        pending += [("place", place, 1)]
                    concl_n = conclusion("rounding", col, ka, kb, v_pos, alt_place)
                else:
                    concl_n = conclusion(other, col, ka, kb, v_pos, place or "hundred")
                pending += [("word", other, 0)]
                v_neg = v_pos                      # the numeral is untouched
                neg_stated_op = other
                claim_neg = assemble(bind_p, calc_p, concl_n)
                if claim_neg == claim_pos:
                    continue

            # ---------------- (c) conclusion mismatch -----------------------
            else:
                up = want_up(dir_count, (detail, dtype))
                same_op = []
                for (_, x) in pool:
                    for (_, y) in pool:
                        if x == y:
                            continue
                        x2, y2 = ((max(x, y), min(x, y))
                                  if dtype in ("difference", "ratio") else (x, y))
                        if dtype in ("difference", "ratio") and y2 <= 0:
                            continue
                        w2 = round_at(x2, step) if dtype == "rounding" else derive(dtype, x2, y2)
                        if w2 is not None:
                            same_op.append(w2)
                if not same_op:
                    continue
                rng.shuffle(same_op)
                v_neg = None
                for w in same_op:
                    if abs(w - correct) < 1e-9 or (w > correct) != up:
                        continue
                    ws = fmt(w)
                    if not (DIGIT_LEN_MIN <= digits(ws) <= DIGIT_LEN_MAX):
                        continue
                    if not surface_ok(v_pos, ws):
                        continue
                    v_neg = ws
                    break
                if v_neg is None:
                    continue
                claim_neg = assemble(bind_p, calc_p,
                                     conclusion(dtype, col, ka, kb, v_neg, place))
                pending += [((detail, dtype), "up" if up else "down")]

            # ---------------- evidence, fitted to the token budget ----------
            rows, chunk = _fit_evidence(tab, lab_ci, ci, keep0, protect,
                                        max(claim_pos, claim_neg, key=len), rng)
            present = present_numbers(chunk)
            need = [va] if single else [va, vb]
            if any(not (canon_set(v) & present) for v in need):
                continue
            if canon_set(v_pos) & present:
                continue
            if v_neg != v_pos and (canon_set(v_neg) & present):
                continue

            for k in pending:
                dir_count[k] += 1
            return {
                "claim_pos": claim_pos, "claim_neg": claim_neg, "chunk": chunk,
                "dtype": dtype, "neg_family": detail, "claim_form": "trace",
                "pair_shape": "word_only" if fam == "b:operation_word" else "numeral_only",
                "serial_form": tab["form"], "doc_id": tab["doc_id"],
                "source": tab["source"], "column": col, "key_a": ka, "key_b": kb,
                "operand_a": float(vi), "operand_b": float(vj),
                "round_step": float(step or 0.0), "round_place": place or "",
                "bind_variant": bind_i, "calc_variant": calc_i,
                "neg_cited_value": cited_wrong or "", "neg_true_value": cited_true or "",
                "neg_stated_op": neg_stated_op,
                "v_pos": v_pos, "v_neg": v_neg, "result_digits": digits(v_pos),
            }
    return None


# --------------------------------------------------------------------------- #
# relational sub-block (R15-B4 / H138) - every asserted value PRESENT
# --------------------------------------------------------------------------- #
def try_rel(tab, arm, rng, gap_count, gap_cap, attempts=4):
    for _ in range(attempts):
        r = _try_rel_once(tab, arm, rng, gap_count, gap_cap)
        if r is not None:
            return r
    return None


def _try_rel_once(tab, arm, rng, gap_count, gap_cap):
    """R15-B4 / H138, v3: the binding statement IS the trace. `bind_col` and
    `bind_row` negatives misquote which cell the table gives; `compare`
    negatives swap the two bindings and then reason correctly from the swap, so
    every defect is found by grounding the trace, never by computing."""
    lab_ci = tab["lab_ci"]
    cols = numeric_columns(tab["hdr"], tab["body"], lab_ci)
    if not cols:
        return None

    def finish(keep, protect, ci_bind, claim_pos, claim_neg, **extra):
        rows, chunk = _fit_evidence(tab, lab_ci, ci_bind, set(keep), set(protect),
                                    max(claim_pos, claim_neg, key=len), rng)
        return rows, chunk, claim_pos, claim_neg, extra

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
                if lx == ly or digits(lx) != digits(ly) or not surface_ok(lx, ly):
                    continue
                ka = tab["body"][ri][lab_ci].strip()
                if not ka:
                    continue
                cp = f"The table lists the {colx} of {ka} as {lx}, so the {colx} of {ka} is {lx}."
                cn = f"The table lists the {colx} of {ka} as {ly}, so the {colx} of {ka} is {ly}."
                keep = {ri}
                others = [r for r in range(len(tab["body"])) if r != ri]
                rng.shuffle(others)
                keep |= set(others[: BODY_ROWS - 1])
                rows, chunk, cp, cn, _ = finish(keep, {ri}, ci, cp, cn)
                pres = present_numbers(chunk)
                if not (canon_set(lx) & pres) or not (canon_set(ly) & pres):
                    continue
                return {"claim_pos": cp, "claim_neg": cn, "chunk": chunk,
                        "arm": "bind_col", "pair_shape": "numeral_only",
                        "serial_form": tab["form"], "doc_id": tab["doc_id"],
                        "source": tab["source"], "column": colx, "column_b": coly,
                        "key_a": ka, "key_b": ka, "neg_cited_value": ly,
                        "neg_true_value": lx, "v_pos": lx, "v_neg": ly,
                        "gap_stratum": ""}
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
            if not surface_ok(la, lb):
                continue
            cp = f"The table lists the {col} of {ka} as {la}, so the {col} of {ka} is {la}."
            cn = f"The table lists the {col} of {ka} as {lb}, so the {col} of {ka} is {lb}."
            keep = {ra, rb}
            others = [r for r in range(len(tab["body"])) if r not in keep]
            rng.shuffle(others)
            keep |= set(others[: BODY_ROWS - 2])
            rows, chunk, cp, cn, _ = finish(keep, {ra, rb}, ci, cp, cn)
            pres = present_numbers(chunk)
            if not (canon_set(la) & pres) or not (canon_set(lb) & pres):
                continue
            return {"claim_pos": cp, "claim_neg": cn, "chunk": chunk,
                    "arm": "bind_row", "pair_shape": "numeral_only",
                    "serial_form": tab["form"], "doc_id": tab["doc_id"],
                    "source": tab["source"], "column": col, "column_b": col,
                    "key_a": ka, "key_b": kb, "neg_cited_value": lb,
                    "neg_true_value": la, "v_pos": la, "v_neg": lb,
                    "gap_stratum": ""}
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
        # Bind the two entities in a random order, drawn independently of label.
        # Fixed larger-first order let a claim-only model read the label off the
        # position of the two numerals (within-pair 0.741): in the positive the
        # first bound value always matched the first value of the comparison.
        e1, e2 = (k_hi, k_lo) if rng.random() < 0.5 else (k_lo, k_hi)
        true_of = {k_hi: lh, k_lo: ll}
        swap_of = {k_hi: ll, k_lo: lh}
        head = (f"The {col} of {{e1}} is {{v1}} and the {col} of {{e2}} is {{v2}}; "
                f"{lh} is greater than {ll}, so ")
        cp = head.format(e1=e1, e2=e2, v1=true_of[e1], v2=true_of[e2]) + \
            f"the {col} of {k_hi} is greater than the {col} of {k_lo}."
        cn = head.format(e1=e1, e2=e2, v1=swap_of[e1], v2=swap_of[e2]) + \
            f"the {col} of {k_lo} is greater than the {col} of {k_hi}."
        keep = {ra, rb}
        others = [r for r in range(len(tab["body"])) if r not in keep]
        rng.shuffle(others)
        keep |= set(others[: BODY_ROWS - 2])
        rows, chunk, cp, cn, _ = finish(keep, {ra, rb}, ci, cp, cn)
        pres = present_numbers(chunk)
        if not (canon_set(lh) & pres) or not (canon_set(ll) & pres):
            continue
        gap_count[stratum] += 1
        return {"claim_pos": cp, "claim_neg": cn, "chunk": chunk, "arm": "compare",
                "pair_shape": "binding_swap", "serial_form": tab["form"],
                "doc_id": tab["doc_id"], "source": tab["source"], "column": col,
                "column_b": col, "key_a": k_hi, "key_b": k_lo,
                "neg_cited_value": ll, "neg_true_value": lh,
                "v_pos": lh, "v_neg": ll, "gap_stratum": stratum}
    return None


# --------------------------------------------------------------------------- #
def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    print("loading corpora (public: TabFact train, FEVEROUS train)...", flush=True)
    raw = tabfact_tables() + feverous_tables()
    tables = []
    for t in raw:
        lab = label_column(t["hdr"], t["body"])
        if lab is None:
            continue
        if not numeric_columns(t["hdr"], t["body"], lab):
            continue
        t["lab_ci"] = lab
        tables.append(t)
    print(f"  candidate tables {len(raw)} -> admitting a label column and a "
          f"usable numeric column: {len(tables)}", flush=True)

    # one serialization form per DOCUMENT, drawn before any claim exists
    forms = list(FORM_WEIGHTS)
    wts = np.array([FORM_WEIGHTS[f] for f in forms], dtype=float)
    wts /= wts.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=wts)):
        t["form"] = forms[int(k)]

    tq, fam_q, sub_q, core_pairs, sub_pairs = build_quotas()
    print(f"  core pairs {core_pairs}, sub-block pairs {sub_pairs}", flush=True)

    dlen_cap = int(DIGIT_LEN_CAP * core_pairs)
    dlen_count = collections.Counter()
    gap_cap = int(sub_q["compare"] / 3) + 1
    gap_count = collections.Counter({"lt10pct": 0, "10to100pct": 0, "gt100pct": 0})
    dir_count = collections.Counter()
    # Seed the wrong-word balancer with each word's FINAL positive count, so it
    # aims at the finished marginal from the first draw. Greedy against a
    # running count leaves an end-game bias - a word cannot be its own type's
    # alternative, so the highest-share types finish short (difference realised
    # 0.115 of negatives against 0.162 of positives).
    for _t, _f in fam_q.items():
        dir_count[("word", _t, 1)] = _f.get("b:operation_word", 0)

    fam_left = {t: dict(f) for t, f in fam_q.items()}
    sub_left = dict(sub_q)
    per_doc = collections.Counter()
    rows, pair_id = [], 0
    dropped_operand_rows = 0
    rejections = 0

    def attempt(tab):
        """Emit at most one pair from `tab`. Returns the pair kind or None."""
        nonlocal pair_id, rejections, dropped_operand_rows
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
                if r == "operand_row_loss":
                    dropped_operand_rows += 1
                    rejections += 1
                    continue
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

    # Surface parity (v2) makes some families genuinely short of supply - an
    # order-of-magnitude misbind that also matches the positive's trailing zeros
    # exists on 1.8% of `mean` tables and 2.6% of `sum` tables, measured. Rather
    # than manufacture one, the unfillable remainder REFLOWS to sibling families
    # inside the same type and every move is recorded.
    REFLOW = {"a:misbound_row": ["a:misbound_col", "b:operation_word"],
              "a:misbound_col": ["a:misbound_row", "b:operation_word"],
              "b:operation_word": ["a:misbound_row", "a:misbound_col"],
              "c:conclusion": ["a:misbound_row", "a:misbound_col"]}
    reflowed = collections.Counter()

    def reflow():
        moved = 0
        for t, fams in fam_left.items():
            for f, n in list(fams.items()):
                if n <= 0:
                    continue
                for dest in REFLOW.get(f, []):
                    if dest in fams:
                        fams[dest] += n
                        fams[f] = 0
                        reflowed[f"{t}:{f}->{dest}"] += n
                        moved += n
                        break
        return moved

    order = list(range(len(tables)))
    for rnd in range(4):
        for cap in (2, 3, 4, 5, 6, 8):   # per-table cap; raised only if supply is short
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
    # claim / chunk / label / tag first - the columns R10-H108_lane.lane_train reads
    lead = ["claim", "chunk", "label", "tag"]
    df = df.select(lead + [c for c in df.columns if c not in lead])
    df.write_parquet(OUT)
    print(f"wrote {OUT}  rows={len(df)}", flush=True)

    manifest = {
        "rows": len(df), "pairs": pair_id, "seed": SEED,
        "core_pairs_target": core_pairs, "sub_pairs_target": sub_pairs,
        "type_quota": tq, "family_quota": fam_q, "sub_quota": sub_q,
        "family_unfilled": {t: {f: n for f, n in d.items() if n} for t, d in fam_left.items()},
        "sub_unfilled": {a: n for a, n in sub_left.items() if n},
        "tuples_rejected_at_construction": rejections,
        "tuples_dropped_for_operand_row_loss": dropped_operand_rows,
        "operand_row_loss_note": ("R15-B1 clause 4 is implemented as row SELECTION (the "
            "retained 6 body rows are chosen to carry the operand rows) rather than a "
            "6-row prefix, so the only operand loss left is the 1,500-character serving "
            "cap truncating a retained row; those tuples are dropped and counted here"),
        "result_digit_length_counts": {str(k): v for k, v in sorted(dlen_count.items())},
        "compare_gap_strata": dict(gap_count),
        "negative_direction_balance": {
            str(k): v for k, v in sorted(dir_count.items(), key=lambda kv: str(kv[0]))},
        "family_slots_reflowed": dict(reflowed),
        "reflow_note": ("surface parity (v2) makes some family/type combinations "
                        "unconstructible; the unfillable remainder is moved to a "
                        "sibling family inside the same type and recorded here"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("rows", "pairs", "family_unfilled", "sub_unfilled")}, indent=1))


if __name__ == "__main__":
    main()
