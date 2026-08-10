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
N2_SHARE = 0.225          # wrong-operator share of non-rounding negatives (see manifest)
N7_SHARE = 0.10           # binding cap - numeral corruption at 10% of negatives
N1_SPLIT = {"oom": 0.50, "arbitrary": 0.30, "near_miss": 0.20}
N2_WEIGHT = {"difference": 2.0, "mean": 2.0}   # "concentrated on difference/mean vs sum"

# R15-B1 clause 4 form weights, re-derived on token cost by L1
FORM_WEIGHTS = {"row_prose": 30, "narrative": 30, "pipe": 15,
                "keyvalue": 10, "markdown": 10, "json_records": 5}

NUM_CELL = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")
NUM_FREE = re.compile(r"(?<![\d.,])[-+]?\d[\d,]*(?:\.\d+)?(?![\d.,])")
WIKILINK = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]")


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


def digits(s):
    return sum(ch.isdigit() for ch in s)


def sigdigits(s):
    return len(s.replace("-", "").replace(".", "").replace(",", "").lstrip("0").rstrip("0")) or 1


# --------------------------------------------------------------------------- #
# serialization - one form per DOCUMENT, drawn independently of everything else
# --------------------------------------------------------------------------- #
def _pipe(cap, hdr, body):
    return f"{cap}\n" + "\n".join(" | ".join(r) for r in [hdr] + body)


def _markdown(cap, hdr, body):
    sep = " | ".join("---" for _ in hdr)
    rows = ["| " + " | ".join(hdr) + " |", "| " + sep + " |"]
    rows += ["| " + " | ".join(r) + " |" for r in body]
    return f"{cap}\n" + "\n".join(rows)


def _keyvalue(cap, hdr, body):
    return "\n".join([cap] + ["; ".join(f"{h}: {c}" for h, c in zip(hdr, r)) for r in body])


def _json_records(cap, hdr, body):
    recs = [{h: c for h, c in zip(hdr, r)} for r in body]
    return cap + "\n" + json.dumps(recs, ensure_ascii=False)


def _narrative(cap, hdr, body):
    out = [f"{cap}."]
    for r in body:
        rest = ", ".join(f"a {h} of {c}" for h, c in zip(hdr[1:], r[1:]) if c)
        out.append(f"{r[0]} has {rest}." if rest else f"{r[0]} is listed.")
    return " ".join(out)


def _row_prose(cap, hdr, body):
    out = [f"The following records are from {cap}."]
    for r in body:
        for h, c in zip(hdr[1:], r[1:]):
            if c:
                out.append(f"The {h} of {r[0]} is {c}.")
    return " ".join(out)


SERIALIZERS = {"pipe": _pipe, "markdown": _markdown, "keyvalue": _keyvalue,
               "json_records": _json_records, "narrative": _narrative,
               "row_prose": _row_prose}


def serialize(form, cap, hdr, body):
    return SERIALIZERS[form](cap, hdr, body).replace("\r\n", "\n")[:CHUNK_MAX]


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
        out.append({"doc_id": f"tabfact:{tid}", "source": "tabfact",
                    "caption": (cap or "table").strip(), "hdr": rows[0], "body": body})
    return out


def feverous_tables():
    d = pl.read_parquet(FEVEROUS_SRC).unique(subset=["evidence"], keep="first")
    out, seen = [], set()
    for i, ev in enumerate(d["evidence"].to_list()):
        ev = WIKILINK.sub(r"\2", ev).replace("[SEP]", "\n").strip()
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
        cap = (segs[0].split("|")[0] or "table").strip()
        out.append({"doc_id": f"feverous:{i}", "source": "feverous",
                    "caption": cap[:120], "hdr": hdr, "body": body})
    return out


def numeric_columns(hdr, body):
    """(col_index, [(row_index, value)]) for every usable numeric column.

    Usable = >= 4 numeric rows, >= 4 distinct values, every contributing row
    carrying a non-empty NON-numeric label in column 0 (P1's rule)."""
    out = []
    for ci in range(1, len(hdr)):
        vals = [(ri, as_num(r[ci])) for ri, r in enumerate(body)]
        vals = [(ri, v) for ri, v in vals if v is not None]
        if len(vals) < 4 or len({v for _, v in vals}) < 4:
            continue
        if any((not body[ri][0].strip()) or as_num(body[ri][0]) is not None for ri, _ in vals):
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
    if dtype == "rounding":
        return f"The {col} of {ka} is approximately {{}}."
    raise ValueError(dtype)


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
        step = 10.0 if abs(a) >= 100 else 1.0
        r = round(a / step) * step
        return None if abs(r - a) < 1e-9 else r
    raise ValueError(dtype)


def swap_operator(dtype, a, b):
    """N2 - the operator-swap negative. difference/mean invert against sum, which
    are the two measured inversions (0.4319, 0.4648)."""
    if dtype in ("difference", "mean", "product"):
        return a + b
    if dtype == "sum":
        return a - b
    if dtype == "ratio":
        return a - b
    if dtype == "pct_change":
        return b - a
    if dtype == "scale_unit":
        return a / 1000
    return None


def perturb_last_digit(s):
    """N7 - last-digit corruption of the correct result string."""
    idx = [i for i, ch in enumerate(s) if ch.isdigit()]
    if not idx:
        return None
    i = idx[-1]
    return s[:i] + str((int(s[i]) + 1) % 10) + s[i + 1:]


def rounding_negative(a, rng):
    """R15-B1 clause 5 - prefix-balanced rounding negatives. Half round in the
    WRONG DIRECTION at the same place; half round to the WRONG PLACE."""
    step = 10.0 if abs(a) >= 100 else 1.0
    correct = round(a / step) * step
    if rng.random() < 0.5:
        wrong = (math.floor(a / step) * step if correct > a else math.ceil(a / step) * step)
        kind = "wrong_direction"
    else:
        wrong = round(a / (step * 10)) * (step * 10)
        kind = "wrong_place"
    return wrong, kind


# --------------------------------------------------------------------------- #
# quotas
# --------------------------------------------------------------------------- #
def build_quotas():
    core_pairs = int(round(N_PAIRS * (1 - SUB_FRAC)))
    sub_pairs = N_PAIRS - core_pairs
    tot = sum(TYPE_SHARES.values())
    tq = {t: int(round(core_pairs * s / tot)) for t, s in TYPE_SHARES.items()}
    # absorb the rounding residual on the largest type
    tq["sum"] += core_pairs - sum(tq.values())

    non_round = {t: n for t, n in tq.items() if t != "rounding"}
    n2_total = int(round(N2_SHARE * sum(non_round.values())))
    wsum = sum(N2_WEIGHT.get(t, 1.0) * n for t, n in non_round.items())
    fam = {}
    for t, n in tq.items():
        if t == "rounding":
            fam[t] = {"N_round": n}
            continue
        n7 = int(round(N7_SHARE * n))
        n2 = min(int(round(n2_total * N2_WEIGHT.get(t, 1.0) * n / wsum)), n - n7)
        n1 = n - n7 - n2
        f = {"N7_numeral": n7, "N2_operator": n2}
        f["N1_oom"] = int(round(n1 * N1_SPLIT["oom"]))
        f["N1_arbitrary"] = int(round(n1 * N1_SPLIT["arbitrary"]))
        f["N1_near_miss"] = n1 - f["N1_oom"] - f["N1_arbitrary"]
        fam[t] = f
    sub = {"bind_col": int(round(sub_pairs * 0.40)),
           "compare": int(round(sub_pairs * 0.40))}
    sub["bind_row"] = sub_pairs - sub["bind_col"] - sub["compare"]
    return tq, fam, sub, core_pairs, sub_pairs


# --------------------------------------------------------------------------- #
# core construction
# --------------------------------------------------------------------------- #
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


def try_core(tab, dtype, want_family, rng, dlen_count, dlen_cap):
    """One core minimal pair over one table, or None."""
    cols = numeric_columns(tab["hdr"], tab["body"])
    if not cols:
        return None
    ci, vals = cols[rng.randrange(len(cols))]
    if len(vals) < 4:
        return None
    col = (tab["hdr"][ci] or f"column {ci}").strip() or f"column {ci}"
    pick = rng.sample(range(len(vals)), 4)
    (ri_i, vi), (ri_j, vj), (ri_k, vk), (ri_l, vl) = [vals[p] for p in pick]

    if dtype == "product" and not (2 <= sigdigits(fmt(vi)) <= 3 and 2 <= sigdigits(fmt(vj)) <= 3):
        return None

    correct = derive(dtype, vi, vj)
    if correct is None:
        return None
    v_pos = fmt(correct)
    if not (DIGIT_LEN_MIN <= digits(v_pos) <= DIGIT_LEN_MAX):
        return None
    if dlen_count[digits(v_pos)] >= dlen_cap:
        return None

    # --- evidence: BODY_ROWS retained, all four operand rows inside it -------
    # R15-B1 clause 4: the retained set is CHOSEN to carry the operand rows
    # (not a prefix), so the "both operands survive" assertion never drops a
    # tuple; the declared price is that the retained rows are non-contiguous.
    keep = {ri_i, ri_j, ri_k, ri_l}
    if len(keep) < 4 or len(keep) > BODY_ROWS:
        return "operand_row_loss"
    others = [r for r in range(len(tab["body"])) if r not in keep]
    rng.shuffle(others)
    keep |= set(others[: BODY_ROWS - len(keep)])
    body6 = [tab["body"][r] for r in sorted(keep)]
    chunk = serialize(tab["form"], tab["caption"], tab["hdr"], body6)
    present = canon_set(chunk)

    ka, kb = tab["body"][ri_i][0].strip(), tab["body"][ri_j][0].strip()
    if not ka or not kb or ka == kb:
        return None

    # --- the negative -------------------------------------------------------
    fam = want_family
    if fam == "N_round":
        wrong, kind = rounding_negative(vi, rng)
        fam_detail = f"N_round:{kind}"
    elif fam == "N7_numeral":
        p = perturb_last_digit(v_pos)
        if p is None:
            return None
        try:
            wrong = float(p)
        except ValueError:
            return None
        fam_detail = fam
    elif fam == "N2_operator":
        wrong = swap_operator(dtype, vi, vj)
        fam_detail = fam
    else:  # N1 wrong-operand, band-targeted
        target = fam.split("_", 1)[1]
        cands = []
        for (ra, va) in vals:
            for (rb, vb) in vals:
                if ra == rb or (ra, rb) == (ri_i, ri_j):
                    continue
                if ra not in keep or rb not in keep:
                    continue
                w = derive(dtype, va, vb)
                if w is None:
                    continue
                cands.append((band(correct, w), w))
        hit = [w for b, w in cands if b == target]
        if not hit:
            return None
        wrong = hit[rng.randrange(len(hit))]
        fam_detail = fam
    if wrong is None:
        return None

    v_neg = fmt(wrong)
    if v_neg == v_pos:
        return None
    if not (DIGIT_LEN_MIN <= digits(v_neg) <= DIGIT_LEN_MAX):
        return None
    if abs(digits(v_neg) - digits(v_pos)) > 1:          # clause 3 - digit-length parity
        return None
    if fam == "N_round" and digits(v_neg) != digits(v_pos):   # clause 5 - prefix-balanced
        return None
    # absence rule - BOTH asserted values absent from the serialized evidence
    if canon_set(v_pos) & present or canon_set(v_neg) & present:
        return None

    # --- claim form: bare assertion vs shown work, drawn independently of label
    tpl = template(dtype, col, ka, kb)
    form = "bare" if rng.random() < BARE_FRAC else "shown"
    if form == "shown":
        li, lj = tab["body"][ri_i][ci].strip(), tab["body"][ri_j][ci].strip()
        if dtype in ("scale_unit", "rounding"):
            pre = f"The {col} of {ka} is {li}, so "
        else:
            pre = f"The {col} of {ka} is {li} and the {col} of {kb} is {lj}, so "
        tpl = pre + tpl[0].lower() + tpl[1:]

    return {
        "claim_pos": tpl.format(v_pos), "claim_neg": tpl.format(v_neg), "chunk": chunk,
        "dtype": dtype, "neg_family": fam_detail, "claim_form": form,
        "serial_form": tab["form"], "doc_id": tab["doc_id"], "source": tab["source"],
        "column": col, "key_a": ka, "key_b": kb,
        "operand_a": float(vi), "operand_b": float(vj),
        "v_pos": v_pos, "v_neg": v_neg, "result_digits": digits(v_pos),
    }


# --------------------------------------------------------------------------- #
# relational sub-block (R15-B4 / H138) - every asserted value PRESENT
# --------------------------------------------------------------------------- #
def try_rel(tab, arm, rng, gap_count, gap_cap):
    cols = numeric_columns(tab["hdr"], tab["body"])
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
                if lx == ly or digits(lx) != digits(ly):   # B4 - digit-length matched
                    continue
                ka = tab["body"][ri][0].strip()
                if not ka:
                    continue
                keep = {ri}
                others = [r for r in range(len(tab["body"])) if r != ri]
                rng.shuffle(others)
                keep |= set(others[: BODY_ROWS - 1])
                body6 = [tab["body"][r] for r in sorted(keep)]
                chunk = serialize(tab["form"], tab["caption"], tab["hdr"], body6)
                pres = canon_set(chunk)
                if not (canon_set(lx) & pres) or not (canon_set(ly) & pres):
                    continue
                tpl = f"The {colx} of {ka} is {{}}."
                return {"claim_pos": tpl.format(lx), "claim_neg": tpl.format(ly),
                        "chunk": chunk, "arm": "bind_col", "serial_form": tab["form"],
                        "doc_id": tab["doc_id"], "source": tab["source"],
                        "column": colx, "column_b": coly, "key_a": ka, "key_b": ka,
                        "v_pos": lx, "v_neg": ly, "gap_stratum": ""}
        return None

    ci, vals = cols[rng.randrange(len(cols))]
    col = (tab["hdr"][ci] or f"column {ci}").strip() or f"column {ci}"
    if len(vals) < 2:
        return None

    if arm == "bind_row":
        for _ in range(6):
            (ra, va), (rb, vb) = [vals[p] for p in rng.sample(range(len(vals)), 2)]
            la, lb = fmt(va), fmt(vb)
            ka, kb = tab["body"][ra][0].strip(), tab["body"][rb][0].strip()
            if la == lb or not ka or not kb or ka == kb:
                continue
            if lb in " | ".join(tab["body"][ra]):
                continue
            keep = {ra, rb}
            others = [r for r in range(len(tab["body"])) if r not in keep]
            rng.shuffle(others)
            keep |= set(others[: BODY_ROWS - 2])
            body6 = [tab["body"][r] for r in sorted(keep)]
            chunk = serialize(tab["form"], tab["caption"], tab["hdr"], body6)
            pres = canon_set(chunk)
            if not (canon_set(la) & pres) or not (canon_set(lb) & pres):
                continue
            tpl = f"The {col} of {ka} is {{}}."
            return {"claim_pos": tpl.format(la), "claim_neg": tpl.format(lb),
                    "chunk": chunk, "arm": "bind_row", "serial_form": tab["form"],
                    "doc_id": tab["doc_id"], "source": tab["source"],
                    "column": col, "column_b": col, "key_a": ka, "key_b": kb,
                    "v_pos": la, "v_neg": lb, "gap_stratum": ""}
        return None

    # arm == "compare": ordering, both operands printed, no computation
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
        hi, lo = (ra, va), (rb, vb)
        if vb > va:
            hi, lo = (rb, vb), (ra, va)
        k_hi, k_lo = tab["body"][hi[0]][0].strip(), tab["body"][lo[0]][0].strip()
        if not k_hi or not k_lo or k_hi == k_lo:
            continue
        keep = {ra, rb}
        others = [r for r in range(len(tab["body"])) if r not in keep]
        rng.shuffle(others)
        keep |= set(others[: BODY_ROWS - 2])
        body6 = [tab["body"][r] for r in sorted(keep)]
        chunk = serialize(tab["form"], tab["caption"], tab["hdr"], body6)
        pres = canon_set(chunk)
        if not (canon_set(fmt(va)) & pres) or not (canon_set(fmt(vb)) & pres):
            continue
        gap_count[stratum] += 1
        return {"claim_pos": f"The {col} of {k_hi} is greater than the {col} of {k_lo}.",
                "claim_neg": f"The {col} of {k_lo} is greater than the {col} of {k_hi}.",
                "chunk": chunk, "arm": "compare", "serial_form": tab["form"],
                "doc_id": tab["doc_id"], "source": tab["source"],
                "column": col, "column_b": col, "key_a": k_hi, "key_b": k_lo,
                "v_pos": fmt(va), "v_neg": fmt(vb), "gap_stratum": stratum}
    return None


# --------------------------------------------------------------------------- #
def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    print("loading corpora (public: TabFact train, FEVEROUS train)...", flush=True)
    tables = tabfact_tables() + feverous_tables()
    print(f"  candidate tables: {len(tables)}", flush=True)

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

    fam_left = {t: dict(f) for t, f in fam_q.items()}
    sub_left = dict(sub_q)
    per_doc = collections.Counter()
    rows, pair_id = [], 0
    dropped_operand_rows = 0
    rejections = 0

    order = list(range(len(tables)))
    for cap in (2, 3, 4, 6):          # per-table cap; raised only if supply is short
        rng.shuffle(order)
        for oi in order:
            if per_doc[oi] >= cap:
                continue
            tab = tables[oi]
            need_core = [t for t, f in fam_left.items() if sum(f.values()) > 0]
            need_sub = [a for a, n in sub_left.items() if n > 0]
            if not need_core and not need_sub:
                break
            made = None
            # bias the draw by remaining quota so shares land where registered
            if need_core and (not need_sub or rng.random() < 0.85):
                rng.shuffle(need_core)
                for dtype in need_core[:4]:
                    fams = [f for f, n in fam_left[dtype].items() if n > 0]
                    if not fams:
                        continue
                    f = fams[rng.randrange(len(fams))]
                    r = try_core(tab, dtype, f, rng, dlen_count, dlen_cap)
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
                continue
            kind, r = made
            per_doc[oi] += 1
            tag = TAG_CORE if kind == "core" else TAG_REL
            base = {k: v for k, v in r.items() if k not in ("claim_pos", "claim_neg")}
            for lab, cl, val in ((1.0, r["claim_pos"], r["v_pos"]),
                                 (0.0, r["claim_neg"], r["v_neg"])):
                rows.append({"claim": cl, "chunk": r["chunk"], "label": lab, "tag": tag,
                             "pair_id": pair_id, "block": kind, "asserted_value": val,
                             **{k: v for k, v in base.items() if k != "chunk"}})
            pair_id += 1
        left = sum(sum(f.values()) for f in fam_left.values()) + sum(sub_left.values())
        print(f"  cap={cap}: pairs={pair_id} remaining_quota={left}", flush=True)
        if left == 0:
            break

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
        "operand_row_loss_note": ("0 by construction - R15-B1 clause 4 is implemented "
            "as row SELECTION (the retained 6 body rows are chosen to carry the operand "
            "rows) rather than a 6-row prefix; the declared price is non-contiguous rows"),
        "result_digit_length_counts": {str(k): v for k, v in sorted(dlen_count.items())},
        "compare_gap_strata": dict(gap_count),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("rows", "pairs", "family_unfilled", "sub_unfilled")}, indent=1))


if __name__ == "__main__":
    main()
