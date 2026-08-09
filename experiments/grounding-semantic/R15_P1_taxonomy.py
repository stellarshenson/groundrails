"""R15-P1: value-driven derivation-type classifier over finqa/tatqa deciding sentences.

For each numeral asserted in a deciding sentence that is ABSENT from the response's
evidence windows, search the operand pool for an arithmetic explanation. Operand pool =
numerals in the sentence that ARE present in evidence (the LLM quotes its operands) plus,
for single-operand ops, the whole evidence value set.
"""
import itertools, json, re
import polars as pl

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
YEAR_RANGE = re.compile(r"(?:19|20)\d\d\s*[-–]\s*(?:19|20)\d\d")
TOL = 2e-3   # relative; catches 2-dp rounding of a percent


def numvals(t):
    out = []
    for m in NUM.finditer(t):
        s = m.group().replace(",", "").rstrip(".")
        try:
            v = float(s)
        except ValueError:
            continue
        out.append((m.group(), v, m.start()))
    return out


def close(a, b):
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) <= TOL * max(1.0, abs(b))


def present(v, evvals):
    return any(abs(e - v) < 1e-6 or close(v, e) for e in evvals)


def explain(v, pool, evvals):
    """Return derivation type for absent value v, or None."""
    P = sorted({p for p in pool if abs(p) > 1e-12})
    # --- unit / scale conversion (single operand) ---
    for e in list(P) + list(evvals):
        for k in (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
            if close(v, e * k):
                return "scale_unit"
    # --- ratio x100 of a present fraction, or fraction of a present pct ---
    for e in list(P):
        if close(v, e * 100) or close(v, e / 100):
            return "pct_scaling"
    pairs = list(itertools.permutations(P, 2))
    # --- percent change ---
    for a, b in pairs:
        if b != 0:
            if close(v, (a - b) / b * 100) or close(v, (a - b) / b):
                return "pct_change"
    # --- ratio / share of total ---
    for a, b in pairs:
        if b != 0 and close(v, a / b):
            return "ratio"
        if b != 0 and close(v, a / b * 100):
            return "ratio"
    # --- difference ---
    for a, b in pairs:
        if close(v, a - b):
            return "difference"
    # --- sum (2..4 operands) ---
    for n in (2, 3, 4):
        for c in itertools.combinations(P, n):
            if close(v, sum(c)):
                return "sum"
    # --- mean (2..4 operands) ---
    for n in (2, 3, 4):
        for c in itertools.combinations(P, n):
            if close(v, sum(c) / n):
                return "mean"
    # --- product ---
    for a, b in itertools.combinations(P, 2):
        if close(v, a * b):
            return "product"
    # --- percent-of (a * b/100) ---
    for a, b in pairs:
        if close(v, a * b / 100):
            return "percent_of"
    # --- count aggregation: small int, plausibly a row/year count ---
    if abs(v - round(v)) < 1e-9 and 0 <= v <= 12:
        return "count_agg"
    return "unexplained"


def main():
    d = pl.read_parquet("/tmp/p1_deciding.parquet")
    rows = []
    for r in d.iter_rows(named=True):
        ev = r["ev"]
        evvals = {v for _, v, _ in numvals(ev)}
        sent = r["sent_text"]
        yr_spans = [m.span() for m in YEAR_RANGE.finditer(sent)]
        toks = numvals(sent)
        pool = [v for _, v, _ in toks if present(v, evvals)]
        absent = []
        for s, v, pos in toks:
            if present(v, evvals):
                continue
            absent.append((s, v, pos))
        # pass 1: explain from evidence-present operands only
        first = {}
        for s, v, pos in absent:
            # parse artifacts
            if any(a <= pos < b for a, b in yr_spans):
                t = "parse_artifact_year"
            elif v in (100.0,) and re.search(r"[x*×]\s*100|100\s*(?:to get|\))", sent):
                t = "formula_constant_100"
            else:
                t = explain(v, pool, evvals)
            first[(s, pos)] = (v, t)
        # pass 2: chain - readmit values explained in pass 1 as operands, re-try the rest
        pool2 = pool + [v for (v, t) in first.values()
                        if t not in ("unexplained", "parse_artifact_year", "formula_constant_100")]
        for (s, pos), (v, t) in first.items():
            if t == "unexplained":
                t2 = explain(v, pool2, evvals)
                t = "unexplained" if t2 == "unexplained" else f"{t2}_chain2"
            rows.append({"subset": r["subset"], "resp_idx": r["resp_idx"],
                         "sent_score": r["sent_score"], "label": r["label"],
                         "resp_label": r["resp_label"], "tok": s, "val": v, "dtype": t})
    out = pl.DataFrame(rows)
    out.write_parquet("/home/lab/workspace/private/ai-assistants/groundrails/tmp/p1_types.parquet")
    print("=== absent-numeral level ===")
    pl.Config.set_tbl_rows(100)
    print(out.group_by(["subset", "dtype"]).agg(pl.len().alias("n")).sort(["subset", "n"], descending=[False, True]))
    # per deciding-sentence: the set of REAL derivation types (artifacts excluded)
    REAL = out.filter(~pl.col("dtype").is_in(["parse_artifact_year", "formula_constant_100"]))
    print("\n=== deciding sentences carrying >=1 numeral of each type ===")
    per = REAL.group_by(["subset", "dtype"]).agg(
        pl.col("resp_idx").n_unique().alias("n_sent"),
        pl.col("sent_score").mean().round(4).alias("mean_score"))
    print(per.sort(["subset", "n_sent"], descending=[False, True]))
    tot = d.group_by("subset").len()
    print(tot)


if __name__ == "__main__":
    main()
