"""Is the residual discrimination on scale_unit / rounding a digit-copy detector?

For each quad, the named entity's own cell value vi is recoverable from claim_a
("The {col} of {key} is {vi}."). Measure the longest common digit prefix (LCDP)
between the asserted value and vi, and check whether score tracks it.
"""
import re
import numpy as np
import polars as pl

q = pl.read_parquet("/home/lab/workspace/private/ai-assistants/groundrails/experiments/grounding-semantic/R15_P1_typeprobe_quads.parquet")
def tail_num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return m[-1].replace(",", "").rstrip(".") if m else ""
def digits(s):
    return re.sub(r"\D", "", s)
def lcdp(a, b):
    a, b = digits(a), digits(b)
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n
q = q.with_columns(pl.col("claim_a").map_elements(tail_num, return_dtype=pl.Utf8).alias("v_i"))
q = q.with_columns([
    pl.struct(["v_b", "v_i"]).map_elements(lambda s: lcdp(s["v_b"], s["v_i"]), return_dtype=pl.Int64).alias("lcdp_b"),
    pl.struct(["v_c", "v_i"]).map_elements(lambda s: lcdp(s["v_c"], s["v_i"]), return_dtype=pl.Int64).alias("lcdp_c"),
])
pl.Config.set_tbl_rows(30)
print(q.group_by("dtype").agg(
    pl.col("lcdp_b").mean().round(2), pl.col("lcdp_c").mean().round(2),
    pl.col("score_b").mean().round(4), pl.col("score_c").mean().round(4),
    ((pl.col("score_b") > pl.col("score_c")) == (pl.col("lcdp_b") > pl.col("lcdp_c"))).mean().round(3).alias("sign_agree_score_vs_lcdp"),
    (pl.col("lcdp_b") > pl.col("lcdp_c")).mean().round(3).alias("frac_lcdp_b_gt_c"),
).sort("dtype"))
# within scale_unit: does score_b - score_c track lcdp_b - lcdp_c?
for t in ("scale_unit", "rounding", "sum", "difference"):
    s = q.filter(pl.col("dtype") == t)
    x = (s["lcdp_b"] - s["lcdp_c"]).to_numpy().astype(float)
    y = (s["score_b"] - s["score_c"]).to_numpy().astype(float)
    if x.std() > 0:
        print(f"{t:12s} pearson(delta_lcdp, delta_score) = {np.corrcoef(x, y)[0,1]:+.4f}  (n={len(s)})")
    else:
        print(f"{t:12s} delta_lcdp constant at {x[0]:.0f} - no variance")
