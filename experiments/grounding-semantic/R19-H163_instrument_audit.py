"""R19-H163 PROBE-BANK INSTRUMENT AUDIT - does the probe bank predict the arena?

ANALYSIS ONLY.  No GPU, no training, no new data.  Reads only banked artifacts.

WHY THIS EXISTS
---------------
Lane R19-H161/L2 measured the probe bank running BACKWARDS against the arena on
both classes it can speak to: table-binding error mass FELL on the arena while
bind_col FELL on the probe, and scale_unit ROSE on the probe while tatqa's
scale/unit arena error mass nearly TRIPLED.  Two observations on one checkpoint
pair is an anecdote.  This audit turns it into a measurement.

Every arm in this campaign has been steered, in part, by probe movement.  If the
probes do not predict arena movement across the checkpoints that carry both
readings, then that steering was never informative and the bank must be demoted
to report-only.  That is a claim about the INSTRUMENT, not about any arm.

METHOD
------
Nine banked checkpoints carry BOTH a probe-bank reading (`*_probes_draw*_result
.json`, field `headline`) and a blind arena windowed reading (`*_windowed_result
.json`, fields `mean` and `per_subset`).  For each probe metric and each arena
quantity, Spearman rank correlation across those checkpoints, with a two-sided
permutation null (the probe vector is shuffled; the arena vector is held).

PRE-REGISTERED TARGETING MAP - written before any correlation was computed.
Each probe is scored against the arena quantity IT CLAIMS TO SPEAK TO, not
against whatever it happens to correlate with.  A probe that predicts the arena
mean but not its own target has not earned its name.

POWER HONESTY
-------------
n = 9 checkpoints.  Spearman's standard error at n=9 is ~0.35, so only |rho|
above ~0.7 is distinguishable from noise at the conventional level, and the
permutation null needs |rho| >= 0.717 for two-sided p < 0.05.  A null result
here is "this audit cannot see an effect of the size the campaign has been
assuming", NOT "the effect is zero".  Both readings are reported.

VERDICT RULE, pre-registered
----------------------------
For each probe: SUPPORTED if its targeted correlation is positive and clears the
permutation null; DEAD-AS-INSTRUMENT if the point estimate is negative (it points
the wrong way); INDETERMINATE otherwise.  A DEAD probe is demoted to report-only
in the canonical log and may not be cited as evidence for or against a lane.

Run:  uv run python experiments/grounding-semantic/R19-H163_instrument_audit.py
"""

import itertools
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R19-H163_instrument_audit.json"
TABLE = HERE / "R19-H163_instrument_audit.parquet"

SEED = 191663
N_PERM = 200_000

# Explicit pairing, probe-reading file -> arena windowed-reading file.  Written
# out rather than inferred so the join is auditable: a wrong pair here would
# silently corrupt every correlation below.
PAIRS = [
    ("R14-H133_probes_draw1_result.json", "R14-H133_arm_draw1_windowed_result.json"),
    ("R14-H133_probes_draw2_result.json", "R14-H133_arm_draw2_windowed_result.json"),
    ("R17-H145_probes_draw1_result.json", "R17-H145_arm_draw1_windowed_result.json"),
    ("R17-H146_probes_draw1_result.json", "R17-H146_arm_draw1_windowed_result.json"),
    ("R18-H150_probes_draw1_result.json", "R18-H150_arm_draw1_windowed_result.json"),
    ("R18-H150-d2_probes_draw2_result.json", "R18-H150_arm_draw2_windowed_result.json"),
    ("R18-H156_probes_draw1_result.json", "R18-H156_arm_draw1_windowed_result.json"),
    ("R19-H159_probes_draw1_result.json", "R19-H159_arm_draw1_windowed_result.json"),
    ("R19-H160-soupB_probes_draw1_result.json", "R19-H160_soup_soupB_windowed_result.json"),
]

# Probe metrics lifted from `headline`.  tier1 is a dict of four algebraic
# variants of the same numeric-reasoning read; all four are carried because the
# campaign has cited different ones at different times.
FLAT_PROBES = [
    "scale_unit_arm",
    "verbatim_mean_arm",
    "auroc_a_vs_b_arm",
    "bind_col_arm",
    "compare_arm",
    "bind_row_arm",
]
TIER1_VARIANTS = ["difference", "ratio", "pct_change", "sum"]

SUBSETS = [
    "covidqa", "delucionqa", "emanual", "expertqa", "finqa",
    "hagrid", "hotpotqa", "pubmedqa", "tatqa", "techqa",
]

# PRE-REGISTERED: what each probe claims to speak to.  "mean" is the arena mean.
TARGETS = {
    "bind_col_arm": ["finqa", "tatqa"],
    "bind_row_arm": ["finqa", "tatqa"],
    "scale_unit_arm": ["tatqa", "finqa"],
    "verbatim_mean_arm": ["delucionqa", "hagrid"],
    "compare_arm": ["hotpotqa"],
    "auroc_a_vs_b_arm": ["mean"],
    "tier1_difference": ["finqa", "tatqa"],
    "tier1_ratio": ["finqa", "tatqa"],
    "tier1_pct_change": ["finqa", "tatqa"],
    "tier1_sum": ["finqa", "tatqa"],
}


def rankdata(a):
    """Average-tie ranks, so Spearman is correct when probes repeat a value."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    for value in np.unique(a):
        mask = a == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def perm_p(x, y, rho, rng, n_perm=N_PERM):
    """Two-sided permutation p on |rho|.

    n=9 gives 362,880 distinct permutations, so for n<=8 the exact null is
    enumerated and for n=9 a large random sample is used; either way the p is a
    permutation p, not a normal approximation, which at this n would be wrong.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if np.isnan(rho):
        return float("nan"), "undefined"
    if n <= 8:
        nulls = [abs(spearman(np.array(p), y)) for p in itertools.permutations(x)]
        return float(np.mean(np.array(nulls) >= abs(rho) - 1e-12)), "exact"
    hits = 0
    xs = x.copy()
    for _ in range(n_perm):
        rng.shuffle(xs)
        if abs(spearman(xs, y)) >= abs(rho) - 1e-12:
            hits += 1
    return float((hits + 1) / (n_perm + 1)), f"sampled/{n_perm}"


def load_pair(probe_file, arena_file):
    p = json.loads((HERE / probe_file).read_text())
    a = json.loads((HERE / arena_file).read_text())
    head = p["headline"]

    row = {
        "probe_file": probe_file,
        "arena_file": arena_file,
        "checkpoint": str(p.get("arm_checkpoint") or p.get("checkpoint") or "?"),
    }
    for k in FLAT_PROBES:
        row[k] = float(head[k]) if k in head else None
    t1 = head.get("tier1_arm") or {}
    for v in TIER1_VARIANTS:
        row[f"tier1_{v}"] = float(t1[v]) if v in t1 else None

    row["mean"] = float(a["mean"])
    per = a.get("per_subset", a)
    for s in SUBSETS:
        val = per.get(s)
        if isinstance(val, dict):
            val = val.get("auroc", val.get("windowed", val.get("value")))
        row[s] = float(val) if val is not None else None
    return row


def main():
    rng = np.random.default_rng(SEED)

    rows = []
    missing = []
    for pf, af in PAIRS:
        if not (HERE / pf).exists() or not (HERE / af).exists():
            missing.append((pf, af, (HERE / pf).exists(), (HERE / af).exists()))
            continue
        rows.append(load_pair(pf, af))

    df = pl.DataFrame(rows)
    df.write_parquet(TABLE)
    n = df.height

    print(f"paired checkpoints: {n}")
    if missing:
        print("MISSING PAIRS (excluded, not substituted):")
        for m in missing:
            print(f"  {m}")
    print()
    print("arena means:", [round(v, 5) for v in df["mean"].to_list()])
    print()

    probe_names = FLAT_PROBES + [f"tier1_{v}" for v in TIER1_VARIANTS]
    arena_names = ["mean"] + SUBSETS

    results = {}
    full_grid = {}
    for pn in probe_names:
        pv = df[pn].to_list()
        if any(v is None for v in pv):
            results[pn] = {"status": "INCOMPLETE - probe absent on some checkpoints"}
            continue
        grid = {}
        for an in arena_names:
            av = df[an].to_list()
            if any(v is None for v in av):
                continue
            rho = spearman(pv, av)
            p, mode = perm_p(pv, av, rho, rng)
            grid[an] = {"rho": round(rho, 4), "p": round(p, 5), "null": mode}
        full_grid[pn] = grid

        tgt = TARGETS[pn]
        if tgt == ["mean"]:
            tgt_vec = df["mean"].to_list()
            tgt_label = "mean"
        else:
            # the probe's own target, as the mean of the subsets it claims
            arr = np.column_stack([np.array(df[s].to_list(), dtype=float) for s in tgt])
            tgt_vec = arr.mean(axis=1).tolist()
            tgt_label = "+".join(tgt)
        rho_t = spearman(pv, tgt_vec)
        p_t, mode_t = perm_p(pv, tgt_vec, rho_t, rng)

        if np.isnan(rho_t):
            verdict = "INDETERMINATE"
        elif rho_t > 0 and p_t < 0.05:
            verdict = "SUPPORTED"
        elif rho_t < 0:
            verdict = "DEAD-AS-INSTRUMENT"
        else:
            verdict = "INDETERMINATE"

        results[pn] = {
            "target": tgt_label,
            "rho_target": round(rho_t, 4),
            "p_target": round(p_t, 5),
            "null": mode_t,
            "rho_arena_mean": grid.get("mean", {}).get("rho"),
            "p_arena_mean": grid.get("mean", {}).get("p"),
            "verdict": verdict,
        }

    print(f"{'probe':<22} {'target':<18} {'rho_tgt':>8} {'p':>8} {'rho_mean':>9}  verdict")
    print("-" * 88)
    for pn in probe_names:
        r = results[pn]
        if "verdict" not in r:
            print(f"{pn:<22} {r['status']}")
            continue
        rm = r["rho_arena_mean"]
        print(
            f"{pn:<22} {r['target']:<18} {r['rho_target']:>8.4f} {r['p_target']:>8.4f} "
            f"{(f'{rm:.4f}' if rm is not None else 'n/a'):>9}  {r['verdict']}"
        )

    counts = {}
    for r in results.values():
        counts[r.get("verdict", r.get("status", "?"))] = (
            counts.get(r.get("verdict", r.get("status", "?")), 0) + 1
        )
    print()
    print("verdict tally:", counts)

    dead = [k for k, v in results.items() if v.get("verdict") == "DEAD-AS-INSTRUMENT"]
    supported = [k for k, v in results.items() if v.get("verdict") == "SUPPORTED"]
    print(f"DEAD (point estimate points the wrong way): {dead}")
    print(f"SUPPORTED (predicts its own target):        {supported}")

    payload = {
        "arm": "R19-H163 probe-bank instrument audit",
        "question": "does any probe in the bank predict the arena quantity it claims to speak to",
        "status": "NOT ADJUDICATED HERE - the coordinator holds the verdict",
        "analysis_only": True,
        "n_checkpoints": n,
        "seed": SEED,
        "pairs": [{"probe": a, "arena": b} for a, b in PAIRS],
        "missing_pairs": missing,
        "targeting_map_preregistered": TARGETS,
        "power": {
            "n": n,
            "spearman_se_approx": round(1 / np.sqrt(max(n - 1, 1)), 4),
            "note": (
                "n is small; a null is 'this audit cannot see an effect of the assumed "
                "size', not 'the effect is zero'. Point-estimate SIGN is the load-bearing "
                "read, because a probe that points the wrong way cannot steer regardless "
                "of significance."
            ),
        },
        "verdict_rule": {
            "SUPPORTED": "rho_target > 0 and permutation p < 0.05",
            "DEAD-AS-INSTRUMENT": "rho_target < 0 (points the wrong way)",
            "INDETERMINATE": "otherwise",
        },
        "results": results,
        "full_grid_probe_x_arena": full_grid,
        "table": str(TABLE.name),
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nresults -> {OUT}")
    print("=== H163 INSTRUMENT AUDIT COMPLETE ===")


if __name__ == "__main__":
    main()
