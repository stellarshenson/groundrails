"""R12-H119 kill-gate - CPU per-rule ablation of the numeric-surface canonicalizer.

Pre-registered gate (binding amendment 1, skeptic 6): adjudicate against the
committed, reproducible per-rule ablation, not the proposal's number. Only rules
with a NON-ZERO measured agreement effect may ship. A rule that alters >= 5% of
characters on covidqa / pubmedqa / hotpotqa / hagrid chunks is DROPPED.

Measurement. On the arena's finqa subset (RAGBench, read-only - no labels are
used here, only the text), extract every numeric surface token from each claim
and ask whether it appears verbatim in that claim's own evidence. Agreement is
the micro fraction over all claim numbers. The transform is applied
symmetrically to claim and evidence, one rule at a time, and the agreement is
recomputed. The effect of a rule is the point change in agreement.

Run:  uv run python experiments/grounding-semantic/R12-H119_surface_audit.py
"""

import os

# CPU-only gate: keep every GPU out of this process (GPU0/GPU1 are in use).
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARENA = _mod("arena", "R8-H77_unseen_arena.py")
CANON = _mod("canon", "R12-H119_canon.py")

OUT = HERE / "R12-H119_audit_result.json"
CHAR_SUBSETS = ("covidqa", "pubmedqa", "hotpotqa", "hagrid")
CHAR_GATE = 0.05

# A numeric surface token: optional currency prefix (with optional space),
# a digit run possibly carrying thousands separators, an optional fractional
# part, an optional trailing percent (with optional space). Anchored so it never
# starts mid-number.
NUM = re.compile(r"(?<![\w.])(?:[$€£¥][ \t]*)?\d[\d,]*(?:\.\d+)?(?:[ \t]*%)?")

# Sensitivity variant: the bare digit run alone, with no currency or percent
# affix. This is the token definition under which the proposal's skeptic
# measured currency and percent at +0.0 - a rule can only move this metric by
# changing the digits themselves. Reported alongside, never used for the gate,
# because the encoder tokenizes the affix too.
NUM_BARE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def numbers(text, pat=NUM):
    return [m.group(0) for m in pat.finditer(text)]


_PAT_CACHE = {}


def _occurs(token, evidence):
    """Boundary-anchored verbatim occurrence. A bare substring test would score
    the claim number 5 as present inside 500 and 1234 as present inside 12345,
    which inflates every rule that shortens a token."""
    pat = _PAT_CACHE.get(token)
    if pat is None:
        pat = re.compile(r"(?<![\d.,])" + re.escape(token) + r"(?![\d])(?![.,]\d)")
        _PAT_CACHE[token] = pat
    return pat.search(evidence) is not None


def agreement(claims, chunk_lists, tf, pat=NUM):
    """Micro fraction of claim numbers appearing verbatim in their own evidence."""
    hit = tot = 0
    for c, ks in zip(claims, chunk_lists, strict=True):
        ev = tf("\n".join(ks))
        for n in numbers(tf(c), pat):
            tot += 1
            hit += _occurs(n, ev)
    return (hit / tot if tot else 0.0), tot


def char_change(chunk_lists, tf):
    """Fraction of characters altered. Every rule here only inserts or deletes
    single characters (commas, spaces), so |len delta| counts them exactly."""
    changed = total = 0
    for ks in chunk_lists:
        for k in ks:
            total += len(k)
            changed += abs(len(tf(k)) - len(k))
    return changed / total if total else 0.0


def main():
    subs = ARENA.load_subsets()
    claims, chunks, _ = subs["finqa"]
    ident = lambda t: t  # noqa: E731

    base_agree, n_nums = agreement(claims, chunks, ident)
    base_bare, n_bare = agreement(claims, chunks, ident, NUM_BARE)
    print(f"finqa: {len(claims)} claims, {n_nums} claim numbers ({n_bare} bare-token variant)")
    print(f"baseline claim/evidence number agreement  {base_agree:.4f}  (bare {base_bare:.4f})\n")

    rules = {}
    for name in CANON.RULES:
        row = {}
        for direction in ("strip", "add"):
            tf = CANON.transform(direction, [name])
            a, _ = agreement(claims, chunks, tf)
            row[f"agreement_{direction}"] = round(a, 4)
            row[f"effect_{direction}_pts"] = round(100 * (a - base_agree), 2)
            b, _ = agreement(claims, chunks, tf, NUM_BARE)
            row[f"effect_{direction}_pts_bare"] = round(100 * (b - base_bare), 2)
        row["char_change"] = {}
        for sub in CHAR_SUBSETS:
            worst = 0.0
            for direction in ("strip", "add"):
                worst = max(worst, char_change(subs[sub][1], CANON.transform(direction, [name])))
            row["char_change"][sub] = round(worst, 5)
        row["max_char_change"] = round(max(row["char_change"].values()), 5)
        nonzero = row["effect_strip_pts"] != 0.0 or row["effect_add_pts"] != 0.0
        row["nonzero_effect"] = bool(nonzero)
        row["char_gate_ok"] = row["max_char_change"] < CHAR_GATE
        row["ships"] = bool(nonzero and row["char_gate_ok"])
        rules[name] = row
        print(
            f"  {name:10s} strip {row['agreement_strip']:.4f} ({row['effect_strip_pts']:+.2f} pts, "
            f"bare {row['effect_strip_pts_bare']:+.2f})  add {row['agreement_add']:.4f} "
            f"({row['effect_add_pts']:+.2f} pts, bare {row['effect_add_pts_bare']:+.2f})  "
            f"max char change {row['max_char_change']:.4%}  ships={row['ships']}"
        )

    shipped = [n for n, r in rules.items() if r["ships"]]

    full = {}
    if shipped:
        for direction in ("strip", "add"):
            tf = CANON.transform(direction, shipped)
            a, _ = agreement(claims, chunks, tf)
            full[f"finqa_agreement_{direction}"] = round(a, 4)
            full[f"finqa_effect_{direction}_pts"] = round(100 * (a - base_agree), 2)
            full[f"char_change_{direction}"] = {
                sub: round(char_change(subs[sub][1], tf), 5) for sub in CHAR_SUBSETS
            }

    verdict = "PROCEED" if shipped else "KILLED-AT-GATE"
    print("\n" + "=" * 92)
    print("R12-H119 CPU AUDIT - per-rule ablation")
    print("=" * 92)
    print(f"  shipped rules: {shipped or '(none)'}")
    if shipped:
        print(
            f"  full transform finqa agreement  strip {full['finqa_agreement_strip']:.4f} "
            f"({full['finqa_effect_strip_pts']:+.2f} pts)   "
            f"add {full['finqa_agreement_add']:.4f} ({full['finqa_effect_add_pts']:+.2f} pts)"
        )
    print(f"  gate verdict: {verdict}")

    OUT.write_text(
        json.dumps(
            {
                "finqa_claims": len(claims),
                "finqa_claim_numbers": n_nums,
                "baseline_agreement": round(base_agree, 4),
                "baseline_agreement_bare": round(base_bare, 4),
                "per_rule": rules,
                "full_transform": full,
                "shipped_rules": shipped,
                "char_gate": CHAR_GATE,
                "char_gate_subsets": list(CHAR_SUBSETS),
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()
