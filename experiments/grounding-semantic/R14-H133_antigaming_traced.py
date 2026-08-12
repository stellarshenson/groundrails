"""R14-H133 anti-gaming eval set, TRACE-CONDITIONED. CPU only, no GPU.

The v3 lane trains a detector that VERIFIES a programmatically attached
reasoning trace; it never computes. An eval set of bare claims therefore asks
the model a question the lane never taught. This script re-issues the banked
anti-gaming pairs in the serving shape.

  POSITIVE   the trace states what the table gives, and the claim's value
             matches it
  NEGATIVE   the claim keeps its banked wrong value; the trace still states
             what the table gives, so the trace CONTRADICTS the claim

The trace is BYTE-IDENTICAL within a pair - it reports the evidence, not the
claim - so it contributes no surface signal of its own. The only asymmetry left
is the banked claim's own numeral, which is inherited and deliberately not
distorted; the residual is measured into the manifest.

Pairs are rebuilt with `R14-H133_antigaming.build_nearmiss` / `.build_bindrow`
at their own SEED, so `pair_id` is the row index of the set that script banks.

Run: uv run python experiments/grounding-semantic/R14-H133_antigaming_traced.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # GPU1 is carrying the H133 arm
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util
import json
import pathlib
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R14-H133_antigaming_traced.parquet"
MANIFEST = HERE / "R14-H133_antigaming_traced_manifest.json"

MAX_LEN = 512
TOKEN_TARGET = 500
CPT_EVIDENCE = 2.24          # pipe-serialized tables, R15_gate_L1_serialtok.json
CPT_CLAIM = 3.4

BIND_RE = re.compile(r"^The (.+) of (.+) is (.+)\.$")
NUM = re.compile(r"\d[\d,.]*")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def changed_numeral(pos, neg):
    """The one numeral the H108 operator rewrote, as (true, wrong).

    Returns None when the corruption changed a WORD and left every numeral
    intact - `scale_word`, `pct_pp` and `comparative_flip` do exactly that, and
    a value-lookup trace cannot contradict them."""
    a, b = NUM.findall(pos), NUM.findall(neg)
    if len(a) != len(b):
        return None
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diff) != 1:
        return None
    return diff[0]


def trim_evidence(ev, claim, keep_values, ntok):
    """Hold the 512-token budget by dropping table BODY lines, never the trace.

    The caption, the header line and every line carrying a value the trace
    quotes are protected, so trimming can never break groundability. Measured
    with the model's own tokenizer - a characters-per-token estimate left 22.3%
    of rows over budget on this set."""
    lines = ev.split("\n")
    if len(lines) <= 2:
        return ev
    head, body = lines[:2], lines[2:]
    while True:
        e = "\n".join(head + body)
        if ntok(claim, e) <= TOKEN_TARGET:
            return e
        drop = next((i for i in range(len(body) - 1, -1, -1)
                     if not any(v in body[i] for v in keep_values)), None)
        if drop is None:
            return e
        body.pop(drop)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(HERE.parent.parent / "models" / "R9-H105-mmbert-dann-clean"))

    def ntok(claim, ev):
        return len(tok(claim, ev, truncation=False)["input_ids"])

    C = _mod("c", "R15_gate_common.py")
    H108D = _mod("h108d", "R10-H108_data.py")
    AG = _mod("ag", "R14-H133_antigaming.py")

    rng = np.random.default_rng(AG.SEED)
    rows = AG.build_nearmiss(C, H108D, rng) + AG.build_bindrow(C, rng)
    n_nm = sum(r["kind"] == "nearmiss" for r in rows)
    print(f"banked pairs rebuilt: {n_nm} near-miss + {len(rows) - n_nm} bind_row", flush=True)

    out, skipped = [], collections.Counter()
    residual = collections.Counter()
    per_family = collections.Counter()

    for pid, r in enumerate(rows):
        ev, cp, cn = r["evidence"], r["claim_pos"], r["claim_neg"]

        if r["kind"] == "bind_row":
            m = BIND_RE.match(cp)
            m2 = BIND_RE.match(cn)
            if not m or not m2:
                skipped["bind_row:unparsable_template"] += 1
                continue
            col, ka, v_true = m.group(1), m.group(2), m.group(3)
            v_wrong = m2.group(3)
            trace = (f"The table lists the {col} of {ka} as {v_true}; "
                     f"the value to assert is therefore {v_true}.")
        else:
            ch = changed_numeral(cp, cn)
            if ch is None:
                skipped[f"nearmiss:word_only_corruption:{r['family']}"] += 1
                continue
            v_true, v_wrong = ch
            if not (C.canon_set(v_true) & C.canon_set(ev)):
                skipped[f"nearmiss:true_value_not_in_evidence:{r['family']}"] += 1
                continue
            trace = (f"Locating the asserted figure in the table: the table gives "
                     f"{v_true}; the value to assert is therefore {v_true}.")

        if v_true == v_wrong:
            skipped[f"{r['kind']}:no_value_contrast"] += 1
            continue

        claim_pos = f"{trace} {cp}"
        claim_neg = f"{trace} {cn}"
        chunk = trim_evidence(ev, max(claim_pos, claim_neg, key=len), {v_true}, ntok)
        if not (C.canon_set(v_true) & C.canon_set(chunk)):
            skipped[f"{r['kind']}:true_value_trimmed_away"] += 1
            continue

        # residual surface asymmetry, inherited from the banked claim and NOT
        # distorted - the trace is byte-identical between the pair members
        d = lambda s: sum(c.isdigit() for c in s)                      # noqa: E731
        tz = lambda s: len(s) - len(s.rstrip("0"))                     # noqa: E731
        ld = lambda s: next((c for c in s if c.isdigit()), "")         # noqa: E731
        if d(v_true) != d(v_wrong):
            residual["digit_count_mismatch"] += 1
        if tz(v_true) != tz(v_wrong):
            residual["trailing_zero_mismatch"] += 1
        if ld(v_true) != ld(v_wrong):
            residual["leading_digit_mismatch"] += 1
        if ("." in v_true) != ("." in v_wrong):
            residual["decimal_presence_mismatch"] += 1
        per_family[r["family"]] += 1

        for lab, cl, av in ((1.0, claim_pos, v_true), (0.0, claim_neg, v_wrong)):
            out.append({
                "claim": cl, "chunk": chunk, "label": lab, "tag": r["kind"],
                "pair_id": pid, "kind": r["kind"], "family": r["family"],
                "table_id": r["table_id"], "trace": trace,
                "claim_untraced": cp if lab == 1.0 else cn,
                "asserted_value": av, "table_value": v_true,
            })

    df = pl.DataFrame(out).with_columns(pl.col("label").cast(pl.Float32))
    df.write_parquet(OUT)

    # realised token budget, measured with the model's own tokenizer
    enc = tok(df["claim"].to_list(), df["chunk"].to_list(), truncation=False)
    lens = np.array([len(x) for x in enc["input_ids"]])
    y = df["label"].to_numpy()
    from sklearn.metrics import roc_auc_score
    yy = np.concatenate([np.ones((y == 1).sum(), int), np.zeros((y == 0).sum(), int)])
    len_auroc = float(roc_auc_score(yy, np.concatenate([lens[y == 1], lens[y == 0]])))

    n_pairs = len(df) // 2
    man = {
        "set": OUT.name,
        "source": "R14-H133_antigaming.build_nearmiss + .build_bindrow, rebuilt at "
                  f"SEED {AG.SEED}; pair_id is the row index of the set that script banks",
        "banked_pairs_rebuilt": {"nearmiss": n_nm, "bind_row": len(rows) - n_nm},
        "traced_pairs": n_pairs, "rows": len(df),
        "traced_pairs_by_family": dict(per_family),
        "rows_by_kind": dict(zip(*df.group_by("tag").len().sort("tag"))),
        "not_trace_attachable": dict(skipped),
        "not_trace_attachable_total": sum(skipped.values()),
        "trace_shape": {
            "identical_within_pair": True,
            "note": "the trace reports the EVIDENCE, so it is byte-identical between a "
                    "pair's two members and contributes no surface signal; the negative "
                    "is detected by the trace contradicting the claim's value",
        },
        "residual_surface_asymmetry": {
            "note": "inherited from the banked claim's own numeral and deliberately NOT "
                    "distorted, per the build order; counts are over traced pairs",
            "traced_pairs": n_pairs, **{k: v for k, v in sorted(residual.items())},
            **{f"{k}_share": round(v / max(n_pairs, 1), 5)
               for k, v in sorted(residual.items())},
        },
        "token_budget": {
            "max_len": MAX_LEN,
            "pair_tokens_median": int(np.median(lens)),
            "pair_tokens_p95": int(np.percentile(lens, 95)),
            "share_over_512": round(float((lens > MAX_LEN).mean()), 5),
            "auroc_from_pair_token_length_alone": round(len_auroc, 4),
            "policy": "evidence body rows trimmed, traces never; the caption, the header "
                      "and every line carrying the trace's quoted value are protected",
        },
    }
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in (
        "traced_pairs", "rows", "traced_pairs_by_family", "not_trace_attachable",
        "residual_surface_asymmetry", "token_budget")}, indent=2))
    print(f"-> {OUT}\n-> {MANIFEST}")


if __name__ == "__main__":
    main()
