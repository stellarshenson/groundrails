"""DR-H115 SPAN-DROP-DIAL kill-gate (registration: docs/experiments/semantic-dataset-enhancements.md).

Mechanism: truth-visible clean encoder (eval, cached encoder_outputs); manual
greedy KV-cached decode loop - the clean prefix is forced verbatim (y_ref token
ids), the target window free-decodes with H111's set_dropout(p) active ONLY
during in-window steps (model.train() in-window, model.eval() outside), and the
suffix is spliced back verbatim at token-aligned char boundaries, so text
outside the span is 100% verbatim by construction (hard invariant asserted on
every sample).

Two arms (binding skeptic amendment):
  BANNED  (~600, 50 per dial cell)  - first-free-step ban of the gold token;
                                      characterizes the shipped configuration
  NO-BAN  (~300, concentrated p=0.2 x 1-2tok) - carries the copy-through KILL
                                      metric: drift < 8% in that cell = KILL

Dial grid: p in {0.1, 0.2, 0.3, 0.4} x span-length bucket {1-2, 3-6, 7-15} tok.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/DR_H115_gate.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import polars as pl
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import importlib.util


def _load_mod(name):
    modname = name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(modname, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


S0 = _load_mod("R10-H111_stage0")
import DR_targeting as DRT

DEV = "cuda"
SEED = 0
P_GRID = [0.1, 0.2, 0.3, 0.4]
BUCKETS = [(1, 2), (3, 6), (7, 15)]  # inclusive token-length bands
BANNED_PER_CELL = 50
NOBAN_KILL_CELL = 100  # p=0.2 x 1-2tok, the copy-through kill cell
NOBAN_PER_CELL = 18
PAIRS = HERE / "R10-H111_stage1_pairs.parquet"
TH = json.loads((HERE / "R10-H111_stage1_progress.json").read_text())["thresholds"]

OUT_PARQUET = HERE / "DR_H115_gate_results.parquet"
OUT_REPORT = HERE / "DR_H115_gate_report.md"

_SPELLED = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "hundred": "100", "thousand": "1000",
    "million": "1000000", "first": "1", "second": "2", "third": "3",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(text: str) -> str:
    t = text.casefold()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t)
    return " ".join(_SPELLED.get(w, w) for w in words)


def degenerate_span(span_text: str) -> bool:
    if not span_text.strip():
        return True
    toks = span_text.lower().split()
    if len(toks) >= 3:
        grams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
        if len(set(grams)) / len(grams) < TH["distinct3_min"]:
            return True
    if len(toks) >= 2:
        run = 1
        for a, b in zip(toks, toks[1:]):
            run = run + 1 if a == b else 1
            if run > TH["maxrun_max"]:
                return True
    crun = best = 1
    for a, b in zip(span_text, span_text[1:]):
        crun = crun + 1 if a == b and not a.isspace() else 1
        best = max(best, crun)
    if best > TH["charrun_max"]:
        return True
    sym = sum(1 for c in span_text if not (c.isalnum() or c in " .,;:'\"()-%$/"))
    return sym / max(len(span_text), 1) > TH["symdens_max"]


def editdist_le2(a: str, b: str) -> bool:
    """True when Levenshtein(a, b) <= 2 (banded DP, early exit)."""
    if abs(len(a) - len(b)) > 2:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        lo = 2 + len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            lo = min(lo, cur[j])
        if lo > 2:
            return False
        prev = cur
    return prev[-1] <= 2


def digits(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def cell_of(p: float, n_sp: int) -> str | None:
    for lo, hi in BUCKETS:
        if lo <= n_sp <= hi:
            return f"p{p}_b{lo}-{hi}"
    return None


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    seeds = (
        pl.read_parquet(PAIRS).select("seed").unique(maintain_order=True)["seed"].to_list()
    )
    random.Random(SEED).shuffle(seeds)
    log(f"{len(seeds)} distinct seeds available")

    tok, model = S0.load_mbart()
    model.eval()
    S0.set_dropout(model, 0.0)
    lang_id = tok.lang_code_to_id["en_XX"]
    eos = tok.eos_token_id
    special = set(tok.all_special_ids) | {lang_id}

    # ---- cell quotas: (arm, cell) -> want
    want: dict[tuple[str, str], int] = {}
    for p in P_GRID:
        for lo, hi in BUCKETS:
            want[("banned", f"p{p}_b{lo}-{hi}")] = BANNED_PER_CELL
            want[("noban", f"p{p}_b{lo}-{hi}")] = NOBAN_PER_CELL
    want[("noban", "p0.2_b1-2")] = NOBAN_KILL_CELL
    have = {k: 0 for k in want}

    results = []
    t0 = time.time()
    sample_idx = 0

    with torch.no_grad():
        for seed_text in seeds:
            if all(have[k] >= want[k] for k in want):
                break
            try:
                spans = DRT.sample_spans(
                    seed_text, 2, rng=random.Random(hash(seed_text) % 2**31)
                )
            except Exception:
                continue

            enc = tok(seed_text, return_tensors="pt", truncation=True, max_length=128,
                      return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")[0].tolist()
            ids = enc["input_ids"][0].tolist()
            content = [i for i, (tid, (a, b)) in enumerate(zip(ids, offsets))
                       if (a, b) != (0, 0)]
            input_ids = enc["input_ids"].to(DEV)
            clean_mask = torch.ones_like(input_ids)
            enc_out = None  # lazy: encode only when a span is usable

            for c0, c1, s_text, ltype, source in spans:
                if not s_text.strip() or c1 <= c0:
                    continue
                span_tok = [i for i in content
                            if offsets[i][1] > c0 and offsets[i][0] < c1]
                if not span_tok or len(span_tok) >= len(content):
                    continue
                n_sp = len(span_tok)
                # token-aligned char boundaries (clean splice seam)
                tc0, tc1 = offsets[span_tok[0]][0], offsets[span_tok[-1]][1]
                true_span = seed_text[tc0:tc1]
                # pick (arm, p) for an unfilled cell matching this bucket
                choice = None
                for arm in ("noban", "banned"):
                    for p in P_GRID:
                        cell = cell_of(p, n_sp)
                        if cell and have.get((arm, cell), 99) < want.get((arm, cell), 0):
                            choice = (arm, p, cell)
                            break
                    if choice:
                        break
                if not choice:
                    continue
                arm, p, cell = choice

                if enc_out is None:
                    enc_out = model.get_encoder()(
                        input_ids=input_ids, attention_mask=clean_mask
                    )

                # forced clean prefix
                first_span_pos = span_tok[0]
                prefix_content = [ids[i] for i in content if i < first_span_pos]
                dec_prefix = torch.tensor([[eos, lang_id] + prefix_content], device=DEV)

                torch.manual_seed(sample_idx)
                out = model(
                    encoder_outputs=enc_out,
                    attention_mask=clean_mask,
                    decoder_input_ids=dec_prefix,
                    use_cache=True,
                )
                past = out.past_key_values
                logits = out.logits[0, -1]

                window_ids = []
                gold_first = ids[first_span_pos]
                model.train()
                S0.set_dropout(model, p)
                for step in range(n_sp):
                    if step == 0 and arm == "banned":
                        logits[gold_first] = float("-inf")
                    nxt = int(torch.argmax(logits))
                    if nxt == eos:
                        break
                    window_ids.append(nxt)
                    out = model(
                        encoder_outputs=enc_out,
                        attention_mask=clean_mask,
                        decoder_input_ids=torch.tensor([[nxt]], device=DEV),
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values
                    logits = out.logits[0, -1]
                model.eval()
                S0.set_dropout(model, 0.0)

                dec_span = tok.decode(window_ids, skip_special_tokens=True).strip()
                corrupted = seed_text[:tc0] + dec_span + seed_text[tc1:]
                # HARD INVARIANT: outside-span text 100% verbatim
                assert corrupted[:tc0] == seed_text[:tc0], "prefix invariant broken"
                assert corrupted[len(corrupted) - (len(seed_text) - tc1):] == seed_text[tc1:], \
                    "suffix invariant broken"

                n_true, n_dec = normalize(true_span), normalize(dec_span)
                drift = n_dec != n_true
                degen = degenerate_span(dec_span)
                guarded = (
                    drift and editdist_le2(dec_span, true_span)
                    and digits(dec_span) == digits(true_span)
                )
                results.append({
                    "seed": seed_text, "c0": tc0, "c1": tc1, "true_span": true_span,
                    "ltype": ltype, "source": source, "arm": arm, "p": p,
                    "cell": cell, "n_span_tok": n_sp, "decoded_span": dec_span,
                    "corrupted": corrupted, "drift": drift, "degen": degen,
                    "guarded": guarded,
                })
                have[(arm, cell)] += 1
                sample_idx += 1
                if sample_idx % 100 == 0:
                    filled = sum(min(have[k], want[k]) for k in want)
                    total = sum(want.values())
                    log(f"  {sample_idx} decoded, quota {filled}/{total} "
                        f"({time.time() - t0:.0f}s)")

    log(f"decoded {len(results)} samples in {time.time() - t0:.0f}s; "
        f"unfilled cells: { {k: (have[k], want[k]) for k in want if have[k] < want[k]} }")

    # ---- NLI fwd on drifted, non-degenerate, unguarded samples
    idx = [i for i, r in enumerate(results)
           if r["drift"] and not r["degen"] and not r["guarded"]]
    log(f"NLI fwd on {len(idx)} drifted samples ...")
    del model
    torch.cuda.empty_cache()
    _, fwd = S0.nli_entail([(results[i]["seed"], results[i]["corrupted"]) for i in idx])
    for i, p_ent in zip(idx, fwd):
        results[i]["nli_fwd"] = float(p_ent)
    for r in results:
        r.setdefault("nli_fwd", None)
        r["certified"] = bool(
            r["drift"] and not r["degen"] and not r["guarded"]
            and r["nli_fwd"] is not None and r["nli_fwd"] < 0.8
        )

    # ---- per cell x arm table
    table = {}
    for arm in ("banned", "noban"):
        for p in P_GRID:
            for lo, hi in BUCKETS:
                cell = f"p{p}_b{lo}-{hi}"
                sub = [r for r in results if r["arm"] == arm and r["cell"] == cell]
                if not sub:
                    continue
                table[f"{arm}:{cell}"] = {
                    "n": len(sub),
                    "drift": round(sum(r["drift"] for r in sub) / len(sub), 3),
                    "degen": round(sum(r["degen"] for r in sub) / len(sub), 3),
                    "guarded": round(sum(r["guarded"] for r in sub) / len(sub), 3),
                    "certified": round(sum(r["certified"] for r in sub) / len(sub), 3),
                }

    # ---- kill metric: NO-BAN arm, p=0.2, bucket 1-2
    kc = [r for r in results if r["arm"] == "noban" and r["cell"] == "p0.2_b1-2"]
    kill_drift = sum(r["drift"] for r in kc) / max(len(kc), 1)
    kill = kill_drift < 0.08
    verdict = "KILL" if kill else "SURVIVES-to-pilot"

    # monotonicity: certified vs p (pooled buckets, banned arm)
    mono_p = {p: round(float(np.mean([r["certified"] for r in results
                                      if r["arm"] == "banned" and r["p"] == p]) or 0), 3)
              for p in P_GRID}
    mono_b = {f"{lo}-{hi}": round(float(np.mean(
        [r["certified"] for r in results
         if r["arm"] == "banned" and cell_of(r["p"], r["n_span_tok"]) and
         lo <= r["n_span_tok"] <= hi]) or 0), 3) for lo, hi in BUCKETS}
    best_cell = max(table.items(), key=lambda kv: kv[1]["certified"])

    summary = {
        "n": len(results),
        "kill_metric_noban_p02_b12_drift": round(kill_drift, 4),
        "kill_bar": "< 0.08 = KILL",
        "verdict": verdict,
        "certified_by_p_banned": mono_p,
        "certified_by_bucket_banned": mono_b,
        "best_cell": {best_cell[0]: best_cell[1]},
        "h111_single_point_yield": 0.302,
        "table": table,
    }
    print(json.dumps(summary, indent=1))

    pl.DataFrame(results, infer_schema_length=None).write_parquet(OUT_PARQUET)

    # ---- report with eyeball examples
    ex = []
    seen_cells = set()
    for r in results:
        key = (r["arm"], r["cell"])
        if r["drift"] and not r["degen"] and key not in seen_cells and len(ex) < 15:
            seen_cells.add(key)
            ex.append(r)
    lines = [
        "# DR-H115 SPAN-DROP-DIAL kill-gate\n",
        "```json", json.dumps(summary, indent=1), "```\n",
        "## Eyeball - 15 decodes across cells\n",
    ]
    for i, r in enumerate(ex, 1):
        lines += [
            f"**S{i}** [{r['arm']}:{r['cell']}] ltype={r['ltype']} "
            f"nli_fwd={r['nli_fwd'] if r['nli_fwd'] is None else round(r['nli_fwd'], 2)} "
            f"certified={r['certified']}",
            f"- true : {r['true_span']}",
            f"- dec  : {r['decoded_span']}",
            f"- seed : {r['seed'][:180]}",
            f"- corr : {r['corrupted'][:180]}",
            "",
        ]
    OUT_REPORT.write_text("\n".join(lines))
    log(f"report -> {OUT_REPORT}")
    log(f"parquet -> {OUT_PARQUET}  ({len(results)} rows)")
    print("=== DR-H115 GATE DONE ===", flush=True)


if __name__ == "__main__":
    main()
