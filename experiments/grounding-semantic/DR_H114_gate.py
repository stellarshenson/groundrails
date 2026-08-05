"""DR-H114 XATTN-BLIND kill-gate (registration: docs/experiments/semantic-dataset-enhancements.md).

Mechanism: encode the FULL clean seed (no mask token); blind the decoder to the
target span by zeroing exactly the span's encoder token positions in the
attention_mask passed to generate(encoder_outputs=clean_enc, ...) - consumed
only by decoder cross-attention. Force the clean prefix, free-run GREEDY for
max_new_tokens = span token length +50% slack (pre-registered termination rule),
cut at the last word boundary, verbatim text-level suffix splice.

Gate metrics (pre-registered): reconstruction = normalized match (casefold,
punctuation stripped, whitespace collapsed, numbers canonicalized; containment
of the true span in the decoded span counts as reconstruction - conservative
against the hypothesis). KILL if reconstruction >= 60% OR degenerate/empty > 30%.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/DR_H114_gate.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from collections import defaultdict
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
    sys.modules[modname] = mod  # required: dataclasses resolve cls.__module__ here
    spec.loader.exec_module(mod)
    return mod


S0 = _load_mod("R10-H111_stage0")
import DR_targeting as DRT

DEV = "cuda"
SEED = 0
N_SEEDS = 500
SPANS_PER_SEED = 2
TARGET_SPANS = 800
PAIRS = HERE / "R10-H111_stage1_pairs.parquet"
TH = json.loads((HERE / "R10-H111_stage1_progress.json").read_text())["thresholds"]

OUT_PARQUET = HERE / "DR_H114_gate_results.parquet"
OUT_REPORT = HERE / "DR_H114_gate_report.md"

_SPELLED = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000", "million": "1000000",
    "billion": "1000000000", "first": "1", "second": "2", "third": "3",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(text: str) -> str:
    """Pre-registered normalization: casefold, punctuation stripped, whitespace
    collapsed, numbers canonicalized (1,000 == 1000 == one thousand)."""
    t = text.casefold()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # thousands separators
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t)
    words = [_SPELLED.get(w, w) for w in words]
    return " ".join(words)


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
    crun, best = 1, 1
    for a, b in zip(span_text, span_text[1:]):
        crun = crun + 1 if a == b and not a.isspace() else 1
        best = max(best, crun)
    if best > TH["charrun_max"]:
        return True
    sym = sum(1 for c in span_text if not (c.isalnum() or c in " .,;:'\"()-%$/"))
    if sym / max(len(span_text), 1) > TH["symdens_max"]:
        return True
    return False


def head_repeats(seed: str, span_text: str, c0: int, c1: int) -> bool:
    """Registered restriction: span head token must not appear elsewhere in seed."""
    head = span_text.split()[0].casefold() if span_text.split() else ""
    if not head or len(head) < 3:
        return False
    rest = (seed[:c0] + " " + seed[c1:]).casefold()
    return re.search(rf"\b{re.escape(head)}\b", rest) is not None


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    seeds = (
        pl.read_parquet(PAIRS)
        .select("seed")
        .unique(maintain_order=True)["seed"]
        .to_list()
    )
    random.Random(SEED).shuffle(seeds)
    log(f"{len(seeds)} distinct seeds available")

    # -------- span selection (CPU, spaCy)
    cases = []
    used = 0
    for seed in seeds:
        if len(cases) >= TARGET_SPANS or used >= N_SEEDS * 2:
            break
        used += 1
        try:
            spans = DRT.sample_spans(seed, SPANS_PER_SEED, rng=random.Random(hash(seed) % 2**31))
        except Exception:
            continue
        for c0, c1, s_text, ltype, source in spans:
            if not s_text.strip() or c1 <= c0:
                continue
            if head_repeats(seed, s_text, c0, c1):
                continue
            cases.append({"seed": seed, "c0": c0, "c1": c1, "span": s_text,
                          "ltype": ltype, "source": source})
    log(f"{len(cases)} spans over {used} seeds after head-repetition filter")

    # -------- generation (GPU)
    tok, model = S0.load_mbart()
    model.eval()  # NO dropout anywhere
    lang_id = tok.lang_code_to_id["en_XX"]
    eos = tok.eos_token_id

    results = []
    t0 = time.time()
    with torch.no_grad():
        for k, case in enumerate(cases):
            seed = case["seed"]
            enc = tok(seed, return_tensors="pt", truncation=True, max_length=128,
                      return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")[0].tolist()
            ids = enc["input_ids"][0].tolist()
            # content token indices (skip lang code / eos: offset (0,0) + special ids)
            special = set(tok.all_special_ids) | {lang_id}
            content = [i for i, (tid, (a, b)) in enumerate(zip(ids, offsets))
                       if not (tid in special and a == b == 0) or (a, b) != (0, 0)]
            content = [i for i in content if offsets[i] != [0, 0]]
            span_tok = [i for i in content
                        if offsets[i][1] > case["c0"] and offsets[i][0] < case["c1"]]
            if not span_tok or len(span_tok) >= len(content):
                continue
            # encoder blind: zero exactly the span positions
            attn = torch.ones(1, len(ids), dtype=torch.long)
            for i in span_tok:
                attn[0, i] = 0
            input_ids = enc["input_ids"].to(DEV)
            clean_mask = torch.ones_like(input_ids)
            enc_out = model.get_encoder()(input_ids=input_ids, attention_mask=clean_mask)
            # decoder forced clean prefix: [eos, lang] + target content tokens before span
            # target-side content tokens = same sentencepiece ids as source content
            first_span_pos = span_tok[0]
            prefix_content = [ids[i] for i in content if i < first_span_pos]
            dec_prefix = torch.tensor([[eos, lang_id] + prefix_content], device=DEV)
            n_sp = len(span_tok)
            budget = max(2, int(np.ceil(n_sp * 1.5)))  # pre-registered +50% slack
            gen = model.generate(
                encoder_outputs=enc_out,
                attention_mask=attn.to(DEV),
                decoder_input_ids=dec_prefix,
                num_beams=1,
                do_sample=False,
                max_new_tokens=budget,
            )
            cont_ids = gen[0][dec_prefix.shape[1]:].tolist()
            hit_eos = eos in cont_ids
            if hit_eos:
                cont_ids = cont_ids[:cont_ids.index(eos)]
            dec_span = tok.decode(cont_ids, skip_special_tokens=True).strip()
            if not hit_eos and " " in dec_span:
                dec_span = dec_span.rsplit(" ", 1)[0]  # cut at last word boundary
            corrupted = seed[:case["c0"]] + dec_span + seed[case["c1"]:]

            n_true, n_dec = normalize(case["span"]), normalize(dec_span)
            recon = bool(n_dec) and bool(n_true) and (n_true == n_dec or n_true in n_dec)
            degen = degenerate_span(dec_span)
            results.append({**case, "decoded_span": dec_span, "corrupted": corrupted,
                            "n_span_tok": n_sp, "recon": recon, "degen": degen,
                            "hit_eos": hit_eos})
            if (k + 1) % 100 == 0:
                log(f"  {k + 1}/{len(cases)} decoded ({time.time() - t0:.0f}s)")

    del model
    torch.cuda.empty_cache()
    log(f"decoded {len(results)} spans in {time.time() - t0:.0f}s")

    # -------- metrics
    n = len(results)
    recon_rate = sum(r["recon"] for r in results) / n
    degen_rate = sum(r["degen"] and not r["recon"] for r in results) / n
    drift = [r for r in results if not r["recon"] and not r["degen"]]
    drift_rate = len(drift) / n
    log(f"reconstruction {recon_rate:.3f}  degenerate {degen_rate:.3f}  usable-drift {drift_rate:.3f}")

    # NLI fwd (seed -> corrupted) on the usable drift band
    log(f"NLI fwd on {len(drift)} drift pairs ...")
    _, fwd_pe = S0.nli_entail([(r["seed"], r["corrupted"]) for r in drift])
    for r, p in zip(drift, fwd_pe):
        r["nli_fwd"] = float(p)
    fwd = np.array(fwd_pe) if fwd_pe else np.array([0.0])
    veto_share = float((fwd >= 0.8).mean()) if len(drift) else 0.0

    per_type = {}
    for lt in sorted({r["ltype"] for r in results}):
        sub = [r for r in results if r["ltype"] == lt]
        per_type[lt] = {
            "n": len(sub),
            "recon": round(sum(r["recon"] for r in sub) / len(sub), 3),
            "degen": round(sum(r["degen"] and not r["recon"] for r in sub) / len(sub), 3),
            "drift": round(sum(not r["recon"] and not r["degen"] for r in sub) / len(sub), 3),
        }

    kill = recon_rate >= 0.60 or degen_rate > 0.30
    verdict = "KILL" if kill else "SURVIVES-to-pilot"
    summary = {
        "n": n, "recon_rate": round(recon_rate, 4), "degen_rate": round(degen_rate, 4),
        "drift_rate": round(drift_rate, 4), "nli_fwd_mean": round(float(fwd.mean()), 4),
        "nli_fwd_ge08_share": round(veto_share, 4),
        "post_veto_yield": round(drift_rate * (1 - veto_share), 4),
        "per_type": per_type, "verdict": verdict,
        "bars": "KILL if recon >= 0.60 or degen > 0.30",
    }
    print(json.dumps(summary, indent=1))

    for r in results:
        r.setdefault("nli_fwd", None)
    pl.DataFrame(results).write_parquet(OUT_PARQUET)

    # -------- report with 20 eyeball examples
    lines = ["# DR-H114 XATTN-BLIND kill-gate", "", f"```json\n{json.dumps(summary, indent=1)}\n```",
             "", "## Eyeball - 20 blinded decodes across locus types", ""]
    rng = random.Random(1)
    by_type = defaultdict(list)
    for r in drift:
        by_type[r["ltype"]].append(r)
    picks = []
    while len(picks) < 20 and any(by_type.values()):
        for lt in list(by_type):
            if by_type[lt] and len(picks) < 20:
                picks.append(by_type[lt].pop(rng.randrange(len(by_type[lt]))))
    for i, r in enumerate(picks, 1):
        lines += [f"**X{i}** [{r['ltype']}] span=\"{r['span']}\" nli_fwd={r['nli_fwd']:.2f}",
                  f"- seed : {r['seed'][:200]}",
                  f"- blind: {r['corrupted'][:200]}",
                  f"- dec  : {r['decoded_span'][:120]}", ""]
    OUT_REPORT.write_text("\n".join(lines))
    log(f"report -> {OUT_REPORT}")
    print("=== DR-H114 GATE DONE ===")


if __name__ == "__main__":
    main()
