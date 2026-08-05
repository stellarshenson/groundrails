"""DR-H112 SPAN-INFILL-BAN kill-gate (registration: docs/experiments/semantic-dataset-enhancements.md).

Arm (a) - infilling-head precondition: 200 seeds, span -> <mask> in the ENCODER
input (mBART-50 text-infilling format) + co-mention occlusion; decoder forced
clean prefix, GREEDY free span (no ban), budget span_len+4, suffix-bigram stop,
word-boundary cut, text splice. KILL THE FAMILY if degeneracy >= 15%.

Arm (b) - banned sampling: 1,500 infills over 1,000 quota-typed seeds, top_p 0.9
temp 0.9, ban list = true span tokenizations (raw / lower / leading-space /
digit+spelled number forms) + <mask> + early-EOS ban (min_new_tokens). Measures
degeneracy, morphological ban-evasion (normalized match OR Levenshtein <= 2),
exact reproduction (regrounding proxy), bidirectional NLI, generic-filler share,
and the binding-amendment bypass eligibility for number/date spans.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/DR_H112_gate.py
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
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


S0 = _load_mod("R10-H111_stage0")
import DR_targeting as DRT

DEV = "cuda"
SEED = 0
ARM_A_N = 200
ARM_B_TARGET = 1500
ARM_B_SEEDS = 1000
PAIRS = HERE / "R10-H111_stage1_pairs.parquet"
TH = json.loads((HERE / "R10-H111_stage1_progress.json").read_text())["thresholds"]

OUT_PARQUET = HERE / "DR_H112_gate_samples.parquet"
OUT_JUDGE_SAMPLE = HERE / "DR_H112_gate_judge_sample.parquet"
OUT_SUMMARY = HERE / "DR_H112_gate_summary.json"

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
_DIGIT_TO_SPELLED = {v: k for k, v in _SPELLED.items() if k not in ("first", "second", "third")}

BOUND_QUALIFIERS = [
    "at least", "at most", "up to", "over", "under", "more than", "less than",
    "approximately", "approx", "roughly", "about", "nearly", "around", "~",
]

FILLER_PHRASES = {
    "the following", "a number of", "various", "a variety of", "some",
    "several", "a lot of", "the same", "a new", "the first", "the most",
    "in general", "and so on", "etc", "different", "the other", "a certain",
    "the main", "important", "significant", "numerous", "many", "a few",
    "the best", "the largest", "general", "common", "typical", "standard",
    "a", "the", "this", "that", "it", "one",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize(text: str) -> str:
    t = text.casefold()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t)
    words = [_SPELLED.get(w, w) for w in words]
    return " ".join(words)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


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
        for x, y in zip(toks, toks[1:]):
            run = run + 1 if x == y else 1
            if run > TH["maxrun_max"]:
                return True
    crun, best = 1, 1
    for x, y in zip(span_text, span_text[1:]):
        crun = crun + 1 if x == y and not x.isspace() else 1
        best = max(best, crun)
    if best > TH["charrun_max"]:
        return True
    sym = sum(1 for c in span_text if not (c.isalnum() or c in " .,;:'\"()-%$/"))
    if sym / max(len(span_text), 1) > TH["symdens_max"]:
        return True
    return False


def is_filler(dec_norm: str) -> bool:
    if not dec_norm:
        return False
    if dec_norm in FILLER_PHRASES:
        return True
    return all(w in FILLER_PHRASES for w in dec_norm.split())


def span_variants(span_text: str) -> list[str]:
    """Ban-list surface variants: raw, lower, digit and spelled number forms."""
    out = {span_text, span_text.lower(), span_text.strip()}
    stripped = re.sub(r"(?<=\d),(?=\d{3}\b)", "", span_text)
    out.add(stripped)
    for w in re.findall(r"[A-Za-z]+|\d[\d,\.]*", span_text):
        lw = w.lower()
        if lw in _SPELLED:
            out.add(span_text.lower().replace(lw, _SPELLED[lw]))
        wd = re.sub(r",", "", w)
        if wd.isdigit() and wd in _DIGIT_TO_SPELLED:
            out.add(span_text.lower().replace(w, _DIGIT_TO_SPELLED[wd]))
    return [v for v in out if v.strip()]


def bad_words_for(tok, span_text: str, mask_id: int) -> list[list[int]]:
    seqs = []
    for v in span_variants(span_text):
        for form in (v, " " + v):
            ids = tok(form, add_special_tokens=False).input_ids
            if ids:
                seqs.append(ids)
    seqs.append([mask_id])
    # dedup
    seen, out = set(), []
    for s in seqs:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            out.append(s)
    return out


def bound_qualified(seed: str, c0: int) -> bool:
    prefix_words = seed[:c0].casefold().split()[-5:]
    window = " ".join(prefix_words)
    return any(q in window for q in BOUND_QUALIFIERS)


class SuffixBigramStop:
    """Stop when the last two generated ids equal the suffix's first two ids."""

    def __init__(self, suffix_ids, prefix_len):
        self.suf = tuple(suffix_ids[:2])
        self.prefix_len = prefix_len

    def __call__(self, input_ids, scores, **kw):
        if len(self.suf) < 2:
            return False
        gen = input_ids[0][self.prefix_len:].tolist()
        return len(gen) >= 2 and tuple(gen[-2:]) == self.suf


def select_cases(seeds, n_target, spans_per_seed, rng):
    cases = []
    for seed in seeds:
        if len(cases) >= n_target:
            break
        try:
            spans = DRT.sample_spans(seed, spans_per_seed,
                                     rng=random.Random(hash(seed) % 2**31))
        except Exception:
            continue
        for c0, c1, s_text, ltype, source in spans:
            if not s_text.strip() or c1 <= c0:
                continue
            cases.append({"seed": seed, "c0": c0, "c1": c1, "span": s_text,
                          "ltype": ltype, "source": source})
    return cases[:n_target]


def run_arm(tok, model, cases, banned: bool, arm_name: str):
    from transformers import StoppingCriteriaList, StoppingCriteria

    class _Stop(StoppingCriteria):
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, input_ids, scores, **kw):
            return self.fn(input_ids, scores)

    lang_id = tok.lang_code_to_id["en_XX"]
    eos = tok.eos_token_id
    mask_id = tok.mask_token_id
    results = []
    t0 = time.time()
    with torch.no_grad():
        for k, case in enumerate(cases):
            seed = case["seed"]
            c0, c1 = case["c0"], case["c1"]
            span_text = case["span"]
            # ---- masked encoder input + co-mention occlusion
            masked = seed[:c0] + tok.mask_token + seed[c1:]
            pat = re.compile(re.escape(span_text), re.IGNORECASE)
            head, tail = masked[:c0 + len(tok.mask_token)], masked[c0 + len(tok.mask_token):]
            tail = pat.sub(tok.mask_token, tail)
            head_pre = pat.sub(tok.mask_token, head[:c0])
            masked = head_pre + head[c0:] + tail
            enc_in = tok(masked, return_tensors="pt", truncation=True,
                         max_length=140).to(DEV)
            # ---- decoder prefix / budget from CLEAN seed tokenization
            enc_clean = tok(seed, return_tensors="pt", truncation=True,
                            max_length=128, return_offsets_mapping=True)
            offsets = enc_clean.pop("offset_mapping")[0].tolist()
            ids = enc_clean["input_ids"][0].tolist()
            content = [i for i in range(len(ids)) if offsets[i] != [0, 0]]
            span_tok = [i for i in content
                        if offsets[i][1] > c0 and offsets[i][0] < c1]
            if not span_tok or len(span_tok) >= len(content):
                continue
            first_span = span_tok[0]
            prefix_content = [ids[i] for i in content if i < first_span]
            suffix_content = [ids[i] for i in content if i > span_tok[-1]]
            dec_prefix = torch.tensor([[eos, lang_id] + prefix_content], device=DEV)
            n_sp = len(span_tok)
            budget = n_sp + 4
            stop = StoppingCriteriaList(
                [_Stop(SuffixBigramStop(suffix_content, dec_prefix.shape[1]))])
            gen_kw = dict(
                input_ids=enc_in["input_ids"],
                attention_mask=enc_in["attention_mask"],
                decoder_input_ids=dec_prefix,
                max_new_tokens=budget,
                min_new_tokens=min(2, budget),
                stopping_criteria=stop,
            )
            if banned:
                gen_kw.update(do_sample=True, top_p=0.9, temperature=0.9,
                              bad_words_ids=bad_words_for(tok, span_text, mask_id))
            else:
                gen_kw.update(num_beams=1, do_sample=False)
            try:
                gen = model.generate(**gen_kw)
            except Exception as e:
                log(f"  generate error at {k}: {e}")
                continue
            cont_ids = gen[0][dec_prefix.shape[1]:].tolist()
            hit_eos = eos in cont_ids
            if hit_eos:
                cont_ids = cont_ids[:cont_ids.index(eos)]
            # strip a trailing suffix bigram if the stopper fired
            if len(cont_ids) >= 2 and tuple(cont_ids[-2:]) == tuple(suffix_content[:2]):
                cont_ids = cont_ids[:-2]
            dec_span = tok.decode(cont_ids, skip_special_tokens=True).strip()
            if not hit_eos and " " in dec_span:
                dec_span = dec_span.rsplit(" ", 1)[0]
            corrupted = seed[:c0] + dec_span + seed[c1:]
            n_true, n_dec = normalize(span_text), normalize(dec_span)
            exact_repro = bool(n_dec) and n_true == n_dec
            evasion = bool(n_dec) and (
                n_true == n_dec or n_true in n_dec or levenshtein(n_true, n_dec) <= 2)
            results.append({**case, "arm": arm_name, "decoded_span": dec_span,
                            "corrupted": corrupted, "n_span_tok": n_sp,
                            "degen": degenerate_span(dec_span),
                            "exact_repro": exact_repro, "evasion": evasion,
                            "filler": is_filler(n_dec), "hit_eos": hit_eos,
                            "bound_q": bound_qualified(seed, c0)})
            if (k + 1) % 100 == 0:
                log(f"  [{arm_name}] {k + 1}/{len(cases)} ({time.time() - t0:.0f}s)")
    log(f"[{arm_name}] decoded {len(results)} in {time.time() - t0:.0f}s")
    return results


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    seeds = (pl.read_parquet(PAIRS).select("seed")
             .unique(maintain_order=True)["seed"].to_list())
    random.Random(SEED).shuffle(seeds)
    log(f"{len(seeds)} distinct seeds; arm-a and arm-b pools disjoint")
    arm_a_seed_pool = seeds[:400]
    arm_b_seed_pool = seeds[400:400 + ARM_B_SEEDS * 2]

    rng = random.Random(SEED)
    cases_a = select_cases(arm_a_seed_pool, ARM_A_N, 1, rng)
    cases_b = select_cases(arm_b_seed_pool, ARM_B_TARGET, 2, rng)
    log(f"arm-a {len(cases_a)} spans, arm-b {len(cases_b)} spans")

    tok, model = S0.load_mbart()
    model.eval()

    # ---------------- arm (a): infilling-head precondition, greedy, NO ban
    res_a = run_arm(tok, model, cases_a, banned=False, arm_name="a")
    degen_a = sum(r["degen"] for r in res_a) / max(len(res_a), 1)
    # differ-rate for the pre-registered bart-large fallback note
    differ_a = sum(1 for r in res_a
                   if normalize(r["decoded_span"]) != normalize(r["span"])) / max(len(res_a), 1)
    log(f"ARM-A degeneracy {degen_a:.3f} (family KILL bar >= 0.15); "
        f"differ-rate {differ_a:.3f}")
    family_kill = degen_a >= 0.15

    # ---------------- arm (b): banned sampling (runs regardless, for the record,
    # unless the family is dead - then we skip the spend)
    if family_kill:
        res_b = []
        log("family KILLED at arm (a) - skipping arm (b)")
    else:
        res_b = run_arm(tok, model, cases_b, banned=True, arm_name="b")

    del model
    torch.cuda.empty_cache()

    summary = {"arm_a": {"n": len(res_a), "degen": round(degen_a, 4),
                         "differ_rate": round(differ_a, 4),
                         "family_kill": family_kill}}

    if res_b:
        n = len(res_b)
        garbage = sum(r["degen"] for r in res_b) / n
        evasion = sum(r["evasion"] and not r["degen"] for r in res_b) / n
        exact = sum(r["exact_repro"] for r in res_b) / n
        usable = [r for r in res_b if not r["degen"] and not r["evasion"]]
        filler = (sum(r["filler"] for r in usable) / max(len(usable), 1))
        log(f"ARM-B garbage {garbage:.3f} evasion {evasion:.3f} "
            f"exact-repro {exact:.3f} filler(usable) {filler:.3f} "
            f"usable {len(usable)}/{n}")
        # bidirectional NLI on the usable band
        log(f"NLI on {len(usable)} usable pairs ...")
        _, fwd = S0.nli_entail([(r["seed"], r["corrupted"]) for r in usable])
        _, bwd = S0.nli_entail([(r["corrupted"], r["seed"]) for r in usable])
        for r, f, b in zip(usable, fwd, bwd):
            r["nli_fwd"], r["nli_bwd"] = float(f), float(b)
        fwd_arr = np.array(fwd) if fwd else np.array([0.0])
        per_type = {}
        for lt in sorted({r["ltype"] for r in res_b}):
            sub = [r for r in res_b if r["ltype"] == lt]
            usab = [r for r in sub if not r["degen"] and not r["evasion"]]
            per_type[lt] = {"n": len(sub),
                            "degen": round(sum(r["degen"] for r in sub) / len(sub), 3),
                            "evasion": round(sum(r["evasion"] and not r["degen"] for r in sub) / len(sub), 3),
                            "usable": round(len(usab) / len(sub), 3)}
        # binding amendment: bypass eligibility among usable number/date
        nd = [r for r in usable if r["ltype"] == "number_date"]
        bypass_eligible = [r for r in nd
                           if normalize(r["span"]) != normalize(r["decoded_span"])
                           and not r["bound_q"]]
        summary["arm_b"] = {
            "n": n, "garbage": round(garbage, 4), "evasion": round(evasion, 4),
            "exact_repro": round(exact, 4), "filler_usable": round(filler, 4),
            "usable": len(usable), "usable_rate": round(len(usable) / n, 4),
            "nli_fwd_mean": round(float(fwd_arr.mean()), 4),
            "nli_fwd_ge08_share": round(float((fwd_arr >= 0.8).mean()), 4),
            "per_type": per_type,
            "number_date_usable": len(nd),
            "bypass_eligible": len(bypass_eligible),
            "bound_qualified_denied": sum(1 for r in nd if r["bound_q"]),
        }
        # ---------------- judge sample: stratified + bypass oversample
        jrng = random.Random(1)
        by_type = defaultdict(list)
        for r in usable:
            by_type[r["ltype"]].append(r)
        picks, seen = [], set()

        def take(pool, k):
            jrng.shuffle(pool)
            for r in pool:
                if len(picks) >= 300:
                    return
                key = id(r)
                if key not in seen:
                    seen.add(key)
                    picks.append(r)
                    k -= 1
                    if k <= 0:
                        return

        take(list(bypass_eligible), min(len(bypass_eligible), 80))
        per_quota = max(1, (300 - len(picks)) // max(len(by_type), 1))
        for lt in sorted(by_type):
            take(list(by_type[lt]), per_quota)
        rest = [r for r in usable if id(r) not in seen]
        take(rest, 300 - len(picks))
        for r in res_b:
            r.setdefault("nli_fwd", None)
            r.setdefault("nli_bwd", None)
        pl.DataFrame([{**r, "bypass_eligible": (r in bypass_eligible)}
                      for r in picks],
                     infer_schema_length=None).write_parquet(OUT_JUDGE_SAMPLE)
        log(f"judge sample: {len(picks)} rows -> {OUT_JUDGE_SAMPLE}")

    all_rows = res_a + res_b
    for r in all_rows:
        r.setdefault("nli_fwd", None)
        r.setdefault("nli_bwd", None)
    pl.DataFrame(all_rows, infer_schema_length=None).write_parquet(OUT_PARQUET)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("=== DR-H112 GEN DONE ===", flush=True)
    if family_kill:
        print("=== DR-H112 GATE DONE ===", flush=True)


if __name__ == "__main__":
    main()
