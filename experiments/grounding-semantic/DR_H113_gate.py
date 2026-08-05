"""DR-H113 TYPED-SWAP kill-gate (registration: docs/experiments/semantic-dataset-enhancements.md).

Deterministic rule surgery on typed factual loci - no neural generator. This is
the KILL-GATE only: ~1,500 samples stratified over 7 operator families, then
fluency gate (gpt2 NLL, H111 stage-0 thresholds), the two-question judge
(DR_H113_judge.py, vllm env), the still-entailed veto MEASUREMENT (nli_fwd on
judge-certified swaps), and the verdict vs the registered bars.

Binding amendments implemented:
- operators restricted to seed loci verified EVIDENCE-ENTAILED at construction
  (locus surface form present - numbers normalized - in the paired chunk)
- judge adjudicates vs the PAIRED EVIDENCE (supported yes/no), not only
  contrastively vs the seed (see DR_H113_judge.py)
- digit-typo edits banned (value edits only, >=5% relative)

Phases:
  uv run python DR_H113_gate.py --generate            # CPU, spaCy
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python DR_H113_gate.py --nll
  (judge: see DR_H113_judge.py, /home/lab/venvs/vllm/bin/python, GPU1)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python DR_H113_gate.py --nli --verdict
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re

import polars as pl

HERE = pathlib.Path(__file__).parent
PAIRS = HERE / "R10-H111_stage1_pairs.parquet"
SAMPLES = HERE / "DR_H113_gate_samples.parquet"
JUDGED = HERE / "DR_H113_gate_judged.parquet"
REPORT = HERE / "DR_H113_gate_report.md"
NLL_MAX = 6.234306645393371  # H111 stage-0 calibrated threshold (95th pct clean recon)

TARGET_PER_OP = 215
TOTAL_CAP = 1600
MAX_SEEDS = 40000

OPS = ["number", "date", "unit", "entity", "hedge", "comparative", "negation"]

DELETION_HEDGES = {
    "possibly", "perhaps", "probably", "likely", "presumably", "apparently",
    "seemingly", "arguably", "roughly", "approximately", "about", "around",
    "nearly", "almost", "generally", "typically", "usually", "often",
    "sometimes", "somewhat", "relatively", "partially", "largely", "mostly",
    "estimated",
}
STRENGTHEN = {
    "may": "will", "might": "will", "could": "will",
    "probably": "certainly", "likely": "certainly", "possibly": "certainly",
    "perhaps": "certainly",
    "approximately": "exactly", "roughly": "exactly", "about": "exactly",
    "around": "exactly", "nearly": "exactly", "almost": "exactly",
    "generally": "always", "typically": "always", "usually": "always",
    "often": "always", "sometimes": "always",
    "somewhat": "entirely", "relatively": "entirely", "partially": "entirely",
    "largely": "entirely", "mostly": "entirely",
    "suggests": "proves", "suggest": "prove",
    "indicates": "proves", "indicate": "prove", "seems": "is",
}
COMP_MAP = {
    "more": "less", "less": "more", "fewer": "more", "higher": "lower",
    "lower": "higher", "larger": "smaller", "smaller": "larger",
    "greater": "smaller", "faster": "slower", "slower": "faster",
    "before": "after", "after": "before", "above": "below", "below": "above",
    "most": "least", "least": "most", "increase": "decrease",
    "decrease": "increase", "increases": "decreases", "decreases": "increases",
    "increased": "decreased", "decreased": "increased", "earliest": "latest",
    "latest": "earliest", "oldest": "newest", "newest": "oldest",
    "best": "worst", "worst": "best", "maximum": "minimum",
    "minimum": "maximum", "first": "last", "last": "first",
}
COMP_PHRASES = [("at least", "at most"), ("at most", "at least"),
                ("more than", "less than"), ("less than", "more than")]
UNIT_DIMS = [
    ["mm", "cm", "km"], ["mg", "kg"], ["seconds", "minutes", "hours", "days",
    "weeks", "months", "years"], ["second", "minute", "hour", "day", "week",
    "month", "year"], ["KB", "MB", "GB", "TB"], ["Hz", "kHz", "MHz", "GHz"],
]
NUMDATE_ENTS = {"CARDINAL", "QUANTITY", "PERCENT", "MONEY", "ORDINAL"}
ENTITY_ENTS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
               "WORK_OF_ART", "FAC", "NORP", "LAW", "LANGUAGE"}
NEG_AUX = {"is", "are", "was", "were", "can", "will", "should"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.casefold()).replace("  ", " ").strip()


def norm_num(s: str) -> str:
    return s.replace(",", "")


def chunk_numbers(chunk: str) -> set[str]:
    return {norm_num(m.group(0)).rstrip(".") for m in NUM_RE.finditer(chunk)}


def fmt_like(value: float, template: str) -> str:
    """Format value using the surface conventions of template."""
    if "." in template:
        dec = len(template.split(".")[1])
        s = f"{value:.{dec}f}"
    else:
        s = str(int(round(value)))
    if "," in template:
        head, _, tail = s.partition(".")
        head = f"{int(head):,}"
        s = head + (("." + tail) if tail else "")
    return s


def word_in(text_cf: str, word: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(word.casefold())}(?![a-z0-9])",
                     text_cf) is not None


# ---------------- operators ----------------
# each returns (claim_new, span_start, span_end, old, new) or None


def op_number(doc, seed, chunk_cf, chunk_nums, rng):
    cands = [e for e in doc.ents if e.label_ in NUMDATE_ENTS
             and NUM_RE.search(e.text)]
    rng.shuffle(cands)
    for ent in cands:
        m = NUM_RE.search(ent.text)
        surface = m.group(0).rstrip(".")
        if norm_num(surface) not in chunk_nums:
            continue  # amendment: locus must be evidence-entailed
        try:
            val = float(norm_num(surface))
        except ValueError:
            continue
        if val == 0:
            continue
        for _ in range(6):
            d = pow(2.718281828, rng.uniform(-2.9957, -0.9163))  # 5%..40%
            newv = val * (1 + d) if rng.random() < 0.5 else val * (1 - d)
            new_surface = fmt_like(newv, surface)
            if new_surface != surface and norm_num(new_surface) not in chunk_nums:
                try:
                    if abs(float(norm_num(new_surface)) - val) / abs(val) >= 0.05:
                        break
                except ValueError:
                    continue
        else:
            continue
        start = ent.start_char + m.start()
        claim_new = seed[:start] + new_surface + seed[start + len(m.group(0)):]
        return claim_new, start, start + len(new_surface), surface, new_surface
    return None


def op_date(doc, seed, chunk_cf, chunk_nums, rng):
    cands = [e for e in doc.ents if e.label_ == "DATE"]
    rng.shuffle(cands)
    for ent in cands:
        ym = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", ent.text)
        if ym:
            year = ym.group(0)
            if year not in chunk_nums and not word_in(chunk_cf, year):
                continue
            shift = rng.choice([-3, -2, -1, 1, 2, 3])
            new = str(int(year) + shift)
            start = ent.start_char + ym.start()
            claim_new = seed[:start] + new + seed[start + 4:]
            return claim_new, start, start + len(new), year, new
        mm = re.search("|".join(MONTHS), ent.text)
        if mm:
            old = mm.group(0)
            if not word_in(chunk_cf, old):
                continue
            new = rng.choice([m for m in MONTHS if m != old])
            start = ent.start_char + mm.start()
            claim_new = seed[:start] + new + seed[start + len(old):]
            return claim_new, start, start + len(new), old, new
    return None


def op_unit(doc, seed, chunk_cf, chunk_nums, rng):
    for dim in UNIT_DIMS:
        for u in dim:
            m = re.search(rf"(\d[\d,\.]*)\s?({u})\b", seed)
            if not m:
                continue
            if not word_in(chunk_cf, m.group(2)):
                continue
            new_u = rng.choice([x for x in dim if x != u])
            start, end = m.start(2), m.end(2)
            claim_new = seed[:start] + new_u + seed[end:]
            return claim_new, start, start + len(new_u), u, new_u
    return None


def op_entity(doc, seed, chunk_cf, rng, chunk_doc, xpool, tag_reg, chunk_raw):
    cands = [e for e in doc.ents if e.label_ in ENTITY_ENTS
             and len(e.text) > 2]
    rng.shuffle(cands)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    pred = root.lemma_.casefold() if root is not None else ""
    for ent in cands:
        if ent.text.casefold() not in chunk_cf:
            continue  # amendment: evidence-entailed
        subs = [e.text for e in chunk_doc.ents
                if e.label_ == ent.label_ and e.text.casefold() != ent.text.casefold()
                and ent.text.casefold() not in e.text.casefold()
                and e.text.casefold() not in ent.text.casefold()]
        good = []
        for s in dict.fromkeys(subs):
            # predicate-co-occurrence rejection guard (same evidence doc)
            bad = False
            for sent in chunk_doc.sents:
                sl = sent.text.casefold()
                if s.casefold() in sl and pred and pred in sl:
                    bad = True
                    break
            if not bad:
                good.append(s)
        if not good:
            # cross-doc fallback pool, same register; must be absent from THIS chunk
            pool = [s for s in xpool.get((tag_reg, ent.label_), [])
                    if s.casefold() not in chunk_cf
                    and s.casefold() != ent.text.casefold()]
            good = pool[:20]
        if not good:
            continue
        new = rng.choice(good)
        claim_new = seed[:ent.start_char] + new + seed[ent.end_char:]
        return (claim_new, ent.start_char, ent.start_char + len(new),
                ent.text, new)
    return None


def op_hedge(doc, seed, chunk_cf, rng):
    toks = [t for t in doc if t.lower_ in DELETION_HEDGES.union(STRENGTHEN)]
    rng.shuffle(toks)
    for t in toks:
        if not word_in(chunk_cf, t.lower_):
            continue  # hedge itself must be evidence-entailed
        if t.lower_ in DELETION_HEDGES and rng.random() < 0.6:
            start, end = t.idx, t.idx + len(t.text)
            if end < len(seed) and seed[end] == " ":
                end += 1
            claim_new = seed[:start] + seed[end:]
            return claim_new, start, start, t.text, ""
        if t.lower_ in STRENGTHEN:
            new = STRENGTHEN[t.lower_]
            if t.text[0].isupper():
                new = new.capitalize()
            claim_new = seed[:t.idx] + new + seed[t.idx + len(t.text):]
            return claim_new, t.idx, t.idx + len(new), t.text, new
    return None


def op_comparative(doc, seed, chunk_cf, rng):
    scf = seed.casefold()
    phr = [(a, b) for a, b in COMP_PHRASES if a in scf and word_in(chunk_cf, a.split()[0])]
    if phr:
        a, b = rng.choice(phr)
        i = scf.find(a)
        old = seed[i:i + len(a)]
        new = b if old[0].islower() else b.capitalize()
        claim_new = seed[:i] + new + seed[i + len(a):]
        return claim_new, i, i + len(new), old, new
    toks = [t for t in doc if t.lower_ in COMP_MAP]
    rng.shuffle(toks)
    for t in toks:
        if not word_in(chunk_cf, t.lower_):
            continue
        new = COMP_MAP[t.lower_]
        if t.text[0].isupper():
            new = new.capitalize()
        claim_new = seed[:t.idx] + new + seed[t.idx + len(t.text):]
        return claim_new, t.idx, t.idx + len(new), t.text, new
    return None


def op_negation(doc, seed, chunk_cf, rng):
    # delete an existing negation
    negs = [t for t in doc if t.dep_ == "neg" or t.lower_ in ("not", "never")]
    rng.shuffle(negs)
    for t in negs:
        head = t.head
        if head is not None and head.lemma_ and not word_in(chunk_cf, head.lemma_.casefold()):
            continue
        if t.lower_ == "n't":
            start, end = t.idx, t.idx + len(t.text)
            claim_new = seed[:start] + seed[end:]
            return claim_new, start, start, t.text, ""
        start, end = t.idx, t.idx + len(t.text)
        if end < len(seed) and seed[end] == " ":
            end += 1
        claim_new = seed[:start] + seed[end:]
        return claim_new, start, start, t.text, ""
    # insert a negation after a bare auxiliary/copula
    for t in doc:
        if t.lower_ in NEG_AUX and t.i + 1 < len(doc) and doc[t.i + 1].lower_ != "not":
            if not word_in(chunk_cf, t.lower_):
                continue
            ins = t.idx + len(t.text)
            claim_new = seed[:ins] + " not" + seed[ins:]
            return claim_new, ins, ins + 4, "", "not"
    return None


# ---------------- generation ----------------


def generate():
    import spacy
    rng = random.Random(0)
    nlp = spacy.load("en_core_web_lg")
    df = (pl.read_parquet(PAIRS).unique(subset=["seed"], keep="first")
          .select("seed", "chunk", "tag").sample(fraction=1.0, seed=0))
    rows = df.head(MAX_SEEDS).to_dicts()
    print(f"seed pool: {len(rows)}", flush=True)

    counts = {op: 0 for op in OPS}
    out = []
    chunk_cache: dict[str, object] = {}
    xpool: dict[tuple, list] = {}

    seeds_txt = [r["seed"] for r in rows]
    for i, doc in enumerate(nlp.pipe(seeds_txt, batch_size=128)):
        r = rows[i]
        seed, chunk, tag = r["seed"], r["chunk"], r["tag"]
        if len(seed) < 30 or len(seed) > 600:
            continue
        tag_reg = tag.rsplit("_", 1)[-1]
        chunk_cf = chunk.casefold()
        chunk_nums = chunk_numbers(chunk)

        applicable = []
        if any(e.label_ in NUMDATE_ENTS for e in doc.ents):
            applicable.append("number")
        if any(e.label_ == "DATE" for e in doc.ents):
            applicable.append("date")
        if any(re.search(rf"\d[\d,\.]*\s?{u}\b", seed) for dim in UNIT_DIMS for u in dim):
            applicable.append("unit")
        if any(e.label_ in ENTITY_ENTS for e in doc.ents):
            applicable.append("entity")
        if any(t.lower_ in DELETION_HEDGES or t.lower_ in STRENGTHEN for t in doc):
            applicable.append("hedge")
        if any(t.lower_ in COMP_MAP for t in doc) or any(
                a in seed.casefold() for a, _ in COMP_PHRASES):
            applicable.append("comparative")
        if any(t.dep_ == "neg" or t.lower_ in ("not", "never") for t in doc) or any(
                t.lower_ in NEG_AUX for t in doc):
            applicable.append("negation")
        applicable = [op for op in applicable if counts[op] < TARGET_PER_OP]
        if not applicable:
            if all(c >= TARGET_PER_OP for c in counts.values()):
                break
            continue
        applicable.sort(key=lambda op: counts[op])
        chosen = applicable[0]

        res = None
        if chosen == "entity":
            if chunk[:2000] not in chunk_cache:
                if len(chunk_cache) > 3000:
                    chunk_cache.clear()
                cd = nlp(chunk[:2000])
                chunk_cache[chunk[:2000]] = cd
                for e in cd.ents:
                    if e.label_ in ENTITY_ENTS:
                        xpool.setdefault((tag_reg, e.label_), []).append(e.text)
            res = op_entity(doc, seed, chunk_cf, rng,
                            chunk_cache[chunk[:2000]], xpool, tag_reg, chunk)
        elif chosen == "number":
            res = op_number(doc, seed, chunk_cf, chunk_nums, rng)
        elif chosen == "date":
            res = op_date(doc, seed, chunk_cf, chunk_nums, rng)
        elif chosen == "unit":
            res = op_unit(doc, seed, chunk_cf, chunk_nums, rng)
        elif chosen == "hedge":
            res = op_hedge(doc, seed, chunk_cf, rng)
        elif chosen == "comparative":
            res = op_comparative(doc, seed, chunk_cf, rng)
        elif chosen == "negation":
            res = op_negation(doc, seed, chunk_cf, rng)
        if res is None:
            continue
        claim_new, s0, s1, old, new = res
        if norm(claim_new) == norm(seed):
            continue
        counts[chosen] += 1
        out.append({"seed": seed, "chunk": chunk, "tag": tag, "op": chosen,
                    "claim": claim_new, "span_start": s0, "span_end": s1,
                    "old": old, "new": new})
        if len(out) >= TOTAL_CAP:
            break
        if len(out) % 100 == 0:
            print(f"  {len(out)} samples  {counts}", flush=True)

    pl.DataFrame(out).write_parquet(SAMPLES)
    print(f"\nDONE generate: {len(out)} samples -> {SAMPLES}", flush=True)
    print(json.dumps(counts, indent=1), flush=True)


def add_nll():
    import sys
    sys.path.insert(0, str(HERE))
    S0 = __import__("R10-H111_stage0".replace("-", "_"), fromlist=["x"]) \
        if False else None
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s0", HERE / "R10-H111_stage0.py")
    s0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s0)
    df = pl.read_parquet(SAMPLES)
    nll = s0.gpt2_nll(df["claim"].to_list())
    df = df.with_columns(pl.Series("nll", [float(x) for x in nll]),
                         pl.Series("fluent", [float(x) <= NLL_MAX for x in nll]))
    df.write_parquet(SAMPLES)
    ok = df["fluent"].sum()
    print(f"DONE nll: fluent {ok}/{len(df)} ({ok/len(df):.3f})  bar>=0.90", flush=True)


def add_nli_and_verdict(do_nli: bool, do_verdict: bool):
    df = pl.read_parquet(JUDGED)
    factual = ["entity-swap", "number-change", "hedge-deletion", "omission",
               "negation", "other-factual"]
    df = df.with_columns(
        (pl.col("delta").is_in(factual) & (pl.col("supported") == "no"))
        .alias("agreed"))
    if do_nli and "nli_fwd" not in df.columns:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "s0", HERE / "R10-H111_stage0.py")
        s0 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s0)
        _, fwd = s0.nli_entail(list(zip(df["seed"].to_list(),
                                        df["claim"].to_list())))
        df = df.with_columns(pl.Series("nli_fwd", [float(x) for x in fwd]))
        df.write_parquet(JUDGED)
    if not do_verdict:
        return

    lines = ["# DR-H113 TYPED-SWAP kill-gate report\n"]
    stats = []
    for op in OPS:
        sub = df.filter(pl.col("op") == op)
        if not len(sub):
            stats.append((op, 0, 0, 0, 0, 0, 0))
            continue
        n = len(sub)
        agree = sub["agreed"].sum() / n
        # regrounding: judge said changed content but it appears in evidence
        reg = 0
        for r in sub.iter_rows(named=True):
            nc = norm(r.get("changed") or "")
            if r["agreed"] and nc and len(nc) > 3 and nc in norm(r["chunk"]):
                reg += 1
        reg_rate = reg / n
        fl = sub["fluent"].sum() / n
        sub_a = sub.filter(pl.col("agreed"))
        subtle = (sub_a.filter(pl.col("severity") == "subtle").height /
                  max(1, sub_a.height))
        veto = (sub_a.filter(pl.col("nli_fwd") >= 0.8).height /
                max(1, sub_a.height))
        stats.append((op, n, agree, reg_rate, fl, subtle, veto))

    lines.append("| op | n | judge agree | regrounding | fluent | subtle (of agreed) | still-entailed veto would kill |")
    lines.append("|---|---|---|---|---|---|---|")
    survivors = []
    for op, n, agree, reg, fl, subtle, veto in stats:
        verdict = "SURVIVES" if (n and agree >= 0.60 and reg <= 0.05) else "KILLED"
        if n and agree >= 0.60 and reg <= 0.05:
            survivors.append(op)
        lines.append(f"| {op} | {n} | {agree:.3f} | {reg:.3f} | {fl:.3f} | "
                     f"{subtle:.3f} | {veto:.3f} | {verdict}")
    dfa = df.filter(pl.col("agreed"))
    pooled = dfa.height / len(df)
    fluent_all = df["fluent"].sum() / len(df)
    subtle_all = (dfa.filter(pl.col("severity") == "subtle").height /
                  max(1, dfa.height))
    veto_all = (dfa.filter(pl.col("nli_fwd") >= 0.8).height / max(1, dfa.height))
    reg_all = sum(1 for r in df.iter_rows(named=True)
                  if r["agreed"] and (nc := norm(r.get("changed") or ""))
                  and len(nc) > 3 and nc in norm(r["chunk"])) / len(df)
    lines += [
        "",
        f"pooled judge agreement: {pooled:.3f} (PASS bar >=0.75, hypothesis-kill <0.60)",
        f"pooled regrounding: {reg_all:.3f} (bar <=0.05)",
        f"fluency pass: {fluent_all:.3f} (bar >=0.90)",
        f"subtle share of agreed: {subtle_all:.3f} (bar >=0.30)",
        f"STILL-ENTAILED VETO measurement: {veto_all:.3f} of judge-certified true "
        f"negatives would be executed by nli_fwd>=0.8 (campaign-wide number)",
        f"surviving operator types: {survivors} ({len(survivors)}/7; hypothesis "
        f"killed if <3)",
        "",
        "## Example swaps per operator (5 each, for the main-session eyeball)",
    ]
    for op in OPS:
        sub = df.filter(pl.col("op") == op).head(5)
        lines.append(f"\n### {op}")
        for r in sub.iter_rows(named=True):
            lines += [f"- old→new: `{r['old']}` → `{r['new']}` | delta={r['delta']} "
                      f"sev={r['severity']} supported={r['supported']} "
                      f"agreed={r['agreed']} nli_fwd={r.get('nli_fwd', float('nan')):.2f}",
                      f"  - seed : {r['seed'][:180]}",
                      f"  - claim: {r['claim'][:180]}"]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines[:25]), flush=True)
    print(f"\nDONE verdict -> {REPORT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--nll", action="store_true")
    ap.add_argument("--nli", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.generate:
        generate()
    if a.nll:
        add_nll()
    if a.nli or a.verdict:
        add_nli_and_verdict(a.nli, a.verdict)
