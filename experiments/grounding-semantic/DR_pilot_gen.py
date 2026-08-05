"""DR-2 pilot-scale generation (registration: docs/experiments/semantic-dataset-enhancements.md).

Generates the DR lane's raw corruption stream from the three surviving engines at
the registered volumes (+25% slack over the measured gate yields), runs the
deterministic degeneracy gates and bidirectional NLI, and checkpoints to parquet.
The contrastive judge pass is NOT here - it runs later on GPU1.

Volumes (registered)
  H112 SPAN-INFILL-BAN  ~31,000 spans, core loci ONLY
                        (number_date / negation / entity / positional)
  H113 TYPED-SWAP       ~5,600 swaps over the four surviving operators at famine
                        proportions number 45 / negation 20 / comparative 20 /
                        unit 15, evidence-entailed loci, NLL fluency gate
  H114 XATTN-BLIND      ~23,000 decodes, negation + number_date loci 2x
                        overweight, seam cleaner BEFORE the degeneracy gates

Long-form: if DR_H116_subgate_result.json says SURVIVES, up to 20% of the H112
budget is delivered as 256-2048-token pysbd-spliced documents carrying the exact
char-offset ledger; otherwise the whole H112 budget is sentence-level.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/DR_pilot_gen.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import polars as pl
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import DR_pilot_engines as ENG  # noqa: E402
import DR_H113_gate as H113G  # noqa: E402
import DR_targeting as DRT  # noqa: E402

PAIRS = HERE / "R10-H111_stage1_pairs.parquet"
SUBGATE = HERE / "DR_H116_subgate_result.json"
OUT = HERE / "DR_pilot_raw.parquet"
CKPT = HERE / "DR_pilot_raw.parquet.ckpt"
SUMMARY = HERE / "DR_pilot_gen_summary.json"

SEED = 0
CKPT_EVERY = 2000
NLL_MAX = H113G.NLL_MAX  # H111 stage-0 calibrated fluency threshold

N_H112 = 31_000
N_H113 = 5_600
N_H114 = 23_000
LONGFORM_SHARE = 0.20  # of the H112 budget, only if the H116 sub-gate survives
H113_MIX = {"number": 0.45, "negation": 0.20, "comparative": 0.20, "unit": 0.15}
H113_LOCUS = {"number": "number_date", "unit": "number_date",
              "negation": "negation", "comparative": "comparative"}
OVERWEIGHT_LOCI = {"negation", "number_date"}  # H114 gate-eyeball amendment

log = ENG.log

ROW_KEYS = [
    "engine", "seed_id", "tag", "register", "seed", "chunk", "claim",
    "span_start", "span_end", "orig_span_start", "orig_span_end",
    "orig_span", "new_span", "locus_type", "operator", "source",
    "degen", "evasion", "exact_repro", "recon", "degen_postseam",
    "raw_new_span", "filler", "hit_eos",
    "bound_q", "seam_cleaned", "nll", "fluent", "degeneracy_pass", "usable",
    "nli_fwd", "nli_bwd", "long_form", "doc_id", "doc_clean", "doc_corrupt",
    "doc_span_start", "doc_span_end", "doc_sent_index", "doc_sent_start",
    "doc_sent_end",
]
ROW_DEFAULTS = {
    "operator": "", "source": "", "degen": False, "evasion": False,
    "exact_repro": False, "recon": False, "degen_postseam": False,
    "raw_new_span": None, "filler": False, "hit_eos": False,
    "bound_q": False, "seam_cleaned": False, "nll": None, "fluent": None,
    "nli_fwd": None, "nli_bwd": None, "long_form": False, "doc_id": None,
    "doc_clean": None, "doc_corrupt": None, "doc_span_start": None,
    "doc_span_end": None, "doc_sent_index": None, "doc_sent_start": None,
    "doc_sent_end": None, "usable": True, "degeneracy_pass": True,
}


class Sink:
    """Row accumulator with incremental parquet checkpointing."""

    def __init__(self):
        self.rows: list[dict] = []
        self._last = 0

    def add(self, **kw):
        row = {**ROW_DEFAULTS, **kw}
        self.rows.append({k: row.get(k) for k in ROW_KEYS})
        if len(self.rows) - self._last >= CKPT_EVERY:
            self.flush()

    def flush(self):
        self._last = len(self.rows)
        pl.DataFrame(self.rows, infer_schema_length=None).write_parquet(CKPT)
        log(f"  ckpt {len(self.rows)} rows -> {CKPT.name}")


# ----------------------------------------------------------------- seed pool


def load_pool():
    df = (pl.read_parquet(PAIRS).select("seed", "chunk", "tag")
          .unique(subset=["seed"], maintain_order=True))
    seeds = df["seed"].to_list()
    chunks = df["chunk"].to_list()
    tags = df["tag"].to_list()
    order = list(range(len(seeds)))
    random.Random(SEED).shuffle(order)
    return seeds, chunks, tags, order


# ------------------------------------------------------------ H112 sentence


def gen_h112(tok, model, sink, seeds, chunks, tags, ids, target):
    made = 0
    t0 = time.time()
    for sid in ids:
        if made >= target:
            break
        seed = seeds[sid]
        for c0, c1, stext, ltype, source in ENG.core_spans(
                seed, 2, random.Random(sid * 7919 + 1)):
            if made >= target:
                break
            res = ENG.infill_span(tok, model, seed, c0, c1, stext)
            if res is None:
                continue
            sink.add(engine="H112", seed_id=sid, tag=tags[sid],
                     register=tags[sid].rsplit("_", 1)[-1], seed=seed,
                     chunk=chunks[sid], claim=res["claim"], span_start=c0,
                     span_end=c0 + len(res["decoded_span"]),
                     orig_span_start=c0, orig_span_end=c1, orig_span=stext,
                     new_span=res["decoded_span"], locus_type=ltype,
                     source=source, degen=res["degen"],
                     evasion=res["evasion"], exact_repro=res["exact_repro"],
                     filler=res["filler"], hit_eos=res["hit_eos"],
                     bound_q=res["bound_q"],
                     degeneracy_pass=not res["degen"],
                     usable=not (res["degen"] or res["evasion"]))
            made += 1
            if made % 2500 == 0:
                log(f"  H112 {made}/{target} ({made / max(time.time() - t0, 1):.1f}/s)")
    log(f"H112 sentence-level: {made} spans in {time.time() - t0:.0f}s")
    return made


# ------------------------------------------------------------ H112 long-form


def gen_h112_longform(tok, model, sink, seeds, chunks, tags, target, ids=None,
                      skip_keys=None, spans_per_doc=1.6):
    """DR-H116 wrapper on the H112 engine: pysbd docs, char-exact splice, ledger.

    `ids` restricts doc assembly to a seed slice (the H112 slice, so the H113 /
    H114 slices stay untouched). `skip_keys` drops spans whose
    seed_id|span_start|new_span key already exists in a previously written lane.
    """
    import pysbd

    rng = random.Random(202)
    hist = ENG.span_count_hist()
    ks, kw = list(hist), [hist[k] for k in hist]
    lens = ENG.precount_tokens(tok, seeds)
    order = list(ids) if ids is not None else list(range(len(seeds)))
    random.Random(4242).shuffle(order)
    skip_keys = skip_keys or set()
    n_skipped = 0
    pools: dict[str, list[tuple]] = {}
    for i in order:
        pools.setdefault(tags[i].rsplit("_", 1)[-1], []).append(
            (seeds[i], i, lens[i]))
    n_docs = max(1, int(math.ceil(target / spans_per_doc)))
    docs = ENG.assemble_docs(pools, rng, n_docs, allow_cycle=True)
    log(f"long-form: {len(docs)} docs assembled (target {target} spans)")

    seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
    made, n_doc_done, intact_fail = 0, 0, 0
    t0 = time.time()
    for d in docs:
        if made >= target:
            break
        text = d["text"]
        sents = [(s.start, s.end) for s in seg.segment(text)
                 if text[s.start:s.end].strip()]
        if len(sents) < 2:
            continue
        k = min(int(rng.choices(ks, weights=kw)[0]),
                max(1, math.ceil(len(sents) / 3)))
        edits = []
        for si in ENG.pick_sentences(sents, len(text), k, rng):
            s0, s1 = sents[si]
            sent = text[s0:s1]
            # the seed that contributed this sentence carries the evidence link
            sid, best = -1, 0
            for a, b, cid in d["components"]:
                ov = min(b, s1) - max(a, s0)
                if ov > best:
                    best, sid = ov, cid
            # the cross-lane dedup check MUST happen before the edit reaches the
            # doc: a span dropped after splicing would leave an unledgered
            # corruption in doc_corrupt and break the H116 char-exact guarantee
            for _attempt in range(3):
                got = ENG.core_spans(sent, 1, random.Random(rng.getrandbits(31)))
                if not got:
                    break
                c0, c1, stext, ltype, source = got[0]
                res = ENG.infill_span(tok, model, sent, c0, c1, stext)
                if res is None:
                    continue
                if f"{sid}|{c0}|{res['decoded_span']}" in skip_keys:
                    n_skipped += 1
                    continue  # resample; this edit never touches the doc
                edits.append({
                    "abs_start": s0 + c0, "abs_end": s0 + c1,
                    "new_span": res["decoded_span"], "orig_span": stext,
                    "sent_index": si, "sent_start": s0, "sent_end": s1,
                    "sent_c0": c0, "sent_c1": c1, "sent_claim": res["claim"],
                    "locus_type": ltype, "source": source, "seed_id": sid,
                    "degen": res["degen"], "evasion": res["evasion"],
                    "exact_repro": res["exact_repro"], "filler": res["filler"],
                    "hit_eos": res["hit_eos"], "bound_q": res["bound_q"],
                })
                break
        if not edits:
            continue
        corrupt, ledger, intact = ENG.splice_doc(text, edits)
        if not intact:
            intact_fail += 1
            continue  # never ship a doc whose splice is not char-exact
        n_doc_done += 1
        for e in ledger:
            sid = e["seed_id"]
            sink.add(engine="H112", seed_id=sid,
                     tag=tags[sid] if sid >= 0 else "",
                     register=d["register"],
                     seed=text[e["sent_start"]:e["sent_end"]],
                     chunk=chunks[sid] if sid >= 0 else "",
                     claim=e["sent_claim"], span_start=e["sent_c0"],
                     span_end=e["sent_c0"] + len(e["new_span"]),
                     orig_span_start=e["sent_c0"], orig_span_end=e["sent_c1"],
                     orig_span=e["orig_span"], new_span=e["new_span"],
                     locus_type=e["locus_type"], source=e["source"],
                     degen=e["degen"], evasion=e["evasion"],
                     exact_repro=e["exact_repro"], filler=e["filler"],
                     hit_eos=e["hit_eos"], bound_q=e["bound_q"],
                     degeneracy_pass=not e["degen"],
                     usable=not (e["degen"] or e["evasion"]),
                     long_form=True, doc_id=d["doc_id"], doc_clean=text,
                     doc_corrupt=corrupt,
                     doc_span_start=e["doc_span_start"],
                     doc_span_end=e["doc_span_end"],
                     doc_sent_index=e["sent_index"],
                     doc_sent_start=e["sent_start"],
                     doc_sent_end=e["sent_end"])
            made += 1
        if n_doc_done % 250 == 0:
            log(f"  H112-LF {made}/{target} spans over {n_doc_done} docs "
                f"({time.time() - t0:.0f}s)")
    log(f"H112 long-form: {made} spans over {n_doc_done} docs, "
        f"{intact_fail} splice rejects, {n_skipped} dedup skips, "
        f"{time.time() - t0:.0f}s")
    return made, n_doc_done, intact_fail, n_skipped


# ------------------------------------------------------------------- H114


def gen_h114(tok, model, sink, seeds, chunks, tags, ids, target):
    rng = random.Random(303)
    made = 0
    t0 = time.time()
    for sid in ids:
        if made >= target:
            break
        seed = seeds[sid]
        try:
            draws = DRT.sample_spans(seed, 3, rng=random.Random(sid * 5701 + 3))
        except Exception:
            continue
        seen = set()
        for c0, c1, stext, ltype, source in draws:
            if made >= target:
                break
            if not stext.strip() or c1 <= c0 or (c0, c1) in seen:
                continue
            # registered 2x overweight of the meaning-flip loci
            if ltype not in OVERWEIGHT_LOCI and rng.random() >= 0.5:
                continue
            if ENG.head_repeats(seed, stext, c0, c1):
                continue
            seen.add((c0, c1))
            res = ENG.blind_span(tok, model, seed, c0, c1, stext)
            if res is None:
                continue
            sink.add(engine="H114", seed_id=sid, tag=tags[sid],
                     register=tags[sid].rsplit("_", 1)[-1], seed=seed,
                     chunk=chunks[sid], claim=res["claim"],
                     span_start=res["c0"],
                     span_end=res["c0"] + len(res["decoded_span"]),
                     orig_span_start=res["c0"], orig_span_end=res["c1"],
                     orig_span=seed[res["c0"]:res["c1"]],
                     new_span=res["decoded_span"], locus_type=ltype,
                     source=source, degen=res["degen"], recon=res["recon"],
                     degen_postseam=res["degen_postseam"],
                     raw_new_span=res["raw_decoded_span"],
                     hit_eos=res["hit_eos"], seam_cleaned=res["seam_cleaned"],
                     degeneracy_pass=not res["degen"],
                     usable=not (res["degen"] or res["recon"]))
            made += 1
            if made % 2500 == 0:
                log(f"  H114 {made}/{target} ({made / max(time.time() - t0, 1):.1f}/s)")
    log(f"H114: {made} decodes in {time.time() - t0:.0f}s")
    return made


# ------------------------------------------------------------------- H113


def _h113_block(nlp, sink, seeds, chunks, tags, block, quota, counts, rng):
    """One seed block under a STRICT per-operator quota. Returns rows made."""
    made = 0
    texts = [seeds[i] for i in block]
    for n, doc in enumerate(nlp.pipe(texts, batch_size=128)):
        if all(counts[op] >= quota[op] for op in quota):
            break
        sid = block[n]
        seed, chunk = seeds[sid], chunks[sid]
        if len(seed) < 30 or len(seed) > 600:
            continue
        chunk_cf = chunk.casefold()
        chunk_nums = H113G.chunk_numbers(chunk)

        applicable = []
        if any(e.label_ in H113G.NUMDATE_ENTS for e in doc.ents):
            applicable.append("number")
        if any(re.search(rf"\d[\d,\.]*\s?{u}\b", seed)
               for dim in H113G.UNIT_DIMS for u in dim):
            applicable.append("unit")
        if any(t.lower_ in H113G.COMP_MAP for t in doc) or any(
                a in seed.casefold() for a, _ in H113G.COMP_PHRASES):
            applicable.append("comparative")
        if any(t.dep_ == "neg" or t.lower_ in ("not", "never") for t in doc) or any(
                t.lower_ in H113G.NEG_AUX for t in doc):
            applicable.append("negation")
        hungry = [op for op in applicable if counts[op] < quota[op]]
        if not hungry:
            continue
        chosen = min(hungry, key=lambda op: counts[op] / max(quota[op], 1))

        if chosen == "number":
            res = H113G.op_number(doc, seed, chunk_cf, chunk_nums, rng)
        elif chosen == "unit":
            res = H113G.op_unit(doc, seed, chunk_cf, chunk_nums, rng)
        elif chosen == "comparative":
            res = H113G.op_comparative(doc, seed, chunk_cf, rng)
        else:
            res = H113G.op_negation(doc, seed, chunk_cf, rng)
        if res is None:
            continue
        claim, s0, s1, old, new = res
        if H113G.norm(claim) == H113G.norm(seed):
            continue
        counts[chosen] += 1
        made += 1
        degen = ENG.degenerate_span(claim)
        sink.add(engine="H113", seed_id=sid, tag=tags[sid],
                 register=tags[sid].rsplit("_", 1)[-1], seed=seed, chunk=chunk,
                 claim=claim, span_start=s0, span_end=s1,
                 orig_span_start=s0, orig_span_end=s0 + len(old),
                 orig_span=old, new_span=new,
                 locus_type=H113_LOCUS[chosen], operator=chosen,
                 source="operator", degen=degen, degeneracy_pass=not degen)
    return made


def gen_h113(sink, seeds, chunks, tags, ids, target):
    """Four surviving operators only, evidence-entailed loci, famine proportions.

    Strict quota per block; whatever an operator's corpus supply cannot deliver
    is redistributed over the operators that did meet their quota.
    """
    import spacy

    rng = random.Random(404)
    nlp = spacy.load("en_core_web_lg")
    quota = {op: max(1, int(round(target * s))) for op, s in H113_MIX.items()}
    counts = {op: 0 for op in H113_MIX}
    log(f"H113 quota: {quota}")
    t0 = time.time()
    made = 0
    block = 15_000
    for b0 in range(0, len(ids), block):
        if made >= target:
            break
        made += _h113_block(nlp, sink, seeds, chunks, tags,
                            ids[b0:b0 + block], quota, counts, rng)
        log(f"  H113 {made}/{target} {counts} ({time.time() - t0:.0f}s)")
        if made < target:
            live = [op for op in H113_MIX if counts[op] >= quota[op]]
            if not live:
                continue
            short = target - made
            w = sum(H113_MIX[op] for op in live)
            for op in live:
                quota[op] += max(1, int(round(short * H113_MIX[op] / w)))
    log(f"H113: {made} swaps in {time.time() - t0:.0f}s, mix {counts}")
    return made, counts


# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1/200 volumes, for a pre-flight correctness check")
    args = ap.parse_args()
    scale = 200 if args.smoke else 1
    n112 = max(20, N_H112 // scale)
    n113 = max(20, N_H113 // scale)
    n114 = max(20, N_H114 // scale)

    t_start = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)

    lf_ok = False
    if SUBGATE.exists():
        lf_ok = json.loads(SUBGATE.read_text()).get("verdict") == "SURVIVES"
    log(f"DR-H116 sub-gate verdict: "
        f"{'SURVIVES - long-form lane ON' if lf_ok else 'KILL/absent - sentence-level only'}")
    n112_lf = int(n112 * LONGFORM_SHARE) if lf_ok else 0
    n112_sent = n112 - n112_lf

    seeds, chunks, tags, order = load_pool()
    log(f"{len(seeds)} distinct seeds in the H111 public pool")
    cut1 = int(len(order) * 0.46)
    cut2 = int(len(order) * 0.92)
    ids112, ids114 = order[:cut1], order[cut1:cut2]
    ids113 = order[cut2:] + order[:cut2]  # CPU operators need the widest supply
    if args.smoke:
        ids112, ids114, ids113 = ids112[:400], ids114[:400], ids113[:4000]

    sink = Sink()
    tok, model = ENG.S0.load_mbart()
    model.eval()

    made112 = gen_h112(tok, model, sink, seeds, chunks, tags, ids112, n112_sent)
    lf_stats = (0, 0, 0, 0)
    if n112_lf:
        lf_stats = gen_h112_longform(tok, model, sink, seeds, chunks, tags,
                                     n112_lf, ids=ids112)
    made114 = gen_h114(tok, model, sink, seeds, chunks, tags, ids114, n114)
    sink.flush()
    del model
    torch.cuda.empty_cache()

    made113, mix113 = gen_h113(sink, seeds, chunks, tags, ids113, n113)
    sink.flush()

    df = pl.DataFrame(sink.rows, infer_schema_length=None)
    log(f"raw rows: {df.height}")

    # ---- dedup exact (seed_id | doc_id, span_start, replacement) within lane
    df = df.with_columns(
        pl.concat_str([
            pl.when(pl.col("doc_id").is_null())
              .then(pl.col("seed_id").cast(pl.Utf8))
              .otherwise(pl.col("doc_id")),
            pl.col("span_start").cast(pl.Utf8),
            pl.col("new_span"),
        ], separator="|").alias("dedup_key"))
    before = df.height
    df = df.unique(subset=["dedup_key"], keep="first", maintain_order=True)
    log(f"dedup: {before} -> {df.height} rows")

    # ---- H113 fluency gate (H111 stage-0 calibrated NLL threshold)
    h113_idx = [i for i, e in enumerate(df["engine"].to_list()) if e == "H113"]
    if h113_idx:
        claims = df["claim"].to_list()
        nll = ENG.S0.gpt2_nll([claims[i] for i in h113_idx])
        col_nll = [None] * df.height
        col_flu = [None] * df.height
        for i, v in zip(h113_idx, nll):
            col_nll[i] = float(v)
            col_flu[i] = float(v) <= NLL_MAX
        df = df.with_columns(pl.Series("nll", col_nll, dtype=pl.Float64),
                             pl.Series("fluent", col_flu, dtype=pl.Boolean))
        # the NLL fluency gate is H113's OWN gate, kept separate from the
        # deterministic degeneracy gate so the debris bar stays gate-comparable
        df = df.with_columns(
            pl.when(pl.col("engine") == "H113")
              .then(pl.col("usable") & pl.col("fluent").fill_null(False))
              .otherwise(pl.col("usable")).alias("usable"))
        ok = sum(1 for v in col_flu if v)
        log(f"H113 fluency: {ok}/{len(h113_idx)} pass (gate measured 0.955)")
    df.write_parquet(CKPT)

    # ---- bidirectional NLI (mDeBERTa, same model + code path as the gates)
    pairs_fwd = list(zip(df["seed"].to_list(), df["claim"].to_list()))
    log(f"NLI forward on {len(pairs_fwd)} pairs ...")
    _, fwd = ENG.S0.nli_entail(pairs_fwd)
    log("NLI backward ...")
    _, bwd = ENG.S0.nli_entail([(b, a) for a, b in pairs_fwd])
    df = df.with_columns(pl.Series("nli_fwd", [float(x) for x in fwd]),
                         pl.Series("nli_bwd", [float(x) for x in bwd]))
    df.write_parquet(OUT)
    log(f"wrote {df.height} rows -> {OUT}")

    # ---- realized-debris readout vs the registered per-engine kill bars
    bars = {"H112": 0.124, "H113": 0.02, "H114": 0.286}
    per_engine = {}
    for eng in ("H112", "H113", "H114"):
        sub = df.filter(pl.col("engine") == eng)
        if not sub.height:
            continue
        debris = sub.filter(~pl.col("degeneracy_pass")).height / sub.height
        per_engine[eng] = {
            "n": sub.height,
            "debris": round(debris, 4),
            "debris_kill_bar": bars[eng],
            "debris_kill": debris > bars[eng],
            "usable": round(sub.filter(pl.col("usable")).height / sub.height, 4),
            "nli_fwd_ge08": round(
                sub.filter(pl.col("nli_fwd") >= 0.8).height / sub.height, 4),
        }
    summary = {
        "generated": {"H112_sentence": made112, "H112_longform": lf_stats[0],
                      "H112_longform_docs": lf_stats[1],
                      "H112_longform_splice_rejects": lf_stats[2],
                      "H113": made113, "H113_mix": mix113, "H114": made114},
        "longform_enabled": lf_ok,
        "rows_after_dedup": df.height,
        "per_engine": per_engine,
        "runtime_s": round(time.time() - t_start, 1),
        "judge_pass": "pending - GPU1, not run here",
    }
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    log(f"summary -> {SUMMARY}")


if __name__ == "__main__":
    main()
