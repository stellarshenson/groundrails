"""Shared span-corruption engines for the DR-2 pilot and the DR-H116 sub-gate.

Lifts the PROVEN gate code paths verbatim in mechanism:
- `infill_span`  = DR_H112_gate arm (b): mask-token infilling in the encoder
  input + co-mention occlusion, forced clean decoder prefix, sampled span under
  the true-fact ban list, suffix-bigram stop, word-boundary cut, text splice
- `blind_span`   = DR_H114_gate: clean encode, span positions zeroed in the
  cross-attention mask, greedy span at span_len+50% slack, word-boundary cut
- `seam_clean`   = DR-H114 pilot amendment: doubled-token seam cleaner applied to
  the DECODED SPAN (not the spliced text) so char offsets never drift

Helper predicates (normalize / degeneracy / ban list / bound-qualifier guard) are
imported from DR_H112_gate so the pilot and the gate share one implementation.

GPU0 is pinned before any torch import (RTX PRO 4000 24GB).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import DR_H112_gate as H112G  # noqa: E402  (also brings S0 + DR_targeting)
import DR_targeting as DRT  # noqa: E402

S0 = H112G.S0
DEV = "cuda"

# DR-H112 pilot locus restriction (registration: hedge/relverb fills read as
# no-delta paraphrase to the judge; core loci sit at 65-71% changed-fact)
CORE_LOCI = ("number_date", "negation", "entity", "positional")

normalize = H112G.normalize
degenerate_span = H112G.degenerate_span
is_filler = H112G.is_filler
levenshtein = H112G.levenshtein
bad_words_for = H112G.bad_words_for
bound_qualified = H112G.bound_qualified


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------- seam cleaner


_LAST_WORD = re.compile(r"(\w+)\W*$")
_FIRST_WORD = re.compile(r"^\W*(\w+)")


def seam_clean(prefix: str, dec: str, suffix: str) -> str:
    """Drop a decoded-span boundary word that duplicates the spliced neighbour.

    Operates on the decoded span alone, so the ledger offsets stay exact
    (span_start is unchanged, span_end follows len(dec)).
    """
    d = dec
    pm, dm = _LAST_WORD.search(prefix), _FIRST_WORD.match(d)
    if pm and dm and pm.group(1).casefold() == dm.group(1).casefold():
        d = d[dm.end(1):].lstrip()
    dm2, sm = _LAST_WORD.search(d), _FIRST_WORD.match(suffix)
    if dm2 and sm and dm2.group(1).casefold() == sm.group(1).casefold():
        d = d[:dm2.start(1)].rstrip()
    return d


# --------------------------------------------------------- token span resolve


def resolve_span(tok, seed: str, c0: int, c1: int, max_length: int = 128):
    """Map char span -> encoder token indices under the CLEAN tokenization."""
    enc = tok(seed, return_tensors="pt", truncation=True, max_length=max_length,
              return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"][0].tolist()
    content = [i for i in range(len(ids)) if offsets[i] != [0, 0]]
    span_tok = [i for i in content if offsets[i][1] > c0 and offsets[i][0] < c1]
    if not span_tok or len(span_tok) >= len(content):
        return None
    return enc, ids, content, span_tok


_STOP_CLS = None


def _stop_class():
    global _STOP_CLS
    if _STOP_CLS is None:
        from transformers import StoppingCriteria

        class _Stop(StoppingCriteria):
            def __init__(self, fn):
                self.fn = fn

            def __call__(self, input_ids, scores, **kw):
                return self.fn(input_ids, scores)

        _STOP_CLS = _Stop
    return _STOP_CLS


# ------------------------------------------------------------- H112 engine


@torch.no_grad()
def infill_span(tok, model, seed: str, c0: int, c1: int, span_text: str) -> dict | None:
    """DR-H112 SPAN-INFILL-BAN on one span. Returns None when unusable."""
    from transformers import StoppingCriteriaList

    lang_id = tok.lang_code_to_id["en_XX"]
    eos, mask_id = tok.eos_token_id, tok.mask_token_id

    # masked encoder input + co-mention occlusion of the same surface form
    masked = seed[:c0] + tok.mask_token + seed[c1:]
    pat = re.compile(re.escape(span_text), re.IGNORECASE)
    cut = c0 + len(tok.mask_token)
    head, tail = masked[:cut], masked[cut:]
    tail = pat.sub(tok.mask_token, tail)
    head_pre = pat.sub(tok.mask_token, head[:c0])
    masked = head_pre + head[c0:] + tail
    enc_in = tok(masked, return_tensors="pt", truncation=True, max_length=140).to(DEV)

    r = resolve_span(tok, seed, c0, c1)
    if r is None:
        return None
    _, ids, content, span_tok = r
    first_span = span_tok[0]
    prefix_content = [ids[i] for i in content if i < first_span]
    suffix_content = [ids[i] for i in content if i > span_tok[-1]]
    dec_prefix = torch.tensor([[eos, lang_id] + prefix_content], device=DEV)
    n_sp = len(span_tok)
    budget = n_sp + 4
    stop = StoppingCriteriaList([_stop_class()(
        H112G.SuffixBigramStop(suffix_content, dec_prefix.shape[1]))])
    try:
        gen = model.generate(
            input_ids=enc_in["input_ids"], attention_mask=enc_in["attention_mask"],
            decoder_input_ids=dec_prefix, max_new_tokens=budget,
            min_new_tokens=min(2, budget), stopping_criteria=stop,
            do_sample=True, top_p=0.9, temperature=0.9,
            bad_words_ids=bad_words_for(tok, span_text, mask_id))
    except Exception as e:  # generation failures are logged, not fatal
        log(f"  infill generate error: {e}")
        return None

    cont_ids = gen[0][dec_prefix.shape[1]:].tolist()
    hit_eos = eos in cont_ids
    if hit_eos:
        cont_ids = cont_ids[:cont_ids.index(eos)]
    if len(cont_ids) >= 2 and tuple(cont_ids[-2:]) == tuple(suffix_content[:2]):
        cont_ids = cont_ids[:-2]
    dec_span = tok.decode(cont_ids, skip_special_tokens=True).strip()
    if not hit_eos and " " in dec_span:
        dec_span = dec_span.rsplit(" ", 1)[0]

    n_true, n_dec = normalize(span_text), normalize(dec_span)
    return {
        "decoded_span": dec_span,
        "claim": seed[:c0] + dec_span + seed[c1:],
        "n_span_tok": n_sp,
        "degen": degenerate_span(dec_span),
        "exact_repro": bool(n_dec) and n_true == n_dec,
        "evasion": bool(n_dec) and (n_true == n_dec or n_true in n_dec
                                    or levenshtein(n_true, n_dec) <= 2),
        "filler": is_filler(n_dec),
        "hit_eos": hit_eos,
        "bound_q": bound_qualified(seed, c0),
    }


# ------------------------------------------------------------- H114 engine


def head_repeats(seed: str, span_text: str, c0: int, c1: int) -> bool:
    """Registered H114 restriction: span head token absent elsewhere in seed."""
    parts = span_text.split()
    head = parts[0].casefold() if parts else ""
    if not head or len(head) < 3:
        return False
    rest = (seed[:c0] + " " + seed[c1:]).casefold()
    return re.search(rf"\b{re.escape(head)}\b", rest) is not None


@torch.no_grad()
def blind_span(tok, model, seed: str, c0: int, c1: int, span_text: str) -> dict | None:
    """DR-H114 XATTN-BLIND on one span, with the pilot seam cleaner applied."""
    lang_id = tok.lang_code_to_id["en_XX"]
    eos = tok.eos_token_id

    r = resolve_span(tok, seed, c0, c1)
    if r is None:
        return None
    enc, ids, content, span_tok = r
    attn = torch.ones(1, len(ids), dtype=torch.long)
    for i in span_tok:
        attn[0, i] = 0
    input_ids = enc["input_ids"].to(DEV)
    enc_out = model.get_encoder()(input_ids=input_ids,
                                  attention_mask=torch.ones_like(input_ids))
    prefix_content = [ids[i] for i in content if i < span_tok[0]]
    dec_prefix = torch.tensor([[eos, lang_id] + prefix_content], device=DEV)
    n_sp = len(span_tok)
    budget = max(2, int(np.ceil(n_sp * 1.5)))  # pre-registered +50% slack
    try:
        gen = model.generate(encoder_outputs=enc_out, attention_mask=attn.to(DEV),
                             decoder_input_ids=dec_prefix, num_beams=1,
                             do_sample=False, max_new_tokens=budget)
    except Exception as e:
        log(f"  blind generate error: {e}")
        return None

    cont_ids = gen[0][dec_prefix.shape[1]:].tolist()
    hit_eos = eos in cont_ids
    if hit_eos:
        cont_ids = cont_ids[:cont_ids.index(eos)]
    dec_span = tok.decode(cont_ids, skip_special_tokens=True).strip()
    if not hit_eos and " " in dec_span:
        dec_span = dec_span.rsplit(" ", 1)[0]
    raw_span = dec_span
    dec_span = seam_clean(seed[:c0], dec_span, seed[c1:])  # BEFORE the gates

    # a fully absorbed span is a pure seam stutter -> the splice becomes a clean
    # deletion; take one adjacent space with it so no double space is left
    c0e, c1e = c0, c1
    if not dec_span.strip():
        dec_span = ""
        if c1 < len(seed) and seed[c1] == " ":
            c1e = c1 + 1
        elif c0 > 0 and seed[c0 - 1] == " ":
            c0e = c0 - 1

    n_true, n_dec = normalize(span_text), normalize(dec_span)
    # degeneracy is measured gate-identically on the RAW decoded span (the seam
    # cleaner is a post-gate text repair introduced by the H114 pilot amendment);
    # a non-empty cleaned span is additionally checked so seam debris is caught
    degen = degenerate_span(raw_span) or (bool(dec_span.strip())
                                          and degenerate_span(dec_span))
    return {
        "decoded_span": dec_span,
        "raw_decoded_span": raw_span,
        "claim": seed[:c0e] + dec_span + seed[c1e:],
        "c0": c0e, "c1": c1e,
        "n_span_tok": n_sp,
        "degen": degen,
        "degen_postseam": degenerate_span(dec_span),
        "recon": bool(n_dec) and bool(n_true) and (n_true == n_dec or n_true in n_dec),
        "hit_eos": hit_eos,
        "seam_cleaned": raw_span != dec_span,
    }


# ------------------------------------------------------------- span sourcing


def core_spans(seed: str, n: int, rng, allowed=CORE_LOCI, tries: int = 3):
    """Draw up to n spans from the shared targeting module, core loci only."""
    out, seen = [], set()
    for _ in range(tries):
        if len(out) >= n:
            break
        try:
            draws = DRT.sample_spans(seed, n, rng=rng)
        except Exception:
            return out
        for c0, c1, text, ltype, source in draws:
            if ltype not in allowed or not text.strip() or c1 <= c0:
                continue
            if (c0, c1) in seen:
                continue
            seen.add((c0, c1))
            out.append((c0, c1, text, ltype, source))
            if len(out) >= n:
                break
    return out


# ------------------------------------------------- DR-H116 long-form wrapper

import json  # noqa: E402
import math  # noqa: E402
import zipfile  # noqa: E402

ROOT = HERE.parents[1]
RAGTRUTH_ZIP = ROOT / "data/external/datasets/dataset-ragtruth.zip"
RAGTRUTH_MEMBER = "wandb__RAGTruth-processed__train.parquet"
QUANTILES = [0.18, 0.29, 0.47, 0.71, 0.85]  # registered late-skewed positions
TOK_MIN, TOK_MAX = 256, 2048


def span_count_hist() -> dict[int, float]:
    """RAGTruth train per-response hallucination-span-count histogram (level 1)."""
    import polars as pl

    with zipfile.ZipFile(RAGTRUTH_ZIP) as z, z.open(RAGTRUTH_MEMBER) as f:
        df = pl.read_parquet(f.read())
    counts: dict[int, int] = {}
    for labels in df["hallucination_labels"].to_list():
        spans = json.loads(labels)
        if spans:
            counts[len(spans)] = counts.get(len(spans), 0) + 1
    tot = sum(counts.values())
    return {k: v / tot for k, v in sorted(counts.items())}


def precount_tokens(tok, texts, batch: int = 512) -> list[int]:
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], add_special_tokens=False).input_ids
        out += [len(x) for x in enc]
    return out


def assemble_docs(pools: dict[str, list[tuple]], rng, n_docs: int,
                  allow_cycle: bool = False):
    """Concatenate same-register seed sentences into 256-2048-token responses.

    `pools[register]` = list of (seed_text, seed_id, n_tok). Returns docs with a
    component ledger so each sentence keeps its own seed_id / evidence link.
    """
    regs = sorted(pools)
    cursor = {r: 0 for r in regs}
    docs = []
    i = 0
    stalls = 0
    while len(docs) < n_docs and stalls < len(regs) * 2:
        reg = regs[i % len(regs)]
        i += 1
        pool = pools[reg]
        if cursor[reg] >= len(pool):
            if allow_cycle:
                rng.shuffle(pool)  # reshuffle so a second pass makes new docs
                cursor[reg] = 0
            else:
                stalls += 1
                continue
        target = int(math.exp(rng.uniform(math.log(TOK_MIN + 48),
                                          math.log(TOK_MAX * 0.55))))
        parts, comps, n_tok, pos = [], [], 0, 0
        while cursor[reg] < len(pool) and n_tok < target:
            s, sid, st = pool[cursor[reg]]
            cursor[reg] += 1
            s = s.strip()
            if not s or n_tok + st > TOK_MAX:
                break
            if parts:
                pos += 1  # the joining space
            comps.append((pos, pos + len(s), sid))
            parts.append(s)
            pos += len(s)
            n_tok += st
        if n_tok < TOK_MIN or len(parts) < 2:
            stalls += 1
            continue
        stalls = 0
        docs.append({"doc_id": f"DR-LF-{len(docs):05d}", "register": reg,
                     "text": " ".join(parts), "n_tok": n_tok,
                     "components": comps})
    return docs


def pick_sentences(sents, doc_len: int, k: int, rng) -> list[int]:
    """Registered targeting level 2: span-bearing sentence char-offset quantiles."""
    chosen: list[int] = []
    qs = rng.sample(QUANTILES, min(k, len(QUANTILES)))
    while len(qs) < k:
        qs.append(rng.choice(QUANTILES))
    for q in qs:
        target = q * doc_len
        order = sorted(range(len(sents)),
                       key=lambda i: abs((sents[i][0] + sents[i][1]) / 2 - target))
        for i in order:
            if i not in chosen:
                chosen.append(i)
                break
    return sorted(chosen)


def splice_doc(text: str, edits: list[dict]):
    """Char-exact splice + ledger + integrity proof.

    Each edit needs abs_start / abs_end (offsets in `text`) and new_span.
    Returns (corrupt_text, ledger, intact) where `intact` is True iff putting
    the originals back at the ledgered offsets reproduces `text` byte-for-byte.
    """
    edits = sorted(edits, key=lambda e: e["abs_start"])
    out, ledger, prev = [], [], 0
    for e in edits:
        if e["abs_start"] < prev:  # overlapping selections are dropped
            continue
        out.append(text[prev:e["abs_start"]])
        new_start = sum(len(x) for x in out)
        out.append(e["new_span"])
        ledger.append({**e, "doc_span_start": new_start,
                       "doc_span_end": new_start + len(e["new_span"])})
        prev = e["abs_end"]
    out.append(text[prev:])
    corrupt = "".join(out)

    rebuild, prev = [], 0
    for e in ledger:
        rebuild.append(corrupt[prev:e["doc_span_start"]])
        rebuild.append(text[e["abs_start"]:e["abs_end"]])
        prev = e["doc_span_end"]
    rebuild.append(corrupt[prev:])
    return corrupt, ledger, "".join(rebuild) == text
