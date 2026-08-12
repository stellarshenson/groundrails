"""R17-H143 Stage A - tiny-reasoner scorer harness (measurement only).

Scores (evidence chunk, derived claim) pairs from R17-H143_evalset.parquet with
a GROUNDED / UNGROUNDED forced-choice verdict, per model, and checkpoints every
batch to R17-H143_scores.parquet.

Two backends:
  * transformers - the five tiny decoders, GPU0
  * vllm        - the teacher ceiling, GPU2. The registered teacher
                  (Mistral-Small-24B) has NO cached weights - only two orphaned
                  .incomplete stubs - so --teacher-model names a substitute.

Prompt discipline (standing repo rule after a past prompt-format failure): one
template per model, rendered through that model's OWN chat template where it has
one, and through its documented native format where it does not. Audited at
startup - the script asserts every special token it emits exists in the model's
tokenizer, and that the GROUNDED / UNGROUNDED first-token id sets are disjoint.

  Baguettotron       ChatML, user-only turn, template forces an open <think>
                     (its README documents user-only turns; system role unused)
  Pleias-RAG-350M    native <|query_start|> / <|source_start|> RAG format - it
                     ships NO chat template, a generic one would be off-format
  Monad 56M          no chat template at all: plain completion prompt
  SmolLM2-360M-Inst  ChatML via its own template, system + user
  Qwen3-0.6B         its own template, enable_thinking=True

Scoring per pair, in two phases. Phase 1 greedy-decodes up to 512 new tokens -
the free reasoning trace, kept for the arithmetic and compliance reads. Phase 2
cuts that trace at the end of reasoning, closes the think block, elicits the
answer, and takes the forced-choice margin on the FIRST answer token:
logsumexp(GROUNDED ids) - logsumexp(UNGROUNDED ids), recorded by a
LogitsProcessor. A fixed answer position is what makes the margin comparable
across models and defined even for a model that never says either word
(Pleias-RAG answers in prose on every pair). Raw fields only are stored; AUROC
and the branch are derived in the analysis pass.

Positive-control gate: the 50 control pairs are scored first. Below 0.90 AUROC
the harness/prompt for that model is defective and the model is skipped rather
than recorded as a low read. Monad is gate-exempt (floor reference).

Run (detached):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  nohup setsid python experiments/grounding-semantic/R17-H143_stageA.py \
      --backend transformers >> logs/R17-H143_stageA.log 2>&1 &

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  nohup setsid /home/lab/venvs/vllm/bin/python \
      experiments/grounding-semantic/R17-H143_stageA.py --backend vllm \
      >> logs/R17-H143_stageA_teacher.log 2>&1 &
"""

import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

HERE = pathlib.Path(__file__).parent
EVALSET = HERE / "R17-H143_evalset.parquet"
SCORES = HERE / "R17-H143_scores.parquet"
GATE_LOG = HERE / "R17-H143_gate.json"
SAMPLES = HERE / "R17-H143_gen_samples.json"

CONTROL_GATE = 0.90
MAX_NEW_TOKENS = 512

INSTRUCTION = (
    "You verify whether a claim is grounded in the evidence. The claim may state "
    "a value derived from numbers in the evidence. Check the arithmetic. Answer "
    "with exactly one word: GROUNDED or UNGROUNDED."
)


def user_block(chunk: str, claim: str) -> str:
    return f"Evidence:\n{chunk}\n\nClaim: {claim}\n\nIs the claim grounded in the evidence?"


# Capped think budget (registered). The score is ALWAYS read at a forced answer
# position: the free trace is truncated at the end of reasoning, the think block
# is closed, the answer is elicited, and the margin is taken on that first
# answer token. Parsing the free trace instead is not sound here - measured on
# the controls, Baguettotron re-emits the instruction after </think>, so a
# last-occurrence scan latches onto the echoed phrase "GROUNDED or UNGROUNDED",
# and its real conclusions are phrased "NOT grounded", which substring-matches
# GROUNDED. The free-trace parse is still recorded, as a compliance diagnostic
# (parse_fail_rate), but it does not drive the score.
FORCE_SUFFIX = {
    "baguettotron": "\n</think>\nAnswer:",
    "chat_think": "\n</think>\n\nAnswer:",
    "chat": "\nAnswer:",
    "completion": "\nAnswer:",
    "pleias_rag": "<|answer_start|>The claim is",
}
STRIP_TAIL = ("<|answer_end|>", "<|end_of_text|>", "<|im_end|>", "<|answer_start|>")
# reasoning styles: the free trace is cut here before the answer is elicited
THINK_CLOSE = {"baguettotron": "</think>", "chat_think": "</think>"}

MODELS = {
    "PleIAs/Baguettotron": dict(params_M=321, style="baguettotron", offline=True, batch=32),
    "PleIAs/Pleias-RAG-350M": dict(params_M=350, style="pleias_rag", offline=True, batch=32),
    "PleIAs/Monad": dict(params_M=56, style="completion", offline=True, batch=64, gate_exempt=True),
    "HuggingFaceTB/SmolLM2-360M-Instruct": dict(params_M=362, style="chat", batch=32),
    "Qwen/Qwen3-0.6B": dict(params_M=596, style="chat_think", batch=32, over_budget=True),
}
TEACHER = "mistralai/Mistral-Small-24B-Instruct-2501"

GROUNDED_VARIANTS = ["GROUNDED", " GROUNDED", "Grounded", " Grounded", "grounded", " grounded"]
UNGROUNDED_VARIANTS = [
    "UNGROUNDED", " UNGROUNDED", "Ungrounded", " Ungrounded", "ungrounded", " ungrounded",
]


# --------------------------------------------------------------------------- #
# prompt construction - one template per model, audited against its tokenizer
# --------------------------------------------------------------------------- #
def build_prompt(style: str, tok, chunk: str, claim: str) -> str:
    body = user_block(chunk, claim)
    if style == "baguettotron":
        # README documents user-only turns; the template appends an open <think>
        return tok.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION}\n\n{body}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if style == "chat":
        return tok.apply_chat_template(
            [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if style == "chat_think":
        return tok.apply_chat_template(
            [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    if style == "pleias_rag":
        # Native structured RAG format - the model ships no chat template. The
        # generation prompt ends at <|language_start|>, the first block the model
        # produces; ending at <|source_analysis_start|> instead makes it enumerate
        # a source 2 and echo the chunk (probed, rejected).
        return (
            f"<|query_start|>Does the source support this statement? "
            f'Statement: "{claim}" '
            f"The statement may give a value derived from numbers in the source; check the "
            f"arithmetic. Reply with exactly one word: GROUNDED if the source supports the "
            f"statement, UNGROUNDED if it does not.<|query_end|>"
            f"<|source_start|><|source_id|>1 {chunk}<|source_end|><|language_start|>"
        )
    if style == "completion":
        # base LM, no chat template: plain completion
        return (
            f"{INSTRUCTION}\n\nEvidence: {chunk}\n\nClaim: {claim}\n\n"
            f"Is the claim grounded in the evidence?\nAnswer:"
        )
    raise ValueError(style)


def audit_prompt(style: str, tok, name: str) -> None:
    """Assert every special token the template emits exists in this tokenizer."""
    sample = build_prompt(style, tok, "The X of Y is 3.", "The X of Y is 3.")
    ids = tok.encode(sample, add_special_tokens=False)
    roundtrip = tok.decode(ids)
    required = {
        "baguettotron": ["<|im_start|>", "<|im_end|>", "<think>"],
        "chat": ["<|im_start|>", "<|im_end|>"],
        "chat_think": ["<|im_start|>", "<|im_end|>"],
        "pleias_rag": [
            "<|query_start|>", "<|query_end|>", "<|source_start|>",
            "<|source_id|>", "<|source_end|>", "<|language_start|>",
        ],
        "completion": [],
    }[style]
    vocab = tok.get_vocab()
    for tokstr in required:
        assert tokstr in vocab, f"{name}: template emits {tokstr!r}, absent from tokenizer"
        assert tokstr in sample, f"{name}: expected {tokstr!r} in rendered prompt"
        # the token must survive tokenization as a single unit
        assert len(tok.encode(tokstr, add_special_tokens=False)) == 1, (
            f"{name}: {tokstr!r} does not tokenize as one unit"
        )
    print(f"[audit] {name} style={style} prompt_tokens={len(ids)}")
    print(f"[audit] {name} rendered head: {sample[:220]!r}")
    assert "GROUNDED" in roundtrip or "GROUNDED" in sample


def verdict_token_ids(tok) -> tuple[list[int], list[int]]:
    def firsts(variants):
        out = set()
        for v in variants:
            ids = tok.encode(v, add_special_tokens=False)
            # SentencePiece tokenizers emit a standalone space token first, which
            # is shared by both verdicts; the discriminating token is the next one
            for t in ids:
                if tok.decode([t]).strip():
                    out.add(t)
                    break
        return sorted(out)

    g, u = firsts(GROUNDED_VARIANTS), firsts(UNGROUNDED_VARIANTS)
    overlap = set(g) & set(u)
    assert not overlap, f"GROUNDED/UNGROUNDED first tokens overlap: {overlap}"
    return g, u


def parse_verdict(text: str, first: bool = False) -> tuple[str | None, int]:
    """A verdict word in the text, with its character offset.

    UNGROUNDED is checked first because it contains GROUNDED as a substring: an
    occurrence of "grounded" starting within two characters of the tail of an
    "ungrounded" match is that same word, not a separate verdict.

    `first=True` is used on a forced-answer continuation, where the answer is the
    opening word; the default scans from the end, for the free trace.
    """
    low = text.lower()
    if first:
        iu = low.find("ungrounded")
        ig = low.find("grounded")
        if iu < 0 and ig < 0:
            return None, -1
        if iu >= 0 and (ig < 0 or ig >= iu):
            return "UNGROUNDED", iu
        return "GROUNDED", ig
    iu, ig = low.rfind("ungrounded"), low.rfind("grounded")
    if iu < 0 and ig < 0:
        return None, -1
    if iu >= 0 and (ig < 0 or ig <= iu + 2):
        return "UNGROUNDED", iu
    return "GROUNDED", ig


def verdict_token_index(pieces: list[str], char_off: int) -> int | None:
    """Token index covering `char_off` in ''.join(pieces).

    Anchoring the margin on the decoded verdict WORD, not on a bare token id:
    several of these tokenizers emit the verdict's first token as a single
    character ('G', 'U'), which any capitalised word would also produce.
    """
    if char_off < 0:
        return None
    acc = 0
    for k, p in enumerate(pieces):
        acc += len(p)
        if acc > char_off:
            return k
    return None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_p, n_n = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


# --------------------------------------------------------------------------- #
# transformers backend
# --------------------------------------------------------------------------- #
def run_transformers(name: str, cfg: dict, rows: pl.DataFrame, samples: dict) -> pl.DataFrame:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

    if cfg.get("offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.convert_ids_to_tokens(0)
    tok.padding_side = "left"
    audit_prompt(cfg["style"], tok, name)
    g_ids, u_ids = verdict_token_ids(tok)
    print(f"[audit] {name} GROUNDED first-ids={g_ids} UNGROUNDED first-ids={u_ids}")

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    # stop on the chat template's own end token as well as the config eos
    eos = set()
    if model.generation_config.eos_token_id is not None:
        e = model.generation_config.eos_token_id
        eos.update(e if isinstance(e, list) else [e])
    for t in ("<|im_end|>", "<|end_of_text|>", "<|answer_end|>"):
        if t in tok.get_vocab():
            eos.add(tok.get_vocab()[t])
    eos_list = sorted(eos)

    gt = torch.tensor(g_ids, device="cuda:0")
    ut = torch.tensor(u_ids, device="cuda:0")

    class Recorder(LogitsProcessor):
        def __init__(self):
            self.g: list = []
            self.u: list = []

        def __call__(self, input_ids, scores):
            lp = torch.log_softmax(scores.float(), dim=-1)
            self.g.append(torch.logsumexp(lp[:, gt], dim=-1).cpu())
            self.u.append(torch.logsumexp(lp[:, ut], dim=-1).cpu())
            return scores

    def generate(prompts: list[str], max_new: int):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        rec = Recorder()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                logits_processor=[rec],
                eos_token_id=eos_list or None,
                pad_token_id=tok.pad_token_id,
            )
        new_ids = gen[:, enc["input_ids"].shape[1] :].cpu()
        return new_ids, torch.stack(rec.g, dim=1), torch.stack(rec.u, dim=1)

    def read_verdict(ids: list[int], g_lp, u_lp, bi: int, first: bool = False):
        for k, t in enumerate(ids):
            if t in eos:
                ids = ids[:k]
                break
        pieces = tok.batch_decode([[t] for t in ids]) if ids else []
        v, off = parse_verdict("".join(pieces), first=first)
        pos = verdict_token_index(pieces, off)
        margin = None
        if pos is not None and pos < g_lp.shape[1] and ids[pos] in g_ids + u_ids:
            margin = float(g_lp[bi, pos] - u_lp[bi, pos])
        return ids, v, margin

    out_rows: list[dict] = []
    batch = cfg["batch"]
    recs = rows.to_dicts()
    for start in range(0, len(recs), batch):
        chunk_rows = recs[start : start + batch]
        prompts = [build_prompt(cfg["style"], tok, r["chunk"], r["claim"]) for r in chunk_rows]
        t0 = time.time()
        new_ids, g_lp, u_lp = generate(prompts, MAX_NEW_TOKENS)

        staged = []
        for bi in range(len(chunk_rows)):
            ids, v, _ = read_verdict(new_ids[bi].tolist(), g_lp, u_lp, bi)
            staged.append(dict(ids=ids, free_verdict=v))

        # the score is always read at a forced answer position, never off the
        # free trace: cut the reasoning, close the block, elicit the answer
        close = THINK_CLOSE.get(cfg["style"])
        fprompts = []
        for s in staged:
            tail = tok.decode(s["ids"], skip_special_tokens=False)
            if close and close in tail:
                tail = tail.split(close)[0]
            for t in STRIP_TAIL:
                tail = tail.replace(t, "")
            fprompts.append(tail)
        fprompts = [p + t + FORCE_SUFFIX[cfg["style"]] for p, t in zip(prompts, fprompts, strict=True)]
        f_ids, fg, fu = generate(fprompts, 8)
        n_forced = 0
        for bi, s in enumerate(staged):
            fids, v2, _ = read_verdict(f_ids[bi].tolist(), fg, fu, bi, first=True)
            # forced-choice margin at THE answer position - the first token of the
            # elicited answer. Reading it wherever the verdict word happens to land
            # leaves it undefined for a model that never says either word (measured:
            # Pleias-RAG answers in prose 50/50 times), and shifts the comparison to
            # a different position per pair.
            s.update(
                verdict=v2,
                margin=float(fg[bi, 0] - fu[bi, 0]),
                answer_text=tok.decode(fids, skip_special_tokens=True),
            )
            n_forced += s["free_verdict"] is None
        need = [1] * n_forced

        dt = time.time() - t0
        per = dt / len(chunk_rows)
        for bi, r in enumerate(chunk_rows):
            s = staged[bi]
            out_rows.append(
                dict(
                    model=name, pair_id=r["pair_id"], label=r["label"], control=r["control"],
                    neg_family=r["neg_family"], claim_form=r["claim_form"],
                    margin=s["margin"], margin_floored=False, verdict=s["verdict"],
                    free_verdict=s["free_verdict"], parse_fail=s["free_verdict"] is None,
                    forced_answer=s["free_verdict"] is None, answer_text=s["answer_text"],
                    gen_text=tok.decode(s["ids"], skip_special_tokens=True),
                    n_gen_tokens=len(s["ids"]), latency_s=per,
                )
            )
        samples.setdefault(name, [])
        if len(samples[name]) < 6:
            samples[name].append(
                dict(prompt=prompts[0][:900], gen=out_rows[-len(chunk_rows)]["gen_text"][:900])
            )
        done = start + len(chunk_rows)
        print(f"[{name}] {done}/{len(recs)} {dt:.1f}s/batch forced={len(need)}", flush=True)

    del model
    torch.cuda.empty_cache()
    return pl.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# vllm backend (teacher)
# --------------------------------------------------------------------------- #
_VLLM: dict = {}


def _teacher_engine():
    """Load the teacher once per process - a 24B init costs minutes."""
    if _VLLM:
        return _VLLM["llm"], _VLLM["tok"], _VLLM["path"], _VLLM["g"], _VLLM["u"]

    from transformers import AutoTokenizer
    from vllm import LLM

    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # the teacher is cached; without this the load stalls indefinitely on
    # unauthenticated Hub metadata requests rather than on the GPU (observed:
    # 20 min at "Loading model from scratch", 15 MB read, GPU memory flat)
    os.environ["HF_HUB_OFFLINE"] = "1"

    # transformers picks its mistral-common backend whenever tekken.json sits
    # beside the HF files, and vLLM cannot wrap that backend. A directory holding
    # only the HF tokenizer files resolves to a normal fast LlamaTokenizer, which
    # both vLLM and the verdict-id derivation below can use.
    import glob
    import shutil
    import tempfile

    snaps = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{TEACHER.replace('/', '--')}/snapshots/*/"
    ))
    hf_tok_dir = None
    if snaps and "mistral" in TEACHER.lower():
        hf_tok_dir = tempfile.mkdtemp(prefix="mistral_hftok_")
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = pathlib.Path(snaps[0]) / f
            if src.exists():
                shutil.copy(src, hf_tok_dir)

    tok = AutoTokenizer.from_pretrained(hf_tok_dir or TEACHER)
    assert getattr(tok, "is_fast", False), f"teacher tokenizer is {type(tok).__name__}, not fast"
    g_ids, u_ids = verdict_token_ids(tok)
    print(f"[audit] teacher tokenizer={type(tok).__name__} "
          f"GROUNDED={g_ids} UNGROUNDED={u_ids}", flush=True)

    hf_tok = dict(tokenizer=hf_tok_dir, tokenizer_mode="auto", config_format="hf") if hf_tok_dir else {}

    llm, path = None, None
    for quant, kwargs in [
        ("fp8-hftok", dict(quantization="fp8", **hf_tok)),
        ("bfloat16-hftok", dict(dtype="bfloat16", **hf_tok)),
        ("fp8", dict(quantization="fp8")),
    ]:
        try:
            llm = LLM(
                model=TEACHER, max_model_len=4096, gpu_memory_utilization=0.92,
                enforce_eager=False, **kwargs,
            )
            path = quant
            print(f"[teacher] loaded via {quant}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[teacher] {quant} init failed: {type(e).__name__}: {e}", flush=True)
    if llm is None:
        raise RuntimeError("teacher failed to initialise on every quantisation path")
    _VLLM.update(llm=llm, tok=tok, path=path, g=g_ids, u=u_ids)
    return llm, tok, path, g_ids, u_ids


def run_vllm(rows: pl.DataFrame, samples: dict) -> tuple[pl.DataFrame, str]:
    from vllm import SamplingParams

    llm, tok, path, g_ids, u_ids = _teacher_engine()

    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": INSTRUCTION},
             {"role": "user", "content": user_block(r["chunk"], r["claim"])}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in rows.to_dicts()
    ]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))
    traces = [o.outputs[0] for o in outs]

    # same discipline as the tiny half: the score is read at a forced answer
    # position, not off the free trace
    fprompts = []
    for p, comp in zip(prompts, traces, strict=True):
        tail = comp.text.split("</think>")[0]
        for t in STRIP_TAIL:
            tail = tail.replace(t, "")
        fprompts.append(p + tail + "\nAnswer:")
    fouts = llm.generate(
        fprompts, SamplingParams(temperature=0.0, max_tokens=8, logprobs=20)
    )
    per = (time.time() - t0) / len(prompts)

    out_rows = []
    for r, comp, fo in zip(rows.to_dicts(), traces, fouts, strict=True):
        fc = fo.outputs[0]
        pieces = tok.batch_decode([[t] for t in fc.token_ids]) if fc.token_ids else []
        v, _ = parse_verdict("".join(pieces), first=True)
        margin, floored = None, False
        if fc.logprobs:
            # forced-choice margin at the answer position (first answer token)
            d = fc.logprobs[0]
            gv = [x.logprob for t, x in d.items() if t in g_ids]
            uv = [x.logprob for t, x in d.items() if t in u_ids]
            floor = min(x.logprob for x in d.values())
            if not gv:
                gv, floored = [floor], True
            if not uv:
                uv, floored = [floor], True
            margin = float(np.logaddexp.reduce(gv) - np.logaddexp.reduce(uv))
        fv, _ = parse_verdict(comp.text)
        out_rows.append(
            dict(model=TEACHER, pair_id=r["pair_id"], label=r["label"], control=r["control"],
                 neg_family=r["neg_family"], claim_form=r["claim_form"], margin=margin,
                 margin_floored=floored, verdict=v, free_verdict=fv, parse_fail=fv is None,
                 forced_answer=fv is None, answer_text=fc.text, gen_text=comp.text,
                 n_gen_tokens=len(comp.token_ids), latency_s=per)
        )
    samples[TEACHER] = [dict(prompt=prompts[0][:900], gen=out_rows[0]["gen_text"][:900])]
    return pl.DataFrame(out_rows), path


# --------------------------------------------------------------------------- #
def checkpoint(df: pl.DataFrame) -> None:
    if SCORES.exists():
        old = pl.read_parquet(SCORES)
        df = pl.concat([old, df], how="diagonal_relaxed")
    # a pair_id carries BOTH its positive and its negative - the row identity is
    # (model, pair_id, label). Keying on (model, pair_id) drops one label per pair
    # and leaves an all-positive set with no computable AUROC.
    df.unique(subset=["model", "pair_id", "label"], keep="last").write_parquet(SCORES)


def already(model: str) -> set[int]:
    """Row keys already scored, as pair_id*2+label."""
    if not SCORES.exists():
        return set()
    d = pl.read_parquet(SCORES).filter(pl.col("model") == model)
    return set((d["pair_id"] * 2 + d["label"]).to_list())


def load_samples() -> dict:
    return json.loads(SAMPLES.read_text()) if SAMPLES.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["transformers", "vllm"], required=True)
    ap.add_argument("--models", default=None, help="comma-separated subset")
    ap.add_argument("--force", action="store_true", help="score the full set despite a failed gate")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap rows per split")
    ap.add_argument(
        "--teacher-model", default=None,
        help="substitute teacher (the registered Mistral-Small-24B has no cached weights)",
    )
    args = ap.parse_args()
    if args.teacher_model:
        global TEACHER
        TEACHER = args.teacher_model

    ev = pl.read_parquet(EVALSET)
    if args.limit:
        global SCORES, GATE_LOG, SAMPLES
        SCORES = HERE / "R17-H143_scores.SMOKE.parquet"
        GATE_LOG = HERE / "R17-H143_gate.SMOKE.json"
        SAMPLES = HERE / "R17-H143_gen_samples.SMOKE.json"
        ev = pl.concat([
            ev.filter(pl.col("control")).head(args.limit),
            ev.filter(~pl.col("control")).head(args.limit),
        ])
    controls = ev.filter(pl.col("control"))
    real = ev.filter(~pl.col("control"))
    samples = load_samples()
    gates = json.loads(GATE_LOG.read_text()) if GATE_LOG.exists() else {}

    targets = [TEACHER] if args.backend == "vllm" else list(MODELS)
    if args.models:
        want = set(args.models.split(","))
        targets = [t for t in targets if t in want or t.split("/")[-1] in want]

    for name in targets:
        cfg = MODELS.get(name, dict(params_M=23600, style="chat", batch=1))
        done = already(name)
        print(f"\n===== {name} (already scored: {len(done)}) =====", flush=True)
        try:
            # --- positive-control gate first -------------------------------- #
            key = pl.col("pair_id") * 2 + pl.col("label")
            todo_ctrl = controls.filter(~key.is_in(list(done))) if done else controls
            if todo_ctrl.height:
                t0 = time.time()
                if args.backend == "vllm":
                    cdf, path = run_vllm(todo_ctrl, samples)
                    gates.setdefault(name, {})["quant_path"] = path
                else:
                    cdf = run_transformers(name, cfg, todo_ctrl, samples)
                checkpoint(cdf)
                print(f"[{name}] controls scored in {time.time() - t0:.0f}s", flush=True)

            cd = pl.read_parquet(SCORES).filter(pl.col("model") == name, pl.col("control"))
            m = cd["margin"].to_numpy().astype(float)
            fallback = np.nanmean(np.abs(m)) if np.isfinite(m).any() else 1.0
            sc = np.where(
                np.isfinite(m), m,
                np.where(cd["verdict"].to_numpy() == "GROUNDED", fallback,
                         np.where(cd["verdict"].to_numpy() == "UNGROUNDED", -fallback, 0.0)),
            )
            ctrl_auroc = auroc(cd["label"].to_numpy(), sc)
            exempt = cfg.get("gate_exempt", False)
            gates.setdefault(name, {}).update(
                control_auroc=ctrl_auroc, gate_exempt=exempt,
                passed=bool(exempt or (ctrl_auroc or 0) >= CONTROL_GATE),
                parse_fail_rate=float(cd["parse_fail"].mean()),
            )
            GATE_LOG.write_text(json.dumps(gates, indent=2))
            SAMPLES.write_text(json.dumps(samples, indent=2))
            print(f"[{name}] CONTROL AUROC = {ctrl_auroc} exempt={exempt}", flush=True)

            if not gates[name]["passed"] and not args.force:
                print(f"[{name}] GATE FAILED (<{CONTROL_GATE}) - skipping full set, fix prompt",
                      flush=True)
                continue

            # --- full eval set ---------------------------------------------- #
            done = already(name)
            key = pl.col("pair_id") * 2 + pl.col("label")
            todo = real.filter(~key.is_in(list(done))) if done else real
            print(f"[{name}] scoring {todo.height} real pairs", flush=True)
            step = 200 if args.backend == "vllm" else 10_000
            for s in range(0, todo.height, step):
                part = todo.slice(s, step)
                if args.backend == "vllm":
                    df, _ = run_vllm(part, samples)
                else:
                    df = run_transformers(name, cfg, part, samples)
                checkpoint(df)
                SAMPLES.write_text(json.dumps(samples, indent=2))
            print(f"[{name}] DONE", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            gates.setdefault(name, {})["error"] = f"{type(e).__name__}: {e}"
            GATE_LOG.write_text(json.dumps(gates, indent=2))
            print(f"[{name}] SKIPPED: {e}", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
