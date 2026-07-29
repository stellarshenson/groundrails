"""R6-H47 vocabulary-coverage pre-flight + R6-H46 token budget.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 6).

A prompt format built from strings the tokenizer does not carry is fed to the
model as sub-word noise in the position where structure was expected, so the
model's trained format is never actually invoked and any score taken under it is
void. This audits every candidate's DECLARED format against its OWN vocabulary
before a single weight is loaded, and measures the assembled serving unit against
the model's real context window.

Two verdicts per candidate:
  R6-H47  does every control string in its declared format exist as one token
  R6-H46  does the serving unit fit the context, single-source and multi-source

Run:  uv run python experiments/grounding-semantic/R6-H47_vocab_gate.py
"""

import json
import pathlib

from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer

CANDIDATES = ["PleIAs/Monad", "PleIAs/Baguettotron", "PleIAs/Pleias-RAG-350M"]
CTX = {"PleIAs/Monad": 2048, "PleIAs/Baguettotron": 4096, "PleIAs/Pleias-RAG-350M": 4096}
CHUNK_CHARS = 1500  # cfg.chunk_max_chars - the serving unit
TOP_K = 3  # cfg.semantic_top_k - sources in a multi-source prompt

# The control strings each declared format emits. ChatML for the two SYNTH
# models (from their chat_template.json); the structured protocol for the RAG
# model, which ships no template because its format IS these tokens.
CHATML = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
PROTOCOL = [
    "<|query_start|>",
    "<|query_end|>",
    "<|source_start|>",
    "<|source_id|>",
    "<|source_end|>",
    "<|language_start|>",
    "<|language_end|>",
    "<|query_analysis_start|>",
    "<|query_analysis_end|>",
    "<|query_report_start|>",
    "<|query_report_end|>",
    "<|source_analysis_start|>",
    "<|source_analysis_end|>",
    "<|source_report_start|>",
    "<|source_report_end|>",
    "<|draft_start|>",
    "<|draft_end|>",
    "<|answer_start|>",
    "<|answer_end|>",
]


# Control tokens that carry no <|...|> shape but are still format-bearing.
EXTRA_CONTROL = frozenset({"<think>", "</think>", "[PAD]", "[UNK]"})


def declared_format(tok, mid):
    """The template the repo ships, or the protocol tokens when it ships none.

    `tok.chat_template` is NOT sufficient: Monad ships its template in a separate
    `chat_template.json`, which this transformers version does not fold into the
    attribute. Reading only the attribute silently classified Monad as a
    protocol model and audited it against the wrong 19 tokens - a right verdict
    for a wrong reason, which is the exact failure this gate exists to catch.
    """
    tmpl = tok.chat_template
    if not tmpl:
        path = try_to_load_from_cache(mid, "chat_template.json")
        if isinstance(path, str) and pathlib.Path(path).exists():
            tmpl = json.loads(pathlib.Path(path).read_text()).get("chat_template")
    if tmpl:
        src = "chat_template attribute" if tok.chat_template else "chat_template.json"
        return f"ChatML ({src})", CHATML, tmpl
    return "structured protocol (no chat_template)", PROTOCOL, None


def coverage(tok, strings):
    out = []
    for s in strings:
        i = tok.convert_tokens_to_ids(s)
        ok = i is not None and i != tok.unk_token_id
        out.append((s, i if ok else None, [] if ok else tok.tokenize(s)))
    return out


def control_tokens_in_vocab(tok, limit=40):
    """Every control-shaped token the model actually has - the real format."""
    found = [
        (i, t)
        for t, i in tok.get_vocab().items()
        if (t.startswith("<|") and t.endswith("|>")) or t in EXTRA_CONTROL
    ]
    return sorted(found)[:limit]


def main():
    prose = pathlib.Path(__file__).parent / "private-prose-forensics" / "sources"
    sample = next(p for p in sorted(prose.iterdir())).read_text(encoding="utf-8")
    claim = "The dam is 271 metres tall and produces 1300 MW of power."
    chunk = sample[:CHUNK_CHARS]

    verdicts = {}
    for mid in CANDIDATES:
        tok = AutoTokenizer.from_pretrained(mid)
        kind, strings, tmpl = declared_format(tok, mid)
        cov = coverage(tok, strings)
        missing = [c for c in cov if c[1] is None]
        print("=" * 96)
        print(f"{mid}   vocab {tok.vocab_size}   ctx {CTX[mid]}   format: {kind}")
        print(
            f"  R6-H47 control-string coverage: {len(cov) - len(missing)}/{len(cov)} in vocabulary"
        )
        for s, i, pieces in cov:
            if i is None:
                print(f"    MISSING  {s:28s} -> {len(pieces)} pieces {pieces[:8]}")
        if not missing:
            print("    all present as single tokens")
        print(
            f"  control tokens actually in vocab: "
            f"{[t for _, t in control_tokens_in_vocab(tok)][:12]}"
        )

        # Sequence-initial token: an EOS prepended as BOS is R6-H41's target.
        first = tok("probe")["input_ids"][0]
        print(f"  sequence-initial token: id={first} ({tok.convert_ids_to_tokens(first)})")
        if tmpl:
            probe = "<|im_start|>user\nX<|im_end|>\n<|im_start|>assistant\n<think>\n"
            n = len(tok(probe, add_special_tokens=False)["input_ids"])
            print(f"  rendered template probe: {n} tokens (12 = clean control tokens)")

        # R6-H46 - the assembled serving unit, both shapes.
        single = f"Evidence:\n{chunk}\n\nClaim: {claim}\n\nIs the claim supported?"
        multi = (
            "\n\n".join(f"Source {i + 1}:\n{chunk}" for i in range(TOP_K)) + f"\n\nClaim: {claim}"
        )
        n1 = len(tok(single)["input_ids"])
        nk = len(tok(multi)["input_ids"])
        print(
            f"  R6-H46 budget: single-source {n1} tok ({n1 / CTX[mid]:.0%} of ctx), "
            f"{TOP_K}-source {nk} tok ({nk / CTX[mid]:.0%} of ctx)"
        )

        verdicts[mid] = {
            "h47": "PASS" if not missing else f"FAIL - {len(missing)}/{len(cov)} missing",
            "h46_single": "PASS" if n1 <= CTX[mid] else "OVERFLOW",
            "h46_multi": "PASS" if nk <= CTX[mid] else "OVERFLOW",
            "initial_token": tok.convert_ids_to_tokens(first),
        }

    print("\n" + "=" * 96)
    print("R6-H47 / R6-H46 GATE RESULT")
    print("=" * 96)
    for mid, v in verdicts.items():
        print(
            f"  {mid:28s} H47 {v['h47']:26s} H46 single {v['h46_single']:9s} "
            f"{TOP_K}-src {v['h46_multi']:9s} init {v['initial_token']}"
        )
    print("\n  a FAIL on H47 voids every score that candidate carries under that format")
    out = pathlib.Path(__file__).parent / "R6-H47_gate.json"
    out.write_text(json.dumps(verdicts, indent=2))
    print(f"  verdicts -> {out}")


if __name__ == "__main__":
    main()
