"""Single-engine OpenVINO int8 semantic cascade - the serving substrate.

The `semantic` effort tier escalates here when the lexical manifold's win is not
clear. Everything runs on one runtime (OpenVINO int8, torch-free): a bge-m3 bi-encoder
pre-filters the evidence chunks, then a bge-reranker (relevance) and an mDeBERTa NLI
(entailment) cross-encoder score the survivors. Per claim the engine returns the
max-over-chunks signals the joint verdict head consumes.

Models are the published int8 IRs on the HuggingFace Hub (built offline by
experiments/grounding-semantic/build_ov_grounder.py); the IR, config and tokenizer ship
together so no base model is needed. Heavy deps (`openvino` is core; `transformers` is
the tokenizer, in the `semantic-grounder` extra) are imported lazily - importing this
module is cheap, and `is_available()` gates the layer when the extra is absent.

This is a lean serving copy of the validated functions in
experiments/grounding-semantic/grounding_openvino.py (the research record); src keeps no
dependency on the experiments tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

# Published OpenVINO int8 IRs (short name -> HF repo id).
HF_REPOS = {
    "bge-m3": "stellars/bge-m3-openvino-int8",
    "bge-reranker-v2-m3": "stellars/bge-reranker-v2-m3-openvino-int8",
    "mDeBERTa-v3-nli": "stellars/mdeberta-v3-base-mnli-xnli-openvino-int8",
}

MAX_LEN = 512

# Operating points fit out-of-fold on the gold (experiments/grounding-semantic):
# claims whose max claim-chunk cosine is outside the gate skip BOTH cross-encoders;
# claims whose reranker max is outside the band skip the NLI forward.
COSINE_GATE = (0.493, 0.739)
CASCADE_BAND = (0.01, 0.66)


def is_available() -> bool:
    """True iff the serving deps (openvino + transformers + huggingface_hub) import."""
    for mod in ("openvino", "transformers", "huggingface_hub"):
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


def install_hint() -> str:
    return (
        "The semantic cascade needs openvino (core) + transformers + huggingface_hub. "
        "Install the extra:\n  pip install 'groundrails[semantic-grounder]'\n"
        "The int8 model IRs (~1.4 GB) download from the HuggingFace Hub on first use."
    )


def download_models():
    """Pre-fetch the int8 cascade IRs into the HuggingFace cache so the first
    ``--semantic 1`` run is warm instead of paying a ~1.4 GB download.

    Yields ``(short_name, repo_id, local_dir)`` as each IR finishes downloading.
    These are the only model weights groundrails downloads - the lexical tiers need
    none. Requires the ``semantic-grounder`` extra; raises ImportError otherwise.
    """
    if not is_available():
        raise ImportError(install_hint())
    from huggingface_hub import snapshot_download

    for name, repo in HF_REPOS.items():
        yield name, repo, snapshot_download(repo)


def _feed(cm, enc):
    names = {i.get_any_name() for i in cm.inputs}
    return {n: enc[n].astype(np.int64) for n in enc if n in names}


def _bucket(chunks):
    """Order chunk indices by length so each padded batch holds similar-length rows.
    The grounder takes the max over chunks, so reordering is result-invariant."""
    return sorted(range(len(chunks)), key=lambda i: len(chunks[i]))


@dataclass
class CascadeScores:
    """Max-over-chunks signals for one claim. ``ran_rr`` / ``ran_nli`` record which
    cross-encoders actually fired (the gate / band may skip them)."""

    cos_max: float = 0.0
    rr_max: float = 0.0
    nli_ent: float = 0.0
    nli_contra: float = 0.0
    ran_rr: bool = False
    ran_nli: bool = False


@dataclass
class SemanticCascade:
    """Lazy-loaded OpenVINO int8 cascade. Call :meth:`score` per claim.

    The three IRs compile on first :meth:`score`. ``top_k`` is the pre-filter survivor
    count; ``gate`` and ``band`` are the cosine / reranker early-exit operating points.
    """

    top_k: int = 8
    gate: tuple[float, float] = COSINE_GATE
    band: tuple[float, float] = CASCADE_BAND
    hint: str = "LATENCY"
    pool: str = "cls"
    _emb: object = field(default=None, repr=False)
    _rr: object = field(default=None, repr=False)
    _nli: object = field(default=None, repr=False)
    _etok: object = field(default=None, repr=False)
    _rtok: object = field(default=None, repr=False)
    _ntok: object = field(default=None, repr=False)
    _ent_idx: int = field(default=0, repr=False)
    _contra_idx: int = field(default=2, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def _load(self) -> None:
        if self._loaded:
            return
        if not is_available():
            raise ImportError(install_hint())
        import openvino as ov
        from transformers import AutoTokenizer

        core = ov.Core()

        def load(name):
            from huggingface_hub import snapshot_download

            d = Path(snapshot_download(HF_REPOS[name]))
            tok = AutoTokenizer.from_pretrained(d, fix_mistral_regex=False)
            cm = core.compile_model(
                core.read_model(str(d / "openvino_model.xml")),
                "CPU",
                {"PERFORMANCE_HINT": self.hint},
            )
            return cm, tok, d

        self._emb, self._etok, _ = load("bge-m3")
        self._rr, self._rtok, _ = load("bge-reranker-v2-m3")
        self._nli, self._ntok, ndir = load("mDeBERTa-v3-nli")
        id2label = {
            int(k): v.lower()
            for k, v in json.loads((ndir / "config.json").read_text("utf-8"))["id2label"].items()
        }
        self._ent_idx = next(i for i, v in id2label.items() if "entail" in v)
        self._contra_idx = next(i for i, v in id2label.items() if "contra" in v)
        self._loaded = True

    # -- scoring helpers (lean copies of grounding_openvino) -------------------

    def _embed(self, texts):
        cm, tok = self._emb, self._etok
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(
                texts[i : i + 32],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="np",
            )
            h = cm(_feed(cm, enc))[cm.output(0)]
            v = (
                h[:, 0]
                if self.pool == "cls"
                else (
                    (h * enc["attention_mask"][..., None]).sum(1)
                    / np.clip(enc["attention_mask"][..., None].sum(1), 1e-9, None)
                )
            )
            out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9))
        return np.concatenate(out)

    def _rerank_max(self, claim, chunks):
        cm, tok = self._rr, self._rtok
        order = _bucket(chunks)
        best = -1e9
        for i in range(0, len(order), 16):
            sub = [chunks[j] for j in order[i : i + 16]]
            enc = tok(
                [claim] * len(sub),
                sub,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="np",
            )
            lg = cm(_feed(cm, enc))[cm.output(0)].reshape(-1)
            best = max(best, float((1.0 / (1.0 + np.exp(-lg))).max()))
        return best

    def _nli_channels_max(self, claim, chunks):
        cm, tok = self._nli, self._ntok
        order = _bucket(chunks)
        ent_best, con_best = -1e9, -1e9
        for i in range(0, len(order), 16):
            sub = [chunks[j] for j in order[i : i + 16]]
            enc = tok(
                sub,
                [claim] * len(sub),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="np",
            )
            lg = cm(_feed(cm, enc))[cm.output(0)]
            ex = np.exp(lg - lg.max(1, keepdims=True))
            p = ex / ex.sum(1, keepdims=True)
            ent_best = max(ent_best, float(p[:, self._ent_idx].max()))
            con_best = max(con_best, float(p[:, self._contra_idx].max()))
        return ent_best, con_best

    def score(self, claim: str, chunks: list[str]) -> CascadeScores:
        """Score one claim against its evidence chunks - returns max-over-chunks signals.

        Applies the cosine gate (extreme tails skip both cross-encoders) and the cascade
        band (out-of-band reranker scores skip the NLI), matching the deployed serving
        path. Signals not computed stay 0 with ``ran_*`` False.
        """
        self._load()
        if not chunks:
            return CascadeScores()
        cv = self._embed([claim])[0]
        dv = self._embed(chunks)
        cos = dv @ cv
        cos_max = float(cos.max())
        res = CascadeScores(cos_max=cos_max)

        # stage-0 cosine gate: extreme tails resolve at embed cost
        if cos_max <= self.gate[0] or cos_max >= self.gate[1]:
            return res

        ranked = [chunks[j] for j in np.argsort(-cos)[: self.top_k]]
        res.rr_max = self._rerank_max(claim, ranked)
        res.ran_rr = True
        # cascade band: only the uncertain reranker scores pay for the NLI forward
        if self.band[0] <= res.rr_max <= self.band[1]:
            res.nli_ent, res.nli_contra = self._nli_channels_max(claim, ranked)
            res.ran_nli = True
        return res
