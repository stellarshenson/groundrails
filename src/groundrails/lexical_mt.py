"""HIGH-tier machine-translation bridge: CTranslate2 + native-OpenVINO SaT sentence splitting.

Translate-then-recall lever for the lexical grounder's HIGH effort tier. Torch-free:
translation runs on CTranslate2 (int8 CPU), sentence segmentation on a native OpenVINO
INT8 SaT (`document_processing.sat`, LATENCY hint) - no onnxruntime.

- reuses the CTranslate2 + tokenizer models argos already downloaded under ~/.local/share/argos-translate/packages/
- handles both argos tokenizer formats - SentencePiece (sentencepiece.model) and subword-nmt BPE (bpe.model)
- imports no torch (SaT replaces argos's stanza segmenter, the stack's last torch dependency)
- pass-through for English / unknown / unsupported source language
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_ARGOS = Path.home() / ".local/share/argos-translate/packages"
_ISO = {"no": "nb", "nn": "nb"}  # langdetect -> argos model code
_MODELS: dict = {}
_SAT = None


def _sat():
    global _SAT
    if _SAT is None:
        from groundrails.sat import SaTSegmenter

        _SAT = SaTSegmenter()  # native OpenVINO INT8 SaT (LATENCY hint), no onnxruntime
    return _SAT


def _find_pkg(src: str) -> Path | None:
    if not _ARGOS.exists():
        return None
    for d in _ARGOS.iterdir():
        meta = d / "metadata.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            if m.get("from_code") == src and m.get("to_code") == "en":
                return d
    return None


def _load(src: str):
    """Load CTranslate2 translator + the package's tokenizer (SentencePiece or BPE)."""
    if src in _MODELS:
        return _MODELS[src]
    import ctranslate2

    d = _find_pkg(src)
    if d is None:
        _MODELS[src] = None
        return None
    tr = ctranslate2.Translator(str(d / "model"), device="cpu", compute_type="int8")
    if (d / "sentencepiece.model").exists():
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(str(d / "sentencepiece.model"))
        _MODELS[src] = {"tr": tr, "kind": "spm", "sp": sp}
    elif (d / "bpe.model").exists():
        from sacremoses import MosesDetokenizer, MosesTokenizer
        from subword_nmt.apply_bpe import BPE

        bpe = BPE(open(d / "bpe.model", encoding="utf-8"))
        _MODELS[src] = {
            "tr": tr,
            "kind": "bpe",
            "bpe": bpe,
            "mtok": MosesTokenizer(lang=src),
            "detok": MosesDetokenizer(lang="en"),
        }
    else:
        _MODELS[src] = None
    return _MODELS[src]


def translate(text: str, src_iso: str) -> str:
    """Translate text into English. Pass-through for English/unknown/unsupported source."""
    code = _ISO.get(src_iso, src_iso)
    if code in ("en", "und", ""):
        return text
    m = _load(code)
    if m is None and ensure_model(code):  # not loaded - try an on-demand argos install
        m = _load(code)
    if m is None:
        logger.warning(
            "argos model for %s->en unavailable (auto-install skipped offline or no such "
            "package) - translate-then-recall skipped for this claim; install manually with "
            "`argospm install translate-%s_en`",
            code,
            code,
        )
        return text
    tr = m["tr"]
    out = []
    for s in _sat().split(text) or [text]:
        s = s.strip()
        if not s:
            continue
        if m["kind"] == "spm":
            tokens = m["sp"].encode(s, out_type=str)
            res = tr.translate_batch([tokens], beam_size=2, max_decoding_length=256)
            out.append("".join(res[0].hypotheses[0]).replace("▁", " ").strip())
        else:  # subword-nmt BPE: moses-tokenise -> apply BPE -> translate -> detokenise
            toks = m["bpe"].process_line(m["mtok"].tokenize(s, return_str=True)).split()
            res = tr.translate_batch([toks], beam_size=2, max_decoding_length=256)
            merged = " ".join(res[0].hypotheses[0]).replace("@@ ", "").replace("@@", "")
            out.append(m["detok"].detokenize(merged.split()).strip())
    return " ".join(o for o in out if o).strip() or text


def has_model(src_iso: str) -> bool:
    """True when the claim's language can reach English for grounding.

    English / undetermined (``en``/``und``/``""``) need no bridge and always pass;
    any other language passes only when an installed argos ``<src>->en`` model
    exists (after the ``no``/``nn`` -> ``nb`` mapping in ``_ISO``). The grounder's
    unsupported-language guard blocks claims for which this returns False.
    """
    code = _ISO.get(src_iso, src_iso)
    if code in ("en", "und", ""):
        return True
    return _find_pkg(code) is not None


def _auto_install_enabled() -> bool:
    """On-demand argos install is on by default, off when offline or explicitly disabled.

    Disabled when ``GROUNDRAILS_ARGOS_AUTO_INSTALL=0`` or when ``HF_HUB_OFFLINE`` is set
    (an offline run must never reach the network), so CI / offline test runs keep the
    pure-check behaviour and download nothing.
    """
    if os.environ.get("GROUNDRAILS_ARGOS_AUTO_INSTALL", "1") == "0":
        return False
    if os.environ.get("HF_HUB_OFFLINE"):
        return False
    return True


def install_model(src_iso: str) -> bool:
    """Download and install the argos ``<src>->en`` model on demand. Returns availability.

    No-op (True) for English / undetermined and for an already-installed pair. Network
    failures and absent packages degrade to a warning + False - never raise.
    """
    code = _ISO.get(src_iso, src_iso)
    if code in ("en", "und", ""):
        return True
    if _find_pkg(code) is not None:
        return True
    try:
        import argostranslate.package as _pkg

        _pkg.update_package_index()
        match = next(
            (
                p
                for p in _pkg.get_available_packages()
                if p.from_code == code and p.to_code == "en"
            ),
            None,
        )
        if match is None:
            logger.warning("no argos %s->en package available to install", code)
            return False
        logger.info("installing argos %s->en model on demand", code)
        _pkg.install_from_path(match.download())
        _MODELS.pop(code, None)  # drop any cached None so _load re-resolves
        return _find_pkg(code) is not None
    except Exception as exc:
        logger.warning("on-demand argos %s->en install failed: %s", code, exc)
        return False


def ensure_model(src_iso: str) -> bool:
    """True when the ``<src>->en`` bridge is available, installing it on demand if missing.

    Same contract as :func:`has_model` (English / undetermined always pass, others need an
    installed pair) but, when the pair is missing and auto-install is enabled, it downloads
    the argos model first. Offline / disabled falls back to the pure check.
    """
    code = _ISO.get(src_iso, src_iso)
    if code in ("en", "und", ""):
        return True
    if _find_pkg(code) is not None:
        return True
    if not _auto_install_enabled():
        return False
    return install_model(code)
