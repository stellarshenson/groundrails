"""groundrails - grounding guardrails for agentic RAG.

Deterministic, torch-free claim verification. Core exports (zero heavy deps):
:func:`ground`, :func:`ground_batch`, :class:`GroundingMatch`, :class:`Location`.

Optional semantic grounding (NLI / cross-encoder + FAISS) lives in
:mod:`groundrails.semantic` and requires ``groundrails[semantic]``. It is
lazy-imported - ``import groundrails`` does NOT load torch, transformers, or faiss.
"""

from groundrails.grounding import (
    GroundingMatch,
    Location,
    UnsupportedLanguageError,
    build_grounding_document,
    ground,
    ground_batch,
    grounding_document,
)

__all__ = [
    "GroundingMatch",
    "Location",
    "UnsupportedLanguageError",
    "build_grounding_document",
    "ground",
    "ground_batch",
    "grounding_document",
]
