"""groundrails - grounding guardrails for agentic RAG.

Deterministic, torch-free claim verification. Core exports (zero heavy deps):
:func:`ground`, :func:`ground_batch`, :class:`GroundingMatch`, :class:`Location`.

Optional semantic grounding (NLI / cross-encoder + FAISS) lives in
:mod:`groundrails.semantic` and requires ``groundrails[semantic]``. It is
lazy-imported - ``import groundrails`` does NOT load torch, transformers, or faiss.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from groundrails.grounding import (
    GroundingMatch,
    Location,
    UnsupportedLanguageError,
    build_grounding_document,
    ground,
    ground_batch,
    grounding_document,
)

try:
    __version__ = _pkg_version("groundrails")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"

__all__ = [
    "GroundingMatch",
    "Location",
    "UnsupportedLanguageError",
    "__version__",
    "build_grounding_document",
    "ground",
    "ground_batch",
    "grounding_document",
]
