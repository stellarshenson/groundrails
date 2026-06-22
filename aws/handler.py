"""AWS Lambda handler exposing the lexical groundrails grounder.

Cold start provisions the grounder from S3 via ``groundrails.init`` (the
calibration JSON is pulled from ``GROUNDRAILS_SOURCE``; models are skipped). At
the low effort tier with same-language claims the lexical path needs no model
weights - recall is exact / fuzzy / BM25 fused by a frozen logistic, and the
claims arrive already split. The SaT sentence segmenter and the CTranslate2
translator are HIGH-tier cross-lingual only (the MT bridge) and stay out of
scope here. Each invoke grounds the event's claims against its sources and
returns the grounding document.

Event shape::

    {
      "claims":  ["The tower is in Paris.", "It is 2000 m tall."],
      "sources": [["evidence.txt", "The Eiffel Tower is in Paris. It is 330 m tall."]],
      "effort":  "low"            # optional; overrides GROUNDRAILS_EFFORT
    }

``sources`` entries may be a plain string or a ``[path, text]`` pair (the path
carries provenance into the support location).
"""

from __future__ import annotations

import os

import groundrails
from groundrails.config import load_document_processing_config

_CFG = None


def _ready(effort: str) -> None:
    """Cold-start provisioning: pull calibration from S3 once, build the config."""
    global _CFG
    if _CFG is not None:
        return
    src = os.environ.get("GROUNDRAILS_SOURCE")
    if src:
        groundrails.init(
            source=src,
            models="none",  # low-effort same-language lexical needs none (SaT/MT is HIGH-tier cross-lingual)
            wordnet=False,
            home="/tmp/groundrails",  # the only writable path in Lambda  # noqa: S108
            aws_region=os.environ.get("AWS_REGION"),
        )
    _CFG = load_document_processing_config().overlay(lexical_effort=effort)


def handler(event, context):
    effort = (event or {}).get("effort") or os.environ.get("GROUNDRAILS_EFFORT", "low")
    _ready(effort)
    claims = (event or {}).get("claims") or []
    raw = (event or {}).get("sources") or []
    sources = [tuple(s) if isinstance(s, (list, tuple)) else s for s in raw]
    doc = groundrails.grounding_document(
        claims,
        sources,
        config=_CFG,
        semantic=False,
        ignore_language=True,
        max_workers=1,
    )
    return {"ok": True, "effort": effort, "grounding": doc}
