"""Claims schema - the pydantic model a `ground` run validates against.

A claims file is one of: a JSON list of strings, a JSON list of ``{claim, ...}`` objects
(as ``extract-claims`` writes), or a plain-text document with one claim per non-empty line.
All three normalise to a list of validated :class:`Claim` objects, so a malformed file
fails with a clear schema error instead of a silent miss. ``extract-claims`` emits through
the same model, so reader and writer share one schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Claim(BaseModel):
    """One claim to ground. ``claim`` text is required and non-empty; ``id``,
    ``line_number`` and the ``char_start`` / ``char_end`` span locate the claim in the
    answer document - all carried through from extract-claims when present."""

    model_config = ConfigDict(extra="ignore")

    claim: str = Field(min_length=1)
    id: str | None = None
    line_number: int | None = None
    char_start: int | None = None  # 0-based offset of the claim in the answer document
    char_end: int | None = None  # 0-based exclusive end offset


_ADAPTER = TypeAdapter(list[Claim])


def parse_claims(raw: object) -> list[Claim]:
    """Validate already-loaded data (a list of strings or ``{claim,...}`` objects)."""
    if not isinstance(raw, list):
        raise ValueError("claims must be a JSON list of strings or {claim,...} objects")
    items = [{"claim": x} if isinstance(x, str) else x for x in raw]
    return _ADAPTER.validate_python(items)


def load_claims(path: str | Path) -> list[Claim]:
    """Read and validate a claims file: JSON (list of strings or objects) or plain text
    (one claim per non-empty line). Raises ``FileNotFoundError``, ``ValueError``, or
    pydantic ``ValidationError`` on a non-conforming file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return parse_claims(raw)
