"""Optional-dependency guards for the dataset pipeline.

The pipeline preprocesses training corpora; it is not part of the grounder
runtime and carries its own extra. Nothing here is imported at
``import groundrails`` time, and every heavy import is deferred to the call that
needs it - so a missing extra surfaces as an install line at the first dataset
call rather than as an import error on an unrelated command.
"""

from __future__ import annotations

import importlib

EXTRA = "pip install groundrails[dataset]"


def require(module: str, purpose: str):
    """Import ``module`` or fail naming the extra that ships it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"the dataset pipeline needs `{module}` for {purpose} - install it with `{EXTRA}`"
        ) from exc


def polars():
    """The frame engine every stage reads and writes with."""
    return require("polars", "reading and writing corpus frames")
