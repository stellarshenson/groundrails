"""Common configuration access interface for every plugin.

One shared module exposing per-plugin schemas (dataclasses) and a
generic yaml loader. **Yaml is the single source of truth** for every
tunable value; the dataclasses here are typed schemas (no defaults)
that describe the expected shape.

File layout (same ``config_<plugin>.yaml`` filename in every location):

- Bundled default (ships with the package):
  ``groundrails/config_<plugin>.yaml``.
- Project-local override:
  ``./.stellars-plugins/config_<plugin>.yaml``.
- User-level override:
  ``~/.stellars-plugins/config_<plugin>.yaml``.

Resolution order for every plugin, first match wins:

1. Explicit ``path`` argument passed to the plugin's load function.
2. ``./.stellars-plugins/config_<plugin>.yaml`` project-local override.
3. ``~/.stellars-plugins/config_<plugin>.yaml`` user-level override.
4. Bundled default at ``<package>/config_<plugin>.yaml``.

Layer 2 beats layer 3 intentionally: a committed project-local override
is an explicit project choice; user-level tuning only applies when no
project override exists.

Any missing required field raises :class:`ConfigError` - no hardcoded
fallbacks. Unknown yaml keys are ignored (forward compatibility). Invalid
yaml syntax raises :class:`ConfigError` wrapping the parser exception.

Add a new plugin by:

1. Declare a ``@dataclass`` with type-annotated (no-default) fields here.
2. Ship the bundled default at
   ``groundrails/config_<plugin>.yaml``.
3. Add a ``load_<plugin>_config(path=None) -> <schema>`` wrapper that
   calls :func:`_load_typed_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import threading
from typing import Literal, TypeVar, get_args, get_origin, get_type_hints

import yaml

PACKAGE_ROOT = Path(__file__).parent
PROJECT_OVERRIDE_DIR = Path(".stellars-plugins")
USER_OVERRIDE_DIR_NAME = ".stellars-plugins"


def _user_override_dir() -> Path:
    """Resolve the user-level override directory at call time.

    Using ``Path.home()`` at call time (rather than at import) lets tests
    monkeypatch ``HOME`` or ``Path.home`` to redirect user config to a
    temp directory.
    """
    return Path.home() / USER_OVERRIDE_DIR_NAME


class ConfigError(RuntimeError):
    """Raised when a subsystem configuration cannot be resolved.

    Common causes: missing yaml, missing required field in yaml, invalid
    yaml syntax, yaml that isn't a top-level mapping. Deliberately loud
    because running with partial or hallucinated defaults would produce
    wrong results silently.
    """


# Parsed-YAML cache keyed on (path, mtime_ns, size). ``ground()`` resolves
# config (and the calibration block) per call, which re-read and re-parsed
# the same yaml for every claim - profiled at ~50% of per-claim runtime on
# a batch. The stat-stamp key keeps override semantics: touching or editing
# any config file invalidates its entry on the next call. Callers must not
# mutate the returned mapping. Unbounded by design: growth is one small
# parsed mapping per DISTINCT config path (4 layers per plugin in normal
# use); only a caller looping over many explicit per-tenant paths grows it.
_RAW_YAML_CACHE: dict[str, tuple[tuple[int, int], object]] = {}
_RAW_YAML_LOCK = threading.Lock()


def load_raw_yaml(path: Path) -> object:
    """Parse ``path`` as YAML with an mtime/size-keyed cache.

    Returns whatever the yaml contains (mapping, ``None`` for an empty
    file, ...) - callers keep their own shape checks. Raises the underlying
    ``OSError`` / ``yaml.YAMLError``; wrapping is the caller's concern.
    """
    st = path.stat()
    key = str(path)
    stamp = (st.st_mtime_ns, st.st_size)
    with _RAW_YAML_LOCK:
        hit = _RAW_YAML_CACHE.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    with _RAW_YAML_LOCK:
        _RAW_YAML_CACHE[key] = (stamp, raw)
    return raw


# --- document-processing schema -------------------------------------------


@dataclass
class GroundingConfig:
    """Tunable parameters for the document-processing grounding pipeline.

    Every field is type-annotated with no default - values are sourced
    from ``config_document_processing.yaml``. Add a new field here and a
    matching entry in the yaml together; never one without the other.
    """

    # ── match_type thresholds ────────────────────────────────────────────
    fuzzy_threshold: float
    """Levenshtein partial-ratio in [0, 1] above which match_type=fuzzy."""

    bm25_threshold: float
    """BM25 token-recall in [0, 1] above which match_type=bm25."""

    bm25_min_claim_tokens: int
    """Minimum unique claim tokens for the BM25 layer to confirm on its own.

    Token recall over a tiny claim saturates - a 3-token claim has a 3-term
    denominator, so a single passage can clear ``bm25_threshold`` on almost
    no evidence. Below this floor the BM25 layer still scores (features,
    agreement) but cannot set ``match_type="bm25"`` by itself; the claim
    falls through to fuzzy/semantic/agreement, which need real overlap."""

    semantic_threshold: float
    """Absolute cosine above which match_type=semantic (pre-percentile)."""

    nli_entailment_min: float
    """Minimum entailment probability for NLI to confirm in the cascade.

    The NLI verdict is decisive only when confident: argmax entailment below
    this floor (e.g. 0.34 on a near-uniform distribution) is a shrug, not
    evidence, and falls through to the lexical ladder instead of confirming
    past every lexical threshold."""

    nli_contradiction_min: float
    """Minimum contradiction probability for NLI to contradict.

    Same confidence principle as ``nli_entailment_min``, other arm: an argmax
    contradiction by a nose must not refuse a claim that strong lexical
    evidence confirms."""

    semantic_threshold_percentile: float
    """Top fraction of random chunk-pair distribution for semantic match."""

    agreement_threshold: float
    """Minimum agreement_score for the agreement-fallback classifier."""

    # ── percentile safety floor ──────────────────────────────────────────
    percentile_floor: float
    """Min value SemanticGrounder.percentile_threshold can return."""

    # ── agreement_score per-layer weights (raw sum, should total 1.0) ───
    agreement_weight_exact: float
    agreement_weight_fuzzy: float
    agreement_weight_bm25: float
    agreement_weight_semantic: float

    # ── agreement_score per-layer ramps (low..high -> v_layer in [0,1]) ─
    fuzzy_ramp_low: float
    fuzzy_ramp_high: float
    bm25_ramp_low: float
    bm25_ramp_high: float
    semantic_abs_ramp_low: float
    semantic_abs_ramp_high: float
    semantic_ratio_ramp_low: float
    semantic_ratio_ramp_high: float

    # ── voter thresholds ────────────────────────────────────────────────
    voter_exact: float
    voter_fuzzy: float
    voter_bm25: float
    voter_semantic_abs: float
    voter_semantic_ratio: float
    voter_semantic_mode: Literal["or", "and", "abs_only", "ratio_only"]
    """How the semantic voter combines absolute score and ratio."""

    voter_bonus_2: float
    voter_bonus_3_plus: float

    # ── entity-presence penalty ─────────────────────────────────────────
    entity_penalty_factor: float
    """Max fraction of agreement_score removed when 100% of claim entities absent from source."""

    # ── lexical co-support gate (WI#5) ──────────────────────────────────
    lexical_cosupport_fuzzy_min: float
    """Min fuzzy score to count as lexical co-support for a semantic hit."""

    lexical_cosupport_bm25_min: float
    """Min bm25_token_recall to count as lexical co-support for a semantic hit."""

    # ── verification threshold proximity (WI#6) ─────────────────────────
    verification_threshold_proximity: float
    """Score-within-this-of-threshold marks match as verification_needed."""

    # ── semantic / chunking ─────────────────────────────────────────────
    chunk_max_chars: int
    chunk_overlap_ratio: float
    percentile_sample_n: int

    # ── sentence-split fallback ─────────────────────────────────────────
    min_passage_chars: int
    single_passage_fallback_length: int

    # ── classifier mode (H11) ───────────────────────────────────────────
    classifier_mode: Literal["absolute", "adaptive_gap"]
    """``absolute`` = use agreement_threshold. ``adaptive_gap`` = batch-mode rank-based."""

    adaptive_gap_min_claims: int

    # ── solution tier ───────────────────────────────────────────────────
    lexical_effort: Literal["low", "medium", "high"]
    """Solution tier for lexical mode - the only user knob. Each tier is an
    indivisible bundle of algorithms plus the manifold trained for it: low (13
    feat), medium (16, + lingua + WordNet), high (18, + MT translate-then-recall).
    The chunk operating point (300/0.1) lives per-manifold, not here."""

    # ── misc ────────────────────────────────────────────────────────────
    context_chars: int
    semantic_top_k: int

    def __post_init__(self) -> None:
        # Literal annotations are not enforced at runtime by dataclasses; without
        # this check an invalid value (e.g. lexical_effort: ultra) would silently
        # fall through manifold lookup to the cascade engine instead of failing.
        hints = get_type_hints(type(self))
        for name, hint in hints.items():
            if get_origin(hint) is Literal:
                value = getattr(self, name)
                allowed = get_args(hint)
                if value not in allowed:
                    raise ConfigError(
                        f"invalid value {value!r} for {name}; allowed: {sorted(allowed)}"
                    )

    def overlay(self, **overrides) -> GroundingConfig:
        """Return a copy with the given fields overridden.

        Used by ``ground`` / ``ground_batch`` to let explicit keyword
        arguments win over the loaded config. ``None`` values are
        ignored ("no override for this field"), so callers can pass
        their full signature through without special-casing.
        """
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        for k, v in overrides.items():
            if v is not None:
                current[k] = v
        return GroundingConfig(**current)


# --- generic loader --------------------------------------------------------


T = TypeVar("T")


def _resolve_config_path(plugin_name: str, explicit_path: Path | str | None) -> Path:
    """Walk the 4-layer precedence order and return the first file that exists.

    Falls back to the bundled default path even if the file does not
    exist there - the caller's ``is_file()`` check will produce the
    loud ``ConfigError``.

    Same ``config_<plugin>.yaml`` filename in all three locations;
    only the directory changes.
    """
    if explicit_path is not None:
        return Path(explicit_path)
    filename = f"config_{plugin_name}.yaml"
    project = PROJECT_OVERRIDE_DIR / filename
    if project.is_file():
        return project
    user = _user_override_dir() / filename
    if user.is_file():
        return user
    return PACKAGE_ROOT / filename


def _load_typed_config(
    schema: type[T],
    plugin_name: str,
    path: Path | str | None = None,
) -> T:
    """Load ``<plugin_name>`` config from disk and instantiate ``schema``.

    See module docstring for resolution order. Raises :class:`ConfigError`
    on any problem (missing file, bad yaml, missing required fields).
    """
    effective_path = _resolve_config_path(plugin_name, path)

    if not effective_path.is_file():
        raise ConfigError(
            f"config yaml not found at {effective_path}; ship "
            f"config_{plugin_name}.yaml with the package or create "
            f".stellars-plugins/config_{plugin_name}.yaml to override"
        )

    try:
        raw = load_raw_yaml(effective_path)
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse {effective_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{effective_path} must contain a yaml mapping at top level")

    required = {f.name for f in fields(schema)}
    missing = required - set(raw.keys())
    if missing:
        raise ConfigError(
            f"{effective_path} is missing required fields: {sorted(missing)}. "
            "Every schema field must be declared in the yaml - there are no "
            "hardcoded defaults."
        )

    filtered = {k: raw[k] for k in required}
    return schema(**filtered)


# --- per-subsystem load helpers -------------------------------------------


def load_document_processing_config(
    path: Path | str | None = None,
) -> GroundingConfig:
    """Load the document-processing grounding config from yaml.

    Looks for ``config_document_processing.yaml`` in order:

    - ``./.stellars-plugins/config_document_processing.yaml`` (project)
    - ``~/.stellars-plugins/config_document_processing.yaml`` (user)
    - ``groundrails/config_document_processing.yaml`` (bundled)
    """
    return _load_typed_config(GroundingConfig, "document_processing", path=path)


# --- backward-compatible aliases ------------------------------------------
#
# ``load_config`` was the Iter 5 name when config lived inside
# ``document_processing/``. Keep it as an alias so existing callers (and
# the archived scripts under ``references/grounding-results/``)
# continue to work.

load_config = load_document_processing_config


__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_OVERRIDE_DIR",
    "ConfigError",
    "GroundingConfig",
    "load_config",
    "load_document_processing_config",
]
