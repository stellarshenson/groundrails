"""Built-in runtime settings for groundrails - no settings file.

Settings are code defaults configured at runtime via the CLI (`groundrails`
flags), :func:`groundrails.init`, or environment variables. Nothing is read
from or written to ``.stellars-plugins/settings.json`` - a persisted settings
file is the wrong fit for a stateless / Lambda deployment, so configuration
lives in-process for the life of the call.

Resource provisioning (the calibration JSON + model weights, from S3 / a local
folder / the HuggingFace Hub) is handled by :mod:`groundrails.bootstrap`
(``groundrails.init``); this module only holds the resolved knobs the grounder
reads and the env vars the engines look at.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
import os
from pathlib import Path

DEFAULT_SEMANTIC_MODEL = "intfloat/multilingual-e5-small"

# Env vars the engines read; ``configure`` / ``init`` set them so a provisioned
# calibration JSON and a local model mirror survive into the lazy model loaders.
ENV_HOME = "GROUNDRAILS_HOME"
ENV_CALIBRATION = "GROUNDRAILS_CALIBRATION_JSON"
ENV_MODELS_DIR = "GROUNDRAILS_MODELS_DIR"
ENV_CONFIG = "GROUNDRAILS_CONFIG"

CONFIG_FILENAME = "groundrails.json"


def default_home() -> Path:
    """``GROUNDRAILS_HOME`` - where init writes the calibration JSON + mirrored models.

    Defaults to ``~/.cache/groundrails`` (use ``/tmp/groundrails`` in Lambda by
    exporting ``GROUNDRAILS_HOME``).
    """
    env = os.environ.get(ENV_HOME)
    return Path(env) if env else Path.home() / ".cache" / "groundrails"


@dataclass
class RuntimeConfig:
    """In-process settings - built-in defaults, overridden by CLI / ``init``."""

    semantic_model: str = DEFAULT_SEMANTIC_MODEL
    semantic_device: str = "auto"
    cache_dir: str = ""  # parquet cache for chunks/embeddings; resolved under home when empty
    calibration_path: str = ""  # provisioned calibration JSON; empty -> bundled default
    models_dir: str = ""  # local mirror dir for model IRs; empty -> HuggingFace cache

    def resolved_cache_dir(self) -> str:
        return self.cache_dir or str(default_home() / "cache")


# Back-compat alias: older callers/tests refer to ``Settings``.
Settings = RuntimeConfig

_RUNTIME = RuntimeConfig()

# Readiness gate: the grounder refuses to run until ``init`` (or a loaded
# ``groundrails.json``) has marked it ready. A stateless Lambda calls
# ``groundrails.init`` in-process; the CLI provisions once and persists
# ``groundrails.json``, which a later ``groundrails ground`` process loads.
_READY = False


def get() -> RuntimeConfig:
    """The active runtime config (built-in defaults unless configured)."""
    return _RUNTIME


# Back-compat: ``load`` returned a Settings object; now it returns the runtime
# config (no file is read). Callers that only need the model/cache knobs keep
# working unchanged.
load = get


def configure(**overrides) -> RuntimeConfig:
    """Override runtime settings in place (``None`` values are ignored).

    Also exports the env vars the lazy model loaders read, so a provisioned
    calibration JSON (``calibration_path``) and a local model mirror
    (``models_dir``) take effect for the rest of the process.
    """
    global _RUNTIME
    valid = {f.name for f in fields(RuntimeConfig)}
    changes = {k: v for k, v in overrides.items() if k in valid and v is not None}
    _RUNTIME = replace(_RUNTIME, **changes)
    if _RUNTIME.calibration_path:
        os.environ[ENV_CALIBRATION] = _RUNTIME.calibration_path
    if _RUNTIME.models_dir:
        os.environ[ENV_MODELS_DIR] = _RUNTIME.models_dir
    return _RUNTIME


def reset() -> None:
    """Reset to built-in defaults and clear readiness (test helper)."""
    global _RUNTIME, _READY
    _RUNTIME = RuntimeConfig()
    _READY = False


# --- readiness gate --------------------------------------------------------


class NotInitializedError(RuntimeError):
    """Raised when the grounder runs before ``groundrails.init`` has provisioned it."""

    def __init__(self) -> None:
        super().__init__(
            "groundrails is not initialized - call groundrails.init() (Python) or run "
            "`groundrails init` (CLI, writes groundrails.json) before grounding"
        )


def mark_ready() -> None:
    """Mark the grounder ready - set by ``init`` and by loading ``groundrails.json``."""
    global _READY
    _READY = True


def is_ready() -> bool:
    """True once ``init`` has run (or a ``groundrails.json`` has been loaded)."""
    return _READY


def require_ready() -> None:
    """Raise :class:`NotInitializedError` unless the grounder has been initialized."""
    if not _READY:
        raise NotInitializedError()


# --- groundrails.json (the init-written runtime config) ---------------------


def config_file_path(explicit: str | None = None) -> Path:
    """Where ``groundrails.json`` is written / read: an explicit path, then
    ``GROUNDRAILS_CONFIG``, then ``$GROUNDRAILS_HOME/groundrails.json``, else
    ``./groundrails.json`` in the current directory (when ``GROUNDRAILS_HOME``
    is unset)."""
    if explicit:
        return Path(explicit)
    env = os.environ.get(ENV_CONFIG)
    if env:
        return Path(env)
    home_env = os.environ.get(ENV_HOME)
    if home_env:
        return Path(home_env) / CONFIG_FILENAME
    return Path.cwd() / CONFIG_FILENAME


def save_config_file(explicit: str | None = None) -> Path | None:
    """Persist the active runtime config to ``groundrails.json`` (best-effort).

    Returns the path written, or ``None`` when the location is not writable (a
    read-only task root in Lambda, where readiness is in-process anyway).
    """
    p = config_file_path(explicit)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(_RUNTIME), indent=2), encoding="utf-8")
        return p
    except OSError:
        return None


def load_config_file(explicit: str | None = None) -> bool:
    """Load ``groundrails.json`` into the runtime config and mark ready.

    Returns ``True`` when a file was found and loaded, ``False`` when none exists.
    """
    p = config_file_path(explicit)
    if not p.is_file():
        return False
    data = json.loads(p.read_text(encoding="utf-8"))
    valid = {f.name for f in fields(RuntimeConfig)}
    configure(**{k: v for k, v in data.items() if k in valid})
    mark_ready()
    return True


# --- semantic optional-deps helpers (kept; used by the CLI) ----------------


def is_semantic_available() -> bool:
    """Check if the (legacy ONNX) semantic-grounding optional deps are importable."""
    for mod in ("onnxruntime", "transformers", "faiss", "pyarrow", "huggingface_hub"):
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


def semantic_install_hint() -> str:
    return (
        "Semantic grounding requires optional dependencies. Install with:\n"
        "  pip install 'groundrails[semantic-grounder]'\n"
        "or individually:\n"
        "  pip install onnxruntime transformers faiss-cpu pyarrow huggingface_hub\n"
    )
