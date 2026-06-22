"""One-call bootstrap: provision calibration + models from S3 / local / HuggingFace.

``groundrails.init`` pulls every required resource through one documented 3-way
resolution so the grounder can run offline (e.g. an AWS Lambda with no
HuggingFace egress):

    1. an explicit ``--``/kwarg override (``s3://…`` | ``https://…`` | ``/local/…``)
    2. else the configured ``source`` base, tried as S3 first, then a local folder
    3. else the HuggingFace Hub (the fallback for model weights)

Resources resolved this way: the calibration JSON (S3 / URL / local; bundled YAML
is the built-in fallback), the int8 cascade IRs (S3 / local / HF), the SaT IR,
the argos MT models for given languages, and the NLTK WordNet corpus.

S3 uses ``botocore`` directly (already a core dep - no boto3), with an optional
named profile and a custom endpoint URL (so S3-compatible stores like RustFS
work). In Lambda omit the profile to use the execution-role credential chain.

No settings file is written: provisioning sets the runtime config + the env vars
the lazy model loaders read (``GROUNDRAILS_CALIBRATION_JSON``,
``GROUNDRAILS_MODELS_DIR``, ``SAT_OV_IR``).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from loguru import logger

from groundrails import settings
from groundrails.calibration import export_calibration  # re-export

__all__ = ["init", "export_calibration"]


# --- source-scheme helpers -------------------------------------------------


def _scheme(uri: str) -> str:
    if uri.startswith("s3://"):
        return "s3"
    if uri.startswith(("http://", "https://")):
        return "url"
    return "local"


def _split_s3(uri: str) -> tuple[str, str]:
    """``s3://bucket/key/parts`` -> ``("bucket", "key/parts")``."""
    bucket, _, key = uri[len("s3://") :].partition("/")
    return bucket, key


def _s3_client(profile: str | None, endpoint_url: str | None, region: str | None):
    """A low-level botocore S3 client (path-style, for S3-compatible endpoints)."""
    from botocore.config import Config
    import botocore.session

    sess = botocore.session.Session(profile=profile) if profile else botocore.session.Session()
    return sess.create_client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        config=Config(s3={"addressing_style": "path"}),
    )


# --- single-object fetch ---------------------------------------------------


def _fetch_to(uri: str, dest: str | Path, *, client=None) -> bool:
    """Fetch one object/file/URL to ``dest``. Returns True on success, False if absent.

    ``client`` is a botocore S3 client (required for ``s3://`` URIs).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    scheme = _scheme(uri)
    if scheme == "s3":
        bucket, key = _split_s3(uri)
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as exc:  # noqa: BLE001 - missing key / auth / endpoint
            logger.debug("S3 fetch miss {}: {}", uri, exc)
            return False
        dest.write_bytes(body)
        return True
    if scheme == "url":
        import urllib.request

        try:
            with urllib.request.urlopen(uri) as resp:  # noqa: S310 - explicit user URL
                data = resp.read()
        except Exception as exc:  # noqa: BLE001
            logger.debug("URL fetch miss {}: {}", uri, exc)
            return False
        dest.write_bytes(data)
        return True
    src = Path(uri)
    if not src.is_file():
        return False
    shutil.copyfile(src, dest)
    return True


# --- calibration provisioning ----------------------------------------------


def _provision_calibration(
    calibration: str | None,
    source: str | None,
    home: Path,
    client,
    summary: dict,
) -> Path | None:
    """Install the provisioned calibration JSON; fall back to the bundled block."""
    target = home / "calibration.json"
    cand: str | None = None
    used: str | None = None
    if calibration:  # explicit override - any of s3 / url / local
        cand, used = calibration, _scheme(calibration)
    elif source:  # default chain off the source base
        sch = _scheme(source)
        if sch == "s3":
            cand, used = source.rstrip("/") + "/calibration.json", "s3"
        elif sch == "local":
            lp = Path(source) / "calibration.json"
            if lp.is_file():
                cand, used = str(lp), "local"

    if not cand:
        summary["calibration"] = {"source_used": "bundled", "path": None}
        return None

    if not _fetch_to(cand, target, client=client):
        if calibration:  # an explicit override that is missing is a hard error
            raise FileNotFoundError(f"calibration not found at {cand}")
        summary["calibration"] = {"source_used": "bundled", "path": None}
        return None

    settings.configure(calibration_path=str(target))
    summary["calibration"] = {"source_used": used, "path": str(target)}
    logger.info("calibration provisioned from {} ({})", cand, used)
    return target


# --- model mirroring -------------------------------------------------------

# repos whose full snapshot dir is mirrored so the cascade tokenizer + IR load
# offline; keys match semantic_ov._resolve_repo_dir lookups (<models_dir>/<name>).
_CASCADE_FILES = ("openvino_model.xml", "openvino_model.bin", "config.json", "tokenizer.json")


def _s3_prefix_to_dir(client, s3_base: str, name: str, dest: Path) -> bool:
    """Mirror every object under ``<s3_base>/<name>/`` into ``dest``. True if any."""
    bucket, key = _split_s3(s3_base.rstrip("/") + "/" + name + "/")
    try:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("S3 list miss {}: {}", key, exc)
        return False
    objs = resp.get("Contents") or []
    if not objs:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    for o in objs:
        rel = o["Key"][len(key) :].lstrip("/")
        if not rel:
            continue
        _fetch_to(f"s3://{bucket}/{o['Key']}", dest / rel, client=client)
    return True


def _mirror_one(name: str, s3_base, local_base, force_hf, models_dir: Path, client) -> str:
    """Resolve one model repo to a local dir; returns the source used (s3/local/hf)."""
    dest = models_dir / name
    if not force_hf and s3_base and _s3_prefix_to_dir(client, s3_base, name, dest):
        return "s3"
    if not force_hf and local_base:
        src = Path(local_base) / name
        if src.is_dir():
            if dest.resolve() != src.resolve():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            return "local"
    # HuggingFace fallback - warm the HF cache (semantic_ov reads it when the
    # mirror lacks this repo). Best-effort; skip quietly if the extra is absent.
    try:
        from huggingface_hub import snapshot_download

        from groundrails import semantic_ov

        snapshot_download(semantic_ov.HF_REPOS[name])
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF prefetch of {} skipped: {}", name, exc)
    return "hf"


def _mirror_models(models, source, home: Path, client, summary: dict) -> None:
    if models == "none":
        summary["models"] = {"source_used": "skipped", "names": []}
        return
    from groundrails import semantic_ov

    models_dir = home / "models"
    s3_base = local_base = None
    force_hf = models == "hf"
    if models and models not in ("hf", "none"):
        sch = _scheme(models)
        if sch == "s3":
            s3_base = models
        elif sch == "local":
            local_base = models
    elif source and not force_hf:
        sch = _scheme(source)
        if sch == "s3":
            s3_base = source.rstrip("/") + "/models"
        elif sch == "local":
            local_base = str(Path(source) / "models")

    used: dict[str, str] = {}
    for name in semantic_ov.HF_REPOS:
        used[name] = _mirror_one(name, s3_base, local_base, force_hf, models_dir, client)
    # SaT int8 IR (xml+bin) - point SAT_OV_IR at the mirror when present
    sat_used = _mirror_one("sat", s3_base, local_base, force_hf, models_dir, client)
    sat_xml = models_dir / "sat" / "openvino_model.xml"
    if sat_used in ("s3", "local") and sat_xml.is_file():
        os.environ["SAT_OV_IR"] = str(sat_xml)
        used["sat"] = sat_used

    if models_dir.exists() and any(models_dir.iterdir()):
        settings.configure(models_dir=str(models_dir))
    summary["models"] = {"source_used": used, "dir": str(models_dir)}


# --- argos MT + WordNet ----------------------------------------------------


def _prefetch_argos(languages, summary: dict) -> None:
    if not languages:
        summary["languages"] = []
        return
    from groundrails import lexical_mt

    done = []
    for lang in languages:
        try:
            if lexical_mt.install_model(lang):
                done.append(lang)
        except Exception as exc:  # noqa: BLE001
            logger.warning("argos {}->en prefetch failed: {}", lang, exc)
    summary["languages"] = done


def _ensure_wordnet(enabled: bool, summary: dict) -> None:
    if not enabled:
        summary["wordnet"] = False
        return
    try:
        import nltk

        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        summary["wordnet"] = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("WordNet ensure skipped: {}", exc)
        summary["wordnet"] = False


# --- public entrypoint -----------------------------------------------------


def init(
    source: str | None = None,
    *,
    calibration: str | None = None,
    models: str | None = None,
    languages: list[str] | None = None,
    wordnet: bool = True,
    semantic_model: str | None = None,
    cache_dir: str | None = None,
    aws_profile: str | None = None,
    aws_endpoint_url: str | None = None,
    aws_region: str | None = None,
    home: str | None = None,
) -> dict:
    """Provision groundrails in one call - calibration + models from S3 / local / HF.

    ``source`` is the default base for the resolution chain (an ``s3://`` prefix
    or a local dir); per-resource overrides (``calibration``, ``models``) win
    over it. ``models`` may be an ``s3://``/local prefix, ``"hf"`` to force the
    Hub, or ``"none"`` to skip model weights. Returns a summary naming which of
    the three ways served each resource.
    """
    home_path = Path(home) if home else settings.default_home()
    home_path.mkdir(parents=True, exist_ok=True)
    # Pin GROUNDRAILS_HOME only when the caller chose a home; otherwise leave it
    # unset so groundrails.json falls back to ./ (the current directory).
    if home:
        os.environ[settings.ENV_HOME] = str(home_path)
    settings.configure(semantic_model=semantic_model, cache_dir=cache_dir)

    client = None
    needs_s3 = (
        _scheme(source or "") == "s3"
        or _scheme(calibration or "") == "s3"
        or (models is not None and _scheme(models) == "s3")
    )
    if needs_s3:
        client = _s3_client(aws_profile, aws_endpoint_url, aws_region)

    summary: dict = {"home": str(home_path)}
    _provision_calibration(calibration, source, home_path, client, summary)
    _mirror_models(models, source, home_path, client, summary)
    _prefetch_argos(languages, summary)
    _ensure_wordnet(wordnet, summary)

    # The grounder is now ready (calibration + any models provisioned). Mark it so the
    # readiness gate passes in-process, and persist the resolved config to groundrails.json
    # so a later CLI process (`groundrails ground`) is ready without re-provisioning.
    settings.mark_ready()
    cfg_file = settings.save_config_file()
    if cfg_file:
        summary["config_file"] = str(cfg_file)

    logger.info("groundrails init complete: {}", summary)
    return summary
