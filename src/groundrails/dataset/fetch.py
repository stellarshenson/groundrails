"""Fetch stage - acquire a corpus from its manifest entry.

One corpus lands one of two ways. An ``archive`` corpus is STAGED as parquet -
one file per split - and only then zipped to ``dataset-<name>.zip``, so a
half-finished download never looks like a complete one. A ``tree`` corpus, whose
source does not fit an archive (a GitHub checkout, a per-topic evidence pull),
lands as an extracted directory ``<name>/``. Both carry a ``_counts.json``
checkpoint recording what the download actually produced.

Three mechanics matter more than the download itself:

* the manifest is the only source of truth - nothing about a corpus is decided here
* a failed corpus is a RESULT, not a crash: it is reported SKIPPED and the run
  continues, because a wave of fetches must not die on one dead mirror
* re-runs are idempotent - a corpus whose checkpoint is on disk is returned
  from cache untouched

Archives and trees are gitignored; what is tracked is the manifest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import zipfile

from groundrails.dataset import manifest as manifest_mod
from groundrails.dataset._deps import polars, require
from groundrails.dataset.manifest import CorpusEntry

#: Bespoke download logic, keyed by the manifest's ``source.fetcher``. A fetcher
#: takes ``(entry, target)`` - a staging dir for an archive corpus, the tree
#: root for a tree corpus - and returns a counts mapping, or ``None`` if the
#: source could not be read.
FETCHERS: dict[str, Callable[[CorpusEntry, Path], dict | None]] = {}


def register_fetcher(name: str):
    """Register a bespoke fetcher under ``name`` (the manifest's ``source.fetcher``)."""

    def deco(fn):
        FETCHERS[name] = fn
        return fn

    return deco


@dataclass
class FetchResult:
    """What one corpus's fetch produced."""

    name: str
    status: str  # fetched | cached | skipped | dry-run
    storage: str  # archive | tree
    counts: dict = field(default_factory=dict)
    path: Path | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("fetched", "cached")


# --------------------------------------------------------------------------- #
# declarative sources
# --------------------------------------------------------------------------- #
def _hf_file(repo: str, filename: str) -> str:
    hub = require("huggingface_hub", "pulling files from the HuggingFace Hub")
    return hub.hf_hub_download(repo, filename, repo_type="dataset")


def _resolve(repo: str, pattern: str) -> str:
    """A literal filename, or the first repo file matching a ``prefix*suffix`` glob.

    Globbing is resolved live so a re-upload under a new content hash still
    fetches instead of 404-ing on a filename frozen into the manifest.
    """
    if "*" not in pattern:
        return pattern
    hub = require("huggingface_hub", "listing files in a HuggingFace dataset repo")
    head, tail = pattern.split("*", 1)
    for f in hub.HfApi().list_repo_files(repo, repo_type="dataset"):
        if f.startswith(head) and f.endswith(tail):
            return f
    raise FileNotFoundError(f"no file matching {pattern!r} in {repo}")


def _read_any(path: str):
    """Read a downloaded parquet / json / jsonl file into a frame."""
    pl = polars()
    p = Path(path)
    if p.suffix == ".parquet":
        return pl.read_parquet(p)
    if p.suffix in (".jsonl", ".ndjson"):
        return pl.DataFrame(
            [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        )
    if p.suffix == ".json":
        return pl.DataFrame(json.loads(p.read_text(encoding="utf-8")))
    raise ValueError(f"unsupported source file type: {p.name}")


def _apply_drop_where(df, drop_where: dict[str, tuple[str, ...]]) -> tuple[object, dict]:
    """The manifest's fetch-time carve-out: rows in the named values never enter the archive."""
    pl = polars()
    dropped = {}
    for col, values in drop_where.items():
        if col not in df.columns:
            continue
        before = df.height
        df = df.filter(~pl.col(col).is_in(list(values)))
        dropped[col] = before - df.height
    return df, dropped


def _fetch_hf_dataset(entry: CorpusEntry, staging: Path) -> dict | None:
    """The default Hub path: ``load_dataset`` per id/subset, one parquet per split."""
    datasets = require("datasets", "the default HuggingFace dataset loader")
    counts: dict = {}
    for repo in entry.source.repos:
        for sub in entry.source.subsets or (None,):
            tag = f"{repo.replace('/', '__')}{'__' + sub if sub else ''}"
            try:
                ds = datasets.load_dataset(repo, sub) if sub else datasets.load_dataset(repo)
            except Exception as exc:  # noqa: BLE001 - a failed split is a result, not a crash
                print(
                    f"    SKIP {repo}{'/' + sub if sub else ''}: "
                    f"{type(exc).__name__}: {str(exc)[:110]}",
                    flush=True,
                )
                continue
            for split, d in ds.items():
                out = staging / f"{tag}__{split}.parquet"
                d.to_parquet(out)
                counts[split] = counts.get(split, 0) + len(d)
                print(f"    {tag}/{split}: {len(d)} rows -> {out.name}", flush=True)
    return counts or None


def _fetch_hf_files(entry: CorpusEntry, staging: Path) -> dict | None:
    """Named files inside one Hub repo, each landing as a split parquet."""
    src = entry.source
    counts: dict = {}
    dropped: dict = {}
    for split, pattern in src.files.items():
        df = _read_any(_hf_file(src.repo, _resolve(src.repo, pattern)))
        if src.drop_where:
            df, d = _apply_drop_where(df, src.drop_where)
            for col, n in d.items():
                dropped[f"{split}:{col}"] = n
        out = staging / f"{entry.name}__{split}.parquet"
        df.write_parquet(out)
        counts[split] = df.height
        print(f"    {entry.name}/{split}: {df.height} rows -> {out.name}", flush=True)
    if dropped:
        counts["_dropped_at_fetch"] = dropped
        counts["_dropped_at_fetch_total"] = sum(dropped.values())
    return counts or None


def _fetch_http_zip(entry: CorpusEntry, tree: Path) -> dict | None:
    """A zip over HTTP, extracted into the tree; ``keep_prefixes`` selects members."""
    import io
    import urllib.request

    req = urllib.request.Request(entry.source.url, headers={"User-Agent": "groundrails/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    root = z.namelist()[0].split("/")[0]
    keep = entry.source.keep_prefixes
    n_files = 0
    for member in z.namelist():
        rel = member[len(root) + 1 :] if member.startswith(root + "/") else member
        if not rel or member.endswith("/"):
            continue
        if keep and not any(rel == p or rel.startswith(p) for p in keep):
            continue
        target = tree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(member))
        n_files += 1
    print(f"    {entry.name}: {n_files} files extracted", flush=True)
    return {"_files": n_files} if n_files else None


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
def checkpoint_path(entry: CorpusEntry, out_dir: Path) -> Path:
    """Where a completed fetch records what it produced."""
    if entry.storage == "tree":
        return out_dir / entry.name / "_counts.json"
    return out_dir / f"dataset-{entry.name}.zip"


def recorded_counts(entry: CorpusEntry, out_dir: Path) -> dict | None:
    """Counts from an earlier completed fetch, or ``None``."""
    try:
        if entry.storage == "tree":
            return json.loads(checkpoint_path(entry, out_dir).read_text())["counts"]
        with zipfile.ZipFile(checkpoint_path(entry, out_dir)) as z:
            return json.loads(z.read("_counts.json"))["counts"]
    except Exception:  # noqa: BLE001 - an unreadable checkpoint is simply absent
        return None


def _run_source(entry: CorpusEntry, target: Path) -> dict | None:
    kind = entry.source.kind
    if kind == "adapter":
        fetcher = FETCHERS.get(entry.source.fetcher)
        if fetcher is None:
            raise KeyError(
                f"corpus {entry.name!r} names fetcher {entry.source.fetcher!r}, which is not "
                f"registered; known: {', '.join(sorted(FETCHERS)) or '(none)'}"
            )
        return fetcher(entry, target)
    if kind == "hf_dataset":
        return _fetch_hf_dataset(entry, target)
    if kind == "hf_files":
        return _fetch_hf_files(entry, target)
    return _fetch_http_zip(entry, target)


def fetch(
    name: str | CorpusEntry,
    out_dir: Path | str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> FetchResult:
    """Acquire one corpus into ``out_dir``.

    Returns a :class:`FetchResult` in every case - a dead source is reported
    SKIPPED, not raised, so a wave of fetches survives one broken mirror. An
    existing checkpoint short-circuits to CACHED unless ``force`` is set.
    """
    entry = manifest_mod.get(name) if isinstance(name, str) else name
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = checkpoint_path(entry, out)

    if dry_run:
        return FetchResult(entry.name, "dry-run", entry.storage, detail=str(entry.source.kind))
    if done.exists() and not force:
        counts = recorded_counts(entry, out) or {}
        target = out / entry.name if entry.storage == "tree" else done
        return FetchResult(entry.name, "cached", entry.storage, counts, target)

    staging = out / f"_staging_{entry.name}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    tree = out / entry.name
    if entry.storage == "tree":
        shutil.rmtree(tree, ignore_errors=True)
        tree.mkdir(parents=True)

    target = tree if entry.storage == "tree" else staging
    try:
        counts = _run_source(entry, target)
    except Exception as exc:  # noqa: BLE001 - a failed corpus is a result, not a crash
        counts = None
        detail = f"{type(exc).__name__}: {str(exc)[:200]}"
    else:
        detail = ""

    if not counts:
        shutil.rmtree(staging, ignore_errors=True)
        if entry.storage == "tree":
            shutil.rmtree(tree, ignore_errors=True)
        return FetchResult(entry.name, "skipped", entry.storage, detail=detail or "no rows")

    record = {
        "counts": counts,
        "fetched_utc": datetime.now(UTC).date().isoformat(),
        "source": entry.source.url or ", ".join(entry.source.repos) or entry.source.repo,
        "licence": entry.licence.tag,
    }
    if entry.storage == "tree":
        (tree / "_counts.json").write_text(json.dumps(record, indent=2))
        shutil.rmtree(staging, ignore_errors=True)
        return FetchResult(entry.name, "fetched", "tree", counts, tree)

    (staging / "_counts.json").write_text(json.dumps(record, indent=2))
    archive = out / f"dataset-{entry.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(staging.iterdir()):
            z.write(f, f.name)
    shutil.rmtree(staging)
    return FetchResult(entry.name, "fetched", "archive", counts, archive)


def read_fetched(entry: CorpusEntry, out_dir: Path | str) -> dict:
    """``split -> frame`` for an archive corpus's staged parquets."""
    import io

    pl = polars()
    archive = Path(out_dir) / f"dataset-{entry.name}.zip"
    if not archive.exists():
        raise FileNotFoundError(f"{entry.name} is not fetched: {archive} is missing")
    z = zipfile.ZipFile(archive)
    parts = {}
    for m in z.namelist():
        if m.endswith(".parquet"):
            parts[m[: -len(".parquet")].split("__")[-1]] = pl.read_parquet(io.BytesIO(z.read(m)))
    return parts
