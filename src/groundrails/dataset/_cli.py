"""`groundrails dataset` - the corpus preprocessing pipeline on the command line.

One subcommand per stage plus `run` for a corpus end to end. Every stage is a
gate as well as a report: a BLOCK verdict, a RED contamination gate or a no-go
census exits non-zero, so a wrapper script can stop before spending a GPU.

Nothing here imports the pipeline at parser-build time - the heavy imports
happen inside the handlers, so `groundrails --help` costs nothing on an install
without the ``dataset`` extra.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

DEFAULT_DATA_DIR = "data/external/datasets"
DEFAULT_LANE_DIR = "data/interim/lanes"


def _presentation(args):
    """The presentation the stage measures: a corpus's declared block, then flag overrides."""
    from groundrails.dataset import manifest as manifest_mod

    base = manifest_mod.get(args.corpus).presentation if getattr(args, "corpus", None) else None
    fields = {
        "window_chars": args.window,
        "stride_chars": args.stride,
        "pairs_per_batch": args.cap,
        "max_rows_per_document": args.max_rows_per_doc,
        "max_pair_share": args.max_pair_share,
        "max_over_cap_fraction": args.max_over_cap_fraction,
    }
    given = {k: v for k, v in fields.items() if v is not None}
    if base is None:
        return manifest_mod.Presentation(**given)
    return base.model_copy(update=given)


def _emit(payload, output: str | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)
    else:
        print(text)


def _names(args) -> list[str]:
    from groundrails.dataset import manifest as manifest_mod

    return list(args.names) or manifest_mod.names()


# --- stage handlers --------------------------------------------------------- #
def cmd_fetch(args: argparse.Namespace) -> int:
    """Stage 1 - acquire corpora into the data directory."""
    from groundrails.dataset.fetch import fetch

    failed = 0
    for name in _names(args):
        res = fetch(name, args.data_dir, dry_run=args.dry_run, force=args.force)
        print(f"{name:<20} {res.status:<8} {res.detail or res.path or ''}", file=sys.stderr)
        failed += res.status == "skipped"
    if failed:
        print(f"{failed} corpora could not be fetched (see above)", file=sys.stderr)
    return 1 if failed else 0


def cmd_contaminate(args: argparse.Namespace) -> int:
    """Stage 2 - n-gram overlap against the walled corpora, with the spike control."""
    from groundrails.dataset.contaminate import check, walled_texts_from_files
    from groundrails.dataset.format import evidence_texts

    walled = walled_texts_from_files(args.walled, text_col=args.walled_col)
    texts = evidence_texts(args.corpus, args.data_dir)
    print(
        f"{args.corpus}: {len(texts)} deduplicated evidence units against "
        f"{sum(len(v) for v in walled.values())} walled units in {len(walled)} buckets",
        file=sys.stderr,
    )
    res = check(
        texts,
        walled,
        n=args.n,
        jaccard=None if args.containment else args.jaccard,
        warn=args.warn,
        kill=args.kill,
        label=args.corpus,
    )
    _emit(res, args.output)
    print(
        f"{args.corpus}: {res['status']} - gate {res['gate']['verdict']} at max fraction "
        f"{res['gate']['max_fraction']}, spike {res['spike_control']['detected_total']}/"
        f"{res['spike_control']['injected']} detected",
        file=sys.stderr,
    )
    return 0 if res["status"] == "GREEN" else 1


def cmd_format(args: argparse.Namespace) -> int:
    """Stage 3 - normalise fetched corpora into the pair schema."""
    from groundrails.dataset import manifest as manifest_mod
    from groundrails.dataset.format import FormatError, format_corpus, write_lane

    failed = 0
    for name in _names(args):
        entry = manifest_mod.get(name)
        if entry.format is None:
            print(f"{name:<20} SKIP     no format block in the manifest", file=sys.stderr)
            continue
        try:
            result = format_corpus(entry, args.data_dir, strict=not args.no_strict)
        except (FormatError, FileNotFoundError) as exc:
            print(f"{name:<20} FAIL     {exc}", file=sys.stderr)
            failed += 1
            continue
        path = write_lane(result, entry, args.lane_dir)
        print(
            f"{name:<20} {result.rows:>8,} rows  {result.frame['doc_id'].n_unique():>7,} docs "
            f" -> {path}",
            file=sys.stderr,
        )
    return 1 if failed else 0


def cmd_shape(args: argparse.Namespace) -> int:
    """Stage 4 - the evidence-shape gate over one or more lane parquets."""
    from groundrails.dataset.shape import BLOCK, MixProjection, shape_lane, table

    presentation = _presentation(args)
    projection = (
        MixProjection(args.mix_name, args.mix_other_pairs, args.mix_other_rows)
        if args.mix_other_pairs is not None
        else None
    )
    reports = [
        shape_lane(p, presentation, name=Path(p).stem, projection=projection) for p in args.lanes
    ]
    if args.json or args.output:
        _emit([r.to_dict() for r in reports], args.output)
    else:
        print(table(reports))
        for r in reports:
            for reason in r.reasons:
                print(f"  {r.name}: {r.verdict} - {reason}")
    return 1 if any(r.verdict == BLOCK for r in reports) else 0


def cmd_assemble(args: argparse.Namespace) -> int:
    """Stage 5 - build a mix from a JSON spec, asserting everything it declares."""
    from groundrails.dataset.assemble import LaneSpec, MixAssertionError, MixSpec, assemble
    from groundrails.dataset.manifest import Presentation

    raw = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    spec = MixSpec(
        name=raw.get("name", Path(args.spec).stem),
        lanes=tuple(LaneSpec(**lane) for lane in raw["lanes"]),
        base=raw.get("base"),
        base_group=raw.get("base_group", "base"),
        expected_rows=raw.get("expected_rows"),
        expected_groups=tuple(raw.get("expected_groups", ())),
        presentation=Presentation(**raw.get("presentation", {})),
        drop_over_cap=raw.get("drop_over_cap", True),
    )
    try:
        mix = assemble(spec)
    except MixAssertionError as exc:
        print(f"MIX ABORT: {exc}", file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        mix.frame.write_parquet(args.out)
        print(f"wrote {mix.rows:,} rows -> {args.out}", file=sys.stderr)
    _emit(mix.to_dict(), args.output)
    return 0


def cmd_census(args: argparse.Namespace) -> int:
    """Stage 6 - the pre-spend report over an assembled mix."""
    from groundrails.dataset._deps import polars
    from groundrails.dataset.census import census

    pl = polars()
    report = census(
        pl.read_parquet(args.mix),
        _presentation(args),
        name=args.name or Path(args.mix).stem,
        group_col=args.group_col,
    )
    if args.json or args.output:
        _emit(report.to_dict(), args.output)
    else:
        from groundrails.dataset.shape import table

        print(table(report.groups))
        print(
            f"\n{report.mix}: {report.rows:,} rows  {report.pairs:,} pairs  "
            f"mean target {report.mean_target}  mean windows {report.mean_windows}  "
            f"max {report.max_windows}"
        )
        print(f"GO: {report.go}" + (f" - blocking: {report.blocking}" if report.blocking else ""))
    return 0 if report.go else 1


def cmd_run(args: argparse.Namespace) -> int:
    """Stages 1-4 for each corpus: fetch, gate, format, shape."""
    from groundrails.dataset.contaminate import walled_texts_from_files
    from groundrails.dataset.pipeline import Pipeline
    from groundrails.dataset.shape import BLOCK

    walled = (
        walled_texts_from_files(args.walled, text_col=args.walled_col) if args.walled else None
    )
    pipe = Pipeline(
        data_dir=Path(args.data_dir),
        lane_dir=Path(args.lane_dir),
        walled=walled,
        strict=not args.no_strict,
    )
    out = []
    for name in _names(args):
        out.append(pipe.run(name, dry_run=args.dry_run))
        print(
            f"{name:<20} {out[-1].get('verdict', out[-1]['fetch']['status'])}",
            file=sys.stderr,
        )
    _emit(out, args.output)
    return 1 if any(r.get("verdict") == BLOCK for r in out) else 0


# --- parser ----------------------------------------------------------------- #
def _add_presentation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--corpus", help="Take the presentation and bars from this corpus's manifest entry"
    )
    p.add_argument("--window", type=int, default=None, help="Evidence window size in chars (1500)")
    p.add_argument("--stride", type=int, default=None, help="Window stride in chars (750)")
    p.add_argument(
        "--cap", type=int, default=None, help="Pairs per training batch - the over-cap bar (96)"
    )
    p.add_argument(
        "--max-rows-per-doc",
        type=float,
        default=None,
        help="BLOCK above this many rows per distinct document (20)",
    )
    p.add_argument(
        "--max-pair-share",
        type=float,
        default=None,
        help="BLOCK above this projected share of the mix's training pairs (0.15)",
    )
    p.add_argument(
        "--max-over-cap-fraction",
        type=float,
        default=None,
        help="BLOCK above this fraction of rows over the batch cap (0.02); below it the rows "
        "are a documented drop",
    )


def add_parser(sub) -> argparse.ArgumentParser:
    """Attach the `dataset` subcommand group to a subparsers object."""
    ds = sub.add_parser(
        "dataset",
        help="Corpus preprocessing pipeline: fetch, contaminate, format, shape, assemble, census.",
        description=(
            "End-to-end preprocessing for grounding training corpora. Corpora are declared in "
            "the packaged manifest (corpora.yaml); each stage consumes the previous one's "
            "output and is independently runnable and idempotent. `shape` is the gate that "
            "runs before any GPU spend: it reports windows per row, document reuse and the "
            "projected share of a mix's training pairs, and BLOCKS a corpus whose evidence "
            "shape would capture the run. Needs the [dataset] extra."
        ),
    )
    stage = ds.add_subparsers(dest="dataset_cmd", required=True)

    f = stage.add_parser("fetch", help="Stage 1 - acquire corpora into the data directory.")
    f.add_argument("names", nargs="*", metavar="NAME", help="Corpora to fetch (default: all)")
    f.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help=f"Default: {DEFAULT_DATA_DIR}")
    f.add_argument("--dry-run", action="store_true", help="Report what would be fetched")
    f.add_argument("--force", action="store_true", help="Re-fetch even if a checkpoint exists")
    f.set_defaults(func=cmd_fetch)

    c = stage.add_parser(
        "contaminate",
        help="Stage 2 - n-gram overlap against walled corpora, with the spike control.",
        description=(
            "Bidirectional n-gram gate over the corpus's deduplicated evidence texts. WARN at "
            "0.5%%, KILL at 2%% of the candidate. The spike control injects known walled units "
            "and requires every one back, so a clean verdict cannot come from a dead gate. "
            "Exit 0 on GREEN, 1 on RED."
        ),
    )
    c.add_argument("corpus", help="Corpus to gate")
    c.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help=f"Default: {DEFAULT_DATA_DIR}")
    c.add_argument(
        "--walled",
        action="append",
        required=True,
        help="Walled corpus file (parquet/jsonl/json/txt), repeatable; one bucket per file",
    )
    c.add_argument("--walled-col", default="chunk", help="Text column in the walled files")
    c.add_argument("--n", type=int, default=8, help="n-gram size (default 8)")
    c.add_argument("--jaccard", type=float, default=0.3, help="Jaccard threshold (default 0.3)")
    c.add_argument(
        "--containment", action="store_true", help="Containment mode instead of Jaccard"
    )
    c.add_argument("--warn", type=float, default=0.005, help="Warn fraction (default 0.005)")
    c.add_argument("--kill", type=float, default=0.02, help="Kill fraction (default 0.02)")
    c.add_argument("--output", help="Write the gate JSON here instead of stdout")
    c.set_defaults(func=cmd_contaminate)

    fm = stage.add_parser(
        "format",
        help="Stage 3 - normalise fetched corpora into the pair schema.",
        description=(
            "Emit `<name>_lane.parquet` plus its manifest: claim, chunk, label, doc_id, source, "
            "tag and the corpus's retained provenance columns. The manifest's expected counts "
            "are held unless --no-strict."
        ),
    )
    fm.add_argument("names", nargs="*", metavar="NAME", help="Corpora to format (default: all)")
    fm.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help=f"Default: {DEFAULT_DATA_DIR}")
    fm.add_argument("--lane-dir", default=DEFAULT_LANE_DIR, help=f"Default: {DEFAULT_LANE_DIR}")
    fm.add_argument(
        "--no-strict", action="store_true", help="Report count mismatches instead of refusing"
    )
    fm.set_defaults(func=cmd_format)

    sh = stage.add_parser(
        "shape",
        help="Stage 4 - the evidence-shape gate over lane parquets.",
        description=(
            "Report windows per row, total pairs, rows over the batch cap with their window "
            "counts, distinct documents and rows-per-document reuse, claim and evidence length "
            "distributions, positive fraction, and the projected share of a mix's training "
            "pairs. Verdicts: PASS, PASS-WITH-DROP (over-cap rows, counted), BLOCK "
            "(structural). Exit 1 if any lane BLOCKs."
        ),
    )
    sh.add_argument("lanes", nargs="+", metavar="LANE", help="Lane parquet(s) to census")
    _add_presentation_args(sh)
    sh.add_argument(
        "--mix-other-pairs",
        type=int,
        default=None,
        help="Training pairs contributed by the REST of the mix - enables the projected share",
    )
    sh.add_argument("--mix-other-rows", type=int, default=None, help="Rows in the rest of the mix")
    sh.add_argument("--mix-name", default="mix", help="Name of the mix being projected into")
    sh.add_argument("--json", action="store_true", help="Emit the full reports as JSON")
    sh.add_argument("--output", help="Write the JSON reports here")
    sh.set_defaults(func=cmd_shape)

    asm = stage.add_parser(
        "assemble",
        help="Stage 5 - build a mix from a JSON spec.",
        description=(
            "The spec names the lanes, their groups and the counts each must reproduce, plus "
            "the mix total and group map. Over-cap rows are dropped by the trainer's own rule "
            "and every drop is recorded. Any assertion that fails aborts the build."
        ),
    )
    asm.add_argument("--spec", required=True, help="Mix spec JSON")
    asm.add_argument("--out", help="Write the assembled mix parquet here")
    asm.add_argument("--output", help="Write the assembly record JSON here instead of stdout")
    asm.set_defaults(func=cmd_assemble)

    cen = stage.add_parser(
        "census",
        help="Stage 6 - the pre-spend report over an assembled mix.",
        description=(
            "Per group: rows, pairs, pair share, mean target and shape verdict; for the mix: "
            "the window census and a go/no-go. Exit 1 on no-go. This is what a trainer wrapper "
            "calls before touching a GPU."
        ),
    )
    cen.add_argument("mix", help="Assembled mix parquet")
    cen.add_argument("--group-col", default="group", help="Group column (default: group)")
    cen.add_argument("--name", default="", help="Mix name for the report")
    _add_presentation_args(cen)
    cen.add_argument("--json", action="store_true", help="Emit the report as JSON")
    cen.add_argument("--output", help="Write the report JSON here")
    cen.set_defaults(func=cmd_census)

    r = stage.add_parser(
        "run",
        help="Stages 1-4 for each corpus: fetch, gate, format, shape.",
        description=(
            "The per-corpus pipeline end to end, stopping at the first stage that refuses the "
            "corpus. The contamination stage runs only when --walled is given. Exit 1 if any "
            "corpus BLOCKs at shape."
        ),
    )
    r.add_argument("names", nargs="*", metavar="NAME", help="Corpora to run (default: all)")
    r.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help=f"Default: {DEFAULT_DATA_DIR}")
    r.add_argument("--lane-dir", default=DEFAULT_LANE_DIR, help=f"Default: {DEFAULT_LANE_DIR}")
    r.add_argument("--walled", action="append", help="Walled corpus file, repeatable")
    r.add_argument("--walled-col", default="chunk", help="Text column in the walled files")
    r.add_argument("--dry-run", action="store_true", help="Report what would be fetched, no work")
    r.add_argument(
        "--no-strict", action="store_true", help="Report count mismatches instead of refusing"
    )
    r.add_argument("--output", help="Write the run record JSON here instead of stdout")
    r.set_defaults(func=cmd_run)

    return ds
