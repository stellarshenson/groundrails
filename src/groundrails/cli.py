"""CLI entry point for the groundrails grounder.

Subcommands:
    ground             - ground claims against source text (regex + Levenshtein + BM25; --semantic adds the bundle)
    extract-claims     - heuristic sentence-to-claim extractor for a document
    check-consistency  - intra-document numeric/entity divergence detector
    config             - show the resolved grounding config + calibration block
    setup              - first-run: write the semantic model/cache config

Sources are read as plain UTF-8 text. Binary document readers (PDF / OCR / docx)
are a document-processing concern and live outside groundrails.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from groundrails import semantic_ov
from groundrails import settings as settings_mod
from groundrails.config import load_document_processing_config
from groundrails.grounding import (
    GroundingMatch,
    UnsupportedLanguageError,
    ground,
    ground_batch,
)


# --- optional semantic / NLI layer (opt-in via --semantic) -----------------
def _build_semantic_grounder(cfg, enabled: bool):
    """Return a SemanticGrounder, or None when the layer is not requested.

    ``--semantic`` is an explicit contract: deps present -> run, deps missing ->
    hard fail (exit 2). Silent degradation would produce misleading grounding
    reports (rows labelled ``(semantic)`` with score 0.000).
    """
    if not enabled:
        return None
    if not settings_mod.is_semantic_available():
        print(
            "ERROR: --semantic requires the [semantic] extras, but "
            "dependencies are missing. Install and rerun:\n"
            + settings_mod.semantic_install_hint(),
            file=sys.stderr,
        )
        sys.exit(2)
    from groundrails.semantic import SemanticGrounder

    return SemanticGrounder(
        model_name=cfg.semantic_model,
        device=cfg.semantic_device,
        cache_dir=cfg.cache_dir,
    )


def _build_nli_grounder(cfg, enabled: bool):
    """Return an NLIGrounder when semantic grounding is requested, else None.

    NLI rides with semantic (no separate switch). The NLI deps are a subset of
    the semantic extras; we still guard and return None if they are missing.
    """
    if not enabled:
        return None
    from groundrails import nli

    if not nli.is_available():
        return None
    return nli.NLIGrounder(model_name=getattr(cfg, "nli_model", None) or nli.DEFAULT_MODEL)


def _nli_calibrated_verdict():
    """Calibrated verdict used when NLI is active so the entailment signal is
    weighed against the other layers rather than hard-overriding the cascade.

    Prefers config-trained weights (``calibration.engine: calibrated``); falls
    back to the prior means. Built from point weights - no per-call PyMC sampling.
    """
    from groundrails import calibration as C

    trained = C.verdict_from_config()
    if trained is not None:
        return trained
    spec = C.load_prior_spec()
    return C.CalibratedVerdict.from_weights(
        {k: mu for k, (mu, _sd) in spec.items()}, threshold=0.5
    )


# --- source / claim / match helpers ----------------------------------------
def _read_sources(paths: list[str]) -> list[tuple[str, str]]:
    """Read each source path as plain UTF-8 text.

    Binary documents are out of scope for groundrails - they get a clear
    skip warning rather than a silent U+FFFD decode that yields zero hits.
    """
    out: list[tuple[str, str]] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            print(f"WARNING: source not found, skipped: {p}", file=sys.stderr)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError) as exc:
            print(
                f"WARNING: source skipped (not UTF-8 text - convert binary docs first): "
                f"{p} ({exc})",
                file=sys.stderr,
            )
            continue
        out.append((str(path), text))
    return out


def _read_claims(path_str: str) -> list[str]:
    """Read + validate a claims file against the Claim schema (``groundrails.claims``)."""
    from pydantic import ValidationError

    from groundrails.claims import load_claims

    try:
        claims = load_claims(path_str)
    except FileNotFoundError:
        print(f"ERROR: claims file not found: {path_str}", file=sys.stderr)
        raise SystemExit(1) from None
    except ValidationError as exc:
        print(f"ERROR: claims file does not conform to the Claim schema:\n{exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    if not claims:
        print(f"ERROR: claims file has no claims: {path_str}", file=sys.stderr)
        raise SystemExit(1)
    return [c.claim for c in claims]


def _loc_str(loc) -> str:
    """Format a Location as 'path L:C ¶para pgN'."""
    parts = []
    if loc.source_path:
        parts.append(loc.source_path)
    parts.append(f"L{loc.line_start}:C{loc.column_start}")
    if loc.line_end != loc.line_start:
        parts[-1] += f"-L{loc.line_end}:C{loc.column_end}"
    parts.append(f"¶{loc.paragraph}")
    if loc.page > 1:
        parts.append(f"pg{loc.page}")
    return " ".join(parts)


def _match_line(m: GroundingMatch) -> str:
    """One-line summary showing all layer scores with winning location."""
    if m.match_type == "exact":
        loc, winning = _loc_str(m.exact_location), m.exact_matched_text
    elif m.match_type == "fuzzy":
        loc, winning = _loc_str(m.fuzzy_location), m.fuzzy_matched_text
    elif m.match_type == "bm25":
        loc, winning = _loc_str(m.bm25_location), m.bm25_matched_text
    elif m.match_type == "semantic":
        loc, winning = _loc_str(m.semantic_location), m.semantic_matched_text
    elif m.match_type == "contradicted":
        if m.semantic_score > 0:
            loc, winning = _loc_str(m.semantic_location), m.semantic_matched_text
        elif m.bm25_score > 0:
            loc, winning = _loc_str(m.bm25_location), m.bm25_matched_text
        else:
            loc, winning = _loc_str(m.fuzzy_location), m.fuzzy_matched_text
    else:
        loc = "(no match)"
        winning = m.semantic_matched_text or m.bm25_matched_text or m.fuzzy_matched_text

    mismatch_info = ""
    if m.numeric_mismatches or m.entity_mismatches:
        mismatch_info = f" mismatches={m.numeric_mismatches + m.entity_mismatches}"

    return (
        f"{m.match_type.upper()} "
        f"exact={m.exact_score:.3f} fuzzy={m.fuzzy_score:.3f} "
        f"bm25={m.bm25_score:.3f} semantic={m.semantic_score:.3f} "
        f"agreement={m.agreement_score:.3f}{mismatch_info} @ {loc} | {winning!r}"
    )


# --- command handlers -------------------------------------------------------
def cmd_ground(args: argparse.Namespace) -> int:
    # Resolve claims and source from the positional `CLAIMS SOURCE` form or the flags.
    # Claims come from --claim (inline), --claims (file), or the first positional; sources
    # from --source (repeatable) and/or the remaining positionals.
    paths = list(args.paths)
    src = list(args.source)
    claim_single = args.claim
    claims_file = args.claims_file
    if claim_single is not None or claims_file is not None:
        src += paths  # claims came from a flag -> every positional is a source
    elif paths:
        if not src:  # positional form: `ground CLAIMS SOURCE`
            if len(paths) < 2:
                print(
                    "ERROR: need a claims file and a source: `ground CLAIMS SOURCE`, "
                    "or --claim/--claims with --source",
                    file=sys.stderr,
                )
                return 2
            claims_file, src = paths[0], paths[1:]
        else:  # source via --source -> the positional is the claims file
            claims_file, src = paths[0], src + paths[1:]
    else:
        print(
            "ERROR: provide claims and a source: `ground CLAIMS SOURCE`, "
            "or --claim/--claims with --source",
            file=sys.stderr,
        )
        return 2

    sources = _read_sources(src)
    if not sources:
        print("ERROR: no readable source provided (see warnings above)", file=sys.stderr)
        return 1
    # --semantic is the orthogonal switch: it turns on the OpenVINO cascade that
    # escalates the uncertain band of whatever --effort tier is selected. Deps present
    # -> run; deps missing -> hard fail (exit 2), never silent degradation.
    semantic = bool(getattr(args, "semantic", False))
    if semantic and not semantic_ov.is_available():
        print(
            "ERROR: --semantic needs the cascade extras (openvino + transformers).\n"
            + semantic_ov.install_hint(),
            file=sys.stderr,
        )
        return 2
    cfg = load_document_processing_config()
    if getattr(args, "effort", None):
        cfg = cfg.overlay(lexical_effort=args.effort)

    # single-claim mode: one match line, exit 0 if grounded
    if claim_single is not None:
        try:
            m = ground(
                claim_single,
                sources,
                fuzzy_threshold=args.threshold,
                bm25_threshold=args.bm25_threshold,
                config=cfg,
                semantic=semantic,
            )
        except UnsupportedLanguageError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
        if args.json:
            print(json.dumps(asdict(m), indent=2, default=str))
        else:
            print(_match_line(m))
        return 0 if m.match_type != "none" else 1

    # batch mode: ground every claim in the file, one report line each
    claims = _read_claims(claims_file)
    try:
        matches = ground_batch(
            claims,
            sources,
            fuzzy_threshold=args.threshold,
            bm25_threshold=args.bm25_threshold,
            config=cfg,
            semantic=semantic,
            primary_source=args.primary_source,
            max_workers=args.workers,
        )
    except UnsupportedLanguageError as exc:
        print(f"ERROR: {exc} (an unsupported-language claim is in the batch)", file=sys.stderr)
        return 3
    if args.json:
        report = json.dumps([asdict(m) for m in matches], indent=2, default=str)
    else:
        report = "\n".join(f"{i + 1}. {_match_line(m)}" for i, m in enumerate(matches))
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"wrote {len(matches)} results to {args.output}", file=sys.stderr)
    else:
        print(report)
    # `ground` is a gate: exit 1 if any claim is not grounded
    return 0 if all(m.match_type != "none" for m in matches) else 1


def cmd_extract_claims(args: argparse.Namespace) -> int:
    """Emit claims.json from a document using the heuristic extractor."""
    from groundrails.extract import extract_claims_from_file

    doc_path = Path(args.document)
    if not doc_path.is_file():
        print(f"ERROR: document not found: {args.document}", file=sys.stderr)
        return 2
    try:
        extracted = extract_claims_from_file(doc_path)
    except UnicodeDecodeError as exc:
        print(
            f"ERROR: {args.document} is not valid UTF-8 at byte {exc.start}: {exc.reason}. "
            f"Convert or re-encode first.",
            file=sys.stderr,
        )
        return 2

    from groundrails.claims import Claim

    # emit through the shared Claim schema so the written claims.json is conformant
    payload = [
        Claim(id=c.id, claim=c.claim, line_number=c.line_number).model_dump() for c in extracted
    ]
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {len(payload)} claims to {out}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))
    print(
        "NOTE: extract-claims uses a heuristic. Review claims.json before "
        "running `ground claims.json <source>` - short/ambiguous sentences may need rewording.",
        file=sys.stderr,
    )
    return 0


def cmd_check_consistency(args: argparse.Namespace) -> int:
    """Flag intra-document divergences: same number / entity category, different values."""
    from groundrails.consistency import check_consistency_in_file, format_consistency_report

    doc_path = Path(args.document)
    if not doc_path.is_file():
        print(f"ERROR: document not found: {args.document}", file=sys.stderr)
        return 2
    try:
        findings = check_consistency_in_file(doc_path)
    except UnicodeDecodeError as exc:
        print(
            f"ERROR: {args.document} is not valid UTF-8 at byte {exc.start}: {exc.reason}.",
            file=sys.stderr,
        )
        return 2

    report = format_consistency_report(findings, document_path=str(doc_path))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"wrote consistency report to {out} ({len(findings)} findings)", file=sys.stderr)
    else:
        print(report)
    return 0 if not findings else 1


def cmd_config(args: argparse.Namespace) -> int:
    """Show the resolved grounding config + calibration block."""
    import yaml

    from groundrails import calibration as C
    from groundrails.config import load_document_processing_config

    cfg = load_document_processing_config()
    print("# resolved groundrails config")
    print(yaml.safe_dump(asdict(cfg), sort_keys=False).rstrip())
    block = C.load_calibration_from_config()
    print("\n# calibration block")
    if block:
        print(yaml.safe_dump({"calibration": block}, sort_keys=False).rstrip())
    else:
        print("calibration: (none - deterministic classifier / default prior in use)")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Pre-fetch the semantic cascade int8 IRs into the HuggingFace cache.

    These are the only model weights groundrails downloads; the lexical tiers need
    none. Hard-fail (exit 2) if the cascade extras are missing - the pull needs
    huggingface_hub.
    """
    if not semantic_ov.is_available():
        print(
            "ERROR: downloading the cascade models needs the cascade extras "
            "(openvino + transformers + huggingface_hub).\n" + semantic_ov.install_hint(),
            file=sys.stderr,
        )
        return 2
    print(
        "Pre-fetching the semantic cascade int8 IRs (~1.4 GB) into the HuggingFace cache:",
        file=sys.stderr,
    )
    for name, repo, d in semantic_ov.download_models():
        print(f"  {name:<18} {repo} -> {d}", file=sys.stderr)
    print("done - the next `--semantic 1` run loads from cache.", file=sys.stderr)
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    if settings_mod.settings_exist() and not args.force:
        cfg = settings_mod.load()
        print(
            f"Settings already present at {settings_mod.settings_path()}.\n"
            f"  semantic_model   = {cfg.semantic_model}\n"
            f"  semantic_device  = {cfg.semantic_device}\n"
            f"  cache_dir        = {cfg.cache_dir}\n"
            "Semantic grounding (+ NLI) is opt-in per call via '--semantic'.\n"
            "Re-run with --force to reconfigure.",
            file=sys.stderr,
        )
        return 0
    settings_mod.prompt_first_run()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundrails",
        description="Grounding guardrails: deterministic, torch-free claim verification.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser(
        "ground",
        help="Ground claims against source text (regex + Levenshtein + BM25; --semantic adds the bundle).",
        description=(
            "Ground claims against a source. Simple form: `ground CLAIMS SOURCE` (a claims "
            "file then a source file). Single claim: `--claim TEXT --source SRC`. Claims file "
            "by flag: `--claims FILE --source SRC`. Three lexical layers always run - regex "
            "exact, Levenshtein partial-ratio, BM25 token-recall; `--semantic 1` adds the "
            "OpenVINO cascade. Exit 0 if grounded, 1 if any claim is not."
        ),
    )
    g.add_argument(
        "paths",
        nargs="*",
        metavar="CLAIMS SOURCE",
        help="Positional form: a claims file then a source file (`ground claims.json evidence.txt`)",
    )
    g.add_argument(
        "--claim", help="A single claim to ground against the source (instead of a claims file)"
    )
    g.add_argument(
        "--claims",
        dest="claims_file",
        help="Claims file: JSON list / {claim,...} objects (from extract-claims), or text one-per-line",
    )
    g.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source/evidence file (repeatable; flag form of the positional SOURCE)",
    )
    g.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Levenshtein ratio threshold for 'fuzzy' (default 0.85)",
    )
    g.add_argument(
        "--bm25-threshold",
        type=float,
        default=0.5,
        help="BM25 token-recall threshold for 'bm25' (default 0.5)",
    )
    g.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.6,
        help="Semantic cosine threshold for 'semantic' (default 0.6)",
    )
    g.add_argument(
        "--semantic-threshold-percentile",
        type=float,
        default=None,
        help="Percentile threshold; overrides --semantic-threshold when set.",
    )
    g.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Lexical solution tier (default from config; low/medium/high). Orthogonal to --semantic.",
    )
    g.add_argument(
        "--semantic",
        type=int,
        choices=[0, 1],
        default=0,
        metavar="{0,1}",
        help="Switch on (1) the OpenVINO cascade: escalates the uncertain band of the --effort "
        "tier to bge-reranker + mDeBERTa-NLI, fused by the joint head. Default 0 (off).",
    )
    g.add_argument(
        "--primary-source",
        dest="primary_source",
        default=None,
        help="Batch mode: the source expected to ground the claims (cross-source flag).",
    )
    g.add_argument(
        "--output", help="Batch mode: write the report to this path instead of stdout"
    )
    g.add_argument(
        "--json", action="store_true", help="Emit JSON instead of the match line / report"
    )
    g.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Batch mode: worker threads for per-claim grounding (default 5; 1 = serial)",
    )
    g.set_defaults(func=cmd_ground)

    ex = sub.add_parser(
        "extract-claims",
        help="Heuristic sentence-to-claim extractor; emits claims.json to feed `ground`.",
        description=(
            "Walk a markdown/text document and emit claim candidates with stable IDs. "
            "LOSSY - review claims.json before grounding."
        ),
    )
    ex.add_argument("--document", required=True, help="Source document (markdown or plain text)")
    ex.add_argument("--output", help="Write claims.json here; omitted -> stdout")
    ex.set_defaults(func=cmd_extract_claims)

    cc = sub.add_parser(
        "check-consistency",
        help="Flag intra-document divergences (same category, different value).",
        description=(
            "Pure intra-document check; no source needed. Extracts numbers and named "
            "entities and reports categories where multiple distinct values appear."
        ),
    )
    cc.add_argument("--document", required=True, help="Document to analyse")
    cc.add_argument("--output", help="Write markdown report here; omitted -> stdout")
    cc.set_defaults(func=cmd_check_consistency)

    cf = sub.add_parser(
        "config",
        help="Show the resolved grounding config + calibration block.",
    )
    cf.set_defaults(func=cmd_config)

    su = sub.add_parser(
        "setup",
        help="First-run setup: write the semantic model/cache config.",
        description=(
            "Write .stellars-plugins/settings.json with the semantic model / cache "
            "config. Semantic grounding (+ NLI) is opt-in per call via --semantic."
        ),
    )
    su.add_argument(
        "--force", action="store_true", help="Re-prompt even if settings already exist"
    )
    su.set_defaults(func=cmd_setup)

    dl = sub.add_parser(
        "download",
        help="Pre-download the semantic cascade models (~1.4 GB) into the HuggingFace cache.",
        description=(
            "Fetch the int8 OpenVINO IRs the --semantic cascade needs (bge-m3 + "
            "bge-reranker + mDeBERTa-NLI) so the first --semantic run is warm. These are "
            "the only model weights groundrails downloads; the lexical tiers need none. "
            "Requires the [semantic-grounder] extra."
        ),
    )
    dl.set_defaults(func=cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
