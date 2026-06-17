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

from groundrails import settings as settings_mod
from groundrails.grounding import GroundingMatch, ground, ground_batch


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


def _read_claims_manifest(path_str: str) -> list[str]:
    """Read a claims manifest: a JSON list of strings or ``{claim, ...}`` objects."""
    claims_path = Path(path_str)
    if not claims_path.is_file():
        print(f"ERROR: claims manifest not found: {path_str}", file=sys.stderr)
        raise SystemExit(1)
    raw = json.loads(claims_path.read_text(encoding="utf-8"))
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return raw
    if isinstance(raw, list) and all(isinstance(x, dict) and "claim" in x for x in raw):
        return [x["claim"] for x in raw]
    print(
        "ERROR: claims manifest must be a JSON list of strings or objects with a 'claim' key",
        file=sys.stderr,
    )
    raise SystemExit(1)


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
    sources = _read_sources(args.source)
    if not sources:
        print("ERROR: no readable --source provided (see warnings above)", file=sys.stderr)
        return 1
    enabled = bool(getattr(args, "semantic", False))
    cfg = settings_mod.ensure_loaded(auto_prompt=False) if enabled else None
    grounder = _build_semantic_grounder(cfg, enabled)
    nli_grounder = _build_nli_grounder(cfg, enabled)
    verdict = _nli_calibrated_verdict() if enabled else None

    if getattr(args, "manifest", None):
        claims = _read_claims_manifest(args.manifest)
        matches = ground_batch(
            claims,
            sources,
            fuzzy_threshold=args.threshold,
            bm25_threshold=args.bm25_threshold,
            semantic_threshold=args.semantic_threshold,
            semantic_threshold_percentile=args.semantic_threshold_percentile,
            semantic_grounder=grounder,
            nli_grounder=nli_grounder,
            calibrated_verdict=verdict,
            primary_source=args.primary_source,
            max_workers=args.workers,
        )
        if args.json:
            report = json.dumps([asdict(m) for m in matches], indent=2, default=str)
        else:
            report = "\n".join(f"{i + 1}. {_match_line(m)}" for i, m in enumerate(matches))
        if args.output:
            Path(args.output).write_text(report, encoding="utf-8")
            print(f"wrote {len(matches)} results to {args.output}", file=sys.stderr)
        else:
            print(report)
        return 0

    m = ground(
        args.claim,
        sources,
        fuzzy_threshold=args.threshold,
        bm25_threshold=args.bm25_threshold,
        semantic_threshold=args.semantic_threshold,
        semantic_threshold_percentile=args.semantic_threshold_percentile,
        semantic_grounder=grounder,
        nli_grounder=nli_grounder,
        calibrated_verdict=verdict,
    )
    if args.json:
        print(json.dumps(asdict(m), indent=2, default=str))
    else:
        print(_match_line(m))
    return 0 if m.match_type != "none" else 1


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

    payload = [{"id": c.id, "claim": c.claim, "line_number": c.line_number} for c in extracted]
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {len(payload)} claims to {out}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))
    print(
        "NOTE: extract-claims uses a heuristic. Review claims.json before "
        "running `ground --manifest` - short/ambiguous sentences may need rewording.",
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
            "One claim via --claim, or many via --manifest (a claims file). Three "
            "lexical layers always run - regex exact, Levenshtein partial-ratio, BM25 "
            "token-recall; all scores reported. --semantic adds the embedding + NLI "
            "entailment + calibrated-verdict bundle. Single-claim mode prints one match "
            "line (exit 0 if grounded, 1 if not); manifest mode writes a report."
        ),
    )
    claim_src = g.add_mutually_exclusive_group(required=True)
    claim_src.add_argument("--claim", help="A single claim to locate (single-claim mode)")
    claim_src.add_argument(
        "--manifest",
        help="Claims manifest path (JSON list of strings or {claim,...}); batch mode -> report",
    )
    g.add_argument(
        "--source",
        action="append",
        default=[],
        required=True,
        help="Source text file path (repeatable; read as UTF-8)",
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
        "--semantic",
        action="store_true",
        help="Enable the semantic bundle (embedding + NLI + calibrated verdict); default off",
    )
    g.add_argument(
        "--primary-source",
        dest="primary_source",
        default=None,
        help="Manifest mode: the source expected to ground the claims (cross-source flag).",
    )
    g.add_argument(
        "--output", help="Manifest mode: write the report to this path instead of stdout"
    )
    g.add_argument(
        "--json", action="store_true", help="Emit JSON instead of the match line / report"
    )
    g.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Manifest mode: worker threads for per-claim grounding (default 5; 1 = serial)",
    )
    g.set_defaults(func=cmd_ground)

    ex = sub.add_parser(
        "extract-claims",
        help="Heuristic sentence-to-claim extractor; emits claims.json for `ground --manifest`.",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
