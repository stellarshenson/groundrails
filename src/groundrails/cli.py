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
    build_grounding_document,
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


def _read_claims(path_str: str):
    """Read + validate a claims file against the Claim schema (``groundrails.claims``).

    Returns the validated ``Claim`` objects (carrying id / line / char span) so the
    grounding document can report each claim's answer-document location."""
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
    return claims


def _extract_claim_objs(document_path: str):
    """Extract claims from a positional document (the answer to check); ``None`` on error,
    with the message already printed. Each extracted claim carries its location in the document."""
    from groundrails.extract import extract_claims_from_file

    p = Path(document_path)
    if not p.is_file():
        print(f"ERROR: document not found: {document_path}", file=sys.stderr)
        return None
    try:
        claims = extract_claims_from_file(p)
    except UnicodeDecodeError as exc:
        print(
            f"ERROR: {document_path} is not valid UTF-8 at byte {exc.start}: {exc.reason}.",
            file=sys.stderr,
        )
        return None
    if not claims:
        print(
            f"ERROR: no claims extracted from {document_path} "
            "(use --claims for a structured claims file)",
            file=sys.stderr,
        )
        return None
    return claims


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
    # Hard gate: grounding requires `groundrails init` first (main() loads groundrails.json,
    # which marks the grounder ready). No file -> not ready -> refuse rather than grounding
    # with un-provisioned defaults.
    if not settings_mod.is_ready():
        print(
            "ERROR: groundrails is not initialized - run `groundrails init` first "
            "(it provisions resources and writes groundrails.json)",
            file=sys.stderr,
        )
        return 2
    # Resolve the claim source - exactly one of: a positional DOCUMENT (claims extracted from
    # it), --claims FILE (a structured claims file), or one-or-more --claim TEXT (inline) - plus
    # the evidence (the remaining positionals and any --source). A claims.json goes through
    # --claims; the positional is always a document to pull claims from.
    paths = list(args.paths)
    src = list(args.source)
    inline = args.claim  # list of inline claims (repeatable) or None
    claims_file = args.claims_file

    if inline and claims_file:
        print("ERROR: choose one claim source - a document, --claims, or --claim", file=sys.stderr)
        return 2

    if inline:  # --claim TEXT (repeatable): every positional is evidence
        from groundrails.claims import Claim

        claim_objs = [Claim(claim=c) for c in inline]
        evidence = src + paths
    elif claims_file:  # --claims FILE: every positional is evidence
        claim_objs = _read_claims(claims_file)
        evidence = src + paths
    else:  # default form: `ground DOCUMENT EVIDENCE ...`
        if not paths:
            print(
                "ERROR: provide a document then evidence (`ground DOCUMENT EVIDENCE ...`), "
                "or --claims/--claim with evidence",
                file=sys.stderr,
            )
            return 2
        document, evidence = paths[0], src + paths[1:]
        if not evidence:
            print("ERROR: need at least one evidence source after the document", file=sys.stderr)
            return 2
        claim_objs = _extract_claim_objs(document)
        if claim_objs is None:
            return 2

    sources = _read_sources(evidence)
    if not sources:
        print("ERROR: no readable evidence provided (see warnings above)", file=sys.stderr)
        return 1
    # --semantic is the orthogonal switch: it turns on the OpenVINO cascade that escalates the
    # uncertain band of whatever --effort tier is selected. Deps present -> run; missing -> hard
    # fail (exit 2), never silent degradation.
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

    try:
        matches = ground_batch(
            [c.claim for c in claim_objs],
            sources,
            fuzzy_threshold=args.threshold,
            bm25_threshold=args.bm25_threshold,
            config=cfg,
            semantic=semantic,
            primary_source=args.primary_source,
            max_workers=args.workers,
            ignore_language=getattr(args, "ignore_language", False),
        )
    except UnsupportedLanguageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if getattr(args, "full_output", False):
        report = json.dumps([asdict(m) for m in matches], indent=2, default=str)
    elif args.json:
        report = json.dumps(
            build_grounding_document(matches, claims=claim_objs, sources=sources),
            indent=2,
            default=str,
        )
    elif len(matches) == 1:
        report = _match_line(matches[0])
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

    # emit through the shared Claim schema so the written claims.json is conformant; the
    # char span locates each claim in the answer document (-1 -> unfound -> omitted as null)
    payload = [
        Claim(
            id=c.id,
            claim=c.claim,
            line_number=c.line_number,
            char_start=c.char_start if c.char_start >= 0 else None,
            char_end=c.char_end if c.char_end >= 0 else None,
        ).model_dump()
        for c in extracted
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
    """Show the built-in runtime settings (no file is written)."""
    cfg = settings_mod.get()
    print(
        "groundrails settings are built in - configure them per call via CLI flags or "
        "`groundrails init` (no settings.json is written).\n"
        f"  semantic_model  = {cfg.semantic_model}\n"
        f"  cache_dir       = {cfg.resolved_cache_dir()}\n"
        f"  calibration     = {cfg.calibration_path or '(bundled default)'}\n"
        f"  models_dir      = {cfg.models_dir or '(HuggingFace cache)'}\n"
        "Semantic grounding (+ NLI) is opt-in per call via '--semantic'.",
        file=sys.stderr,
    )
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Provision calibration + models from S3 / local / HuggingFace in one call."""
    from groundrails.bootstrap import init as _init

    langs = [s.strip() for s in args.languages.split(",") if s.strip()] if args.languages else None
    summary = _init(
        source=args.source,
        calibration=args.calibration,
        models=args.models,
        languages=langs,
        wordnet=not args.no_wordnet,
        semantic_model=args.semantic_model,
        cache_dir=args.cache_dir,
        aws_profile=args.aws_profile,
        aws_endpoint_url=args.aws_endpoint_url,
        aws_region=args.aws_region,
        home=args.home,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_calibration_export(args: argparse.Namespace) -> int:
    """Export the active calibration block to a JSON file - the provisioned artifact."""
    from groundrails.calibration import export_calibration

    p = export_calibration(args.output, source=args.source)
    print(f"calibration exported -> {p}", file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundrails",
        description="Grounding guardrails: deterministic, torch-free claim verification.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser(
        "ground",
        help="Ground a document's claims against evidence (regex + Levenshtein + BM25; --semantic adds the bundle).",
        description=(
            "Ground claims against evidence. Default: `ground DOCUMENT EVIDENCE ...` - claims are "
            "extracted from the one document and checked against the evidence sources. Or pass "
            "claims explicitly: `--claims FILE EVIDENCE ...` (a claims file) or `--claim TEXT "
            "[--claim TEXT ...] EVIDENCE ...` (inline, repeatable). Three lexical layers always "
            "run - regex exact, Levenshtein partial-ratio, BM25 token-recall; `--semantic 1` adds "
            "the OpenVINO cascade. Exit 0 if every claim is grounded, 1 if any is not."
        ),
    )
    g.add_argument(
        "paths",
        nargs="*",
        metavar="DOCUMENT EVIDENCE",
        help="`ground DOCUMENT EVIDENCE ...`: one document to extract claims from, then one or "
        "more evidence sources. With --claims/--claim, every positional is an evidence source.",
    )
    g.add_argument(
        "--claim",
        action="append",
        help="An inline claim to ground (repeatable: pass --claim several times). Positionals are evidence.",
    )
    g.add_argument(
        "--claims",
        dest="claims_file",
        help="A claims file (JSON list / {claim,...} objects from extract-claims, or one claim per "
        "line). The only way to pass a structured claims file; positionals are evidence.",
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
        "--ignore-language",
        dest="ignore_language",
        action="store_true",
        help="Skip the HIGH-tier non-English hard-block: score every claim as-is instead of "
        "refusing claims whose language is detected as non-English with no installed MT model.",
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
    g.add_argument("--output", help="Batch mode: write the report to this path instead of stdout")
    g.add_argument(
        "--json",
        action="store_true",
        help="Emit the grounding document as JSON: per claim -> verdict, final score, and "
        "support provenance (claim location + evidence location). The business-end machine "
        "interface; internal per-scorer detail is omitted unless --full-output is set.",
    )
    g.add_argument(
        "--full-output",
        dest="full_output",
        action="store_true",
        help="JSON with the full per-scorer GroundingMatch (exact/fuzzy/bm25/semantic scores "
        "and every layer's location), not just the business-end document.",
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
        help="Show the built-in runtime settings (no file written).",
        description=(
            "Print the resolved built-in runtime settings. groundrails writes no "
            "settings.json - configure per call via CLI flags or `groundrails init`."
        ),
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

    ini = sub.add_parser(
        "init",
        help="Provision calibration + models from S3 / local / HuggingFace in one call.",
        description=(
            "One-call bootstrap for offline / Lambda use. Every required resource resolves "
            "S3 -> local folder -> HuggingFace, each overridable by a flag. Writes the "
            "provisioned calibration JSON + a local model mirror under GROUNDRAILS_HOME; "
            "no settings.json is written."
        ),
    )
    ini.add_argument(
        "--source", help="Default base for the chain: s3://bucket/prefix or a local dir"
    )
    ini.add_argument(
        "--calibration", help="Calibration override: s3://… | https://… | /local/calibration.json"
    )
    ini.add_argument(
        "--models", help="Model source: s3://…/models | /local/models | 'hf' | 'none'"
    )
    ini.add_argument("--languages", help="Comma-separated argos MT langs to prefetch (e.g. fr,de)")
    ini.add_argument(
        "--no-wordnet", action="store_true", help="Skip ensuring the NLTK WordNet corpus"
    )
    ini.add_argument(
        "--semantic-model", dest="semantic_model", help="Override the semantic bi-encoder model id"
    )
    ini.add_argument("--cache-dir", dest="cache_dir", help="Parquet cache dir override")
    ini.add_argument(
        "--aws-profile", dest="aws_profile", help="botocore profile for S3 (omit in Lambda)"
    )
    ini.add_argument(
        "--aws-endpoint-url",
        dest="aws_endpoint_url",
        help="S3-compatible endpoint URL (e.g. RustFS)",
    )
    ini.add_argument("--aws-region", dest="aws_region", help="AWS region for S3")
    ini.add_argument(
        "--home", help="GROUNDRAILS_HOME for fetched assets (default ~/.cache/groundrails)"
    )
    ini.set_defaults(func=cmd_init)

    ca = sub.add_parser(
        "calibration", help="Calibration utilities (export the active calibration to JSON)."
    )
    ca_sub = ca.add_subparsers(dest="cal_cmd", required=True)
    ca_exp = ca_sub.add_parser(
        "export",
        help="Write the active calibration block to a JSON file (the provisioned artifact).",
    )
    ca_exp.add_argument("-o", "--output", required=True, help="Path to write calibration.json")
    ca_exp.add_argument(
        "--source", help="Read calibration from this override instead of the active one"
    )
    ca_exp.set_defaults(func=cmd_calibration_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Load a provisioned groundrails.json (written by `groundrails init`) so grounding
    # commands run ready; `init` itself writes the file, so it is skipped here.
    if getattr(args, "func", None) is not cmd_init:
        settings_mod.load_config_file()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
