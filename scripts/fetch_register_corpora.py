"""Fetch the register-gap corpora admitted by the author's 2026-08-09 rulings.

Registered against R14-H136 in docs/experiments/semantic-grounding-experiments.md
("Author rulings (2026-08-09): corpus admissions"); scouting paper trail in
experiments/grounding-semantic/R14_corpus_scout.md.

Four corpora, two measured training-register gaps:

  GAP A - financial discourse (382x prevalence gap vs the finqa arena subset)
    edgar-restricted  EDGAR-CORPUS, non-S&P-500 filers AND filing year >= 2020 only

  GAP B - procedural / product-manual register (1,645x gap vs delucionqa)
    army-tm           US Army operator / maintenance technical manuals
    faa-amt           FAA Aviation Maintenance Technician handbooks

  Science-claim supplement, admitted on upstream AI2 terms
    scifact           AI2 SciFact claims + abstracts

Every subcommand is idempotent and resumable: files already on disk are skipped,
progress is checkpointed to `_state.json` next to the payload, and a kill at any
point loses at most the file in flight. Re-run the same command to continue.

Run:  uv run python scripts/fetch_register_corpora.py list
      uv run python scripts/fetch_register_corpora.py scifact
      uv run python scripts/fetch_register_corpora.py faa-amt
      uv run python scripts/fetch_register_corpora.py edgar-restricted
      uv run python scripts/fetch_register_corpora.py army-tm --per-day 100
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "external" / "datasets"

# SEC asks automated clients to identify themselves; faa.gov rejects the default
# urllib agent on HTML pages.
UA = "groundrails-research/0.1 (konrad.jelen@kolomolo.com)"


# --------------------------------------------------------------------------- #
# sidecars - one source of truth, same shape as dataset-ragtruth.md
# --------------------------------------------------------------------------- #

SIDECAR = """# {title}

{why}

{bullets}
## Caveats

{caveats}

## Provenance

{provenance}

Fetched by `scripts/fetch_register_corpora.py {name}`. The downloaded data under
`data/external/datasets/{name}/` is gitignored; this sidecar is tracked.
"""

EDGAR_SPEC = {
    "title": "EDGAR-CORPUS, restricted slice (non-S&P-500 filers, filing year >= 2020)",
    "licence": "Apache-2.0 (HuggingFace dataset tag)",
    "why": """The only corpus carrying 10-K management-discussion discourse at volume, which the
R14 register-gap audit measured as the largest finqa-side training deficit - 904 rows carrying
financial vocabulary, 0.13% of the mix, against 50.3% of finqa arena rows, a 382x gap. Admitted
only as a restricted slice, because raw EDGAR sits on the same document population FinQA was
built from.""",
    "bullets": """- **Source** - `eloukas/edgar-corpus` on HuggingFace; the year-2020 `train` /
  `validate` / `test` shards fetched as raw JSONL from the Hub resolve endpoint
- **Licence** - Apache-2.0 (HuggingFace dataset tag); the underlying 10-K filings are US
  government-published public records carrying no copyright
- **Restriction** - **non-S&P-500 filers AND filing year >= 2020, both clauses, no relaxation**.
  EDGAR-CORPUS ends at 2020, so the year clause selects the `2020` shards alone; the filer clause
  drops every CIK resolving to an S&P 500 constituent at any point in 1999-2019
- **Reason for the restriction** - FinQA's source population is S&P 500 annual reports 1999-2019,
  reached via FinTabNet. Excluding those filers and those years makes the slice document-disjoint
  from FinQA's population by company and by year, which is what turns the corpus from SUSPECT-HIGH
  into gateable
- **S&P 500 exclusion list** - union of constituents over 1999-2019 from `fja05680/sp500`
  (`S&P 500 Historical Components & Changes.csv`), tickers resolved to CIK through the SEC's
  `company_tickers.json`: 981 distinct tickers, 550 resolving to 546 distinct CIKs, 431
  unresolved because the company no longer trades under that ticker
- **Size** - **6,379 filings kept** of 6,851 raw 2020 filings; 472 dropped by the S&P 500 clause,
  0 by the year clause (the 2020 shards are already year-pure). 4,384 of the survivors carry an
  MD&A section over 500 characters. Full breakdown in `edgar-restricted/_counts.json`
- **Languages** - English
- **How negatives were made** - none ship with the corpus; it arrives as unlabeled clean prose and
  negatives are manufactured by the admitted DR corruption engine at lane build
- **How labels were made** - unlabeled
- **Mapping onto our task** - section_7 (MD&A) prose chunked to the project's 1,500-char window →
  evidence; claims manufactured at lane build
""",
    "caveats": """The 8-gram Jaccard provenance gate against the finqa and tatqa arena documents
(`experiments/grounding-semantic/provenance_gate.py`, the SciFact gate pattern, KILL at > 2%
overlap) has **not** run at fetch time - it runs at LANE BUILD, and the slice enters no training
mix until it passes.

The S&P 500 exclusion resolves historical tickers through a present-day SEC ticker-to-CIK map, so
constituents delisted, acquired or renamed before that map was published do not resolve and are
not excluded. The provenance gate is the backstop for that residue.

This slice overturns the standing no-EDGAR ruling for itself alone (`R10-H108_gate_report.md`);
the ban stands for every other EDGAR packaging.""",
    "provenance": """Admitted by the author's ruling of 2026-08-09, clause 1 ("EDGAR
admit-with-restriction"), recorded in the final block of
`docs/experiments/semantic-grounding-experiments.md` and carried by hypothesis R14-H136. Scouted
in `experiments/grounding-semantic/R14_corpus_scout.md` section 1, where the corpus is rated
register fit A-strong and wall verdict SUSPECT-HIGH pending exactly this filter.""",
}

SCIFACT_SPEC = {
    "title": "SciFact",
    "licence": "CC BY 4.0 (claims) + ODC-By (abstracts), per upstream AI2",
    "why": """Expert-written scientific claims paired with the abstracts that support or refute
them - a near-miss construction in a register no other admitted corpus covers, and the R13 gate
already measured it clean against the arena.""",
    "bullets": """- **Source** - the upstream AI2 release,
  `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`
- **Licence** - **CC BY 4.0 (claims) + ODC-By (abstracts)**, per the upstream AI2 release. These
  terms are authoritative over the HuggingFace mirror `allenai/scifact`, which tags the dataset
  `cc-by-nc-2.0`; the discrepancy is recorded under Caveats
- **Size** - 1,258 labelled (claim, abstract) rows from train+dev - 508 SUPPORT / 265 CONTRADICT /
  485 NEI; 5,183 abstracts in the corpus file
- **Languages** - English
- **How negatives were made** - expert claim rewriting: annotators negate or alter a claim drawn
  from a citation sentence so the same abstract refutes it
- **How labels were made** - human expert annotation with rationale sentences (SUPPORT /
  CONTRADICT / NOINFO)
- **Mapping onto our task** - claim → claim; cited abstract → evidence; SUPPORT → 1, CONTRADICT
  and NEI → 0
""",
    "caveats": """**Licence discrepancy, recorded deliberately**: the HuggingFace mirror
`allenai/scifact` carries a `cc-by-nc-2.0` tag while the upstream AI2 release states CC BY 4.0 for
the claims and ODC-By for the abstracts. Data here is taken from the AI2 S3 release and the
upstream terms are treated as authoritative, per the author's ruling of 2026-08-09. A
non-commercial reading would bar any shipped model trained on it.

The test split is blind (no labels) and unusable for training or evaluation. Biomedical-literature
register, not conversational RAG.""",
    "provenance": """Admitted by the author's ruling of 2026-08-09, clause 2 ("NC-class"); iFixit
was refused in the same clause. The provenance gate is already recorded in the canonical log: 0 of
5,183 abstracts match pubmedqa / covidqa / expertqa arena documents at 8-gram Jaccard >= 0.3 (max
best-Jaccard 0.0163, p99 0.0046, spike controls 9/9) -
`experiments/grounding-semantic/R13-scifact_gates_result.json`.""",
}

ARMY_SPEC = {
    "title": "US Army technical manuals (operator and maintenance)",
    "licence": "Public domain - US Government work (17 U.S.C. 105)",
    "why": """The procedural / product-manual register is the most extreme measured training gap -
181 rows from 30 distinct documents, 0.026% of the mix, against 43.4% of delucionqa arena rows, a
1,645x gap that no re-weighting can close because the rows are absent. This corpus is the only
candidate that fixes document diversity rather than row count, and it is public domain.""",
    "bullets": """- **Source** - `https://www.liberatedmanuals.com/`, index `all.mpl`, one PDF per
  manual identifier
- **Licence** - **public domain**; works of US federal employees carry no copyright
  (17 U.S.C. 105), and the mirror states the manuals "are NOT subject to copyright, can be freely
  copied and redistributed"
- **Size** - 4,792 distinct PDFs in the index, filtered to the **1,766 operator / unit /
  direct-support maintenance manuals**: identifier tails `-10`, `-12`, `-13`, `-14`, `-20`, `-23`,
  `-24`, `-34` with no `P` anywhere in the identifier. The 2,203 parts-list PDFs are excluded as
  catalogue tables, the wrong register
- **Languages** - English
- **Register fit** - measured `procden` 23.45 on `TM-10-3930-630-12` against the arena-median bar
  11.33: warnings, cautions, numbered steps, torque values, "refer to paragraph" cross-references
- **How negatives were made** - none ship with the corpus; unlabeled clean prose, negatives
  manufactured by the admitted DR corruption engine at lane build
- **How labels were made** - unlabeled
- **Mapping onto our task** - manual text chunked to the project's 1,500-char window → evidence;
  claims manufactured at lane build
""",
    "caveats": """**Acquisition is rate-limited, not instant.** The mirror allows roughly 100
manuals per IP per day, so the full 1,766-file pull takes about 18 days of polite crawling. The
downloader is detached, budgeted per day and resumable: it sleeps to the next daily window when
the budget or the server's limit is reached, and re-running continues from `army-tm/_state.json`.
The mirror also refuses individual requests transiently - observed once at file 10, with the same
URL serving normally seconds later - so a refusal backs off for two minutes and only a run of five
consecutive refusals is read as the daily allowance being spent.

**The archive.org mirror route does not cover this set.** Probed before acquisition: the
`military-manuals` collection holds 678 items against the 1,766 targets, direct identifier
resolution hit 0 of 12 sampled targets, and full-text search hit 2 of 10 with one of those two a
wrong-document match. Archive.org cannot supply the measured set, so the rate-limited primary
mirror is the route taken.

Older scanned TMs extract with OCR noise ("Highgway", "Diagramn" observed in a 1970s document), so
a text-quality filter is required at lane build. The provenance gate against the arena documents
runs at lane build, not at fetch.""",
    "provenance": """Admitted by the author's ruling of 2026-08-09, clause 3 ("Army TMs"),
acquisition immediate, under hypothesis R14-H136. Scouted and measured in
`experiments/grounding-semantic/R14_corpus_scout.md` section 8, wall verdict CLEAN - no RAGBench
subset draws on US military documentation (delucionqa is Jeep, emanual is Samsung, techqa is
IBM).""",
}

FAA_SPEC = {
    "title": "FAA Aviation Maintenance Technician handbooks",
    "licence": "Public domain - US Government work (17 U.S.C. 105)",
    "why": """Born-digital procedural text that widens the maintenance vocabulary beyond the Army
TMs' ground vehicles - inspection procedures, warnings and cautions, fastener and connector
vocabulary, torque and tolerance numerics - with the clean text extraction the older Army scans
cannot offer.""",
    "bullets": """- **Source** - `https://www.faa.gov/regulations_policies/handbooks_manuals/`
  `aviation/`, direct PDF URLs
- **Licence** - **public domain**; works of the US Government carry no copyright (17 U.S.C. 105)
- **Size** - 3 handbooks, roughly 1,500 pages combined: AMT General (FAA-H-8083-30B), AMT Airframe
  (FAA-H-8083-31B), AMT Powerplant (FAA-H-8083-32B)
- **Languages** - English
- **How negatives were made** - none ship with the corpus; unlabeled clean prose, negatives
  manufactured by the admitted DR corruption engine at lane build
- **How labels were made** - unlabeled
- **Mapping onto our task** - handbook text chunked to the project's 1,500-char window → evidence;
  claims manufactured at lane build
""",
    "caveats": """Low document diversity by construction - three documents, which is exactly the
failure mode the register-gap audit diagnosed. This is a SUPPLEMENT to the Army TMs and must never
form a lane alone.

`faa.gov` answers HTTP 403 to an unrecognised user agent, on the PDF URLs as well as the HTML
pages, so the downloader sends a browser-shaped agent. The provenance gate against the arena
documents runs at lane build, not at fetch.""",
    "provenance": """Admitted by the author's ruling of 2026-08-09, clause 3, as the born-digital
supplement to the Army TMs, under hypothesis R14-H136. Scouted in
`experiments/grounding-semantic/R14_corpus_scout.md` section 9, wall verdict CLEAN, register fit
B-strong for the AMT handbooks specifically.""",
}

SPECS = {
    "edgar-restricted": EDGAR_SPEC,
    "scifact": SCIFACT_SPEC,
    "army-tm": ARMY_SPEC,
    "faa-amt": FAA_SPEC,
}


def write_sidecar(name: str) -> Path:
    spec = SPECS[name]
    path = OUT / f"dataset-{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SIDECAR.format(name=name, **{k: v for k, v in spec.items() if k != "licence"}))
    return path


# --------------------------------------------------------------------------- #
# shared plumbing - checkpoints and resumable downloads
# --------------------------------------------------------------------------- #


def load_state(name: str) -> dict:
    p = OUT / name / "_state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(name: str, state: dict) -> None:
    p = OUT / name / "_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    tmp.replace(p)


def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download(url: str, dest: Path, timeout: int = 900, agent: str = UA) -> tuple[bool, int]:
    """Download to `dest`, skipping if it is already there. Writes through a
    `.part` file so a kill mid-transfer never leaves a truncated file looking
    complete. Returns (downloaded_now, size_bytes)."""
    if dest.exists() and dest.stat().st_size > 0:
        return False, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(req, timeout=timeout) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    part.replace(dest)
    return True, dest.stat().st_size


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# scifact
# --------------------------------------------------------------------------- #

SCIFACT_TARBALL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_LICENCE = "https://raw.githubusercontent.com/allenai/scifact/master/LICENSE.md"


def cmd_scifact(args: argparse.Namespace) -> None:
    d = OUT / "scifact"
    d.mkdir(parents=True, exist_ok=True)

    tar = d / "data.tar.gz"
    new, size = download(SCIFACT_TARBALL, tar)
    log(f"data.tar.gz {'downloaded' if new else 'already present'}: {size / 1e6:.1f} MB")

    lic = d / "LICENSE.md"
    if not lic.exists():
        lic.write_bytes(fetch_bytes(SCIFACT_LICENCE))
        log("LICENSE.md downloaded from allenai/scifact")

    if not (d / "data").exists():
        with tarfile.open(tar) as t:
            t.extractall(d, filter="data")
        log("data.tar.gz extracted")

    files = sorted(p.name for p in (d / "data").rglob("*.jsonl"))
    save_state("scifact", {"complete": True, "files": files})
    log(f"scifact COMPLETE - {len(files)} jsonl files: {', '.join(files)}")


# --------------------------------------------------------------------------- #
# faa-amt
# --------------------------------------------------------------------------- #

FAA_BASE = "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/"
# faa.gov answers 403 to an unrecognised agent, on PDFs as well as HTML
FAA_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
FAA_FILES = {
    "FAA-H-8083-30B_AMT_General.pdf": "amtg_handbook.pdf",
    "FAA-H-8083-31B_AMT_Airframe.pdf": (
        "FAA-H-8083-31B_Aviation_Maintenance_Technician_Handbook.pdf"
    ),
    "FAA-H-8083-32B_AMT_Powerplant.pdf": "amt_powerplant_handbook.pdf",
}


def cmd_faa_amt(args: argparse.Namespace) -> None:
    d = OUT / "faa-amt" / "pdf"
    done = set(load_state("faa-amt").get("done", []))
    for local, remote in FAA_FILES.items():
        dest = d / local
        new, size = download(FAA_BASE + remote, dest, agent=FAA_UA)
        if size < 100_000 or dest.read_bytes()[:4] != b"%PDF":
            log(f"FAILED {local}: not a PDF ({size} bytes) - left on disk for inspection")
            continue
        done.add(local)
        log(f"{'downloaded' if new else 'present'} {local}: {size / 1e6:.1f} MB")
    save_state("faa-amt", {"done": sorted(done), "complete": len(done) == len(FAA_FILES)})
    log(f"faa-amt: {len(done)}/{len(FAA_FILES)} handbooks on disk")


# --------------------------------------------------------------------------- #
# army-tm
# --------------------------------------------------------------------------- #

LM_INDEX = "https://www.liberatedmanuals.com/all.mpl"
LM_PDF = "https://www.liberatedmanuals.com/{}.pdf"
# operator / unit / direct-support maintenance tails; a P anywhere is a parts list
TM_TAILS = ("10", "12", "13", "14", "20", "23", "24", "34")


def army_targets() -> list[str]:
    """The 1,766 operator/maintenance manuals, reproduced from the index exactly as
    R14_corpus_scout.md's verification commands do."""
    html = fetch_bytes(LM_INDEX).decode("latin-1")
    ids = sorted({m.group(1) for m in re.finditer(r"(?i)HREF=/([A-Za-z0-9._-]+)\.pdf", html)})
    tail = re.compile(r"-(" + "|".join(TM_TAILS) + r")(-[0-9]+)?$")
    return [i for i in ids if tail.search(i) and "P" not in i]


def cmd_army_tm(args: argparse.Namespace) -> None:
    d = OUT / "army-tm" / "pdf"
    d.mkdir(parents=True, exist_ok=True)
    state = load_state("army-tm")

    targets = state.get("targets") or army_targets()
    failed = dict(state.get("failed", {}))
    # whatever the checkpoint says, a PDF on disk is done
    done = set(state.get("done", [])) | {p.stem for p in d.glob("*.pdf") if p.stat().st_size > 0}
    save_state("army-tm", {"targets": targets, "done": sorted(done), "failed": failed})
    log(f"army-tm: {len(targets)} targets, {len(done)} already on disk")

    while True:
        pending = [t for t in targets if t not in done and failed.get(t, 0) < 3]
        if not pending:
            save_state(
                "army-tm",
                {
                    "targets": targets,
                    "done": sorted(done),
                    "failed": failed,
                    "complete": True,
                },
            )
            log(f"army-tm COMPLETE - {len(done)}/{len(targets)}, {len(failed)} permanently failed")
            return

        fetched, blocked, refusals = 0, False, 0
        for tid in pending[: args.per_day]:
            try:
                blob = fetch_bytes(LM_PDF.format(tid), timeout=args.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                failed[tid] = failed.get(tid, 0) + 1
                log(f"  ERROR {tid}: {type(e).__name__}: {str(e)[:90]}")
                time.sleep(args.delay)
                continue
            if blob[:4] != b"%PDF":
                # the mirror serves a short HTML notice instead of a PDF both for
                # transient throttling and for the spent per-IP daily allowance.
                # Only a run of consecutive refusals means the daily allowance;
                # a single one is a hiccup and backs off briefly.
                refusals += 1
                log(f"  refused {tid} ({len(blob)} non-PDF bytes), refusal {refusals}")
                if refusals >= args.refusals_before_backoff:
                    blocked = True
                    break
                time.sleep(args.backoff)
                continue
            refusals = 0
            (d / f"{tid}.pdf").write_bytes(blob)
            done.add(tid)
            fetched += 1
            log(f"  {tid}.pdf {len(blob) / 1e6:.2f} MB  ({len(done)}/{len(targets)})")
            time.sleep(args.delay)

        save_state("army-tm", {"targets": targets, "done": sorted(done), "failed": failed})
        if args.once:
            log(f"army-tm batch done - {fetched} fetched, --once given, exiting")
            return
        wait = args.day_seconds if (blocked or fetched >= args.per_day) else args.delay
        log(f"army-tm sleeping {wait}s ({len(done)}/{len(targets)} done)")
        time.sleep(wait)


# --------------------------------------------------------------------------- #
# edgar-restricted
# --------------------------------------------------------------------------- #

EDGAR_SHARDS = ("train", "validate", "test")
EDGAR_URL = "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/2020/{}.jsonl"
SP500_CSV = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
)
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
EDGAR_FLOOR = 1000  # below this the slice is reported, never silently relaxed


def sp500_excluded_ciks(d: Path) -> tuple[set[int], dict]:
    """CIKs of every company that was an S&P 500 constituent at any point in
    1999-2019 - FinQA's source population."""
    csv_path = d / "sp500_historical_components.csv"
    if not csv_path.exists():
        csv_path.write_bytes(fetch_bytes(SP500_CSV))
    tickers: set[str] = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if "1999-01-01" <= row["date"] <= "2019-12-31":
                # constituents carry a delisting suffix, e.g. AAL-199702
                tickers |= {t.split("-")[0].upper() for t in row["tickers"].split(",") if t}

    map_path = d / "sec_company_tickers.json"
    if not map_path.exists():
        map_path.write_bytes(fetch_bytes(SEC_TICKERS))
    by_ticker = {
        v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(map_path.read_text()).values()
    }

    resolved = {t for t in tickers if t in by_ticker}
    return {by_ticker[t] for t in resolved}, {
        "sp500_tickers_1999_2019": len(tickers),
        "tickers_resolved_to_cik": len(resolved),
        "tickers_unresolved": len(tickers) - len(resolved),
    }


def _write_part(rows: list[dict], d: Path, shard: str, index: int) -> Path:
    import polars as pl

    p = d / f"_{shard}_part{index:03d}.parquet"
    pl.DataFrame(rows).write_parquet(p)
    return p


def filter_shard(raw: Path, out: Path, excluded: set[int], shard: str) -> dict:
    """Stream the shard, keep filings passing BOTH admitted clauses, write parquet."""
    import polars as pl

    counts = {"raw": 0, "dropped_year_lt_2020": 0, "dropped_sp500_filer": 0, "kept": 0}
    batch: list[dict] = []
    parts: list[Path] = []
    with raw.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts["raw"] += 1
            if int(row.get("year", 0)) < 2020:
                counts["dropped_year_lt_2020"] += 1
                continue
            if int(row.get("cik", -1)) in excluded:
                counts["dropped_sp500_filer"] += 1
                continue
            counts["kept"] += 1
            batch.append(row)
            if len(batch) >= 500:
                parts.append(_write_part(batch, out.parent, shard, len(parts)))
                batch = []
    if batch:
        parts.append(_write_part(batch, out.parent, shard, len(parts)))
    if parts:
        pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed").write_parquet(out)
        for p in parts:
            p.unlink()
    return counts


def cmd_edgar_restricted(args: argparse.Namespace) -> None:
    import polars as pl

    d = OUT / "edgar-restricted"
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "filings").mkdir(parents=True, exist_ok=True)

    excluded, meta = sp500_excluded_ciks(d)
    log(f"S&P 500 exclusion list: {meta}, distinct CIKs {len(excluded)}")

    counts: dict = {"excluded_cik_count": len(excluded), **meta, "shards": {}}
    for shard in EDGAR_SHARDS:
        raw = d / "raw" / f"{shard}.jsonl"
        new, size = download(EDGAR_URL.format(shard), raw)
        log(f"{shard}.jsonl {'downloaded' if new else 'present'}: {size / 1e6:.0f} MB")

        out = d / "filings" / f"{shard}.parquet"
        if out.exists():
            n = pl.scan_parquet(out).select(pl.len()).collect().item()
            counts["shards"][shard] = {"kept": n, "cached": True}
            log(f"{shard}: already filtered, {n} filings kept")
            continue

        counts["shards"][shard] = filter_shard(raw, out, excluded, shard)
        log(f"{shard}: {counts['shards'][shard]}")
        (d / "_counts.json").write_text(json.dumps(counts, indent=1))

    total = sum(s.get("kept", 0) for s in counts["shards"].values())
    counts["total_kept"] = total
    (d / "_counts.json").write_text(json.dumps(counts, indent=1))
    save_state("edgar-restricted", {"complete": True, "total_kept": total})
    log(f"edgar-restricted TOTAL KEPT: {total} filings")

    if total < EDGAR_FLOOR:
        log(
            "STOP: the admitted joint slice (non-S&P-500 AND year >= 2020) is below the "
            f"~{EDGAR_FLOOR}-filing floor at {total}. Neither restriction may be relaxed here - "
            "the fallback is the author's call. Measured counts are in _counts.json."
        )
        sys.exit(2)


# --------------------------------------------------------------------------- #

COMMANDS = {
    "edgar-restricted": cmd_edgar_restricted,
    "scifact": cmd_scifact,
    "army-tm": cmd_army_tm,
    "faa-amt": cmd_faa_amt,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the corpora and their licences")
    for name in COMMANDS:
        p = sub.add_parser(name, help=SPECS[name]["title"])
        p.add_argument("--sidecar-only", action="store_true", help="write the sidecar and stop")
    army = sub.choices["army-tm"]
    army.add_argument("--per-day", type=int, default=100, help="mirror allowance per IP per day")
    army.add_argument("--delay", type=float, default=8.0, help="seconds between manuals")
    army.add_argument("--day-seconds", type=int, default=86_400, help="wait for the daily reset")
    army.add_argument("--backoff", type=float, default=120.0, help="pause after one refusal")
    army.add_argument(
        "--refusals-before-backoff",
        type=int,
        default=5,
        help="consecutive non-PDF responses that mean the daily allowance is spent",
    )
    army.add_argument("--timeout", type=int, default=300)
    army.add_argument("--once", action="store_true", help="one batch, then exit")

    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "list":
        for name, spec in SPECS.items():
            print(f"{name:18s} {spec['title']}")
            print(f"{'':18s} licence - {spec['licence']}")
        return

    log(f"=== {args.cmd} - {SPECS[args.cmd]['title']}")
    log(f"sidecar -> {write_sidecar(args.cmd).name}")
    if args.sidecar_only:
        return
    COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
