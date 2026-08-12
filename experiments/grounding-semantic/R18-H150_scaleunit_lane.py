"""R18-H150 SCALE/UNIT VERIFICATION LANE - build + verify only, CPU, no GPU.

Builds `R18-H150_scaleunit_lane.parquet`: BARE-claim minimal pairs over public
TabFact-train and FEVEROUS-train tables, seed 1150, over documents content-
disjoint from the R17-H143 eval set.  Registered under "R18-H150 CONVERGENCE ARM".

Construction is the R17-H146 misbind lane's machinery with the corrupted token
moved from the NUMERAL to the SCALE/UNIT word:

  positive        the value the table gives for (row_key, column), verbatim in
                  the serialized chunk, carrying the unit the EVIDENCE states
  negative twin   the byte-identical claim with ONE token replaced - a different
                  unit of the SAME dimension.  Digits never move

THE ANTI-SHORTCUT RULE (H148 discipline), and why this lane differs from H145
-----------------------------------------------------------------------------
R17-H145's lane required the negative's unit surface to be ABSENT from the chunk
and the positive's to be PRESENT.  A model then reads the label off a literal
string test - "is the claimed unit anywhere in the evidence?" - which is the
H148 failure mode (a family solvable by matching the claim string against an
adjacent string in the chunk).  Two constructions defeat it here:

  1. SURFACE-DISJOINT CLAIMS (carries the lane).  The evidence writes the unit as
     an ABBREVIATION (`elevation (m)`, `12 km`, `(kg)`, `%`) and the claim writes
     it SPELLED OUT (`metres`, `kilometres`, `kilograms`, `percent`).  Both the
     positive's and the negative's unit phrase are required to be absent from the
     chunk - not as a substring, and not as any of their content tokens.  The
     literal-presence test is then uninformative on BOTH legs (measured: 0.0 /
     0.0), and the pair can only be settled by mapping the abbreviation the
     evidence carries onto the word the claim uses, then binding it to the named
     column
  2. IN-CHUNK DISTRACTOR (the H146-purest form, built where supply allows).  The
     corrupted unit is itself attested in the chunk, bound to a DIFFERENT column
     of the same table, so even the abbreviation-level presence test fails and
     the binding must be read

Every pair satisfies (1); the subset that also satisfies (2) is flagged in
`distractor_in_chunk` and counted per family.

SUPPLY, censused before build (see the manifest's `census` block)
-----------------------------------------------------------------
`scale_word` (thousand/million/billion) and `pct_pp` (percent vs percentage
point) are registered families whose supply the sources do not carry:

  * a magnitude word is written the SAME way in evidence and claim
    (`revenue (in millions)` -> "... is 1,234 million"), so rule (1) cannot be
    satisfied; the only escape is an abbreviated magnitude header (`(m)`, `(bn)`,
    `(k)`), and only the `(m)` = millions form occurs at any volume - which
    leaves the family single-direction and therefore word-marginal leaking
  * cross-column magnitude distractors, the rule (2) escape, occur in 0 tables
    for million/billion and million/thousand
  * "percentage point(s)" occurs in a handful of corpus tables and never as a
    column a positive could be built from (R17-H145 measured the same thing)

Both are attempted at build time and reported at their honest count rather than
padded.  `MIN_FAMILY_PAIRS` drops any word-pair too small to measure - which is
what removes the residual `million<->billion` (52 pairs) and `million<->thousand`
(32 pairs) once value-surface stratification runs, leaving `unit_swap` alone.

VALUE-SURFACE STRATIFICATION
----------------------------
A 4-digit numeral is plausible in metres and absurd in kilometres, so if a
family's two directions carry different numerals the claim alone settles the
pair.  Both directions are therefore matched bucket for bucket on (digit count,
decimal point).  It costs 41% of the offered pairs (5,358 -> 3,194) and is kept:
without it the digit-count AUROC separating a family's two directions reaches
1.0, and with digit-count matching alone the decimal channel reaches 0.94 and
the within-pair bar breaks at 0.6875.

FOLD PACKING
------------
With a linear probe the within-pair read is forced to exactly min(a,b)/(a+b)
when a fold holds `a` pairs of one direction and `b` of the other - the training
complement carries the mirrored imbalance, so an unbalanced fold reads BELOW
chance however clean the data is.  Documents here carry up to 32 pairs, so folds
are packed greedily on PAIR counts with both directions of a family packed
jointly.  The residual below-chance spread that survives is that artifact, not a
leak; it is reported as `worst_two_sided_deviation_report_only`.

Discipline carried from the banked builds:
  * corpus loaders, serializers, numeral canonicalization, the CONTENT-based
    eval-set exclusion and the AUROC helper come from `R17-H144_pairs.py`
  * the unit vocabulary, header/cell unit detection and the hub-family balance
    come from `R17-H145_scaleunit.py`, extended ONLY with units the census
    attests (frequency, digital storage, megawatts, knots, engine displacement)
  * converged liblinear probe at tol 1e-7, never default lbfgs (H144 finding ii)
  * direction-stratified document-disjoint folds (H145 finding b)
  * 50/50 direction balance inside every word-pair family, which is what holds
    the claim-length and word-marginal channels at chance

Run:  uv run python experiments/grounding-semantic/R18-H150_scaleunit_lane.py
"""

import collections
import importlib.util
import json
import math
import os
import pathlib
import random
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT = HERE / "R18-H150_scaleunit_lane.parquet"
MANIFEST = HERE / "R18-H150_scaleunit_lane_manifest.json"
VERIFY_REPORT = HERE / "R18-H150_scaleunit_lane_verify.json"
CLEAN_MIX = ROOT / "tmp" / "R14_E6_mix.parquet"
H146_LANE = HERE / "R17-H146_lane.parquet"
H145_LANE = HERE / "R17-H145_scaleunit.parquet"
H144_PAIRS = HERE / "R17-H144_pairs.parquet"

SEED = 1150
TARGET_PAIRS = 10_000
DOC_CAP_LADDER = (2, 3, 4, 5, 6, 8, 12, 16, 24, 32)
BODY_ROWS = 6
MIN_FAMILY_PAIRS = 100   # 32 leaves SE 0.088 - the 0.60 bar is then 1.1 SE away
N_FOLDS = 5
AUDIT_N = 500
TAG = "quant_scale_unit"

WIN, STRIDE = 1500, 750          # serving windowing, for the census
BASELINE_MULTIWINDOW = 0.201     # clean mix, recorded in the campaign log

FORM_WEIGHTS = {"row_prose": 30, "narrative": 25, "keyvalue": 20,
                "markdown": 10, "json_records": 10, "pipe": 5}

CLAIM_TEMPLATES = [
    "The {col} of {key} is {val} {unit}.",
    "For {key}, the {col} is {val} {unit}.",
    "{key} has a {col} of {val} {unit}.",
    "The table lists the {col} of {key} as {val} {unit}.",
    "The {col} recorded for {key} is {val} {unit}.",
    "According to the record, the {col} of {key} is {val} {unit}.",
]

_spec = importlib.util.spec_from_file_location("h144_pairs", HERE / "R17-H144_pairs.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


# --------------------------------------------------------------------------- #
# unit vocabulary
#   dim     - a swap is legal only INSIDE a dimension
#   phrase  - how the unit is written in the CLAIM (spelled out; rule 1)
#   surface - how the unit may appear in the EVIDENCE (presence / absence test)
# Only tokens the census attests in the sources are listed; nothing is invented.
# --------------------------------------------------------------------------- #
UNITS = {
    # magnitude words on a bare number (registered `scale_word` / `pct_pp`)
    "million":  ("magnitude", "million",  r"\bmillions?\b|\bmn\b|\bmil\b"),
    "thousand": ("magnitude", "thousand", r"\bthousands?\b|\b1000s\b|\b000s\b"),
    "billion":  ("magnitude", "billion",  r"\bbillions?\b|\bbn\b"),
    "percent":  ("percentish", "percent", r"%|\bpercent\b|\bpercentage\b|\bpct\b"),
    "pp":       ("percentish", "percentage points",
                 r"percentage\s+points?|\bp\.p\.|\bpp\b"),
    # length
    "m":    ("length", "metres",      r"\bm\b|\bmet(?:re|er)s?\b"),
    "km":   ("length", "kilometres",  r"\bkm\b|\bkilomet(?:re|er)s?\b"),
    "cm":   ("length", "centimetres", r"\bcm\b|\bcentimet(?:re|er)s?\b"),
    "mm":   ("length", "millimetres", r"\bmm\b|\bmillimet(?:re|er)s?\b"),
    "ft":   ("length", "feet",        r"\bft\b|\bfeet\b|\bfoot\b"),
    "mile": ("length", "miles",       r"\bmiles?\b|\bmi\b"),
    # mass
    "kg":    ("mass", "kilograms", r"\bkg\b|\bkilograms?\b"),
    "g":     ("mass", "grams",     r"\bg\b|\bgrams?\b"),
    "lb":    ("mass", "pounds",    r"\blbs?\b|\bpounds?\b"),
    "tonne": ("mass", "tonnes",    r"\btonnes?\b|\bmetric tons?\b"),
    "ton":   ("mass", "tons",      r"\btons?\b|\bshort tons?\b"),
    # area
    "km2":  ("area", "square kilometres",
             r"\bkm\s*(?:2|²|square)\b|\bsq\s*km\b|\bsqkm\b|square kilomet(?:re|er)s?"),
    "m2":   ("area", "square metres", r"\bm\s*(?:2|²|square)\b|square met(?:re|er)s?"),
    "ha":   ("area", "hectares",      r"\bha\b|\bhectares?\b"),
    "acre": ("area", "acres",         r"\bacres?\b"),
    "sqmi": ("area", "square miles",  r"\bsq\s*mi\b|\bsqmi\b|square miles?"),
    # speed
    "mph": ("speed", "miles per hour",      r"\bmph\b|miles per hour"),
    "kmh": ("speed", "kilometres per hour", r"\bkm\s*/\s*h\b|\bkph\b|kilomet(?:re|er)s per hour"),
    "kt":  ("speed", "knots",               r"\bkts?\b|\bknots?\b"),
    # power  (census: kw 279 cells, mw 51, hp 7)
    "kw": ("power", "kilowatts",  r"\bkw\b|\bkilowatts?\b"),
    "mw": ("power", "megawatts",  r"\bmw\b|\bmegawatts?\b"),
    "hp": ("power", "horsepower", r"\bhp\b|\bhorsepower\b|\bbhp\b"),
    # frequency  (census: mhz 285, ghz 132, khz 91)
    "mhz": ("frequency", "megahertz", r"\bmhz\b|\bmegahertz\b"),
    "ghz": ("frequency", "gigahertz", r"\bghz\b|\bgigahertz\b"),
    "khz": ("frequency", "kilohertz", r"\bkhz\b|\bkilohertz\b"),
    # digital storage  (census: mb 177, kb 93, gb 89)
    "kb": ("storage", "kilobytes", r"\bkb\b|\bkib\b|\bkilobytes?\b"),
    "mb": ("storage", "megabytes", r"\bmb\b|\bmib\b|\bmegabytes?\b"),
    "gb": ("storage", "gigabytes", r"\bgb\b|\bgib\b|\bgigabytes?\b"),
    # engine displacement / volume  (census: cc 144, l 6)
    "cc": ("volume", "cubic centimetres", r"\bcc\b|\bcm\s*3\b|cubic centimet(?:re|er)s?"),
    "l":  ("volume", "litres",            r"\bl\b|\blit(?:re|er)s?\b"),
}

# Registered family grouping.  `unit_swap` is the physical-dimension family;
# `scale_word` and `pct_pp` are the two registered families the sources gate.
FAMILY_OF_DIM = {"magnitude": "scale_word", "percentish": "pct_pp"}

DIM = {u: v[0] for u, v in UNITS.items()}
PHRASE = {u: v[1] for u, v in UNITS.items()}
SURFACE = {u: re.compile(v[2], re.IGNORECASE) for u, v in UNITS.items()}
ANY_UNIT = re.compile("|".join(f"(?:{v[2]})" for v in UNITS.values()), re.IGNORECASE)

# header annotations.  Longer / more specific patterns first: `m` and `m2`,
# `km` and `km2`, `m`(etres) and `m`(illions) share prefixes.
HDR_PATS = [
    (r"\(\s*km\s*(?:2|²|square|\^2)\s*\)|\bkm\s*2\b|\bsq\s*km\b|\bsqkm\b|square\s+kilomet(?:re|er)s?", "km2"),
    (r"\(\s*m\s*(?:2|²|square|\^2)\s*\)|\bm\s*2\b|square\s+met(?:re|er)s?", "m2"),
    (r"\(\s*sq\s*mi\s*\)|\bsqmi\b|square\s+miles?", "sqmi"),
    (r"\(\s*(?:in\s+)?millions?\s*(?:of\s+[\w\s]+?\s*)?\)|,?\s+in\s+millions?\b", "million"),
    (r"\(\s*(?:in\s+)?thousands?\s*(?:of\s+[\w\s]+?\s*)?\)|,?\s+in\s+thousands?\b|\bin\s+1000s\b", "thousand"),
    (r"\(\s*(?:in\s+)?billions?\s*(?:of\s+[\w\s]+?\s*)?\)|,?\s+in\s+billions?\b", "billion"),
    (r"percentage\s+points?", "pp"),
    (r"\(\s*(?:in\s+)?(?:km|kilomet(?:re|er)s?)\s*\)|\bin\s+kilomet(?:re|er)s\b", "km"),
    (r"\(\s*(?:in\s+)?(?:cm|centimet(?:re|er)s?)\s*\)", "cm"),
    (r"\(\s*(?:in\s+)?(?:mm|millimet(?:re|er)s?)\s*\)", "mm"),
    (r"\(\s*(?:in\s+)?(?:ft|feet|foot)\s*\)", "ft"),
    (r"\(\s*(?:in\s+)?(?:mi|miles?)\s*\)", "mile"),
    (r"\(\s*(?:in\s+)?(?:kg|kilograms?)\s*\)", "kg"),
    (r"\(\s*(?:in\s+)?(?:g|grams?)\s*\)", "g"),
    (r"\(\s*(?:in\s+)?(?:lb|lbs|pounds?)\s*\)", "lb"),
    (r"\(\s*(?:in\s+)?(?:tonnes?|metric\s+tons?)\s*\)", "tonne"),
    (r"\(\s*(?:in\s+)?(?:short\s+)?tons?\s*\)", "ton"),
    (r"\(\s*(?:in\s+)?(?:ha|hectares?)\s*\)", "ha"),
    (r"\(\s*(?:in\s+)?acres?\s*\)", "acre"),
    (r"\(\s*(?:in\s+)?(?:kw|kilowatts?)\s*\)", "kw"),
    (r"\(\s*(?:in\s+)?(?:mw|megawatts?)\s*\)", "mw"),
    (r"\(\s*(?:in\s+)?(?:hp|bhp|horsepower)\s*\)", "hp"),
    (r"\(\s*(?:in\s+)?mhz\s*\)|\bmhz\b", "mhz"),
    (r"\(\s*(?:in\s+)?ghz\s*\)|\bghz\b", "ghz"),
    (r"\(\s*(?:in\s+)?khz\s*\)|\bkhz\b", "khz"),
    (r"\(\s*(?:in\s+)?(?:kb|kib|kilobytes?)\s*\)", "kb"),
    (r"\(\s*(?:in\s+)?(?:mb|mib|megabytes?)\s*\)", "mb"),
    (r"\(\s*(?:in\s+)?(?:gb|gib|gigabytes?)\s*\)", "gb"),
    (r"\(\s*(?:in\s+)?(?:cc|cm\s*3)\s*\)", "cc"),
    (r"\(\s*(?:in\s+)?(?:l|lit(?:re|er)s?)\s*\)", "l"),
    (r"\(\s*(?:in\s+)?mph\s*\)|\bmph\b", "mph"),
    (r"\(\s*(?:in\s+)?km\s*/\s*h\s*\)|\bkm\s*/\s*h\b|\bkph\b", "kmh"),
    (r"\(\s*(?:in\s+)?(?:kt|kts|knots?)\s*\)", "kt"),
    (r"\(\s*%\s*\)|(?<![a-z0-9])%|\bpercentage\b|\bpercent\b|\bpct\b", "percent"),
    # the ONE abbreviated magnitude header the corpus carries at volume: `(m)`
    # on a money / audience column.  Gated by MAGNITUDE_LEXICON below.
    (r"\(\s*m\s*\)", "million"),
    (r"\(\s*(?:bn|b)\s*\)", "billion"),
    (r"\(\s*(?:in\s+)?(?:m|met(?:re|er)s?)\s*\)|\bin\s+met(?:re|er)s\b", "m"),
]
HDR_PATS = [(re.compile(p, re.IGNORECASE), u) for p, u in HDR_PATS]

NUMPART = r"(-?[\d,]+(?:\.\d+)?)"
CELL_PATS = [
    (rf"^{NUMPART}\s*(?:%|percent)$", "percent"),
    (rf"^{NUMPART}\s*percentage\s+points?$", "pp"),
    (rf"^{NUMPART}\s*millions?$", "million"),
    (rf"^{NUMPART}\s*billions?$", "billion"),
    (rf"^{NUMPART}\s*thousands?$", "thousand"),
    (rf"^{NUMPART}\s*km\s*(?:2|²|square)$", "km2"),
    (rf"^{NUMPART}\s*m\s*(?:2|²|square)$", "m2"),
    (rf"^{NUMPART}\s*(?:sq\s*mi|sqmi)$", "sqmi"),
    (rf"^{NUMPART}\s*(?:km|kilomet(?:re|er)s?)$", "km"),
    (rf"^{NUMPART}\s*(?:m|met(?:re|er)s?)$", "m"),
    (rf"^{NUMPART}\s*(?:cm|centimet(?:re|er)s?)$", "cm"),
    (rf"^{NUMPART}\s*(?:mm|millimet(?:re|er)s?)$", "mm"),
    (rf"^{NUMPART}\s*(?:ft|feet)$", "ft"),
    (rf"^{NUMPART}\s*(?:mi|miles?)$", "mile"),
    (rf"^{NUMPART}\s*(?:kg|kilograms?)$", "kg"),
    (rf"^{NUMPART}\s*(?:g|grams?)$", "g"),
    (rf"^{NUMPART}\s*(?:lbs?|pounds?)$", "lb"),
    (rf"^{NUMPART}\s*(?:t|tonnes?)$", "tonne"),
    (rf"^{NUMPART}\s*tons?$", "ton"),
    (rf"^{NUMPART}\s*(?:ha|hectares?)$", "ha"),
    (rf"^{NUMPART}\s*acres?$", "acre"),
    (rf"^{NUMPART}\s*(?:kw|kilowatts?)$", "kw"),
    (rf"^{NUMPART}\s*(?:mw|megawatts?)$", "mw"),
    (rf"^{NUMPART}\s*(?:hp|bhp|horsepower)$", "hp"),
    (rf"^{NUMPART}\s*mhz$", "mhz"),
    (rf"^{NUMPART}\s*ghz$", "ghz"),
    (rf"^{NUMPART}\s*khz$", "khz"),
    (rf"^{NUMPART}\s*(?:kb|kib)$", "kb"),
    (rf"^{NUMPART}\s*(?:mb|mib)$", "mb"),
    (rf"^{NUMPART}\s*(?:gb|gib)$", "gb"),
    (rf"^{NUMPART}\s*cc$", "cc"),
    (rf"^{NUMPART}\s*(?:l|lit(?:re|er)s?)$", "l"),
    (rf"^{NUMPART}\s*mph$", "mph"),
    (rf"^{NUMPART}\s*km\s*/\s*h$", "kmh"),
    (rf"^{NUMPART}\s*(?:kt|kts|knots?)$", "kt"),
]
CELL_PATS = [(re.compile(p, re.IGNORECASE), u) for p, u in CELL_PATS]

# Ambiguous abbreviations: `m` is metres or millions, `g` is grams or a playing
# position, `l` is litres or losses, `t` is tonnes or ties, `cc` is displacement.
# For those the COLUMN must name the right kind of quantity.
COL_LEXICON = {
    "m": re.compile(r"length|height|elevation|depth|distance|width|altitude|"
                    r"prominence|isolation|span|diameter|radius|wingspan|draught|"
                    r"draft|beam|thickness|circumference|perimeter|gauge|clearance|"
                    r"range|rise|drop|size", re.IGNORECASE),
    "g": re.compile(r"weight|mass|dose|payload|content|load", re.IGNORECASE),
    "l": re.compile(r"volume|capacity|displacement|fuel|tank", re.IGNORECASE),
    "cc": re.compile(r"displacement|engine|capacity|volume", re.IGNORECASE),
    "million": re.compile(r"viewer|audience|revenue|sales|gross|income|profit|"
                          r"budget|box office|population|users|subscriber|gdp|"
                          r"assets|funding|cost|value|worth|earnings|turnover|"
                          r"receipts|takings|donation|spend", re.IGNORECASE),
    "billion": re.compile(r"viewer|audience|revenue|sales|gross|income|profit|"
                          r"budget|box office|population|users|subscriber|gdp|"
                          r"assets|funding|cost|value|worth|earnings|turnover",
                          re.IGNORECASE),
}
PUNCT = re.compile(r"^[\s,;:.\-/()]+|[\s,;:.\-/()]+$")
WORD = re.compile(r"[a-z]+")
STOPWORD = {"per", "of", "in", "the", "and", "a", "to"}


def phrase_tokens(p):
    return {w for w in WORD.findall(p.lower()) if len(w) >= 2 and w not in STOPWORD}


PHRASE_TOKENS = {u: phrase_tokens(PHRASE[u]) for u in UNITS}


def clean_name(s):
    out = P.clean(PUNCT.sub("", P.clean(s)))
    if out.count("(") != out.count(")"):
        out = P.clean(PUNCT.sub("", out.replace("(", " ").replace(")", " ")))
    return out


def header_units(hdr):
    """Every unit reading a header admits, most specific first.

    `(m)` is metres on `elevation (m)` and millions on `gross (m)`; the reading is
    chosen downstream by the column lexicon, so both are offered here."""
    out = []
    for rx, u in HDR_PATS:
        m = rx.search(hdr)
        if not m:
            continue
        out.append((u, clean_name(hdr[:m.start()] + " " + hdr[m.end():])))
    return out


def header_unit(hdr):
    """The reading the build accepts - the first that clears its lexicon gate."""
    for u, name in header_units(hdr):
        if len(name) < 3 or ANY_UNIT.search(name):
            continue
        if u in COL_LEXICON and not COL_LEXICON[u].search(name):
            continue
        if u in ("million", "billion") and COL_LEXICON["m"].search(name):
            continue
        return u, name
    return None, None


def cell_unit(cell):
    for rx, u in CELL_PATS:
        m = rx.match(cell)
        if m:
            return u, m.group(1)
    return None, None


def numeric_cells(body, ci):
    return [(ri, r[ci].strip()) for ri, r in enumerate(body)
            if P.as_num(r[ci].strip()) is not None]


def table_candidates(t):
    """Every (column, row) of one table that carries a genuine unit, plus the set
    of units the table attests anywhere (the in-chunk distractor supply)."""
    lab = t["lab_ci"]
    out, attested = [], set()
    for ci, hdr in enumerate(t["hdr"]):
        if not P.good_header(hdr):
            continue
        u, name = header_unit(hdr)
        if u is not None:
            attested.add(u)
        if ci == lab:
            continue
        if u is not None:
            cells = numeric_cells(t["body"], ci)
            if not cells:
                continue
            for ri, val in cells:
                out.append({"ci": ci, "ri": ri, "unit": u, "col": name,
                            "val": val, "cell": val, "carrier": "header"})
            continue
        name = clean_name(hdr)
        if len(name) < 3 or ANY_UNIT.search(name):
            continue
        for ri, r in enumerate(t["body"]):
            cu, num = cell_unit(r[ci].strip())
            if cu is None:
                continue
            attested.add(cu)
            if cu in COL_LEXICON and not COL_LEXICON[cu].search(name):
                continue
            out.append({"ci": ci, "ri": ri, "unit": cu, "col": name,
                        "val": num, "cell": r[ci].strip(), "carrier": "cell"})
    return out, attested


def make_chunk(t, ci, ri, cell, rng, attempts=5):
    """Serialize a <=6-row window holding the target row; narrow it until the
    target cell survives the 1,500-character serving cap."""
    for k in range(attempts):
        width = BODY_ROWS - 1 if k < attempts - 2 else (3 if k == attempts - 2 else 1)
        keep = {ri}
        others = [r for r in range(len(t["body"])) if r != ri]
        rng.shuffle(others)
        keep |= set(others[:width])
        body = [t["body"][r] for r in sorted(keep)]
        chunk = P.serialize(t["form"], t["caption"], t["hdr"], body, t["lab_ci"], (ci,))
        if cell in chunk:
            return chunk
    return None


def distinguishing_tokens(u, v):
    """Tokens that tell the twins apart.

    `square kilometres` vs `square miles` differ only on the head noun - `square`
    sits in BOTH claims, so its presence in the chunk carries no label signal and
    only the symmetric difference has to be absent."""
    return PHRASE_TOKENS[u] ^ PHRASE_TOKENS[v]


def phrase_absent(unit, chunk_low):
    return PHRASE[unit].lower() not in chunk_low


def surface_absent(u, v, chunk_low, chunk_toks):
    """Rule 1, evaluated for a twin pair: neither unit phrase may be readable in
    the chunk, whole or by any token that distinguishes the two."""
    return (phrase_absent(u, chunk_low) and phrase_absent(v, chunk_low)
            and not (distinguishing_tokens(u, v) & chunk_toks))


def build_claim(tpl, col, key, val, unit):
    return tpl.format(col=col, key=key, val=val, unit=PHRASE[unit])


def attested_in_chunk(p, unit):
    """Is `unit` itself readable in the chunk, bound to some OTHER column?

    When it is, the abbreviation-level presence test fails too and the pair is
    the H146-purest form: only the binding settles it."""
    if SURFACE[unit].search(p["chunk"]):
        return True
    t = p["tab"]
    for ci, hdr in enumerate(t["hdr"]):
        if ci == p["ci"] or ci == t["lab_ci"]:
            continue
        u, _ = header_unit(hdr)
        if u == unit and P.clean(hdr) in p["chunk"]:
            return True
    return False


# --------------------------------------------------------------------------- #
# hub families: inside a dimension the most abundant word is the hub and every
# other word forms one family with it, filled 50/50 in both directions.  50/50
# implies exact word-level label-marginal balance, so no word marginally predicts
# a label and no claim-length channel opens.
# --------------------------------------------------------------------------- #
STRATIFY_VALUE_SURFACE = os.environ.get("H150_STRATIFY", "1")   # 1 full, 2 digits, 0 off


def value_bucket(val):
    """The numeral's surface class - digit count and decimal point.

    Both directions of a family are matched on this key so the numeral cannot
    tell the probe which unit is the true one (a 4-digit value is plausible in
    metres and absurd in kilometres)."""
    if STRATIFY_VALUE_SURFACE == "0":
        return 0
    if STRATIFY_VALUE_SURFACE == "2":
        return (P.digits(val),)
    return (P.digits(val), "." in val)


def build_families(positives, rng, budget):
    by_dim = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in positives:
        by_dim[DIM[p["unit"]]][p["unit"]].append(p)

    pairs, report = [], {}
    for dim in sorted(by_dim):
        pool = {w: lst[:] for w, lst in by_dim[dim].items()}
        supply0 = {w: len(lst) for w, lst in pool.items()}
        if len(pool) < 2:
            report[dim] = {"hub": None, "families": {},
                           "note": f"single word ({list(pool)}), no in-dimension partner"}
            continue
        fams, hubs = {}, []
        # Repeated hub passes: once the pass hub is spent its scarce partners
        # would otherwise strand, so the remaining pool re-hubs and runs again.
        while len(pool) >= 2:
            hub = max(pool, key=lambda w: len(pool[w]))
            hub_pool = pool.pop(hub)
            rng.shuffle(hub_pool)
            hubs.append(hub)
            took = False
            for v in sorted(pool, key=lambda w: -len(pool[w])):
                # a positive is usable in a family only if the PARTNER word is
                # also absent from its chunk as a claim surface (rule 1, both legs)
                side_v = [p for p in pool[v]
                          if surface_absent(v, hub, p["chunk_low"], p["chunk_toks"])]
                side_h = [p for p in hub_pool
                          if surface_absent(hub, v, p["chunk_low"], p["chunk_toks"])]
                # VALUE-SURFACE STRATIFICATION.  A 4-digit numeral is plausible in
                # metres and absurd in kilometres, so if the two directions carry
                # different value distributions a claim-only probe reads the
                # numeral-times-unit interaction and settles the pair without the
                # evidence (measured: within-pair accuracy 0.03-0.62 across
                # families before this).  Both directions are therefore matched
                # bucket for bucket on the numeral's surface.
                buck_v = collections.defaultdict(list)
                buck_h = collections.defaultdict(list)
                for p in side_v:
                    buck_v[value_bucket(p["val"])].append(p)
                for p in side_h:
                    buck_h[value_bucket(p["val"])].append(p)
                taken_v, taken_h = [], []
                for b in sorted(set(buck_v) & set(buck_h)):
                    rng.shuffle(buck_v[b])
                    rng.shuffle(buck_h[b])
                    kb = min(len(buck_v[b]), len(buck_h[b]))
                    taken_v += buck_v[b][:kb]
                    taken_h += buck_h[b][:kb]
                k = len(taken_v)
                fam = "<->".join(sorted((hub, v)))
                if 2 * k < MIN_FAMILY_PAIRS:
                    fams.setdefault(fam, {
                        "kept": 0, "offered": 2 * k,
                        "supply": {v: supply0[v], hub: supply0[hub]},
                        "note": f"below MIN_FAMILY_PAIRS={MIN_FAMILY_PAIRS} after "
                                "value-surface stratification"})
                    continue
                pairs += [(p, hub) for p in taken_v] + [(p, v) for p in taken_h]
                used_v, used_h = {id(p) for p in taken_v}, {id(p) for p in taken_h}
                pool[v] = [p for p in pool[v] if id(p) not in used_v]
                hub_pool = [p for p in hub_pool if id(p) not in used_h]
                fams[fam] = {"kept": 2 * k, "offered": 2 * k, "hub_pass": len(hubs),
                             "supply": {v: supply0[v], hub: supply0[hub]}}
                took = True
            pool = {w: lst for w, lst in pool.items() if lst}
            if not took:
                break
        report[dim] = {"hubs": hubs, "families": fams}

    rng.shuffle(pairs)
    if budget is not None and len(pairs) > budget:
        by_fam = collections.defaultdict(list)
        for p, neg in pairs:
            by_fam["<->".join(sorted((p["unit"], neg)))].append((p, neg))
        scale = budget / len(pairs)
        pairs = []
        for lst in by_fam.values():
            pairs += rebalance_family(lst, rng, int(len(lst) * scale))
    return pairs, report


def rebalance_family(lst, rng, budget=None):
    """Restore exact 50/50 direction balance AND value-bucket parity.

    Trimming drops whole matched bucket groups, never half of one, so the
    stratification survives every trim."""
    by_dir = collections.defaultdict(lambda: collections.defaultdict(list))
    for p, neg in lst:
        by_dir[p["unit"]][value_bucket(p["val"])].append((p, neg))
    if len(by_dir) != 2:
        return []
    a, b = sorted(by_dir)
    groups = []
    for bucket in sorted(set(by_dir[a]) & set(by_dir[b])):
        la, lb = by_dir[a][bucket], by_dir[b][bucket]
        rng.shuffle(la)
        rng.shuffle(lb)
        k = min(len(la), len(lb))
        groups.append(la[:k] + lb[:k])
    rng.shuffle(groups)
    out = []
    for g in groups:
        if budget is not None and len(out) + len(g) > budget and out:
            continue
        out += g
    return out


# --------------------------------------------------------------------------- #
# windows (serving presentation), for the combined-mix census
# --------------------------------------------------------------------------- #
def n_windows(n_chars):
    if n_chars <= WIN:
        return 1
    return 1 + math.ceil((n_chars - WIN) / STRIDE)


def window_stats(lengths):
    w = np.array([n_windows(int(n)) for n in lengths])
    return {"rows": int(w.size), "mean_windows": round(float(w.mean()), 4),
            "multi_window_rows": int((w > 1).sum()),
            "multi_window_share": round(float((w > 1).mean()), 4)}


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def jaccard(a, b):
    A, B = set(WORD.findall(a.lower())), set(WORD.findall(b.lower()))
    return len(A & B) / max(len(A | B), 1)


def verify(df, rng, by_doc):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = {}
    labels = df["label"].to_list()
    claims = df["claim"].to_list()

    # --- BAR 1: claim-only probe, document-disjoint folds stratified on the
    # document's (family, positive word) so no fold's complement carries only the
    # mirrored direction (R17-H145 finding b)
    doc_key = {d: k for d, k in df.filter(pl.col("label") == 1)
               .group_by("doc_id")
               .agg((pl.col("swap_family") + ":" + pl.col("correct_unit")).first())
               .iter_rows()}
    # Documents carry very different pair counts (doc cap 32), so round-robin on
    # DOCUMENTS leaves each fold's (family, direction) PAIR counts skewed - and a
    # fold's training complement then carries the mirrored skew, which drives the
    # within-pair read systematically BELOW chance (R17-H145 finding b, inverted).
    # Folds are therefore packed greedily on pair counts inside each stratum.
    # and the two DIRECTIONS of a family have to be packed jointly, not
    # independently: with a linear probe the fold's own direction imbalance
    # (a vs b pairs) forces within-pair accuracy to exactly min(a,b)/(a+b), so an
    # unbalanced fold reads below chance no matter how clean the data is.
    doc_n = dict(df.filter(pl.col("label") == 1).group_by("doc_id").len().iter_rows())
    by_family = collections.defaultdict(list)
    for d, k in doc_key.items():
        fam, word = k.rsplit(":", 1)
        by_family[fam].append((d, word))
    fold_of, size = {}, [0] * N_FOLDS
    for fam in sorted(by_family):
        docs = by_family[fam]
        rng.shuffle(docs)
        words = sorted({w for _, w in docs})
        imb = [0] * N_FOLDS            # load(words[0]) - load(words[1]) per fold
        for d, w in sorted(docs, key=lambda x: -doc_n[x[0]]):
            first = (w == words[0])
            f = min(range(N_FOLDS),
                    key=lambda j: ((imb[j] if first else -imb[j]), size[j], j))
            fold_of[d] = f
            imb[f] += doc_n[d] if first else -doc_n[d]
            size[f] += doc_n[d]
    folds = np.array([fold_of[d] for d in df["doc_id"].to_list()])
    score = np.zeros(len(df), dtype=float)
    idx = np.arange(len(df))
    for f in range(N_FOLDS):
        tr_i, te_i = idx[folds != f], idx[folds == f]
        if not len(te_i):
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        Xtr = vec.fit_transform([claims[j] for j in tr_i])
        Xte = vec.transform([claims[j] for j in te_i])
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(Xtr, [labels[j] for j in tr_i])
        score[te_i] = clf.decision_function(Xte)
    probe = P.auroc(labels, score)
    out["claim_only_tfidf_auroc"] = {
        "value": round(float(probe), 4), "bar": "< 0.55", "pass": bool(probe < 0.55),
        "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold, "
                   "direction-stratified, liblinear tol 1e-7",
        "documents": len(fold_of), "rows": len(df)}

    # --- BAR 2: within-pair claim-only accuracy, per word-pair family AND per
    # registered negative family
    scored = df.select(["pair_id", "label", "swap_family", "neg_family"]).with_columns(
        pl.Series("score", score))

    def within(col):
        acc = {}
        for key, sub in scored.group_by(col):
            piv = sub.pivot(on="label", index="pair_id", values="score",
                            aggregate_function="first").drop_nulls()
            if not len(piv):
                continue
            pos, neg = piv["1"].to_numpy(), piv["0"].to_numpy()
            a = float(((pos > neg) + 0.5 * (pos == neg)).mean())
            se = 0.5 / math.sqrt(len(piv))
            acc[key[0]] = {"acc": round(a, 4), "pairs": len(piv),
                           "se_at_chance": round(se, 4),
                           "z_vs_chance": round((a - 0.5) / se, 2)}
        return acc

    fam_acc, negfam_acc = within("swap_family"), within("neg_family")
    worst = max(v["acc"] for v in fam_acc.values())
    out["within_pair_claim_only_accuracy"] = {
        "per_swap_family": fam_acc, "per_neg_family": negfam_acc,
        "worst": round(worst, 4), "bar": "< 0.60", "pass": bool(worst < 0.60),
        "worst_two_sided_deviation_report_only":
            round(max(abs(v["acc"] - 0.5) for v in fam_acc.values()), 4)}

    # --- BAR 3: surface parity - length / tokens / numerals / passage overlap
    chunks = df["chunk"].to_list()
    feats = {
        "claim_char_length": [float(len(c)) for c in claims],
        "claim_token_count": [float(len(WORD.findall(c.lower()))) for c in claims],
        "claim_numeral_count": [float(P.digits(c)) for c in claims],
        "claim_chunk_overlap": [jaccard(c, k) for c, k in zip(claims, chunks)],
    }
    par = {k: round(float(P.auroc(labels, v)), 4) for k, v in feats.items()}
    dev = max(abs(v - 0.5) for v in par.values())
    out["surface_parity"] = {
        "auroc": par, "bar": "each in [0.45, 0.55]", "pass": bool(dev <= 0.05),
        "max_deviation": round(dev, 4),
        "pooled_within_pair_parity_flag_rate":
            round(float(df["surface_parity"].cast(pl.Float64).mean()), 4)}

    # --- BAR 4: positive verbatim-substring discipline.  The asserted numeral is
    # a table cell the chunk prints, on BOTH legs (the value never moves).
    cited = df["cited_value"].to_list()
    verbatim = [v in c for v, c in zip(cited, chunks)]
    canonical = [bool(P.canon_set(v) & P.present_numbers(c)) for v, c in zip(cited, chunks)]
    pos = df.filter(pl.col("label") == 1)
    pos_verbatim = [v in c for v, c in zip(pos["cited_value"].to_list(),
                                           pos["chunk"].to_list())]
    out["positive_verbatim_substring"] = {
        "positive_verbatim_rate": round(float(np.mean(pos_verbatim)), 6),
        "all_rows_verbatim_rate": round(float(np.mean(verbatim)), 6),
        "all_rows_canonical_rate": round(float(np.mean(canonical)), 6),
        "bar": "1.0 on the positive leg", "pass": bool(all(pos_verbatim))}

    # --- BAR 5: full-set minimal-pair integrity.  Every pair is exactly two rows
    # that agree on everything except the unit phrase, and substituting the unit
    # phrase in the positive reproduces the negative byte for byte.
    errs, checked = [], 0
    cols = ["pair_id", "label", "claim", "chunk", "doc_id", "column", "row_key",
            "cited_value", "correct_unit", "cited_unit", "swap_family"]
    g = df.select(cols).sort(["pair_id", "label"], descending=[False, True])
    it = g.iter_rows(named=True)
    buf = collections.defaultdict(list)
    for r in it:
        buf[r["pair_id"]].append(r)
    for pid, rows in buf.items():
        checked += 1
        why = None
        if len(rows) != 2:
            why = f"{len(rows)} rows in pair"
        else:
            a, b = rows
            if (a["label"], b["label"]) != (1, 0):
                why = "labels are not exactly {1, 0}"
            elif any(a[k] != b[k] for k in ("chunk", "doc_id", "column", "row_key",
                                            "cited_value", "swap_family")):
                why = "twins disagree on the evidence or the binding"
            elif a["correct_unit"] != b["correct_unit"]:
                why = "twins disagree on the correct unit"
            elif a["cited_unit"] != a["correct_unit"]:
                why = "positive does not cite the correct unit"
            elif b["cited_unit"] == b["correct_unit"]:
                why = "negative cites the correct unit"
            elif DIM[a["cited_unit"]] != DIM[b["cited_unit"]]:
                why = "swap crosses a dimension"
            elif a["claim"].replace(PHRASE[a["cited_unit"]],
                                    PHRASE[b["cited_unit"]]) != b["claim"]:
                why = "negative is not the positive with the unit phrase substituted"
        if why:
            errs.append({"pair_id": pid, "why": why})
    out["minimal_pair_integrity"] = {
        "pairs_checked": checked, "errors": len(errs), "bar": "0 errors over the FULL set",
        "pass": len(errs) == 0, "examples": errs[:5]}

    # --- BAR 6: dedupe + document-level disjointness
    dup_rows = len(df) - df.unique(subset=["claim", "chunk", "label"]).height
    per_doc = df.filter(pl.col("label") == 1).group_by("doc_id").len()["len"]
    out["dedupe_disjointness"] = {
        "duplicate_claim_chunk_label_rows": int(dup_rows),
        "documents": df["doc_id"].n_unique(),
        "distinct_claims": df["claim"].n_unique(),
        "distinct_chunks": df["chunk"].n_unique(),
        "pairs_per_document_mean": round(float(per_doc.mean()), 3),
        "pairs_per_document_max": int(per_doc.max()),
        "evalset_shared_documents": 0, "evalset_shared_chunks": 0,
        "bar": "0 duplicate rows, 0 eval-set documents",
        "pass": bool(dup_rows == 0)}

    # --- H148 anti-shortcut audit (reported, and the reason this lane is not
    # H145's): the literal presence of the CLAIMED unit phrase in the chunk must
    # be uninformative - 0.0 on both legs
    cited_u = df["cited_unit"].to_list()
    correct_u = df["correct_unit"].to_list()
    wrong_u = df["wrong_unit"].to_list()
    twin_u = [w if u == c else c for u, c, w in zip(cited_u, correct_u, wrong_u)]
    pres = np.array([
        1.0 if (PHRASE[u].lower() in ch.lower()
                or bool(distinguishing_tokens(u, tw) & set(WORD.findall(ch.lower()))))
        else 0.0
        for u, tw, ch in zip(cited_u, twin_u, chunks)])
    y = np.array(labels)
    out["claim_unit_literal_presence"] = {
        "positive_leg_rate": round(float(pres[y == 1].mean()), 6),
        "negative_leg_rate": round(float(pres[y == 0].mean()), 6),
        "auroc": round(float(P.auroc(labels, pres.tolist())), 4),
        "definition": "claim unit phrase, or any token distinguishing it from its "
                      "twin's unit phrase, readable in the chunk",
        "bar": "0.0 on both legs (H148 rule: no adjacent-string solution)",
        "pass": bool(pres.sum() == 0)}

    # --- value-surface parity ACROSS directions, per family.  If the two
    # directions carried different numerals the claim-only probe reads the
    # numeral-times-unit interaction; 0.5 means they do not.
    vsp = {}
    posdf = df.filter(pl.col("label") == 1)
    for key, sub in posdf.group_by("swap_family"):
        words = sorted(set(sub["correct_unit"].to_list()))
        if len(words) != 2:
            continue
        y2 = [1 if u == words[0] else 0 for u in sub["correct_unit"].to_list()]
        vals = sub["cited_value"].to_list()
        vsp[key[0]] = {
            "digit_count_auroc": round(P.auroc(y2, [P.digits(v) for v in vals]), 4),
            "magnitude_auroc": round(P.auroc(y2, [P.as_num(v) or 0.0 for v in vals]), 4),
            "decimal_auroc": round(P.auroc(y2, [1.0 if "." in v else 0.0 for v in vals]), 4),
            "pairs": len(sub)}
    out["value_surface_direction_parity"] = {
        "per_swap_family": vsp,
        "max_digit_count_deviation": round(
            max(abs(v["digit_count_auroc"] - 0.5) for v in vsp.values()), 4),
        "max_decimal_deviation": round(
            max(abs(v["decimal_auroc"] - 0.5) for v in vsp.values()), 4),
        "bar": "digit-count AUROC in [0.45, 0.55] (report-only; enforced by "
               "bucket-matched construction)"}

    # --- word-level label-marginal balance
    wm = collections.defaultdict(lambda: [0, 0])
    for u, l in zip(df["cited_unit"].to_list(), labels):
        wm[u][0 if l == 1 else 1] += 1
    out["word_label_marginal_balance"] = {
        "per_word": {u: {"as_positive": a, "as_negative": b,
                         "skew": round(abs(a - b) / max(a + b, 1), 4)}
                     for u, (a, b) in sorted(wm.items())},
        "max_skew": round(max(abs(a - b) / max(a + b, 1) for a, b in wm.values()), 4),
        "bar": "0.0 by construction (50/50 inside every family)"}

    # --- digit-surface channels: the numeral never moves inside a pair, so these
    # are exactly chance by construction; measured, not assumed
    out["digit_surface_report_only"] = {
        "trailing_zero_auroc": round(P.auroc(labels, [P.trailing_zeros(v) for v in cited]), 4),
        "digit_count_auroc": round(P.auroc(labels, [P.digits(v) for v in cited]), 4),
        "leading_digit_auroc": round(P.auroc(
            labels, [float(P.leading_digit(v) or 0) for v in cited]), 4),
        "value_magnitude_auroc": round(P.auroc(labels, [P.as_num(v) or 0.0 for v in cited]), 4),
    }

    # --- mechanical re-derivation audit on sampled negatives
    neg = df.filter(pl.col("label") == 0)
    samp = neg.sample(n=min(AUDIT_N, len(neg)), seed=SEED)
    aerrs = []
    for r in samp.iter_rows(named=True):
        t = by_doc.get(r["doc_id"])
        why = None
        if t is None:
            why = "table not found"
        else:
            body, hdr = t["body"], t["hdr"]
            ri, ci = r["row_index"], r["column_index"]
            if not (0 <= ri < len(body) and 0 <= ci < len(hdr)):
                why = "index out of range"
            elif r["unit_carrier"] == "header":
                if body[ri][ci].strip() != r["cited_value"]:
                    why = "value is not the table's cell for the claimed binding"
                elif header_unit(hdr[ci])[0] != r["correct_unit"]:
                    why = "column header does not carry the recorded unit"
            else:
                cu, num = cell_unit(body[ri][ci].strip())
                if cu != r["correct_unit"] or num != r["cited_value"]:
                    why = "cell does not carry the recorded unit/value"
            if why is None and body[ri][t["lab_ci"]].strip() != r["row_key"]:
                why = "row key does not name the claimed row"
            if why is None and SURFACE[r["cited_unit"]].search(r["chunk"]) \
                    and not r["distractor_in_chunk"]:
                why = "corrupted unit is attested in the chunk but not flagged"
            if why is None:
                # the negative must be FALSE: no other column of the same row may
                # state the same numeral under the corrupted unit
                for cj, hdr_j in enumerate(hdr):
                    if cj in (ci, t["lab_ci"]):
                        continue
                    s = body[ri][cj].strip()
                    hu, _ = header_unit(hdr_j)
                    cu, num = cell_unit(s)
                    if ((hu == r["cited_unit"] and s == r["cited_value"])
                            or (cu == r["cited_unit"] and num == r["cited_value"])):
                        why = "negative twin is satisfied elsewhere in the same row"
                        break
        if why:
            aerrs.append({"pair_id": r["pair_id"], "doc_id": r["doc_id"], "why": why})
    out["unit_rederivation_audit"] = {
        "sampled": len(samp), "errors": len(aerrs), "bar": "0 errors",
        "pass": len(aerrs) == 0, "examples": aerrs[:5]}

    out["all_bars_pass"] = all(out[k]["pass"] for k in (
        "claim_only_tfidf_auroc", "within_pair_claim_only_accuracy", "surface_parity",
        "positive_verbatim_substring", "minimal_pair_integrity", "dedupe_disjointness"))
    out["h148_and_audit_pass"] = bool(out["claim_unit_literal_presence"]["pass"]
                                      and out["unit_rederivation_audit"]["pass"])
    return out


def overlap(df, path, name):
    if not path.exists():
        return {"file": path.name, "present": False}
    other = pl.read_parquet(path, columns=["doc_id", "chunk"])
    return {"file": path.name, "present": True,
            "shared_doc_ids": len(set(other["doc_id"]) & set(df["doc_id"])),
            "shared_chunks": len(set(other["chunk"]) & set(df["chunk"])),
            "their_docs": other["doc_id"].n_unique(),
            "note": f"{name}: permitted, measured not enforced"}


# --------------------------------------------------------------------------- #
def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    excluded_ids, prints, eval_rows, unmatched = P.evalset_documents()
    print(f"eval set: {eval_rows} rows -> {len(excluded_ids)} doc_ids, "
          f"{len(prints)} content fingerprints ({unmatched} unmatched)", flush=True)

    print("loading corpora (public: TabFact train, FEVEROUS train)...", flush=True)
    raw = P.tabfact_tables() + P.feverous_tables()
    drop_idx = P.excluded_tables(raw, prints)
    tables, dropped_eval = [], 0
    for ti, t in enumerate(raw):
        if ti in drop_idx or t["doc_id"] in excluded_ids:
            dropped_eval += 1
            continue
        lab = P.label_column(t["hdr"], t["body"])
        if lab is None:
            continue
        t["lab_ci"] = lab
        tables.append(t)
    print(f"  {len(raw)} candidate tables; {len(drop_idx)} carry eval content; "
          f"{dropped_eval} dropped; {len(tables)} admitted", flush=True)

    forms = list(FORM_WEIGHTS)
    w = np.array([FORM_WEIGHTS[f] for f in forms], dtype=float)
    w /= w.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=w)):
        t["form"] = forms[int(k)]
    by_doc = {t["doc_id"]: t for t in tables}

    # ---- enumerate positives ------------------------------------------------
    census = {"unit_cells": collections.Counter(), "unit_tables": collections.Counter(),
              "unit_bearing_tables": 0, "dropped_claim_surface_present": 0,
              "dropped_no_chunk": 0, "dropped_carrier_truncated": 0}
    positives, seen_key, per_doc = [], set(), collections.Counter()
    order = list(range(len(tables)))
    rng.shuffle(order)
    for cap in DOC_CAP_LADDER:
        for ti in order:
            t = tables[ti]
            if per_doc[t["doc_id"]] >= cap:
                continue
            if "cands" not in t:
                t["cands"], t["attested"] = table_candidates(t)
                if t["cands"]:
                    census["unit_bearing_tables"] += 1
                    for u in {c["unit"] for c in t["cands"]}:
                        census["unit_tables"][u] += 1
                    for c in t["cands"]:
                        census["unit_cells"][c["unit"]] += 1
                rng.shuffle(t["cands"])
            for c in t["cands"]:
                if per_doc[t["doc_id"]] >= cap:
                    break
                key = (t["doc_id"], c["ci"], c["ri"])
                if key in seen_key:
                    continue
                lab_ci = t["lab_ci"]
                row_key = t["body"][c["ri"]][lab_ci].strip()
                if not row_key or len(row_key) > 60 or P.as_num(row_key) is not None:
                    continue
                chunk = make_chunk(t, c["ci"], c["ri"], c["cell"], rng)
                if chunk is None or row_key not in chunk:
                    census["dropped_no_chunk"] += 1
                    continue
                low = chunk.lower()
                toks = set(WORD.findall(low))
                if not phrase_absent(c["unit"], low):
                    census["dropped_claim_surface_present"] += 1
                    continue
                # the evidence must STATE the positive's unit: the annotated
                # header (header carrier) or the annotated cell (cell carrier)
                # has to survive the 1,500-character serving cap
                carrier_text = (P.clean(t["hdr"][c["ci"]]) if c["carrier"] == "header"
                                else c["cell"])
                if carrier_text not in chunk:
                    census["dropped_carrier_truncated"] += 1
                    continue
                seen_key.add(key)
                per_doc[t["doc_id"]] += 1
                positives.append({**c, "doc_id": t["doc_id"], "source": t["source"],
                                  "chunk": chunk, "chunk_low": low, "chunk_toks": toks,
                                  "row_key": row_key, "form": t["form"],
                                  "lab_ci": lab_ci, "tab": t})
        print(f"  cap {cap}: {len(positives)} positives over "
              f"{len(per_doc)} documents", flush=True)
        if len(positives) >= 4 * TARGET_PAIRS:
            break

    print(f"positives: {len(positives)}", flush=True)
    print("  by unit: " + json.dumps(
        dict(collections.Counter(p["unit"] for p in positives).most_common())), flush=True)

    # ---- assemble hub families ---------------------------------------------
    fam_pairs, fam_report = build_families(positives, rng, TARGET_PAIRS)
    print(f"family pairs offered: {len(fam_pairs)}", flush=True)

    # a negative must be FALSE: no other column of the SAME row may state the
    # same numeral under the corrupted unit (which would make the twin true)
    def negative_is_false(p, neg_unit):
        t, row = p["tab"], p["tab"]["body"][p["ri"]]
        for ci, hdr in enumerate(t["hdr"]):
            if ci == p["ci"] or ci == t["lab_ci"]:
                continue
            s = row[ci].strip()
            hu, _ = header_unit(hdr)
            cu, num = cell_unit(s)
            if (hu == neg_unit and s == p["val"]) or (cu == neg_unit and num == p["val"]):
                return False
        return True

    before = len(fam_pairs)
    fam_pairs = [(p, n) for p, n in fam_pairs if negative_is_false(p, n)]
    dropped_true_twin = before - len(fam_pairs)
    print(f"  dropped {dropped_true_twin} pairs whose negative twin was satisfiable",
          flush=True)

    # restore exact 50/50 direction balance inside every family after the drop,
    # and drop families that fell below the measurable floor
    by_fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for p, n in fam_pairs:
        by_fam["<->".join(sorted((p["unit"], n)))][p["unit"]].append((p, n))
    fam_pairs, dropped_small = [], {}
    for fam, dirs in sorted(by_fam.items()):
        kept = rebalance_family([x for v in dirs.values() for x in v], rng)
        if len(kept) < MIN_FAMILY_PAIRS:
            dropped_small[fam] = {"kept": 0, "offered": len(kept),
                                  "reason": f"below MIN_FAMILY_PAIRS={MIN_FAMILY_PAIRS}"}
            continue
        fam_pairs += kept
    print(f"  {len(fam_pairs)} pairs after rebalance; dropped families: "
          f"{sorted(dropped_small)}", flush=True)

    rows = []
    for pid, (p, neg_unit) in enumerate(sorted(
            fam_pairs, key=lambda x: (x[0]["doc_id"], x[0]["ci"], x[0]["ri"], x[1]))):
        tpl = CLAIM_TEMPLATES[pid % len(CLAIM_TEMPLATES)]
        pos_claim = build_claim(tpl, p["col"], p["row_key"], p["val"], p["unit"])
        neg_claim = build_claim(tpl, p["col"], p["row_key"], p["val"], neg_unit)
        distractor = attested_in_chunk(p, neg_unit)
        dim = DIM[p["unit"]]
        base = dict(chunk=p["chunk"], doc_id=p["doc_id"], source=p["source"],
                    column=p["col"], column_index=p["ci"], row_key=p["row_key"],
                    row_index=p["ri"], cited_value=p["val"],
                    correct_unit=p["unit"], wrong_unit=neg_unit,
                    dimension=dim,
                    neg_family=FAMILY_OF_DIM.get(dim, "unit_swap"),
                    swap_family="<->".join(sorted((p["unit"], neg_unit))),
                    direction=f"{p['unit']}->{neg_unit}",
                    unit_carrier=p["carrier"], distractor_in_chunk=distractor,
                    surface_parity=bool(abs(len(pos_claim) - len(neg_claim)) <= 2),
                    serial_form=p["form"], template_id=pid % len(CLAIM_TEMPLATES),
                    tag=TAG)
        rows.append(dict(pair_id=pid, label=1, claim=pos_claim,
                         cited_unit=p["unit"], **base))
        rows.append(dict(pair_id=pid, label=0, claim=neg_claim,
                         cited_unit=neg_unit, **base))

    df = pl.DataFrame(rows).unique(subset=["claim", "chunk", "label"],
                                   keep="first", maintain_order=True)
    keep_pairs = df.group_by("pair_id").len().filter(pl.col("len") == 2)["pair_id"]
    df = df.filter(pl.col("pair_id").is_in(keep_pairs)).sort(
        ["pair_id", "label"], descending=[False, True])
    n_pairs = df["pair_id"].n_unique()
    df.write_parquet(OUT)
    print(f"\n{df.height} rows / {n_pairs} pairs over "
          f"{df['doc_id'].n_unique()} documents", flush=True)

    res = verify(df, rng, by_doc)

    # ---- lane-side window statistics ----------------------------------------
    # The COMBINED census is `R18-H150_window_census.py`: the clean mix has to be
    # read untruncated (the cached mix is already cut to 1,500 chars, which would
    # read every clean row as single-window and understate the baseline).
    win_census = {
        "windowing": f"{WIN}/{STRIDE}",
        "scaleunit_lane_H150": window_stats([len(c) for c in df["chunk"].to_list()]),
        "combined": "see R18-H150_window_census.json (clean mix read untruncated)",
        "baseline_multi_window_share": BASELINE_MULTIWINDOW}

    fam = {k: v for k, v in df.group_by("neg_family").len().iter_rows()}
    swap = {k: v for k, v in df.group_by("swap_family").len().iter_rows()}
    man = dict(
        experiment="R18-H150 scale/unit verification lane (convergence arm)",
        seed=SEED, target_pairs=TARGET_PAIRS, rows=df.height, pairs=n_pairs,
        documents=df["doc_id"].n_unique(),
        pairs_per_document=round(n_pairs / max(df["doc_id"].n_unique(), 1), 3),
        doc_cap=DOC_CAP_LADDER[-1], tag=TAG,
        families=fam,
        family_shares={k: round(v / df.height, 4) for k, v in fam.items()},
        swap_families=swap,
        directions={f"{a}": n for a, n in df.group_by("direction").len().iter_rows()},
        dimensions={k: v for k, v in df.group_by("dimension").len().iter_rows()},
        unit_carrier={k: v for k, v in df.group_by("unit_carrier").len().iter_rows()},
        distractor_in_chunk={str(k): v for k, v in
                             df.group_by("distractor_in_chunk").len().iter_rows()},
        diversity=dict(
            serial_forms={k: v for k, v in df.group_by("serial_form").len().iter_rows()},
            templates={str(k): v for k, v in df.group_by("template_id").len().iter_rows()},
            sources={k: v for k, v in df.group_by("source").len().iter_rows()},
            n_templates=len(CLAIM_TEMPLATES),
            distinct_claims=df["claim"].n_unique(),
            distinct_chunks=df["chunk"].n_unique(),
            distinct_columns=df["column"].n_unique()),
        census=dict(
            corpus_tables=len(raw), admitted_tables=len(tables),
            unit_bearing_tables=census["unit_bearing_tables"],
            unit_cells_enumerated=dict(census["unit_cells"].most_common()),
            unit_tables=dict(census["unit_tables"].most_common()),
            positives_built=len(positives),
            dropped_claim_phrase_present=census["dropped_claim_surface_present"],
            dropped_carrier_truncated=census["dropped_carrier_truncated"],
            dropped_no_chunk=census["dropped_no_chunk"],
            dropped_satisfiable_negative=dropped_true_twin,
            dropped_small_families=dropped_small),
        stratification_ablation={
            "note": "same build, same seed, only the value-surface bucket changes "
                    "(H150_STRATIFY); the shipped setting is `full`",
            "off": {"pairs": 5358, "worst_within_pair": 0.5298,
                    "max_digit_count_deviation": 0.5, "max_decimal_deviation": None,
                    "verdict": "bars pass, but the numeral alone separates a "
                               "family's two directions - a claim-only plausibility "
                               "channel a nonlinear model can cash in"},
            "digits_only": {"pairs": 3948, "worst_within_pair": 0.6875,
                            "max_digit_count_deviation": 0.0,
                            "max_decimal_deviation": 0.4444,
                            "verdict": "FAILS the within-pair bar; the decimal "
                                       "channel carries the leak on its own"},
            "full": {"pairs": "as built", "verdict": "shipped"}},
        family_construction=fam_report,
        eval_disjointness=dict(evalset_rows=eval_rows,
                               excluded_doc_ids=len(excluded_ids),
                               content_matched_tables=len(drop_idx),
                               tables_dropped=dropped_eval,
                               tables_admitted=len(tables),
                               shared_documents=0, shared_chunks=0,
                               method="content-based (R17-H144 method), enforced"),
        overlap_permitted=[overlap(df, H144_PAIRS, "H144 pair corpus"),
                           overlap(df, H145_LANE, "H145 scale/unit lane"),
                           overlap(df, H146_LANE, "H146 misbind lane")],
        window_census=win_census,
        verify=res)
    MANIFEST.write_text(json.dumps(man, indent=2))
    VERIFY_REPORT.write_text(json.dumps(
        {"rows": df.height, "pairs": n_pairs, "verify": res,
         "window_census": win_census}, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "documents", "families", "swap_families",
                       "distractor_in_chunk", "window_census", "verify")},
                     indent=2)[:12000], flush=True)
    ok = res["all_bars_pass"] and res["h148_and_audit_pass"]
    print(f"=== R18-H150 SCALE/UNIT LANE {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
