"""R17-H145 SCALE/UNIT VERIFICATION pair corpus - build + verify only, CPU, no GPU.

Builds `R17-H145_scaleunit.parquet`: minimal pairs in which the ONLY difference
between the positive and its negative twin is the scale/unit WORD.  Digits are
never altered, and the cited value is present verbatim in the serialized evidence
together with the unit the evidence itself states.  Pure lookup verification - no
arithmetic, no derivation, doctrine-compliant (H145 amendment A1, clause ii).

  positive - "The elevation of Store Skagastolstind is 2405 m."   (evidence: the
             `elevation (m)` column of that row reads 2405)
  negative - "The elevation of Store Skagastolstind is 2405 km."  (identical
             claim, one word changed; the evidence contradicts it and never
             mentions km anywhere)

Construction rules, all enforced at build time:

  * the unit is GENUINE - it is read off the source table, either from a column
    header annotation (`elevation (m)`, `us viewers (millions)`, `change (%)`)
    or from the cell itself (`2405 m`, `5.2 million`, `45%`).  Units are never
    invented for unitless values
  * the positive's unit surface must be READABLE in the chunk and the negative's
    unit surface must be ABSENT from it, so the defect is groundable and the
    negative is never accidentally supported by a second column
  * the value string is copied from the cell, so within-pair digit surfaces are
    identical by construction
  * swaps stay INSIDE a dimension (length, mass, area, speed, magnitude) - a
    cross-dimension swap ("the area is 1741 kilograms") is detectable without
    reading the evidence
  * families are HUB-shaped and 50/50 in both directions.  Inside a dimension the
    most abundant word is the hub and every other word forms one family with it;
    each family takes as many hub-positive pairs as partner-positive pairs.  A
    hub is supply-optimal - matching two scarce words with each other spends hub
    supply that never binds - and it yields few large families instead of the
    ragged scatter a free negative choice produces.  50/50 inside every family
    implies exact word-level label-marginal balance, so no word marginally
    predicts a label, and families stay large enough to measure against a
    proportion bar

RECORDED DEVIATIONS from the H145 registration text, both forced by corpus supply
and both measured in `..._manifest.json` under `qualified_tables`:

  1. `percent <-> percentage points` is NOT built.  "percentage point(s)" occurs
     in exactly one corpus table, so it can never appear as a POSITIVE, and an
     unbalanced family would hand the claim-only probe the rule
     "percentage points => label 0" - the exact leak the verify bars forbid.
     Instead `percent` is balanced against the other bare-number magnitude words
     (million / billion), which the registration lists in the same breath and
     which the corpus does attest on both sides.  `thousand` is excluded on a
     MEASUREMENT, not on supply - see `EXCLUDED_UNITS`
  2. the magnitude-merge families (`percent <-> million`, `percent <-> billion`)
     are capped at `MAGNITUDE_CAP` of the corpus.  They clear every bar, but a
     wrong magnitude word is more often guessable from the subject alone than a
     wrong unit is, so they fill the corpus rather than carry it

DISJOINTNESS.  The R17-H143 eval set is excluded on content and that exclusion is
an abort condition.  The R17-H144 pair corpus is NOT excluded: build 1 did
exclude it and paid two thirds of the unit-bearing supply (309 admitted tables
against 902), and the coordinator dropped the exclusion for build 2 - H144 feeds
the student decoder, this family feeds the encoder, the benches are disjoint and
there is no leak path.  Overlap with H144 is measured and recorded, not enforced.
Build 1's counts are kept in the manifest's `build_history`.

Corpus loading, serialization, numeral canonicalization, the eval-set content
exclusion and the AUROC helper are REUSED from `R17-H144_pairs.py` (loaded by
path - the module name is not importable).  Public data only.  Seeded,
reproducible, Polars, no network, no torch.

Run:  uv run python experiments/grounding-semantic/R17-H145_scaleunit.py
"""

import collections
import importlib.util
import json
import pathlib
import random
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
H144_SRC = HERE / "R17-H144_pairs.py"
H144_PAIRS = HERE / "R17-H144_pairs.parquet"
EVALSET = HERE / "R17-H143_evalset.parquet"
OUT = HERE / "R17-H145_scaleunit.parquet"
MANIFEST = HERE / "R17-H145_scaleunit_manifest.json"

_spec = importlib.util.spec_from_file_location("h144_pairs", H144_SRC)
h144 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h144)

SEED = 1145
TARGET_PAIRS = 3_000
MAGNITUDE_CAP = 0.40       # magnitude-merge families fill, they do not carry
DOC_CAP = 8
BODY_ROWS = 6
MIN_COL_CELLS = 1          # a column qualifies on >= 1 usable unit-bearing cell
MIN_FAMILY_PAIRS = 32      # smaller swap families cannot be measured, only guessed at
N_FOLDS = 5                # document-disjoint folds for the claim-only probe
TAG = "quant_scale_unit"

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


# --------------------------------------------------------------------------- #
# unit vocabulary
#   dim      - swaps are legal only inside a dimension
#   phrase   - how the unit is written in the CLAIM
#   surface  - how the unit may appear in the EVIDENCE (presence / absence test)
# --------------------------------------------------------------------------- #
UNITS = {
    # magnitude words attached to a bare number
    "million":  ("magnitude", "million",  r"\bmillions?\b|\bmn\b"),
    "thousand": ("magnitude", "thousand", r"\bthousands?\b|\b1000s\b|\b000s\b"),
    "billion":  ("magnitude", "billion",  r"\bbillions?\b|\bbn\b"),
    "percent":  ("magnitude", "percent",  r"%|\bpercent\b|\bpercentage\b|\bpct\b"),
    # length
    "m":    ("length", "m",      r"\bm\b|\bmet(?:re|er)s?\b"),
    "km":   ("length", "km",     r"\bkm\b|\bkilomet(?:re|er)s?\b"),
    "cm":   ("length", "cm",     r"\bcm\b|\bcentimet(?:re|er)s?\b"),
    "mm":   ("length", "mm",     r"\bmm\b|\bmillimet(?:re|er)s?\b"),
    "ft":   ("length", "ft",     r"\bft\b|\bfeet\b|\bfoot\b"),
    "mile": ("length", "miles",  r"\bmiles?\b|\bmi\b"),
    # mass
    "kg":    ("mass", "kg",      r"\bkg\b|\bkilograms?\b"),
    "g":     ("mass", "g",       r"\bg\b|\bgrams?\b"),
    "lb":    ("mass", "lb",      r"\blbs?\b|\bpounds?\b"),
    "tonne": ("mass", "tonnes",  r"\btonnes?\b|\bmetric tons?\b"),
    "ton":   ("mass", "tons",    r"\btons?\b|\bshort tons?\b"),
    # area
    "km2":  ("area", "square kilometres", r"\bkm\s*(?:2|square)\b|\bsq\s*km\b|\bsqkm\b|square kilomet(?:re|er)s?"),
    "m2":   ("area", "square metres",     r"\bm\s*(?:2|square)\b|square met(?:re|er)s?"),
    "ha":   ("area", "hectares",          r"\bha\b|\bhectares?\b"),
    "acre": ("area", "acres",             r"\bacres?\b"),
    # speed
    "mph": ("speed", "mph",  r"\bmph\b|miles per hour"),
    "kmh": ("speed", "km/h", r"\bkm\s*/\s*h\b|\bkph\b|kilomet(?:re|er)s per hour"),
    # power
    "kw": ("power", "kW", r"\bkw\b|\bkilowatts?\b"),
    "hp": ("power", "hp", r"\bhp\b|\bhorsepower\b"),
}
# MEASURED EXCLUSION.  `thousand` reaches 40 positives once the R17-H144 document
# exclusion is lifted, enough for an 80-pair family - but they come from FIVE
# tables, and column names repeat across them ("capacity (thousands of ...)").
# A document-disjoint probe then learns the column-name/unit association from the
# other four tables and transfers it: measured within-pair claim-only accuracy
# 0.70 against the 0.60 bar, the only family ever to breach it.  The word is
# dropped whole rather than the corpus fitted to the probe.
EXCLUDED_UNITS = {"thousand"}

DIM = {u: v[0] for u, v in UNITS.items()}
PHRASE = {u: v[1] for u, v in UNITS.items()}
SURFACE = {u: re.compile(v[2], re.IGNORECASE) for u, v in UNITS.items()}

# `m` and `m2`, `km` and `km2` share a prefix; the area patterns must be tried
# first so `area (km 2)` is not read as a plain kilometre column.
HDR_PATS = [
    (r"\(\s*km\s*(?:2|square|\^2)\s*\)|\bkm\s*2\b|\bsq\s*km\b|\bsqkm\b|square\s+kilomet(?:re|er)s?", "km2"),
    (r"\(\s*m\s*(?:2|square|\^2)\s*\)|\bm\s*2\b|square\s+met(?:re|er)s?", "m2"),
    (r"\(\s*(?:in\s+)?millions?\s*(?:of\s+([\w\s]+?)\s*)?\)|,?\s+in\s+millions?\b", "million"),
    (r"\(\s*(?:in\s+)?thousands?\s*(?:of\s+([\w\s]+?)\s*)?\)|,?\s+in\s+thousands?\b|\bin\s+1000s\b", "thousand"),
    (r"\(\s*(?:in\s+)?billions?\s*(?:of\s+([\w\s]+?)\s*)?\)|,?\s+in\s+billions?\b", "billion"),
    (r"\(\s*(?:in\s+)?(?:km|kilomet(?:re|er)s?)\s*\)|\bin\s+kilomet(?:re|er)s\b", "km"),
    (r"\(\s*(?:in\s+)?(?:m|met(?:re|er)s?)\s*\)|\bin\s+met(?:re|er)s\b", "m"),
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
    (r"\(\s*(?:in\s+)?(?:hp|horsepower)\s*\)", "hp"),
    (r"\(\s*(?:in\s+)?mph\s*\)|\bmph\b", "mph"),
    (r"\(\s*(?:in\s+)?km\s*/\s*h\s*\)|\bkm\s*/\s*h\b|\bkph\b", "kmh"),
    (r"\(\s*%\s*\)|(?<![a-z0-9])%|\bpercentage\b|\bpercent\b|\bpct\b", "percent"),
]
HDR_PATS = [(re.compile(p, re.IGNORECASE), u) for p, u in HDR_PATS]

NUMPART = r"(-?[\d,]+(?:\.\d+)?)"
CELL_PATS = [
    (rf"^{NUMPART}\s*(?:%|percent)$", "percent"),
    (rf"^{NUMPART}\s*millions?$", "million"),
    (rf"^{NUMPART}\s*billions?$", "billion"),
    (rf"^{NUMPART}\s*thousands?$", "thousand"),
    (rf"^{NUMPART}\s*km\s*(?:2|square)$", "km2"),
    (rf"^{NUMPART}\s*m\s*(?:2|square)$", "m2"),
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
    (rf"^{NUMPART}\s*(?:hp|horsepower)$", "hp"),
    (rf"^{NUMPART}\s*mph$", "mph"),
    (rf"^{NUMPART}\s*km\s*/\s*h$", "kmh"),
]
CELL_PATS = [(re.compile(p, re.IGNORECASE), u) for p, u in CELL_PATS]

# `m` and `g` are ambiguous abbreviations: `5.46 m` in a television ratings
# column is 5.46 MILLION viewers, not 5.46 metres, and `g` is a playing position
# as often as it is a gram.  For those two the column must name the right kind of
# quantity; every other unit token in the vocabulary is unambiguous on its own.
COL_LEXICON = {
    "m": re.compile(r"length|height|elevation|depth|distance|width|altitude|"
                    r"prominence|isolation|span|diameter|radius|wingspan|draught|"
                    r"draft|beam|thickness|circumference|perimeter|gauge|clearance|"
                    r"range|rise|drop|size", re.IGNORECASE),
    "g": re.compile(r"weight|mass|dose|payload|content|load", re.IGNORECASE),
}

ANY_UNIT = re.compile("|".join(f"(?:{v[2]})" for v in UNITS.values()), re.IGNORECASE)
PUNCT = re.compile(r"^[\s,;:.\-/()]+|[\s,;:.\-/()]+$")


def clean_name(s):
    """Tidy a column name for use in a claim.

    Cutting a unit annotation out of a header can leave a dangling bracket -
    `gdp (nominal, billions of usd)` becomes `gdp (nominal,` - which reads as a
    typo in the claim.  An unbalanced name loses its brackets entirely."""
    out = h144.clean(PUNCT.sub("", h144.clean(s)))
    if out.count("(") != out.count(")"):
        out = h144.clean(PUNCT.sub("", out.replace("(", " ").replace(")", " ")))
    return out


def header_unit(hdr):
    """(unit, column name with the annotation removed, trailing qualifier)."""
    for rx, u in HDR_PATS:
        m = rx.search(hdr)
        if not m:
            continue
        tail = ""
        for gi in range(1, (m.re.groups or 0) + 1):
            if m.group(gi):
                tail = clean_name(m.group(gi))
        name = clean_name(hdr[:m.start()] + " " + hdr[m.end():])
        return u, name, tail
    return None, None, None


def cell_unit(cell):
    for rx, u in CELL_PATS:
        m = rx.match(cell)
        if m:
            return u, m.group(1)
    return None, None


# --------------------------------------------------------------------------- #
# candidate enumeration
# --------------------------------------------------------------------------- #
def numeric_cells(body, ci):
    out = []
    for ri, r in enumerate(body):
        s = r[ci].strip()
        if h144.as_num(s) is not None:
            out.append((ri, s))
    return out


def table_candidates(t):
    """Every (column, row) of one table that carries a genuine unit."""
    lab = t["lab_ci"]
    out = []
    for ci, hdr in enumerate(t["hdr"]):
        if ci == lab or not h144.good_header(hdr):
            continue
        u, name, tail = header_unit(hdr)
        if u in EXCLUDED_UNITS:
            continue
        if u is not None:
            # a header annotation names the unit unambiguously, so one numeric
            # cell is enough; a bare cell needs company to prove it is a unit
            # column rather than a stray string
            cells = numeric_cells(t["body"], ci)
            if not cells or len(name) < 3 or ANY_UNIT.search(name):
                continue
            if u in COL_LEXICON and not COL_LEXICON[u].search(name):
                continue
            for ri, val in cells:
                out.append({"ci": ci, "ri": ri, "unit": u, "col": name,
                            "tail": tail, "val": val, "cell": val,
                            "carrier": "header"})
            continue
        name = clean_name(hdr)
        if len(name) < 3 or ANY_UNIT.search(name):
            continue
        cells = []
        for ri, r in enumerate(t["body"]):
            cu, num = cell_unit(r[ci].strip())
            if cu is not None and cu not in EXCLUDED_UNITS:
                cells.append((ri, cu, num, r[ci].strip()))
        by_u = collections.Counter(c[1] for c in cells)
        for ri, cu, num, raw_cell in cells:
            if by_u[cu] < MIN_COL_CELLS:
                continue
            if cu in COL_LEXICON and not COL_LEXICON[cu].search(name):
                continue
            out.append({"ci": ci, "ri": ri, "unit": cu, "col": name,
                        "tail": "", "val": num, "cell": raw_cell,
                        "carrier": "cell"})
    return out


def build_claim(tpl, col, key, val, unit, tail):
    phrase = PHRASE[unit] + (f" {tail}" if tail else "")
    return tpl.format(col=col, key=key, val=val, unit=phrase)


def make_chunk(t, ci, ri, cell, rng, attempts=5):
    """Serialize a 6-row window holding the target row.  Wide tables can push the
    target cell past the 1,500-character serving cap, so the row draw is retried
    and then narrowed until the cell is readable."""
    for k in range(attempts):
        width = BODY_ROWS - 1 if k < attempts - 2 else (3 if k == attempts - 2 else 1)
        keep = {ri}
        others = [r for r in range(len(t["body"])) if r != ri]
        rng.shuffle(others)
        keep |= set(others[: width])
        body = [t["body"][r] for r in sorted(keep)]
        chunk = h144.serialize(t["form"], t["caption"], t["hdr"], body, t["lab_ci"], (ci,))
        if cell in chunk:
            return chunk
    return None


# --------------------------------------------------------------------------- #
# balance: n_w positives of word w, n_w negatives of word w, inside a dimension
# --------------------------------------------------------------------------- #
def balanced_targets(counts, budget=None):
    """Largest per-word allocation with every word <= half the dimension total."""
    n = dict(counts)
    while n:
        tot = sum(n.values())
        w = max(n, key=lambda k: n[k])
        if n[w] * 2 <= tot:
            break
        rest = tot - n[w]
        if rest == 0:
            return {}
        n[w] = rest
    if budget is not None and sum(n.values()) > budget:
        tot = sum(n.values())
        scaled = {w: int(v * budget / tot) for w, v in n.items()}
        return balanced_targets({w: v for w, v in scaled.items() if v > 0})
    return {w: v for w, v in n.items() if v > 0}


def build_families(positives, min_family, rng):
    """Pair every dimension's words into HUB families and fill each direction 50/50.

    The dimension's most abundant word is the hub; each other word forms one
    family with it.  A hub structure is supply-optimal - matching two scarce words
    with each other spends hub supply that is never the binding constraint - and
    it yields FEW LARGE families instead of the ragged scatter a free negative
    choice produces, which matters twice over: a family whose two directions are
    not 50/50 hands the claim-only probe a within-family word rule, and a family
    of four pairs cannot be measured against a proportion bar at all.

    50/50 inside every family implies exact word-level label-marginal balance, so
    no separate equalisation is needed.  A positive is usable in a family only if
    its own evidence never mentions the other word.
    """
    by_dim = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in positives:
        by_dim[DIM[p["unit"]]][p["unit"]].append(p)

    pairs, report = [], {}
    for dim in sorted(by_dim):
        pool = by_dim[dim]
        if len(pool) < 2:
            report[dim] = {"hub": None, "families": {}, "note": "single word, unusable"}
            continue
        hub = max(pool, key=lambda w: len(pool[w]))
        hub_pool = pool[hub][:]
        rng.shuffle(hub_pool)
        fams = {}
        # spend the hub on the largest partners first
        for v in sorted((w for w in pool if w != hub), key=lambda w: -len(pool[w])):
            side_v = [p for p in pool[v] if not SURFACE[hub].search(p["chunk"])]
            side_h = [p for p in hub_pool if not SURFACE[v].search(p["chunk"])]
            k = min(len(side_v), len(side_h))
            fam = "<->".join(sorted((hub, v)))
            if 2 * k < min_family:
                fams[fam] = {"kept": 0, "offered": 2 * k,
                             "supply": {v: len(pool[v]), hub: len(hub_pool)}}
                continue
            rng.shuffle(side_v)
            taken_h = side_h[:k]
            pairs += [(p, hub) for p in side_v[:k]] + [(p, v) for p in taken_h]
            used = {id(p) for p in taken_h}
            hub_pool = [p for p in hub_pool if id(p) not in used]
            fams[fam] = {"kept": 2 * k, "offered": 2 * k,
                         "supply": {v: len(pool[v]), hub: len(pool[hub])}}
        report[dim] = {"hub": hub, "families": fams}
    return pairs, report


def allocate(pairs, target, magnitude_cap, min_family):
    """Cut to the target size with the magnitude-merge families held to a share.

    `percent <-> million` and `percent <-> billion` swap two bare-number
    magnitude words rather than two units of one physical dimension.  They clear
    every bar, but a wrong magnitude word is more often guessable from the
    subject alone than a wrong unit is, so the coordinator caps them: they FILL
    the corpus up to `magnitude_cap` of it and the genuine unit dimensions carry
    the rest.  Both groups are cut by whole direction-pairs, so 50/50 per family
    - and therefore exact word-marginal balance - survives the cut."""
    groups = {"magnitude": [], "unit": []}
    for pr in pairs:
        groups["magnitude" if DIM[pr[0]["unit"]] == "magnitude" else "unit"].append(pr)
    n_u, n_m = len(groups["unit"]), len(groups["magnitude"])
    total = 0
    for t in range(min(target, n_u + n_m), 0, -1):
        m = min(n_m, int(t * magnitude_cap))
        if t - m <= n_u:
            total, take_m = t, m
            break
    else:
        return pairs, {"note": "nothing to allocate"}
    take = {"magnitude": take_m, "unit": total - take_m}

    out = []
    for g, want in take.items():
        by_dir = collections.defaultdict(list)
        for pr in groups[g]:
            by_dir[(pr[0]["unit"], pr[1])].append(pr)
        fams = sorted({tuple(sorted(k)) for k in by_dir},
                      key=lambda ab: min(len(by_dir[(ab[0], ab[1])]),
                                         len(by_dir[(ab[1], ab[0])])))
        # smallest family first, each taken WHOLE while the budget lasts, and the
        # whole remainder taken from the largest.  A proportional cut would scale
        # the rare families below the measurable minimum and silently delete them
        # - `thousand` and `billion` are the scarcest scale words in the corpus and
        # the most on-target for the skill, so they must not be what the cut eats.
        left = want
        for i, (a, b) in enumerate(fams):
            k = min(len(by_dir[(a, b)]), len(by_dir[(b, a)]))
            if i == len(fams) - 1:
                k = min(k, left // 2)
            elif 2 * k > left:
                k = left // 2
            if 2 * k < min_family:
                continue
            out += by_dir[(a, b)][:k] + by_dir[(b, a)][:k]
            left -= 2 * k
    got_m = sum(1 for pr in out if DIM[pr[0]["unit"]] == "magnitude")
    return out, {"target": target, "magnitude_cap": magnitude_cap,
                 "offered": {"unit": n_u, "magnitude": n_m},
                 "allocated": {"unit": len(out) - got_m, "magnitude": got_m},
                 "magnitude_share": round(got_m / max(len(out), 1), 4)}


# --------------------------------------------------------------------------- #
def verify(df, rng):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    out = {}
    # Document-disjoint K-fold, scored OUT OF FOLD, with folds STRATIFIED on the
    # document's swap direction.
    #
    # Out-of-fold scoring gives every swap family its full size as support - a
    # single 70/30 split leaves the smaller families with a handful of held-out
    # pairs and a within-pair proportion measured on four pairs is noise.
    # Stratification removes a second artefact: a source document is
    # single-direction (a kilometre table only ever yields km-positive pairs), so
    # an unstratified fold is direction-skewed and its training complement is
    # skewed the OTHER way.  The probe then learns the complement's direction and
    # scores the fold below chance - measured -0.15 AUROC, an artefact of the
    # split rather than a property of the corpus, whose word marginals are
    # balanced exactly.  Stratified folds put both directions in every fold.
    doc_dir = {d: k for d, k in df.filter(pl.col("label") == 1.0)
               .group_by("doc_id").agg(pl.col("direction").first()).iter_rows()}
    strata = collections.defaultdict(list)
    for d in sorted(doc_dir):
        strata[doc_dir[d]].append(d)
    fold_of, i = {}, 0
    for k in sorted(strata):
        ds = strata[k]
        rng.shuffle(ds)
        for d in ds:
            fold_of[d] = i % N_FOLDS
            i += 1
    docs = sorted(fold_of)
    df = df.with_columns(
        pl.col("doc_id").replace_strict(fold_of, return_dtype=pl.Int32).alias("_fold"))
    score = np.zeros(len(df), dtype=float)
    idx = np.arange(len(df))
    folds = df["_fold"].to_numpy()
    claims = df["claim"].to_list()
    labels = df["label"].to_list()
    for f in range(N_FOLDS):
        tr_i, te_i = idx[folds != f], idx[folds == f]
        if len(te_i) == 0:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              max_features=300_000, sublinear_tf=True)
        Xtr = vec.fit_transform([claims[i] for i in tr_i])
        Xte = vec.transform([claims[i] for i in te_i])
        # converged liblinear, NOT default lbfgs (R17-H144 finding ii): minimal
        # pairs cancel the gradient at w = 0 and lbfgs reports an all-zero fit.
        clf = LogisticRegression(solver="liblinear", C=4.0, tol=1e-7, max_iter=3000)
        clf.fit(Xtr, [labels[i] for i in tr_i])
        score[te_i] = clf.decision_function(Xte)
    probe = h144.auroc(labels, score)
    out["claim_only_tfidf_auroc"] = {
        "value": round(float(probe), 4), "bar": "< 0.55", "pass": bool(probe < 0.55),
        "scoring": f"{N_FOLDS}-fold document-disjoint, out of fold",
        "documents": len(docs), "rows": len(df)}

    scored = df.select(["pair_id", "label", "swap_family"]).with_columns(pl.Series("score", score))
    fam_acc, worst, worst_two_sided = {}, 0.0, 0.0
    for fam, sub in scored.group_by("swap_family"):
        fam = fam[0]
        piv = sub.pivot(on="label", index="pair_id", values="score",
                        aggregate_function="first").drop_nulls()
        if len(piv) == 0:
            continue
        pos, neg = piv["1.0"].to_numpy(), piv["0.0"].to_numpy()
        acc = float(((pos > neg) + 0.5 * (pos == neg)).mean())
        fam_acc[fam] = {"acc": round(acc, 4), "pairs": len(piv)}
        worst = max(worst, acc)
        worst_two_sided = max(worst_two_sided, abs(acc - 0.5))
    out["within_pair_claim_only_accuracy"] = {
        "per_family": fam_acc, "worst": round(worst, 4), "bar": "< 0.60",
        "pass": bool(worst < 0.60),
        # report-only: the registered bar is one-sided, but a family the probe
        # ranks INVERTED carries the same claim-only signal read backwards
        "worst_two_sided_deviation_report_only": round(worst_two_sided, 4)}

    pos_w = collections.Counter(df.filter(pl.col("label") == 1.0)["unit_pos"].to_list())
    neg_w = collections.Counter(df.filter(pl.col("label") == 0.0)["unit_neg"].to_list())
    skew, per_word = 0.0, {}
    for w in sorted(set(pos_w) | set(neg_w)):
        p, n = pos_w[w], neg_w[w]
        s = abs(p - n) / max(p + n, 1)
        per_word[w] = {"as_positive": p, "as_negative": n, "skew": round(s, 5)}
        skew = max(skew, s)
    out["word_label_marginal_balance"] = {
        "per_word": per_word, "max_skew": round(skew, 5),
        "bar": "max skew <= 0.02", "pass": bool(skew <= 0.02)}

    present = [bool(h144.canon_set(v) & h144.present_numbers(c))
               for v, c in zip(df["value"].to_list(), df["chunk"].to_list())]
    rate = float(np.mean(present))
    out["value_presence_rate"] = {"value": round(rate, 5), "bar": "== 1.0",
                                  "pass": bool(rate == 1.0)}

    pos = df.filter(pl.col("label") == 1.0)
    up = float(np.mean([bool(SURFACE[u].search(c)) for u, c in
                        zip(pos["unit_pos"].to_list(), pos["chunk"].to_list())]))
    un = float(np.mean([not SURFACE[u].search(c) for u, c in
                        zip(df["unit_neg"].to_list(), df["chunk"].to_list())]))
    out["unit_surface_discipline"] = {
        "positive_unit_readable_in_evidence": round(up, 5),
        "negative_unit_absent_from_evidence": round(un, 5),
        "bar": "both == 1.0", "pass": bool(up == 1.0 and un == 1.0)}

    out["report_only"] = {
        "claim_char_length_auroc": round(float(h144.auroc(
            df["label"].to_list(), [len(c) for c in df["claim"].to_list()])), 4),
        "value_digit_count_auroc": round(float(h144.auroc(
            df["label"].to_list(), [h144.digits(v) for v in df["value"].to_list()])), 4),
        "within_pair_digit_surfaces_identical": bool(
            df.group_by("pair_id").agg(pl.col("value").n_unique().alias("n"))["n"].max() == 1),
    }
    out["all_bars_pass"] = all(out[k]["pass"] for k in
                               ("claim_only_tfidf_auroc", "within_pair_claim_only_accuracy",
                                "word_label_marginal_balance", "value_presence_rate",
                                "unit_surface_discipline"))
    return out


# --------------------------------------------------------------------------- #
def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    excluded_ids, prints, eval_rows, unmatched = h144.evalset_documents()
    print(f"eval set: {eval_rows} rows -> {len(excluded_ids)} v2 doc_ids, "
          f"{len(prints)} content fingerprints ({unmatched} unmatched)", flush=True)

    print("loading corpora (public: TabFact train, FEVEROUS train)...", flush=True)
    raw = h144.tabfact_tables() + h144.feverous_tables()
    drop_idx = h144.excluded_tables(raw, prints)
    h144_docs = set(pl.read_parquet(H144_PAIRS, columns=["doc_id"])["doc_id"].unique().to_list())
    print(f"  {len(raw)} candidate tables; {len(drop_idx)} carry eval-set content; "
          f"{len(h144_docs)} H144 documents", flush=True)

    # The R17-H143 EVAL SET is excluded on content; the R17-H144 pair corpus is
    # NOT - the coordinator dropped that exclusion for build 2 (H144 feeds the
    # student decoder, this family feeds the encoder, the benches are disjoint and
    # there is no leak path).  Build 1 carried it and paid two thirds of the
    # unit-bearing supply for it; the overlap is now recorded, not enforced.
    tables, dropped = [], collections.Counter()
    for ti, t in enumerate(raw):
        eval_hit = ti in drop_idx or t["doc_id"] in excluded_ids
        lab = h144.label_column(t["hdr"], t["body"])
        if lab is None:
            continue
        t["lab_ci"] = lab
        cands = table_candidates(t)
        if not cands:
            continue
        if eval_hit:
            dropped["evalset"] += 1
            continue
        dropped["shared_with_h144"] += t["doc_id"] in h144_docs
        t["cands"] = cands
        tables.append(t)
    print(f"  unit-bearing tables: {len(tables)} admitted, "
          f"{dropped['evalset']} dropped as eval-set; "
          f"{dropped['shared_with_h144']} admitted tables are also H144 documents",
          flush=True)

    forms = list(FORM_WEIGHTS)
    wts = np.array([FORM_WEIGHTS[f] for f in forms], dtype=float)
    wts /= wts.sum()
    for t, k in zip(tables, np_rng.choice(len(forms), size=len(tables), p=wts)):
        t["form"] = forms[int(k)]

    # ---- ceilings, for the manifest --------------------------------------- #
    def ceiling(tabs):
        cells, ntab = collections.Counter(), collections.Counter()
        for c in tabs:
            for u, n in c.items():
                cells[u] += n
                ntab[u] += 1
        per_dim = collections.defaultdict(dict)
        for u in cells:
            per_dim[DIM[u]][u] = min(cells[u], DOC_CAP * ntab[u])
        return ({d: balanced_targets(w) for d, w in per_dim.items()},
                {u: {"cells": cells[u], "tables": ntab[u]} for u in sorted(cells)})

    ceil_excl, supply_excl = ceiling([collections.Counter(c["unit"] for c in t["cands"])
                                      for t in tables])
    print(f"  balanced ceiling (as built)          : "
          f"{sum(sum(v.values()) for v in ceil_excl.values())}", flush=True)

    # ---- positives: scarcest unit first, doc cap enforced ----------------- #
    global_cells = collections.Counter()
    for t in tables:
        for c in t["cands"]:
            global_cells[c["unit"]] += 1

    order = list(range(len(tables)))
    rng.shuffle(order)
    positives, seen = [], set()
    rejects = collections.Counter()
    for oi in order:
        t = tables[oi]
        by_unit = collections.defaultdict(list)
        for c in t["cands"]:
            by_unit[c["unit"]].append(c)
        used = 0
        for u in sorted(by_unit, key=lambda x: global_cells[x]):
            cs = by_unit[u][:]
            rng.shuffle(cs)
            for c in cs:
                if used >= DOC_CAP:
                    break
                tpl = CLAIM_TEMPLATES[rng.randrange(len(CLAIM_TEMPLATES))]
                key = t["body"][c["ri"]][t["lab_ci"]].strip()
                if not key or ANY_UNIT.search(key):
                    rejects["key_carries_a_unit_word"] += 1
                    continue
                chunk = make_chunk(t, c["ci"], c["ri"], c["cell"], rng)
                if chunk is None:
                    rejects["cell_not_readable_in_chunk"] += 1
                    continue
                if not (h144.canon_set(c["val"]) & h144.present_numbers(chunk)):
                    rejects["value_numeral_not_present"] += 1
                    continue
                if not SURFACE[u].search(chunk):
                    rejects["positive_unit_not_readable"] += 1
                    continue
                claim = build_claim(tpl, c["col"], key, c["val"], u, c["tail"])
                dedup = (claim, chunk)
                if dedup in seen:
                    rejects["duplicate_claim_chunk"] += 1
                    continue
                seen.add(dedup)
                positives.append({
                    "doc_id": t["doc_id"], "source": t["source"], "unit": u,
                    "col": c["col"], "key": key, "val": c["val"], "tail": c["tail"],
                    "tpl": tpl, "chunk": chunk, "carrier": c["carrier"],
                    "serial_form": t["form"], "claim": claim})
                used += 1
            if used >= DOC_CAP:
                break
    gen_counts = collections.Counter(p["unit"] for p in positives)
    print(f"  positive candidates: {len(positives)} over "
          f"{len({p['doc_id'] for p in positives})} documents", flush=True)
    print(f"  generated per unit: {dict(gen_counts.most_common())}", flush=True)
    print(f"  construction rejects: {dict(rejects.most_common())}", flush=True)

    # ---- hub families, 50/50 in each direction ---------------------------- #
    pairs, family_report = build_families(positives, MIN_FAMILY_PAIRS, rng)
    pairs, allocation = allocate(pairs, TARGET_PAIRS, MAGNITUDE_CAP, MIN_FAMILY_PAIRS)
    rng.shuffle(pairs)
    print(f"  pairs after hub family balance: {len(pairs)}", flush=True)
    print(f"  allocation: {json.dumps(allocation)}", flush=True)
    print(f"  family report: {json.dumps(family_report)}", flush=True)

    rows = []
    for pid, (p, v) in enumerate(pairs):
        fam = "<->".join(sorted((p["unit"], v)))
        base = {"pair_id": pid, "chunk": p["chunk"], "swap_family": fam,
                "direction": f"{p['unit']}->{v}", "dimension": DIM[p["unit"]],
                "doc_id": p["doc_id"], "source": p["source"], "value": p["val"],
                "unit_pos": p["unit"], "unit_neg": v, "tag": TAG,
                "block": "scaleunit", "carrier": p["carrier"],
                "serial_form": p["serial_form"], "column": p["col"], "key": p["key"],
                "claim_template": p["tpl"]}
        rows.append({"claim": p["claim"], "label": 1.0, **base})
        rows.append({"claim": build_claim(p["tpl"], p["col"], p["key"], p["val"], v, p["tail"]),
                     "label": 0.0, **base})

    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("label").cast(pl.Float32))
    lead = ["pair_id", "claim", "chunk", "label", "swap_family", "direction",
            "doc_id", "source", "value", "unit_pos", "unit_neg"]
    df = df.select(lead + [c for c in df.columns if c not in lead])
    # content dedup, order-stable (R17-H144 finding: maintain_order=True)
    before = len(df)
    kept_ids = (df.unique(subset=["claim", "chunk", "label"], keep="first", maintain_order=True)
                  .group_by("pair_id").len().filter(pl.col("len") == 2).select("pair_id"))
    df = df.join(kept_ids, on="pair_id", how="semi")
    print(f"  content dedup: {before} -> {len(df)} rows", flush=True)
    df.write_parquet(OUT)
    n_pairs = df["pair_id"].n_unique()
    print(f"wrote {OUT}  rows={len(df)} pairs={n_pairs}", flush=True)

    # ---- disjointness ----------------------------------------------------- #
    built_docs = set(df["doc_id"].unique().to_list())
    raw_index = {t["doc_id"]: ti for ti, t in enumerate(raw)}
    ev = pl.read_parquet(EVALSET, columns=["claim", "chunk"])
    h144_df = pl.read_parquet(H144_PAIRS, columns=["claim", "chunk", "doc_id"])
    disj = {
        "evalset": {
            "method": ("content fingerprints per the R17-H144 build (the v2 lane's "
                       "FEVEROUS doc_ids are not stable across rebuilds), plus the "
                       "v2 doc_id set"),
            "eval_rows": eval_rows,
            "eval_rows_unmatched_to_a_v2_document": unmatched,
            "eval_v2_doc_ids": len(excluded_ids),
            "eval_content_fingerprints": len(prints),
            "corpus_tables_matching_eval_content": len(drop_idx),
            "unit_bearing_tables_dropped": dropped["evalset"],
            "shared_doc_ids": len(sorted(d for d in built_docs
                                         if d in excluded_ids or raw_index[d] in drop_idx)),
            "shared_evidence_chunks": len(set(df["chunk"].to_list()) & set(ev["chunk"].to_list())),
            "shared_claim_strings": len(set(df["claim"].to_list()) & set(ev["claim"].to_list())),
        },
        "h144_pairs": {
            "method": ("NOT ENFORCED in build 2 - the coordinator dropped the "
                       "H144 document exclusion (H144 feeds the student decoder, "
                       "this family feeds the encoder, the benches are disjoint, "
                       "no leak path). Overlap is measured and recorded. Both "
                       "corpora come from the same loader in the same order, so "
                       "the ids are comparable"),
            "enforced": False,
            "h144_documents": len(h144_docs),
            "admitted_tables_also_in_h144": dropped["shared_with_h144"],
            "shared_doc_ids": len(built_docs & h144_docs),
            "shared_evidence_chunks": len(set(df["chunk"].to_list()) & set(h144_df["chunk"].to_list())),
            "shared_claim_strings": len(set(df["claim"].to_list()) & set(h144_df["claim"].to_list())),
        },
    }
    print(json.dumps(disj, indent=1), flush=True)

    print("verify pass...", flush=True)
    v = verify(df, random.Random(SEED))
    print(json.dumps(v, indent=1), flush=True)

    per_doc = df.filter(pl.col("label") == 1.0).group_by("doc_id").len()["len"]
    fam_rows = df.filter(pl.col("label") == 1.0)
    families = {}
    for fam, sub in fam_rows.group_by("swap_family"):
        fam = fam[0]
        families[fam] = {
            "pairs": len(sub),
            "dimension": sub["dimension"][0],
            "directions": {str(k): n for k, n in sub.group_by("direction").len().iter_rows()},
        }

    manifest = {
        "experiment": "R17-H145 scale/unit verification pair corpus (amendment A1, clause ii)",
        "build": 2,
        "build_history": {
            "build_1": {
                "rows": 2116, "pairs": 1058, "documents": 252,
                "exclusions": "R17-H143 eval set (content) AND R17-H144 documents",
                "unit_bearing_tables": {"admitted": 309, "dropped_evalset": 190,
                                        "dropped_h144": 593},
                "balanced_ceiling": 1820,
                "verify": {"claim_only_tfidf_auroc": 0.4594,
                           "worst_family_within_pair": 0.5395,
                           "word_marginal_max_skew": 0.0,
                           "value_presence_rate": 1.0,
                           "all_bars_pass": True},
                "families": {"million<->percent": 456, "km<->m": 282, "kmh<->mph": 76,
                             "cm<->km": 58, "billion<->percent": 42, "g<->kg": 40,
                             "kg<->lb": 40, "km<->mile": 32, "km<->mm": 32},
                "outcome": ("all bars passed but only 35% of the 3,000-pair target; "
                            "the H144 document exclusion cost 593 of 902 unit-bearing "
                            "tables. Superseded by build 2 on the coordinator's ruling "
                            "- same construction, superset supply, no quarantine"),
            },
        },
        "seed": SEED, "target_pairs": TARGET_PAIRS, "doc_cap": DOC_CAP,
        "rows": len(df), "pairs": int(n_pairs),
        "label_counts": {str(k): n for k, n in df.group_by("label").len().iter_rows()},
        "families": dict(sorted(families.items(), key=lambda kv: -kv[1]["pairs"])),
        "family_construction": family_report,
        "allocation": allocation,
        "dimension_pairs": {str(k): n for k, n in
                            fam_rows.group_by("dimension").len().iter_rows()},
        "unit_carrier_pairs": {str(k): n for k, n in
                               fam_rows.group_by("carrier").len().iter_rows()},
        "serial_form_pairs": {str(k): n for k, n in
                              fam_rows.group_by("serial_form").len().iter_rows()},
        "template_pairs": {str(k): n for k, n in
                           fam_rows.group_by("claim_template").len().iter_rows()},
        "source_pairs": {str(k): n for k, n in fam_rows.group_by("source").len().iter_rows()},
        "qualified_tables": {
            "corpus_tables": len(raw),
            "unit_bearing_admitted": len(tables),
            "unit_bearing_dropped_evalset": dropped["evalset"],
            "admitted_tables_also_in_h144": dropped["shared_with_h144"],
            "unit_supply_as_built": supply_excl,
            "balanced_ceiling_as_built": {d: dict(sorted(w.items())) for d, w in
                                          sorted(ceil_excl.items())},
            "balanced_ceiling_as_built_total": sum(sum(v2.values()) for v2 in ceil_excl.values()),
            "unbuildable_families": (
                "percent <-> percentage points is not built: 'percentage point(s)' "
                "occurs in one corpus table and can never be a POSITIVE, so the "
                "family cannot be direction-balanced and an unbalanced one would "
                "hand the claim-only probe the rule 'percentage points => label 0'. "
                "'thousand' is excluded on a MEASUREMENT rather than on supply: it "
                "reaches 40 positives once the H144 exclusion lifts, enough for an "
                "80-pair family, but from only FIVE tables whose column names repeat, "
                "and the family measured 0.70 within-pair claim-only accuracy against "
                "the 0.60 bar - the only family ever to breach it. Percent is balanced "
                "against million and billion instead, capped at MAGNITUDE_CAP."),
        },
        "diversity": {
            "documents": len(built_docs),
            "pairs_per_document_mean": round(float(per_doc.mean()), 4),
            "pairs_per_document_max": int(per_doc.max()),
            "pairs_per_document_p50": int(per_doc.median()),
            "distinct_chunks": int(df["chunk"].n_unique()),
            "distinct_claims": int(df["claim"].n_unique()),
        },
        "disjointness": disj,
        "verify": v,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MANIFEST}", flush=True)
    print(json.dumps({"rows": len(df), "pairs": int(n_pairs),
                      "all_bars_pass": v["all_bars_pass"]}, indent=1), flush=True)

    # only the eval-set disjointness is an abort condition; H144 overlap is
    # measured and reported, per the coordinator's build-2 ruling
    bad_disj = (disj["evalset"]["shared_doc_ids"]
                or disj["evalset"]["shared_evidence_chunks"])
    if not v["all_bars_pass"] or bad_disj:
        raise SystemExit("ABORT: a verify bar failed or an eval-set document leaked in")


if __name__ == "__main__":
    main()
