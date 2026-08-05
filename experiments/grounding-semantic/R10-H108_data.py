"""R10-H108 data build - quantitative near-miss register pairs (ZERO GPU).

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 10).
Builds R10-H108_pairs.parquet with four DANN groups:

  quant_feverous - FEVEROUS claims involving tables/numeric reasoning
                   (materialized-evidence mirror KingTechnician/feverous-label-evidence;
                   labels per its card: 0=SUPPORTS, 1=REFUTES, 2=NEI -> ours 1/0/0)
  quant_infotabs - InfoTabS infobox hypotheses (mirror table-benchmark/infotabs;
                   official corpus Apache-2.0, github.com/infotabs/infotabs)
  quant_scitab   - SciTab science-paper table claims (MIT, XinyuanLu00/SciTab)
  quant_corrupt  - deterministic unit/period/scale corruption negatives built from
                   already-POSITIVE numeric claims in TabFact/FEVEROUS/InfoTabS only

SEM-TAB-FACTS dropped at gate: no programmatic distribution (Google Drive only).
Contamination wall: no ConvFinQA / TAT-HQA / MultiHiertt / FinanceBench / FinTabNet /
EDGAR material; sources are Wikipedia tables/infoboxes and open-access paper tables.

Run: uv run python experiments/grounding-semantic/R10-H108_data.py
"""

import io
import json
import pathlib
import random
import re
import urllib.request
import zipfile
from collections import Counter

import polars as pl

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"
OUT = HERE / "R10-H108_pairs.parquet"
CHUNK_MAX = 1500  # one serving window
SEED = 0
CORRUPT_TARGET = 45_000
SCITAB_URL = "https://raw.githubusercontent.com/XinyuanLu00/SciTab/main/dataset/sci_tab.json"

rng = random.Random(SEED)


def feverous():
    """Table/numeric FEVEROUS claims with materialized evidence text."""
    df = pl.read_parquet(
        "hf://datasets/KingTechnician/feverous-label-evidence/data/train-00000-of-00001.parquet"
    )
    keep = df.filter(
        pl.col("challenge").is_in(["Combining Tables and Text", "Numerical Reasoning"])
    )
    rows = []
    for claim, lab, ev in zip(
        keep["claim"].to_list(), keep["label"].to_list(), keep["evidence"].to_list(), strict=True
    ):
        ev = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", ev)  # [[page|text]] -> text
        ev = ev.replace("[SEP]", "\n").strip()
        if len(ev) > CHUNK_MAX or len(ev) < 30:
            continue  # non-localisable in one window, or empty
        rows.append((claim, ev, 1.0 if lab == 0 else 0.0, "quant_feverous"))
    return rows


def infotabs():
    """InfoTabS hypotheses over linearized Wikipedia infoboxes."""
    df = pl.read_parquet(
        "hf://datasets/table-benchmark/infotabs/data/train-00000-of-00001.parquet"
    )
    rows = []
    for tbl, title, hyp, ans in zip(
        df["table"].to_list(), df["table_title"].to_list(), df["question"].to_list(),
        df["answer"].to_list(), strict=True,
    ):
        try:
            t = json.loads(tbl.replace("'", '"')) if isinstance(tbl, str) else tbl
        except (json.JSONDecodeError, TypeError):
            t = None
        if isinstance(t, dict):
            ev = (title or "") + "\n" + "\n".join(
                f"{k} | {', '.join(map(str, v)) if isinstance(v, list) else v}"
                for k, v in t.items()
            )
        else:
            ev = (title or "") + "\n" + str(tbl)
        ev = ev.strip()
        if len(ev) > CHUNK_MAX or len(ev) < 30:
            continue
        rows.append((hyp, ev, 1.0 if ans == "Entailment" else 0.0, "quant_infotabs"))
    return rows


def scitab():
    """SciTab claims over science-paper tables, TabFact-style serialization."""
    raw = json.loads(urllib.request.urlopen(SCITAB_URL, timeout=60).read())
    rows = []
    for x in raw:
        header = " | ".join(x["table_column_names"])
        body = "\n".join(" | ".join(r) for r in x["table_content_values"])
        ev = f"{x['table_caption']}\n{header}\n{body}".replace("[BOLD] ", "")
        if len(ev) > CHUNK_MAX:
            continue
        rows.append((x["claim"], ev, 1.0 if x["label"] == "supports" else 0.0, "quant_scitab"))
    return rows


def tabfact_positives():
    z = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    df = pl.read_parquet(
        io.BytesIO(z.read(next(x for x in z.namelist() if x.endswith("__train.parquet"))))
    ).filter((pl.col("label") == 1) & (pl.col("statement").str.len_chars() > 10))
    return [
        (s, f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")[:CHUNK_MAX])
        for s, cap, tbl in zip(
            df["statement"].to_list(), df["table_caption"].to_list(),
            df["table_text"].to_list(), strict=True,
        )
    ]


# --- corruption families (>= 6, deterministic, label 0 by construction) ---

_SCALE = [("million", "billion"), ("billion", "million"), ("thousand", "million"),
          ("hundred", "thousand")]
_CMP = [("more than", "less than"), ("less than", "more than"), ("higher", "lower"),
        ("lower", "higher"), ("largest", "smallest"), ("smallest", "largest"),
        ("increase", "decrease"), ("decrease", "increase"), ("most", "fewest"),
        ("at least", "at most"), ("at most", "at least"), ("over", "under")]


def _numbers(s):
    return re.findall(r"\d[\d,.]*", s)


def f_scale_word(c):
    for a, b in _SCALE:
        if re.search(rf"\b{a}\b", c):
            return re.sub(rf"\b{a}\b", b, c, count=1), None
    return None


def f_digit_perturb(c):
    nums = _numbers(c)
    if not nums:
        return None
    n = rng.choice(nums)
    digits = [i for i, ch in enumerate(n) if ch.isdigit()]
    if not digits:
        return None
    i = rng.choice(digits)
    old = n[i]
    new = str((int(old) + rng.randint(1, 8)) % 10)
    if new == old or (i == 0 and new == "0"):
        return None
    n2 = n[:i] + new + n[i + 1:]
    return c.replace(n, n2, 1), n2


def f_pct_pp(c):
    if re.search(r"\bpercentage points?\b", c):
        return re.sub(r"\bpercentage points?\b", "percent", c, count=1), None
    if re.search(r"\bpercent\b", c):
        return re.sub(r"\bpercent\b", "percentage points", c, count=1), None
    if "%" in c:
        return c.replace("%", " percentage points", 1), None
    return None


def f_year_shift(c):
    m = re.search(r"\b(19\d\d|20[0-3]\d)\b", c)
    if not m:
        return None
    y = int(m.group(0))
    y2 = y + rng.choice([-3, -2, -1, 1, 2, 3])
    return c[:m.start()] + str(y2) + c[m.end():], str(y2)


def f_comparative_flip(c):
    low = c.lower()
    for a, b in _CMP:
        i = low.find(a)
        if i >= 0:
            return c[:i] + b + c[i + len(a):], None
    return None


def f_magnitude_shift(c):
    nums = [n for n in _numbers(c) if "," not in n]
    if not nums:
        return None
    n = rng.choice(nums)
    if "." in n:
        n2 = n.replace(".", "")  # 4.9 -> 49 (x10)
    else:
        n2 = n + "0"
    return c.replace(n, n2, 1), n2


FAMILIES = [
    ("scale_word", f_scale_word), ("digit_perturb", f_digit_perturb),
    ("pct_pp", f_pct_pp), ("year_shift", f_year_shift),
    ("comparative_flip", f_comparative_flip), ("magnitude_shift", f_magnitude_shift),
]
VALUE_CHECKED = {"digit_perturb", "year_shift", "magnitude_shift"}


def corruptions(sources):
    """sources: list of (claim, chunk, source_tag). Returns corrupt rows + histogram."""
    per_family_cap = int(CORRUPT_TARGET / len(FAMILIES) * 1.5)
    hist, src_hist, rows = Counter(), Counter(), []
    cands = [x for x in sources if re.search(r"\d", x[0])]
    rng.shuffle(cands)
    tabfact_cap = int(CORRUPT_TARGET * 0.6)
    for claim, chunk, src in cands:
        if len(rows) >= CORRUPT_TARGET:
            break
        if src == "tabfact" and src_hist["tabfact"] >= tabfact_cap:
            continue
        fams = FAMILIES[:]
        rng.shuffle(fams)
        for name, fn in fams:
            if hist[name] >= per_family_cap:
                continue
            out = fn(claim)
            if not out:
                continue
            new_claim, new_val = out
            if new_claim == claim:
                continue
            # a corrupted value already present in the evidence could silently
            # re-ground the claim - drop those (registered hard filter)
            if name in VALUE_CHECKED and new_val and new_val in chunk:
                continue
            rows.append((new_claim, chunk, 0.0, "quant_corrupt"))
            hist[name] += 1
            src_hist[src] += 1
            break
    return rows, hist, src_hist


def main():
    groups = {}
    print("fetching FEVEROUS (materialized mirror)...")
    groups["quant_feverous"] = feverous()
    print(f"  kept {len(groups['quant_feverous'])}")
    print("fetching InfoTabS...")
    groups["quant_infotabs"] = infotabs()
    print(f"  kept {len(groups['quant_infotabs'])}")
    print("fetching SciTab...")
    groups["quant_scitab"] = scitab()
    print(f"  kept {len(groups['quant_scitab'])}")

    print("building corruption negatives...")
    sources = [(c, ch, "tabfact") for c, ch in tabfact_positives()]
    sources += [(c, ch, "feverous") for c, ch, y, _ in groups["quant_feverous"] if y == 1.0]
    sources += [(c, ch, "infotabs") for c, ch, y, _ in groups["quant_infotabs"] if y == 1.0]
    corrupt, hist, src_hist = corruptions(sources)
    groups["quant_corrupt"] = corrupt

    all_rows = [r for g in groups.values() for r in g]
    df = pl.DataFrame(
        {"claim": [r[0] for r in all_rows], "chunk": [r[1] for r in all_rows],
         "label": [r[2] for r in all_rows], "tag": [r[3] for r in all_rows]},
        schema={"claim": pl.Utf8, "chunk": pl.Utf8, "label": pl.Float32, "tag": pl.Utf8},
    )
    df.write_parquet(OUT)

    print("\n" + "=" * 88)
    print("R10-H108 PAIRS - counts per group/label")
    print(df.group_by(["tag", "label"]).len().sort(["tag", "label"]))
    print(f"\ncorruption family histogram: {dict(hist)}")
    print(f"corruption source histogram: {dict(src_hist)}")
    print(f"\nTOTAL {len(df)} pairs -> {OUT}")

    print("\nQA samples (10 per group/label):")
    for tag in sorted(groups):
        for lab in (1.0, 0.0):
            sub = df.filter((pl.col("tag") == tag) & (pl.col("label") == lab))
            if not len(sub):
                continue
            print(f"\n--- {tag} label={lab} (n={len(sub)}) ---")
            for r in sub.sample(min(10, len(sub)), seed=SEED).iter_rows():
                print(f"  CLAIM: {r[0][:140]}")
                print(f"  CHUNK: {r[1][:110]}...")


if __name__ == "__main__":
    main()
