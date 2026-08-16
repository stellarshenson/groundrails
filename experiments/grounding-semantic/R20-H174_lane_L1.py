"""R20-H174 LANE L1 `frame_reject` - vacuous_claim_reject, build + verify, CPU.

Registered in docs/experiments/semantic-grounding-experiments.md, block "R20-H174
HAGRID/EMANUAL PORTFOLIO ARM": "L1 vacuous_claim_reject (~5-10k rows, rule-
generated, WITH label-1 frame+content rows so MIL learns frame-NEUTRAL,
protecting emanual's grounded recap items)".

THE DEFECT THE LANE TEACHES AGAINST
-----------------------------------
Four hagrid responses are the string "Based on the given context ," and nothing
else.  All four are labelled unsupported, all four score POSITIVE (+1.41 to
+2.75 on flagship draw 1) at token containment 0.000, and together they carry
21.2% of hagrid's misrank mass; ranking them last lifts the subset +0.076
(R19-H162_hagrid_mechanisms.json, `frame_drop_control`).  The training mix
cannot teach otherwise: the clean public 685,670 rows contain ZERO discourse
frames against 4.1% in the hagrid sample (`training_mix_wrapper_census`).

CONSTRUCTION - the frame is a CONSTANT, the content is the variable
------------------------------------------------------------------
Every pair is two claims over the SAME evidence chunk:

  label 1   a genuine supported claim (MiniCheck label-1 / VitaminC SUPPORTS)
  label 0   a claim assembled ONLY from a contentless inventory - provenance
            frames, discourse fillers, citation marks, bare reference lines

and each pair carries a `framed` coin that BOTH legs obey:

  framed    both claims open with the byte-identical provenance frame head, so
            the frame prefix is present on the supported leg exactly as often as
            on the vacuous one.  This is the registration's frame-NEUTRAL
            requirement: MIL must learn "a frame decides nothing", never "a
            frame means unsupported", which would sink emanual's grounded recap
            sentences (24 recap-ending items already read 0.55 = chance,
            R19-H162_procedural_mechanisms.json)
  unframed  neither claim carries a frame; the negative is built from markers,
            reference lines and non-frame discourse fillers instead

THE LENGTH SHORTCUT, closed by construction
-------------------------------------------
A frame-only claim is short and a frame+content claim is long, so an unmatched
lane teaches "short implies unsupported" - the exact confound the probe design
in `R19-H162_hagrid_mechanisms.json` requires a control against.  The negative
is therefore ASSEMBLED TO THE POSITIVE'S LENGTH: contentless fragments are
appended (and, matching the hagrid artifact, sometimes truncated mid-frame)
until the two claims sit within LEN_TOL characters of one another.

The negative's whole vocabulary is a closed inventory written in this file, so
"no negative claim carries verifiable content" is checked mechanically rather
than asserted: `negative_contentless_audit` fails the build if any negative
claim contains a token the inventory does not.

Sources: MiniCheck (MIT) and VitaminC train (CC-BY-SA-3.0) supply the evidence
chunks and the label-1 claims.  No new corpus, no new licence.

Run:  uv run python experiments/grounding-semantic/R20-H174_lane_L1.py
"""

import collections
import importlib.util as _ilu
import json
from pathlib import Path
import random
import re
import sys

import polars as pl

_spec = _ilu.spec_from_file_location("h174common", Path(__file__).parent / "R20-H174_lane_common.py")
C = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(C)

HERE = Path(__file__).parent
OUT = HERE / "R20-H174_lane_L1.parquet"
MANIFEST = HERE / "R20-H174_lane_L1_manifest.json"

SEED = 1174
TAG = "frame_reject"
N_PAIRS = 4_000                 # 8,000 rows, inside the registered 5-10k band
MINICHECK_SHARE = 0.60
FRAME_SHARE = 0.65              # share of pairs whose BOTH legs carry a frame
DOC_CAP = 2
LEN_TOL = 12                    # chars; negative assembled to the positive's length
MAX_ASSEMBLY_TRIES = 40
MIN_POS_CHARS, MAX_POS_CHARS = 25, 460

# The banked detector, transcribed from R19-H162_hagrid_mechanisms.py, used to
# MEASURE the lane rather than to build it.
RE_FRAME_BANKED = re.compile(
    r"^\s*(Based on|According to) the (given|provided) contexts?|"
    r"^\s*The (given|provided) contexts? (mentions|states)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# the contentless inventory - the ONLY vocabulary a negative claim may use
# --------------------------------------------------------------------------- #
FRAME_HEADS = [
    "Based on the given context", "Based on the given contexts",
    "Based on the provided context", "Based on the provided contexts",
    "Based on the provided passages", "Based on the context provided",
    "Based on the above passages", "Based on the given documents",
    "According to the given context", "According to the given contexts",
    "According to the provided context", "According to the provided contexts",
    "According to the passages provided", "According to the documents provided",
    "According to the given references", "According to the above sources",
    "The given context states", "The provided context states",
    "The given contexts mention", "The provided contexts mention",
    "The given passages indicate", "The provided documents show",
    "Drawing on the provided sources", "Referring to the given references",
    "In light of the provided context", "As stated in the given passages",
    "Per the provided documents", "From the given excerpts",
]

FRAME_CONTINUATIONS = [
    "the following can be stated", "the answer is as follows",
    "here is what applies", "we can conclude the following",
    "I can provide the following information", "this can be summarised as follows",
    "the relevant details are given below", "the response is provided below",
    "the following applies", "an answer can be given",
]

FRAME_DISCOURSE = [
    "See the passages above for details.",
    "Further information appears in the references.",
    "This is discussed in the material provided.",
    "The above summarises the relevant information.",
    "Additional detail is available in the sources.",
    "Refer to the excerpts provided.",
    "No further elaboration is given here.",
    "The remainder follows from the context.",
    "The provided material is summarised above.",
]

MARKERS = [
    "( Context )", "[ Context ]", "( context )", "[Context]", "[1]", "[2]",
    "[3]", "[1, 2]", "[2, 3]", "[1, 2, 3]", "[ 12 ]", "[4, 5]", "(cf.)",
    "( ... )", "...", "[citation]", "[ref]", "N/A", "[ - ]",
]

REFLINES = [
    "Available: https://www.example.org/a/b",
    "Retrieved from https://example.com/page/7",
    "Source: https://www.example.org/index.html",
    "See https://example.com/reference for details.",
    "Available: http://example.net/doc/12",
    "Retrieved from https://www.example.org/archive",
    "\"\", Available: http://example.net/doc",
]

NOFRAME_DISCOURSE = [
    "See above.", "See below for details.", "Details follow.",
    "This is elaborated further on.", "Further reading is listed at the end.",
    "The relevant material is cited above.", "As noted earlier.",
    "More information can be found in the references listed.",
    "The remainder is omitted here.", "Nothing further is added.",
]

TAILS = [",", " ,", ".", " .", ":", "", ";"]

INVENTORY = (FRAME_HEADS + FRAME_CONTINUATIONS + FRAME_DISCOURSE + MARKERS
             + REFLINES + NOFRAME_DISCOURSE + TAILS)
INVENTORY_VOCAB = {t for s in INVENTORY for t in C.tokens(s)}


def _jitter(text, rng):
    """Casing / spacing variation of the kind the hagrid artifacts carry."""
    r = rng.random()
    if r < 0.06:
        text = text[0].lower() + text[1:]
    elif r < 0.09:
        text = text.upper()
    if rng.random() < 0.08:
        text = text.replace(" ", "  ", 1)
    return text


def _truncate_words(text, target, rng):
    """Cut at a word boundary - the hagrid artifact 'Based on the given context ,'
    is itself a truncated frame."""
    if len(text) <= target:
        return text
    out = []
    for w in text.split(" "):
        cand = " ".join(out + [w])
        if len(cand) > target:
            break
        out.append(w)
    cut = " ".join(out) if out else text[:target]
    if rng.random() < 0.5:
        cut += rng.choice([" ,", ",", " .", "..."])
    return cut


def build_negative(rng, framed, prefix, target):
    """Assemble a contentless claim of length ~= `target`.

    On a framed pair `prefix` is the pair's SHARED frame - byte-identical to the
    positive's opening - and the assembly may never cut into it, so frame
    presence is exactly equal on the two legs and carries zero label
    information."""
    pools = ([FRAME_CONTINUATIONS, FRAME_DISCOURSE, MARKERS]
             if framed else [NOFRAME_DISCOURSE, MARKERS, REFLINES])
    floor = len(prefix)
    best = None
    for _ in range(MAX_ASSEMBLY_TRIES):
        text = prefix if framed else rng.choice(pools[rng.randrange(len(pools))])
        guard = 0
        while len(text) < target - LEN_TOL and guard < 8:
            frag = rng.choice(pools[rng.randrange(len(pools))])
            text = f"{text} {frag}".strip()
            guard += 1
        if len(text) > max(target + LEN_TOL, floor):
            text = _truncate_words(text, max(target + LEN_TOL, floor), rng)
        if not framed:
            text = _jitter(text, rng)
        d = abs(len(text) - target)
        if best is None or d < best[0]:
            best = (d, text)
        if d <= LEN_TOL:
            break
    return best[1], best[0]


# --------------------------------------------------------------------------- #
# supply
# --------------------------------------------------------------------------- #
def supply(rng):
    """(claim, chunk, doc_id, source) rows whose claim is genuinely supported."""
    mc = C.minicheck().filter(pl.col("label") == 1)
    rows_mc = [
        {"claim": c, "chunk": d, "doc_id": i, "source": "minicheck"}
        for c, d, i in mc.select(["claim", "doc", "doc_id"]).iter_rows()
    ]

    vc = C.vitaminc("train")
    vc_pos = vc.filter(pl.col("label") == 1)
    pages = collections.defaultdict(list)
    for page, ev in vc.select(["page", "evidence"]).unique().iter_rows():
        pages[page].append(ev)
    rows_vc = []
    for claim, ev, page, did in vc_pos.select(
            ["claim", "evidence", "page", "doc_id"]).iter_rows():
        rows_vc.append({"claim": claim, "chunk": None, "evidence": ev,
                        "page": page, "doc_id": did, "source": "vitaminc"})
    rng.shuffle(rows_mc)
    rng.shuffle(rows_vc)
    return rows_mc, rows_vc, pages


def already_built():
    """Idempotence: a lane whose parquet and manifest are on disk and whose own
    verify block passed is not rebuilt.  `--force` overrides."""
    if "--force" in sys.argv or not (OUT.exists() and MANIFEST.exists()):
        return False
    try:
        man = json.loads(MANIFEST.read_text())
        rows = pl.read_parquet(OUT).height
    except Exception:
        return False
    if man.get("verify", {}).get("all_bars_pass") and rows == man.get("rows"):
        print(f"{OUT.name}: {rows} rows already built and passing - skipping "
              f"(pass --force to rebuild)", flush=True)
        return True
    return False


def main():
    if already_built():
        return
    rng = random.Random(SEED)
    print(f"=== R20-H174 lane L1 ({TAG}) seed {SEED}", flush=True)
    rows_mc, rows_vc, pages = supply(rng)
    print(f"supply: minicheck label-1 {len(rows_mc)}, vitaminc SUPPORTS "
          f"{len(rows_vc)} over {len(pages)} pages", flush=True)

    want_mc = int(round(N_PAIRS * MINICHECK_SHARE))
    plan = ([("minicheck", r) for r in rows_mc] , [("vitaminc", r) for r in rows_vc])
    per_doc = collections.Counter()
    out, pid, len_err = [], 0, []
    n_framed = 0

    for src_rows, want in ((plan[0], want_mc), (plan[1], N_PAIRS - want_mc)):
        got = 0
        for _src, r in src_rows:
            if got >= want:
                break
            if per_doc[r["doc_id"]] >= DOC_CAP:
                continue
            claim = r["claim"].strip()
            if not (MIN_POS_CHARS <= len(claim) <= MAX_POS_CHARS):
                continue
            if r["source"] == "vitaminc":
                chunk = C.vitaminc_passage_for(pages[r["page"]], r["evidence"], rng)
            else:
                chunk = r["chunk"]
            if len(chunk) < 200:
                continue
            framed = rng.random() < FRAME_SHARE
            head = rng.choice(FRAME_HEADS) if framed else ""
            # the frame is drawn ONCE and used byte-identically on both legs
            prefix = _jitter(head + rng.choice(TAILS), rng) if framed else ""
            pos_claim = f"{prefix} {claim}".strip() if framed else claim
            neg_claim, err = build_negative(rng, framed, prefix, len(pos_claim))
            if not neg_claim.strip():
                continue
            len_err.append(err)
            n_framed += int(framed)
            base = {"chunk": chunk, "doc_id": r["doc_id"], "source": r["source"],
                    "tag": TAG, "framed": framed,
                    "neg_family": "vacuous_frame" if framed else "vacuous_marker",
                    "pos_family": "frame_plus_content" if framed else "bare_content",
                    "frame_head": head, "frame_prefix": prefix,
                    "genuine_claim": claim}
            out.append(dict(pair_id=pid, label=1, claim=pos_claim, **base))
            out.append(dict(pair_id=pid, label=0, claim=neg_claim, **base))
            per_doc[r["doc_id"]] += 1
            pid += 1
            got += 1
        print(f"  built {got} pairs from {_src}", flush=True)

    df = C.dedupe(pl.DataFrame(out))
    df.write_parquet(OUT)
    print(f"{df.height} rows / {df['pair_id'].n_unique()} pairs -> {OUT.name}", flush=True)

    res = verify(df, rng)
    man = build_manifest(df, res, len_err, n_framed)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("rows", "pairs", "label_balance", "families", "sources",
                       "window_census", "verify")}, indent=2), flush=True)
    ok = res["all_bars_pass"]
    print(f"=== R20-H174 LANE L1 {'BUILT' if ok else 'FAILED BARS'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(df, rng):
    out = {}
    out["pair_integrity"] = C.pair_integrity(df)

    # --- the registration's frame-NEUTRAL requirement, measured with the BANKED
    # detector: a frame must be as frequent on the supported leg as on the
    # vacuous one, so frame presence carries no label information.
    hit = [bool(RE_FRAME_BANKED.search(c)) for c in df["claim"].to_list()]
    y = df["label"].to_list()
    pos_rate = sum(h for h, l in zip(hit, y) if l == 1) / max(sum(y), 1)
    neg_rate = sum(h for h, l in zip(hit, y) if l == 0) / max(len(y) - sum(y), 1)
    fa = C.auroc(y, [float(h) for h in hit])
    out["frame_presence_neutrality"] = {
        "detector": "R19-H162 RE_FRAME (banked), applied to the claim",
        "label1_frame_rate": round(pos_rate, 6),
        "label0_frame_rate": round(neg_rate, 6),
        "rate_difference": round(abs(pos_rate - neg_rate), 6),
        "frame_presence_auroc": round(fa, 4),
        "bar": "rate difference <= 0.005 and AUROC in [0.45, 0.55]",
        "pass": bool(abs(pos_rate - neg_rate) <= 0.005 and abs(fa - 0.5) <= 0.05)}

    # --- no negative claim may carry verifiable content.  Mechanical: every
    # token of every negative claim must come from this file's closed inventory.
    errs = []
    for r in df.filter(pl.col("label") == 0).iter_rows(named=True):
        stray = sorted(set(C.tokens(r["claim"])) - INVENTORY_VOCAB)
        if stray:
            errs.append({"pair_id": r["pair_id"], "stray_tokens": stray[:5]})
    out["negative_contentless_audit"] = {
        "negatives": int(df.filter(pl.col("label") == 0).height),
        "inventory_vocabulary": len(INVENTORY_VOCAB),
        "rows_with_out_of_inventory_token": len(errs),
        "bar": "0 rows - the negative leg is closed-vocabulary by construction",
        "pass": not errs, "examples": errs[:5]}

    # --- surface parity: length is the confound the lane is built to avoid
    out["surface_parity"] = C.surface_parity(
        df, report_only=("claim_chunk_containment",))

    # --- claim-only probe.  REPORT ONLY, and the reason is registered: L1
    # deliberately teaches a CLAIM-SIDE rule (a claim asserting nothing is not
    # supported by anything), so a claim-only probe MUST separate.  What must
    # not separate is the frame itself, which the neutrality block above bars.
    probe, score = C.claim_only_probe(df["claim"].to_list(), y,
                                      df["doc_id"].to_list(), rng)
    out["claim_only_tfidf_auroc_report_only"] = {
        "value": round(probe, 4),
        "within_pair": C.within_pair_accuracy(df, score, by="neg_family"),
        "note": "no bar - the lane's content is a claim-side rule; the binding "
                "bar is frame_presence_neutrality"}

    # --- the positive leg really is supported by its chunk (dataset property,
    # measured not assumed) and the negative leg is not
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)
    out["containment_report_only"] = {
        "label1_mean": round(sum(C.containment(c, k) for c, k in
                                 zip(pos["genuine_claim"], pos["chunk"])) / pos.height, 4),
        "label0_mean": round(sum(C.containment(c, k) for c, k in
                                 zip(neg["claim"], neg["chunk"])) / neg.height, 4),
        "note": "label-1 figure is the GENUINE claim against its chunk; label-0 "
                "residue is inventory function words, not content"}

    out["all_bars_pass"] = all(
        out[k]["pass"] for k in ("pair_integrity", "frame_presence_neutrality",
                                 "negative_contentless_audit", "surface_parity"))
    return out


def build_manifest(df, res, len_err, n_framed):
    y = df["label"].to_list()
    return dict(
        experiment="R20-H174 lane L1 - vacuous_claim_reject (frame_reject)",
        registration="docs/experiments/semantic-grounding-experiments.md, "
                     "block 'R20-H174 HAGRID/EMANUAL PORTFOLIO ARM'",
        tag=TAG,
        dann_group=TAG,
        mix_loader="drop-in for R18-H150_arm_run.make_build_mix - columns "
                   "claim / chunk / label / pair_id / neg_family; chunk is read "
                   "UNTRUNCATED and windowed 1500/750 by the loader",
        seed=SEED,
        rows=df.height,
        pairs=int(df["pair_id"].n_unique()),
        documents=int(df["doc_id"].n_unique()),
        label_balance={"label_1": int(sum(y)), "label_0": int(len(y) - sum(y)),
                       "positive_share": round(sum(y) / len(y), 4)},
        families={k: v for k, v in df.group_by("neg_family").len().iter_rows()},
        pos_families={k: v for k, v in df.group_by("pos_family").len().iter_rows()},
        framed_pairs=n_framed,
        sources={k: C.SOURCES[k] for k in ("minicheck", "vitaminc")},
        source_rows={k: v for k, v in df.group_by("source").len().iter_rows()},
        generator=dict(
            n_pairs_target=N_PAIRS, minicheck_share=MINICHECK_SHARE,
            frame_share=FRAME_SHARE, doc_cap=DOC_CAP, len_tol=LEN_TOL,
            min_pos_chars=MIN_POS_CHARS, max_pos_chars=MAX_POS_CHARS,
            inventory_sizes={"frame_heads": len(FRAME_HEADS),
                             "frame_continuations": len(FRAME_CONTINUATIONS),
                             "frame_discourse": len(FRAME_DISCOURSE),
                             "markers": len(MARKERS), "reflines": len(REFLINES),
                             "noframe_discourse": len(NOFRAME_DISCOURSE),
                             "tails": len(TAILS)},
            length_match_error={"mean": round(sum(len_err) / max(len(len_err), 1), 3),
                                "max": max(len_err) if len_err else 0}),
        char_stats=dict(
            claim_label1=C.char_stats(df.filter(pl.col("label") == 1)["claim"].to_list()),
            claim_label0=C.char_stats(df.filter(pl.col("label") == 0)["claim"].to_list()),
            chunk=C.char_stats(df["chunk"].to_list())),
        diversity=dict(distinct_claims=int(df["claim"].n_unique()),
                       distinct_chunks=int(df["chunk"].n_unique()),
                       distinct_frame_heads=int(df["frame_head"].n_unique())),
        window_census=C.window_census(df["chunk"].to_list()),
        verify=res)


if __name__ == "__main__":
    main()
