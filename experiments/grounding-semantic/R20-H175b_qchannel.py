"""R20-H175b QUESTION CHANNEL - the arm's single intervention, and its guards.

Registered in docs/experiments/semantic-grounding-experiments.md, blocks
"R20-H175b QUESTION CONDITIONING (measurement only)" (2026-08-16 ~23:30),
"R20-H175b STAGE 0 COMPLETE" (2026-08-17 ~00:10, MANDATORY loader assertion) and
"QUEUE AMENDMENT Q1" (2026-08-17 ~06:05, draw 1 only).

THE INTERVENTION, and it is the ONLY one. The cross-encoder's text-A side is the
claim string. This module composes an OPTIONAL question prefix into it:

    question present ->  "<question[:256]> [SEP] <claim>"
    question absent  ->  "<claim>"                      (byte-identical to flagship)

Nothing else changes: the evidence side, the windowing, the objective, the DANN
groups, the schedule and the seeding are the flagship's. Because every training
and read path takes its text-A from the same `claims` list
(`R16-H142_G1_arm.encode_batch`, `R19-H160_split_exec.split_train_step`,
`R15_gate_common.score`), composing at mix-assembly time is the whole change and
no trainer or reader is edited.

The question is a PREFIX, so tokenizer truncation can never remove it: HF's
`longest_first` strategy drops tokens from the END of whichever side is longer,
and a 1,500-char evidence window is longer than a 256-char question plus a claim
for all but a handful of rows - and even there the drop comes off the claim's
tail, never off the leading question. The 256-char cap bounds the growth of the
text-A side so the window keeps its budget.

WHERE THE QUESTIONS COME FROM. `R10-H108_lane.public_train()` returns
(claims, chunks, y, tags) and drops every question field it reads. Rather than
edit that banked loader, `clean_questions()` REPLAYS its source order and emits
the parallel question list, and `assert_alignment()` proves the replay is
row-for-row identical by comparing the replayed claim list against the loader's
own output element by element. A single mis-ordered filter is a hard abort, not
a silent misalignment.

Three corpora carry a question field, exactly as the registration enumerates:

    ragtruth_en   `query`     15,090 rows  (QA 5,034 real questions; Summary and
                              Data2txt carry a task instruction - counted by the
                              registration's own 13-14% arithmetic, and reported
                              separately by `coverage()`)
    halueval-qa   `question`  20,000 rows  (both legs of each item)
    psiloqa       `question`  the filtered train split

    + R20-H175b_qlane   `question`  17,972 rows - the contrast lane, where the
                        label lives ENTIRELY in the question

THE MANDATORY LOADER ASSERTION (stage-0 disposition 4, registered before this
launch). Both rows of a qlane pair carry the same claim and the same chunk. If
the question is dropped the lane becomes label-contradictory duplicate rows -
pure label noise at ~2.4% of the mix - and the arm would measure nothing while
looking healthy. `assert_lane_channel()` therefore hard-aborts unless a question
is composed for EVERY row of the lane, and `assert_pairs_differ()` proves
positively that the two legs of a pair produce different composed strings and
different TOKENIZED inputs. Both run inside the arm's `build_mix`, before a card
is touched.
"""

import importlib.util
import io
import pathlib
import zipfile

import polars as pl

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"

Q_MAX_CHARS = 256
Q_SEP = " [SEP] "

LANE_FILE = "R20-H175b_qlane.parquet"
LANE_GROUP = "qrel_contrast"
LANE_ROWS = 17_972
LANE_PAIRS = 8_986

# How many qlane pairs the positive tokenization check samples. Every pair is
# checked at the string level; this is the tokenizer-level sample.
TOKEN_CHECK_PAIRS = 400


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compose(question, claim):
    """The intervention. Empty question -> the flagship's own claim string."""
    q = (question or "").strip()
    if not q:
        return claim
    return q[:Q_MAX_CHARS] + Q_SEP + claim


# --------------------------------------------------------------------------- #
# the clean mix's question channel - a replay of public_train's source order
# --------------------------------------------------------------------------- #
def clean_questions():
    """(questions, claims_replay, segments) for the 685,670-row clean mix.

    `claims_replay` exists ONLY to be compared against the banked loader's own
    claim list - it is the alignment proof, not data. `segments` records each
    source's row span so coverage can be reported per corpus.
    """
    questions, claims, segments = [], [], []

    def seg(name, qs, cs):
        start = len(questions)
        questions.extend(qs)
        claims.extend(cs)
        segments.append({"source": name, "start": start, "rows": len(qs),
                         "with_question": sum(1 for q in qs if (q or "").strip())})

    # 1. RAGTruth EN - `query`
    z = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    n = next(x for x in z.namelist() if x.endswith("__train.parquet"))
    df = pl.read_parquet(io.BytesIO(z.read(n))).filter(
        pl.col("context").str.len_chars() > 50)
    seg("ragtruth_en", df["query"].to_list(), df["output"].to_list())

    # 2. RAGTruth translations - the `prompt` IS the evidence; no question field
    zt = zipfile.ZipFile(DATA / "dataset-ragtruth-translated.zip")
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        nm = next(x for x in zt.namelist()
                  if f"ragtruth-{lg}-" in x and x.endswith("__train.parquet"))
        d = pl.read_parquet(io.BytesIO(zt.read(nm))).filter(
            pl.col("prompt").str.len_chars() > 50)
        seg(f"ragtruth_{lg}", [""] * d.height, d["answer"].to_list())

    # 3. HaluEval - the QA half carries `question`, the summarization half does not
    zh = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    for cfg, ev_col, pos_col, neg_col in (
        ("qa", "knowledge", "right_answer", "hallucinated_answer"),
        ("summarization", "document", "right_summary", "hallucinated_summary"),
    ):
        hits = [x for x in zh.namelist() if f"__{cfg}__" in x]
        if not hits:
            continue
        d = pl.read_parquet(io.BytesIO(zh.read(hits[0])))
        if not {ev_col, pos_col, neg_col} <= set(d.columns):
            continue
        qs, cs = [], []
        qcol = d["question"].to_list() if "question" in d.columns else [""] * d.height
        for q, pos, neg in zip(qcol, d[pos_col].to_list(), d[neg_col].to_list(),
                               strict=True):
            qs += [q, q]          # the loader emits the two legs back to back
            cs += [pos, neg]
        seg(f"halueval_{cfg}", qs, cs)

    # 4. PsiloQA - `question`
    zp = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    dp = pl.read_parquet(io.BytesIO(zp.read(
        next(x for x in zp.namelist() if x.endswith("__train.parquet"))))).filter(
        (pl.col("wiki_passage").str.len_chars() > 50)
        & (pl.col("llm_answer").str.len_chars() > 10))
    seg("psiloqa", dp["question"].to_list(), dp["llm_answer"].to_list())

    # 5. VitaminC - no question field
    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    dv = pl.read_parquet(io.BytesIO(zv.read(
        next(x for x in zv.namelist() if x.endswith("__train.parquet")))))
    cl_col = next(c for c in ("claim", "output", "answer") if c in dv.columns)
    seg("vitaminc", [""] * dv.height, dv[cl_col].to_list())

    # 6. TabFact - no question field
    zt2 = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    dt = pl.read_parquet(io.BytesIO(zt2.read(
        next(x for x in zt2.namelist() if x.endswith("__train.parquet"))))).filter(
        pl.col("statement").str.len_chars() > 10)
    seg("tabfact", [""] * dt.height, dt["statement"].to_list())

    return questions, claims, segments


def assert_alignment(replay_claims, loader_claims):
    """Row-for-row proof that the replayed question channel lines up with the
    banked loader's output. Any drift aborts before a card is touched."""
    if len(replay_claims) != len(loader_claims):
        raise SystemExit(
            f"QUESTION-CHANNEL ABORT: replay has {len(replay_claims)} rows, the "
            f"banked loader {len(loader_claims)} - the source order changed")
    bad = [i for i, (a, b) in enumerate(zip(replay_claims, loader_claims, strict=True))
           if a != b]
    if bad:
        raise SystemExit(
            f"QUESTION-CHANNEL ABORT: {len(bad)} of {len(loader_claims)} replayed "
            f"claims differ from the banked loader's (first at row {bad[0]}) - the "
            "question channel is NOT aligned with the mix; do not train")
    return True


# --------------------------------------------------------------------------- #
# the MANDATORY loader assertion (stage-0 disposition 4)
# --------------------------------------------------------------------------- #
def assert_lane_channel(questions, composed, claims, tags):
    """Hard abort unless a question is composed for EVERY row of the contrast
    lane. Registered BEFORE this launch, on the R20-H174 census-rebind precedent.
    """
    idx = [i for i, t in enumerate(tags) if t == LANE_GROUP]
    if len(idx) != LANE_ROWS:
        raise SystemExit(
            f"QLANE ABORT: {len(idx)} rows tagged {LANE_GROUP} in the mix, "
            f"expected {LANE_ROWS} - the lane did not load")
    empty = [i for i in idx if not (questions[i] or "").strip()]
    if empty:
        raise SystemExit(
            f"QLANE ABORT: {len(empty)} of {len(idx)} lane rows carry NO question. "
            "The lane's two legs share claim and chunk and differ only in the "
            "question, so without it the lane is label-contradictory duplicate "
            "rows - pure label noise. Registered hard abort; do not train")
    unchanged = [i for i in idx if composed[i] == claims[i]]
    if unchanged:
        raise SystemExit(
            f"QLANE ABORT: {len(unchanged)} of {len(idx)} lane rows have a composed "
            "text-A identical to the bare claim - the intervention is a no-op on "
            "the lane; do not train")
    return {"lane_rows": len(idx), "with_question": len(idx),
            "coverage": 1.0, "first_row": idx[0], "last_row": idx[-1]}


def assert_pairs_differ(composed, tags, tokenizer=None, chunk_of=None,
                        n_sample=TOKEN_CHECK_PAIRS):
    """Positive verification: the two legs of a contrast pair must NOT be the
    same model input. Checked at the STRING level for all 8,986 pairs and, when a
    tokenizer is supplied, at the TOKEN level for a sample.

    `chunk_of(i)` returns the evidence window that row i's leg is scored against;
    the token check pairs each composed claim with its own first window, which is
    the same string for both legs by the lane's construction.
    """
    idx = [i for i, t in enumerate(tags) if t == LANE_GROUP]
    lane = pl.read_parquet(HERE / LANE_FILE, columns=["pair_id"])
    if lane.height != len(idx):
        raise SystemExit(
            f"QLANE ABORT: parquet {lane.height} rows vs {len(idx)} tagged rows")
    pid = lane["pair_id"].to_numpy()

    by_pair = {}
    for off, p in enumerate(pid):
        by_pair.setdefault(int(p), []).append(idx[off])
    if len(by_pair) != LANE_PAIRS or any(len(v) != 2 for v in by_pair.values()):
        raise SystemExit(
            f"QLANE ABORT: {len(by_pair)} pairs, not {LANE_PAIRS} two-row pairs")

    same = [p for p, (a, b) in by_pair.items() if composed[a] == composed[b]]
    if same:
        raise SystemExit(
            f"QLANE ABORT: {len(same)} of {LANE_PAIRS} pairs produce an IDENTICAL "
            "composed text-A on both legs - the question is not reaching the "
            "model input. This is the R20-H175a failure mode; do not train")

    report = {"pairs": len(by_pair), "pairs_string_identical": 0}
    if tokenizer is not None:
        pairs = sorted(by_pair)[:n_sample]
        a_txt = [composed[by_pair[p][0]] for p in pairs]
        b_txt = [composed[by_pair[p][1]] for p in pairs]
        w = [chunk_of(by_pair[p][0]) for p in pairs]
        ea = tokenizer(a_txt, w, truncation=True, max_length=512)["input_ids"]
        eb = tokenizer(b_txt, w, truncation=True, max_length=512)["input_ids"]
        ident = sum(1 for x, yv in zip(ea, eb, strict=True) if x == yv)
        if ident:
            raise SystemExit(
                f"QLANE ABORT: {ident} of {len(pairs)} sampled pairs TOKENIZE "
                "identically on both legs - the arm is void; do not train")
        report.update({"pairs_token_checked": len(pairs),
                       "pairs_token_identical": 0,
                       "mean_token_len_leg_a": round(
                           sum(len(x) for x in ea) / len(ea), 1)})
    return report


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def coverage(questions, tags):
    """Question-coverage fraction of the built mix, overall and per DANN group."""
    n = len(questions)
    have = [bool((q or "").strip()) for q in questions]
    per_group = {}
    for g in sorted(set(tags)):
        rows = [i for i, t in enumerate(tags) if t == g]
        k = sum(have[i] for i in rows)
        per_group[g] = {"rows": len(rows), "with_question": k,
                        "coverage": round(k / len(rows), 4)}
    return {"mix_rows": n, "rows_with_question": sum(have),
            "coverage": round(sum(have) / n, 4), "per_group": per_group}
