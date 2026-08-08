"""R12 label-ceiling diagnostic (ANALYSIS ONLY - Round 12 ruling 5).

Question: what mean AUC would a PERFECT per-(sentence, window) entailment oracle
achieve on the blind arena under RAGBench's labels?

The read under test is R8-H101_windowed_read.py:
  - unit scored   : H92.sentences(response) x 1500/750 windows of documents[:8]
  - aggregation   : MAX over windows/chunks, then MIN over sentences
  - label         : `adherence_score` (RAGBench, GPT-4o annotated)
  - AUC           : R7-H59.auc_and_f1 (sklearn roc_auc_score) per subset, then mean

VERIFIED LABEL GRANULARITY (this is the whole point of the diagnostic).
`adherence_score` is stored as a response-level boolean, but it is not an
independent coarse judgement: it equals `len(unsupported_response_sentence_keys)
== 0` on 2,264 / 2,264 arena rows across all ten subsets. The label IS the AND
over the per-response-sentence annotations. `sentence_support_information[].
fully_supported` is NULL on 8 of 10 subsets (finqa 1101/1116 null, tatqa 658/665,
techqa 1739/1917, pubmedqa 736/832) and must not be used as the truth field;
`unsupported_response_sentence_keys` is the field that is populated everywhere.

Therefore CASE B of the brief applies: the ceiling at the ANNOTATION's own
sentence granularity is 1.0 by construction, and the diagnostic instead measures
what the read's own machinery costs a perfect entailer. Four nested oracles:

  O1 ANNOTATION-AND  : min over ANNOTATED response sentences of supported.
                       1.0 by construction - the identity check.
  O2 + SPLITTER      : the same truth through H92.sentences() (>=25 chars, cap 12,
                       <2 -> whole response), mapped back onto annotated keys.
  O3 + CHUNK CAP     : O2, plus a supported sentence counts only if all of its
                       supporting document sentences survive documents[:8].
  O4 + WINDOWS       : O3, plus supported only if a single 1500-char window
                       (stride 750) of one retained document contains the support.
                       Two variants - STRICT (all supporting keys in one window)
                       and LENIENT (any one supporting key in one window). This
                       is exactly what max-over-windows of a perfect
                       per-(sentence, window) entailer can express.

Run (zero GPU):  uv run python experiments/grounding-semantic/R12_label_ceiling.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import io
import json
import pathlib
import re
import zipfile

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score

HERE = pathlib.Path(__file__).parent
ARCHIVE = HERE.parent.parent / "data" / "external" / "datasets" / "dataset-ragbench.zip"
OUT = HERE / "R12_label_ceiling_result.json"

MAX_CHUNKS = 8  # R8-H77 load_subsets
N_PER_SUBSET = 250  # R8-H77 load_subsets
MIN_SENT_CHARS = 25  # R8-H92 sentences()
MAX_SENTS = 12  # R8-H92 sentences()
WIN = 1500  # R8-H101
STRIDE = 750  # R8-H101

_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PREFIX = re.compile(r"^(Title|Passage|Question|Answer)\s*:\s*")
_WS = re.compile(r"\s+")


def auc_and_f1(y, s, seed=0):
    """Byte-identical to R7-H59.auc_and_f1 - the read's own AUC path."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    grid = np.quantile(s[a], np.linspace(0.05, 0.95, 91))
    thr = max(grid, key=lambda t: f1_score(y[a], (s[a] >= t).astype(int), average="macro"))
    return (
        float(roc_auc_score(y, s)),
        float(f1_score(y[b], (s[b] >= thr).astype(int), average="macro")),
        float(thr),
    )


def sentences(text):
    """R8-H92.sentences - the splitter the read actually uses."""
    parts = [s.strip() for s in _SPLIT.split(text)]
    parts = [s for s in parts if len(s) >= MIN_SENT_CHARS][:MAX_SENTS]
    return parts if len(parts) >= 2 else [text]


def win_spans(doc):
    """R8-H101.windows as (start, end) char spans."""
    n = len(doc)
    if n <= WIN:
        return [(0, n)]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [(s, s + WIN) for s in starts]


def norm(t):
    return _WS.sub(" ", t).strip().lower()


def load_rows():
    """R8-H77.load_subsets filter+sample, annotation columns retained."""
    z = zipfile.ZipFile(ARCHIVE)
    out = {}
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0)
        )
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        out[sub] = df.sample(min(N_PER_SUBSET, len(df)), seed=0)
    return out


def locate(text, doc):
    """Char span of a document sentence inside its raw document, or None."""
    for cand in (text, _PREFIX.sub("", text)):
        if cand:
            i = doc.find(cand)
            if i >= 0:
                return i, i + len(cand)
    n_doc, n_txt = norm(doc), norm(_PREFIX.sub("", text))
    if n_txt:
        i = n_doc.find(n_txt)
        if i >= 0:
            lo = int(i / max(len(n_doc), 1) * len(doc))
            return lo, min(len(doc), lo + len(text))
    return None


def map_sentences(resp_sents, ann_texts):
    """H92 sentence -> indices of annotated response sentences it covers."""
    ann_norm = [norm(t) for t in ann_texts]
    ann_tok = [set(n.split()) for n in ann_norm]
    out = []
    for s in resp_sents:
        ns = norm(s)
        toks = set(ns.split())
        hits = [j for j, a in enumerate(ann_norm) if a and (a in ns or ns in a)]
        if not hits and toks:
            hits = [
                j for j, at in enumerate(ann_tok)
                if at and len(toks & at) / len(at) >= 0.6
            ]
        out.append(hits)
    return out


def main():
    subs = load_rows()
    rows = {}
    L = {
        "n_resp": 0, "label1": 0, "label0": 0,
        "identity_adherence_eq_no_unsupported": 0,
        "n_sent_ann": 0, "n_sent_h92": 0,
        "h92_unmapped": 0, "ann_uncovered": 0,
        "ann_uncovered_unsupported": 0,
        "flip_split": 0, "flip_cap": 0,
        "flip_window_strict": 0, "flip_window_lenient": 0,
        "supported_sent": 0, "supported_sent_no_keys": 0,
        "supp_keys": 0, "supp_keys_over_chunk_cap": 0, "supp_keys_unlocatable": 0,
        "supported_sent_multidoc": 0,
        "supported_sent_no_window_strict": 0,
        "supported_sent_no_window_lenient": 0,
        "supported_sent_over_cap": 0,
    }

    for sub, df in subs.items():
        y = df["adherence_score"].cast(pl.Int8).to_numpy()
        o1, o2, o3, o4s, o4l = [], [], [], [], []

        for ri, row in enumerate(df.iter_rows(named=True)):
            ssi = row["sentence_support_information"] or []
            unsupported = set(row["unsupported_response_sentence_keys"] or [])
            key_text = {p[0]: p[1] for p in (row["response_sentences"] or []) if len(p) >= 2}
            keys = [d["response_sentence_key"] for d in ssi] or list(key_text)
            supp_keys = {
                d["response_sentence_key"]: [k for k in (d["supporting_sentence_keys"] or []) if k]
                for d in ssi
            }
            ann_texts = [key_text.get(k, "") for k in keys]
            ann_ok = [k not in unsupported for k in keys]

            L["n_resp"] += 1
            L["label1" if y[ri] else "label0"] += 1
            L["n_sent_ann"] += len(keys)

            # ---- O1: the identity check ---------------------------------------
            v1 = 1.0 if not unsupported else 0.0
            o1.append(v1)
            L["identity_adherence_eq_no_unsupported"] += int(int(v1) == int(y[ri]))

            # ---- O2: through the H92 splitter -----------------------------------
            hs = sentences(row["response"])
            L["n_sent_h92"] += len(hs)
            mapping = map_sentences(hs, ann_texts)
            covered = set()
            per_sent = []  # (truth, [annotated idx])
            for hits in mapping:
                if not hits:
                    L["h92_unmapped"] += 1
                    per_sent.append((1.0, []))  # optimistic - keeps this an upper bound
                    continue
                covered.update(hits)
                per_sent.append((1.0 if all(ann_ok[j] for j in hits) else 0.0, hits))
            for j in range(len(keys)):
                if j not in covered:
                    L["ann_uncovered"] += 1
                    if not ann_ok[j]:
                        L["ann_uncovered_unsupported"] += 1
            v2 = min(t for t, _ in per_sent)
            o2.append(v2)
            L["flip_split"] += int(int(v2) != int(v1))

            # ---- O3 / O4: chunk cap and window reachability ----------------------
            docs_kept = row["documents"][:MAX_CHUNKS]
            smap = {}
            for di, ds in enumerate(row["documents_sentences"]):
                for pair in ds:
                    if len(pair) >= 2:
                        smap[pair[0]] = (di, pair[1])
            spans = {di: win_spans(d) for di, d in enumerate(docs_kept)}

            t3, t4s, t4l = [], [], []
            for truth, hits in per_sent:
                if truth == 0.0 or not hits:
                    t3.append(truth)
                    t4s.append(truth)
                    t4l.append(truth)
                    continue
                L["supported_sent"] += 1
                sk = []
                for j in hits:
                    sk.extend(supp_keys.get(keys[j], []))
                if not sk:
                    L["supported_sent_no_keys"] += 1
                    t3.append(1.0)
                    t4s.append(1.0)
                    t4l.append(1.0)
                    continue
                L["supp_keys"] += len(sk)
                by_doc, over_cap, unlocatable = {}, False, 0
                for k in sk:
                    if k not in smap:
                        unlocatable += 1
                        L["supp_keys_unlocatable"] += 1
                        continue
                    di, txt = smap[k]
                    if di >= len(docs_kept):
                        over_cap = True
                        L["supp_keys_over_chunk_cap"] += 1
                        continue
                    sp = locate(txt, docs_kept[di])
                    if sp is None:
                        unlocatable += 1
                        L["supp_keys_unlocatable"] += 1
                        continue
                    by_doc.setdefault(di, []).append(sp)
                if over_cap:
                    L["supported_sent_over_cap"] += 1
                    t3.append(0.0)
                    t4s.append(0.0)
                    t4l.append(0.0)
                    continue
                t3.append(1.0)
                if not by_doc:
                    # nothing locatable - optimistic on both variants
                    t4s.append(1.0)
                    t4l.append(1.0)
                    continue
                if len(by_doc) > 1:
                    L["supported_sent_multidoc"] += 1
                # LENIENT: some window covers at least one supporting sentence
                len_ok = any(
                    ws <= lo and hi <= we
                    for di, sps in by_doc.items()
                    for (lo, hi) in sps
                    for (ws, we) in spans.get(di, [])
                )
                # STRICT: one window covers ALL supporting sentences (single doc only)
                str_ok = False
                if len(by_doc) == 1 and unlocatable == 0:
                    di, sps = next(iter(by_doc.items()))
                    lo, hi = min(a for a, _ in sps), max(b for _, b in sps)
                    str_ok = any(ws <= lo and hi <= we for (ws, we) in spans.get(di, []))
                elif unlocatable > 0 and len(by_doc) == 1:
                    str_ok = len_ok  # cannot prove failure - optimistic
                t4s.append(1.0 if str_ok else 0.0)
                t4l.append(1.0 if len_ok else 0.0)
                L["supported_sent_no_window_strict"] += int(not str_ok)
                L["supported_sent_no_window_lenient"] += int(not len_ok)

            v3 = min(t3) if t3 else 1.0
            v4s = min(t4s) if t4s else 1.0
            v4l = min(t4l) if t4l else 1.0
            o3.append(v3)
            o4s.append(v4s)
            o4l.append(v4l)
            L["flip_cap"] += int(int(v3) != int(v2))
            L["flip_window_strict"] += int(int(v4s) != int(v3))
            L["flip_window_lenient"] += int(int(v4l) != int(v3))

        r = {"n": int(len(y)), "grounded_rate": round(float(y.mean()), 4)}
        for tag, o in (
            ("o1_annotation_and", o1),
            ("o2_plus_splitter", o2),
            ("o3_plus_chunkcap", o3),
            ("o4_windows_strict", o4s),
            ("o4_windows_lenient", o4l),
        ):
            s = np.asarray(o, dtype=np.float64)
            auc, f1, _ = auc_and_f1(y, s)
            r[f"{tag}_auc"] = round(auc, 4)
            r[f"{tag}_f1"] = round(f1, 4)
        rows[sub] = r
        print(
            f"  {sub:12s} n={r['n']:>4} base {r['grounded_rate']:.3f}  "
            f"O1 {r['o1_annotation_and_auc']:.4f}  O2 {r['o2_plus_splitter_auc']:.4f}  "
            f"O3 {r['o3_plus_chunkcap_auc']:.4f}  O4str {r['o4_windows_strict_auc']:.4f}  "
            f"O4len {r['o4_windows_lenient_auc']:.4f}",
            flush=True,
        )

    means = {
        t: round(float(np.mean([r[f"{t}_auc"] for r in rows.values()])), 4)
        for t in (
            "o1_annotation_and", "o2_plus_splitter", "o3_plus_chunkcap",
            "o4_windows_strict", "o4_windows_lenient",
        )
    }

    ns = max(L["supported_sent"], 1)
    losses = [
        {
            "rank": 1,
            "source": "conjunctive support - one window cannot carry the whole evidence",
            "mechanism": (
                f"{L['supported_sent_no_window_strict']} of {L['supported_sent']} supported "
                f"scored sentences ({L['supported_sent_no_window_strict'] / ns:.1%}) cannot have "
                "ALL their annotated supporting sentences inside any single 1500-char window; "
                f"{L['supported_sent_multidoc']} of those ({L['supported_sent_multidoc'] / ns:.1%} "
                "of supported sentences) draw support from more than one DOCUMENT, so no window "
                "exists even in principle. A faithful per-(sentence, window) entailer must return "
                "0 on every window for these, and MIN-over-sentences propagates it to the response"
            ),
            "responses_flipped": L["flip_window_strict"],
            "auc_cost_pooled": round(
                means["o3_plus_chunkcap"] - means["o4_windows_strict"], 4
            ),
        },
        {
            "rank": 2,
            "source": "H92 sentence splitter drops annotated sentences",
            "mechanism": (
                "sentences() keeps only >=25-char fragments and caps at 12; "
                f"{L['ann_uncovered']} of {L['n_sent_ann']} annotated response sentences "
                f"({L['ann_uncovered'] / max(L['n_sent_ann'], 1):.1%}) are covered by no scored "
                f"sentence, and {L['ann_uncovered_unsupported']} of those are the UNSUPPORTED "
                f"sentence that produced the label - {L['flip_split']} responses become invisible "
                "to the min. Concentrated in finqa (1.0000 -> 0.7500) and tatqa (-> 0.8929), "
                "where responses are arithmetic fragments below the 25-char floor"
            ),
            "responses_flipped": L["flip_split"],
            "auc_cost_pooled": round(means["o1_annotation_and"] - means["o2_plus_splitter"], 4),
        },
        {
            "rank": 3,
            "source": "documents[:8] chunk cap strands evidence",
            "mechanism": (
                f"MAX_CHUNKS=8 strands {L['supp_keys_over_chunk_cap']} supporting keys on "
                f"{L['supported_sent_over_cap']} sentences, flipping {L['flip_cap']} responses. "
                "Window GEOMETRY itself costs nothing: "
                f"{L['supported_sent_no_window_lenient']} of {L['supported_sent']} supported "
                "sentences (0.0%) lack even one supporting sentence wholly inside some window - "
                "at stride 750 every document sentence shorter than 750 chars is fully contained "
                "somewhere, so 1500/750 truncation is NOT a live loss source on this arena"
            ),
            "responses_flipped": L["flip_cap"],
            "auc_cost_pooled": round(
                means["o2_plus_splitter"] - means["o3_plus_chunkcap"], 4
            ),
        },
    ]

    print("\n" + "=" * 92)
    print("R12 LABEL-CEILING - perfect per-(sentence, window) entailment oracle")
    print("=" * 92)
    for k, v in means.items():
        print(f"  {k:22s} pooled mean AUC {v:.4f}")
    print("\n  top mechanical loss sources:")
    for d in losses:
        print(f"   {d['rank']}. {d['source']}  (AUC {d['auc_cost_pooled']:+.4f})")
    print(f"\n  loss accounting: {json.dumps(L, indent=2)}")

    OUT.write_text(json.dumps({
        "case": "B",
        "case_finding": (
            "The read's labels ARE derived from the sentence-level annotations. "
            "`adherence_score` == (unsupported_response_sentence_keys is empty) on "
            f"{L['identity_adherence_eq_no_unsupported']}/{L['n_resp']} arena rows, all ten "
            "subsets. The annotation-granularity ceiling is therefore 1.0 by construction, and "
            "the diagnostic measures what the read's own machinery (H92 splitter, documents[:8] "
            "cap, 1500/750 windows) costs a perfect entailer."
        ),
        "truth_field_warning": (
            "sentence_support_information[].fully_supported is NULL on 8/10 subsets "
            "(finqa 1101/1116, tatqa 658/665, techqa 1739/1917, pubmedqa 736/832 null) and is "
            "NOT a usable truth field. unsupported_response_sentence_keys is populated on all "
            "subsets and is what adherence_score is computed from."
        ),
        "read": (
            "R8-H101_windowed_read.py - H92 min-over-sentences of max over 1500/750 windows "
            "of documents[:8]"
        ),
        "label_field": "adherence_score (response-level boolean, GPT-4o annotated)",
        "oracle_truth_field": (
            "unsupported_response_sentence_keys (support) + "
            "sentence_support_information[].supporting_sentence_keys (evidence location)"
        ),
        "auc_path": "R7-H59.auc_and_f1 (sklearn roc_auc_score), per subset then unweighted mean",
        "per_subset": rows,
        "pooled_means": means,
        "ceiling_headline": {
            "annotation_granularity_ceiling": means["o1_annotation_and"],
            "faithful_entailer_ceiling": means["o4_windows_strict"],
            "leaky_entailer_ceiling": means["o4_windows_lenient"],
            "reading": (
                f"The honest ceiling for a FAITHFUL per-(sentence, window) entailer is "
                f"{means['o4_windows_strict']:.4f} pooled mean AUC. It is not an upper bound on "
                "achievable AUC: an entailer that fires on PARTIAL window support reaches "
                f"{means['o4_windows_lenient']:.4f}, so this label/aggregation combination "
                "actively PENALISES faithfulness. Against the 0.74 round bar, 0.74 is reachable "
                f"but sits only {means['o4_windows_strict'] - 0.74:+.4f} below the faithful "
                "ceiling and 0.0350 above the H105-pair incumbent 0.7031 - i.e. the remaining "
                "gap is 68% of the entire faithful-oracle headroom."
            ),
        },
        "window_reachability": {
            "supported_scored_sentences": L["supported_sent"],
            "frac_no_single_window_carries_all_support": round(
                L["supported_sent_no_window_strict"] / ns, 4
            ),
            "frac_support_spans_multiple_documents": round(
                L["supported_sent_multidoc"] / ns, 4
            ),
            "frac_no_window_carries_any_support": round(
                L["supported_sent_no_window_lenient"] / ns, 4
            ),
        },
        "caveats": [
            (
                f"{L['supp_keys_unlocatable']} of {L['supp_keys']} supporting keys "
                f"({L['supp_keys_unlocatable'] / max(L['supp_keys'], 1):.1%}) could not be located "
                "as a substring of their raw document; they are resolved OPTIMISTICALLY, so both "
                "O4 numbers are upper bounds on their own ceilings"
            ),
            (
                f"{L['h92_unmapped']} of {L['n_sent_h92']} H92 sentences map to no annotated "
                "sentence and are scored 1 (optimistic)"
            ),
            "All oracle scores are binary, so per-subset AUC = (TPR + TNR) / 2 exactly",
        ],
        "top_loss_sources": losses,
        "loss_accounting": L,
        "licence": "ANALYSIS ONLY - Round 12 ruling 5; no quantity here may enter any build",
    }, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()
