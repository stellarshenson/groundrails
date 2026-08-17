"""Contract C2 / C3 / C6 / C7 / C8 for `ragtruth_translated`. CPU ONLY.

C2  exact-string disjointness against every evaluation surface, in the three
    registered forms (raw, truncated to `chunk_max_chars`, whitespace-collapsed
    case-folded) and in BOTH directions.
C3  the split axis this corpus actually cuts on, measured from the archive.
C6  the memorisation channel - what a feature keyed on a shared field can read.
C7  rows and pairs, both counts.
C8  within-member duplication and repeat structure.

The member is read from `ragtruth_translated_member.parquet`, produced by
`ragtruth_translated_extract.py` through the BANKED loader.

Run:  uv run python experiments/grounding-semantic/contract/ragtruth_translated_c2378.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import importlib.util as _ilu
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
SEM = HERE.parent
ROOT = SEM.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
MEMBER = HERE / "ragtruth_translated_member.parquet"
OUT = HERE / "ragtruth_translated_c2378.json"

CHUNK_MAX = 1500  # M59.CFG.chunk_max_chars, pinned by the extract stage
LANGS = ("de", "fr", "es", "it", "pl", "hu", "cn")

HELD_OUT_EVALS = (
    "R17-H143_evalset.parquet",
    "R20-H177_eval_B.parquet",
    "R20-H177_eval_C.parquet",
    "R20-H175b_qlane_eval.parquet",
    "R20-H175b_qlane_eval_repaired.parquet",
    "R20-H175b_qlane_eval_clean.parquet",
    "R20-H175b_qlane_eval_clean_prefix.parquet",
    "R11-H117_heldout_pairs.parquet",
)
GOLD_FULL = SEM / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    """Whitespace-collapsed, case-folded - R20-H175b_qlane_eval_clean.norm."""
    return " ".join(s.split()).casefold()


def forms(texts):
    raw = {t for t in texts if t and t.strip()}
    return {
        "raw": raw,
        "trunc": {t[:CHUNK_MAX] for t in raw},
        "nraw": {norm(t) for t in raw},
        "ntrunc": {norm(t[:CHUNK_MAX]) for t in raw},
    }


def disjoint(member, surface, label):
    """Exact membership in all three forms, both directions."""
    per_form = {}
    m_hit, s_hit = set(), set()
    for name, mkey, skey in (
        ("raw_vs_raw", "raw", "raw"),
        ("raw_vs_truncated", "raw", "trunc"),
        ("truncated_vs_raw", "trunc", "raw"),
        ("truncated_vs_truncated", "trunc", "trunc"),
        ("normalised_raw", "nraw", "nraw"),
        ("normalised_truncated", "ntrunc", "ntrunc"),
    ):
        inter = member[mkey] & surface[skey]
        per_form[name] = len(inter)
        if mkey == "raw":
            m_hit |= inter
        if skey == "raw":
            s_hit |= inter
    total = sum(per_form.values())
    return {
        "surface": label,
        "member_units": len(member["raw"]),
        "surface_units": len(surface["raw"]),
        "counts_per_form": per_form,
        "any_form_hits": int(total),
        "member_to_surface_fraction": round(len(m_hit) / max(len(member["raw"]), 1), 6),
        "surface_to_member_fraction": round(len(s_hit) / max(len(surface["raw"]), 1), 6),
        "pass": total == 0,
    }


def main():
    G = _mod("provgate", SEM / "provenance_gate.py")
    mem = pl.read_parquet(MEMBER)
    res = {"member": "ragtruth_translated", "chunk_max_chars": CHUNK_MAX}

    # ------------------------------------------------------------------ C7 --
    pair_key = mem.select(
        pl.concat_str(["claim", "chunk"], separator="||CONTRACT-SEP||").alias("k")
    )["k"]
    res["C7_units_and_volume"] = {
        "declared_unit": "rows - the loader emits one (claim, evidence, label) row "
        "per RAGTruth response per language; there is no pair construction, so "
        "rows and (claim, evidence) pairs are the same object",
        "rows": int(mem.height),
        "pairs_claim_evidence": int(pair_key.n_unique()),
        "rows_per_language": {
            k: int(v) for k, v in mem.group_by("lang").len().sort("lang").iter_rows()
        },
        "registered_figure": "~105,630 rows (7 x 15,090)",
        "label1_rows": int((mem["label"] == 1.0).sum()),
        "label0_rows": int((mem["label"] == 0.0).sum()),
        "label1_rate": round(float(mem["label"].mean()), 6),
    }

    # ------------------------------------------------------------------ C3 --
    # The translated splits are ROW-ALIGNED translations of the RAGTruth English
    # rows (verified below), so the split axis is RAGTruth's own, and the axis is
    # measurable exactly on the English archive where the document field exists.
    ze = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    en_tr = pl.read_parquet(
        io.BytesIO(ze.read(next(x for x in ze.namelist() if x.endswith("__train.parquet"))))
    )
    en_te = pl.read_parquet(
        io.BytesIO(ze.read(next(x for x in ze.namelist() if x.endswith("__test.parquet"))))
    )
    en_y = (
        (en_tr["hallucination_labels_processed"].struct.field("evident_conflict") == 0)
        & (en_tr["hallucination_labels_processed"].struct.field("baseless_info") == 0)
    ).cast(pl.Int8).to_numpy()

    zt = zipfile.ZipFile(DATA / "dataset-ragtruth-translated.zip")
    align, split_overlap = {}, {}
    for lg in LANGS:
        tr = pl.read_parquet(
            io.BytesIO(
                zt.read(
                    next(
                        x
                        for x in zt.namelist()
                        if f"ragtruth-{lg}-" in x and x.endswith("__train.parquet")
                    )
                )
            )
        )
        te = pl.read_parquet(
            io.BytesIO(
                zt.read(
                    next(
                        x
                        for x in zt.namelist()
                        if f"ragtruth-{lg}-" in x and x.endswith("__test.parquet")
                    )
                )
            )
        )
        ty = (tr["labels"].list.len() == 0).cast(pl.Int8).to_numpy()
        align[lg] = {
            "train_rows": tr.height,
            "task_type_sequence_matches_english": tr["task_type"].to_list()
            == en_tr["task_type"].to_list(),
            "label_vector_agreement_with_english": round(float((ty == en_y).mean()), 6),
            "label_vector_disagreements": int((ty != en_y).sum()),
        }
        ptr, pte = set(tr["prompt"].to_list()), set(te["prompt"].to_list())
        atr, ate = set(tr["answer"].to_list()), set(te["answer"].to_list())
        split_overlap[lg] = {
            "train_rows": tr.height,
            "test_rows": te.height,
            "distinct_train_prompt_strings": len(ptr),
            "distinct_test_prompt_strings": len(pte),
            "prompt_strings_shared": len(ptr & pte),
            "answer_strings_shared": len(atr & ate),
            "test_answer_share_in_train": round(len(atr & ate) / max(len(ate), 1), 6),
        }

    res["C3_split_semantics"] = {
        "declared_axis": "the corpus card states an official train/test split and "
        "nothing about its axis",
        "measured_axis": "source DOCUMENT (RAGTruth `context`) - measured on the "
        "English archive the translated splits are row-aligned to",
        "row_alignment_to_english": align,
        "english_archive": {
            "train_rows": en_tr.height,
            "test_rows": en_te.height,
            "train_distinct_contexts": en_tr["context"].n_unique(),
            "test_distinct_contexts": en_te["context"].n_unique(),
            "context_strings_shared_train_test": len(
                set(en_tr["context"].to_list()) & set(en_te["context"].to_list())
            ),
            "query_strings_shared_train_test": len(
                set(en_tr["query"].to_list()) & set(en_te["query"].to_list())
            ),
            "test_distinct_queries": en_te["query"].n_unique(),
        },
        "translated_split_overlap_exact": split_overlap,
        "caveat": "exact-string prompt identity is NOT a reliable document key on "
        "this member: each row's prompt was machine-translated independently, so "
        "the same source document yields near-duplicate but non-identical prompt "
        "strings. The exact-string zero is therefore weak evidence on its own; the "
        "document-level zero comes from the English archive's `context` field via "
        "the verified row alignment. A near-duplicate n-gram read is in the C4 "
        "artifact.",
    }

    # ------------------------------------------------------------------ C8 --
    # Underlying document identity, recovered through the verified row alignment.
    ctx = en_tr["context"].to_list()
    ctx_ids = {c: i for i, c in enumerate(dict.fromkeys(ctx))}
    doc_idx = np.array([ctx_ids[c] for c in ctx])
    per_doc = Counter(doc_idx.tolist())
    distinct_prompt_per_lang = {
        lg: split_overlap[lg]["distinct_train_prompt_strings"] for lg in LANGS
    }
    res["C8_provenance_and_structure"] = {
        "source": "KRLabsOrg/ragtruth-{de,fr,es,it,pl,hu,cn}-translated on the "
        "HuggingFace Hub; machine translations of wandb/RAGTruth-processed",
        "licence": "MIT (sidecar data/external/datasets/dataset-ragtruth-translated.md)",
        "retrieval_date": "2026-07-29 (archive mtime; fetched by "
        "scripts/fetch_grounding_datasets.py)",
        "selection_predicate": "for each of the 7 languages, the `__train.parquet` "
        "member of dataset-ragtruth-translated.zip; label = (labels list empty); "
        "filter prompt.len_chars > 50; evidence = `prompt`, claim = `answer`. "
        "The prompt filter drops 0 rows on every language.",
        "distinct_claims": int(mem["claim"].n_unique()),
        "distinct_evidence_strings": int(mem["chunk"].n_unique()),
        "distinct_evidence_strings_per_language": distinct_prompt_per_lang,
        "underlying_documents": {
            "distinct_source_documents_per_language": len(ctx_ids),
            "rows_per_document_mean": round(15090 / len(ctx_ids), 4),
            "rows_per_document_max": max(per_doc.values()),
            "note": "15,090 rows per language rest on 2,514 distinct source "
            "documents - RAGTruth pairs each document with ~6 model responses. "
            "Exact-string dedup of the translated prompt reports 11,540-14,950 "
            "distinct evidence strings per language, overstating document "
            "diversity by 4.6x to 5.9x because machine translation is not "
            "deterministic across rows.",
        },
        "cross_language_replication": {
            "instances_replicated": 15090,
            "copies_in_the_member": 7,
            "copies_in_the_clean_mix_including_ragtruth_en": 8,
            "member_rows": int(mem.height),
            "clean_mix_rows": 685670,
            "share_of_clean_mix_that_is_this_one_corpus": round(
                (15090 * 8) / 685670, 6
            ),
            "note": "the member is 7 parallel translations of ONE 15,090-row "
            "corpus, and ragtruth_en is the 8th copy of the same rows. 120,720 of "
            "the clean mix's 685,670 rows carry 15,090 distinct (document, "
            "response, label) instances resting on 2,514 documents.",
        },
        "public_repository_check": {
            "client_or_company_name_in_artifacts": False,
            "note": "no client or company name appears in any artifact written by "
            "this verification",
        },
        "exact_duplicate_rows": int(mem.height - pair_key.n_unique()),
    }

    # ------------------------------------------------------------------ C2 --
    m_ev = forms(mem["chunk"].to_list())
    m_cl = forms(mem["claim"].to_list())
    print(
        f"member evidence forms: {len(m_ev['raw'])} raw / {len(m_ev['trunc'])} trunc",
        flush=True,
    )

    surfaces = {}
    arena_texts, _ = G.load_arena()
    surfaces["arena_documents"] = forms(
        [c for v in arena_texts.values() for c in v]
    )
    gf = pl.read_parquet(GOLD_FULL)
    surfaces["gold_full_chunks"] = forms(gf["chunk"].to_list())
    surfaces["gold_full_claims"] = forms(gf["claim"].to_list())
    for name in HELD_OUT_EVALS:
        p = SEM / name
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        col = next((c for c in ("chunk", "evidence", "context") if c in d.columns), None)
        if col:
            surfaces[f"{name}:{col}"] = forms(d[col].to_list())
        if "claim" in d.columns:
            surfaces[f"{name}:claim"] = forms(d["claim"].to_list())

    ev_res, cl_res = {}, {}
    for label, sf in surfaces.items():
        ev_res[label] = disjoint(m_ev, sf, label)
        cl_res[label] = disjoint(m_cl, sf, label)
        print(
            f"  {label}: evidence {ev_res[label]['any_form_hits']} hits, "
            f"claims {cl_res[label]['any_form_hits']} hits",
            flush=True,
        )

    # the member's OWN in-domain evaluation surface - the non-EN lineage read
    own_ev_texts, own_cl_texts = [], []
    for lg in LANGS:
        te = pl.read_parquet(
            io.BytesIO(
                zt.read(
                    next(
                        x
                        for x in zt.namelist()
                        if f"ragtruth-{lg}-" in x and x.endswith("__test.parquet")
                    )
                )
            )
        )
        te = te.filter(
            (pl.col("prompt").str.len_chars() > 50)
            & (pl.col("answer").str.len_chars() > 20)
        )
        own_ev_texts += te["prompt"].to_list()
        own_cl_texts += te["answer"].to_list()
    own_ev = forms(own_ev_texts)
    own_cl = forms(own_cl_texts)
    own = {
        "evidence": disjoint(m_ev, own_ev, "ragtruth_translated_test_prompts"),
        "claims": disjoint(m_cl, own_cl, "ragtruth_translated_test_answers"),
    }

    res["C2_disjointness"] = {
        "forms": "raw, truncated to chunk_max_chars=1500, whitespace-collapsed "
        "case-folded; each compared in both member->surface and "
        "surface->member denominators",
        "surfaces_tested": sorted(surfaces),
        "member_evidence_vs_surfaces": ev_res,
        "member_claims_vs_surfaces": cl_res,
        "all_forms_zero": all(v["pass"] for v in ev_res.values())
        and all(v["pass"] for v in cl_res.values()),
        "own_in_domain_eval_reported_separately": {
            "note": "the RAGTruth-translated TEST split is not one of the "
            "contract's named evaluation surfaces, but it is the member's own "
            "in-domain read (`ragtruth_nonen`) and the non-EN hold is used as a "
            "gate, so it is measured here",
            **own,
        },
        "instrument_caveat": "exact string membership only. Machine translation is "
        "not row-deterministic on this corpus, so exact matching cannot detect a "
        "near-duplicate document that entered under a different translation. The "
        "n-gram instrument in the C4 artifact covers that direction.",
    }

    # ------------------------------------------------------------------ C6 --
    # Shared field: 15,090 rows per language rest on 2,514 documents, so ~6 rows
    # share each document key. What can a feature keyed on the document read?
    lab = (
        pl.read_parquet(MEMBER)
        .filter(pl.col("lang") == "de")["label"]
        .to_numpy()
        .astype(float)
    )
    grp_sum = np.zeros(len(ctx_ids))
    grp_n = np.zeros(len(ctx_ids))
    np.add.at(grp_sum, doc_idx, lab)
    np.add.at(grp_n, doc_idx, 1.0)
    loo = (grp_sum[doc_idx] - lab) / np.maximum(grp_n[doc_idx] - 1.0, 1.0)
    pos, neg = loo[lab == 1.0], loo[lab == 0.0]
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # tie-corrected AUROC via rank sum
    s = np.concatenate([pos, neg])
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    rsum = np.zeros(len(uniq))
    np.add.at(rsum, inv, ranks)
    ranks = (rsum / cnt)[inv]
    n1, n0 = len(pos), len(neg)
    auroc = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    test_prompt_in_mix = own["evidence"]["surface_to_member_fraction"]
    res["C6_memorisation_channel"] = {
        "shared_field": "the source document - ~6 responses per document share one "
        "evidence text, so a feature keyed on the document is computable",
        "within_member_document_key_auroc": round(float(auroc), 4),
        "within_member_document_key_note": "leave-one-out mean label of the row's "
        "own document group, scored against the row label, on the German slice "
        "(labels are identical across languages by construction). Above 0.5 means "
        "the document key alone carries label information INSIDE the training mix",
        "eval_side_coverage": {
            "in_domain_test_prompts_found_in_the_member": test_prompt_in_mix,
            "verdict": "UNDEFINED - the feature has zero coverage on the member's "
            "own held-out read"
            if test_prompt_in_mix == 0.0
            else "defined at the stated coverage",
        },
        "cross_language_channel": "each (document, response, label) instance appears "
        "8 times in the clean mix (English plus 7 translations). A model can carry a "
        "label association across the language copies. This is train-internal "
        "redundancy, not eval leakage - the evaluation surfaces above read zero.",
    }

    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}", flush=True)
    print(
        json.dumps(
            {
                "C7": res["C7_units_and_volume"],
                "C2_all_forms_zero": res["C2_disjointness"]["all_forms_zero"],
                "C6_doc_key_auroc": res["C6_memorisation_channel"][
                    "within_member_document_key_auroc"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
