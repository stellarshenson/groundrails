"""Exact-equivalence gate: groundrails must reproduce the scoring the grounder
produced in the parent stellars-claude-code-plugins package.

`tests/data/grounding_golden.json` was captured from the parent grounder before
the code was migrated. This test runs the same public fixture through the
groundrails grounder and asserts byte-for-byte identical scoring. A drift here
means the migration changed behaviour - the whole point was that it must not.
"""

import json
from pathlib import Path

from groundrails.config import load_document_processing_config
from groundrails.grounding import ground

GOLDEN = Path(__file__).parent / "data" / "grounding_golden.json"

# Same public/synthetic fixture used to capture the parent snapshot - no client data.
FIXTURE = [
    {"id": "support_exact", "claim": "The Eiffel Tower is in Paris.",
     "sources": ["The Eiffel Tower is located in Paris, France, on the Champ de Mars."]},
    {"id": "support_para", "claim": "Water boils at 100 degrees Celsius at sea level.",
     "sources": ["At sea level atmospheric pressure, water reaches its boiling point at 100 C."]},
    {"id": "numeric_contra", "claim": "The model has 512 transformer layers.",
     "sources": ["The model is built from 1000 transformer layers in total."]},
    {"id": "entity_contra", "claim": "The training used H100 GPUs.",
     "sources": ["All training runs were performed on A100 GPUs in the cluster."]},
    {"id": "hallucination", "claim": "The capital of Australia is Sydney.",
     "sources": ["Canberra is the capital city of Australia; Sydney is the largest city."]},
    {"id": "unsupported_novel", "claim": "The quarterly revenue grew by 37 percent.",
     "sources": ["The report describes the company's new logo and office relocation."]},
    {"id": "xling_support_de", "claim": "Der Eiffelturm steht in Paris.",
     "sources": ["The Eiffel Tower is located in Paris, France."]},
    {"id": "xling_contra_de", "claim": "Der Eiffelturm steht in Berlin.",
     "sources": ["The Eiffel Tower is located in Paris, France."]},
]

FIELDS = [
    "exact_score", "fuzzy_score", "bm25_score", "bm25_token_recall",
    "semantic_score", "semantic_ratio", "agreement_score", "combined_score",
    "verdict_probability", "verdict_uncertainty", "lexical_co_support",
    "verification_needed", "numeric_mismatches", "entity_mismatches", "entities_absent",
]


def _rnd(v):
    if isinstance(v, float):
        return round(v, 10)
    if isinstance(v, (list, tuple)):
        return [_rnd(x) for x in v]
    return v


def _snapshot():
    cfg = load_document_processing_config()
    out = {}
    for item in FIXTURE:
        m = ground(item["claim"], item["sources"], config=cfg)
        rec = {"match_type": str(getattr(m, "match_type", None))}
        for f in FIELDS:
            rec[f] = _rnd(getattr(m, f, None))
        out[item["id"]] = rec
    # round-trip through json to normalise tuples->lists like the golden capture
    return json.loads(json.dumps(out, sort_keys=True, default=str))


def test_grounding_matches_parent_golden():
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = _snapshot()
    assert actual.keys() == expected.keys()
    mismatches = {k: (expected[k], actual[k]) for k in expected if actual[k] != expected[k]}
    assert not mismatches, f"scoring drift vs parent golden: {json.dumps(mismatches, indent=2)}"
