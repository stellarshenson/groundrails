"""MT bridge (translate-then-recall) and cross-lingual grounding tests.

The HIGH tier translates a non-English claim to English (argos / CTranslate2 +
SaT segmentation) and recall-matches against the evidence. Cross-lingual cases
need the argos ``de->en`` model installed, so they skip when it is absent.
"""

import pytest

from groundrails import ground, ground_batch
from groundrails import lexical_mt as mt

_skip_de = pytest.mark.skipif(not mt.has_model("de"), reason="argos de->en model not installed")


def test_translate_passthrough_english():
    text = "The cat sat on the mat in the afternoon sun."
    assert mt.translate(text, "en") == text


def test_translate_passthrough_undetermined():
    assert mt.translate("alpha beta gamma delta", "und") == "alpha beta gamma delta"


@_skip_de
def test_translate_german_to_english():
    out = mt.translate("Der Eiffelturm steht in Paris.", "de").lower()
    assert "eiffel" in out and "paris" in out


@_skip_de
def test_cross_lingual_grounding_supported():
    # German claim grounds against English evidence via the MT bridge (HIGH tier)
    m = ground(
        "Der Eiffelturm steht in Paris.",
        ["The Eiffel Tower is located in Paris, France, on the Champ de Mars."],
    )
    assert m.match_type != "none"


@_skip_de
def test_cross_lingual_batch_grounds_both():
    matches = ground_batch(
        ["Der Eiffelturm steht in Paris.", "Wasser kocht bei hundert Grad Celsius."],
        ["The Eiffel Tower is in Paris. Water boils at 100 degrees Celsius at sea level."],
    )
    assert len(matches) == 2
    assert all(m.match_type != "none" for m in matches)
