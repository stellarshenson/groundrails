"""Tests for the self-contained re-calibration mechanism (API + CLI).

Covers the in-library dogfood path the CLI ``calibration fit`` / ``eval`` drive:
``build_feature_frame`` re-grounds labeled records, ``calibrate`` fits the
manifold, ``verdict_to_block`` serialises to the config ``calibration:`` block,
and that block round-trips through ``verdict_from_config`` so a deployed
``ground()`` would run on the fitted weights. The two CLI subcommands are
exercised end to end, including column mapping and input-validation errors.
"""

from __future__ import annotations

import json
import warnings

import pytest

from groundrails import calibration as C
from groundrails.cli import main as cli_main

warnings.filterwarnings("ignore")


def _pytensor_compiles() -> bool:
    """True when pytensor can compile (the bambi/PyMC fit). Skip otherwise."""
    try:
        import numpy as np
        import pymc as pm

        with pm.Model():
            mu = pm.Normal("mu", 0.0, 1.0)
            pm.Normal("obs", mu=mu, sigma=1.0, observed=np.array([0.0, 1.0]))
            pm.sample(
                draws=2, tune=2, chains=1, progressbar=False, compute_convergence_checks=False
            )
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pytensor_compiles(),
    reason="pytensor C compilation unavailable - calibration (bambi/PyMC) tests skipped",
)

# Small sampler settings keep the suite fast; the records are trivially separable
# so behaviour (not convergence) is what we assert.
DRAWS = 150
TUNE = 150

_SRC_A = "The Eiffel Tower is in Paris. It was completed in 1889 for the World's Fair."
_SRC_B = "Mercury is the smallest planet. It orbits the Sun every 88 days."

# Strong-signal features for a verbatim hit vs an absent fabrication - used to
# probe the fitted verdict without re-grounding.
_HIT = {"exact": 1.0, "fuzzy": 1.0, "bm25_recall": 1.0, "voters": 1.0, "lexical_cosupport": 1.0}
_MISS = {"exact": 0.0, "fuzzy": 0.0, "bm25_recall": 0.0, "voters": 0.0, "lexical_cosupport": 0.0}


def _raw_records() -> list[dict]:
    """Trivially separable labeled records: grounded claims are verbatim
    substrings of their source (exact/fuzzy/bm25 fire); hallucinations are absent."""
    grounded = [
        ("The Eiffel Tower is in Paris.", _SRC_A),
        ("It was completed in 1889 for the World's Fair.", _SRC_A),
        ("Mercury is the smallest planet.", _SRC_B),
        ("It orbits the Sun every 88 days.", _SRC_B),
    ]
    hallucinated = [
        ("The tower is built from solid titanium plates.", _SRC_A),
        ("Penguins migrate across the Sahara each winter.", _SRC_A),
        ("Mercury has a thick breathable atmosphere of oxygen.", _SRC_B),
        ("The planet completes one orbit every seven hundred years.", _SRC_B),
    ]
    recs = [{"claim": c, "source_text": s, "label": 1, "lang": "en"} for c, s in grounded]
    recs += [{"claim": c, "source_text": s, "label": 0, "lang": "en"} for c, s in hallucinated]
    return recs


@pytest.fixture(scope="module")
def fitted():
    """A verdict calibrated once on the raw records (reused across API tests)."""
    return C.calibrate(_raw_records(), draws=DRAWS, tune=TUNE, random_seed=0)


def _write_jsonl(path, recs) -> None:
    path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")


# ---------------------------------------------------------------------------
# API: build_feature_frame
# ---------------------------------------------------------------------------


class TestBuildFeatureFrame:
    def test_frame_has_predictors_label_and_lang(self):
        frame = C.build_feature_frame(_raw_records())
        assert len(frame) == 8
        for col in C.PREDICTORS:
            assert col in frame.columns
        assert C.RESPONSE in frame.columns
        assert "lang" in frame.columns

    def test_grounded_records_outscore_hallucinations(self):
        frame = C.build_feature_frame(_raw_records())
        g = frame[frame[C.RESPONSE] == 1.0]
        h = frame[frame[C.RESPONSE] == 0.0]
        # verbatim claims fire the exact pass and carry more voter signal;
        # fabrications fire nothing
        assert g["exact"].mean() > h["exact"].mean()
        assert g["voters"].mean() > h["voters"].mean()

    def test_source_text_accepts_list_of_chunks(self):
        recs = [
            {
                "claim": "The Eiffel Tower is in Paris.",
                "source_text": ["unrelated text", _SRC_A],
                "label": 1,
            }
        ]
        frame = C.build_feature_frame(recs)
        assert frame.loc[0, "exact"] == 1.0
        # lang carried only when present in the record
        assert "lang" not in frame.columns


# ---------------------------------------------------------------------------
# API: calibrate / verdict_to_block / round-trip
# ---------------------------------------------------------------------------


class TestCalibrate:
    def test_returns_verdict_that_separates(self, fitted):
        assert isinstance(fitted, C.CalibratedVerdict)
        assert fitted.predict_proba([_HIT])[0] > fitted.predict_proba([_MISS])[0]
        assert fitted.confirmed(_HIT) is True

    def test_verdict_to_block_shape(self, fitted):
        block = C.verdict_to_block(fitted)
        assert block["engine"] == "calibrated"
        assert block["mode"] == "lexical"
        assert block["threshold"] == 0.5
        assert "Intercept" in block["weights"]
        # at least one varying lexical predictor was estimated
        assert any(k in block["weights"] for k in ("exact", "bm25_recall", "voters"))

    def test_block_mode_semantic_label(self):
        # the mode tag follows the semantic flag (no grounding cost asserted here)
        block = C.verdict_to_block(C.default_calibrator(draws=DRAWS, tune=TUNE), mode="semantic")
        assert block["mode"] == "semantic"

    def test_block_roundtrips_through_config(self, tmp_path, fitted):
        block = C.verdict_to_block(fitted)
        p = tmp_path / "calibration.json"
        p.write_text(json.dumps(block), encoding="utf-8")
        reloaded = C.verdict_from_config(p)
        assert reloaded is not None
        direct = C.CalibratedVerdict.from_weights(block["weights"], threshold=block["threshold"])
        assert reloaded.predict_proba([_HIT])[0] == pytest.approx(
            direct.predict_proba([_HIT])[0], abs=1e-9
        )

    def test_evaluate_reports_macro_f1(self, fitted):
        frame = C.build_feature_frame(_raw_records())
        metrics = C.evaluate(fitted, frame, group_col="lang")
        assert "f1_macro" in metrics
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert "by_lang" in metrics


# ---------------------------------------------------------------------------
# CLI: calibration fit / eval
# ---------------------------------------------------------------------------


class TestCli:
    def test_fit_writes_calibration_block(self, tmp_path, capsys):
        inp = tmp_path / "recs.jsonl"
        _write_jsonl(inp, _raw_records())
        out = tmp_path / "cal.json"
        rc = cli_main(
            [
                "calibration",
                "fit",
                "--input",
                str(inp),
                "-o",
                str(out),
                "--draws",
                str(DRAWS),
                "--tune",
                str(TUNE),
            ]
        )
        assert rc == 0
        block = json.loads(out.read_text())
        assert block["engine"] == "calibrated"
        assert block["mode"] == "lexical"
        assert "Intercept" in block["weights"]

    def test_eval_reports_macro_f1(self, tmp_path, capsys):
        inp = tmp_path / "recs.jsonl"
        _write_jsonl(inp, _raw_records())
        out = tmp_path / "cal.json"
        cli_main(
            [
                "calibration",
                "fit",
                "--input",
                str(inp),
                "-o",
                str(out),
                "--draws",
                str(DRAWS),
                "--tune",
                str(TUNE),
            ]
        )
        capsys.readouterr()  # clear fit output
        rc = cli_main(["calibration", "eval", "--input", str(inp), "--calibration", str(out)])
        assert rc == 0
        metrics = json.loads(capsys.readouterr().out)
        assert "f1_macro" in metrics
        assert "by_lang" in metrics

    def test_fit_with_custom_columns(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame(_raw_records()).rename(
            columns={"claim": "q", "source_text": "ev", "label": "y", "lang": "loc"}
        )
        csv = tmp_path / "recs.csv"
        df.to_csv(csv, index=False)
        out = tmp_path / "cal.json"
        rc = cli_main(
            [
                "calibration",
                "fit",
                "--input",
                str(csv),
                "-o",
                str(out),
                "--claim-col",
                "q",
                "--source-col",
                "ev",
                "--label-col",
                "y",
                "--lang-col",
                "loc",
                "--draws",
                str(DRAWS),
                "--tune",
                str(TUNE),
            ]
        )
        assert rc == 0
        assert json.loads(out.read_text())["engine"] == "calibrated"

    def test_fit_missing_column_errors(self, tmp_path):
        inp = tmp_path / "bad.jsonl"
        inp.write_text(json.dumps({"claim": "x", "evidence": "y", "label": 1}), encoding="utf-8")
        with pytest.raises(SystemExit):
            cli_main(
                [
                    "calibration",
                    "fit",
                    "--input",
                    str(inp),
                    "-o",
                    str(tmp_path / "o.json"),
                    "--draws",
                    str(DRAWS),
                    "--tune",
                    str(TUNE),
                ]
            )

    def test_unsupported_format_errors(self, tmp_path):
        bad = tmp_path / "recs.txt"
        bad.write_text("nope", encoding="utf-8")
        with pytest.raises(SystemExit):
            cli_main(
                [
                    "calibration",
                    "fit",
                    "--input",
                    str(bad),
                    "-o",
                    str(tmp_path / "o.json"),
                    "--draws",
                    str(DRAWS),
                    "--tune",
                    str(TUNE),
                ]
            )
