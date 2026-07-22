"""HuggingFace-token warning: anonymous Hub downloads are bandwidth-throttled, so
``settings.warn_if_no_hf_token`` warns once before a model IR is pulled from the Hub.

The helper returns True iff it warned, which is what these tests assert (no need to
capture loguru output). Each case pins one branch; the wiring test pins that the
dominant cascade fetch (``semantic_ov._resolve_repo_dir``) actually calls it.
"""

import huggingface_hub
import pytest

from groundrails import semantic_ov, settings


@pytest.fixture(autouse=True)
def _clean_token_env(monkeypatch):
    """Deterministic token state: clear the env vars and the login-cache lookup so a
    real ``~/.cache/huggingface/token`` on the host cannot leak into the assertions."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    settings.reset()
    settings.mark_ready()
    yield
    settings.reset()


def test_warns_once_when_no_token():
    assert settings.warn_if_no_hf_token() is True
    # memoized: a second call in the same process stays quiet
    assert settings.warn_if_no_hf_token() is False


def test_silent_when_token_present(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_dummy")
    assert settings.warn_if_no_hf_token() is False


def test_silent_when_hf_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert settings.warn_if_no_hf_token() is False


def test_cascade_resolve_checks_token(monkeypatch):
    """``_resolve_repo_dir`` must warn before the Hub snapshot when no local mirror."""
    monkeypatch.delenv("GROUNDRAILS_MODELS_DIR", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo: "/tmp/fake-ir")
    called = []
    monkeypatch.setattr(settings, "warn_if_no_hf_token", lambda: called.append(True))
    assert semantic_ov._resolve_repo_dir("bge-m3") == "/tmp/fake-ir"
    assert called == [True]
