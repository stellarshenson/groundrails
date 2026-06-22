"""Tests for the one-call bootstrap + 3-way resource resolution (S3 -> local -> HF).

Model-free and network-free: the S3 client is a fake (monkeypatched), URL fetch
monkeypatches urllib, and model weights are never downloaded - the cascade
fallback is not exercised because every test supplies the resource via S3 or
local, or skips models.
"""

from __future__ import annotations

import io
import json
import os

import pytest

from groundrails import bootstrap, calibration, settings


# --- fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env():
    """Snapshot + restore the provisioning env vars so a test's mirror paths
    (``SAT_OV_IR`` / ``GROUNDRAILS_MODELS_DIR``, set via direct ``os.environ``)
    never leak into later test files and break real model loading."""
    keys = (settings.ENV_HOME, settings.ENV_CALIBRATION, settings.ENV_MODELS_DIR, "SAT_OV_IR")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    settings.reset()
    yield
    for k in keys:
        if saved[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = saved[k]
    settings.reset()


class FakeS3:
    """Minimal botocore-S3 stand-in over an in-memory ``{(bucket, key): bytes}``."""

    def __init__(self, store):
        self.store = store

    def get_object(self, Bucket, Key):  # noqa: N803 - botocore casing
        if (Bucket, Key) not in self.store:
            raise KeyError(f"no such key s3://{Bucket}/{Key}")
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix):  # noqa: N803
        contents = [{"Key": k} for (b, k) in self.store if b == Bucket and k.startswith(Prefix)]
        return {"Contents": contents} if contents else {}


def _bundled_calibration_bytes(tmp_path):
    p = bootstrap.export_calibration(tmp_path / "src.json")
    return p.read_bytes()


# --- scheme + split --------------------------------------------------------


def test_scheme_and_split():
    assert bootstrap._scheme("s3://b/k") == "s3"
    assert bootstrap._scheme("https://x/y") == "url"
    assert bootstrap._scheme("/local/path") == "local"
    assert bootstrap._split_s3("s3://bucket/a/b/c.json") == ("bucket", "a/b/c.json")


# --- _fetch_to: the three transports ---------------------------------------


def test_fetch_to_local(tmp_path):
    src = tmp_path / "a.json"
    src.write_text('{"x": 1}')
    dest = tmp_path / "out" / "a.json"
    assert bootstrap._fetch_to(str(src), dest) is True
    assert json.loads(dest.read_text()) == {"x": 1}


def test_fetch_to_local_missing(tmp_path):
    assert bootstrap._fetch_to(str(tmp_path / "nope.json"), tmp_path / "o.json") is False


def test_fetch_to_url(tmp_path, monkeypatch):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda uri: _Resp(b'{"u": 2}'), raising=True
    )
    dest = tmp_path / "u.json"
    assert bootstrap._fetch_to("https://host/c.json", dest) is True
    assert json.loads(dest.read_text()) == {"u": 2}


def test_fetch_to_s3(tmp_path):
    client = FakeS3({("bkt", "k.json"): b'{"s": 3}'})
    dest = tmp_path / "s.json"
    assert bootstrap._fetch_to("s3://bkt/k.json", dest, client=client) is True
    assert json.loads(dest.read_text()) == {"s": 3}
    assert bootstrap._fetch_to("s3://bkt/missing.json", tmp_path / "m.json", client=client) is False


# --- calibration resolution: the precedence branches -----------------------


def _init_caltest(monkeypatch, store, tmp_path, **kw):
    monkeypatch.setattr(bootstrap, "_s3_client", lambda *a, **k: FakeS3(store))
    return bootstrap.init(
        home=str(tmp_path / "home"), models="none", wordnet=False, **kw
    )


def test_resolution_explicit_override_wins(tmp_path, monkeypatch):
    cal = _bundled_calibration_bytes(tmp_path)
    store = {("groundrails-dev", "calibration.json"): cal}
    # an explicit s3 override is used even though a local source is also given
    summary = _init_caltest(
        monkeypatch,
        store,
        tmp_path,
        calibration="s3://groundrails-dev/calibration.json",
        source=str(tmp_path),
    )
    assert summary["calibration"]["source_used"] == "s3"


def test_resolution_s3_from_source(tmp_path, monkeypatch):
    cal = _bundled_calibration_bytes(tmp_path)
    store = {("groundrails-dev", "calibration.json"): cal}
    summary = _init_caltest(monkeypatch, store, tmp_path, source="s3://groundrails-dev")
    assert summary["calibration"]["source_used"] == "s3"
    # provisioned calibration is now active
    assert calibration.load_calibration_from_config() is not None


def test_resolution_local_fallback(tmp_path, monkeypatch):
    src = tmp_path / "assets"
    src.mkdir()
    blk = json.loads(bootstrap.export_calibration(tmp_path / "x.json").read_text())
    blk["threshold"] = 0.654
    (src / "calibration.json").write_text(json.dumps(blk))
    summary = _init_caltest(monkeypatch, {}, tmp_path, source=str(src))
    assert summary["calibration"]["source_used"] == "local"
    assert calibration.load_calibration_from_config()["threshold"] == 0.654


def test_resolution_bundled_when_no_source(tmp_path, monkeypatch):
    summary = _init_caltest(monkeypatch, {}, tmp_path)
    assert summary["calibration"]["source_used"] == "bundled"
    assert summary["calibration"]["path"] is None


def test_explicit_calibration_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "_s3_client", lambda *a, **k: FakeS3({}))
    with pytest.raises(FileNotFoundError):
        bootstrap.init(
            calibration="s3://groundrails-dev/calibration.json",
            home=str(tmp_path / "home"),
            models="none",
            wordnet=False,
        )


# --- calibration install + active + no settings.json -----------------------


def test_provisioned_calibration_becomes_active(tmp_path, monkeypatch):
    src = tmp_path / "assets"
    src.mkdir()
    blk = json.loads(bootstrap.export_calibration(tmp_path / "x.json").read_text())
    blk["threshold"] = 0.777
    (src / "calibration.json").write_text(json.dumps(blk))
    home = tmp_path / "home"
    bootstrap.init(source=str(src), models="none", wordnet=False, home=str(home))
    assert (home / "calibration.json").is_file()
    assert calibration.load_calibration_from_config()["threshold"] == 0.777
    # no settings file is ever written
    assert not list(home.rglob("settings.json"))


def test_init_writes_no_settings_json(tmp_path, monkeypatch):
    home = tmp_path / "home"
    bootstrap.init(
        semantic_model="custom/model",
        cache_dir=str(tmp_path / "cache"),
        models="none",
        wordnet=False,
        home=str(home),
    )
    assert settings.get().semantic_model == "custom/model"
    # no `.stellars-plugins/settings.json` anywhere - the rejected convention
    assert not list(tmp_path.rglob("settings.json"))
    # init DOES write the runtime config to <home>/groundrails.json
    assert (home / "groundrails.json").is_file()


def test_grounding_before_init_raises():
    """Hard gate: grounding before init() refuses rather than running un-provisioned."""
    from groundrails import grounding

    settings.reset()  # clears the autouse ready flag - simulate "init never called"
    with pytest.raises(settings.NotInitializedError):
        grounding.ground_batch(["a claim"], ["some evidence"])


def test_cli_ground_refuses_without_init(tmp_path, monkeypatch, capsys):
    """`groundrails ground` with no groundrails.json exits 2 with an init hint."""
    from groundrails import cli

    settings.reset()
    # point at a guaranteed-absent config so no stray ~/.cache/groundrails/groundrails.json loads
    monkeypatch.setenv(settings.ENV_CONFIG, str(tmp_path / "absent.json"))
    rc = cli.main(["ground", "--claim", "x", str(tmp_path / "ev.txt")])
    assert rc == 2
    assert "not initialized" in capsys.readouterr().err


# --- model mirror ----------------------------------------------------------


def test_models_none_skips(tmp_path, monkeypatch):
    summary = _init_caltest(monkeypatch, {}, tmp_path)
    assert summary["models"]["source_used"] == "skipped"


def test_models_mirror_from_s3(tmp_path, monkeypatch):
    from groundrails import semantic_ov

    store = {}
    names = [*semantic_ov.HF_REPOS.keys(), "sat"]
    for name in names:
        store[("groundrails-dev", f"models/{name}/openvino_model.xml")] = b"<ir/>"
        store[("groundrails-dev", f"models/{name}/openvino_model.bin")] = b"\x00"
    monkeypatch.setattr(bootstrap, "_s3_client", lambda *a, **k: FakeS3(store))
    home = tmp_path / "home"
    summary = bootstrap.init(
        source="s3://groundrails-dev", wordnet=False, home=str(home)
    )
    used = summary["models"]["source_used"]
    for name in semantic_ov.HF_REPOS:
        assert used[name] == "s3"
    # the local mirror + SaT env are wired for offline loading
    assert settings.get().models_dir == str(home / "models")
    assert (home / "models" / "bge-m3" / "openvino_model.xml").is_file()
    assert os.environ.get("SAT_OV_IR", "").endswith("openvino_model.xml")


# --- summary shape ---------------------------------------------------------


def test_summary_keys(tmp_path, monkeypatch):
    summary = _init_caltest(monkeypatch, {}, tmp_path, languages=None)
    assert set(summary) >= {"home", "calibration", "models", "languages", "wordnet"}
