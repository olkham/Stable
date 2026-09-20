"""Tests for lazy, checksummed model-weight download.

Every download here is served from a ``file://`` URL in a temp directory, so the
whole acquisition path — transfer, verification, caching, licence gating — is
exercised offline and in milliseconds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from stable.core.errors import WeightsError
from stable.core.paths import data_root
from stable.trackers.cotracker3 import COTRACKER3_OFFLINE, CoTracker3Tracker
from stable.trackers.tapir import BOOTSTAPIR, TapirTracker
from stable.trackers.weights import (
    ENV_ACCEPT_NONCOMMERCIAL,
    ENV_ALLOW_UNPINNED,
    ENV_MODEL_DIR,
    ModelSpec,
    clear_weights,
    describe_models,
    ensure_weights,
    expected_sha256,
    is_downloaded,
    local_path,
    model_root,
    register_model,
    registered_models,
    sha256_file,
)

PAYLOAD = b"pretend this is 900 MB of transformer weights\n" * 64
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def cache(tmp_path, monkeypatch) -> Path:
    """Point the model cache at a temp directory for the duration of a test."""
    root = tmp_path / "models"
    monkeypatch.setenv(ENV_MODEL_DIR, str(root))
    monkeypatch.delenv(ENV_ALLOW_UNPINNED, raising=False)
    monkeypatch.delenv(ENV_ACCEPT_NONCOMMERCIAL, raising=False)
    return root


@pytest.fixture
def served(tmp_path) -> str:
    """Write the fake checkpoint to disk and return a file:// URL for it."""
    source = tmp_path / "served" / "weights.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PAYLOAD)
    return source.as_uri()


def spec_for(url: str, **overrides) -> ModelSpec:
    """Build a throwaway ModelSpec pointing at ``url``."""
    fields = {"key": "test_model", "url": url, "sha256": DIGEST}
    fields.update(overrides)
    return ModelSpec(**fields)


# -- cache layout ---------------------------------------------------------


def test_cache_root_defaults_to_the_data_root(monkeypatch):
    monkeypatch.delenv(ENV_MODEL_DIR, raising=False)
    assert model_root() == data_root() / "models"


def test_cache_root_is_overridable(cache):
    assert model_root() == cache
    assert local_path(spec_for("file:///x")) == cache / "test_model" / "x"


def test_filename_defaults_to_the_url_basename():
    assert ModelSpec(key="k", url="https://host/dir/model_v2.pth").filename == "model_v2.pth"
    assert ModelSpec(key="k", url="https://host/dir/x.pth", filename="mine.pth").filename


# -- the happy path -------------------------------------------------------


def test_download_verifies_and_caches(cache, served):
    spec = spec_for(served)
    assert not is_downloaded(spec)

    seen: list[float] = []
    path = ensure_weights(spec, progress=lambda f, m="": seen.append(f))

    assert path.read_bytes() == PAYLOAD
    assert path == cache / "test_model" / "weights.bin"
    assert is_downloaded(spec)
    assert seen and seen[-1] == 1.0


def test_a_cached_checkpoint_is_never_re_downloaded(cache, served, tmp_path):
    spec = spec_for(served)
    ensure_weights(spec)
    Path(tmp_path / "served" / "weights.bin").unlink()  # the server goes away

    assert ensure_weights(spec).read_bytes() == PAYLOAD  # still fine: it is cached


def test_verification_is_cached_between_calls(cache, served, monkeypatch):
    spec = spec_for(served)
    ensure_weights(spec)

    calls = {"n": 0}
    real = sha256_file

    def counting(path, **kwargs):
        calls["n"] += 1
        return real(path, **kwargs)

    monkeypatch.setattr("stable.trackers.weights.sha256_file", counting)
    ensure_weights(spec)
    ensure_weights(spec)
    assert calls["n"] == 0  # the size/mtime stamp answers instead of re-hashing


def test_touching_the_file_invalidates_the_stamp(cache, served):
    spec = spec_for(served)
    path = ensure_weights(spec)
    path.write_bytes(PAYLOAD + b"tampered")

    with pytest.raises(WeightsError, match="corrupt"):
        ensure_weights(spec)


# -- failure modes --------------------------------------------------------


def test_checksum_mismatch_deletes_the_download(cache, served):
    spec = spec_for(served, sha256="0" * 64)
    with pytest.raises(WeightsError, match="checksum mismatch"):
        ensure_weights(spec)
    assert not local_path(spec).exists()  # a bad file is never left lying around


def test_a_missing_url_is_reported_not_swallowed(cache, tmp_path):
    spec = spec_for((tmp_path / "nope.bin").as_uri())
    with pytest.raises(WeightsError, match="could not download"):
        ensure_weights(spec)
    assert not local_path(spec).exists()


def test_offline_mode_refuses_to_reach_the_network(cache, served):
    spec = spec_for(served)
    with pytest.raises(WeightsError, match="downloading is disabled"):
        ensure_weights(spec, allow_download=False)


def test_unpinned_weights_need_an_explicit_opt_in(cache, served, caplog):
    spec = spec_for(served, sha256=None)
    with pytest.raises(WeightsError, match="no sha256 is pinned"):
        ensure_weights(spec)

    path = ensure_weights(spec, allow_unpinned=True)
    assert path.read_bytes() == PAYLOAD
    assert any(DIGEST in record.getMessage() for record in caplog.records), (
        "the digest of an unpinned download must be logged so it can be pinned"
    )


def test_unpinned_opt_in_via_environment(cache, served, monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_UNPINNED, "1")
    assert ensure_weights(spec_for(served, sha256=None)).exists()


def test_non_commercial_weights_need_an_explicit_opt_in(cache, served, monkeypatch):
    spec = spec_for(served, non_commercial=True, license="CC-BY-NC-4.0")
    with pytest.raises(WeightsError, match="non-commercial"):
        ensure_weights(spec)

    assert ensure_weights(spec, accept_noncommercial=True).exists()
    clear_weights(spec)
    monkeypatch.setenv(ENV_ACCEPT_NONCOMMERCIAL, "1")
    assert ensure_weights(spec).exists()


def test_an_already_downloaded_non_commercial_model_is_usable(cache, served):
    spec = spec_for(served, non_commercial=True)
    ensure_weights(spec, accept_noncommercial=True)
    # The gate is on *acquiring* the weights, not on running them.
    assert ensure_weights(spec).exists()


def test_environment_pin_overrides_the_spec(cache, served, monkeypatch):
    spec = spec_for(served, sha256=None)
    monkeypatch.setenv("STABLE_MODEL_SHA256_TEST_MODEL", DIGEST.upper())
    assert expected_sha256(spec) == DIGEST
    assert ensure_weights(spec).exists()  # pinned by the env, so no opt-in needed

    monkeypatch.setenv("STABLE_MODEL_SHA256_TEST_MODEL", "nonsense")
    with pytest.raises(WeightsError, match="64 hex"):
        expected_sha256(spec)


def test_spec_validation():
    with pytest.raises(WeightsError, match="key"):
        ModelSpec(key="  ", url="https://host/x")
    with pytest.raises(WeightsError, match="url"):
        ModelSpec(key="k", url="")
    with pytest.raises(WeightsError, match="64 hex"):
        ModelSpec(key="k", url="https://host/x", sha256="abc")


def test_clear_removes_the_file_and_its_stamp(cache, served):
    spec = spec_for(served)
    path = ensure_weights(spec)
    assert clear_weights(spec) is True
    assert not path.exists()
    assert not list(path.parent.glob("*.sha256.json"))
    assert clear_weights(spec) is False


# -- the model table ------------------------------------------------------


def test_registering_the_same_spec_twice_is_a_no_op():
    spec = ModelSpec(key="dup_model", url="https://host/x", sha256=DIGEST)
    register_model(spec)
    register_model(ModelSpec(key="dup_model", url="https://host/x", sha256=DIGEST))
    with pytest.raises(WeightsError, match="already registered"):
        register_model(ModelSpec(key="dup_model", url="https://host/other"))


def test_the_shipped_plugins_register_their_checkpoints():
    keys = {spec.key for spec in registered_models()}
    assert {"cotracker3_offline", "bootstapir"} <= keys

    rows = {row["key"]: row for row in describe_models()}
    assert rows["cotracker3_offline"]["non_commercial"] is True
    assert rows["cotracker3_offline"]["license"] == "CC-BY-NC-4.0"
    assert rows["bootstapir"]["non_commercial"] is False
    assert rows["bootstapir"]["license"] == "Apache-2.0"
    assert set(rows["bootstapir"]) == {
        "key",
        "filename",
        "url",
        "license",
        "non_commercial",
        "pinned",
        "path",
        "downloaded",
    }


def test_plugin_licences_match_their_weights():
    assert CoTracker3Tracker.license == COTRACKER3_OFFLINE.license == "CC-BY-NC-4.0"
    assert TapirTracker.license == BOOTSTAPIR.license == "Apache-2.0"


def test_plugins_report_whether_their_weights_are_cached(cache):
    assert CoTracker3Tracker.weights_downloaded() is False
    assert TapirTracker.weights_downloaded() is False
    assert CoTracker3Tracker.weights_path().endswith("cotracker3_scaled_offline.pth")
    assert TapirTracker.weights_path().endswith("bootstapir_checkpoint_v2.pt")
