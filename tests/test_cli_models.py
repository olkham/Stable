"""Tests for the ``stable models`` CLI subcommand.

Downloads are served from ``file://`` URLs (see ``tests/test_weights.py``),
so this runs offline. Each test registers its own throwaway
:class:`~stable.trackers.weights.ModelSpec` under a unique key, since the
model table is process-wide (shared with the real shipped specs).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from stable.cli.main import main
from stable.trackers.weights import ENV_MODEL_DIR, ModelSpec, is_downloaded, register_model

PAYLOAD = b"pretend checkpoint bytes\n" * 8
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture(autouse=True)
def _model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    monkeypatch.setenv(ENV_MODEL_DIR, str(root))
    monkeypatch.delenv("STABLE_ALLOW_UNPINNED_WEIGHTS", raising=False)
    monkeypatch.delenv("STABLE_ACCEPT_NONCOMMERCIAL", raising=False)
    return root


@pytest.fixture
def served(tmp_path: Path) -> str:
    source = tmp_path / "served" / "weights.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PAYLOAD)
    return source.as_uri()


def _unique_key() -> str:
    return "clitest_" + uuid4().hex[:10]


def test_list_includes_the_shipped_checkpoints(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "bootstapir" in out
    assert "cotracker3_offline" in out


def test_list_json_is_machine_readable(capsys):
    assert main(["models", "--json"]) == 0
    rows = {row["key"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["bootstapir"]["license"] == "Apache-2.0"
    assert rows["cotracker3_offline"]["non_commercial"] is True


def test_download_then_remove_round_trip(served, capsys):
    key = _unique_key()
    spec = register_model(ModelSpec(key=key, url=served, sha256=DIGEST))
    assert not is_downloaded(spec)

    assert main(["models", "download", key]) == 0
    assert "ready at" in capsys.readouterr().out
    assert is_downloaded(spec)

    assert main(["models", "remove", key]) == 0
    assert "removed" in capsys.readouterr().out
    assert not is_downloaded(spec)


def test_download_unpinned_fails_closed_without_the_flag(served, capsys):
    key = _unique_key()
    register_model(ModelSpec(key=key, url=served, sha256=None))

    assert main(["models", "download", key]) == 1
    assert "sha256" in capsys.readouterr().err

    assert main(["models", "download", key, "--allow-unpinned"]) == 0


def test_download_non_commercial_fails_closed_without_the_flag(served, capsys):
    key = _unique_key()
    register_model(
        ModelSpec(key=key, url=served, sha256=DIGEST, non_commercial=True, license="CC-BY-NC-4.0")
    )

    assert main(["models", "download", key]) == 1
    assert "non-commercial" in capsys.readouterr().err

    assert main(["models", "download", key, "--accept-noncommercial"]) == 0


def test_download_reports_an_unknown_key_cleanly(capsys):
    assert main(["models", "download", "not_a_real_model"]) == 1
    assert "not_a_real_model" in capsys.readouterr().err


def test_remove_of_an_uncached_checkpoint_is_not_an_error(served, capsys):
    key = _unique_key()
    register_model(ModelSpec(key=key, url=served, sha256=DIGEST))
    assert main(["models", "remove", key]) == 0
    assert "was not cached" in capsys.readouterr().out
