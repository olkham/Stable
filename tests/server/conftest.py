"""Shared fixtures for the Phase 4 server tests.

Each test gets an isolated workspace directory and a fresh
:class:`~stable.server.jobs.JobManager` (its worker thread is a daemon and
harmless to leak across tests, but a fresh instance means no job from a
previous test can be observed by a later one).

Helper *functions* (``wait_job``, ``upload_clip``) live in ``_helpers.py``, not
here — see that module's docstring for why.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stable.server.app import create_app
from stable.server.jobs import reset_job_manager
from stable.server.storage import Workspace, reset_workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """A fresh, isolated :class:`Workspace` rooted under pytest's tmp_path."""
    ws = reset_workspace(tmp_path / "workspace")
    reset_job_manager()
    return ws


@pytest.fixture
def client(workspace: Workspace) -> Iterator[TestClient]:
    """A :class:`TestClient` wired to a fresh app over the isolated workspace."""
    with TestClient(create_app()) as test_client:
        yield test_client
