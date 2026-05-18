"""Subprocess job runner tests."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from reconcile_tiers.jobs import JobRegistry


@pytest.fixture
def registry(tmp_path: Path) -> JobRegistry:
    return JobRegistry(workspace_root=tmp_path)


def _wait_until_done(reg: JobRegistry, job_id: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = reg.get(job_id)
        if job and job.finished_at is not None:
            return
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_job_succeeds_and_captures_stdout(registry: JobRegistry) -> None:
    job = registry.start([sys.executable, "-c", "print('hello-job')"])
    _wait_until_done(registry, job.id)
    assert job.status == "succeeded"
    assert job.returncode == 0
    assert "hello-job" in job.log_tail()


def test_job_failure_marks_failed(registry: JobRegistry) -> None:
    job = registry.start([sys.executable, "-c", "import sys; sys.exit(3)"])
    _wait_until_done(registry, job.id)
    assert job.status == "failed"
    assert job.returncode == 3


def test_job_env_overrides_visible_to_child(registry: JobRegistry) -> None:
    job = registry.start(
        [
            sys.executable,
            "-c",
            "import os; print('TEST_VAR=' + os.environ.get('TEST_VAR', '?'))",
        ],
        env_overrides={"TEST_VAR": "1825a812"},
    )
    _wait_until_done(registry, job.id)
    assert "TEST_VAR=1825a812" in job.log_tail()


def test_to_public_payload_shape(registry: JobRegistry) -> None:
    job = registry.start([sys.executable, "-c", "print('x')"])
    _wait_until_done(registry, job.id)
    public = job.to_public()
    assert public["job_id"] == job.id
    assert public["status"] == "succeeded"
    assert "log_tail" in public
    assert "env_overrides" in public


def test_kill_terminates_running_job(registry: JobRegistry) -> None:
    job = registry.start([sys.executable, "-c", "import time; time.sleep(30)"])
    time.sleep(0.2)
    assert registry.kill(job.id)
    _wait_until_done(registry, job.id, timeout=5.0)
    assert job.status == "killed"


def test_list_returns_started_jobs(registry: JobRegistry) -> None:
    a = registry.start([sys.executable, "-c", "print('a')"])
    b = registry.start([sys.executable, "-c", "print('b')"])
    ids = {j.id for j in registry.list()}
    assert {a.id, b.id} <= ids
