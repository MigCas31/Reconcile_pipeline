"""Subprocess job runner for dev-pipeline operations.

The Gemini chat surface dispatches mutating operations (rebuild, audit,
score) into here. Each job runs ``subprocess.Popen`` in the workspace root,
captures stdout/stderr, and exposes a ``status`` + ``log_tail`` for polling.

Jobs are kept in-memory only — restart the server to clear them. This is a
local dev tool, not production infrastructure.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid as _uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

_LOG_TAIL_LINES = 200


@dataclass
class Job:
    id: str
    cmd: list[str]
    env_overrides: dict[str, str]
    cwd: Path
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    returncode: int | None = None
    status: str = "running"  # running | succeeded | failed | killed
    _log: deque[str] = field(default_factory=lambda: deque(maxlen=_LOG_TAIL_LINES))
    _proc: subprocess.Popen | None = None

    def log_tail(self) -> str:
        return "\n".join(self._log)

    def to_public(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "cmd": self.cmd,
            "env_overrides": self.env_overrides,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "status": self.status,
            "log_tail": self.log_tail(),
        }


class JobRegistry:
    """Tracks running and finished jobs."""

    def __init__(self, workspace_root: Path = WORKSPACE_ROOT) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.workspace_root = workspace_root

    def start(
        self,
        cmd: Sequence[str],
        env_overrides: Mapping[str, str] | None = None,
    ) -> Job:
        env = os.environ.copy()
        overrides = dict(env_overrides or {})
        env.update(overrides)
        job = Job(
            id=_uuid.uuid4().hex[:12],
            cmd=list(cmd),
            env_overrides=overrides,
            cwd=self.workspace_root,
        )
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(self.workspace_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job._proc = proc
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._reader_thread, args=(job,), daemon=True).start()
        return job

    def _reader_thread(self, job: Job) -> None:
        proc = job._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                job._log.append(line.rstrip("\n"))
        finally:
            proc.wait()
            job.returncode = proc.returncode
            job.finished_at = time.time()
            job.status = (
                "succeeded"
                if proc.returncode == 0
                else ("killed" if job.status == "killed" else "failed")
            )

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def kill(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job._proc is None:
            return False
        job.status = "killed"
        try:
            job._proc.terminate()
            return True
        except Exception:
            return False

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


# Global singleton used by the HTTP server.
REGISTRY = JobRegistry()
