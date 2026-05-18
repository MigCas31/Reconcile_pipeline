"""Dev / pipeline tools exposed to the Gemini chat surface only.

These wrap the CLI patterns the user runs most often per
``tracking_progress.md``. Mutating tools (``rebuild_*``, ``apply_threshold_tweak``,
``audit_*``) require an explicit ``confirmed=True`` argument; without it
they return ``{"requires_confirmation": True, "command": ..., ...}`` so the
agent's caller can show a dialog and re-issue once the user approves.

These tools are **never** exposed to the right-click menu. They live behind
the ``/gemini/chat`` endpoint and are gated by ``GEMINI_API_KEY``.

Note: this module deliberately does **not** use ``from __future__ import
annotations``. The Gemini SDK's automatic-function-calling argument coercion
(``google.genai._extra_utils.convert_if_exist_pydantic_model``) does
``isinstance(value, annotation)``, which fails when annotations are strings.
Keep parameter types as real type objects.
"""

import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reconcile_tiers import jobs, snapshots, threshold_registry

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
TRACKING_PROGRESS = WORKSPACE_ROOT / "tracking_progress.md"
PIPELINE_OUTPUTS = WORKSPACE_ROOT / "pipeline-outputs"


# ---- helpers ------------------------------------------------------------------


def _confirm_envelope(
    confirmed: bool,
    *,
    command: list[str],
    summary: str,
    est_runtime_s: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a confirmation envelope when not yet confirmed; else None."""
    if confirmed:
        return None
    payload: dict[str, Any] = {
        "requires_confirmation": True,
        "summary": summary,
        "command": " ".join(command),
        "argv": command,
    }
    if est_runtime_s is not None:
        payload["est_runtime_s"] = est_runtime_s
    if extra:
        payload.update(extra)
    return payload


def _start_job(
    cmd: list[str], env_overrides: Mapping[str, str] | None = None
) -> dict[str, Any]:
    job = jobs.REGISTRY.start(cmd, env_overrides=env_overrides)
    return {
        "job_id": job.id,
        "command": " ".join(cmd),
        "env_overrides": dict(env_overrides or {}),
    }


# ---- tools --------------------------------------------------------------------


def validate_building(uuid: str) -> dict[str, Any]:
    """Validate one building (no rebuild). Pure read-only check."""
    cmd = [
        sys.executable,
        "-m",
        "reconcile_tiers.build",
        "--uuid",
        uuid,
        "--validate-only",
    ]
    return _start_job(cmd)


def rebuild_building(
    uuid: str,
    env_overrides: dict | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Rebuild one building. Mutating; takes a snapshot first."""
    cmd = [
        sys.executable,
        "-m",
        "reconcile_tiers.build",
        "--uuid",
        uuid,
        "--force",
    ]
    envelope = _confirm_envelope(
        confirmed,
        command=cmd,
        summary=f"Rebuild building {uuid} (overwrites tier_payload.json)",
        est_runtime_s=8,
        extra={"env_overrides": env_overrides or {}},
    )
    if envelope is not None:
        return envelope
    snapshot_path: str | None = None
    try:
        snapshot_path = str(snapshots.take(uuid))
    except FileNotFoundError:
        snapshot_path = None
    result = _start_job(cmd, env_overrides=env_overrides)
    if snapshot_path:
        result["pre_rebuild_snapshot"] = snapshot_path
    return result


def rebuild_corpus(
    env_overrides: dict | None = None,
    parallelism: int = 8,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Rebuild ALL buildings. Slow + high blast radius; double confirmation."""
    cmd = [
        sys.executable,
        "-m",
        "reconcile_tiers.build",
        "--all",
        "--force",
        "-j",
        str(parallelism),
    ]
    envelope = _confirm_envelope(
        confirmed,
        command=cmd,
        summary=(
            "Rebuild ALL buildings (~272s, overwrites every tier_payload.json). "
            "Snapshot the suspect ones first if you care about diffs."
        ),
        est_runtime_s=300,
        extra={"env_overrides": env_overrides or {}, "destructive": True},
    )
    if envelope is not None:
        return envelope
    return _start_job(cmd, env_overrides=env_overrides)


def audit_building(uuid: str) -> dict[str, Any]:
    """Run the cohort_scan defect rules on one building. Pure."""
    cmd = [
        sys.executable,
        "-m",
        "reconcile_tiers.audit.cohort_scan",
        "--uuid",
        uuid,
    ]
    return _start_job(cmd)


def audit_corpus(confirmed: bool = False) -> dict[str, Any]:
    """Run the cohort_scan defect rules across all buildings. Slow."""
    cmd = [sys.executable, "-m", "reconcile_tiers.audit.cohort_scan"]
    envelope = _confirm_envelope(
        confirmed,
        command=cmd,
        summary="Run defect audit across all 200+ buildings (slow, read-only).",
        est_runtime_s=120,
    )
    if envelope is not None:
        return envelope
    return _start_job(cmd)


def list_thresholds() -> dict[str, Any]:
    """Return the threshold registry the model can act on."""
    return {"entries": threshold_registry.list_entries()}


def read_threshold(name: str) -> dict[str, Any]:
    """Look up the current value of a threshold from the source file."""
    value, path, line = threshold_registry.read_current(name)
    return {"name": name, "value": value, "file": path, "line": line}


def apply_threshold_tweak(
    name: str,
    new_value: float,
    rebuild_uuid: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Edit a threshold constant in source.

    Always confirmation-gated. ``rebuild_uuid`` optionally chains a single
    building rebuild after the edit.
    """
    diff = threshold_registry.make_diff(name, float(new_value))
    summary_lines = [
        f"Edit {name} in {diff['file']}:{diff['line']}",
        f"  - {diff['diff']['-']}",
        f"  + {diff['diff']['+']}",
    ]
    if rebuild_uuid:
        summary_lines.append(f"Then rebuild building {rebuild_uuid}.")
    envelope = _confirm_envelope(
        confirmed,
        command=["edit", diff["file"]],
        summary="\n".join(summary_lines),
        extra={"diff": diff, "rebuild_uuid": rebuild_uuid},
    )
    if envelope is not None:
        return envelope
    threshold_registry.apply(name, float(new_value))
    out: dict[str, Any] = {"applied": diff}
    if rebuild_uuid:
        out["rebuild"] = rebuild_building(rebuild_uuid, confirmed=True)
    return out


def read_tracking_progress(grep_term: str, max_chars: int = 4000) -> dict[str, Any]:
    """Return matching context blocks from tracking_progress.md.

    Returns up to ``max_chars`` characters of paragraphs that contain the
    grep term (case-insensitive). The model uses this to avoid suggesting
    abandoned approaches or to recall past parameter tries.
    """
    if not TRACKING_PROGRESS.exists():
        return {"matches": [], "note": "tracking_progress.md not found"}
    text = TRACKING_PROGRESS.read_text()
    needle = grep_term.lower()
    paragraphs = re.split(r"\n\n+", text)
    matches: list[str] = []
    total = 0
    for para in paragraphs:
        if needle in para.lower():
            if total + len(para) > max_chars:
                matches.append(para[: max_chars - total])
                break
            matches.append(para)
            total += len(para)
    return {"grep_term": grep_term, "match_count": len(matches), "matches": matches}


def compare_before_after(
    uuid: str, before: str | None = None, after: str | None = None
) -> dict[str, Any]:
    """Diff two snapshots of a building's tier_payload."""
    available = snapshots.list_snapshots(uuid)
    if not available:
        return {"error": f"no snapshots for {uuid}"}
    before_path = (
        Path(before) if before else available[-2 if len(available) >= 2 else 0]
    )
    current = PIPELINE_OUTPUTS / uuid / "tier_payload.json"
    after_path = Path(after) if after else current
    return snapshots.diff(before_path, after_path)


def take_snapshot(uuid: str) -> dict[str, Any]:
    """Manually capture the current tier_payload as a snapshot."""
    path = snapshots.take(uuid)
    return {"uuid": uuid, "snapshot": str(path)}


def inspect_building(uuid: str) -> dict[str, Any]:
    """Return a concise overview of a building's current tier_payload.

    The full inspect-building skill emits a multi-file report; this is a
    lightweight in-process variant suitable for chat answers. Pure.
    """
    payload_path = PIPELINE_OUTPUTS / uuid / "tier_payload.json"
    if not payload_path.exists():
        return {"error": f"tier_payload.json missing for {uuid}"}
    payload = json.loads(payload_path.read_text())
    rooms = payload.get("rooms") or []
    summary = {
        "uuid": uuid,
        "address": payload.get("address"),
        "classification": payload.get("classification"),
        "story_labels": payload.get("story_labels"),
        "counts": {
            "rooms": len(rooms),
            "walls": sum(len(r.get("walls") or []) for r in rooms),
            "ceiling": len(payload.get("ceiling") or []),
            "knee_walls": len(payload.get("knee_walls") or []),
            "dormer_faces": len(payload.get("dormer_faces") or []),
            "gaps": len(payload.get("gaps") or []),
        },
    }
    return summary


def debug_element(locator: str) -> dict[str, Any]:
    """Resolve a locator to its tier_payload entry + adjacency.

    A pure-read counterpart to the /debug-element skill, suitable for the
    model to examine before suggesting a fix.
    """
    from reconcile_tiers import quick_actions as qa

    info = qa.element_info(locator)
    neighbors = qa.neighbors(locator).get("neighbors") or []
    return {"info": info, "neighbors": neighbors}


def score_ridge_eave(uuid: str) -> dict[str, Any]:
    """Run the ridge/eave scorer prototype on one building."""
    cmd = [
        sys.executable,
        "scripts/score_candidates_ridge_eave.py",
        "--uuid",
        uuid,
    ]
    return _start_job(cmd)


# ---- registry -----------------------------------------------------------------


@dataclass(frozen=True)
class DevTool:
    name: str
    description: str
    fn: Callable[..., dict[str, Any]]


REGISTRY: dict[str, DevTool] = {
    tool.name: tool
    for tool in [
        DevTool(
            "validate_building",
            "Validate one building's tier_payload without overwriting it. Args: "
            "uuid:str.",
            validate_building,
        ),
        DevTool(
            "rebuild_building",
            "Rebuild one building (overwrites tier_payload.json). Args: uuid:str, "
            "env_overrides:dict, confirmed:bool=False.",
            rebuild_building,
        ),
        DevTool(
            "rebuild_corpus",
            (
                "Rebuild ALL ~200 buildings. Slow + destructive; double "
                "confirmation. Args: env_overrides:dict, parallelism:int=8, "
                "confirmed:bool=False."
            ),
            rebuild_corpus,
        ),
        DevTool(
            "audit_building",
            "Run defect rules on one building. Args: uuid:str.",
            audit_building,
        ),
        DevTool(
            "audit_corpus",
            "Run defect rules across all buildings. Args: confirmed:bool=False.",
            audit_corpus,
        ),
        DevTool(
            "list_thresholds",
            "List the tunable threshold constants exposed to apply_threshold_tweak.",
            list_thresholds,
        ),
        DevTool(
            "read_threshold",
            "Read the current source-of-truth value of a threshold. Args: name:str.",
            read_threshold,
        ),
        DevTool(
            "apply_threshold_tweak",
            "Edit a threshold constant; optionally rebuild one building. Args: "
            "name:str, new_value:float, rebuild_uuid:str|None, confirmed:bool=False.",
            apply_threshold_tweak,
        ),
        DevTool(
            "read_tracking_progress",
            (
                "Search tracking_progress.md for past attempts (grep). "
                "Args: grep_term:str, max_chars:int=4000."
            ),
            read_tracking_progress,
        ),
        DevTool(
            "compare_before_after",
            "Diff two snapshots of a building's tier_payload. Args: uuid:str, "
            "before:str|None, after:str|None.",
            compare_before_after,
        ),
        DevTool(
            "take_snapshot",
            "Manually snapshot a building's current tier_payload. Args: uuid:str.",
            take_snapshot,
        ),
        DevTool(
            "inspect_building",
            "Quick in-process overview of a building's tier_payload. Args: uuid:str.",
            inspect_building,
        ),
        DevTool(
            "debug_element",
            "Resolve a locator to its tier_payload entry + neighbors. Args: "
            "locator:str.",
            debug_element,
        ),
        DevTool(
            "score_ridge_eave",
            "Run the ridge/eave scorer prototype on one building. Args: uuid:str.",
            score_ridge_eave,
        ),
    ]
}


def dispatch(name: str, **params: Any) -> dict[str, Any]:
    if name not in REGISTRY:
        raise KeyError(f"unknown dev tool: {name!r}")
    return REGISTRY[name].fn(**params)


# Static context the agent should carry into every call.
KNOWN_PIVOTS: list[str] = [
    "GBM segment classifier abandoned in favour of BIP-style optimization.",
    "Shape-based ridge/eave scorer abandoned in favour of mirror-parity scoring.",
    "Per-room ceiling heuristics abandoned in favour of building-wide cell "
    "decomposition.",
    "Continuation surfaces demoted from committed shell to unresolved_region.",
    "Synthetic fallback ceilings demoted to unresolved (don't fake confidence).",
    "AZIMUTH_FILTER_THRESHOLD must remain 180°; 90° caused production regressions.",
    "tier_payload.json is derived; persistent fixes go upstream (extract / classify / "
    "build), not by editing the payload.",
]


__all__ = ["KNOWN_PIVOTS", "REGISTRY", "DevTool", "dispatch"]
