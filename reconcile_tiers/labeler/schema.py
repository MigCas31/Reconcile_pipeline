"""Schemas for the generic labeler.

The labeler works on *runs*, each pinned to a single ``decision_type``. A run
contains one ``RunMeta`` and many ``Case`` records. Humans contribute
``Label`` records, appended to a separate JSONL.

Geometry passed to the viewer is intentionally minimal: each option carries a
list of triangle-fan polygons (``corners`` per polygon, in world coordinates)
plus a colour hint. The labeler renders these on top of the building's
``tier_payload.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CaseOption:
    """One candidate reconstruction the human can pick.

    ``polygons`` is a list of polygons; each polygon is a list of [x, y, z]
    corners in world coordinates. The viewer renders each polygon as a filled
    convex/triangulated mesh.
    """

    id: str
    label: str
    polygons: list[list[list[float]]]
    color: str = "#7dd3fc"


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    building_uuid: str
    decision_type: str
    locator_id: str
    options: list[CaseOption]
    camera_target: list[float]
    camera_offset: list[float] = field(default_factory=lambda: [12.0, 8.0, 12.0])
    features: dict[str, Any] = field(default_factory=dict)
    heuristic_label: str | None = None
    notes: str | None = None
    # Building elements (rooms, walls, floors) whose locator_id is in this
    # list will be drawn in a brighter colour to give the labeler architectural
    # context — typically the room containing the element under decision.
    highlight_locators: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Label:
    case_id: str
    decision_type: str
    selected_option_id: str | None
    label_kind: str  # "select" | "skip" | "unsure"
    reasons: list[str]
    ts: float
    labeler: str


@dataclass(frozen=True, slots=True)
class RunMeta:
    run_id: str
    decision_type: str
    generator: str
    generator_version: str
    created_at: float
    total_cases: int
    description: str = ""


def case_to_dict(case: Case) -> dict[str, Any]:
    out = asdict(case)
    out["options"] = [asdict(opt) for opt in case.options]
    return out


def label_to_dict(label: Label) -> dict[str, Any]:
    return asdict(label)


def run_meta_to_dict(meta: RunMeta) -> dict[str, Any]:
    return asdict(meta)
