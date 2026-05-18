"""End-user quick actions for the viewer right-click menu.

These are pure or read-only operations on a single building's tier_payload
entry. They are called from the frontend via ``POST /context-action``.
**Never** invoke pipeline reruns, edit source files, or call external APIs;
those operations live in ``reconcile_tiers/dev_tools/`` and are exposed only
through the Gemini chat surface.

Each action takes a typed input model and returns a JSON-serializable dict.
The frontend is responsible for applying the result (swapping mesh corners,
showing an info panel, hiding a mesh, etc.).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_OUTPUTS = WORKSPACE_ROOT / "pipeline-outputs"

# ---- locator + payload helpers ------------------------------------------------

_LOCATOR_SCOPE_TO_PAYLOAD_KEY: dict[str, str] = {
    "ceiling-flat": "ceiling",
    "ceiling-slanted": "ceiling",
    "ceiling": "ceiling",
    "knee-wall": "knee_walls",
    "dormer-face": "dormer_faces",
    "gable-closure": "gable_closures",
    "gap-floor": "gaps",
    "gap-ceiling": "gaps",
    "gap-side": "gaps",
    "gap-stitch": "gaps",
    "gap-exterior-side": "gaps",
    "gap-exterior-floor": "gaps",
    "gap-exterior-ceiling": "gaps",
    "gap-stitch-floor": "gaps",
    "gap-stitch-ceiling": "gaps",
}


def parse_locator(locator: str) -> tuple[str, str, list[str]]:
    """Return (uuid, scope, parts) for a tier-* locator string."""
    segments = locator.split("::", 2)
    if len(segments) != 3:
        raise ValueError(f"invalid locator: {locator!r}")
    uuid_part, kind_part, id_part = segments
    if not kind_part.startswith("tier-"):
        raise ValueError(f"only tier-* locators supported, got {kind_part!r}")
    scope = kind_part[len("tier-") :]
    parts = id_part.split(":") if id_part else []
    return uuid_part, scope, parts


def _payload_path(uuid: str) -> Path:
    return PIPELINE_OUTPUTS / uuid / "tier_payload.json"


def load_payload(uuid: str) -> dict[str, Any]:
    path = _payload_path(uuid)
    if not path.exists():
        raise FileNotFoundError(f"tier_payload.json not found for uuid {uuid}")
    return json.loads(path.read_text())


def find_entry(payload: dict[str, Any], locator: str) -> dict[str, Any] | None:
    """Walk the payload structure to find the entry whose locator_id matches."""
    _, scope, _ = parse_locator(locator)

    # tier-room and tier-wall live nested inside payload['rooms'][i]
    if scope == "room":
        for room in payload.get("rooms") or []:
            if room.get("locator_id") == locator:
                return room
        return None

    if scope == "wall":
        for room in payload.get("rooms") or []:
            for wall in room.get("walls") or []:
                if wall.get("locator_id") == locator:
                    return wall
        return None

    key = _LOCATOR_SCOPE_TO_PAYLOAD_KEY.get(scope)
    if key:
        for entry in payload.get(key) or []:
            if entry.get("locator_id") == locator:
                return entry
    return None


def resolve(locator: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Return (payload, entry, scope) for a locator. Raises if not found."""
    uuid, scope, _ = parse_locator(locator)
    payload = load_payload(uuid)
    entry = find_entry(payload, locator)
    if entry is None:
        raise LookupError(f"no entry found for {locator}")
    return payload, entry, scope


# ---- corner math --------------------------------------------------------------


def _corner(c: Any) -> tuple[float, float, float]:
    if isinstance(c, dict):
        return float(c.get("x", 0.0)), float(c.get("y", 0.0)), float(c.get("z", 0.0))
    return float(c[0]), float(c[1]), float(c[2])


def _to_corner_dict(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def _centroid_xz(corners: list[Any]) -> tuple[float, float]:
    if not corners:
        return 0.0, 0.0
    xs = [_corner(c)[0] for c in corners]
    zs = [_corner(c)[2] for c in corners]
    return sum(xs) / len(xs), sum(zs) / len(zs)


def _mean_y(corners: list[Any]) -> float:
    if not corners:
        return 0.0
    ys = [_corner(c)[1] for c in corners]
    return sum(ys) / len(ys)


# ---- preview transforms -------------------------------------------------------
#
# Each "preview_*" returns the full updated entry-shape so the frontend can
# swap geometry directly. They never mutate disk; the caller is responsible
# for keeping originals in client-side state for "Reset preview".


def preview_make_flat(locator: str) -> dict[str, Any]:
    """Project all corners to the mean-y plane; plane becomes y = mean_y."""
    _, entry, _ = resolve(locator)
    corners = entry.get("corners") or []
    if not corners:
        raise ValueError(f"entry has no corners: {locator}")
    y = _mean_y(corners)
    new_corners = [_to_corner_dict(_corner(c)[0], y, _corner(c)[2]) for c in corners]
    return {
        "locator": locator,
        "corners": new_corners,
        "plane": {"a": 0.0, "b": 1.0, "c": 0.0, "d": -y},
    }


def preview_make_slanted(
    locator: str, slope_deg: float = 20.0, azimuth_deg: float = 0.0
) -> dict[str, Any]:
    """Tilt the polygon about its XZ centroid.

    The hinge axis is horizontal, perpendicular to the chosen azimuth.
    Azimuth 0° means "uphill points in +Z". slope_deg=0 leaves the polygon
    flat at its mean y.
    """
    _, entry, _ = resolve(locator)
    corners = entry.get("corners") or []
    if not corners:
        raise ValueError(f"entry has no corners: {locator}")

    cx, cz = _centroid_xz(corners)
    base_y = _mean_y(corners)
    slope = math.radians(slope_deg)
    az = math.radians(azimuth_deg)
    # uphill direction in XZ
    ux, uz = math.sin(az), math.cos(az)
    tan_slope = math.tan(slope)

    new_corners: list[dict[str, float]] = []
    for c in corners:
        x, _, z = _corner(c)
        # signed distance along uphill direction from centroid
        s = (x - cx) * ux + (z - cz) * uz
        new_corners.append(_to_corner_dict(x, base_y + s * tan_slope, z))

    # plane equation: n . p = d, with n pointing "up" relative to the slope
    nx = -ux * math.sin(slope)
    ny = math.cos(slope)
    nz = -uz * math.sin(slope)
    d = -(nx * cx + ny * base_y + nz * cz)
    return {
        "locator": locator,
        "corners": new_corners,
        "plane": {"a": nx, "b": ny, "c": nz, "d": d},
        "slope_deg": slope_deg,
        "azimuth_deg": azimuth_deg,
    }


def preview_delete(locator: str) -> dict[str, Any]:
    """Signal the frontend to hide / remove the element.

    Server returns a marker; the client tracks the original geometry so
    "Reset preview" can restore it.
    """
    _, _entry, scope = resolve(locator)
    return {"locator": locator, "scope": scope, "deleted": True}


def preview_toggle_gap(locator: str) -> dict[str, Any]:
    """Toggle inclusion (visibility) of a gap element."""
    _, _, scope = resolve(locator)
    if not scope.startswith("gap"):
        raise ValueError(f"toggle_gap only valid for tier-gap-* scopes, got {scope}")
    return {"locator": locator, "scope": scope, "deleted": True}


# ---- info getters -------------------------------------------------------------


def _trim_corners(corners: list[Any]) -> list[dict[str, float]]:
    return [_to_corner_dict(*_corner(c)) for c in corners[:32]]


def element_info(locator: str) -> dict[str, Any]:
    """Read-only summary panel for the right-click menu."""
    payload, entry, scope = resolve(locator)
    uuid, _, _ = parse_locator(locator)
    corners = entry.get("corners") or []

    summary: dict[str, Any] = {
        "locator": locator,
        "scope": scope,
        "uuid": uuid,
        "address": payload.get("address"),
        "corner_count": len(corners),
    }
    if corners:
        ys = [_corner(c)[1] for c in corners]
        xs = [_corner(c)[0] for c in corners]
        zs = [_corner(c)[2] for c in corners]
        summary["bounds"] = {
            "x": [min(xs), max(xs)],
            "y": [min(ys), max(ys)],
            "z": [min(zs), max(zs)],
        }
    for key in (
        "source",
        "role",
        "kind",
        "story",
        "scope",
        "support_quality",
        "synthetic",
        "plane",
    ):
        if key in entry:
            summary[key] = entry[key]
    holes = entry.get("holes") or []
    if holes:
        summary["hole_count"] = len(holes)
    adjacency = entry.get("adjacency") or []
    if adjacency:
        summary["adjacency_count"] = len(adjacency)
    return summary


def neighbors(locator: str) -> dict[str, Any]:
    """Return adjacent locators for the selected element."""
    _, entry, scope = resolve(locator)
    raw_adj = entry.get("adjacency") or []
    out: list[str] = []
    for a in raw_adj:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict):
            for k in ("locator_id", "locator", "neighbor", "id"):
                if isinstance(a.get(k), str):
                    out.append(a[k])
                    break
    return {"locator": locator, "scope": scope, "neighbors": out}


# ---- registry -----------------------------------------------------------------


@dataclass(frozen=True)
class QuickAction:
    name: str
    description: str
    fn: Callable[..., dict[str, Any]]


REGISTRY: dict[str, QuickAction] = {
    "preview_make_flat": QuickAction(
        name="preview_make_flat",
        description="Project a polygon's corners to its mean-y plane (visual A/B; not "
        "persisted).",
        fn=preview_make_flat,
    ),
    "preview_make_slanted": QuickAction(
        name="preview_make_slanted",
        description="Tilt a polygon by slope_deg about its XZ centroid; azimuth_deg "
        "sets uphill direction.",
        fn=preview_make_slanted,
    ),
    "preview_delete": QuickAction(
        name="preview_delete",
        description="Hide/remove an element in the viewer (visual A/B; not persisted).",
        fn=preview_delete,
    ),
    "preview_toggle_gap": QuickAction(
        name="preview_toggle_gap",
        description="Toggle visibility of a tier-gap-* element (visual A/B; not "
        "persisted).",
        fn=preview_toggle_gap,
    ),
    "element_info": QuickAction(
        name="element_info",
        description="Return a read-only summary of the element (source, role, bounds, "
        "plane, etc.).",
        fn=element_info,
    ),
    "neighbors": QuickAction(
        name="neighbors",
        description="List adjacent element locators per the tier_payload graph.",
        fn=neighbors,
    ),
}


def dispatch(action: str, **params: Any) -> dict[str, Any]:
    if action not in REGISTRY:
        raise KeyError(f"unknown quick action: {action!r}")
    return REGISTRY[action].fn(**params)


__all__ = [
    "REGISTRY",
    "QuickAction",
    "dispatch",
    "element_info",
    "find_entry",
    "load_payload",
    "neighbors",
    "parse_locator",
    "preview_delete",
    "preview_make_flat",
    "preview_make_slanted",
    "preview_toggle_gap",
    "resolve",
]
