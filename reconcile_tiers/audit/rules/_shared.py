"""Shared helpers, thresholds, and the FlagItem type used by every rule.

`roof_defects.py` and a couple of diagnostic modules import `_y_range`,
`_make_item`, `FlagItem`, `RULE_HYPOTHESES`, and `_building_envelope`
directly. Anything renamed here will break those external consumers.
"""

from __future__ import annotations

import uuid as _uuid
from collections import defaultdict
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ---- thresholds (mirror render-tuning.js + legacy inspect_building) ----------

MIN_POLYGON_AREA_M2 = 1e-4
OPENING_MIN_DIM = 0.05
ENVELOPE_BUFFER_M = 0.5
MIN_OUTSIDE_AREA_M2 = 0.25
MIN_OUTSIDE_RATIO = 0.05
CEILING_COVERAGE_MIN = 0.50
CEILING_NORMAL_Y_MIN = 0.10
KNEE_WALL_TOP_STORY_TOL_M = 1.0
DORMER_HOST_SLOPE_MAX_DIST_M = 0.30
CEILING_BELOW_FLOOR_Y_SLACK_M = 0.05
CEILING_COVERAGE_GAP_Y_SLACK_M = 0.5
WALL_CUTOUT_DRIFT_TOL_M = 1e-3
CEILING_YSPAN_EXCESSIVE_M = 2.0
STORY_OVERSAT_ROOMS_MAX = 2
STORY_OVERSAT_CEILS_MIN = 5
STORY_RATIO_MAX = 3.0


FlagItem = dict[str, Any]


# ---- static root-cause hypotheses --------------------------------------------
#
# Per-rule first-pass framings shown by the viewer's "Why?" button. They are
# guesses, not guarantees — accurate enough to answer "is this worth flagging
# right now?" without paying a debug-element round-trip per click.

RULE_HYPOTHESES: dict[str, dict[str, Any]] = {
    "ceiling_orientation_inverted": {
        "step": "reconcile_tiers/_core/newell.py",
        "files": [
            "reconcile_tiers/_core/newell.py",
            "reconcile_tiers/assemble/ceiling_painter.py",
        ],
        "summary": (
            "Ceiling normal points sideways or down. Likely a polygon traversal "
            "reversed at emission, or a coplanar merge that flipped winding."
        ),
    },
    "ceiling_below_floor": {
        "step": "reconcile_tiers/extract/height_align.py",
        "files": [
            "reconcile_tiers/extract/height_align.py",
            "reconcile_tiers/extract/stories.py",
        ],
        "summary": (
            "Ceiling Y is below the room floor it overlaps in plan. Either a "
            "basement-ceiling mis-matched to an upper-story room, or a Y-frame "
            "inversion at extract time."
        ),
    },
    "out_of_envelope": {
        "step": "reconcile_tiers/roof/clipping.py",
        "files": [
            "reconcile_tiers/roof/clipping.py",
            "reconcile_tiers/_core/footprint.py",
        ],
        "summary": (
            "Geometry extends past the convex footprint + 0.5 m buffer. Usually "
            "roof clipping against the wrong polygon, or footprint missing this "
            "story's contribution."
        ),
    },
    "ceiling_coverage_gap": {
        "step": "reconcile_tiers/assemble/ceiling_painter.py",
        "files": [
            "reconcile_tiers/assemble/ceiling_painter.py",
            "reconcile_tiers/build.py",
        ],
        "summary": (
            "Less than 50% of this room's floor is covered by ceiling pieces "
            "above. Either the painter dropped a piece via the noisy-raw gate, "
            "or no ceiling was emitted for this room at all."
        ),
    },
    "knee_wall_misplaced": {
        "step": "reconcile_tiers/extract/extension.py",
        "files": [
            "reconcile_tiers/extract/extension.py",
            "reconcile_tiers/assemble/walls_to_rooms.py",
        ],
        "summary": (
            "Knee wall isn't within ~1 m of the top story's wall top. Either "
            "misclassified (should be a regular wall), or attached to the wrong "
            "story."
        ),
    },
    "dormer_without_host_slope": {
        "step": "reconcile_tiers/assemble/dormer_reconstruction.py",
        "files": [
            "reconcile_tiers/assemble/dormer_reconstruction.py",
            "reconcile_tiers/roof/clipping.py",
        ],
        "summary": (
            "Dormer face has no oblique ceiling within 0.3 m. Either the source "
            "oblique was clipped away, or this geometry shouldn't have been "
            "classified as a dormer face."
        ),
    },
    "silent_drop": {
        "step": "reconcile_tiers/web/render-tuning.js",
        "files": [
            "reconcile_tiers/web/render-tuning.js",
        ],
        "summary": (
            "Polygon would be silently dropped by the renderer (corners < 3 or "
            "area too small). Usually upstream — the emission shouldn't produce "
            "these."
        ),
    },
    "wall_cutout_outside_outline": {
        "step": "reconcile_tiers/payload/adjacency.py",
        "files": [
            "reconcile_tiers/payload/adjacency.py",
            "reconcile_tiers/assemble/walls_to_rooms.py",
            "reconcile_tiers/build.py",
        ],
        "summary": (
            "Wall cutout corner sits outside the wall's own outer polygon. The "
            "renderer's THREE.ShapeUtils.triangulateShape degenerates and draws "
            "a band over the window. Caused upstream by a wall transform that "
            "didn't update its cutouts."
        ),
    },
    "ceiling_yspan_excessive": {
        "step": "reconcile_tiers/extract/ceilings.py",
        "files": [
            "reconcile_tiers/extract/ceilings.py",
            "reconcile_tiers/assemble/ceiling_painter.py",
        ],
        "summary": (
            "A single ceiling piece spans more than 2 m vertically. A flat or "
            "oblique surface should not have that much Y variation; usually a "
            "ceiling assembled from corners belonging to different physical "
            "surfaces, or a primitive that was clipped against the wrong plane."
        ),
    },
    "story_ceiling_overcount": {
        "step": "reconcile_tiers/extract/stories.py",
        "files": [
            "reconcile_tiers/extract/stories.py",
            "reconcile_tiers/assemble/ceiling_painter.py",
        ],
        "summary": (
            "Story has many more ceiling pieces than rooms. Either the story "
            "splitter under-segmented rooms, or the painter attributed pieces "
            "to the wrong story. Downstream selectors and audits become "
            "uninformative on this story until it is resolved."
        ),
    },
}


# ---- helpers -----------------------------------------------------------------


def _corners_xz(corners: list[dict[str, float]] | None) -> list[tuple[float, float]]:
    return [(c.get("x", 0.0), c.get("z", 0.0)) for c in corners or []]


def _corners_xyz(corners: list[dict[str, float]] | None) -> np.ndarray:
    return np.array(
        [[c.get("x", 0.0), c.get("y", 0.0), c.get("z", 0.0)] for c in corners or []],
        dtype=float,
    )


def _newell_normal(corners: list[dict[str, float]] | None) -> np.ndarray | None:
    pts = _corners_xyz(corners)
    if len(pts) < 3:
        return None
    n = np.zeros(3)
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return None
    return n / norm


def _polygon_area_3d(corners: list[dict[str, float]] | None) -> float:
    pts = _corners_xyz(corners)
    if len(pts) < 3:
        return 0.0
    n = np.zeros(3)
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return float(np.linalg.norm(n) * 0.5)


def _safe_polygon(corners_xz: list[tuple[float, float]]) -> Polygon | None:
    if len(corners_xz) < 3:
        return None
    try:
        poly = Polygon(corners_xz)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-9:
            return None
        return poly
    except Exception:
        return None


def _y_range(corners: list[dict[str, float]] | None) -> tuple[float, float] | None:
    pts = _corners_xyz(corners)
    if not len(pts):
        return None
    return float(pts[:, 1].min()), float(pts[:, 1].max())


def _room_floor_pieces(room: dict[str, Any]) -> list[dict[str, Any]]:
    """Tolerate both legacy `floor: HorizontalLid` and current
    `floor: list[HorizontalLid]`.
    """
    floor = room.get("floor")
    if isinstance(floor, list):
        return [piece for piece in floor if isinstance(piece, dict)]
    if isinstance(floor, dict):
        return [floor]
    return []


def _parse_locator(locator_id: str | None) -> tuple[str, list[str]]:
    """Split a locator into (kind, parts). Empty kind if locator missing/malformed."""
    if not locator_id:
        return "", []
    segments = locator_id.split("::", 2)
    if len(segments) != 3:
        return "", []
    kind = segments[1]
    parts = segments[2].split(":") if segments[2] else []
    return kind, parts


def _make_item(
    locator_id: str | None,
    rule: str,
    severity: str,
    evidence: dict[str, Any],
) -> FlagItem:
    kind, parts = _parse_locator(locator_id)
    return {
        "id": _uuid.uuid4().hex[:12],
        "locator": locator_id,
        "kind": kind,
        "parts": parts,
        "rule": rule,
        "note": None,
        "severity": severity,
        "evidence": evidence,
        "hypothesis": RULE_HYPOTHESES.get(rule),
    }


def _building_envelope(rooms: list[dict[str, Any]]) -> Polygon | None:
    """Buffer-union of every room floor across all stories.

    Mirrors `roof/footprint.py:build_building_footprint` (buffer 0.3, union,
    keep largest component). Previously this used a ground-story convex hull
    which disagreed with the pipeline on multi-story buildings — the diagnostic
    pass (`audit/story_envelope_diagnostic.py`) showed 100% of `out_of_envelope`
    flags were that audit-pipeline mismatch, not real overshoot.
    """
    polys = []
    for room in rooms:
        for piece in _room_floor_pieces(room):
            corners = piece.get("corners") or []
            poly = _safe_polygon(_corners_xz(corners))
            if poly:
                polys.append(poly.buffer(ENVELOPE_BUFFER_M, join_style="mitre"))
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.is_empty:
        return None
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    return merged


def _story_y_bands(rooms: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    """Per story, the (y_min, y_max) of all walls in that story."""
    bands: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for room in rooms:
        story = room.get("story")
        if story is None:
            continue
        for wall in room.get("walls") or []:
            yr = _y_range(wall.get("corners") or [])
            if yr is not None:
                bands[story].append(yr)
    out = {}
    for story, ranges in bands.items():
        if not ranges:
            continue
        out[story] = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    return out
