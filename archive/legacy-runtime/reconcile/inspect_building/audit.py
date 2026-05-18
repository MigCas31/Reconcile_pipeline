"""Audit a building's tier_payload.json for defects the user would see in viewer-tiers.

The tiers viewer (reconcile_tiers/web/tier-preview.js) renders exactly these
groups from tier_payload.json:
  - rooms[].floor + rooms[].walls + rooms[].doors + rooms[].windows
  - gaps[]
  - ceiling[]      (rendered as 'roof' material)
  - knee_walls[]   (rendered as 'dormer' material)

Several silent-drop conditions exist in the renderer:
  - polygonGeometry returns null if corners.length < 3
  - polygonGeometry returns null if 3D area < RENDER_TUNING.minPolygonAreaM2
  - addOpeningBox skips if width or height < RENDER_TUNING.opening.minDim

This audit walks the same data the renderer sees, applies these visibility
filters, and reports defects that map to "what the user looks at and notices
is wrong":

  1. ceiling_coverage_per_room  — room with no ceiling overhead
  2. ceiling_orientation        — ceiling normal not roughly upward
  3. out_of_envelope            — geometry extending past the building footprint
  4. silent_drops               — polygons / openings the renderer drops
  5. story_coverage_census      — rooms vs ceilings per story
  6. knee_wall_height           — knee walls not at the top story
  7. gap_census                 — group gaps by kind x scope

Every issue carries the original locator_id so it can be pasted into the
viewer's search box.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

MIN_POLYGON_AREA_M2 = 1e-4  # mirrors RENDER_TUNING.minPolygonAreaM2
OPENING_MIN_DIM = 0.05  # mirrors RENDER_TUNING.opening.minDim
ENVELOPE_BUFFER_M = 0.5  # how far outside the convex footprint counts as "outside"
MIN_OUTSIDE_AREA_M2 = 0.25  # below this, "out of envelope" is noise
MIN_OUTSIDE_RATIO = 0.05  # below this, "out of envelope" is noise
CEILING_COVERAGE_MIN = 0.50  # room is "uncovered" if union of ceilings covers <50%
CEILING_NORMAL_Y_MIN = 0.10  # ceiling normal Y component below this = wrong orientation
KNEE_WALL_TOP_STORY_TOL_M = 1.0  # knee wall should sit within this of the top story Y


# ----- geometry helpers --------------------------------------------------------


def _corners_xz(corners):
    return [(c.get("x", 0.0), c.get("z", 0.0)) for c in corners or []]


def _corners_xyz(corners):
    return np.array(
        [[c.get("x", 0.0), c.get("y", 0.0), c.get("z", 0.0)] for c in corners or []],
        dtype=float,
    )


def _newell_normal(corners):
    """Newell's method for a 3D polygon normal. Returns unit vector or None."""
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
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    return n / norm


def _polygon_area_3d(corners):
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


def _safe_polygon(corners_xz):
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


def _y_range(corners):
    pts = _corners_xyz(corners)
    if not len(pts):
        return None
    return float(pts[:, 1].min()), float(pts[:, 1].max())


def _mean_y(corners):
    pts = _corners_xyz(corners)
    if not len(pts):
        return None
    return float(pts[:, 1].mean())


# ----- audit checks ------------------------------------------------------------


def _floor_pieces(room):
    """`room["floor"]` is a list[FloorPiece] in the current schema; older
    payloads stored a single dict. Normalise to a list of dicts."""
    floor = room.get("floor") or []
    if isinstance(floor, list):
        return [p for p in floor if isinstance(p, dict)]
    if isinstance(floor, dict):
        return [floor]
    return []


def _building_envelope(rooms):
    """Convex hull of ground-story room floors, buffered."""
    stories = [r.get("story") for r in rooms if r.get("story") is not None]
    if not stories:
        return None
    ground = min(stories)
    polys = []
    for room in rooms:
        if room.get("story") != ground:
            continue
        for piece in _floor_pieces(room):
            corners = piece.get("corners") or []
            poly = _safe_polygon(_corners_xz(corners))
            if poly:
                polys.append(poly)
    if not polys:
        return None
    return unary_union(polys).convex_hull.buffer(ENVELOPE_BUFFER_M)


def _check_out_of_envelope(envelope, payload):
    """For every visible polygon, what fraction sits outside the envelope."""
    if envelope is None:
        return []
    issues: list[dict] = []
    sources: list[tuple[str, list]] = []
    for room in payload.get("rooms") or []:
        for wall in room.get("walls") or []:
            sources.append(("room.wall", [wall]))
    sources.append(("ceiling", payload.get("ceiling") or []))
    sources.append(("gap", payload.get("gaps") or []))
    sources.append(("knee_wall", payload.get("knee_walls") or []))
    for kind, items in sources:
        for item in items:
            corners = item.get("corners")
            poly = _safe_polygon(_corners_xz(corners))
            if poly is None:
                continue
            outside = poly.difference(envelope)
            if outside.is_empty:
                continue
            ratio = outside.area / max(poly.area, 1e-9)
            if outside.area < MIN_OUTSIDE_AREA_M2 or ratio < MIN_OUTSIDE_RATIO:
                continue
            cx, cy = outside.representative_point().coords[0]
            issues.append(
                {
                    "kind": kind,
                    "locator_id": item.get("locator_id"),
                    "polygon_area_xz_m2": float(poly.area),
                    "outside_area_xz_m2": float(outside.area),
                    "outside_ratio": float(ratio),
                    "outside_at_xz": [float(cx), float(cy)],
                }
            )
    issues.sort(key=lambda e: -e["outside_area_xz_m2"])
    return issues


def _check_ceiling_coverage(payload):
    """Per room, what fraction of the floor footprint is covered by ceilings above."""
    rooms = payload.get("rooms") or []
    ceilings = payload.get("ceiling") or []
    issues: list[dict] = []
    for room in rooms:
        floor = []
        for piece in _floor_pieces(room):
            floor.extend(piece.get("corners") or [])
        if not floor:
            continue
        floor_poly = _safe_polygon(_corners_xz(floor))
        if floor_poly is None or floor_poly.area < 1.0:
            continue  # room with implausibly small floor — skip
        floor_y = _mean_y(floor)
        # Determine room Y top from walls.
        wall_y_max = None
        for wall in room.get("walls") or []:
            yr = _y_range(wall.get("corners") or [])
            if yr is None:
                continue
            wall_y_max = yr[1] if wall_y_max is None else max(wall_y_max, yr[1])
        # Ceilings above this room: sloped roofs can start at low eaves, so use
        # the high side of the patch rather than rejecting by its lowest point.
        slack_m = 0.5
        candidate_polys = []
        for piece in ceilings:
            corners = piece.get("corners") or []
            yr = _y_range(corners)
            if yr is None:
                continue
            if wall_y_max is not None and yr[1] < wall_y_max - slack_m:
                # Ceiling sits well below the room top. That is wrong for
                # flat slabs, but valid sloped roofs often have a low eave and
                # a high ridge; count oblique/raw roof evidence when it still
                # sits above the room floor.
                source = piece.get("source")
                if not (
                    source in {"computed_oblique", "raw_scan"}
                    and floor_y is not None
                    and yr[1] >= floor_y + 0.25
                ):
                    continue
            poly = _safe_polygon(_corners_xz(corners))
            if poly:
                candidate_polys.append(poly)
        if not candidate_polys:
            covered_area = 0.0
        else:
            union = unary_union(candidate_polys)
            covered_area = floor_poly.intersection(union).area
        coverage = covered_area / floor_poly.area
        if coverage < CEILING_COVERAGE_MIN:
            issues.append(
                {
                    "room_locator_id": room.get("locator_id"),
                    "story": room.get("story"),
                    "floor_area_m2": float(floor_poly.area),
                    "covered_area_m2": float(covered_area),
                    "coverage_ratio": float(coverage),
                }
            )
    issues.sort(key=lambda e: e["coverage_ratio"])
    return issues


def _check_ceiling_orientation(payload):
    """Ceilings whose normal does not point upward."""
    issues: list[dict] = []
    for piece in payload.get("ceiling") or []:
        corners = piece.get("corners") or []
        n = _newell_normal(corners)
        if n is None:
            continue
        if n[1] < CEILING_NORMAL_Y_MIN:
            issues.append(
                {
                    "locator_id": piece.get("locator_id"),
                    "normal": [float(x) for x in n],
                    "normal_y": float(n[1]),
                    "source": piece.get("source"),
                }
            )
    issues.sort(key=lambda e: e["normal_y"])
    return issues


def _check_silent_drops(payload):
    """Polygons / openings the viewer would silently drop."""
    drops: list[dict] = []
    sources: list[tuple[str, list]] = [
        ("ceiling", payload.get("ceiling") or []),
        ("knee_wall", payload.get("knee_walls") or []),
        ("gap", payload.get("gaps") or []),
    ]
    for room in payload.get("rooms") or []:
        for wall in room.get("walls") or []:
            sources.append(("room.wall", [wall]))
        for piece in _floor_pieces(room):
            sources.append(
                (
                    "room.floor",
                    [
                        {
                            "locator_id": piece.get("locator_id")
                            or room.get("locator_id"),
                            "corners": piece.get("corners") or [],
                        }
                    ],
                )
            )
    for kind, items in sources:
        for item in items:
            corners = item.get("corners") or []
            if len(corners) < 3:
                drops.append(
                    {
                        "kind": kind,
                        "locator_id": item.get("locator_id"),
                        "reason": f"corners.length={len(corners)} (<3)",
                    }
                )
                continue
            area = _polygon_area_3d(corners)
            if area < MIN_POLYGON_AREA_M2:
                drops.append(
                    {
                        "kind": kind,
                        "locator_id": item.get("locator_id"),
                        "reason": f"area={area:.2e} m² (<{MIN_POLYGON_AREA_M2})",
                    }
                )
    # Openings (doors/windows) below opening.minDim
    for room in payload.get("rooms") or []:
        for opening_kind in ("doors", "windows"):
            for index, opening in enumerate(room.get(opening_kind) or []):
                corners = opening.get("corners") or []
                if len(corners) < 3:
                    drops.append(
                        {
                            "kind": f"room.{opening_kind}",
                            "locator_id": (
                                f"{room.get('locator_id')}:{opening_kind[:-1]}:{index}"
                            ),
                            "reason": f"corners.length={len(corners)} (<3)",
                        }
                    )
                    continue
                # Approximate opening width/height: use 3D bbox extents on the
                # two longest principal axes.
                pts = _corners_xyz(corners)
                ext = pts.max(axis=0) - pts.min(axis=0)
                # Sort, drop the smallest (which is the wall-thickness direction).
                sorted_ext = sorted(ext.tolist(), reverse=True)
                w, h = sorted_ext[0], sorted_ext[1]
                if w < OPENING_MIN_DIM or h < OPENING_MIN_DIM:
                    drops.append(
                        {
                            "kind": f"room.{opening_kind}",
                            "locator_id": (
                                f"{room.get('locator_id')}:{opening_kind[:-1]}:{index}"
                            ),
                            "reason": f"width={w:.3f} m, height={h:.3f} m "
                            f"(min={OPENING_MIN_DIM})",
                        }
                    )
    return drops


def _check_story_coverage(payload):
    """Per story: room count vs ceiling-piece count assignable to that story."""
    rooms = payload.get("rooms") or []
    ceilings = payload.get("ceiling") or []
    if not rooms:
        return {}
    # Story Y bands from rooms (use wall y-range per story).
    stories: dict[int, dict] = defaultdict(
        lambda: {
            "room_count": 0,
            "y_min": float("inf"),
            "y_max": float("-inf"),
            "ceiling_count": 0,
        }
    )
    for room in rooms:
        story = room.get("story")
        if story is None:
            continue
        stories[story]["room_count"] += 1
        for wall in room.get("walls") or []:
            yr = _y_range(wall.get("corners") or [])
            if yr is None:
                continue
            stories[story]["y_min"] = min(stories[story]["y_min"], yr[0])
            stories[story]["y_max"] = max(stories[story]["y_max"], yr[1])
    for piece in ceilings:
        corners = piece.get("corners") or []
        yr = _y_range(corners)
        if yr is None:
            continue
        # Assign to whichever story's Y band contains the ceiling mean Y, with slack.
        mid = (yr[0] + yr[1]) * 0.5
        best_story = None
        best_dist = float("inf")
        for story, info in stories.items():
            if info["y_max"] == float("-inf"):
                continue
            ref = info["y_max"]
            d = abs(mid - ref)
            if d < best_dist:
                best_dist = d
                best_story = story
        if best_story is not None:
            stories[best_story]["ceiling_count"] += 1
    out = {}
    for story, info in sorted(stories.items()):
        out[str(story)] = {
            "room_count": info["room_count"],
            "ceiling_count": info["ceiling_count"],
            "y_min": None if info["y_min"] == float("inf") else info["y_min"],
            "y_max": None if info["y_max"] == float("-inf") else info["y_max"],
        }
    return out


def _check_knee_walls(payload):
    """Knee walls should sit within ~1 m of the top story's Y_max."""
    knee_walls = payload.get("knee_walls") or []
    if not knee_walls:
        return []
    story_info = _check_story_coverage(payload)
    top_y = None
    for info in story_info.values():
        if info["y_max"] is not None:
            top_y = info["y_max"] if top_y is None else max(top_y, info["y_max"])
    if top_y is None:
        return []
    issues: list[dict] = []
    for wall in knee_walls:
        yr = _y_range(wall.get("corners") or [])
        if yr is None:
            continue
        # Wall midpoint Y vs top_y.
        mid_y = (yr[0] + yr[1]) * 0.5
        delta = abs(mid_y - top_y)
        if delta > KNEE_WALL_TOP_STORY_TOL_M:
            issues.append(
                {
                    "locator_id": wall.get("locator_id"),
                    "y_range": [float(yr[0]), float(yr[1])],
                    "top_story_y_max": float(top_y),
                    "delta_m": float(delta),
                }
            )
    issues.sort(key=lambda e: -e["delta_m"])
    return issues


def _gap_census(payload):
    counter = Counter()
    for gap in payload.get("gaps") or []:
        kind = gap.get("kind") or "?"
        scope = gap.get("scope") or "?"
        counter[(kind, scope)] += 1
    return [
        {"kind": k, "scope": s, "count": c} for (k, s), c in sorted(counter.items())
    ]


# ----- entry point -------------------------------------------------------------


def audit(payload: dict) -> dict:
    rooms = payload.get("rooms") or []
    envelope = _building_envelope(rooms)
    out_of_envelope = _check_out_of_envelope(envelope, payload)
    ceiling_coverage = _check_ceiling_coverage(payload)
    ceiling_orientation = _check_ceiling_orientation(payload)
    silent_drops = _check_silent_drops(payload)
    story_coverage = _check_story_coverage(payload)
    knee_walls = _check_knee_walls(payload)
    gap_census = _gap_census(payload)

    flags: list[str] = []
    if out_of_envelope:
        flags.append("out_of_envelope")
    if ceiling_coverage:
        flags.append("ceiling_coverage_gaps")
    if ceiling_orientation:
        flags.append("ceiling_orientation_wrong")
    if silent_drops:
        flags.append("silent_drops")
    if knee_walls:
        flags.append("knee_walls_misplaced")

    return {
        "flags": flags,
        "envelope_present": envelope is not None,
        "envelope_area_m2": float(envelope.area) if envelope is not None else None,
        "out_of_envelope": out_of_envelope,
        "ceiling_coverage_gaps": ceiling_coverage,
        "ceiling_orientation_wrong": ceiling_orientation,
        "silent_drops": silent_drops,
        "story_coverage": story_coverage,
        "knee_walls_misplaced": knee_walls,
        "gap_census": gap_census,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier-payload", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not args.tier_payload.exists():
        args.out.write_text(json.dumps({"error": f"Missing: {args.tier_payload}"}))
        return 1

    payload = json.loads(args.tier_payload.read_text())
    result = audit(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
