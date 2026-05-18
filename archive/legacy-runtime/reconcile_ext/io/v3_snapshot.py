"""Build an ``ExtSnapshot`` from a V3 result JSON + raw buildings JSON.

For the first-cut pipeline we consume the *full* V3 result (so slanted roofs
are available), not the "branch point" partial state. Moving the branch
point into the live V3 pipeline is a follow-up once we trust the signals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..models import (
    ExtSnapshot,
    SnapshotPart,
    SnapshotRoof,
    SnapshotRoom,
    SnapshotWall,
)

Vec3 = tuple[float, float, float]


def _to_vec3_list(corners: list[Any]) -> list[Vec3]:
    out: list[Vec3] = []
    for c in corners or []:
        if len(c) >= 3:
            out.append((float(c[0]), float(c[1]), float(c[2])))
    return out


def _wall_features(
    corners: list[Vec3],
) -> tuple[float, float, float, float, float] | None:
    """Return (azimuth_deg, plane_offset_m, top_y, bottom_y, length_m).

    Azimuth is the compass direction of the wall's outward normal. We
    approximate by taking the first two non-coincident XZ-distinct corners
    as one edge of the wall and treating its perpendicular as the normal
    direction. For a rectangular wall this is exact; for warped corners it
    is a reasonable approximation.
    """
    if len(corners) < 3:
        return None
    pts_xz = [(c[0], c[2]) for c in corners]
    # Find the longest XZ-edge as the "along" direction.
    best_len = 0.0
    best_ab: tuple[tuple[float, float], tuple[float, float]] | None = None
    for i in range(len(pts_xz)):
        a = pts_xz[i]
        b = pts_xz[(i + 1) % len(pts_xz)]
        dx, dz = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dz)
        if length > best_len:
            best_len = length
            best_ab = (a, b)
    if best_ab is None or best_len < 1e-6:
        return None
    a, b = best_ab
    dx, dz = b[0] - a[0], b[1] - a[1]
    # Outward normal convention: rotate the along-edge +90° in XZ.
    # (Which of the two normals is "outward" doesn't matter for our step
    # check — we compare signed offsets within the same azimuth bucket.)
    nx, nz = -dz / best_len, dx / best_len
    azimuth_rad = math.atan2(nx, nz)  # 0 = +Z (North-ish)
    azimuth_deg = math.degrees(azimuth_rad) % 360.0
    plane_offset = nx * a[0] + nz * a[1]
    top_y = max(c[1] for c in corners)
    bottom_y = min(c[1] for c in corners)
    return azimuth_deg, plane_offset, top_y, bottom_y, best_len


def _slanted_roof_features(
    plane: tuple[float, float, float, float], corners: list[Vec3]
) -> tuple[float, float, float, float]:
    """Return (azimuth_deg, inclination_deg, ridge_y, eave_y)."""
    a, b, c, _d = plane
    # Normal = (a, b, c). Inclination = angle between normal and +Y.
    norm = math.sqrt(a * a + b * b + c * c) or 1.0
    inclination_rad = math.acos(max(-1.0, min(1.0, abs(b) / norm)))
    inclination_deg = math.degrees(inclination_rad)
    # Azimuth of the *downhill* direction: project normal onto XZ, flip sign.
    # Downhill azimuth = direction water runs off.
    if math.hypot(a, c) < 1e-6:
        azimuth_deg = 0.0
    else:
        # +Z = north, +X = east; azimuth clockwise from north.
        azimuth_deg = math.degrees(math.atan2(a, c)) % 360.0
    if not corners:
        return azimuth_deg, inclination_deg, 0.0, 0.0
    ys = [c[1] for c in corners]
    return azimuth_deg, inclination_deg, max(ys), min(ys)


def build_snapshot(
    v3_results: dict[str, Any],
    buildings_json_path: Path,
) -> ExtSnapshot:
    """Merge a per-building V3 result dict with the raw building record."""
    uuid = v3_results["building_uuid"]

    with open(buildings_json_path) as fh:
        buildings = json.load(fh)
    raw = next((b for b in buildings if b.get("uuid") == uuid), None)
    if raw is None:
        raise KeyError(f"Building {uuid} not found in {buildings_json_path}")

    # Parts
    parts: list[SnapshotPart] = []
    for p in v3_results.get("parts", []) or []:
        parts.append(
            SnapshotPart(
                id=p["id"],
                footprint_xz=_to_vec3_list(p.get("footprint_xz") or []),
                room_ids=tuple(p.get("room_ids") or []),
                stories=tuple(p.get("stories") or []),
            )
        )

    # Rooms — pull floor/wall geometry from raw; pull ceiling Y from walls' tops.
    rooms: list[SnapshotRoom] = []
    for idx, r in enumerate(raw.get("rooms") or []):
        room_id = r.get("identifier") or f"room:{idx}"
        story = int(r.get("story", 0))
        floor_polygon = _to_vec3_list(r.get("floor_polygon") or [])
        floor_y = floor_polygon[0][1] if floor_polygon else 0.0
        walls_src = r.get("walls_computed") or r.get("walls_merged") or []
        walls: list[SnapshotWall] = []
        ceiling_y_candidates: list[float] = []
        for wi, w in enumerate(walls_src):
            wall_corners = _to_vec3_list(w.get("corners") or [])
            feats = _wall_features(wall_corners)
            if feats is None:
                continue
            azimuth_deg, plane_offset, top_y, bottom_y, length_m = feats
            walls.append(
                SnapshotWall(
                    id=w.get("id") or f"{room_id}::wall:{wi}",
                    corners=wall_corners,
                    azimuth_deg=azimuth_deg,
                    plane_offset_m=plane_offset,
                    top_y=top_y,
                    bottom_y=bottom_y,
                    length_m=length_m,
                )
            )
            ceiling_y_candidates.append(top_y)
        ceiling_y = None
        if ceiling_y_candidates:
            ceiling_y_candidates.sort()
            ceiling_y = ceiling_y_candidates[len(ceiling_y_candidates) // 2]
        rooms.append(
            SnapshotRoom(
                id=room_id,
                story=story,
                floor_polygon=floor_polygon,
                floor_y=floor_y,
                ceiling_y=ceiling_y,
                walls=walls,
            )
        )

    # V3's closure decisions — geometry V3 already committed to as "part of
    # one continuous building". Unioned with part footprints before reflex
    # detection so V3's calls don't get re-litigated as seams:
    #   1. ``slabs`` with ``room_id=None`` — floor slabs for gaps V3 closed
    #      (``close_obvious_gaps`` in ``reconcile_v3/stages/gaps.py``).
    #   2. ``flat_ceilings`` with ``over="fill"`` — ``fill_remaining`` caps
    #      uncovered top-story footprint pieces with a flat ceiling,
    #      committing them as part of the envelope.
    # ``flat_ceilings`` with ``over="gap"`` is *not* added: it's the ceiling
    # counterpart of the same closed gap already covered by the floor slab;
    # their XZ projections drift by ~wall-thickness so unioning both injects
    # staircase edges and *increases* false reflex count.
    # Per-room slabs and ``flat_ceilings`` with ``over="room"`` are not
    # added — already represented inside ``V3Part.footprint_xz``.
    gap_fill_footprints: list[list[Vec3]] = []
    for s in v3_results.get("slabs", []) or []:
        if s.get("room_id") is not None:
            continue
        poly = _to_vec3_list(s.get("polygon") or [])
        if len(poly) >= 3:
            gap_fill_footprints.append(poly)
    for fc in v3_results.get("flat_ceilings", []) or []:
        if fc.get("over") != "fill":
            continue
        poly = _to_vec3_list(fc.get("footprint_xz") or [])
        if len(poly) >= 3:
            gap_fill_footprints.append(poly)

    # Slanted roofs
    slanted_roofs: list[SnapshotRoof] = []
    for sr in v3_results.get("slanted_roofs", []) or []:
        plane = sr.get("plane")
        corners = _to_vec3_list(sr.get("corners") or [])
        if not plane or len(plane) != 4 or not corners:
            continue
        plane_t = (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))
        azimuth_deg, inclination_deg, ridge_y, eave_y = _slanted_roof_features(
            plane_t, corners
        )
        slanted_roofs.append(
            SnapshotRoof(
                id=sr["id"],
                corners=corners,
                plane=plane_t,
                azimuth_deg=azimuth_deg,
                inclination_deg=inclination_deg,
                ridge_y=ridge_y,
                eave_y=eave_y,
            )
        )

    return ExtSnapshot(
        building_uuid=uuid,
        address=v3_results.get("address") or raw.get("address"),
        parts=parts,
        rooms=rooms,
        slanted_roofs=slanted_roofs,
        gap_fill_footprints=gap_fill_footprints,
    )


def load_v3_building(v3_results_path: Path, uuid: str) -> dict[str, Any]:
    """Pull one building out of the V3 aggregate JSON."""
    with open(v3_results_path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        for entry in data:
            if entry.get("building_uuid") == uuid:
                return entry
        raise KeyError(f"Building {uuid} not found in {v3_results_path}")
    if isinstance(data, dict):
        if data.get("building_uuid") == uuid:
            return data
        raise KeyError(f"Building {uuid} not found in {v3_results_path}")
    raise ValueError(f"Unexpected V3 results shape in {v3_results_path}")


def iter_v3_buildings(v3_results_path: Path):
    with open(v3_results_path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield data
