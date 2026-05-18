"""Load the subset of buildings_3d.json that corresponds to viewer
pipeline steps 1 ("Apple merged") and 2 ("Room segments") — floor
polygons, walls_computed, story assignment. Everything downstream is
redesigned by reconcile_v3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Polygon

Vec3 = tuple[float, float, float]


_FLOOR_SIMPLIFY_TOL_M = 0.02


def _sanitize_floor_polygon(corners: list[Vec3]) -> list[Vec3]:
    """Repair self-intersecting floor polygons.

    Raw floor polygons often contain sub-scan-noise spurs and tails that
    produce self-intersecting topology. buffer(0) heals bow-ties; a small
    simplify removes the hair-thin spurs that earcut renders as weird
    triangles. Y is uniform on a floor ring, so we keep the incoming Y.
    """
    if len(corners) < 3:
        return list(corners)
    try:
        poly = Polygon([(c[0], c[2]) for c in corners])
    except Exception:
        return list(corners)
    if poly.is_empty or poly.area <= 0:
        return list(corners)
    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
        except Exception:
            return list(corners)
        if poly.is_empty or not poly.is_valid:
            return list(corners)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    simplified = poly.simplify(_FLOOR_SIMPLIFY_TOL_M, preserve_topology=True)
    if simplified.is_empty or not simplified.is_valid or simplified.area <= 0:
        simplified = poly
    if isinstance(simplified, MultiPolygon):
        simplified = max(simplified.geoms, key=lambda g: g.area)
    y = float(corners[0][1])
    coords = list(simplified.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        return list(corners)
    return [(float(x), y, float(z)) for x, z in coords]


@dataclass
class V3Room:
    index: int
    identifier: str
    story: int
    floor_polygon: list[Vec3]
    walls: list[dict[str, Any]] = field(default_factory=list)  # each: {id, corners}
    doors: list[dict[str, Any]] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class V3Input:
    uuid: str
    address: str | None
    classification: str | None
    rooms: list[V3Room]


def _to_vec3(c: list[float]) -> Vec3:
    return (float(c[0]), float(c[1]), float(c[2]))


def _room_walls(room: dict) -> list[dict[str, Any]]:
    walls = room.get("walls_computed") or []
    if not walls:
        walls = room.get("walls_merged") or []
    return [
        {
            "id": w.get("id") or f"wall:{i}",
            "corners": [_to_vec3(c) for c in w.get("corners", [])],
            "source": w.get("source"),
        }
        for i, w in enumerate(walls)
    ]


def _room_from_dict(index: int, room: dict) -> V3Room:
    return V3Room(
        index=index,
        identifier=room.get("identifier") or f"room:{index}",
        story=int(room.get("story", 0)),
        floor_polygon=_sanitize_floor_polygon(
            [_to_vec3(c) for c in (room.get("floor_polygon") or [])]
        ),
        walls=_room_walls(room),
        doors=[
            {
                "id": d.get("id") or f"door:{i}",
                "corners": [_to_vec3(c) for c in d.get("corners", [])],
            }
            for i, d in enumerate(room.get("doors") or [])
        ],
        windows=[
            {
                "id": w.get("id") or f"window:{i}",
                "corners": [_to_vec3(c) for c in w.get("corners", [])],
            }
            for i, w in enumerate(room.get("windows") or [])
        ],
    )


def load_building(buildings_json_path: Path, uuid: str) -> V3Input:
    with open(buildings_json_path) as handle:
        buildings = json.load(handle)
    for b in buildings:
        if b.get("uuid") == uuid:
            return V3Input(
                uuid=b["uuid"],
                address=b.get("address"),
                classification=b.get("classification"),
                rooms=[
                    _room_from_dict(i, r) for i, r in enumerate(b.get("rooms") or [])
                ],
            )
    raise KeyError(f"Building {uuid} not found in {buildings_json_path}")


def iter_building_uuids(buildings_json_path: Path) -> list[str]:
    with open(buildings_json_path) as handle:
        buildings = json.load(handle)
    return [b["uuid"] for b in buildings if b.get("uuid")]
