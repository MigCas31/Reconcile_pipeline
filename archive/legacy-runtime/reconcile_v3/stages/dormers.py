"""dormers - V3 adapter around
``reconcile.roof_algorithms_py.dormer_detection.detect_dormers``.

Re-derives oblique clusters from V3 rooms (cheap; O(#walls)), synthesises
the ``oblique_roof_surfaces`` dicts that ``detect_dormers`` expects, and
forwards to the green primitive. All graph-coupled kwargs pass as ``None``
so the function falls back to its geometry-only heuristics.
"""

from __future__ import annotations

from math import cos, sin

from reconcile.roof_algorithms_py.dormer_detection import detect_dormers
from reconcile.roof_algorithms_py.oblique_clustering import cluster_oblique_segments

from ..audit import HypothesisTrace
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import V3Dormer, V3SlantedRoof
from .slanted_roofs import _collect_room_oblique_segments


def _bldg_adapter(rooms: list[V3Room]) -> dict:
    return {
        "rooms": [
            {
                "story": r.story,
                "floor_polygon": [list(c) for c in r.floor_polygon],
                "walls_computed": [
                    {
                        "id": w.get("id"),
                        "corners": [list(c) for c in w.get("corners") or []],
                    }
                    for w in r.walls
                ],
                "walls_merged": [
                    {
                        "id": w.get("id"),
                        "corners": [list(c) for c in w.get("corners") or []],
                    }
                    for w in r.walls
                ],
            }
            for r in rooms
        ]
    }


def _cluster_surfaces(
    rooms: list[V3Room], slanted_roofs: list[V3SlantedRoof]
) -> list[dict]:
    segments: list[dict] = []
    for r in rooms:
        segments.extend(_collect_room_oblique_segments(r))
    clusters = cluster_oblique_segments(segments) if segments else []

    surfaces: list[dict] = []
    for idx, cl in enumerate(clusters):
        xs = [e for s in cl["segs"] for e in (s["a"][0], s["b"][0])]
        ys = [e for s in cl["segs"] for e in (s["a"][1], s["b"][1])]
        zs = [e for s in cl["segs"] for e in (s["a"][2], s["b"][2])]
        cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)

        az_rad = float(cl["avgAzimuth"]) * 3.141592653589793 / 180.0
        ridge_dx, ridge_dz = -cos(az_rad), sin(az_rad)
        projections = [
            x * ridge_dx + z * ridge_dz for x, z in zip(xs, zs, strict=False)
        ]
        ridge_min = min(projections) if projections else 0.0
        ridge_max = max(projections) if projections else 0.0

        dominant_story = 0
        story_counts: dict[int, int] = {}
        for s in cl["segs"]:
            story_counts[int(s.get("story", 0))] = (
                story_counts.get(int(s.get("story", 0)), 0) + 1
            )
        if story_counts:
            dominant_story = max(story_counts.items(), key=lambda kv: kv[1])[0]

        surfaces.append(
            {
                "cluster": cl,
                "dominant_story": dominant_story,
                "corners": [(cx, cy, cz)],
                "center": {"x": cx, "y": cy, "z": cz},
                "ridge": {
                    "x": ridge_dx,
                    "z": ridge_dz,
                    "min": ridge_min,
                    "max": ridge_max,
                },
                "roof_hypothesis_id": None,
                "_v3_roof_index": idx if idx < len(slanted_roofs) else None,
            }
        )
    return surfaces


def detect_dormers_v3(
    building_uuid: str,
    rooms: list[V3Room],
    slanted_roofs: list[V3SlantedRoof],
) -> list[V3Dormer]:
    if not slanted_roofs:
        return []

    bldg = _bldg_adapter(rooms)
    surfaces = _cluster_surfaces(rooms, slanted_roofs)
    if not surfaces:
        return []

    raw = detect_dormers(bldg, surfaces, story_floor_y={})

    dormers: list[V3Dormer] = []
    for idx, dormer in enumerate(raw):
        front_wall = dormer.get("front_wall") or {}
        surf_idx = dormer.get("roof_surface_index")
        roof_idx = (
            surfaces[surf_idx].get("_v3_roof_index")
            if isinstance(surf_idx, int) and 0 <= surf_idx < len(surfaces)
            else None
        )
        roof_surface_id = (
            slanted_roofs[roof_idx].id
            if isinstance(roof_idx, int) and 0 <= roof_idx < len(slanted_roofs)
            else ""
        )
        corners = [tuple(c) for c in (front_wall.get("corners") or [])]
        dormers.append(
            V3Dormer(
                id=make_element_id(building_uuid, "v3-dormer", f"dormer-{idx}"),
                front_wall_id=str(front_wall.get("id") or f"wall-{idx}"),
                roof_surface_id=roof_surface_id,
                corners=corners,
                trace=HypothesisTrace(
                    stage="dormers",
                    rule="dormer-protrusion",
                    inputs={
                        "roof_surface_index": surf_idx,
                        "room_index": front_wall.get("room_index"),
                    },
                    decision_reason=(
                        "Wall protrudes through an oblique roof surface per "
                        "detect_dormers' geometric criteria."
                    ),
                ),
            )
        )
    return dormers
