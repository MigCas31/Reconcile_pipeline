#!/usr/bin/env python3
"""Analyze selected ridge/eave plane-groups that pair with multiple planes.

This script answers a specific architectural question:

* When a selected plane-group intersects more than one other selected plane-group,
  what kind of building situation is that?

The intent is to separate physically different scenarios before changing scorer
logic. In practice, this highlights whether we are seeing:

* one roof side over-segmented into multiple local runs
* a hub/fan on one roof mass
* multi-wing / multi-story articulated buildings
* rare single-room / single-story edge cases
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"
RIDGE_EAVE_SCORES_PATH = REPO / "reports" / "ridge_eave_scores_20260420" / "scores.json"
DEFAULT_OUT = REPO / ".context" / "multi_partner_plane_groups_analysis.json"


def _wrapped_angle_delta_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs(float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


def _room_floor_union(building: dict[str, Any]) -> Polygon | None:
    polys: list[Polygon] = []
    for room in building.get("rooms") or []:
        floor = room.get("floor_polygon") or []
        pts = [(float(c[0]), float(c[2])) for c in floor if len(c) >= 3]
        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        polys.append(poly)
    if not polys:
        return None
    try:
        merged = unary_union(polys)
    except Exception:
        return None
    if isinstance(merged, Polygon):
        return merged
    geoms = getattr(merged, "geoms", [])
    if not geoms:
        return None
    return max(
        (geom for geom in geoms if isinstance(geom, Polygon)),
        key=lambda geom: float(geom.area),
        default=None,
    )


def _footprint_concavity(poly: Polygon | None) -> tuple[float | None, int | None]:
    if poly is None or poly.is_empty:
        return None, None
    hull = poly.convex_hull
    if hull.is_empty or float(hull.area) <= 1e-9:
        return 0.0, 0
    concavity_ratio = 1.0 - float(poly.area) / float(hull.area)

    pts = list(poly.exterior.coords)[:-1]
    if len(pts) < 4:
        return concavity_ratio, 0
    signed_area_2 = 0.0
    for idx, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(idx + 1) % len(pts)]
        signed_area_2 += x1 * y2 - x2 * y1
    orientation = 1.0 if signed_area_2 >= 0.0 else -1.0
    concave_vertices = 0
    for idx, (bx, by) in enumerate(pts):
        ax, ay = pts[idx - 1]
        cx, cy = pts[(idx + 1) % len(pts)]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if orientation * cross < -1e-8:
            concave_vertices += 1
    return concavity_ratio, concave_vertices


def _building_part_count(roof_result: dict[str, Any] | None) -> int:
    if not roof_result:
        return 0
    part_graph = roof_result.get("building_part_graph") or {}
    parts = part_graph.get("parts")
    if isinstance(parts, dict):
        return len(parts)
    if isinstance(parts, list):
        return len(parts)
    nodes = part_graph.get("nodes")
    if isinstance(nodes, dict):
        return len(nodes)
    if isinstance(nodes, list):
        return len(nodes)
    return 0


def _building_shape_summary(
    building: dict[str, Any], roof_result: dict[str, Any] | None
) -> dict[str, Any]:
    stories = sorted(
        {int(room.get("story", 0)) for room in building.get("rooms") or []}
    )
    top_story = stories[-1] if stories else None
    top_story_room_count = (
        sum(
            1
            for room in building.get("rooms") or []
            if int(room.get("story", 0)) == top_story
        )
        if top_story is not None
        else 0
    )
    floor_union = _room_floor_union(building)
    concavity_ratio, concave_vertices = _footprint_concavity(floor_union)
    part_count = _building_part_count(roof_result)
    return {
        "story_count": len(stories),
        "top_story_room_count": top_story_room_count,
        "building_part_count": part_count,
        "footprint_concavity_ratio": round(float(concavity_ratio), 6)
        if concavity_ratio is not None
        else None,
        "footprint_concave_vertex_count": concave_vertices,
    }


def _classify_architectural_context(shape: dict[str, Any]) -> str:
    story_count = int(shape.get("story_count") or 0)
    top_story_room_count = int(shape.get("top_story_room_count") or 0)
    part_count = int(shape.get("building_part_count") or 0)
    concavity_ratio = float(shape.get("footprint_concavity_ratio") or 0.0)

    if top_story_room_count == 1:
        return "single_top_room_or_attic"
    if story_count <= 1 and (part_count >= 2 or concavity_ratio > 0.08):
        return "single_story_extension_or_LTU"
    if story_count >= 2 and part_count >= 3 and concavity_ratio > 0.08:
        return "multi_story_articulated_mass"
    if story_count >= 2 and part_count >= 3:
        return "multi_story_multi_part"
    if concavity_ratio > 0.08:
        return "concave_footprint"
    return "compact_or_unclear"


def _neighbor_family_count(
    node_id: str,
    neighbors: set[str],
    plane_groups_by_id: dict[str, dict[str, Any]],
    *,
    azimuth_tol_deg: float = 30.0,
) -> int:
    reps: list[float] = []
    for neighbor_id in sorted(neighbors):
        azimuth = plane_groups_by_id.get(neighbor_id, {}).get("azimuth_deg")
        if azimuth is None:
            continue
        placed = False
        for rep in reps:
            delta = _wrapped_angle_delta_deg(rep, azimuth)
            if delta is not None and delta <= azimuth_tol_deg:
                placed = True
                break
        if not placed:
            reps.append(float(azimuth))
    return max(len(reps), 1 if neighbors else 0)


def _same_side_sibling_count(
    node_id: str,
    plane_groups_by_id: dict[str, dict[str, Any]],
    *,
    azimuth_tol_deg: float = 30.0,
    inclination_tol_deg: float = 10.0,
) -> int:
    subject = plane_groups_by_id.get(node_id) or {}
    partner_id = subject.get("best_partner_plane_group_id")
    subject_azimuth = subject.get("azimuth_deg")
    subject_inclination = subject.get("inclination_deg")
    if partner_id is None or subject_azimuth is None or subject_inclination is None:
        return 0
    count = 0
    for other_id, other in plane_groups_by_id.items():
        if other_id == node_id:
            continue
        if other.get("best_partner_plane_group_id") != partner_id:
            continue
        if other.get("azimuth_deg") is None or other.get("inclination_deg") is None:
            continue
        azimuth_delta = _wrapped_angle_delta_deg(
            subject_azimuth, other.get("azimuth_deg")
        )
        if azimuth_delta is None or azimuth_delta > azimuth_tol_deg:
            continue
        if (
            abs(float(subject_inclination) - float(other.get("inclination_deg")))
            > inclination_tol_deg
        ):
            continue
        count += 1
    return count


def _classify_node_pattern(
    node_id: str,
    neighbors: set[str],
    plane_groups_by_id: dict[str, dict[str, Any]],
) -> str:
    family_count = _neighbor_family_count(node_id, neighbors, plane_groups_by_id)
    same_side_siblings = _same_side_sibling_count(node_id, plane_groups_by_id)
    degree = len(neighbors)
    if degree == 2 and family_count == 1:
        return "one_plane_two_same_side_neighbors"
    if degree >= 3 and family_count == 1:
        return "fan_same_opposite_family"
    if family_count >= 2:
        return "multi_neighbor_families"
    if same_side_siblings >= 1:
        return "same_side_siblings_share_partner"
    return "unclassified"


def analyze_multi_partner_plane_groups(
    buildings: list[dict[str, Any]],
    roof_results: dict[str, Any],
    ridge_eave_scores: dict[str, Any],
) -> dict[str, Any]:
    buildings_by_uuid = {
        str(building.get("uuid")): building
        for building in buildings
        if building.get("uuid")
    }
    entries = ridge_eave_scores.get("buildings") or []

    node_pattern_counts: Counter[str] = Counter()
    building_pattern_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    degree_counts: Counter[int] = Counter()
    examples_by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    building_rows: list[dict[str, Any]] = []

    for entry in entries:
        uuid = str(entry.get("building_uuid") or "")
        if not uuid:
            continue
        building = buildings_by_uuid.get(uuid)
        if building is None:
            continue
        selected_plane_groups = [
            plane_group
            for plane_group in (entry.get("plane_groups") or [])
            if plane_group.get("selected") is not False
        ]
        plane_groups_by_id = {
            str(plane_group["id"]): plane_group
            for plane_group in selected_plane_groups
            if plane_group.get("id")
        }
        adjacency: dict[str, set[str]] = defaultdict(set)
        for pair in entry.get("pairs") or []:
            a_id = pair.get("a_plane_group_id")
            b_id = pair.get("b_plane_group_id")
            if a_id not in plane_groups_by_id or b_id not in plane_groups_by_id:
                continue
            adjacency[str(a_id)].add(str(b_id))
            adjacency[str(b_id)].add(str(a_id))

        multi_partner_nodes = [
            node_id for node_id, neighbors in adjacency.items() if len(neighbors) > 1
        ]
        if not multi_partner_nodes:
            continue

        shape = _building_shape_summary(building, roof_results.get(uuid))
        context = _classify_architectural_context(shape)
        context_counts[context] += 1

        node_rows: list[dict[str, Any]] = []
        building_patterns: set[str] = set()
        for node_id in sorted(multi_partner_nodes):
            neighbors = adjacency[node_id]
            degree = len(neighbors)
            degree_counts[degree] += 1
            pattern = _classify_node_pattern(node_id, neighbors, plane_groups_by_id)
            node_pattern_counts[pattern] += 1
            building_patterns.add(pattern)
            row = {
                "plane_group_id": node_id,
                "degree": degree,
                "neighbor_plane_group_ids": sorted(neighbors),
                "neighbor_family_count": _neighbor_family_count(
                    node_id, neighbors, plane_groups_by_id
                ),
                "same_side_sibling_count": _same_side_sibling_count(
                    node_id, plane_groups_by_id
                ),
                "best_partner_plane_group_id": plane_groups_by_id.get(node_id, {}).get(
                    "best_partner_plane_group_id"
                ),
                "azimuth_deg": plane_groups_by_id.get(node_id, {}).get("azimuth_deg"),
                "inclination_deg": plane_groups_by_id.get(node_id, {}).get(
                    "inclination_deg"
                ),
                "pattern": pattern,
            }
            node_rows.append(row)
            if len(examples_by_pattern[pattern]) < 12:
                examples_by_pattern[pattern].append(
                    {
                        "uuid": uuid,
                        **row,
                        "architectural_context": context,
                        "story_count": shape["story_count"],
                        "building_part_count": shape["building_part_count"],
                        "footprint_concavity_ratio": shape["footprint_concavity_ratio"],
                    }
                )

        for pattern in building_patterns:
            building_pattern_counts[pattern] += 1
        building_rows.append(
            {
                "uuid": uuid,
                "architectural_context": context,
                **shape,
                "n_selected_plane_groups": len(selected_plane_groups),
                "n_multi_partner_plane_groups": len(multi_partner_nodes),
                "max_multi_partner_degree": max(
                    len(adjacency[node_id]) for node_id in multi_partner_nodes
                ),
                "patterns": sorted(building_patterns),
                "multi_partner_plane_groups": node_rows,
            }
        )

    building_rows.sort(
        key=lambda row: (
            -int(row["max_multi_partner_degree"]),
            -int(row["n_multi_partner_plane_groups"]),
            str(row["uuid"]),
        )
    )

    return {
        "summary": {
            "n_scored_buildings": len(entries),
            "n_buildings_with_multi_partner_selected_plane_groups": len(building_rows),
            "share_buildings_with_multi_partner_selected_plane_groups": round(
                len(building_rows) / max(len(entries), 1),
                6,
            ),
            "node_pattern_counts": dict(node_pattern_counts),
            "building_pattern_counts": dict(building_pattern_counts),
            "architectural_context_counts": dict(context_counts),
            "degree_counts": {str(k): v for k, v in sorted(degree_counts.items())},
        },
        "examples_by_pattern": dict(examples_by_pattern),
        "buildings": building_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--roof-results", type=Path, default=ROOF_RESULTS_PATH)
    parser.add_argument(
        "--ridge-eave-scores", type=Path, default=RIDGE_EAVE_SCORES_PATH
    )
    parser.add_argument(
        "--uuid", help="Only emit analysis rows for a single building UUID"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with args.buildings.open() as handle:
        buildings = json.load(handle)
    with args.roof_results.open() as handle:
        roof_results = json.load(handle)
    with args.ridge_eave_scores.open() as handle:
        ridge_eave_scores = json.load(handle)

    analysis = analyze_multi_partner_plane_groups(
        buildings, roof_results, ridge_eave_scores
    )
    if args.uuid:
        analysis["buildings"] = [
            row
            for row in analysis["buildings"]
            if str(row.get("uuid")) == str(args.uuid)
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    summary = analysis["summary"]
    print(f"Scored buildings: {summary['n_scored_buildings']}")
    print(
        "Buildings with selected plane-groups intersecting >1 selected plane-group: "
        f"{summary['n_buildings_with_multi_partner_selected_plane_groups']} "
        f"({summary['share_buildings_with_multi_partner_selected_plane_groups']:.3f})"
    )
    print(f"Node patterns: {summary['node_pattern_counts']}")
    print(f"Architectural contexts: {summary['architectural_context_counts']}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
