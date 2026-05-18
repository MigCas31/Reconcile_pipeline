from __future__ import annotations

from collections import defaultdict
from typing import Any

from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid

from .graph_utils import room_key as _room_key
from .roof_cell_complex import _poly_xz_from_3d

AREA_EPS = 0.01
EPS = 1e-6


def _poly_xz(corners: list) -> Polygon | None:
    poly = _poly_xz_from_3d(corners)
    if poly is None or poly.is_empty or poly.area <= AREA_EPS:
        return None
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda geom: geom.area)
        except Exception:
            return None
    return poly if isinstance(poly, Polygon) else None


def _angle_diff(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _knee_wall_bottom_span(knee_wall: dict[str, Any]) -> float:
    corners = knee_wall.get("corners") or []
    if len(corners) < 2:
        return 0.0
    return float(
        LineString(
            [
                (float(corners[0][0]), float(corners[0][2])),
                (float(corners[1][0]), float(corners[1][2])),
            ]
        ).length
    )


def build_roof_evidence_graph(
    *,
    exposed_rooms: list[dict[str, Any]],
    room_partitions: list[dict[str, Any]],
    selected_oblique_surfaces: list[dict[str, Any]],
    building_part_graph: dict[str, Any],
    roof_coverage_graph: dict[str, Any],
    roof_cell_complex: dict[str, Any],
) -> dict[str, Any]:
    room_polys = {
        int(room["room_index"]): _poly_xz(room.get("fp") or [])
        for room in exposed_rooms
    }
    room_polys = {
        room_index: poly for room_index, poly in room_polys.items() if poly is not None
    }

    subparts_by_id = {
        str(subpart["id"]): subpart
        for subpart in (roof_coverage_graph.get("subparts") or [])
        if isinstance(subpart, dict)
    }
    atom_subpart_membership = roof_coverage_graph.get("atom_subpart_membership") or {}
    room_subpart_membership = roof_coverage_graph.get("room_subpart_membership") or {}
    coverage_by_atom = roof_coverage_graph.get("atom_coverage") or {}
    building_part_graph.get("room_membership") or {}

    hypothesis_azimuths = {}
    for surface in selected_oblique_surfaces:
        hypothesis_id = surface.get("roof_hypothesis_id")
        cluster = surface.get("cluster") or {}
        if hypothesis_id and cluster.get("avgAzimuth") is not None:
            hypothesis_azimuths[str(hypothesis_id)] = float(cluster.get("avgAzimuth"))

    cells_by_atom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in roof_cell_complex.get("cells") or []:
        base_atom_id = cell.get("base_atom_id")
        if base_atom_id:
            cells_by_atom[str(base_atom_id)].append(cell)

    knee_walls_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for knee_wall in roof_cell_complex.get("knee_walls") or []:
        room_index = knee_wall.get("room_index")
        if isinstance(room_index, int):
            knee_walls_by_room[room_index].append(knee_wall)

    atom_evidence: dict[str, dict[str, Any]] = {}
    room_evidence: dict[str, dict[str, Any]] = {}
    subpart_evidence: dict[str, dict[str, Any]] = {}
    part_evidence: dict[str, dict[str, Any]] = {}

    for room_partition in room_partitions:
        room_index = int(room_partition["room_index"])
        room_id = _room_key(room_index)
        room_poly = room_polys.get(room_index)
        if room_poly is None:
            continue
        room_boundary = room_poly.boundary

        room_sloped_perimeter = 0.0
        room_flat_cap_under_slope = 0
        room_sloped_atom_count = 0
        room_upper_void_cell_count = 0
        room_attic_cell_count = 0
        hypothesis_ids_on_room: set[str] = set()
        semantic_kinds_on_room: set[str] = set()

        for atom in room_partition.get("partitions") or []:
            atom_id = str(atom["id"])
            atom_poly = _poly_xz(atom.get("poly") or [])
            if atom_poly is None:
                continue
            coverage = coverage_by_atom.get(atom_id) or {}
            cells = cells_by_atom.get(atom_id, [])
            subpart_ids = [
                str(subpart_id)
                for subpart_id in atom_subpart_membership.get(atom_id, [])
            ]
            semantic_kinds = sorted(
                {
                    str(subparts_by_id[subpart_id].get("semantic_kind", "slope_run"))
                    for subpart_id in subpart_ids
                    if subpart_id in subparts_by_id
                }
            )
            try:
                perimeter_len = float(
                    atom_poly.boundary.intersection(room_boundary).length
                )
            except Exception:
                perimeter_len = 0.0
            sloped_context = atom.get("kind") == "oblique" or str(
                coverage.get("sloped_state", "none")
            ) in {
                "confirmed",
                "partial",
            }
            sloped_perimeter_len = perimeter_len if sloped_context else 0.0
            if sloped_context:
                room_sloped_perimeter += sloped_perimeter_len
                room_sloped_atom_count += 1
            if atom.get("kind") == "flat" and str(
                coverage.get("sloped_state", "none")
            ) in {"confirmed", "partial"}:
                room_flat_cap_under_slope += 1
            room_upper_void_cell_count += sum(
                1 for cell in cells if cell.get("cell_kind") == "upper_void"
            )
            room_attic_cell_count += sum(
                1 for cell in cells if cell.get("cell_kind") == "attic"
            )
            hypothesis_id = atom.get("roof_hypothesis_id") or coverage.get(
                "sloped_hypothesis_id"
            )
            if hypothesis_id:
                hypothesis_ids_on_room.add(str(hypothesis_id))
            semantic_kinds_on_room.update(semantic_kinds)

            atom_evidence[atom_id] = {
                "atom_id": atom_id,
                "room_id": room_id,
                "room_index": room_index,
                "kind": atom.get("kind"),
                "perimeter_span_m": round(perimeter_len, 6),
                "sloped_perimeter_span_m": round(sloped_perimeter_len, 6),
                "sloped_context": sloped_context,
                "flat_cap_under_slope": atom.get("kind") == "flat"
                and str(coverage.get("sloped_state", "none"))
                in {"confirmed", "partial"},
                "subpart_ids": subpart_ids,
                "semantic_kinds": semantic_kinds,
                "upper_void_cell_count": sum(
                    1 for cell in cells if cell.get("cell_kind") == "upper_void"
                ),
                "attic_cell_count": sum(
                    1 for cell in cells if cell.get("cell_kind") == "attic"
                ),
            }

        knee_walls = knee_walls_by_room.get(room_index, [])
        knee_wall_span = sum(
            _knee_wall_bottom_span(knee_wall) for knee_wall in knee_walls
        )
        room_boundary_len = max(float(room_boundary.length), EPS)
        opposed_pair_count = 0
        room_hypothesis_ids = sorted(hypothesis_ids_on_room)
        for idx, left_id in enumerate(room_hypothesis_ids):
            left_az = hypothesis_azimuths.get(left_id)
            if left_az is None:
                continue
            for right_id in room_hypothesis_ids[idx + 1 :]:
                right_az = hypothesis_azimuths.get(right_id)
                if right_az is None:
                    continue
                if 140.0 <= _angle_diff(left_az, right_az) <= 220.0:
                    opposed_pair_count += 1

        strong_perimeter_sloped = (
            room_sloped_perimeter >= 1.5
            or (room_sloped_perimeter / room_boundary_len) >= 0.25
        )
        strong_knee_wall_signal = len(knee_walls) > 0 or knee_wall_span >= 0.75
        strong_gable_context = (
            "gable_run" in semantic_kinds_on_room or opposed_pair_count > 0
        )
        strong_upper_void_context = room_upper_void_cell_count > 0 or (
            room_flat_cap_under_slope > 0
            and strong_perimeter_sloped
            and strong_knee_wall_signal
        )
        strong_attic_context = room_attic_cell_count > 0 or (
            room_flat_cap_under_slope > 0
            and strong_gable_context
            and not strong_upper_void_context
        )
        evidence_score = (
            (2 if strong_perimeter_sloped else 0)
            + (2 if strong_knee_wall_signal else 0)
            + (2 if strong_gable_context else 0)
            + min(room_flat_cap_under_slope, 2)
            + min(room_upper_void_cell_count + room_attic_cell_count, 2)
        )

        room_evidence[room_id] = {
            "room_id": room_id,
            "room_index": room_index,
            "sloped_perimeter_span_m": round(room_sloped_perimeter, 6),
            "sloped_perimeter_ratio": round(
                room_sloped_perimeter / room_boundary_len, 6
            ),
            "sloped_atom_count": room_sloped_atom_count,
            "flat_cap_under_slope_count": room_flat_cap_under_slope,
            "knee_wall_count": len(knee_walls),
            "knee_wall_span_m": round(knee_wall_span, 6),
            "upper_void_cell_count": room_upper_void_cell_count,
            "attic_cell_count": room_attic_cell_count,
            "opposed_slope_pair_count": opposed_pair_count,
            "semantic_kinds": sorted(semantic_kinds_on_room),
            "strong_perimeter_sloped": strong_perimeter_sloped,
            "strong_knee_wall_signal": strong_knee_wall_signal,
            "strong_gable_context": strong_gable_context,
            "strong_upper_void_context": strong_upper_void_context,
            "strong_attic_context": strong_attic_context,
            "evidence_score": evidence_score,
        }

    for subpart_id, subpart in subparts_by_id.items():
        room_indices = [
            int(room_index)
            for room_index in (subpart.get("room_indices") or [])
            if isinstance(room_index, int)
        ]
        room_ids = [_room_key(room_index) for room_index in room_indices]
        strong_perimeter_room_count = sum(
            1
            for room_id in room_ids
            if (room_evidence.get(room_id) or {}).get("strong_perimeter_sloped")
        )
        knee_wall_room_count = sum(
            1
            for room_id in room_ids
            if (room_evidence.get(room_id) or {}).get("knee_wall_count", 0) > 0
        )
        flat_cap_room_count = sum(
            1
            for room_id in room_ids
            if (room_evidence.get(room_id) or {}).get("flat_cap_under_slope_count", 0)
            > 0
        )
        evidence_score = (
            strong_perimeter_room_count * 2
            + knee_wall_room_count * 2
            + flat_cap_room_count
            + (2 if subpart.get("semantic_kind", "slope_run") == "gable_run" else 0)
            + (1 if subpart.get("semantic_kind", "slope_run") == "l_t_branch" else 0)
        )
        subpart_evidence[subpart_id] = {
            "subpart_id": subpart_id,
            "roof_hypothesis_id": subpart.get("roof_hypothesis_id"),
            "semantic_kind": subpart.get("semantic_kind", "slope_run"),
            "room_indices": room_indices,
            "part_ids": [str(part_id) for part_id in (subpart.get("part_ids") or [])],
            "room_count": len(room_indices),
            "strong_perimeter_room_count": strong_perimeter_room_count,
            "knee_wall_room_count": knee_wall_room_count,
            "flat_cap_room_count": flat_cap_room_count,
            "evidence_score": evidence_score,
        }

    for part in building_part_graph.get("nodes") or []:
        part_id = str(part["id"])
        room_ids = [str(room_id) for room_id in (part.get("room_ids") or [])]
        subpart_ids = sorted(
            {
                str(subpart_id)
                for room_id in room_ids
                for subpart_id in room_subpart_membership.get(room_id, [])
            }
        )
        gable_subpart_count = sum(
            1
            for subpart_id in subpart_ids
            if (subpart_evidence.get(subpart_id) or {}).get("semantic_kind")
            == "gable_run"
        )
        branch_subpart_count = sum(
            1
            for subpart_id in subpart_ids
            if (subpart_evidence.get(subpart_id) or {}).get("semantic_kind")
            == "l_t_branch"
        )
        strong_sloped_room_count = sum(
            1
            for room_id in room_ids
            if (room_evidence.get(room_id) or {}).get("strong_perimeter_sloped")
        )
        knee_wall_room_count = sum(
            1
            for room_id in room_ids
            if (room_evidence.get(room_id) or {}).get("knee_wall_count", 0) > 0
        )
        opposed_slope_pair_count = sum(
            int((room_evidence.get(room_id) or {}).get("opposed_slope_pair_count", 0))
            for room_id in room_ids
        )
        flat_cap_room_count = sum(
            1
            for room_id in room_ids
            if (room_evidence.get(room_id) or {}).get("flat_cap_under_slope_count", 0)
            > 0
        )
        if gable_subpart_count > 0 or opposed_slope_pair_count > 0:
            refined_family = "gable_or_multi_slope"
        elif (
            strong_sloped_room_count > 0
            or knee_wall_room_count > 0
            or flat_cap_room_count > 0
        ):
            refined_family = "mixed_or_partial"
        else:
            refined_family = str(part.get("roof_family_guess", "flat_or_capped"))
        part_evidence[part_id] = {
            "part_id": part_id,
            "room_ids": room_ids,
            "subpart_ids": subpart_ids,
            "gable_subpart_count": gable_subpart_count,
            "branch_subpart_count": branch_subpart_count,
            "strong_sloped_room_count": strong_sloped_room_count,
            "knee_wall_room_count": knee_wall_room_count,
            "flat_cap_room_count": flat_cap_room_count,
            "opposed_slope_pair_count": opposed_slope_pair_count,
            "refined_roof_family_guess": refined_family,
        }

    return {
        "atom_evidence": atom_evidence,
        "room_evidence": room_evidence,
        "subpart_evidence": subpart_evidence,
        "part_evidence": part_evidence,
        "metadata": {
            "atom_count": len(atom_evidence),
            "room_count": len(room_evidence),
            "subpart_count": len(subpart_evidence),
            "part_count": len(part_evidence),
        },
    }


def annotate_roof_coverage_graph(
    *,
    roof_coverage_graph: dict[str, Any],
    roof_evidence_graph: dict[str, Any],
) -> dict[str, Any]:
    subpart_evidence = roof_evidence_graph.get("subpart_evidence") or {}
    annotated_subparts: list[dict[str, Any]] = []
    for subpart in roof_coverage_graph.get("subparts") or []:
        annotated = dict(subpart)
        evidence = subpart_evidence.get(str(subpart.get("id"))) or {}
        annotated["support_evidence_score"] = int(
            evidence.get("evidence_score", 0) or 0
        )
        annotated["strong_perimeter_room_count"] = int(
            evidence.get("strong_perimeter_room_count", 0) or 0
        )
        annotated["knee_wall_room_count"] = int(
            evidence.get("knee_wall_room_count", 0) or 0
        )
        annotated["flat_cap_room_count"] = int(
            evidence.get("flat_cap_room_count", 0) or 0
        )
        annotated_subparts.append(annotated)

    annotated = dict(roof_coverage_graph)
    annotated["subparts"] = annotated_subparts
    metadata = dict(roof_coverage_graph.get("metadata") or {})
    metadata["evidence_strong_subpart_count"] = sum(
        1
        for subpart in annotated_subparts
        if int(subpart.get("support_evidence_score", 0) or 0) >= 3
    )
    annotated["metadata"] = metadata
    return annotated
