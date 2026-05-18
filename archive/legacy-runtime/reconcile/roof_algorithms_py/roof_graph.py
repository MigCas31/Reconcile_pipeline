from __future__ import annotations

from math import acos, sqrt
from typing import Any

from reconcile_v2.cell_decomposition import split_planar_faces_exact_on_lattice

from .graph_utils import stable_hash as _stable_hash


def _bbox_xz(corners: list[list[float]]) -> tuple[float, float, float, float] | None:
    if len(corners) < 3:
        return None
    xs = [float(c[0]) for c in corners]
    zs = [float(c[2]) for c in corners]
    return min(xs), min(zs), max(xs), max(zs)


def _bbox_gap(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float:
    if a is None or b is None:
        return float("inf")
    ax0, az0, ax1, az1 = a
    bx0, bz0, bx1, bz1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dz = max(bz0 - az1, az0 - bz1, 0.0)
    return sqrt(dx * dx + dz * dz)


def _normal_angle_deg(a: list[float] | None, b: list[float] | None) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 3 or len(b) < 3:
        return 180.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(3))
    dot = max(-1.0, min(1.0, dot))
    return acos(dot) * 180.0 / 3.141592653589793


def _edge(
    *,
    edge_type: str,
    from_id: str,
    to_id: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"edge:{edge_type.lower()}:{
            _stable_hash(
                [edge_type, from_id, to_id, str(evidence)],
                20,
            )
        }",
        "type": edge_type,
        "from": from_id,
        "to": to_id,
        "evidence": evidence,
    }


def build_roof_boundary_graph(
    boundary_model: dict[str, Any], graph=None
) -> dict[str, Any]:
    faces_by_id = {
        face["id"]: face
        for face in boundary_model.get("faces") or []
        if isinstance(face, dict)
    }
    roof_boundaries = [
        boundary
        for boundary in (boundary_model.get("boundaries") or [])
        if boundary.get("role") == "roof"
    ]
    nodes = [
        {
            "id": boundary["id"],
            "type": "Boundary",
            "role": "roof",
            "room_id": boundary.get("room_id"),
            "face_id": boundary.get("face_id"),
            "source": boundary.get("source") or {},
        }
        for boundary in roof_boundaries
    ]
    edges: list[dict[str, Any]] = []

    room_adjacency = {
        (edge.from_id, edge.to_id): edge
        for edge in (graph.edges if graph is not None else [])
        if edge.type == "ADJACENT_TO"
    }

    split_inputs: list[dict[str, Any]] = []
    for boundary in roof_boundaries:
        face = faces_by_id.get(boundary.get("face_id"))
        if not face:
            continue
        split_inputs.append({**face, "boundary_id": boundary["id"]})
    partition = split_planar_faces_exact_on_lattice(split_inputs)

    incidence_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for incidence in partition.get("adjacency", []):
        left_boundary = incidence.get("from_boundary_id")
        right_boundary = incidence.get("to_boundary_id")
        if not left_boundary or not right_boundary or left_boundary == right_boundary:
            continue
        key = tuple(sorted((str(left_boundary), str(right_boundary))))
        entry = incidence_by_pair.setdefault(
            key,
            {
                "shared_edge_length": 0.0,
                "atom_pairs": [],
            },
        )
        entry["shared_edge_length"] += float(incidence.get("shared_edge_length", 0.0))
        entry["atom_pairs"].append(
            [
                incidence.get("from_atom_id"),
                incidence.get("to_atom_id"),
            ]
        )

    roof_boundary_by_id = {boundary["id"]: boundary for boundary in roof_boundaries}
    for (left_id, right_id), incidence in incidence_by_pair.items():
        left = roof_boundary_by_id.get(left_id)
        right = roof_boundary_by_id.get(right_id)
        if left is None or right is None:
            continue
        left_face = faces_by_id.get(left.get("face_id"))
        right_face = faces_by_id.get(right.get("face_id"))
        if not left_face or not right_face:
            continue
        left_src = left.get("source") or {}
        right_src = right.get("source") or {}
        room_a = left.get("room_id")
        room_b = right.get("room_id")
        adjacency_edge = room_adjacency.get((room_a, room_b)) or room_adjacency.get(
            (
                room_b,
                room_a,
            )
        )
        adjacency_state = (
            (adjacency_edge.evidence or {}).get("relation_state")
            if adjacency_edge is not None
            else "unknown"
        )
        shared_edge_length = round(float(incidence["shared_edge_length"]), 6)
        normal_delta = _normal_angle_deg(
            left_face.get("normal"), right_face.get("normal")
        )
        same_story = left_face.get("story") == right_face.get("story")
        same_kind = left_src.get("surface_kind") == right_src.get("surface_kind")
        same_plane_family = normal_delta <= 7.0

        if not (
            same_story and same_kind and same_plane_family and shared_edge_length > 0.01
        ):
            continue

        relation_state = "confirmed" if shared_edge_length >= 0.25 else "partial"
        evidence = {
            "relation_state": relation_state,
            "continuation_kind": left_src.get("surface_kind"),
            "same_story": same_story,
            "shared_edge_length_m": shared_edge_length,
            "normal_delta_deg": round(normal_delta, 6),
            "adjacency_state": adjacency_state,
            "room_ids": [room_a, room_b],
            "exact_face_incidence": True,
            "partition_atom_pairs": incidence["atom_pairs"],
        }
        edges.append(
            _edge(
                edge_type="CONTINUES_AS",
                from_id=left["id"],
                to_id=right["id"],
                evidence=evidence,
            )
        )
        edges.append(
            _edge(
                edge_type="CONTINUES_AS",
                from_id=right["id"],
                to_id=left["id"],
                evidence=evidence,
            )
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "partition_face_count": len(partition.get("atoms", [])),
        },
    }
