from __future__ import annotations

import math
from typing import Any

from .graph_support import match_graph_room, room_surface_bindings
from .graph_utils import stable_hash as _stable_hash

LATTICE_SCALE_MM = 1000


def _snap(value: float) -> int:
    return round(float(value) * LATTICE_SCALE_MM)


def _unsnap(value: int) -> float:
    return value / LATTICE_SCALE_MM


def _snapped_corners(corners: list) -> list[list[float]]:
    out = []
    for c in corners:
        if not isinstance(c, (list, tuple)) or len(c) < 3:
            continue
        out.append(
            [
                _unsnap(_snap(c[0])),
                _unsnap(_snap(c[1])),
                _unsnap(_snap(c[2])),
            ]
        )
    return out


def _face_normal(corners: list[list[float]]) -> list[float]:
    if len(corners) < 3:
        return [0.0, 0.0, 0.0]
    a, b, c = corners[0], corners[1], corners[2]
    ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 1e-9:
        return [0.0, 0.0, 0.0]
    return [nx / norm, ny / norm, nz / norm]


def _make_face(
    *,
    role: str,
    kind: str,
    story: int | None,
    corners: list,
    source: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapped = _snapped_corners(corners)
    return {
        "id": f"face:{_stable_hash([role, kind, str(story), str(snapped)], 20)}",
        "role": role,
        "kind": kind,
        "story": story,
        "corners": snapped,
        "corners_lattice": [[_snap(c[0]), _snap(c[1]), _snap(c[2])] for c in snapped],
        "normal": _face_normal(snapped),
        "source": source or {},
        "properties": properties or {},
    }


def _boundary_record(
    *,
    face_id: str,
    role: str,
    room_id: str | None,
    graph_surface_id: str | None = None,
    boundary_id: str | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": boundary_id
        or f"boundary:{_stable_hash([role, room_id or '', face_id], 20)}",
        "face_id": face_id,
        "role": role,
        "room_id": room_id,
        "graph_surface_id": graph_surface_id,
        "source": source or {},
    }


def _roof_room_ids(bldg: dict, graph, room_indices: list[int]) -> list[str]:
    room_ids: set[str] = set()
    rooms = bldg.get("rooms", [])
    for room_idx in room_indices:
        if not isinstance(room_idx, int) or room_idx < 0 or room_idx >= len(rooms):
            continue
        room = rooms[room_idx] or {}
        graph_room = match_graph_room(
            graph,
            int(room.get("story", 0)),
            room.get("floor_polygon") or [],
        )
        if graph_room is not None:
            room_ids.add(graph_room.id)
    return sorted(room_ids)


def build_boundary_face_model(
    bldg: dict, roof_result: dict, graph=None
) -> dict[str, Any]:
    faces: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []

    for room_idx, room in enumerate(bldg.get("rooms", [])):
        story = int(room.get("story", 0))
        graph_room = match_graph_room(graph, story, room.get("floor_polygon") or [])
        room_id = graph_room.id if graph_room is not None else None
        surface_bindings = room_surface_bindings(graph, room_id) if room_id else {}
        floor_polygon = room.get("floor_polygon") or []
        if len(floor_polygon) >= 3:
            face = _make_face(
                role="slab",
                kind="floor",
                story=story,
                corners=floor_polygon,
                source={"room_index": room_idx, "room_id": room_id},
                properties={"surface_role": "floor_slab"},
            )
            faces.append(face)
            floor_binding = next(
                (
                    binding
                    for binding in surface_bindings.values()
                    if binding.get("surface_kind") == "floor"
                ),
                {},
            )
            boundaries.append(
                _boundary_record(
                    face_id=face["id"],
                    role="slab",
                    room_id=room_id,
                    graph_surface_id=floor_binding.get("graph_surface_id"),
                    boundary_id=floor_binding.get("boundary_id"),
                    source={"room_index": room_idx},
                )
            )
        for wall in room.get("walls_computed", []) or []:
            corners = wall.get("corners") or []
            if len(corners) >= 3:
                face = _make_face(
                    role="wall",
                    kind="wall",
                    story=story,
                    corners=corners,
                    source={
                        "room_index": room_idx,
                        "room_id": room_id,
                        "wall_id": wall.get("id"),
                    },
                    properties={"surface_role": "wall"},
                )
                faces.append(face)
                wall_binding = surface_bindings.get(str(wall.get("id") or ""), {})
                boundaries.append(
                    _boundary_record(
                        face_id=face["id"],
                        role="wall",
                        room_id=room_id,
                        graph_surface_id=wall_binding.get("graph_surface_id"),
                        boundary_id=wall_binding.get("boundary_id"),
                        source={"room_index": room_idx, "wall_id": wall.get("id")},
                    )
                )
            for strip in wall.get("extension_strip") or []:
                if len(strip) >= 3:
                    face = _make_face(
                        role="wall",
                        kind="wall_extension",
                        story=story,
                        corners=strip,
                        source={
                            "room_index": room_idx,
                            "room_id": room_id,
                            "wall_id": wall.get("id"),
                        },
                        properties={"surface_role": "wall_extension"},
                    )
                    faces.append(face)
                    wall_binding = surface_bindings.get(str(wall.get("id") or ""), {})
                    boundaries.append(
                        _boundary_record(
                            face_id=face["id"],
                            role="wall",
                            room_id=room_id,
                            graph_surface_id=wall_binding.get("graph_surface_id"),
                            boundary_id=wall_binding.get("boundary_id"),
                            source={
                                "room_index": room_idx,
                                "wall_id": wall.get("id"),
                                "extension": True,
                            },
                        )
                    )

    oblique_out: list[dict[str, Any]] = []
    for idx, surf in enumerate(
        (roof_result.get("roof_surfaces") or {}).get("oblique", [])
    ):
        corners = surf.get("corners") or []
        if len(corners) < 3:
            continue
        face = _make_face(
            role="roof",
            kind="roof_oblique",
            story=surf.get("dominant_story"),
            corners=corners,
            source={"roof_index": idx},
            properties={"surface_role": "roof_oblique", "legacy_surface": surf},
        )
        faces.append(face)
        oblique_out.append(face)
        roof_room_ids = _roof_room_ids(
            bldg,
            graph,
            (surf.get("cluster") or {}).get("room_indices", []),
        )
        for room_id in roof_room_ids or [None]:
            boundaries.append(
                _boundary_record(
                    face_id=face["id"],
                    role="roof",
                    room_id=room_id,
                    boundary_id=f"boundary:roof:{
                        _stable_hash(
                            [face['id'], room_id or '', 'oblique'],
                            20,
                        )
                    }",
                    source={"roof_index": idx, "surface_kind": "roof_oblique"},
                )
            )

    flat_out: list[dict[str, Any]] = []
    for idx, surf in enumerate(
        (roof_result.get("roof_surfaces") or {}).get("flat", [])
    ):
        corners = surf.get("corners") or []
        if len(corners) < 3:
            continue
        face = _make_face(
            role="roof",
            kind="roof_flat",
            story=surf.get("story"),
            corners=corners,
            source={"roof_index": idx},
            properties={"surface_role": "roof_flat", "legacy_surface": surf},
        )
        faces.append(face)
        flat_out.append(face)
        room_id = None
        if graph is not None:
            story = surf.get("story")
            corners = surf.get("corners") or []
            fp = [[c[0], c[1], c[2]] for c in corners] if corners else []
            graph_room = (
                match_graph_room(graph, story, fp) if isinstance(story, int) else None
            )
            room_id = graph_room.id if graph_room is not None else None
        boundaries.append(
            _boundary_record(
                face_id=face["id"],
                role="roof",
                room_id=room_id,
                boundary_id=f"boundary:roof:{
                    _stable_hash(
                        [face['id'], room_id or '', 'flat'],
                        20,
                    )
                }",
                source={"roof_index": idx, "surface_kind": "roof_flat"},
            )
        )

    return {
        "faces": faces,
        "boundaries": boundaries,
        "roof_faces": {
            "oblique": oblique_out,
            "flat": flat_out,
        },
        "metadata": {
            "backend": "exact_lattice_boundary_faces_v1",
            "numeric_model": "fixed_point_mm",
            "face_count": len(faces),
            "boundary_count": len(boundaries),
            "roof_face_count": len(oblique_out) + len(flat_out),
        },
    }


def derive_roof_surfaces_from_boundary_model(
    boundary_model: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    boundaries_by_face = {}
    for boundary in boundary_model.get("boundaries") or []:
        boundaries_by_face.setdefault(boundary.get("face_id"), []).append(boundary)
    oblique = []
    flat = []
    for face in (boundary_model.get("roof_faces") or {}).get("oblique", []):
        legacy = dict((face.get("properties") or {}).get("legacy_surface") or {})
        legacy["kind"] = str(legacy.get("kind") or "oblique")
        legacy["surface_kind"] = str(legacy.get("surface_kind") or "oblique")
        legacy["corners"] = face["corners"]
        legacy["boundary_face_id"] = face["id"]
        legacy["space_boundary_ids"] = [
            boundary["id"] for boundary in boundaries_by_face.get(face["id"], [])
        ]
        oblique.append(legacy)
    for face in (boundary_model.get("roof_faces") or {}).get("flat", []):
        legacy = dict((face.get("properties") or {}).get("legacy_surface") or {})
        legacy["kind"] = str(legacy.get("kind") or "flat")
        legacy["surface_kind"] = str(legacy.get("surface_kind") or "flat")
        legacy["corners"] = face["corners"]
        legacy["boundary_face_id"] = face["id"]
        legacy["space_boundary_ids"] = [
            boundary["id"] for boundary in boundaries_by_face.get(face["id"], [])
        ]
        flat.append(legacy)
    return {"oblique": oblique, "flat": flat}
