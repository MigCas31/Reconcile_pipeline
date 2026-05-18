"""Backend resolver for shareable viewer element IDs.

Element IDs use the format:
<building_uuid>::<kind>::<id>

Two kind families are supported:

* Legacy kinds (``floor``, ``wall-merged``, ``roof-oblique``, ...) resolve
  against ``buildings_3d.json``.
* Ontology kinds (``ontology-renderable-*``, ``ontology-base-*``,
  ``ontology-knee-wall``, ``ontology-unresolved-coverage``,
  ``ontology-fallback-ceiling``) resolve against
  ``reconcile/roof_algorithms_py_results.json``. Their inner id is
  ``renderable:<category>:<source_id>`` (see
  ``reconcile/viewer_server.py:_renderable_surface_from_atom``), where
  ``source_id`` is one of:

  - a semantic atom id (``ceiling-partition:<hash>``, ``knee-wall:<hash>``, ...)
  - a cell/face composite — either a bare cell (``roof-cell:<kind>:<hash>``)
    or a face within a cell
    (``roof-cell:<kind>:<hash>:arr-face:<hash>``), resolved against
    ``roof_cell_complex.cells[*]`` and its ``.faces[*]``
  - a viewer-assembled ``roof-atom-patch:{kind}:`` prefix wrapping an
    atom id (e.g. ``roof-atom-patch:flat:ceiling-partition:<hash>``) —
    the prefix is stripped automatically during lookup.
* Raw split kinds (``raw-eave-split``, ``raw-eave-split-v1``,
  ``raw-eave-split-v2``) resolve against scorer sidecars.
* Tier payload kinds (``tier-room``, ``tier-wall``, ``tier-ceiling``,
  ``tier-knee-wall``, ``tier-dormer-face``, ...) resolve against
  ``pipeline-outputs/<uuid>/tier_payload.json``.
* ``wall-stitch`` resolves against ``building["stitch_walls"]`` via each
  entry's ``id`` field. The generator stamps IDs of the form
  ``stitch:<story>:<index>`` for vertical quads and
  ``stitch_floor:<story>:<index>`` / ``stitch_ceiling:<story>:<index>``
  for cap triangles; the index is a monotonic counter into
  ``stitch_walls`` at emission time. Snapped stitches override the id
  with ``snap:<wall_id>`` (set in
  ``reconcile/extract3d/stitch.py:snap_stitches_to_non_owner_walls``).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedElementId:
    """Parsed viewer element locator token."""

    building_uuid: str
    kind: str
    element_id: str


def parse_element_id(token: str) -> ParsedElementId:
    """Parse a shareable element token.

    Expected format: ``building_uuid::kind::id``.
    """
    parts = token.strip().split("::", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Invalid element ID. Expected format: <building_uuid>::<kind>::<id>"
        )
    return ParsedElementId(
        building_uuid=parts[0],
        kind=parts[1],
        element_id=parts[2],
    )


RAW_EAVE_SPLIT_KINDS = {"raw-eave-split", "raw-eave-split-v1", "raw-eave-split-v2"}


def _raw_eave_split_version(kind: str) -> str:
    if kind == "raw-eave-split-v2":
        return "v2"
    return "v1"


def _iter_room_elements(building: dict, room_key: str):
    for room_index, room in enumerate(building.get("rooms", [])):
        story = room.get("story")
        for element_index, element in enumerate(room.get(room_key, [])):
            yield room_index, story, element_index, element


def _split_wall_id_suffix(element_id: str) -> tuple[str, int | None, int | None]:
    """Split optional :<story>:<room_index> suffix appended by the full-model viewer.

    Full-model render path (viewer-main.js:1924) produces ``wall_id:story:ri``.
    Layer render path (viewer-main.js:1311) produces the bare ``wall_id``.
    Returns (wall_id, story_hint, room_index_hint); hints are None when absent.
    """
    parts = element_id.rsplit(":", 2)
    if len(parts) == 3:
        try:
            return parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            pass
    return element_id, None, None


def _find_in_room_collection(
    building: dict, room_key: str, element_id: str
) -> dict | None:
    # Try exact match first (layer-mode IDs: bare wall UUID).
    # Also try stripping the :<story>:<room_index> suffix produced by the
    # full-model viewer (viewer-main.js:1924).
    bare_id, story_hint, ri_hint = _split_wall_id_suffix(element_id)
    for room_index, story, element_index, element in _iter_room_elements(
        building, room_key
    ):
        raw_id = str(element.get("id", ""))
        exact = raw_id == element_id
        stripped = (
            raw_id == bare_id
            and (story_hint is None or story == story_hint)
            and (ri_hint is None or room_index == ri_hint)
        )
        if exact or stripped:
            return {
                "json_path": f"rooms[{room_index}].{room_key}[{element_index}]",
                "room_index": room_index,
                "story": story,
                "source": element.get("source"),
                "corners_count": len(element.get("corners", [])),
                "element": element,
            }
    return None


def _find_in_building_collection(
    building: dict, collection_key: str, element_id: str
) -> dict | None:
    for index, element in enumerate(building.get(collection_key, [])):
        candidate_id = element.get("id")
        if candidate_id is None:
            candidate_id = f"{element.get('type', '')}:{index}"
        if str(candidate_id) == element_id:
            return {
                "json_path": f"{collection_key}[{index}]",
                "story": element.get("story"),
                "source": element.get("source") or element.get("confidence"),
                "corners_count": len(
                    element.get("corners", [])
                    or element.get("element_corners", [])
                    or element.get("wall_corners", [])
                ),
                "element": element,
            }
    if collection_key == "gap_walls":
        fallback = _find_gap_wall_by_viewer_fallback_id(building, element_id)
        if fallback is not None:
            index, element = fallback
            return {
                "json_path": f"{collection_key}[{index}]",
                "story": element.get("story"),
                "source": element.get("source") or element.get("confidence"),
                "corners_count": len(
                    element.get("corners", [])
                    or element.get("element_corners", [])
                    or element.get("wall_corners", [])
                ),
                "element": element,
            }
    return None


def _is_meshable_xz_polygon(corners: list[list[float]]) -> bool:
    """Approximate viewer mesh creation success from corner cardinality."""
    return sum(1 for corner in corners if len(corner) >= 3) >= 3


def _find_gap_wall_by_viewer_fallback_id(
    building: dict, element_id: str
) -> tuple[int, dict] | None:
    """Resolve legacy gap-wall IDs minted from ``groups.gaps.children.length``.

    Newer artifacts persist deterministic ``gw:...`` IDs directly on
    ``building["gap_walls"]``. This fallback only exists so older artifacts
    that used the viewer child-counter scheme still resolve.
    """
    match = re.fullmatch(r"([^:]+):(\d+)", str(element_id))
    if match is None:
        return None
    target_counter = int(match.group(2))

    gaps_children_count = 0
    for gap in building.get("cross_floor_gaps", []):
        corners = gap.get("corners") or []
        if len(corners) < 3 or gap.get("type") == "cross_story":
            continue
        if _is_meshable_xz_polygon(corners):
            gaps_children_count += 1
        gaps_children_count += 1  # edge loop

    for gap_wall_index, gap_wall in enumerate(building.get("gap_walls", [])):
        corners = gap_wall.get("corners") or []
        if len(corners) < 3:
            continue
        candidate = f"{gap_wall.get('type', 'wall')}:{gaps_children_count}"
        if candidate == element_id and gaps_children_count == target_counter:
            return gap_wall_index, gap_wall
        if _is_meshable_xz_polygon(corners):
            gaps_children_count += 1
        gaps_children_count += 1  # edge loop
    return None


def _find_legacy_roofish_element_in_roof_results(
    roof_results_for_uuid: dict,
    parsed: ParsedElementId,
) -> dict | None:
    roofish_kind_to_path: dict[str, tuple[str, str]] = {
        "roof-oblique": ("roof_surfaces", "oblique"),
        "roof-flat": ("roof_surfaces", "flat"),
        "ceiling-flat": ("ceiling", "flat"),
        "ceiling-oblique": ("ceiling", "oblique"),
        "ceiling-simple-slant": ("ceiling", "simple_slant"),
    }
    section_path = roofish_kind_to_path.get(parsed.kind)
    if section_path is None:
        return None
    section, subsection = section_path
    try:
        _, idx_str = parsed.element_id.rsplit(":", 1)
        idx = int(idx_str)
    except ValueError:
        return None
    items = (roof_results_for_uuid.get(section) or {}).get(subsection) or []
    if idx < 0 or idx >= len(items):
        return None
    element = items[idx]
    corners = (
        element.get("corners") or element.get("poly") or element.get("polygon") or []
    )
    return {
        "json_path": f"{section}.{subsection}[{idx}]",
        "room_index": element.get("room_index"),
        "story": element.get("story") or element.get("dominant_story"),
        "source": parsed.kind,
        "corners_count": len(corners),
        "element": element,
    }


def is_ontology_kind(kind: str) -> bool:
    """Return True for kinds produced by ``full-model-ontology.js``."""
    return kind.startswith("ontology-")


def is_tier_kind(kind: str) -> bool:
    """Return True for kinds produced by ``reconcile_tiers/web``."""
    return kind.startswith("tier-")


def is_v3_kind(kind: str) -> bool:
    """Return True for kinds produced by ``reconcile_v3``."""
    return kind.startswith("v3-")


_RECONSTRUCTION_KINDS = {
    "ceiling-reconstruction-dormer": "dormer",
    "ceiling-reconstruction-wing": "wing",
}


def is_reconstruction_kind(kind: str) -> bool:
    """Return True for Phase-2 raw-ceiling prototype reconstruction kinds."""
    return kind in _RECONSTRUCTION_KINDS


def _tier_floor_pieces(room: dict) -> list[dict]:
    floor = room.get("floor") or []
    if isinstance(floor, dict):
        return [floor]
    if isinstance(floor, list):
        return [piece for piece in floor if isinstance(piece, dict)]
    return []


def _tier_corners(element: dict) -> list:
    corners = element.get("corners")
    if corners:
        return corners
    floor_pieces = _tier_floor_pieces(element)
    if not floor_pieces:
        return []
    if len(floor_pieces) == 1:
        return floor_pieces[0].get("corners") or []
    out = []
    for piece in floor_pieces:
        out.extend(piece.get("corners") or [])
    return out


def find_tier_payload_element(tier_payload: dict, parsed: ParsedElementId) -> dict:
    """Resolve a ``reconcile_tiers`` static-viewer locator against tier_payload."""
    token = f"{parsed.building_uuid}::{parsed.kind}::{parsed.element_id}"

    for room_index, room in enumerate(tier_payload.get("rooms") or []):
        if room.get("locator_id") == token:
            floor_pieces = _tier_floor_pieces(room)
            corners = _tier_corners(room)
            json_path = (
                f"rooms[{room_index}].floor"
                if len(floor_pieces) != 1
                else f"rooms[{room_index}].floor[0]"
                if isinstance(room.get("floor"), list)
                else f"rooms[{room_index}].floor"
            )
            return {
                "json_path": json_path,
                "room_index": room_index,
                "story": room.get("story"),
                "source": "tier_payload",
                "corners_count": len(corners),
                "element": {"id": token, "corners": corners, "room": room},
            }
        for wall_index, wall in enumerate(room.get("walls") or []):
            if wall.get("locator_id") == token:
                corners = wall.get("corners") or []
                return {
                    "json_path": f"rooms[{room_index}].walls[{wall_index}]",
                    "room_index": room_index,
                    "story": room.get("story"),
                    "source": "tier_payload",
                    "corners_count": len(corners),
                    "element": wall,
                }
            if f"{wall.get('locator_id')}:extension" == token:
                corners = wall.get("extension_strip") or []
                return {
                    "json_path": (
                        f"rooms[{room_index}].walls[{wall_index}].extension_strip"
                    ),
                    "room_index": room_index,
                    "story": room.get("story"),
                    "source": "tier_payload",
                    "corners_count": len(corners),
                    "element": {"id": token, "corners": corners, "parent_wall": wall},
                }
        for collection_key in ("doors", "windows"):
            for element_index, element in enumerate(room.get(collection_key) or []):
                candidate = (
                    f"{room.get('locator_id')}:{collection_key[:-1]}:{element_index}"
                )
                if candidate == token:
                    corners = element.get("corners") or []
                    return {
                        "json_path": (
                            f"rooms[{room_index}].{collection_key}[{element_index}]"
                        ),
                        "room_index": room_index,
                        "story": room.get("story"),
                        "source": "tier_payload",
                        "corners_count": len(corners),
                        "element": element,
                    }

    top_level = (
        ("gaps", "gaps"),
        ("ceiling", "ceiling"),
        ("knee_walls", "knee_walls"),
        ("dormer_faces", "dormer_faces"),
    )
    for collection_key, json_key in top_level:
        for index, element in enumerate(tier_payload.get(collection_key) or []):
            if element.get("locator_id") != token:
                continue
            corners = _tier_corners(element)
            return {
                "json_path": f"{json_key}[{index}]",
                "room_index": element.get("room_index"),
                "story": element.get("story"),
                "source": element.get("source")
                or element.get("kind")
                or "tier_payload",
                "corners_count": len(corners),
                "element": element,
            }

    raise LookupError(
        f"Tier element not found for kind='{parsed.kind}' and id='{parsed.element_id}'"
    )


def find_reconstruction_element(reconstructions: dict, parsed: ParsedElementId) -> dict:
    """Resolve a ``ceiling-reconstruction-*`` token against the sidecar JSON.

    ``reconstructions`` is the object loaded from
    ``reports/raw_ceiling_prototype/reconstructions.json``; its shape is
    ``{"buildings": {"<uuid>": {"dormer": [...], "wing": [...]}}}``.
    """
    subkey = _RECONSTRUCTION_KINDS[parsed.kind]
    by_uuid = (reconstructions.get("buildings") or {}).get(parsed.building_uuid)
    if not by_uuid:
        raise LookupError(
            f"No reconstructions for building {parsed.building_uuid}. "
            "Run the prototype scripts under scripts/prototype_*.py."
        )
    pieces = by_uuid.get(subkey) or []
    token = f"{parsed.building_uuid}::{parsed.kind}::{parsed.element_id}"
    for idx, piece in enumerate(pieces):
        if piece.get("element_id") == token:
            corners = piece.get("corners") or []
            return {
                "building_uuid": parsed.building_uuid,
                "kind": parsed.kind,
                "id": parsed.element_id,
                "json_path": f"buildings[{parsed.building_uuid}].{subkey}[{idx}]",
                "room_index": piece.get("room_index"),
                "story": piece.get("story"),
                "source": piece.get("piece_role"),
                "corners_count": len(corners),
                "element": piece,
            }
    raise LookupError(
        f"Reconstruction piece not found: {token}. The sidecar may be stale — "
        "rerun scripts/prototype_dormer_reconstruction.py / "
        "prototype_wing_reconstruction.py."
    )


def find_raw_eave_split_element(
    raw_ceiling_plane_splits: dict, parsed: ParsedElementId
) -> dict:
    """Resolve a ``raw-eave-split`` element against the scorer sidecar."""
    by_uuid = (raw_ceiling_plane_splits.get("buildings") or {}).get(
        parsed.building_uuid
    )
    if not by_uuid:
        raise LookupError(
            f"No raw-eave split payload for building {parsed.building_uuid}. "
            "Run scripts/prototype_raw_ceiling_plane_scorer.py first."
        )

    for idx, piece in enumerate(by_uuid):
        piece_id = str(piece.get("piece_id") or "")
        target_id = str(piece.get("target_element_id") or "")
        if parsed.element_id not in {piece_id, target_id}:
            continue
        corners = piece.get("corners") or []
        return {
            "building_uuid": parsed.building_uuid,
            "kind": parsed.kind,
            "id": parsed.element_id,
            "json_path": f"buildings[{parsed.building_uuid}][{idx}]",
            "room_index": None,
            "story": piece.get("story"),
            "source": (
                f"{piece.get('piece_role') or 'unknown'}:"
                f"{piece.get('target_kind') or 'unknown'}"
            ),
            "corners_count": len(corners),
            "element": piece,
        }

    raise LookupError(
        f"Raw-eave split piece not found for id='{parsed.element_id}' "
        f"in building {parsed.building_uuid}"
    )


def _id_tail(token: str | None) -> str:
    if not token:
        return ""
    parts = str(token).split("::", 2)
    if len(parts) == 3:
        return parts[2]
    return str(token)


def _sidecar_entry_for_uuid(data: dict | list | None, uuid: str) -> dict | None:
    if data is None:
        return None
    if isinstance(data, list):
        return next(
            (
                entry
                for entry in data
                if isinstance(entry, dict) and entry.get("building_uuid") == uuid
            ),
            None,
        )
    if isinstance(data, dict):
        buildings = data.get("buildings")
        if isinstance(buildings, dict):
            value = buildings.get(uuid)
            if isinstance(value, dict):
                return value
            if isinstance(value, list):
                return {"building_uuid": uuid, "rows": value}
        if isinstance(buildings, list):
            return next(
                (
                    entry
                    for entry in buildings
                    if isinstance(entry, dict) and entry.get("building_uuid") == uuid
                ),
                None,
            )
    return None


def _sidecar_rows_for_uuid(data: dict | list | None, uuid: str) -> list[dict]:
    entry = _sidecar_entry_for_uuid(data, uuid)
    if entry is None:
        return []
    if isinstance(entry.get("rows"), list):
        return [row for row in entry.get("rows", []) if isinstance(row, dict)]
    if isinstance(entry.get("faces"), list):
        return [row for row in entry.get("faces", []) if isinstance(row, dict)]
    if isinstance(entry.get("selected_faces"), list):
        return [row for row in entry.get("selected_faces", []) if isinstance(row, dict)]
    if isinstance(entry, dict) and isinstance(entry.get("candidates"), list):
        return [row for row in entry.get("candidates", []) if isinstance(row, dict)]
    return []


def find_ceiling_raw_element(building: dict, parsed: ParsedElementId) -> dict:
    """Resolve a ``ceiling-raw`` token against ``rooms[*].raw_ceiling_planes``."""
    try:
        story_str, room_index_str, plane_index_str = parsed.element_id.split(":", 2)
        story = int(story_str)
        room_index = int(room_index_str)
        plane_index = int(plane_index_str)
    except ValueError as exc:
        raise LookupError(
            f"ceiling-raw id must be <story>:<room_index>:<plane_index>, got "
            f"'{parsed.element_id}'"
        ) from exc

    rooms = building.get("rooms") or []
    if room_index < 0 or room_index >= len(rooms):
        raise LookupError(
            f"Room index out of range for ceiling-raw id '{parsed.element_id}'"
        )
    room = rooms[room_index]
    planes = room.get("raw_ceiling_planes") or []
    if plane_index < 0 or plane_index >= len(planes):
        raise LookupError(
            f"Plane index out of range for ceiling-raw id '{parsed.element_id}'"
        )
    plane = planes[plane_index]
    corners = plane.get("corners") or []
    return {
        "json_path": f"rooms[{room_index}].raw_ceiling_planes[{plane_index}]",
        "room_index": room_index,
        "story": story,
        "source": room.get("raw_ceiling_source") or "scan",
        "corners_count": len(corners),
        "element": plane,
    }


def find_thermal_ceiling_element(
    building: dict | None,
    roof_results_for_uuid: dict | None,
    parsed: ParsedElementId,
) -> dict:
    """Resolve ``thermal-ceiling`` IDs from gap overlays or roof thermal output."""
    parts = parsed.element_id.split(":")
    if len(parts) == 3 and parts[1] == "gap":
        if building is None:
            raise LookupError("thermal-ceiling gap ids require buildings payload")
        idx = int(parts[2])
        gaps = building.get("cross_floor_gaps") or []
        gap = gaps[idx]
        corners = gap.get("ceiling_corners") or gap.get("corners") or []
        return {
            "json_path": f"cross_floor_gaps[{idx}]",
            "room_index": None,
            "story": gap.get("story"),
            "source": "cross_floor_gap",
            "corners_count": len(corners),
            "element": {
                "id": parsed.element_id,
                "kind": "thermal-ceiling",
                "corners": corners,
                "gap_type": gap.get("type"),
            },
        }
    if len(parts) == 3 and parts[1] == "closure-ceil":
        if building is None:
            raise LookupError("thermal-ceiling closure ids require buildings payload")
        idx = int(parts[2])
        closures = building.get("gap_closures") or []
        closure = closures[idx]
        corners = closure.get("corners") or []
        return {
            "json_path": f"gap_closures[{idx}]",
            "room_index": None,
            "story": closure.get("story"),
            "source": "gap_closure",
            "corners_count": len(corners),
            "element": {
                "id": parsed.element_id,
                "kind": "thermal-ceiling",
                "corners": corners,
                "closure_type": closure.get("type"),
            },
        }
    if len(parts) == 3:
        try:
            story = int(parts[1])
            idx = int(parts[2])
        except ValueError as exc:
            raise LookupError(
                f"Unrecognized thermal-ceiling id '{parsed.element_id}'"
            ) from exc
        thermal = ((roof_results_for_uuid or {}).get("ceiling") or {}).get(
            "thermal"
        ) or []
        if idx < 0 or idx >= len(thermal):
            raise LookupError(
                f"thermal-ceiling index out of range for id '{parsed.element_id}'"
            )
        element = thermal[idx]
        corners = element.get("poly") or element.get("corners") or []
        return {
            "json_path": f"ceiling.thermal[{idx}]",
            "room_index": element.get("room_index"),
            "story": element.get("story", story),
            "source": "roof_results_thermal",
            "corners_count": len(corners),
            "element": element,
        }
    raise LookupError(f"Unrecognized thermal-ceiling id '{parsed.element_id}'")


def find_candidate_face_element(
    candidate_faces: dict | list, parsed: ParsedElementId
) -> dict:
    per_uuid = _sidecar_entry_for_uuid(candidate_faces, parsed.building_uuid)
    if per_uuid is None:
        raise LookupError(
            f"No candidate-face payload for building {parsed.building_uuid}. "
            "Run scripts/build_candidate_faces.py first."
        )
    collection_key = "faces"
    faces = [face for face in (per_uuid.get("faces") or []) if isinstance(face, dict)]
    if not faces:
        collection_key = "candidates"
        faces = [
            face
            for face in (per_uuid.get("candidates") or [])
            if isinstance(face, dict)
        ]
    full_id = f"{parsed.building_uuid}::candidate::{parsed.element_id}"
    for idx, face in enumerate(faces):
        candidate_id = str(face.get("id") or "")
        if (
            parsed.element_id in {candidate_id, _id_tail(candidate_id)}
            or candidate_id == full_id
        ):
            corners = face.get("footprint_xz") or []
            return {
                "building_uuid": parsed.building_uuid,
                "kind": parsed.kind,
                "id": parsed.element_id,
                "json_path": (
                    f"candidate_faces[{parsed.building_uuid}].{collection_key}[{idx}]"
                ),
                "room_index": face.get("room_index"),
                "story": face.get("story"),
                "source": "candidate-face",
                "corners_count": len(corners),
                "element": face,
            }
    raise LookupError(
        f"candidate-face not found for id='{parsed.element_id}' "
        f"in building {parsed.building_uuid}"
    )


def find_reconstruction_face_element(
    reconstruction: dict | list,
    candidate_faces: dict | list | None,
    parsed: ParsedElementId,
) -> dict:
    per_uuid = _sidecar_entry_for_uuid(reconstruction, parsed.building_uuid)
    if per_uuid is None:
        raise LookupError(
            f"No reconstruction payload for building {parsed.building_uuid}. "
            "Run scripts/run_reconstruction_solver.py first."
        )

    selected_faces = [
        face
        for face in (per_uuid.get("selected_faces") or [])
        if isinstance(face, dict)
    ]
    if not selected_faces and candidate_faces is not None:
        candidate_entry = (
            _sidecar_entry_for_uuid(candidate_faces, parsed.building_uuid) or {}
        )
        candidate_rows = (
            candidate_entry.get("faces") or candidate_entry.get("candidates") or []
        )
        candidate_by_id = {
            str(face.get("id") or ""): face
            for face in candidate_rows
            if isinstance(face, dict) and face.get("id")
        }
        selected_faces = [
            candidate_by_id[selected_id]
            for selected_id in (per_uuid.get("selected_face_ids") or [])
            if selected_id in candidate_by_id
        ]

    for idx, face in enumerate(selected_faces):
        face_id = str(face.get("id") or "")
        if parsed.element_id in {face_id, _id_tail(face_id)}:
            corners = face.get("footprint_xz") or []
            return {
                "building_uuid": parsed.building_uuid,
                "kind": parsed.kind,
                "id": parsed.element_id,
                "json_path": (
                    f"reconstruction[{parsed.building_uuid}].selected_faces[{idx}]"
                ),
                "room_index": face.get("room_index"),
                "story": face.get("story"),
                "source": per_uuid.get("decision") or per_uuid.get("status"),
                "corners_count": len(corners),
                "element": {
                    **face,
                    "decision": per_uuid.get("decision"),
                    "status": per_uuid.get("status"),
                },
            }

    raise LookupError(
        f"reconstruction-face not found for id='{parsed.element_id}' "
        f"in building {parsed.building_uuid}"
    )


def find_ridge_eave_candidate_element(
    ridge_eave_scores: dict | list,
    parsed: ParsedElementId,
) -> dict:
    per_uuid = _sidecar_entry_for_uuid(ridge_eave_scores, parsed.building_uuid)
    if per_uuid is None:
        raise LookupError(
            f"No ridge/eave scores for building {parsed.building_uuid}. "
            "Run scripts/score_candidates_ridge_eave.py first."
        )
    element_id = parsed.element_id
    base_plane_group_id = element_id.removesuffix("::below-ridge").removesuffix(
        "::above-ridge"
    )
    for idx, group in enumerate(per_uuid.get("plane_groups") or []):
        group_id = str(group.get("id") or "")
        group_tail = _id_tail(group_id)
        if base_plane_group_id in {
            group_id,
            group_tail,
            f"plane-group::{group_tail}",
        }:
            return {
                "building_uuid": parsed.building_uuid,
                "kind": parsed.kind,
                "id": parsed.element_id,
                "json_path": (
                    f"ridge_eave_scores[{parsed.building_uuid}].plane_groups[{idx}]"
                ),
                "room_index": None,
                "story": group.get("story"),
                "source": "plane-group",
                "corners_count": len(group.get("union_xz") or []),
                "element": group,
            }
    for idx, candidate in enumerate(per_uuid.get("candidates") or []):
        candidate_id = str(candidate.get("id") or "")
        if element_id in {candidate_id, _id_tail(candidate_id)}:
            return {
                "building_uuid": parsed.building_uuid,
                "kind": parsed.kind,
                "id": parsed.element_id,
                "json_path": (
                    f"ridge_eave_scores[{parsed.building_uuid}].candidates[{idx}]"
                ),
                "room_index": candidate.get("room_index"),
                "story": candidate.get("story"),
                "source": "scored-candidate",
                "corners_count": 0,
                "element": candidate,
            }
    raise LookupError(
        f"ridge-eave-candidate not found for id='{parsed.element_id}' "
        f"in building {parsed.building_uuid}"
    )


def find_roof_overextend_element(
    computed_overextend: dict,
    parsed: ParsedElementId,
) -> dict:
    rows = ((computed_overextend or {}).get("buildings") or {}).get(
        parsed.building_uuid
    ) or []
    for idx, piece in enumerate(rows):
        piece_id = f"{piece.get('surface_kind')}:{piece.get('surface_index')}"
        if piece_id != parsed.element_id:
            continue
        corners = piece.get("corners") or []
        return {
            "building_uuid": parsed.building_uuid,
            "kind": parsed.kind,
            "id": parsed.element_id,
            "json_path": (
                f"computed_overextend.buildings[{parsed.building_uuid}][{idx}]"
            ),
            "room_index": piece.get("room_index"),
            "story": piece.get("story"),
            "source": "computed-overextend",
            "corners_count": len(corners),
            "element": piece,
        }
    raise LookupError(
        f"roof-overextend not found for id='{parsed.element_id}' "
        f"in building {parsed.building_uuid}"
    )


def find_raw_disagreement_element(
    raw_disagreement: dict, parsed: ParsedElementId
) -> dict:
    rows = ((raw_disagreement or {}).get("buildings") or {}).get(
        parsed.building_uuid
    ) or []
    for idx, piece in enumerate(rows):
        if str(piece.get("pair_index")) != parsed.element_id:
            continue
        corners = piece.get("corners") or []
        return {
            "building_uuid": parsed.building_uuid,
            "kind": parsed.kind,
            "id": parsed.element_id,
            "json_path": f"raw_disagreement.buildings[{parsed.building_uuid}][{idx}]",
            "room_index": piece.get("room_index"),
            "story": piece.get("story"),
            "source": "raw-disagreement",
            "corners_count": len(corners),
            "element": piece,
        }
    raise LookupError(
        f"raw-disagreement not found for id='{parsed.element_id}' "
        f"in building {parsed.building_uuid}"
    )


def find_clean_ceiling_element(
    ceiling_replacement: dict, parsed: ParsedElementId
) -> dict:
    rows = ((ceiling_replacement or {}).get("buildings") or {}).get(
        parsed.building_uuid
    ) or []
    full = f"{parsed.building_uuid}::clean-ceiling::{parsed.element_id}"
    for idx, piece in enumerate(rows):
        piece_id = str(piece.get("element_id") or "")
        if parsed.element_id not in {piece_id, _id_tail(piece_id)} and piece_id != full:
            continue
        corners = piece.get("corners") or []
        return {
            "building_uuid": parsed.building_uuid,
            "kind": parsed.kind,
            "id": parsed.element_id,
            "json_path": (
                f"ceiling_replacement.buildings[{parsed.building_uuid}][{idx}]"
            ),
            "room_index": piece.get("room_index"),
            "story": piece.get("story"),
            "source": piece.get("replacement_mode") or "clean-ceiling",
            "corners_count": len(corners),
            "element": piece,
        }
    raise LookupError(
        f"clean-ceiling not found for id='{parsed.element_id}' "
        f"in building {parsed.building_uuid}"
    )


_V3_COLLECTION_BY_KIND: dict[str, str] = {
    "v3-part": "parts",
    "v3-gap": "gaps",
    "v3-slab": "slabs",
    "v3-wall-extension": "wall_extensions",
    "v3-flat-ceiling": "flat_ceilings",
    "v3-slanted-roof": "slanted_roofs",
    "v3-roof-proposal": "roof_proposals",
    "v3-merged-roof-segment": "merged_roof_segments",
    "v3-dormer": "dormers",
    "v3-unresolved": "unresolved",
}

_V3_GABLE_KINDS = {"v3-gable-status", "v3-gable-ridge", "v3-gable-extension-target"}


def find_v3_element(v3_results_for_uuid: dict, parsed: ParsedElementId) -> dict:
    """Resolve a ``v3-*`` element ID against reconcile_v3_results.json."""
    if parsed.kind in _V3_GABLE_KINDS:
        return _find_v3_gable_element(v3_results_for_uuid, parsed)
    collection_key = _V3_COLLECTION_BY_KIND.get(parsed.kind)
    if collection_key is None:
        raise LookupError(f"Unknown v3 kind: {parsed.kind}")
    collection = v3_results_for_uuid.get(collection_key) or []
    token = f"{parsed.building_uuid}::{parsed.kind}::{parsed.element_id}"
    for idx, element in enumerate(collection):
        if element.get("id") == token:
            audit_entries = [
                entry
                for entry in (v3_results_for_uuid.get("audit_log") or [])
                if entry.get("element_id") == token
            ]
            return {
                "building_uuid": parsed.building_uuid,
                "kind": parsed.kind,
                "id": parsed.element_id,
                "json_path": f"{collection_key}[{idx}]",
                "element": element,
                "audit": audit_entries,
            }
    raise LookupError(
        f"v3 element not found for kind='{parsed.kind}' id='{parsed.element_id}'"
    )


def _find_v3_gable_element(v3_results_for_uuid: dict, parsed: ParsedElementId) -> dict:
    """Resolve gable decorations — all carry the part's inner id (e.g. ``part-0``)."""
    part_token = f"{parsed.building_uuid}::v3-part::{parsed.element_id}"
    parts = v3_results_for_uuid.get("parts") or []
    for idx, part in enumerate(parts):
        if part.get("id") != part_token:
            continue
        gable = part.get("gable_extension")
        if gable is None:
            raise LookupError(f"Part {parsed.element_id} has no gable_extension record")
        audit_token = f"{part_token}::gable"
        audit_entries = [
            entry
            for entry in (v3_results_for_uuid.get("audit_log") or [])
            if entry.get("element_id") == audit_token
        ]
        return {
            "building_uuid": parsed.building_uuid,
            "kind": parsed.kind,
            "id": parsed.element_id,
            "json_path": f"parts[{idx}].gable_extension",
            "element": gable,
            "audit": audit_entries,
        }
    raise LookupError(
        f"v3 gable element not found for kind='{parsed.kind}' id='{parsed.element_id}'"
    )


# Maps ontology kind -> (file:line, note) list of thresholds that influenced
# surfaces of this kind. Cited locations are authoritative constants only —
# keep in sync with the source files when constants move.
KIND_THRESHOLDS: dict[str, list[tuple[str, str]]] = {
    "ontology-renderable-ceiling": [
        (
            "reconcile/roof_algorithms_py/roof_partitioning.py:22-23",
            "ROOM_TOP_MIN_CLEARANCE_M=0.15, ROOM_TOP_SHELL_TOL_M=0.08",
        ),
        (
            "reconcile/roof_algorithms_py/thermal_ceiling.py:551",
            "THRESHOLD_M=0.30 (knee-wall gap)",
        ),
        (
            "reconcile/roof_algorithms_py/oblique_clustering.py:19-20",
            "COPLANAR_TOL=0.5m, Δazimuth=30°, Δinclination=15°, MIN_CLUSTER_SIZE=2",
        ),
        (
            "reconcile/roof_algorithms_py/simple_slant.py:21",
            "_AZIMUTH_MULTI_THRESHOLD=90° (simple-slant fallback)",
        ),
    ],
    "ontology-renderable-roof": [
        (
            "reconcile/roof_algorithms_py/oblique_clustering.py:19-20",
            "COPLANAR_TOL=0.5m, Δazimuth=30°, Δinclination=15°",
        ),
        (
            "reconcile/roof_algorithms_py/roof_envelope_continuation.py:17",
            "MAX_CONTINUATION_DISTANCE_M=4.0",
        ),
        (
            "reconcile/roof_algorithms_py/roof_partitioning.py:22-23",
            "ROOM_TOP_MIN_CLEARANCE_M=0.15, ROOM_TOP_SHELL_TOL_M=0.08",
        ),
    ],
    "ontology-knee-wall": [
        (
            "reconcile/roof_algorithms_py/thermal_ceiling.py:551",
            "THRESHOLD_M=0.30",
        ),
        (
            "reconcile/roof_algorithms_py/roof_partitioning.py:22-23",
            "ROOM_TOP_* tolerances",
        ),
    ],
    "ontology-unresolved-coverage": [
        (
            "reconcile/roof_algorithms_py/roof_coverage_graph.py:15",
            "SEED_ROOM_BUFFER_M=0.75",
        ),
    ],
    "ontology-fallback-ceiling": [
        (
            "reconcile/roof_algorithms_py/simple_slant.py:21",
            "_AZIMUTH_MULTI_THRESHOLD=90° (simple-slant fallback)",
        ),
        (
            "reconcile/roof_algorithms_py/roof_partitioning.py:22-23",
            "ROOM_TOP_* tolerances",
        ),
    ],
    "ontology-base-exterior-wall": [],
    "ontology-base-interior-wall": [],
    "ontology-base-floor": [],
    "ontology-base-ceiling": [],
    "ontology-renderable-wall": [],
    "ontology-renderable-room-wall": [],
    "ontology-renderable-floor": [],
    "ontology-base-window": [],
    "ontology-base-door": [],
    "ontology-base-opening": [],
}


# Maps atom id prefix -> (file, human description). Used by --trace to point at
# the pipeline step that minted an atom given its id.
ATOM_PREFIX_TO_STEP: dict[str, tuple[str, str]] = {
    "ceiling-partition:": (
        "reconcile/roof_algorithms_py/roof_partitioning.py",
        "derive_room_ceiling_partitions / flat+oblique/simple-slant partitioning",
    ),
    "knee-wall:": (
        "reconcile/roof_algorithms_py/thermal_ceiling.py",
        "knee-wall detection",
    ),
    "implicit-flat-atom:": (
        "reconcile/roof_algorithms_py/occupied_room_cell_complex.py",
        "implicit flat attic / remainder cell synthesis",
    ),
    "occupied-cell:": (
        "reconcile/roof_algorithms_py/occupied_room_cell_complex.py",
        "occupied-room cell complex build",
    ),
    "boundary:": (
        "reconcile/roof_algorithms_py/boundary_model.py",
        "space-boundary assignment (IFC-style)",
    ),
    "face:": (
        "reconcile/roof_algorithms_py/boundary_model.py",
        "boundary face id (top shell surface)",
    ),
    "edge:occupied-atom:": (
        "reconcile/roof_algorithms_py/occupied_room_cell_complex.py",
        "occupied cell ↔ atom edge",
    ),
    "edge:atom-room:": (
        "reconcile/roof_algorithms_py/top_boundary_graph.py",
        "top-boundary atom ↔ room edge",
    ),
    "roof-atom-patch:": (
        "reconcile/viewer_server.py",
        "flat/oblique ceiling-partition atom emitted as exterior_roof patch "
        "(viewer-assembled; strip roof-atom-patch:{kind}: to get the atom id)",
    ),
    "roof-cell:": (
        "reconcile/roof_algorithms_py/roof_cell_complex.py",
        "arranged polyhedral roof cell (attic/upper_void) and its per-face "
        "arrangement (arr-face:<hash>). Composite source_id is "
        "<cell_id>[:arr-face:<face_hash>].",
    ),
}


_FACE_PREFIXES: tuple[str, ...] = ("arr-face:",)


def _split_roof_cell_composite(source_id: str) -> tuple[str, str | None] | None:
    """Parse ``roof-cell:<kind>:<hash>[:arr-face:<face_hash>]``.

    Returns ``(cell_id, face_id)`` — ``face_id`` is ``None`` for bare
    cell references. Returns ``None`` when ``source_id`` isn't a
    roof-cell composite.
    """
    if not source_id.startswith("roof-cell:"):
        return None
    for prefix in _FACE_PREFIXES:
        sep = f":{prefix}"
        idx = source_id.find(sep)
        if idx != -1:
            return source_id[:idx], source_id[idx + 1 :]
    return source_id, None


def _resolve_roof_cell_source(
    roof_results_for_uuid: dict, source_id: str
) -> dict | None:
    """Resolve a cell/face composite against ``roof_cell_complex.cells``.

    Returns a dict with ``atom`` (the face dict, or the cell when no face
    suffix is present), ``provenance_paths``, and ``parent_cell`` metadata
    — or ``None`` if ``source_id`` doesn't match the composite shape or
    isn't present in the cell complex.
    """
    split = _split_roof_cell_composite(source_id)
    if split is None:
        return None
    cell_id, face_id = split
    cells = (roof_results_for_uuid.get("roof_cell_complex") or {}).get("cells") or []
    for cell_idx, cell in enumerate(cells):
        if str(cell.get("id")) != cell_id:
            continue
        parent_cell = {
            "id": cell.get("id"),
            "cell_kind": cell.get("cell_kind"),
            "room_id": cell.get("room_id"),
            "room_index": cell.get("room_index"),
            "story": cell.get("story"),
            "part_id": cell.get("part_id"),
            "base_atom_id": cell.get("base_atom_id"),
            "roof_hypothesis_id": cell.get("roof_hypothesis_id"),
            "roof_surface_kind": cell.get("roof_surface_kind"),
            "volume_m3": cell.get("volume_m3"),
            "bbox_xyz": cell.get("bbox_xyz"),
        }
        if face_id is None:
            return {
                "atom": cell,
                "provenance_paths": [f"roof_cell_complex.cells[{cell_idx}]"],
                "parent_cell": parent_cell,
            }
        for face_idx, face in enumerate(cell.get("faces") or []):
            if str(face.get("id")) != face_id:
                continue
            return {
                "atom": face,
                "provenance_paths": [
                    f"roof_cell_complex.cells[{cell_idx}].faces[{face_idx}]"
                ],
                "parent_cell": parent_cell,
            }
        return None
    return None


def _parse_renderable_element_id(element_id: str) -> dict | None:
    """Split ``renderable:<category>:<source_id>`` into its parts.

    ``source_id`` may itself contain colons (atom IDs like
    ``ceiling-partition:<hash>``), so we only split the first two ``:``.
    """
    parts = element_id.split(":", 2)
    if len(parts) != 3 or parts[0] != "renderable":
        return None
    return {"category": parts[1], "source_id": parts[2]}


def _atom_step_for(source_id: str) -> tuple[str, str] | None:
    for prefix, step in ATOM_PREFIX_TO_STEP.items():
        if source_id.startswith(prefix):
            return step
    return None


def find_ontology_element(
    roof_results_for_uuid: dict,
    parsed: ParsedElementId,
) -> dict:
    """Resolve an ``ontology-*`` element ID against a per-building roof result.

    ``roof_results_for_uuid`` is the value from
    ``roof_algorithms_py_results.json`` indexed by building UUID.
    """
    info = _parse_renderable_element_id(parsed.element_id)
    source_id = parsed.element_id if info is None else info["source_id"]
    # viewer_server.py assembles roof-atom-patch:{kind}:{atom_id} ids at HTTP
    # time; strip the prefix to get the underlying atom id for JSON lookups.
    atom_id = source_id
    if source_id.startswith("roof-atom-patch:"):
        _, _, atom_id = source_id.split(":", 2)
    atom: dict | None = None
    provenance_paths: list[str] = []
    parent_cell: dict | None = None

    cell_resolution = _resolve_roof_cell_source(roof_results_for_uuid, atom_id)
    if cell_resolution is not None:
        atom = cell_resolution["atom"]
        provenance_paths.extend(cell_resolution["provenance_paths"])
        parent_cell = cell_resolution["parent_cell"]

    ceiling_partitions = roof_results_for_uuid.get("ceiling_partitions", {}) or {}
    for subkind in ("oblique", "flat"):
        for idx, part in enumerate(ceiling_partitions.get(subkind, []) or []):
            if str(part.get("id", "")) == atom_id:
                atom = atom or part
                provenance_paths.append(f"ceiling_partitions.{subkind}[{idx}]")
    for idx, rp in enumerate(ceiling_partitions.get("room_partitions", []) or []):
        for p_idx, part in enumerate(rp.get("partitions", []) or []):
            if str(part.get("id", "")) == atom_id:
                atom = atom or part
                provenance_paths.append(
                    f"ceiling_partitions.room_partitions[{idx}].partitions[{p_idx}]"
                )

    for idx, kw in enumerate(roof_results_for_uuid.get("knee_walls", []) or []):
        if str(kw.get("id", "")) == atom_id:
            atom = atom or kw
            provenance_paths.append(f"knee_walls[{idx}]")

    for idx, node in enumerate(
        (roof_results_for_uuid.get("building_part_graph") or {}).get("nodes") or []
    ):
        if str((node or {}).get("id", "")) == atom_id:
            atom = atom or node
            provenance_paths.append(f"building_part_graph.nodes[{idx}]")

    for idx, subpart in enumerate(
        (roof_results_for_uuid.get("roof_coverage_graph") or {}).get("subparts") or []
    ):
        if str((subpart or {}).get("id", "")) == atom_id:
            atom = atom or subpart
            provenance_paths.append(f"roof_coverage_graph.subparts[{idx}]")
        if (
            atom is None
            and atom_id.startswith("roof-coverage-patch:")
            and f":{(subpart or {}).get('id', '')!s}:" in atom_id
        ):
            atom = {
                "id": atom_id,
                "coverage_subpart_id": subpart.get("id"),
                "roof_hypothesis_id": str(atom_id).split(":", 4)[3]
                if str(atom_id).startswith("roof-coverage-patch:roof-hypothesis:")
                else None,
                "subpart": subpart,
            }
            provenance_paths.append(f"roof_coverage_graph.subparts[{idx}]")

    continuation_regions = (
        roof_results_for_uuid.get("roof_continuation_diagnostics") or {}
    ).get("continuation_regions") or []
    for idx, region in enumerate(continuation_regions):
        if str((region or {}).get("id", "")) == atom_id:
            atom = atom or region
            provenance_paths.append(
                f"roof_continuation_diagnostics.continuation_regions[{idx}]"
            )

    if atom is None and atom_id in {
        "building-part:full-building",
        "building-part:unassigned",
    }:
        atom = {"id": atom_id, "synthetic": True}
        provenance_paths.append("synthetic")

    if atom is None and atom_id.startswith("unresolved-coverage:"):
        room_id = atom_id.removeprefix("unresolved-coverage:")
        room_summary = (
            (roof_results_for_uuid.get("top_boundary_graph") or {}).get(
                "room_summaries"
            )
            or {}
        ).get(room_id)
        if isinstance(room_summary, dict):
            atom = {"id": atom_id, "room_id": room_id, **room_summary}
            provenance_paths.append(f"top_boundary_graph.room_summaries[{room_id}]")

    for dormer_idx, dormer in enumerate(roof_results_for_uuid.get("dormers") or []):
        for cheek_idx, cheek in enumerate((dormer or {}).get("cheeks") or []):
            dormer_id = str((dormer or {}).get("id") or f"dormer:{dormer_idx}")
            cheek_id = f"{dormer_id}:cheek:{cheek_idx}"
            if atom_id == cheek_id:
                atom = atom or cheek
                provenance_paths.append(f"dormers[{dormer_idx}].cheeks[{cheek_idx}]")
        header_id = str((dormer or {}).get("id") or f"dormer:{dormer_idx}")
        if atom_id == header_id and isinstance((dormer or {}).get("header"), dict):
            atom = atom or (dormer or {}).get("header")
            provenance_paths.append(f"dormers[{dormer_idx}].header")

    atom_evidence_map = (
        roof_results_for_uuid.get("roof_evidence_graph", {}) or {}
    ).get("atom_evidence", {}) or {}
    evidence = atom_evidence_map.get(atom_id)
    if evidence is not None:
        provenance_paths.append(f"roof_evidence_graph.atom_evidence[{atom_id}]")

    # Mirror viewer_server.py: build partition_by_id from room_partitions so
    # the TBG node (which carries role/roof_hypothesis_id) can be enriched with
    # the stored polygon geometry.
    _partition_by_id: dict[str, dict] = {}
    for _rp in ceiling_partitions.get("room_partitions", []) or []:
        for _part in _rp.get("partitions") or []:
            _pid = str(_part.get("id") or "")
            if _pid:
                _partition_by_id[_pid] = _part

    top_boundary = roof_results_for_uuid.get("top_boundary_graph", {}) or {}
    for idx, node in enumerate(top_boundary.get("nodes", []) or []):
        if not isinstance(node, dict) or str(node.get("id", "")) != atom_id:
            continue
        # Use the TBG node as the authoritative atom (it carries role,
        # roof_hypothesis_id, flat_role, etc.) and fill in geometry from the
        # matching room_partition entry — exactly what viewer_server.py does
        # when assembling semantic_atoms.
        if node.get("type") == "TopBoundaryAtom":
            merged = dict(node)
            _partition = _partition_by_id.get(atom_id) or {}
            merged["poly"] = _partition.get("poly") or merged.get("poly") or []
            merged["top_y_m"] = (
                _partition.get("top_y_m") if "top_y_m" not in node else node["top_y_m"]
            )
            atom = merged
        provenance_paths.append(f"top_boundary_graph.nodes[{idx}]")

    if atom is None and not provenance_paths:
        raise LookupError(
            f"Ontology source '{source_id}' not found in roof results for "
            f"building {parsed.building_uuid}. The results file may be stale "
            f"(hash changed after the ID was captured — try rerunning the "
            f"pipeline for this building). If this is a viewer-assembled "
            "composite id not yet handled by the locator, fetch the live "
            "payload with "
            "'curl http://127.0.0.1:8080/ontology-artifacts?uuid=<uuid>&view=full-model'."
        )

    return {
        "building_uuid": parsed.building_uuid,
        "kind": parsed.kind,
        "id": parsed.element_id,
        "category": (info or {}).get("category"),
        "source_id": source_id,
        "atom": atom,
        "provenance_paths": provenance_paths,
        "evidence": evidence,
        "parent_cell": parent_cell,
    }


def build_trace(result: dict) -> dict:
    """Attach threshold and pipeline-step hints to an ontology resolution."""
    kind = result.get("kind", "")
    thresholds = [
        {"location": loc, "note": note} for loc, note in KIND_THRESHOLDS.get(kind, [])
    ]
    step_info = None
    source_id = result.get("source_id", "")
    if source_id:
        # Unwrap viewer-assembled prefix before step lookup so we point at the
        # pipeline step that minted the atom, not viewer_server.py.
        step_source_id = source_id
        if source_id.startswith("roof-atom-patch:"):
            _, _, step_source_id = source_id.split(":", 2)
        hit = _atom_step_for(step_source_id)
        if hit is not None:
            step_info = {"file": hit[0], "description": hit[1]}
    return {
        **result,
        "thresholds": thresholds,
        "pipeline_step": step_info,
    }


def find_element(
    buildings: list[dict],
    token: str,
    *,
    tier_payloads: dict | None = None,
    roof_results: dict | None = None,
    v3_results: list[dict] | None = None,
    reconstructions: dict | None = None,
    raw_ceiling_plane_splits: dict | None = None,
    raw_ceiling_plane_splits_v2: dict | None = None,
    candidate_faces: dict | list | None = None,
    reconstruction: dict | list | None = None,
    ridge_eave_scores: dict | list | None = None,
    computed_overextend: dict | None = None,
    raw_disagreement: dict | None = None,
    ceiling_replacement: dict | None = None,
) -> dict:
    """Resolve a viewer token to an element in buildings_3d-like data.

    ``roof_results`` (contents of ``roof_algorithms_py_results.json``) is
    required for ``ontology-*`` kinds. ``v3_results`` (contents of
    ``reconcile_v3_results.json``, a list of per-building dicts) is
    required for ``v3-*`` kinds. ``reconstructions`` (contents of
    ``reports/raw_ceiling_prototype/reconstructions.json``) is required
    for ``ceiling-reconstruction-*`` kinds.
    ``raw_ceiling_plane_splits`` (contents of
    ``reports/raw_ceiling_plane_scorer/plane_extent_splits.json``) is
    required for ``raw-eave-split`` and ``raw-eave-split-v1`` kinds.
    ``raw_ceiling_plane_splits_v2`` is required for ``raw-eave-split-v2``
    kinds.
    """
    parsed = parse_element_id(token)

    if is_tier_kind(parsed.kind):
        if tier_payloads is None:
            raise LookupError(
                f"Tier kind '{parsed.kind}' requires tier payload data; pass "
                "tier_payloads=... or use --pipeline-dir on the CLI."
            )
        tier_payload = tier_payloads.get(parsed.building_uuid)
        if tier_payload is None:
            raise LookupError(
                f"Building {parsed.building_uuid} not found in tier payloads"
            )
        result = find_tier_payload_element(tier_payload, parsed)
        return {
            "building_uuid": parsed.building_uuid,
            "building_address": tier_payload.get("address"),
            "kind": parsed.kind,
            "id": parsed.element_id,
            **result,
        }

    if parsed.kind == "candidate-face":
        if candidate_faces is None:
            raise LookupError(
                "candidate-face requires candidate faces sidecar; pass "
                "candidate_faces=... or use --candidate-faces on the CLI."
            )
        return find_candidate_face_element(candidate_faces, parsed)
    if parsed.kind == "reconstruction-face":
        if reconstruction is None:
            raise LookupError(
                "reconstruction-face requires reconstruction sidecar; pass "
                "reconstruction=... or use --reconstruction-results on the CLI."
            )
        return find_reconstruction_face_element(
            reconstruction,
            candidate_faces,
            parsed,
        )
    if parsed.kind == "ridge-eave-candidate":
        if ridge_eave_scores is None:
            raise LookupError(
                "ridge-eave-candidate requires ridge/eave scores sidecar; pass "
                "ridge_eave_scores=... or use --ridge-eave-scores on the CLI."
            )
        return find_ridge_eave_candidate_element(ridge_eave_scores, parsed)
    if parsed.kind == "roof-overextend":
        if computed_overextend is None:
            raise LookupError(
                "roof-overextend requires computed overextend sidecar; pass "
                "computed_overextend=... or use --computed-overextend on the CLI."
            )
        return find_roof_overextend_element(computed_overextend, parsed)
    if parsed.kind == "raw-disagreement":
        if raw_disagreement is None:
            raise LookupError(
                "raw-disagreement requires raw disagreement sidecar; pass "
                "raw_disagreement=... or use --raw-disagreement on the CLI."
            )
        return find_raw_disagreement_element(raw_disagreement, parsed)
    if parsed.kind == "clean-ceiling":
        if ceiling_replacement is None:
            raise LookupError(
                "clean-ceiling requires replacement sidecar; pass "
                "ceiling_replacement=... or use --ceiling-replacement on the CLI."
            )
        return find_clean_ceiling_element(ceiling_replacement, parsed)

    if parsed.kind in RAW_EAVE_SPLIT_KINDS:
        version = _raw_eave_split_version(parsed.kind)
        sidecar = (
            raw_ceiling_plane_splits_v2 if version == "v2" else raw_ceiling_plane_splits
        )
        if sidecar is None:
            if version == "v2":
                raise LookupError(
                    "raw-eave-split-v2 requires V2 plane_extent_splits sidecar; "
                    "pass raw_ceiling_plane_splits_v2=... or use "
                    "--raw-ceiling-plane-splits-v2 on the CLI."
                )
            raise LookupError(
                f"{parsed.kind} requires plane_extent_splits sidecar; pass "
                "raw_ceiling_plane_splits=... or use --raw-ceiling-plane-splits "
                "on the CLI."
            )
        return find_raw_eave_split_element(sidecar, parsed)

    if is_reconstruction_kind(parsed.kind):
        if reconstructions is None:
            raise LookupError(
                f"Reconstruction kind '{parsed.kind}' requires reconstructions "
                "sidecar; pass reconstructions=... or use --reconstructions on "
                "the CLI."
            )
        return find_reconstruction_element(reconstructions, parsed)

    if is_ontology_kind(parsed.kind):
        if roof_results is None:
            raise LookupError(
                f"Ontology kind '{parsed.kind}' requires "
                f"roof_algorithms_py_results.json; "
                "pass roof_results=... or use --roof-results on the CLI."
            )
        per_uuid = roof_results.get(parsed.building_uuid)
        if per_uuid is None:
            raise LookupError(
                f"Building {parsed.building_uuid} not found in roof results"
            )
        return find_ontology_element(per_uuid, parsed)

    if is_v3_kind(parsed.kind):
        if v3_results is None:
            raise LookupError(
                f"v3 kind '{parsed.kind}' requires reconcile_v3_results.json; "
                "pass v3_results=... or use --v3-results on the CLI."
            )
        per_uuid = next(
            (b for b in v3_results if b.get("building_uuid") == parsed.building_uuid),
            None,
        )
        if per_uuid is None:
            raise LookupError(
                f"Building {parsed.building_uuid} not found in v3 results"
            )
        return find_v3_element(per_uuid, parsed)

    roof_results_for_uuid = None
    if roof_results is not None:
        roof_results_for_uuid = roof_results.get(parsed.building_uuid)

    building = next(
        (b for b in buildings if b.get("uuid") == parsed.building_uuid),
        None,
    )
    if building is None and roof_results_for_uuid is None:
        raise LookupError(f"Building not found: {parsed.building_uuid}")

    room_kind_to_key = {
        "wall-merged": "walls_merged",
        "wall-computed": "walls_computed",
        "wall-extension": "walls_computed",
        "door": "doors",
        "window": "windows",
        "wall-clipped-original": "walls_computed",
    }
    building_kind_to_key = {
        "gap-cross-story": "cross_floor_gaps",
        "gap-within-story": "cross_floor_gaps",
        "wall-stitch": "stitch_walls",
        "gap-wall": "gap_walls",
        "exterior-gap-element": "exterior_gap_indicators",
        "exterior-gap-wall": "exterior_gap_indicators",
        "gap-closure": "gap_closures",
    }
    roof_kind_to_path: dict[str, tuple[str, str]] = {
        "roof-oblique": ("roof_surfaces", "oblique"),
        "roof-flat": ("roof_surfaces", "flat"),
        "ceiling-flat": ("ceiling", "flat"),
        "ceiling-oblique": ("ceiling", "oblique"),
        "ceiling-simple-slant": ("ceiling", "simple_slant"),
    }

    details: dict | None = None
    if parsed.kind == "ceiling-raw":
        if building is not None:
            try:
                details = find_ceiling_raw_element(building, parsed)
            except (LookupError, ValueError, IndexError, KeyError):
                details = None
    elif parsed.kind == "thermal-ceiling":
        try:
            details = find_thermal_ceiling_element(
                building,
                roof_results_for_uuid,
                parsed,
            )
        except (LookupError, ValueError, IndexError, KeyError):
            details = None
    elif parsed.kind == "floor" or parsed.kind == "floor-overlap":
        try:
            story_str, room_index_str = parsed.element_id.split(":", 1)
            room_index = int(room_index_str)
            if building is None:
                details = None
                raise LookupError
            room = building.get("rooms", [])[room_index]
            key = (
                "floor_overlap_region"
                if parsed.kind == "floor-overlap"
                else "floor_polygon"
            )
            corners = room.get(key, [])
            details = {
                "json_path": f"rooms[{room_index}].{key}",
                "room_index": room_index,
                "story": int(story_str),
                "source": key,
                "corners_count": len(corners),
                "element": {"id": parsed.element_id, "corners": corners},
            }
        except (LookupError, ValueError, IndexError, KeyError):
            details = None
    elif parsed.kind in room_kind_to_key:
        if building is not None:
            details = _find_in_room_collection(
                building, room_kind_to_key[parsed.kind], parsed.element_id
            )
    elif parsed.kind in building_kind_to_key:
        if building is not None:
            details = _find_in_building_collection(
                building, building_kind_to_key[parsed.kind], parsed.element_id
            )
    elif parsed.kind in roof_kind_to_path:
        section, subsection = roof_kind_to_path[parsed.kind]
        if building is not None:
            try:
                # ID format: "<sub_kind>:<index>" e.g. "oblique:0"
                _, idx_str = parsed.element_id.rsplit(":", 1)
                idx = int(idx_str)
                items = building.get(section, {}).get(subsection, [])
                element = items[idx]
                corners = element.get("corners") or element.get("poly", [])
                details = {
                    "json_path": f"{section}.{subsection}[{idx}]",
                    "room_index": None,
                    "story": element.get("story") or element.get("dominant_story"),
                    "source": parsed.kind,
                    "corners_count": len(corners),
                    "element": element,
                }
            except (ValueError, IndexError, KeyError):
                details = None
        if details is None and roof_results_for_uuid is not None:
            details = _find_legacy_roofish_element_in_roof_results(
                roof_results_for_uuid, parsed
            )

    elif parsed.kind in ("dormer-cheek", "dormer-header"):
        if building is not None:
            try:
                parts = parsed.element_id.split(":")
                dormer_idx = int(parts[1])
                dormers = building.get("dormers", [])
                dormer = dormers[dormer_idx]
                if parsed.kind == "dormer-cheek":
                    cheek_idx = int(parts[2])
                    element = dormer["cheeks"][cheek_idx]
                else:
                    element = dormer["header"]
                corners = element.get("corners", [])
                details = {
                    "json_path": f"dormers[{dormer_idx}].{parsed.kind.split('-')[1]}",
                    "room_index": dormer.get("room_index"),
                    "story": None,
                    "source": parsed.kind,
                    "corners_count": len(corners),
                    "element": element,
                }
            except (ValueError, IndexError, KeyError):
                details = None

    if details is None:
        raise LookupError(
            f"Element not found for kind='{parsed.kind}' and id='{parsed.element_id}'"
        )

    return {
        "building_uuid": (
            building.get("uuid") if building is not None else parsed.building_uuid
        ),
        "building_address": building.get("address") if building is not None else None,
        "kind": parsed.kind,
        "id": parsed.element_id,
        **details,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve shareable viewer element IDs. Legacy kinds resolve from "
            "buildings_3d.json; ontology-* kinds resolve from "
            "roof_algorithms_py_results.json."
        )
    )
    parser.add_argument(
        "--buildings-json",
        type=Path,
        default=Path("reconcile/buildings_3d.json"),
        help="Path to buildings_3d.json (default: reconcile/buildings_3d.json)",
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path("pipeline-outputs"),
        help="Directory containing per-building tier_payload.json files (default: "
        "pipeline-outputs)",
    )
    parser.add_argument(
        "--roof-results",
        type=Path,
        default=Path("reconcile/roof_algorithms_py_results.json"),
        help="Path to roof_algorithms_py_results.json (needed for ontology kinds)",
    )
    parser.add_argument(
        "--v3-results",
        type=Path,
        default=Path("reconcile/reconcile_v3_results.json"),
        help="Path to reconcile_v3_results.json (needed for v3-* kinds)",
    )
    parser.add_argument(
        "--reconstructions",
        type=Path,
        default=Path("reports/raw_ceiling_prototype/reconstructions.json"),
        help=(
            "Path to reports/raw_ceiling_prototype/reconstructions.json "
            "(needed for ceiling-reconstruction-* kinds)"
        ),
    )
    parser.add_argument(
        "--raw-ceiling-plane-splits",
        type=Path,
        default=Path("reports/raw_ceiling_plane_scorer/plane_extent_splits.json"),
        help=(
            "Path to reports/raw_ceiling_plane_scorer/plane_extent_splits.json "
            "(needed for raw-eave-split / raw-eave-split-v1 kinds)"
        ),
    )
    parser.add_argument(
        "--raw-ceiling-plane-splits-v2",
        type=Path,
        default=Path(
            ".context/raw_ceiling_plane_scorer_v2_full/plane_extent_splits.json"
        ),
        help=(
            "Path to .context/raw_ceiling_plane_scorer_v2_full/"
            "plane_extent_splits.json (needed for raw-eave-split-v2 kinds)"
        ),
    )
    parser.add_argument(
        "--candidate-faces",
        type=Path,
        default=Path("reports/candidate_faces_20260419/candidates.json"),
        help=(
            "Path to candidate faces sidecar (needed for candidate-face / "
            "reconstruction-face / ridge-eave-candidate kinds)"
        ),
    )
    parser.add_argument(
        "--reconstruction-results",
        type=Path,
        default=Path("reports/reconstruction_20260419_topologyfix/selections.json"),
        help=(
            "Path to reconstruction selections sidecar (needed for reconstruction-face "
            "kinds)"
        ),
    )
    parser.add_argument(
        "--ridge-eave-scores",
        type=Path,
        default=Path("reports/ridge_eave_scores_20260420/scores.json"),
        help="Path to ridge/eave scores sidecar (needed for ridge-eave-candidate "
        "kinds)",
    )
    parser.add_argument(
        "--computed-overextend",
        type=Path,
        default=Path("reports/computed_extent_vs_raw/overextend_polygons.json"),
        help="Path to computed overextend sidecar (needed for roof-overextend kinds)",
    )
    parser.add_argument(
        "--raw-disagreement",
        type=Path,
        default=Path("reports/raw_orientation_disagreement/disagreement_polygons.json"),
        help="Path to raw disagreement sidecar (needed for raw-disagreement kinds)",
    )
    parser.add_argument(
        "--ceiling-replacement",
        type=Path,
        default=Path("reports/noisy_slanted_ceilings/replacement_polygons.json"),
        help="Path to clean ceiling replacement sidecar (needed for clean-ceiling "
        "kinds)",
    )
    parser.add_argument(
        "--element-id",
        required=True,
        help="Element token in format <building_uuid>::<kind>::<id>",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Enrich the result with threshold citations and the pipeline step "
            "that produced the atom."
        ),
    )
    args = parser.parse_args()

    parsed = parse_element_id(args.element_id)

    def _load_if_exists(path: Path):
        if not path.exists():
            return None
        return json.loads(path.read_text())

    roof_results = None
    if args.roof_results.exists():
        roof_results = json.loads(args.roof_results.read_text())
    raw_ceiling_plane_splits = None
    if args.raw_ceiling_plane_splits.exists():
        raw_ceiling_plane_splits = json.loads(args.raw_ceiling_plane_splits.read_text())
    raw_ceiling_plane_splits_v2 = None
    if args.raw_ceiling_plane_splits_v2.exists():
        raw_ceiling_plane_splits_v2 = json.loads(
            args.raw_ceiling_plane_splits_v2.read_text()
        )
    candidate_faces = _load_if_exists(args.candidate_faces)
    reconstruction = _load_if_exists(args.reconstruction_results)
    ridge_eave_scores = _load_if_exists(args.ridge_eave_scores)
    computed_overextend = _load_if_exists(args.computed_overextend)
    raw_disagreement = _load_if_exists(args.raw_disagreement)
    ceiling_replacement = _load_if_exists(args.ceiling_replacement)
    sidecar_only_kinds = {
        "candidate-face",
        "reconstruction-face",
        "ridge-eave-candidate",
        "roof-overextend",
        "raw-disagreement",
        "clean-ceiling",
    }
    tier_payloads = None
    if is_tier_kind(parsed.kind):
        payload_path = args.pipeline_dir / parsed.building_uuid / "tier_payload.json"
        if not payload_path.exists():
            raise LookupError(f"Tier payload not found: {payload_path}")
        tier_payloads = {parsed.building_uuid: json.loads(payload_path.read_text())}

    if parsed.kind in RAW_EAVE_SPLIT_KINDS:
        result = find_element(
            [],
            args.element_id,
            raw_ceiling_plane_splits=raw_ceiling_plane_splits,
            raw_ceiling_plane_splits_v2=raw_ceiling_plane_splits_v2,
            candidate_faces=candidate_faces,
            reconstruction=reconstruction,
            ridge_eave_scores=ridge_eave_scores,
            computed_overextend=computed_overextend,
            raw_disagreement=raw_disagreement,
            ceiling_replacement=ceiling_replacement,
        )
    elif is_tier_kind(parsed.kind):
        result = find_element(
            [],
            args.element_id,
            tier_payloads=tier_payloads,
        )
    elif is_reconstruction_kind(parsed.kind):
        reconstructions = json.loads(args.reconstructions.read_text())
        result = find_element(
            [],
            args.element_id,
            reconstructions=reconstructions,
            raw_ceiling_plane_splits=raw_ceiling_plane_splits,
            raw_ceiling_plane_splits_v2=raw_ceiling_plane_splits_v2,
            candidate_faces=candidate_faces,
            reconstruction=reconstruction,
            ridge_eave_scores=ridge_eave_scores,
            computed_overextend=computed_overextend,
            raw_disagreement=raw_disagreement,
            ceiling_replacement=ceiling_replacement,
        )
    elif is_ontology_kind(parsed.kind):
        result = find_element(
            [],
            args.element_id,
            roof_results=roof_results,
            raw_ceiling_plane_splits=raw_ceiling_plane_splits,
            raw_ceiling_plane_splits_v2=raw_ceiling_plane_splits_v2,
            candidate_faces=candidate_faces,
            reconstruction=reconstruction,
            ridge_eave_scores=ridge_eave_scores,
            computed_overextend=computed_overextend,
            raw_disagreement=raw_disagreement,
            ceiling_replacement=ceiling_replacement,
        )
    elif is_v3_kind(parsed.kind):
        v3_results = json.loads(args.v3_results.read_text())
        result = find_element(
            [],
            args.element_id,
            v3_results=v3_results,
            raw_ceiling_plane_splits=raw_ceiling_plane_splits,
            raw_ceiling_plane_splits_v2=raw_ceiling_plane_splits_v2,
            candidate_faces=candidate_faces,
            reconstruction=reconstruction,
            ridge_eave_scores=ridge_eave_scores,
            computed_overextend=computed_overextend,
            raw_disagreement=raw_disagreement,
            ceiling_replacement=ceiling_replacement,
        )
    elif parsed.kind in sidecar_only_kinds:
        result = find_element(
            [],
            args.element_id,
            roof_results=roof_results,
            raw_ceiling_plane_splits=raw_ceiling_plane_splits,
            raw_ceiling_plane_splits_v2=raw_ceiling_plane_splits_v2,
            candidate_faces=candidate_faces,
            reconstruction=reconstruction,
            ridge_eave_scores=ridge_eave_scores,
            computed_overextend=computed_overextend,
            raw_disagreement=raw_disagreement,
            ceiling_replacement=ceiling_replacement,
        )
    else:
        buildings = json.loads(args.buildings_json.read_text())
        result = find_element(
            buildings,
            args.element_id,
            roof_results=roof_results,
            raw_ceiling_plane_splits=raw_ceiling_plane_splits,
            raw_ceiling_plane_splits_v2=raw_ceiling_plane_splits_v2,
            candidate_faces=candidate_faces,
            reconstruction=reconstruction,
            ridge_eave_scores=ridge_eave_scores,
            computed_overextend=computed_overextend,
            raw_disagreement=raw_disagreement,
            ceiling_replacement=ceiling_replacement,
        )

    if args.trace and is_ontology_kind(parsed.kind):
        result = build_trace(result)

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
