"""Building-level merge across per-room polyhedra (plan §3 Increment 7).

The per-room flow (``repair_room``) produces one watertight polyhedron per
room. A multi-story building is therefore a stack of separate room boxes:

- Adjacent rooms within a story share a wall — both rooms emit it as a
  full-height tile, so the wall appears TWICE in the building.
- Story N's ceiling coincides with Story N+1's floor — both tiles are
  emitted at the same Y, overlapping in XZ.

For visualization we want only the building's exterior envelope: interior
shared walls and storey-boundary slab pairs must be hidden. This module
runs ``repair_room`` on every room, classifies each face as EXTERIOR or
one of two INTERIOR variants, and emits an aggregated building-level
envelope candidate carrying only exterior faces.

Per plan §3 step 3, the default is *keep both* (no FACE_COLLAPSE). The
interior labels live alongside the faces so a downstream consumer that
wants both rooms' walls — e.g. a thermal sim that needs the inner
boundary — can opt back in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.polyhedron.manifold_repair import (
    RoomRepairResult,
    collect_room_tiles,
    envelope_candidate_from_repair,
    repair_room,
)

__all__ = [
    "BuildingRepairResult",
    "FaceClassification",
    "envelope_candidate_from_building",
    "repair_building",
]


@dataclass(frozen=True, slots=True)
class FaceClassification:
    """One face's identity + classification within the building.

    Kinds:

    - ``exterior``: face is part of the building envelope.
    - ``interior_shared_wall``: two rooms point opposite-normal walls at
      each other across a shared interior wall.
    - ``interior_storey_boundary``: lower room's ceiling and upper
      room's floor meet at the same Y plane with opposite normals.
    - ``duplicate``: two rooms both emit the SAME surface (same plane,
      same outward normal) because ``collect_room_tiles`` assigned the
      same payload tile to multiple rooms. Keep one, drop the other in
      the default envelope.
    """

    room_index: int
    face_id: int
    corners: tuple[tuple[float, float, float], ...]
    plane: Plane
    source: str
    locator_id: str
    story: int | None
    kind: str  # "exterior" | "interior_shared_wall" | "interior_storey_boundary" | "duplicate"
    interior_partner: tuple[int, int] | None = None  # (other_room_index, other_face_id)


@dataclass(slots=True)
class BuildingRepairResult:
    """Aggregated building-level result.

    ``room_results`` carries every per-room ``RoomRepairResult`` in
    iteration order. ``faces`` is the classified face list spanning all
    rooms; ``exterior_faces`` is the convenience filter the viewer
    consumes by default.
    """

    rooms: list[RoomRepairResult]
    faces: list[FaceClassification] = field(default_factory=list)

    @property
    def exterior_faces(self) -> list[FaceClassification]:
        return [f for f in self.faces if f.kind == "exterior"]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def repair_building(
    payload: Mapping[str, Any],
    *,
    corner_tol: float = 0.02,
    coord_tol: float = 1e-3,
    plane_normal_tol: float = 0.02,
    plane_offset_tol_m: float = 0.05,
    interior_overlap_ratio_min: float = 0.50,
    duplicate_overlap_ratio_min: float = 0.85,
    storey_boundary_y_tol_m: float = 1.0,
) -> BuildingRepairResult:
    """Run ``repair_room`` for every room, then classify faces as
    EXTERIOR / INTERIOR_SHARED_WALL / INTERIOR_STOREY_BOUNDARY.

    A pair of faces (face_a in room_a, face_b in room_b) is interior when:

    1. ``face_a.plane`` and ``face_b.plane`` have aligned-or-opposite
       normals (within ``plane_normal_tol``) AND matching offsets
       (within ``plane_offset_tol_m`` after normalising for normal
       direction).
    2. The two faces' XZ-or-plane-2D projections overlap by at least
       ``overlap_ratio_min`` of the smaller face's area.

    Interior types are further distinguished by the plane orientation:

    - Near-horizontal plane (|b| > 0.85): ``interior_storey_boundary``.
    - Otherwise: ``interior_shared_wall``.
    """
    rooms_data = list(payload.get("rooms") or [])
    rooms: list[RoomRepairResult] = []
    candidates: list[Any] = []
    for room in rooms_data:
        repair = repair_room(
            payload, room, corner_tol=corner_tol, coord_tol=coord_tol
        )
        envelope = envelope_candidate_from_repair(repair)
        rooms.append(repair)
        candidates.append(envelope)

    # Visualization fidelity beats topology: we emit tier_payload tiles
    # verbatim (via ``collect_room_tiles``) PLUS any fillers that the
    # per-room repair added to close genuine holes. Bypassing the
    # half-edge build's tile pool prevents visual_shells / gable_closures
    # from being silently dropped due to ``duplicate_consecutive_corners``
    # or ``duplicate_directed_edge_both_windings`` rejection during build.
    faces: list[FaceClassification] = []
    raw_index: list[tuple[int, int, dict]] = []
    for room_index, room in enumerate(rooms_data):
        story = room.get("story")
        for tile in collect_room_tiles(payload, room, corner_tol=corner_tol):
            raw_index.append(
                (
                    room_index,
                    tile.face_id,
                    {
                        "corners": tuple(tile.corners),
                        "plane": tile.plane,
                        "source": tile.source,
                        "locator_id": tile.locator_id,
                        "story": story,
                    },
                )
            )
        # Add fillers from per-room repair so genuine geometric gaps still
        # get covered when the user wants closed envelopes downstream.
        envelope = candidates[room_index]
        if envelope is None:
            continue
        for face in envelope.faces:
            if not face.source.startswith("polyhedron_v3_filler"):
                continue
            raw_index.append(
                (
                    room_index,
                    _face_id_from_locator(face.locator_id),
                    {
                        "corners": tuple(
                            (float(c[0]), float(c[1]), float(c[2]))
                            for c in face.corners
                        ),
                        "plane": face.plane,
                        "source": face.source,
                        "locator_id": face.locator_id,
                        "story": face.story,
                    },
                )
            )

    classifications = ["exterior"] * len(raw_index)
    partners: list[tuple[int, int] | None] = [None] * len(raw_index)

    # Bucket faces by plane signature so we don't compare O(N^2) candidates
    # across the whole building.
    buckets: dict[tuple[int, int, int, int], list[int]] = {}
    for idx, (_ri, _fid, info) in enumerate(raw_index):
        plane = info["plane"]
        signature = _plane_signature(plane)
        if signature is None:
            continue
        buckets.setdefault(signature, []).append(idx)

    room_centroids = _room_centroids(candidates)

    # Pass 1: strict plane-support match. Catches shared walls (rooms on
    # opposite sides of a single wall plane), aligned-normal duplicates,
    # storey boundaries that happen to share a single plane, AND
    # same-room filler-vs-tile coincidences (filler picked
    # neighbor-plane extension → coplanar with the tile it extends; for
    # the building envelope the filler is redundant).
    for bucket_indices in buckets.values():
        if len(bucket_indices) < 2:
            continue
        for ai_pos in range(len(bucket_indices)):
            i = bucket_indices[ai_pos]
            if classifications[i] != "exterior":
                continue
            ri_a, _fid_a, info_a = raw_index[i]
            for j in bucket_indices[ai_pos + 1 :]:
                if classifications[j] != "exterior":
                    continue
                ri_b, _fid_b, info_b = raw_index[j]
                is_same_room = ri_a == ri_b
                a_is_filler = info_a["source"].startswith(
                    "polyhedron_v3_filler"
                )
                b_is_filler = info_b["source"].startswith(
                    "polyhedron_v3_filler"
                )
                if is_same_room and not (a_is_filler or b_is_filler):
                    continue  # tier_payload may legitimately emit
                    # multiple tiles per surface within one room
                if not _planes_share_support(
                    info_a["plane"],
                    info_b["plane"],
                    normal_tol=plane_normal_tol,
                    offset_tol_m=plane_offset_tol_m,
                ):
                    continue
                # Use TWO overlap metrics:
                # - interior pair (shared wall, storey boundary) needs only
                #   ``interior_overlap_ratio_min`` of the smaller face — the
                #   two surfaces are conceptually the same boundary but the
                #   two scans of it (one from each room) often only line up
                #   partially.
                # - duplicate (same surface emitted twice) needs the candidate
                #   to be nearly entirely inside its partner (``duplicate_overlap_
                #   ratio_min`` of EACH face). Without this, two genuinely
                #   distinct tiles that happen to share a plane and touch at
                #   a corner can be wrongly classified as duplicates,
                #   leaving 2D gaps in the building envelope.
                inter_area, area_a, area_b = _face_overlap_areas(
                    info_a["corners"], info_b["corners"], info_a["plane"]
                )
                if inter_area <= 0 or area_a <= 0 or area_b <= 0:
                    continue
                overlap_smaller = inter_area / min(area_a, area_b)
                overlap_each_min = inter_area / max(area_a, area_b)
                if is_same_room:
                    if overlap_each_min < duplicate_overlap_ratio_min:
                        continue
                    if a_is_filler:
                        classifications[i] = "duplicate"
                        partners[i] = (ri_b, raw_index[j][1])
                    else:
                        classifications[j] = "duplicate"
                        partners[j] = (ri_a, raw_index[i][1])
                    break
                kind = _classify_pair_by_source(
                    info_a["source"],
                    info_b["source"],
                    info_a["plane"],
                    room_centroids.get(ri_a),
                    room_centroids.get(ri_b),
                )
                if kind == "exterior":
                    continue
                if kind == "duplicate":
                    # Cross-room duplicate: both tiles must cover essentially
                    # the same surface to drop one without creating a gap.
                    if overlap_each_min < duplicate_overlap_ratio_min:
                        continue
                    classifications[j] = "duplicate"
                    partners[j] = (ri_a, raw_index[i][1])
                else:
                    if overlap_smaller < interior_overlap_ratio_min:
                        continue
                    classifications[i] = kind
                    classifications[j] = kind
                    partners[i] = (ri_b, raw_index[j][1])
                    partners[j] = (ri_a, raw_index[i][1])
                break

    # Pass 2: loose-Y storey-boundary match. tier_payload's scans of the
    # same storey-boundary surface from below (lower room's ceiling) and
    # above (upper room's floor) usually drift 20–40 cm in Y, so they
    # don't share a single supporting plane. Pair them anyway when the
    # XZ extents overlap and Y values are within ``storey_boundary_y_tol_m``.
    floor_indices = [
        i for i, (_ri, _fid, info) in enumerate(raw_index)
        if _is_horizontal_plane(info["plane"])
        and _base_source(info["source"]) == "floor"
        and classifications[i] == "exterior"
    ]
    ceiling_indices = [
        i for i, (_ri, _fid, info) in enumerate(raw_index)
        if _is_horizontal_plane(info["plane"])
        and _base_source(info["source"]) in ("ceiling", "visual_shell", "gable_closure")
        and classifications[i] == "exterior"
    ]
    for ci in ceiling_indices:
        if classifications[ci] != "exterior":
            continue
        ri_c, _fid_c, info_c = raw_index[ci]
        ycen_c = _face_y_centroid(info_c["corners"])
        for fi in floor_indices:
            if classifications[fi] != "exterior":
                continue
            ri_f, _fid_f, info_f = raw_index[fi]
            if ri_c == ri_f:
                continue
            ycen_f = _face_y_centroid(info_f["corners"])
            if abs(ycen_f - ycen_c) > storey_boundary_y_tol_m:
                continue
            # XZ overlap check: use the ceiling's plane (close enough to
            # horizontal that the 2D projection is the XZ plane regardless).
            inter_area, ca, fa = _face_overlap_areas(
                info_c["corners"], info_f["corners"], info_c["plane"]
            )
            if inter_area <= 0 or min(ca, fa) <= 0:
                continue
            if inter_area / min(ca, fa) < interior_overlap_ratio_min:
                continue
            classifications[ci] = "interior_storey_boundary"
            classifications[fi] = "interior_storey_boundary"
            partners[ci] = (ri_f, raw_index[fi][1])
            partners[fi] = (ri_c, raw_index[ci][1])
            break

    for idx, (room_index, face_id, info) in enumerate(raw_index):
        faces.append(
            FaceClassification(
                room_index=room_index,
                face_id=face_id,
                corners=info["corners"],
                plane=info["plane"],
                source=info["source"],
                locator_id=info["locator_id"],
                story=info["story"],
                kind=classifications[idx],
                interior_partner=partners[idx],
            )
        )

    return BuildingRepairResult(rooms=rooms, faces=faces)


def envelope_candidate_from_building(
    building: BuildingRepairResult,
    *,
    include_interior: bool = False,
) -> object | None:
    """Aggregate the building's exterior faces into one EnvelopeCandidate.

    When ``include_interior`` is True, every face (exterior + interior)
    is emitted with its kind tag in ``source`` so downstream consumers
    can filter. Default behaviour drops interior faces so the viewer
    shows only the building envelope.
    """
    from reconcile_tiers.polyhedron import payload_adapter as pa

    selected = (
        building.faces if include_interior else building.exterior_faces
    )
    if not selected:
        return None

    out_faces: list[pa.PayloadFace] = []
    for f in selected:
        source = f.source
        if include_interior and f.kind != "exterior":
            source = f"{f.source}::{f.kind}"
        out_faces.append(
            pa.PayloadFace(
                kind=_source_to_kind(f.source),
                locator_id=f.locator_id,
                corners=[list(c) for c in f.corners],
                plane=f.plane,
                source=source,
                room_index=f.room_index,
                story=f.story,
            )
        )

    return pa.EnvelopeCandidate(
        locator_id="building-envelope",
        faces=out_faces,
        footprint_area_m2=0.0,
        top_source="manifold_repair_building",
        top_overlap_ratio=1.0,
        selector="manifold_repair_building",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _face_id_from_locator(locator: str) -> int:
    """Extract a stable integer id from the face's locator string.

    Falls back to hash on unparseable input; the value just needs to be
    deterministic within a single ``repair_building`` invocation.
    """
    if "::filler::" in locator:
        try:
            return int(locator.rsplit("::filler::", 1)[1])
        except ValueError:
            pass
    if ":face:" in locator:
        try:
            return int(locator.rsplit(":face:", 1)[1])
        except ValueError:
            pass
    return hash(locator) & 0xFFFFFFFF


def _plane_signature(
    plane: Plane,
) -> tuple[int, int, int, int] | None:
    """Quantize a plane to a hashable key for cheap bucketing.

    Two planes with the same normal direction (or anti-parallel — we
    take the dominant-component-positive convention) and the same offset
    within a bucket size land in the same bucket. We then run the more
    precise ``_planes_match`` only on bucket members.
    """
    n = np.array([plane.a, plane.b, plane.c], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return None
    n /= norm
    d = float(plane.d) / norm
    dom = int(np.argmax(np.abs(n)))
    if n[dom] < 0:
        n = -n
        d = -d
    NORMAL_BUCKET = 0.02  # ~1.1° tolerance
    OFFSET_BUCKET_M = 0.05
    return (
        int(round(n[0] / NORMAL_BUCKET)),
        int(round(n[1] / NORMAL_BUCKET)),
        int(round(n[2] / NORMAL_BUCKET)),
        int(round(d / OFFSET_BUCKET_M)),
    )


def _planes_share_support(
    a: Plane,
    b: Plane,
    *,
    normal_tol: float,
    offset_tol_m: float,
) -> bool:
    """True when two planes describe the same supporting surface (aligned
    or anti-parallel normals, matching offset)."""
    na = np.array([a.a, a.b, a.c], dtype=float)
    nb = np.array([b.a, b.b, b.c], dtype=float)
    na_norm = float(np.linalg.norm(na))
    nb_norm = float(np.linalg.norm(nb))
    if na_norm <= 1e-12 or nb_norm <= 1e-12:
        return False
    na /= na_norm
    nb /= nb_norm
    da = float(a.d) / na_norm
    db = float(b.d) / nb_norm
    cos = float(na @ nb)
    if abs(abs(cos) - 1.0) > normal_tol:
        return False
    db_compare = db if cos > 0 else -db
    return abs(da - db_compare) <= offset_tol_m


def _classify_pair_by_source(
    source_a: str,
    source_b: str,
    plane: Plane,
    centroid_a: np.ndarray | None,
    centroid_b: np.ndarray | None,
) -> str:
    """Use semantic source labels (floor / wall / ceiling / …) to decide
    whether two faces on the same supporting plane represent an interior
    pair or a duplicate emission. tier_payload's plane equations are
    unreliable (often ``a=b=c=d=0``) so we cannot rely on the fitted
    normal direction.
    """
    base_a = _base_source(source_a)
    base_b = _base_source(source_b)
    pair = frozenset({base_a, base_b})

    floor_kinds = {"floor"}
    ceiling_kinds = {"ceiling", "visual_shell", "gable_closure"}

    if (base_a in floor_kinds and base_b in ceiling_kinds) or (
        base_b in floor_kinds and base_a in ceiling_kinds
    ):
        return "interior_storey_boundary"

    if base_a in ceiling_kinds and base_b in ceiling_kinds:
        return "duplicate"
    if base_a in floor_kinds and base_b in floor_kinds:
        return "duplicate"

    if pair == {"wall"}:
        # Distinguish shared interior wall (rooms on opposite sides) from
        # duplicate emission (rooms on the same side) using their floor
        # centroids relative to the wall plane.
        if centroid_a is None or centroid_b is None:
            return "interior_shared_wall"
        n = np.array([plane.a, plane.b, plane.c], dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm <= 1e-12:
            return "interior_shared_wall"
        n /= n_norm
        d = float(plane.d) / n_norm
        side_a = float(n @ centroid_a) - d
        side_b = float(n @ centroid_b) - d
        if abs(side_a) < 0.05 or abs(side_b) < 0.05:
            return "interior_shared_wall"
        if side_a * side_b < 0:
            return "interior_shared_wall"
        return "duplicate"

    return "exterior"


def _base_source(source: str) -> str:
    """Strip the optional ``polyhedron_v3_filler:`` prefix so a filler
    inherits the classification of the surface it extends."""
    if source.startswith("polyhedron_v3_filler:"):
        return source.split(":", 1)[1]
    if source == "polyhedron_v3_filler":
        return "ceiling"
    return source


def _is_horizontal_plane(plane: Plane) -> bool:
    n = np.array([plane.a, plane.b, plane.c], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return False
    return abs(float(n[1] / norm)) > 0.85


def _face_y_centroid(
    corners: tuple[tuple[float, float, float], ...],
) -> float:
    return sum(c[1] for c in corners) / len(corners)


def _room_centroids(candidates: list) -> dict[int, np.ndarray]:
    """Approximate each room's interior point as the centroid of its
    floor corners (averaged in 3D). Used by the wall-pair shared/duplicate
    classifier."""
    out: dict[int, np.ndarray] = {}
    for room_index, envelope in enumerate(candidates):
        if envelope is None:
            continue
        floor_pts: list[tuple[float, float, float]] = []
        for face in envelope.faces:
            if _base_source(face.source) != "floor":
                continue
            for c in face.corners:
                floor_pts.append((float(c[0]), float(c[1]), float(c[2])))
        if not floor_pts:
            continue
        arr = np.array(floor_pts, dtype=float)
        out[room_index] = arr.mean(axis=0)
    return out


def _face_overlap_areas(
    corners_a: tuple[tuple[float, float, float], ...],
    corners_b: tuple[tuple[float, float, float], ...],
    plane: Plane,
) -> tuple[float, float, float]:
    """Return ``(intersection_area, area_a, area_b)`` for two coplanar
    polygons projected onto the given plane's 2D frame. Callers pick
    the overlap metric (intersection / smaller for permissive interior
    pairing; intersection / larger for strict duplicate detection)."""
    try:
        from shapely.geometry import Polygon as ShPoly
    except ImportError:
        return 0.0, 0.0, 0.0

    u, v, origin = _plane_2d_frame(plane)
    if u is None:
        return 0.0, 0.0, 0.0

    def to_2d(pt):
        arr = np.asarray(pt, dtype=float) - origin
        return (float(arr @ u), float(arr @ v))

    poly_a = ShPoly([to_2d(c) for c in corners_a])
    poly_b = ShPoly([to_2d(c) for c in corners_b])
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0, float(poly_a.area or 0.0), float(poly_b.area or 0.0)
    inter = poly_a.intersection(poly_b)
    inter_area = 0.0 if inter.is_empty else float(inter.area)
    return inter_area, float(poly_a.area), float(poly_b.area)


def _plane_2d_frame(plane: Plane):
    """Return (u, v, origin) — an orthonormal 2D basis on the plane."""
    n = np.array([plane.a, plane.b, plane.c], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return None, None, None
    n /= norm
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(n @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    origin = n * (float(plane.d) / 1.0)
    return u, v, origin


def _source_to_kind(source: str):
    if source == "floor":
        return "floor"
    if source == "wall":
        return "wall"
    return "ceiling"
