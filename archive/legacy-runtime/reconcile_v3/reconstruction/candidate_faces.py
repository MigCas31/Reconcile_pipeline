"""Phase A -- candidate face generation for the reconstruction BIP.

Given one building's merged roof-segment hypotheses plus the scan-derived
footprint, produce a set of **candidate clipped faces** that the
downstream binary integer program (``solver.py``) selects among.

Each candidate face carries:

* the parent plane equation (a, b, c, d),
* a 2-D XZ polygon -- the footprint the face occupies if selected,
* azimuth / inclination / area for objective terms,
* a neighbour list (candidate face IDs that share a ridge) for the
  topology constraint,
* a support weight (scan evidence -- currently parent-segment area),
* an ``extended`` flag (``True`` when the candidate was extended beyond
  the raw merged segment up to the ridge against an opposing plane).

This v1 emits **one candidate per merged segment**, extended when the
upstream V3 pipeline identified opposing planes, otherwise returned
unextended. That keeps Phase A linear in the number of merged segments
-- critical for buildings with 2000+ segments. Phase B can relax this
later (e.g. emit both an "extended" and an "unextended" variant and let
the BIP pick).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import atan2, degrees, sqrt

from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from ..audit import note_geom_skip

Plane = tuple[float, float, float, float]  # ax + by + cz + d = 0, (a,b,c) unit
Point2D = tuple[float, float]

_RIDGE_HALFPLANE_NORMAL_MIN = 1e-6
_MIN_FACE_AREA_M2 = 0.1
_ZONE_RUN_MERGE_TOL_M = 0.12
_EXTENDED_AREA_RATIO_MIN = 1.01
_FALLBACK_PIECE_MIN_AREA_M2 = 0.15


@dataclass
class CandidateFace:
    id: str
    building_uuid: str
    parent_segment_id: str
    plane: Plane
    footprint_xz: list[Point2D]
    area_m2: float
    azimuth_deg: float
    inclination_deg: float
    neighbors: list[str] = field(default_factory=list)
    support_m2: float = 0.0
    extended: bool = False
    gbm_prior: float | None = None
    zone_id: str | None = None
    part_id: str | None = None
    zone_source: str | None = None
    zone_room_ids: list[str] = field(default_factory=list)
    zone_confidence: float | None = None
    zone_seed_subpart_ids: list[str] = field(default_factory=list)
    zone_seed_hypothesis_ids: list[str] = field(default_factory=list)
    zone_semantic_kinds: list[str] = field(default_factory=list)
    zone_avg_azimuths_deg: list[float] = field(default_factory=list)
    zone_fallback_kind: str | None = None
    source_gap_token: str | None = None
    source_gap_major_axis_azimuth_deg: float | None = None


def _plane_azimuth_inclination(plane: Plane) -> tuple[float, float]:
    """Downslope azimuth (0=+Z, clockwise) and inclination (degrees)."""
    a, b, c, _ = plane
    norm = sqrt(a * a + b * b + c * c)
    if norm < 1e-9:
        return 0.0, 0.0
    a, b, c = a / norm, b / norm, c / norm
    if b < 0:
        a, b, c = -a, -b, -c
    horiz = sqrt(a * a + c * c)
    incl = degrees(atan2(horiz, b))
    azimuth = degrees(atan2(-a, -c)) % 360.0
    return azimuth, incl


def _to_xz_ring(footprint_xz: list) -> list[Point2D]:
    """Accept either [x, y, z] triples or [x, z] pairs; return [x, z] pairs."""
    out: list[Point2D] = []
    for pt in footprint_xz:
        if len(pt) >= 3:
            out.append((float(pt[0]), float(pt[2])))
        elif len(pt) == 2:
            out.append((float(pt[0]), float(pt[1])))
    return out


def _poly_from_ring(ring: list[Point2D]) -> ShapelyPolygon | None:
    if len(ring) < 3:
        return None
    try:
        poly = ShapelyPolygon(ring)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _exterior_ring(poly: ShapelyPolygon) -> list[Point2D]:
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(float(x), float(z)) for (x, z) in coords]


def _expand_polygons(geom) -> list[ShapelyPolygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, ShapelyPolygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [
            poly
            for poly in geom.geoms
            if isinstance(poly, ShapelyPolygon) and not poly.is_empty
        ]
    out: list[ShapelyPolygon] = []
    for poly in getattr(geom, "geoms", []):
        if isinstance(poly, ShapelyPolygon) and not poly.is_empty:
            out.append(poly)
    return out


def _zone_slug(zone_id: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", zone_id).strip("_")
    return (token or "zone")[-40:]


def _canonical_slug(cluster_canonical_id: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", cluster_canonical_id).strip("_")
    return (token or "canonical")[-40:]


def _gap_token_from_segment_id(segment_id: str) -> str | None:
    match = re.search(r":gap-([^:]+):", segment_id)
    if not match:
        return None
    return str(match.group(1) or "") or None


def _azimuth_diff_deg(a: float, b: float) -> float:
    raw = abs(float(a) - float(b)) % 360.0
    return min(raw, 360.0 - raw)


def _axis_metrics_from_ring(
    ring: list[Point2D],
) -> tuple[float | None, float | None]:
    poly = _poly_from_ring(ring)
    if poly is None or poly.area <= 0.0:
        return None, None
    try:
        rect = poly.minimum_rotated_rectangle
    except Exception:
        return None, None
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return None, None
    best_edge: tuple[float, float] | None = None
    shortest = None
    for idx in range(len(coords) - 1):
        x0, z0 = coords[idx]
        x1, z1 = coords[idx + 1]
        dx = float(x1) - float(x0)
        dz = float(z1) - float(z0)
        length = sqrt(dx * dx + dz * dz)
        if length <= 1e-6:
            continue
        azimuth = degrees(atan2(dx, dz)) % 360.0
        if best_edge is None or length > best_edge[0]:
            best_edge = (length, azimuth)
        if shortest is None or length < shortest:
            shortest = length
    if best_edge is None or shortest is None:
        return None, None
    return best_edge[1], max(best_edge[0] / max(shortest, 1e-6), 1.0)


def _ridge_halfplane_coeffs(
    plane_self: Plane, plane_other: Plane
) -> tuple[float, float, float] | None:
    """Coefficients (A, C, D) of the seam line ``A*x + C*z + D = 0``
    between ``plane_self`` and ``plane_other``.

    Derivation: ``y_i = -(a_i*x + c_i*z + d_i) / b_i`` (roof-up convention
    ``b_i > 0``). The seam is where ``y_self = y_other``; the half-plane
    ``A*x + C*z + D >= 0`` corresponds to ``y_self <= y_other`` with

        (a_s*b_o - a_o*b_s)*x + (c_s*b_o - c_o*b_s)*z + (d_s*b_o - d_o*b_s).

    The caller picks which *side* of this seam the candidate belongs on
    (see ``_extend_against_opposing`` -- we orient via the segment's own
    centroid so extension always grows into the segment's own half).

    Returns ``None`` if either plane is near-vertical (``|b| < eps``) or
    the coefficients degenerate to zero (parallel planes).
    """
    a_s, b_s, c_s, d_s = plane_self
    a_o, b_o, c_o, d_o = plane_other
    if abs(b_s) < 1e-6 or abs(b_o) < 1e-6:
        return None
    # Orient so both planes face up (b > 0) before forming the half-plane.
    if b_s < 0:
        a_s, b_s, c_s, d_s = -a_s, -b_s, -c_s, -d_s
    if b_o < 0:
        a_o, b_o, c_o, d_o = -a_o, -b_o, -c_o, -d_o
    A = a_s * b_o - a_o * b_s
    C = c_s * b_o - c_o * b_s
    D = d_s * b_o - d_o * b_s
    if sqrt(A * A + C * C) < _RIDGE_HALFPLANE_NORMAL_MIN:
        return None
    return A, C, D


def _clip_by_halfplane(
    poly: ShapelyPolygon, coeffs: tuple[float, float, float]
) -> ShapelyPolygon | None:
    """Intersect ``poly`` with the half-plane ``A*x + C*z + D >= 0``.

    Builds a large rectangle on the half-plane's positive side and takes
    the intersection -- robust for any convex/concave input.
    """
    A, C, D = coeffs
    minx, minz, maxx, maxz = poly.bounds
    pad = max(maxx - minx, maxz - minz, 10.0) * 10.0 + 10.0
    cx = (minx + maxx) / 2.0
    cz = (minz + maxz) / 2.0
    # Unit vector pointing into the positive side of the half-plane.
    norm = sqrt(A * A + C * C)
    ux, uz = A / norm, C / norm
    # Offset the rectangle centre along +u by ``pad`` so the whole
    # rectangle lies on the positive side of A*x + C*z + D = 0.
    # The boundary line passes through the point closest to the origin:
    # (-A*D / (A^2+C^2), -C*D / (A^2+C^2)).
    t0 = -D / (norm * norm)  # signed distance from origin to the line
    # Anchor the rectangle half-plane side starting at the boundary line
    # through poly's centroid projected onto the line, extended ``pad``
    # units into +u.
    base_x = cx - (A * cx + C * cz + D) * A / (norm * norm)
    base_z = cz - (A * cx + C * cz + D) * C / (norm * norm)
    _ = t0  # silence lint
    # Perpendicular direction (tangent to the boundary).
    tx, tz = -uz, ux
    # Build a rectangle: from (base - pad*t) to (base + pad*t) on the
    # boundary line, extruded ``2*pad`` into +u.
    corners = [
        (base_x - pad * tx, base_z - pad * tz),
        (base_x + pad * tx, base_z + pad * tz),
        (base_x + pad * tx + 2 * pad * ux, base_z + pad * tz + 2 * pad * uz),
        (base_x - pad * tx + 2 * pad * ux, base_z - pad * tz + 2 * pad * uz),
    ]
    try:
        rect = ShapelyPolygon(corners)
        clipped = poly.intersection(rect)
    except Exception:
        return None
    if clipped.is_empty or not clipped.is_valid:
        return None
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda g: g.area)
    if clipped.geom_type != "Polygon" or clipped.area <= 0:
        return None
    return clipped


def _extend_against_opposing(
    plane_self: Plane,
    seg_poly: ShapelyPolygon,
    scan_footprint_poly: ShapelyPolygon | None,
    opposing_planes: list[Plane],
) -> tuple[ShapelyPolygon | None, bool]:
    """Return ``(clipped_poly, extended)``.

    If the segment has no opposing planes we cannot responsibly extend it
    -- there is no ridge geometry to bound the extension -- so we keep the
    segment's own footprint (clipped to the scan footprint if provided)
    and return ``extended=False``. With opposing planes, start from the
    scan footprint (or segment footprint as fallback), and for each
    opposing plane clip to the half of the seam that the segment's own
    centroid already lives on -- so extension always grows the segment
    *within* its own side of the ridge rather than flipping it across.

    This sidedness-by-centroid avoids depending on upstream conventions:
    ``merged_slanted_roof_proposals.py`` emits segments on BOTH sides of
    each opposing seam ("rain" and "covered"), and we have no direct way
    to know which from ``opposing_planes`` alone. The centroid lookup
    answers it from the data itself.
    """
    if not opposing_planes:
        if scan_footprint_poly is None or scan_footprint_poly.is_empty:
            return seg_poly, False
        try:
            clipped = seg_poly.intersection(scan_footprint_poly)
        except Exception:
            return seg_poly, False
        if clipped.is_empty or not clipped.is_valid:
            return None, False
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if clipped.geom_type != "Polygon" or clipped.area < _MIN_FACE_AREA_M2:
            return None, False
        return clipped, False

    if scan_footprint_poly is not None and not scan_footprint_poly.is_empty:
        work = scan_footprint_poly
    else:
        work = seg_poly

    # Use the segment's own centroid (not the scan footprint's) to decide
    # sidedness: scan_footprint spans the whole building, but each segment
    # lives on a specific side of each ridge.
    seg_centroid = seg_poly.centroid
    cx, cz = float(seg_centroid.x), float(seg_centroid.y)

    for plane_other in opposing_planes:
        coeffs = _ridge_halfplane_coeffs(plane_self, plane_other)
        if coeffs is None:
            continue
        A, C, D = coeffs
        # Pick the half-plane that the segment already sits on. Value > 0
        # means ``y_self < y_other`` at the centroid (segment is on the
        # "covered" side); value < 0 means the opposite (rain side).
        if A * cx + C * cz + D < 0.0:
            coeffs = (-A, -C, -D)
        clipped = _clip_by_halfplane(work, coeffs)
        if clipped is None:
            # Half-plane clipped to empty -- face would vanish.
            return None, False
        work = clipped

    if work.area < _MIN_FACE_AREA_M2:
        return None, False

    # Did we actually extend beyond the original segment footprint?
    extended = work.area > seg_poly.area * _EXTENDED_AREA_RATIO_MIN
    return work, extended


def _segment_plane(segment: dict) -> Plane | None:
    plane = segment.get("merged_plane")
    if not plane or len(plane) < 4:
        return None
    a, b, c, d = (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))
    if abs(a) + abs(b) + abs(c) < 1e-9:
        return None
    return a, b, c, d


def _consolidate_same_zone_canonical_slices(
    candidate_records: list[dict],
) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    passthrough: list[dict] = []
    for record in candidate_records:
        zone_id = record.get("zone_id")
        cluster_canonical_id = record.get("cluster_canonical_id")
        if not isinstance(zone_id, str) or not zone_id:
            passthrough.append(record)
            continue
        if not isinstance(cluster_canonical_id, str) or not cluster_canonical_id:
            passthrough.append(record)
            continue
        grouped.setdefault((zone_id, cluster_canonical_id), []).append(record)

    consolidated: list[dict] = list(passthrough)
    for (zone_id, cluster_canonical_id), records in grouped.items():
        if len(records) == 1:
            consolidated.extend(records)
            continue
        members_with_polys: list[tuple[dict, ShapelyPolygon]] = []
        for record in records:
            poly = _poly_from_ring(record["candidate"].footprint_xz)
            if poly is None or poly.area < _MIN_FACE_AREA_M2:
                continue
            members_with_polys.append((record, poly))
        if not members_with_polys:
            consolidated.extend(records)
            continue
        merged = unary_union([poly for _, poly in members_with_polys])
        try:
            merged = merged.buffer(_ZONE_RUN_MERGE_TOL_M).buffer(-_ZONE_RUN_MERGE_TOL_M)
        except Exception as exc:
            note_geom_skip(exc, "candidate_faces.zone_run_merge")
        pieces = sorted(
            _expand_polygons(merged),
            key=lambda poly: (-float(poly.area), poly.bounds),
        )
        if not pieces:
            consolidated.extend(records)
            continue
        for piece_index, piece in enumerate(pieces):
            piece_members: list[dict] = []
            support_sum = 0.0
            for record, poly in members_with_polys:
                try:
                    overlap_area = float(poly.intersection(piece).area)
                except Exception:
                    overlap_area = 0.0
                if overlap_area < _MIN_FACE_AREA_M2:
                    continue
                piece_members.append(record)
                support_sum += float(record["candidate"].support_m2 or 0.0)
            if not piece_members:
                continue
            template = max(
                piece_members,
                key=lambda member: float(member["candidate"].area_m2),
            )
            candidate = template["candidate"]
            ring = _exterior_ring(piece)
            if len(ring) < 3:
                continue
            consolidated_candidate = CandidateFace(
                id=(
                    f"{candidate.building_uuid}::candidate::{_canonical_slug(cluster_canonical_id)}"
                    f":zone-{_zone_slug(zone_id)}"
                    + (f":run-{piece_index}" if len(pieces) > 1 else "")
                ),
                building_uuid=candidate.building_uuid,
                parent_segment_id=candidate.parent_segment_id,
                plane=candidate.plane,
                footprint_xz=ring,
                area_m2=float(piece.area),
                azimuth_deg=candidate.azimuth_deg,
                inclination_deg=candidate.inclination_deg,
                neighbors=[],
                support_m2=min(float(piece.area), support_sum),
                extended=any(
                    bool(member["candidate"].extended) for member in piece_members
                ),
                gbm_prior=candidate.gbm_prior,
                zone_id=candidate.zone_id,
                part_id=candidate.part_id,
                zone_source=candidate.zone_source,
                zone_room_ids=list(candidate.zone_room_ids),
                zone_confidence=candidate.zone_confidence,
                zone_seed_subpart_ids=list(candidate.zone_seed_subpart_ids),
                zone_seed_hypothesis_ids=list(candidate.zone_seed_hypothesis_ids),
                zone_semantic_kinds=list(candidate.zone_semantic_kinds),
                zone_avg_azimuths_deg=list(candidate.zone_avg_azimuths_deg),
                zone_fallback_kind=candidate.zone_fallback_kind,
                source_gap_token=candidate.source_gap_token,
                source_gap_major_axis_azimuth_deg=candidate.source_gap_major_axis_azimuth_deg,
            )
            opposing_cluster_canonicals: set[str] = set()
            for member in piece_members:
                opposing_cluster_canonicals.update(
                    member.get("opposing_cluster_canonicals") or set()
                )
            consolidated.append(
                {
                    "candidate": consolidated_candidate,
                    "cluster_canonical_id": cluster_canonical_id,
                    "opposing_cluster_canonicals": opposing_cluster_canonicals,
                    "zone_id": zone_id,
                }
            )
    return consolidated


def build_candidate_faces(
    building_uuid: str,
    merged_segments: list[dict],
    building_footprint_xz: list[Point2D] | None = None,
    building_zones: list[dict] | None = None,
    building_gaps: list[dict] | None = None,
) -> list[CandidateFace]:
    """Produce one candidate face per merged segment, ridge-extended where
    opposing planes are available.

    Args:
        building_uuid: used as the prefix for candidate IDs.
        merged_segments: list of dicts with keys ``id``, ``merged_plane``,
            ``footprint_xz``, ``opposing_planes`` (optional, list of
            plane tuples), ``features`` (optional).
        building_footprint_xz: XZ-plane ring of the scan-derived
            footprint. If provided, candidates are clipped to it.
        building_zones: optional authoritative part/zone footprints.
            When present, every candidate is further split by overlapping
            zones and tagged with ``zone_id`` / ``part_id`` so downstream
            selection can stay local to the owning roof zone.
        building_gaps: optional V3 gap rows. Gap-derived segments carry the
            source gap's local axis so fallback zones can use discovered
            gap geometry rather than only whole-zone footprint shape.

    Returns:
        List of ``CandidateFace``.
    """
    scan_poly: ShapelyPolygon | None = None
    if building_footprint_xz:
        scan_poly = _poly_from_ring(list(building_footprint_xz))

    gap_axes_by_token: dict[str, float] = {}
    for gap in building_gaps or []:
        gap_id = str(gap.get("id") or "")
        if "::v3-gap::" not in gap_id:
            continue
        gap_token = gap_id.split("::v3-gap::", 1)[1]
        gap_ring = _to_xz_ring(gap.get("footprint_xz") or [])
        gap_axis_azimuth_deg, _gap_aspect_ratio = _axis_metrics_from_ring(gap_ring)
        if gap_axis_azimuth_deg is None:
            continue
        gap_axes_by_token[gap_token] = gap_axis_azimuth_deg

    prepared_zones: list[dict] = []
    for zone in building_zones or []:
        zone_id = str(zone.get("id") or "")
        if not zone_id:
            continue
        zone_ring = _to_xz_ring(zone.get("footprint_xz") or [])
        zone_poly = _poly_from_ring(zone_ring)
        if zone_poly is None or zone_poly.area < _MIN_FACE_AREA_M2:
            continue
        prepared_zones.append(
            {
                "id": zone_id,
                "part_id": str(zone.get("part_id") or zone_id),
                "source": str(zone.get("source") or "unknown"),
                "room_ids": [
                    str(room_id) for room_id in (zone.get("room_ids") or []) if room_id
                ],
                "confidence": (
                    float(zone.get("confidence"))
                    if zone.get("confidence") is not None
                    else None
                ),
                "seed_subpart_ids": [
                    str(value)
                    for value in (zone.get("seed_subpart_ids") or [])
                    if value
                ],
                "seed_hypothesis_ids": [
                    str(value)
                    for value in (zone.get("seed_hypothesis_ids") or [])
                    if value
                ],
                "semantic_kinds": [
                    str(value) for value in (zone.get("semantic_kinds") or []) if value
                ],
                "avg_azimuths_deg": [
                    float(value)
                    for value in (zone.get("avg_azimuths_deg") or [])
                    if value is not None
                ],
                "fallback_kind": (
                    str(zone.get("fallback_kind"))
                    if zone.get("fallback_kind") is not None
                    else None
                ),
                "poly": zone_poly,
            }
        )

    # Pre-compute each segment's shapely poly + plane + opposing planes.
    prepared: list[dict] = []
    for seg in merged_segments:
        plane = _segment_plane(seg)
        if plane is None:
            continue
        ring = _to_xz_ring(seg.get("footprint_xz") or [])
        seg_poly = _poly_from_ring(ring)
        if seg_poly is None or seg_poly.area < _MIN_FACE_AREA_M2:
            continue
        opp_raw = seg.get("opposing_planes") or []
        opposing: list[Plane] = []
        for p in opp_raw:
            if p and len(p) >= 4:
                opposing.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
        prepared.append(
            {
                "segment": seg,
                "plane": plane,
                "seg_poly": seg_poly,
                "opposing": opposing,
            }
        )

    candidate_records: list[dict] = []
    for entry in prepared:
        seg = entry["segment"]
        plane = entry["plane"]
        seg_poly = entry["seg_poly"]
        opposing = entry["opposing"]

        clipped_poly, extended = _extend_against_opposing(
            plane, seg_poly, scan_poly, opposing
        )
        if clipped_poly is None:
            # Fall back to the raw segment poly clipped to the scan footprint.
            if scan_poly is not None:
                try:
                    fb = seg_poly.intersection(scan_poly)
                except Exception:
                    fb = None
                if fb is not None and not fb.is_empty and fb.area >= _MIN_FACE_AREA_M2:
                    if fb.geom_type == "MultiPolygon":
                        fb = max(fb.geoms, key=lambda g: g.area)
                    if fb.geom_type == "Polygon":
                        clipped_poly = fb
                        extended = False
            if clipped_poly is None:
                clipped_poly = seg_poly
                extended = False

        azimuth, inclination = _plane_azimuth_inclination(plane)
        feat = seg.get("features") or {}
        base_support = float(feat.get("area_m2") or seg_poly.area)
        base_area = max(float(clipped_poly.area), 1e-9)
        base_id = f"{building_uuid}::candidate::{seg['id'].split('::')[-1]}"
        source_gap_token = _gap_token_from_segment_id(str(seg.get("id") or ""))
        source_gap_major_axis_azimuth_deg = (
            gap_axes_by_token.get(source_gap_token)
            if source_gap_token is not None
            else None
        )

        zone_pieces: list[tuple[ShapelyPolygon, dict | None, int]] = []
        had_zone_overlap = False
        if prepared_zones:
            for zone in prepared_zones:
                try:
                    overlap = clipped_poly.intersection(zone["poly"])
                except Exception as exc:
                    note_geom_skip(exc, "candidate_faces.zone_overlap")
                    continue
                for piece_index, piece in enumerate(_expand_polygons(overlap)):
                    if piece.is_empty or piece.area < _MIN_FACE_AREA_M2:
                        continue
                    had_zone_overlap = True
                    overlap_ratio = float(piece.area) / base_area
                    avg_azimuths = zone["avg_azimuths_deg"]
                    azimuth_incompatible = bool(avg_azimuths) and all(
                        _azimuth_diff_deg(azimuth, zone_azimuth) > 35.0
                        for zone_azimuth in avg_azimuths
                    )
                    if (
                        zone["fallback_kind"] is None
                        and float(piece.area) < _FALLBACK_PIECE_MIN_AREA_M2
                        and overlap_ratio < 0.10
                        and azimuth_incompatible
                    ):
                        continue
                    zone_pieces.append((piece, zone, piece_index))
        if not zone_pieces and not had_zone_overlap:
            zone_pieces.append((clipped_poly, None, 0))
        elif not zone_pieces:
            continue

        for piece, zone, piece_index in zone_pieces:
            ring = _exterior_ring(piece)
            if len(ring) < 3:
                continue
            if zone is None:
                cand_id = base_id
            else:
                cand_id = f"{base_id}:zone-{_zone_slug(zone['id'])}"
                if piece_index:
                    cand_id += f":piece-{piece_index}"
            support = base_support * (float(piece.area) / base_area)
            candidate = CandidateFace(
                id=cand_id,
                building_uuid=building_uuid,
                parent_segment_id=seg["id"],
                plane=plane,
                footprint_xz=ring,
                area_m2=float(piece.area),
                azimuth_deg=azimuth,
                inclination_deg=inclination,
                neighbors=[],
                support_m2=support,
                extended=extended,
                zone_id=None if zone is None else zone["id"],
                part_id=None if zone is None else zone["part_id"],
                zone_source=None if zone is None else zone["source"],
                zone_room_ids=[] if zone is None else list(zone["room_ids"]),
                zone_confidence=None if zone is None else zone["confidence"],
                zone_seed_subpart_ids=[]
                if zone is None
                else list(zone["seed_subpart_ids"]),
                zone_seed_hypothesis_ids=[]
                if zone is None
                else list(zone["seed_hypothesis_ids"]),
                zone_semantic_kinds=[]
                if zone is None
                else list(zone["semantic_kinds"]),
                zone_avg_azimuths_deg=[]
                if zone is None
                else list(zone["avg_azimuths_deg"]),
                zone_fallback_kind=None if zone is None else zone["fallback_kind"],
                source_gap_token=source_gap_token,
                source_gap_major_axis_azimuth_deg=source_gap_major_axis_azimuth_deg,
            )
            candidate_records.append(
                {
                    "candidate": candidate,
                    "cluster_canonical_id": seg.get("cluster_canonical_id"),
                    "opposing_cluster_canonicals": {
                        str(value)
                        for value in (seg.get("opposing_cluster_canonicals") or [])
                        if value
                    },
                    "zone_id": None if zone is None else zone["id"],
                }
            )

    candidate_records = _consolidate_same_zone_canonical_slices(candidate_records)
    candidates = [record["candidate"] for record in candidate_records]
    for record in candidate_records:
        neighbours_ids: list[str] = []
        for other in candidate_records:
            if other is record:
                continue
            if (
                other["cluster_canonical_id"]
                not in record["opposing_cluster_canonicals"]
            ):
                continue
            record_zone_id = record["zone_id"]
            other_zone_id = other["zone_id"]
            if (
                record_zone_id is not None
                and other_zone_id is not None
                and record_zone_id != other_zone_id
            ):
                continue
            other_id = other["candidate"].id
            if other_id not in neighbours_ids:
                neighbours_ids.append(other_id)
        record["candidate"].neighbors.extend(neighbours_ids)

    return candidates
