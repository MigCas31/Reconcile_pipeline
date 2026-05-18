"""Raw-ceiling-fragment gate used by `reconcile_tiers.build`.

Drops `RAW_FALLBACK` ceiling candidates that are noisy fragments
(`junk_triangle`), already explained by a higher-priority candidate
(`redundant`), or fail the v2 raw-quality score in flat rooms
(`low_quality`). See SKILL.md / tracking_progress.md for the cohort
calibration behind each threshold.
"""

from __future__ import annotations

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.build_internals.constants import (
    RAW_CEILING_JUNK_TRIANGLE_MAX_AREA_M2,
    RAW_CEILING_JUNK_TRIANGLE_MIN_YSPAN_M,
    RAW_CEILING_LOW_QUALITY_THRESHOLD,
    RAW_CEILING_PEAK_YSPAN_MIN_M,
    RAW_CEILING_REDUNDANT_COVERAGE_RATIO,
    RAW_CEILING_REDUNDANT_LEFTOVER_MAX_M2,
    RAW_CEILING_REDUNDANT_Y_TOL_M,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import CeilingSource
from reconcile_tiers.roof.roof import RoofModel

_RAW_GATE_HIGHER_PRIORITY_SOURCES = frozenset(
    {
        CeilingSource.FLAT_EMIT,
        CeilingSource.ROOF_ARRANGEMENT,
        CeilingSource.ROOF_ARRANGEMENT_ATTIC,
        CeilingSource.THERMAL_CAP,
    }
)


def _parse_raw_ceiling_locator(locator_id: str) -> tuple[int, int] | None:
    marker = "::tier-ceiling-raw::"
    if marker not in locator_id:
        return None
    suffix = locator_id.rsplit(marker, 1)[1]
    parts = suffix.split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _planes_match_over_xz(
    raw_plane: Plane,
    higher_plane: Plane,
    overlap_poly,
    *,
    y_tol_m: float = RAW_CEILING_REDUNDANT_Y_TOL_M,
) -> bool:
    if overlap_poly is None or overlap_poly.is_empty:
        return False
    samples: list[tuple[float, float]] = []
    try:
        point = overlap_poly.representative_point()
        samples.append((float(point.x), float(point.y)))
    except Exception:
        pass
    geoms = getattr(overlap_poly, "geoms", [overlap_poly])
    for geom in geoms:
        coords = list(getattr(getattr(geom, "exterior", None), "coords", []) or [])
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]
        samples.extend((float(x), float(z)) for x, z in coords[:8])
    if not samples:
        return False

    deltas: list[float] = []
    for x, z in samples:
        raw_y = raw_plane.y_at(x, z)
        higher_y = higher_plane.y_at(x, z)
        if raw_y is None or higher_y is None:
            continue
        deltas.append(abs(float(raw_y) - float(higher_y)))
    if not deltas:
        return False
    return max(deltas) <= y_tol_m


def _filter_noisy_raw_candidates(
    candidates: list,
    model: BuildingModel,
    roof: RoofModel,
    drops_sink: list | None = None,
) -> list:
    """Drop raw-ceiling candidates that are noisy fragments or already
    redundant with higher-priority candidates on the same story.

    Three rules, each with a physical justification (see threshold constants
    above for cohort calibration data):

    1. junk_triangle: 3-corner, sub-area, tilted plane = RoomPlan artifact.
    2. redundant: leftover after subtracting higher-priority story-mates is
       smaller than a meaningful sliver.
    3. low_quality: in FLAT-classified rooms, score_raw_plane() below the v2
       gate threshold means the plane fits poorly / floats / disagrees with
       the obliques.

    Sloped rooms keep raws ungated by rule 3 -- there the raw IS the ceiling
    source. Rule 1 still applies (a tilted triangle is junk regardless).

    `drops_sink` (when provided) receives one dict per dropped candidate for
    the audit log written next to tier_payload.json.
    """
    from shapely.geometry import Point as _Point
    from shapely.geometry import Polygon as _Poly
    from shapely.ops import unary_union as _union

    from reconcile_tiers._core.shapely2 import make_valid as _make_valid
    from reconcile_tiers.assemble.raw_quality import score_raw_plane

    rooms_by_index = {room.index: room for room in model.rooms}

    # Build XZ polygons of higher-priority candidates per story so we can
    # measure leftover area cheaply with one union per story.
    higher_entries_by_story: dict[int | None, list[tuple[object, Plane]]] = {}
    for cand in candidates:
        if cand.source not in _RAW_GATE_HIGHER_PRIORITY_SOURCES:
            continue
        if len(cand.corners) < 3:
            continue
        try:
            poly = _make_valid(
                _Poly([(float(c[0]), float(c[2])) for c in cand.corners])
            )
        except Exception:
            continue
        if poly is None or poly.is_empty or poly.area <= 1e-9:
            continue
        # `make_valid` occasionally returns a multi-piece result that GEOS
        # itself still flags invalid (precision artefact at near-tangent
        # boundaries). Including it poisons `unary_union` for the whole
        # story; drop it instead so the gate stays computable.
        if not poly.is_valid:
            continue
        higher_entries_by_story.setdefault(cand.story, []).append((poly, cand.plane))

    kept: list = []
    for cand in candidates:
        # Only gate RAW_FALLBACK. ATTIC_FLAT_LID is a synthesised attic lid
        # (build.py: is_top_gable_room and _is_horizontal); it's already
        # specifically vetted geometry, not raw scan noise.
        if cand.source != CeilingSource.RAW_FALLBACK:
            kept.append(cand)
            continue

        corners = cand.corners
        n = len(corners)
        if n < 3:
            kept.append(cand)
            continue
        ys = [c[1] for c in corners]
        y_span = max(ys) - min(ys)
        try:
            raw_poly = _make_valid(_Poly([(float(c[0]), float(c[2])) for c in corners]))
        except Exception:
            raw_poly = None
        area = raw_poly.area if raw_poly is not None and not raw_poly.is_empty else 0.0

        reason: str | None = None

        if (
            n == 3
            and area < RAW_CEILING_JUNK_TRIANGLE_MAX_AREA_M2
            and y_span > RAW_CEILING_JUNK_TRIANGLE_MIN_YSPAN_M
        ):
            reason = "junk_triangle"

        if reason is None and raw_poly is not None and area > 0:
            higher_entries = higher_entries_by_story.get(cand.story) or []
            compatible_overlaps = []
            for higher_poly, higher_plane in higher_entries:
                try:
                    overlap = higher_poly.intersection(raw_poly)
                except Exception:
                    continue
                if overlap.is_empty or overlap.area <= 1e-9:
                    continue
                if _planes_match_over_xz(cand.plane, higher_plane, overlap):
                    compatible_overlaps.append(overlap)
            if compatible_overlaps:
                try:
                    compatible_union = (
                        _union(compatible_overlaps)
                        if len(compatible_overlaps) > 1
                        else compatible_overlaps[0]
                    )
                    covered = compatible_union.intersection(raw_poly).area
                except Exception:
                    covered = 0.0
                leftover = area - covered
                coverage_ratio = covered / area if area > 0 else 0.0
                peak_uncovered = False
                if y_span > RAW_CEILING_PEAK_YSPAN_MIN_M:
                    max_y = max(ys)
                    peak_points = [
                        (float(c[0]), float(c[2]))
                        for c in corners
                        if max_y - float(c[1]) <= RAW_CEILING_REDUNDANT_Y_TOL_M
                    ]
                    if peak_points:
                        peak_uncovered = not all(
                            compatible_union.covers(_Point(x, z))
                            for x, z in peak_points
                        )
                if not peak_uncovered and (
                    leftover < RAW_CEILING_REDUNDANT_LEFTOVER_MAX_M2
                    or coverage_ratio >= RAW_CEILING_REDUNDANT_COVERAGE_RATIO
                ):
                    reason = "redundant"

        if reason is None:
            parsed = _parse_raw_ceiling_locator(cand.locator_id)
            if parsed is not None:
                room_idx, plane_idx = parsed
                room = rooms_by_index.get(room_idx)
                if (
                    room is not None
                    and room.ceiling_type == "flat"
                    and 0 <= plane_idx < len(room.raw_ceiling_planes)
                ):
                    score = score_raw_plane(
                        room.raw_ceiling_planes[plane_idx], room, roof.oblique
                    )
                    if score < RAW_CEILING_LOW_QUALITY_THRESHOLD:
                        reason = "low_quality"

        if reason is None:
            kept.append(cand)
            continue

        if drops_sink is not None:
            drops_sink.append(
                {
                    "locator_id": cand.locator_id,
                    "reason": reason,
                    "n_corners": n,
                    "area_xz_m2": round(area, 3),
                    "y_span_m": round(y_span, 3),
                    "story": cand.story,
                }
            )
    return kept
