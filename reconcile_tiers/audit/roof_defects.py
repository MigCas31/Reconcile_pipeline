"""Roof-architecture-prior defect classifiers over tier_payload.json.

Five rules check each oblique ceiling piece (or the building as a whole)
against the architectural priors that motivate the planned primitive-library
roof reconstructor:

    roof_off_axis              slope direction not parallel/perpendicular to building
    axis
    roof_fragmented            >=2 pieces share the same plane (one surface, split)
    roof_cross_extension       single piece spans >=2 wings of the footprint
    roof_irregular_footprint   building-level: footprint not rectilinear enough for
                               primitive-library reconstruction (informational gate,
                               not a defect)

`roof_off_axis` only fires on rectilinear-enough footprints (rectilinearity
coverage >= 0.70) — the comparison axis isn't meaningful otherwise.
"""

from __future__ import annotations

from math import atan2, degrees, hypot
from typing import Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.plane import (
    planes_equivalent as _planes_equivalent,
)
from reconcile_tiers._core.wing_decomposition import (
    decompose_to_wings,
)
from reconcile_tiers.audit.rules import (
    RULE_HYPOTHESES,
    FlagItem,
    _corners_xz,
    _make_item,
    _room_floor_pieces,
    _safe_polygon,
)

# ---- thresholds --------------------------------------------------------------

# Off-axis slope direction tolerance (mod 90). The cohort report sweeps
# 5/10/15/20° to expose threshold sensitivity.
OFF_AXIS_TOL_DEG = 15.0

# Inclination band defining "this is an oblique roof piece". Borrowed from
# RAW_OBLIQUE_MIN_INCL_DEG / RAW_OBLIQUE_MAX_INCL_DEG in roof/obliques.py.
OBLIQUE_MIN_INCL_DEG = 10.0
OBLIQUE_MAX_INCL_DEG = 75.0

# Cross-extension: a piece must overlap >=2 wings by at least this area to flag.
CROSS_EXTENSION_MIN_OVERLAP_M2 = 1.0

# Rectilinearity gate matches wing_decomposition.decompose_to_wings (line 363).
RECTILINEARITY_GATE = 0.70


# ---- root-cause hypotheses (merged into rules.RULE_HYPOTHESES on import) -----

ROOF_DEFECT_HYPOTHESES: dict[str, dict[str, Any]] = {
    "roof_off_axis": {
        "step": "reconcile_tiers/roof/clustering.py",
        "files": [
            "reconcile_tiers/roof/clustering.py",
            "reconcile_tiers/roof/obliques.py",
        ],
        "summary": (
            "Slope direction is not aligned with the building's principal axis "
            "(parallel/perpendicular). Either azimuth clustering split a single "
            "direction into two, or the input segments are too noisy to snap to "
            "the dominant axis. A yaw-snap prior would resolve this."
        ),
    },
    "roof_fragmented": {
        "step": "reconcile_tiers/roof/clipping.py",
        "files": [
            "reconcile_tiers/roof/clipping.py",
            "reconcile_tiers/roof/clustering.py",
        ],
        "summary": (
            "Multiple ceiling pieces share the same plane within tolerance. "
            "Likely one continuous slope split during clipping or per-room "
            "emission. A continuity-prior merge would resolve this."
        ),
    },
    "roof_cross_extension": {
        "step": "reconcile_tiers/_core/wing_decomposition.py",
        "files": [
            "reconcile_tiers/_core/wing_decomposition.py",
            "reconcile_tiers/roof/clipping.py",
        ],
        "summary": (
            "A single oblique surface spans >=2 footprint wings. Either the "
            "slope legitimately crosses the wing boundary (rare) or the wing "
            "decomposition over-split. Per-part roof reconstruction would tell."
        ),
    },
    "roof_irregular_footprint": {
        "step": "reconcile_tiers/_core/wing_decomposition.py",
        "files": [
            "reconcile_tiers/_core/wing_decomposition.py",
        ],
        "summary": (
            "Footprint is not rectilinear enough for primitive-library "
            "reconstruction. The building needs a hybrid mesh-patch fallback. "
            "This flag is informational, not a defect."
        ),
    },
    "extension_strip_crosses_wing": {
        "step": "reconcile_tiers/extract/extension.py",
        "files": [
            "reconcile_tiers/extract/extension.py",
            "reconcile_tiers/_core/wing_decomposition_v2.py",
        ],
        "summary": (
            "A descent or uplift strip's XZ projection spans >=2 footprint "
            "wings. The strip should be confined to the wing of the upper "
            "wall (descent) or lower wall (uplift) it serves; crossing wing "
            "boundaries means the synthesised inter-story void was filled "
            "from a support on a different wing. Likely v1 extension "
            "synthesis with no wing constraint, or a wing-aware run where "
            "the wall's wing membership was misclassified."
        ),
    },
    "roof_corners_not_planar": {
        "step": "reconcile_tiers/roof_primitive/gable.py | "
        "reconcile_tiers/assemble/ceiling_painter.py",
        "files": [
            "reconcile_tiers/roof_primitive/gable.py",
            "reconcile_tiers/assemble/ceiling_painter.py",
            "reconcile_tiers/assemble/synthesis.py",
        ],
        "summary": (
            "An oblique ceiling piece's corners deviate from its declared "
            "plane by more than a few mm. This means the piece is a non-planar "
            "quadrilateral — the renderer triangulates it with an arbitrary "
            "diagonal, producing visible criss-cross artifacts. Caused by "
            "synthesis bugs (eave/ridge length mismatch in primitive "
            "construction) or downstream clipping that didn't preserve "
            "planarity. Catches Phase 4 regressions early."
        ),
    },
}

RULE_HYPOTHESES.update(ROOF_DEFECT_HYPOTHESES)


# ---- helpers -----------------------------------------------------------------


def _building_footprint_polygon(payload: dict[str, Any]) -> Polygon | None:
    """Union of ground-story floor pieces as a single polygon (or None)."""
    rooms = payload.get("rooms") or []
    stories = [r.get("story") for r in rooms if r.get("story") is not None]
    if not stories:
        return None
    ground = min(stories)
    polys: list[Polygon] = []
    for room in rooms:
        if room.get("story") != ground:
            continue
        for piece in _room_floor_pieces(room):
            poly = _safe_polygon(_corners_xz(piece.get("corners") or []))
            if poly:
                polys.append(poly)
    if not polys:
        return None
    union = polys[0] if len(polys) == 1 else unary_union(polys)
    if not isinstance(union, Polygon):
        try:
            union = max(union.geoms, key=lambda g: g.area)
        except (AttributeError, ValueError):
            return None
    return union if isinstance(union, Polygon) and not union.is_empty else None


def _piece_orientation(plane: dict[str, float]) -> tuple[float, float] | None:
    """
    Return (slope_azimuth_deg, inclination_deg) for an upward-facing oblique slope.
    """
    a = float(plane.get("a", 0.0))
    b = float(plane.get("b", 0.0))
    c = float(plane.get("c", 0.0))
    norm = (a * a + b * b + c * c) ** 0.5
    if norm < 1e-9 or b <= 0:
        return None
    a, b, c = a / norm, b / norm, c / norm
    incl = degrees(atan2(hypot(a, c), abs(b)))
    if incl < OBLIQUE_MIN_INCL_DEG or incl > OBLIQUE_MAX_INCL_DEG:
        return None
    azimuth = degrees(atan2(-a, -c)) % 360.0
    return azimuth, incl


def _angle_mod90_delta(a: float, b: float) -> float:
    """Smallest absolute difference between two angles modulo 90 degrees.

    Both arguments must be in the SAME angular convention (math or compass).
    Mixing conventions silently aliases by 90 deg and produces phantom 30-60 deg
    misalignment on correctly-aligned roofs — see `_axis_misalignment_deg` for
    the cross-convention check used by the off-axis / eave rules.
    """
    diff = abs((a - b) % 90.0)
    return min(diff, 90.0 - diff)


def _axis_misalignment_deg(slope_az_compass: float, wall_axis_math: float) -> float:
    """Distance (deg, in [0, 45]) from axis-alignment between a slope and walls.

    `slope_az_compass` is the down-slope direction in compass convention
    (`atan2(-a, -c)` from the plane normal — 0 deg = +Z, 90 deg = +X).
    `wall_axis_math` is the dominant wall direction in math convention
    (`atan2(dz, dx)` from a wall edge — 0 deg = +X, 90 deg = +Z).

    A slope is axis-aligned (parallel or perpendicular to walls) iff
    `slope_az_compass + wall_axis_math` is a multiple of 90 deg, because
    compass and math are reflections (`compass = 90 - math`). Returning the
    distance from a multiple of 90 (folded to [0, 45]) gives the misalignment
    angle independent of whether the slope is parallel or perpendicular.
    """
    sum_mod_90 = (slope_az_compass + wall_axis_math) % 90.0
    return min(sum_mod_90, 90.0 - sum_mod_90)


def _locator_kind(locator: str | None) -> str:
    if not locator:
        return ""
    parts = locator.split("::", 2)
    return parts[1] if len(parts) == 3 else ""


def _oblique_pieces(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter ceiling pieces to oblique roof slopes by computed orientation.

    Skips per-room emissions (locator kinds ending in `-room`) — those are
    rendering-scope copies of a building-level surface, not independent physical
    pieces. Returned list still contains one entry per emitted piece (a single
    physical surface can appear as multiple pieces after the split fan-out).
    Use `_oblique_physical_surfaces` when you want one entry per physical
    surface (off-axis / eave-not-parallel rules); `_oblique_pieces` is the right
    input for `roof_fragmented` (which exists to count emission duplicates).
    """
    out: list[dict[str, Any]] = []
    for piece in payload.get("ceiling") or []:
        kind = _locator_kind(piece.get("locator_id"))
        if kind.endswith("-room"):
            continue
        plane = piece.get("plane") or {}
        orient = _piece_orientation(plane)
        if orient is None:
            continue
        out.append(
            {
                "raw": piece,
                "plane": plane,
                "azimuth": orient[0],
                "inclination": orient[1],
            }
        )
    return out


def _oblique_physical_surfaces(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Group `_oblique_pieces` by plane equivalence; return one representative
    per group (the piece with largest XZ polygon area).

    Use this for rules that ask "is this physical surface defective?" — those
    rules should count surfaces, not emission redundancy. The split fan-out in
    `_emit_wing_clipped` produces multiple pieces per (source, room) combo,
    all sharing the source's plane. Without this dedup, building-level rates
    are inflated by the room-cell count.
    """
    pieces = _oblique_pieces(payload)
    n = len(pieces)
    if n <= 1:
        return pieces
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _planes_equivalent(pieces[i]["plane"], pieces[j]["plane"]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[dict[str, Any]] = []
    for indices in groups.values():
        # Representative = piece with largest XZ polygon area.
        def _area_of(i: int) -> float:
            poly = _safe_polygon(_corners_xz(pieces[i]["raw"].get("corners") or []))
            return poly.area if poly is not None else 0.0

        out.append(pieces[max(indices, key=_area_of)])
    return out


def _wall_axis_and_coverage(payload: dict[str, Any]) -> tuple[float, float] | None:
    """Derive principal axis from wall XZ projections.

    Thin wrapper around `_core/wall_axis.axis_from_payload`. The shared kernel
    de-duplicates the same algorithm that `roof/architectural_priors.py` uses
    on `BuildingModel`-typed inputs.
    """
    from reconcile_tiers._core.wall_axis import axis_from_payload

    return axis_from_payload(payload)


def _building_axis(payload: dict[str, Any]) -> tuple[float, float, Polygon] | None:
    """Return (principal_axis_deg, wall_coverage, footprint_polygon).

    Axis comes from wall directions (cleaner signal than floor union).
    The footprint polygon (still unioned floors) is returned so callers can
    feed it into `decompose_to_wings` for the cross-extension rule.
    """
    wall_info = _wall_axis_and_coverage(payload)
    footprint = _building_footprint_polygon(payload)
    if wall_info is None or footprint is None:
        return None
    axis, coverage = wall_info
    return axis, coverage, footprint


# ---- rules -------------------------------------------------------------------


def rule_roof_off_axis(payload: dict[str, Any]) -> list[FlagItem]:
    axis_info = _building_axis(payload)
    if axis_info is None:
        return []
    axis, coverage, _ = axis_info
    if coverage < RECTILINEARITY_GATE:
        return []
    items: list[FlagItem] = []
    for piece in _oblique_physical_surfaces(payload):
        slope_az = piece["azimuth"]
        delta = _axis_misalignment_deg(slope_az, axis)
        if delta > OFF_AXIS_TOL_DEG:
            items.append(
                _make_item(
                    piece["raw"].get("locator_id"),
                    rule="roof_off_axis",
                    severity="med",
                    evidence={
                        "slope_azimuth_deg": float(slope_az),
                        "inclination_deg": float(piece["inclination"]),
                        "building_axis_deg": float(axis),
                        "delta_mod90_deg": float(delta),
                        "tolerance_deg": OFF_AXIS_TOL_DEG,
                        "rectilinearity_coverage": float(coverage),
                    },
                )
            )
    items.sort(key=lambda it: -it["evidence"]["delta_mod90_deg"])
    return items


def rule_roof_fragmented(payload: dict[str, Any]) -> list[FlagItem]:
    pieces = _oblique_pieces(payload)
    n = len(pieces)
    if n < 2:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _planes_equivalent(pieces[i]["plane"], pieces[j]["plane"]):
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    items: list[FlagItem] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        sibling_locators = [pieces[i]["raw"].get("locator_id") for i in indices]
        for idx in indices:
            piece = pieces[idx]["raw"]
            siblings = [
                loc for loc in sibling_locators if loc != piece.get("locator_id")
            ]
            items.append(
                _make_item(
                    piece.get("locator_id"),
                    rule="roof_fragmented",
                    severity="med",
                    evidence={
                        "group_size": len(indices),
                        "sibling_locators": siblings,
                        "plane": pieces[idx]["plane"],
                    },
                )
            )
    return items


def _wing_overlaps(poly: Polygon, wings) -> list[int]:
    return [
        wing.index
        for wing in wings
        if poly.intersection(wing.polygon).area >= CROSS_EXTENSION_MIN_OVERLAP_M2
    ]


def _emit_cross_wing_flag(
    locator_id: str | None,
    overlapping: list[int],
    n_wings: int,
    rule: str,
    *,
    extra_evidence: dict[str, Any] | None = None,
) -> FlagItem:
    evidence: dict[str, Any] = {
        "wing_indices": overlapping,
        "wing_count_total": n_wings,
        "min_overlap_m2": CROSS_EXTENSION_MIN_OVERLAP_M2,
    }
    if extra_evidence:
        evidence.update(extra_evidence)
    return _make_item(locator_id, rule=rule, severity="med", evidence=evidence)


def _wings_for_payload(payload: dict[str, Any]):
    axis_info = _building_axis(payload)
    if axis_info is None:
        return None
    _, _, footprint = axis_info
    wings = decompose_to_wings(footprint)
    return wings if len(wings) >= 2 else None


def rule_roof_cross_extension(payload: dict[str, Any]) -> list[FlagItem]:
    """Flag physical roof surfaces (deduped by plane) whose XZ footprint spans
    >=2 wings. Uses the union of all emission pieces in a plane group as the
    surface's footprint — that recovers the pre-split extent, which the
    representative-piece approach would miss when the largest piece happens
    to be confined to one wing.
    """
    wings = _wings_for_payload(payload)
    if wings is None:
        return []
    pieces = _oblique_pieces(payload)
    n = len(pieces)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _planes_equivalent(pieces[i]["plane"], pieces[j]["plane"]):
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    items: list[FlagItem] = []
    for indices in groups.values():
        polys_with_idx: list[tuple[int, Polygon]] = []
        for i in indices:
            poly = _safe_polygon(_corners_xz(pieces[i]["raw"].get("corners") or []))
            if poly is not None:
                polys_with_idx.append((i, poly))
        if not polys_with_idx:
            continue
        merged_poly = (
            polys_with_idx[0][1]
            if len(polys_with_idx) == 1
            else unary_union([p for _, p in polys_with_idx])
        )
        overlapping = _wing_overlaps(merged_poly, wings)
        if len(overlapping) >= 2:
            largest_i, _ = max(polys_with_idx, key=lambda pair: pair[1].area)
            items.append(
                _emit_cross_wing_flag(
                    pieces[largest_i]["raw"].get("locator_id"),
                    overlapping,
                    len(wings),
                    rule="roof_cross_extension",
                )
            )
    return items


# Max corner-to-plane residual before an oblique piece is flagged as
# non-planar. 5 mm — looser than the synthesis-time invariant (1 mm) so
# downstream clipping snaps don't false-flag, but tight enough to catch
# real bugs (eave/ridge length mismatch produced ~0.6 m residuals before
# the Phase 4 planarity fix).
PLANARITY_MAX_RESIDUAL_M = 0.005


def rule_roof_corners_not_planar(payload: dict[str, Any]) -> list[FlagItem]:
    """Flag oblique ceiling pieces whose corners deviate from their
    declared plane by more than `PLANARITY_MAX_RESIDUAL_M`.

    A piece's `plane` is supposed to be the unique plane its corners
    lie on. Any non-planar quadrilateral is a synthesis or clipping
    bug — the renderer will triangulate the piece into two tris with
    an arbitrary diagonal, producing visible criss-cross artifacts in
    the viewer. Catches the Phase 4 non-planar gable bug and any
    similar regression in future primitive synthesisers / clip code.
    """
    out: list[FlagItem] = []
    for piece in payload.get("ceiling") or []:
        plane = piece.get("plane") or {}
        try:
            a = float(plane["a"])
            b = float(plane["b"])
            c = float(plane["c"])
            d = float(plane["d"])
        except (KeyError, TypeError, ValueError):
            continue
        # Only oblique pieces — flat ceilings always pass this trivially.
        if hypot(a, c) < 1e-4:
            continue
        corners = piece.get("corners") or []
        if len(corners) < 3:
            continue
        max_res = 0.0
        for corner in corners:
            try:
                x = float(corner["x"])
                y = float(corner["y"])
                z = float(corner["z"])
            except (KeyError, TypeError, ValueError):
                continue
            res = abs(a * x + b * y + c * z - d)
            if res > max_res:
                max_res = res
        if max_res > PLANARITY_MAX_RESIDUAL_M:
            out.append(
                _make_item(
                    piece.get("locator_id"),
                    rule="roof_corners_not_planar",
                    severity="warning",
                    evidence={
                        "max_residual_m": round(max_res, 4),
                        "tolerance_m": PLANARITY_MAX_RESIDUAL_M,
                    },
                )
            )
    return out


def rule_extension_strip_crosses_wing(payload: dict[str, Any]) -> list[FlagItem]:
    """Flag descent/uplift strips whose XZ projection spans >=2 wings.

    A descent strip belongs to one upper-story wall and should be
    confined to that wall's wing; same for an uplift strip and its lower
    wall. A strip crossing wings means the inter-story void was filled
    using a support on a different wing — the visual signature is a
    spurious synthesised wall sitting under a different mass.
    """
    wings = _wings_for_payload(payload)
    if wings is None:
        return []

    items: list[FlagItem] = []
    for room in payload.get("rooms") or []:
        for wall in room.get("walls") or []:
            for strip_kind in ("descent_strip", "uplift_strip"):
                strip = wall.get(strip_kind)
                if not strip or len(strip) < 3:
                    continue
                xz_corners = []
                for corner in strip:
                    try:
                        xz_corners.append([float(corner["x"]), float(corner["z"])])
                    except (KeyError, TypeError, ValueError):
                        continue
                poly = _safe_polygon(xz_corners)
                if poly is None or poly.area < CROSS_EXTENSION_MIN_OVERLAP_M2:
                    continue
                overlapping = _wing_overlaps(poly, wings)
                if len(overlapping) >= 2:
                    items.append(
                        _emit_cross_wing_flag(
                            wall.get("locator_id"),
                            overlapping,
                            len(wings),
                            rule="extension_strip_crosses_wing",
                            extra_evidence={"strip_kind": strip_kind},
                        )
                    )
    return items


def rule_roof_irregular_footprint(payload: dict[str, Any]) -> list[FlagItem]:
    axis_info = _building_axis(payload)
    if axis_info is None:
        return []
    _, coverage, _ = axis_info
    if coverage >= RECTILINEARITY_GATE:
        return []
    uuid = payload.get("uuid") or payload.get("building_uuid")
    locator = f"{uuid}::building::footprint" if uuid else None
    return [
        _make_item(
            locator,
            rule="roof_irregular_footprint",
            severity="info",
            evidence={
                "rectilinearity_coverage": float(coverage),
                "gate": RECTILINEARITY_GATE,
            },
        )
    ]


ROOF_DEFECT_RULES = (
    rule_roof_off_axis,
    rule_roof_fragmented,
    rule_roof_cross_extension,
    rule_extension_strip_crosses_wing,
    rule_roof_irregular_footprint,
    rule_roof_corners_not_planar,
)
