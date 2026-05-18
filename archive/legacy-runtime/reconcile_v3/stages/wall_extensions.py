"""wall_extensions — raise walls to the slab above, knee-aware.

Decide knee-vs-extend **before** raising any wall (see plan §
"Knee-wall decision"). Consumes per-room ``SlopeHypothesis`` from
``slanted_roofs`` so the 'slab short of wall' branch can tell a knee
wall from a genuine data hole.
"""

from __future__ import annotations

from shapely.geometry import Point, Polygon

from reconcile.extract3d.ceilings import extend_wall_to_slab

from ..audit import HypothesisTrace
from ..constants import EAVE_OVERHANG_M, KNEE_DELTA_M, SCAN_NOISE_M
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import V3Slab, V3UnresolvedRegion, V3WallExtension
from .slanted_roofs import SlopeHypothesis


def _wall_xz_midpoint(corners: list) -> tuple[float, float]:
    xs = [c[0] for c in corners]
    zs = [c[2] for c in corners]
    return (sum(xs) / len(xs), sum(zs) / len(zs))


def _slab_polygon(slab: V3Slab) -> Polygon | None:
    if len(slab.polygon) < 3:
        return None
    try:
        poly = Polygon([(c[0], c[2]) for c in slab.polygon])
    except Exception:
        return None
    if not poly.is_valid or poly.is_empty:
        return None
    return poly


def _slab_y(slab: V3Slab) -> float:
    return (
        float(sum(c[1] for c in slab.polygon) / len(slab.polygon))
        if slab.polygon
        else 0.0
    )


def _pick_slab_above(
    wall_corners: list,
    wall_top_y: float,
    slabs_with_polys: list[tuple[V3Slab, Polygon, float]],
) -> tuple[V3Slab, Polygon, float] | None:
    """Nearest slab whose Y strictly exceeds the wall top by at least SCAN_NOISE_M."""
    mx, mz = _wall_xz_midpoint(wall_corners)
    pt = Point(mx, mz)
    best = None
    best_dist = float("inf")
    for slab, poly, y in slabs_with_polys:
        if y <= wall_top_y + SCAN_NOISE_M:
            continue
        d = poly.distance(pt)
        if d < best_dist:
            best_dist = d
            best = (slab, poly, y)
    return best


def extend_walls_to_slab_above(
    building_uuid: str,
    rooms: list[V3Room],
    slabs: list[V3Slab],
    slope_hypotheses: dict[str, SlopeHypothesis] | None = None,
) -> tuple[list[V3WallExtension], list[V3UnresolvedRegion]]:
    slope_hypotheses = slope_hypotheses or {}

    slabs_with_polys: list[tuple[V3Slab, Polygon, float]] = []
    for s in slabs:
        poly = _slab_polygon(s)
        if poly is None:
            continue
        slabs_with_polys.append((s, poly, _slab_y(s)))

    extensions: list[V3WallExtension] = []
    unresolved: list[V3UnresolvedRegion] = []

    for room in rooms:
        room_centroid = (
            (
                sum(c[0] for c in room.floor_polygon) / max(1, len(room.floor_polygon)),
                sum(c[2] for c in room.floor_polygon) / max(1, len(room.floor_polygon)),
            )
            if room.floor_polygon
            else (0.0, 0.0)
        )

        for wall in room.walls:
            corners = wall.get("corners") or []
            if len(corners) < 3:
                continue
            y_top = max(c[1] for c in corners)
            picked = _pick_slab_above(corners, y_top, slabs_with_polys)
            if picked is None:
                continue
            slab, slab_poly, y_slab = picked
            delta = y_slab - y_top
            wall_id = str(wall.get("id") or "wall:?")
            inner = f"{room.identifier}::{wall_id}"

            if delta > KNEE_DELTA_M:
                unresolved.append(
                    V3UnresolvedRegion(
                        id=make_element_id(
                            building_uuid, "v3-unresolved", f"slab-too-far-{inner}"
                        ),
                        room_id=room.identifier,
                        footprint_xz=[(float(c[0]), 0.0, float(c[2])) for c in corners],
                        y_range=(float(y_top), float(y_slab)),
                        reason="slab-too-far-above-wall",
                        context={"delta_m": round(delta, 3)},
                        trace=HypothesisTrace(
                            stage="wall_extensions",
                            rule="delta-exceeds-knee-threshold",
                            inputs={
                                "delta_m": round(delta, 3),
                                "threshold": KNEE_DELTA_M,
                            },
                            decision_reason=(
                                "Slab-above is more than KNEE_DELTA_M higher than the "
                                "wall top — can't extend without guessing."
                            ),
                        ),
                    )
                )
                continue

            if delta < SCAN_NOISE_M:
                ret = extend_wall_to_slab(corners, y_slab)
                if ret is None:
                    continue
                strips = ret.get("extension_strip") or []
                strip_corners = [tuple(p) for p in (strips[0] if strips else [])]
                extensions.append(
                    V3WallExtension(
                        id=make_element_id(
                            building_uuid, "v3-wall-extension", f"trivial-{inner}"
                        ),
                        wall_id=wall_id,
                        room_id=room.identifier,
                        strip_corners=strip_corners,
                        target_slab_id=slab.id,
                        behind_knee_wall=False,
                        trace=HypothesisTrace(
                            stage="wall_extensions",
                            rule="trivial-extend",
                            inputs={
                                "delta_m": round(delta, 3),
                                "slab_id": slab.id,
                            },
                            decision_reason="Δ under SCAN_NOISE_M — raise wall to slab "
                            "directly.",
                        ),
                    )
                )
                continue

            # SCAN_NOISE_M ≤ Δ ≤ KNEE_DELTA_M → project wall onto slab footprint.
            mx, mz = _wall_xz_midpoint(corners)
            pt = Point(mx, mz)
            dir_out = (mx - room_centroid[0], mz - room_centroid[1])
            dnorm = (dir_out[0] ** 2 + dir_out[1] ** 2) ** 0.5 or 1.0
            dir_out = (dir_out[0] / dnorm, dir_out[1] / dnorm)

            if slab_poly.contains(pt):
                inside_dist = slab_poly.boundary.distance(pt)
                if inside_dist > EAVE_OVERHANG_M:
                    extend_branch = "interior"
                else:
                    extend_branch = "cantilever"
                ret = extend_wall_to_slab(corners, y_slab)
                if ret is None:
                    continue
                strips = ret.get("extension_strip") or []
                strip_corners = [tuple(p) for p in (strips[0] if strips else [])]
                extensions.append(
                    V3WallExtension(
                        id=make_element_id(
                            building_uuid,
                            "v3-wall-extension",
                            f"{extend_branch}-{inner}",
                        ),
                        wall_id=wall_id,
                        room_id=room.identifier,
                        strip_corners=strip_corners,
                        target_slab_id=slab.id,
                        behind_knee_wall=False,
                        trace=HypothesisTrace(
                            stage="wall_extensions",
                            rule=f"{extend_branch}-extend",
                            inputs={
                                "delta_m": round(delta, 3),
                                "inside_dist_m": round(float(inside_dist), 3),
                                "slab_id": slab.id,
                            },
                            decision_reason=(
                                "Slab above contains wall midpoint — raise wall to "
                                "slab."
                            ),
                        ),
                    )
                )
            else:
                outside_dist = slab_poly.distance(pt)
                if outside_dist <= EAVE_OVERHANG_M:
                    ret = extend_wall_to_slab(corners, y_slab)
                    if ret is None:
                        continue
                    strips = ret.get("extension_strip") or []
                    strip_corners = [tuple(p) for p in (strips[0] if strips else [])]
                    extensions.append(
                        V3WallExtension(
                            id=make_element_id(
                                building_uuid,
                                "v3-wall-extension",
                                f"cantilever-{inner}",
                            ),
                            wall_id=wall_id,
                            room_id=room.identifier,
                            strip_corners=strip_corners,
                            target_slab_id=slab.id,
                            behind_knee_wall=False,
                            trace=HypothesisTrace(
                                stage="wall_extensions",
                                rule="cantilever-extend",
                                inputs={
                                    "delta_m": round(delta, 3),
                                    "outside_dist_m": round(float(outside_dist), 3),
                                    "slab_id": slab.id,
                                },
                                decision_reason=(
                                    "Slab terminates within eave-overhang of wall line "
                                    "— cantilever case, extend."
                                ),
                            ),
                        )
                    )
                else:
                    # Slab short of wall by > EAVE_OVERHANG_M.
                    has_hypothesis = room.identifier in slope_hypotheses
                    if has_hypothesis:
                        extensions.append(
                            V3WallExtension(
                                id=make_element_id(
                                    building_uuid, "v3-wall-extension", f"knee-{inner}"
                                ),
                                wall_id=wall_id,
                                room_id=room.identifier,
                                strip_corners=[],
                                target_slab_id=None,
                                behind_knee_wall=True,
                                trace=HypothesisTrace(
                                    stage="wall_extensions",
                                    rule="knee-wall-detected",
                                    inputs={
                                        "delta_m": round(delta, 3),
                                        "outside_dist_m": round(float(outside_dist), 3),
                                        "hypothesis_sources": list(
                                            slope_hypotheses[room.identifier].sources
                                        ),
                                    },
                                    decision_reason=(
                                        "Slab short of wall by > EAVE_OVERHANG_M AND "
                                        "room has a SlopeHypothesis — treat as knee "
                                        "wall; "
                                        "do NOT extend."
                                    ),
                                ),
                            )
                        )
                        unresolved.append(
                            V3UnresolvedRegion(
                                id=make_element_id(
                                    building_uuid, "v3-unresolved", f"knee-{inner}"
                                ),
                                room_id=room.identifier,
                                footprint_xz=[
                                    (float(c[0]), 0.0, float(c[2])) for c in corners
                                ],
                                y_range=(float(y_top), float(y_slab)),
                                reason="knee-wall",
                                context={
                                    "wall_id": wall_id,
                                    "outside_dist_m": round(float(outside_dist), 3),
                                },
                                trace=HypothesisTrace(
                                    stage="wall_extensions",
                                    rule="knee-wall-region",
                                    inputs={
                                        "delta_m": round(delta, 3),
                                        "outside_dist_m": round(float(outside_dist), 3),
                                    },
                                    decision_reason=(
                                        "Knee wall region above wall top — "
                                        "flat_ceilings "
                                        "and fill_remaining must skip it."
                                    ),
                                ),
                            )
                        )
                    else:
                        unresolved.append(
                            V3UnresolvedRegion(
                                id=make_element_id(
                                    building_uuid,
                                    "v3-unresolved",
                                    f"knee-indet-{inner}",
                                ),
                                room_id=room.identifier,
                                footprint_xz=[
                                    (float(c[0]), 0.0, float(c[2])) for c in corners
                                ],
                                y_range=(float(y_top), float(y_slab)),
                                reason="knee-wall-indeterminate",
                                context={
                                    "wall_id": wall_id,
                                    "outside_dist_m": round(float(outside_dist), 3),
                                },
                                trace=HypothesisTrace(
                                    stage="wall_extensions",
                                    rule="knee-wall-indeterminate",
                                    inputs={
                                        "delta_m": round(delta, 3),
                                        "outside_dist_m": round(float(outside_dist), 3),
                                    },
                                    decision_reason=(
                                        "Slab short of wall AND no SlopeHypothesis "
                                        "corroborates a knee wall — cannot decide."
                                    ),
                                ),
                            )
                        )

    return extensions, unresolved
