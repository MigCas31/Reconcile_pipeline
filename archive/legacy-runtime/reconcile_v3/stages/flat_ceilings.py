"""flat_ceilings — ceilings over rooms whose wall-top geometry is unambiguous.

Unambiguous means: ≥ 3 walls, and the spread of wall-top Y values is
≤ scan noise. Anything else emits V3UnresolvedRegion so the failure is
visible in the viewer rather than silently filled.
"""

from __future__ import annotations

import numpy as np

from ..audit import HypothesisTrace
from ..constants import (
    FLAT_CEILING_SPREAD_M,
    KNEE_DELTA_M,
    OUTLIER_FRACTION_MAX,
    SCAN_NOISE_M,
)
from ..ids import make_element_id
from ..io.load import V3Room
from ..models import V3FlatCeiling, V3UnresolvedRegion


def _wall_top_y(wall: dict) -> float | None:
    corners = wall.get("corners") or []
    if len(corners) < 3:
        return None
    return float(max(c[1] for c in corners))


def _wall_len_xz(wall: dict) -> float:
    corners = wall.get("corners") or []
    if len(corners) < 2:
        return 0.0
    bot_y = min(c[1] for c in corners)
    bot = [(c[0], c[2]) for c in corners if c[1] < bot_y + SCAN_NOISE_M]
    if len(bot) < 2:
        return 0.0
    pts = np.array(bot)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _outlier_fraction(walls: list[dict], top_ys: list[float]) -> float:
    """Fraction of room perimeter whose wall-top Y is at the outlier (min) height."""
    if not top_ys:
        return 1.0
    min_top = min(top_ys)
    wall_data = [(_wall_top_y(w), _wall_len_xz(w)) for w in walls]
    total_len = sum(length for _, length in wall_data if length > 0)
    if total_len == 0:
        return 1.0
    outlier_len = sum(
        length
        for y, length in wall_data
        if y is not None and y < min_top + SCAN_NOISE_M
    )
    return outlier_len / total_len


def flat_ceilings_unambiguous(
    building_uuid: str, rooms: list[V3Room]
) -> tuple[list[V3FlatCeiling], list[V3UnresolvedRegion]]:
    ceilings: list[V3FlatCeiling] = []
    unresolved: list[V3UnresolvedRegion] = []

    for room in rooms:
        if len(room.floor_polygon) < 3:
            continue
        top_ys = [y for y in (_wall_top_y(w) for w in room.walls) if y is not None]
        footprint = [(float(c[0]), 0.0, float(c[2])) for c in room.floor_polygon]
        room_eid_inner = f"room-{room.index}"

        if len(top_ys) < 3:
            unresolved.append(
                V3UnresolvedRegion(
                    id=make_element_id(
                        building_uuid, "v3-unresolved", f"ceiling-{room_eid_inner}"
                    ),
                    room_id=room.identifier,
                    footprint_xz=footprint,
                    y_range=None,
                    reason="too-few-walls",
                    context={"wall_top_count": len(top_ys)},
                    trace=HypothesisTrace(
                        stage="flat_ceilings",
                        rule="require-3-wall-tops",
                        inputs={
                            "room_id": room.identifier,
                            "wall_top_count": len(top_ys),
                        },
                        decision_reason="Fewer than 3 wall tops to derive a flat "
                        "ceiling.",
                    ),
                )
            )
            continue

        spread = float(max(top_ys) - min(top_ys))
        if spread > KNEE_DELTA_M:
            unresolved.append(
                V3UnresolvedRegion(
                    id=make_element_id(
                        building_uuid, "v3-unresolved", f"ceiling-{room_eid_inner}"
                    ),
                    room_id=room.identifier,
                    footprint_xz=footprint,
                    y_range=(float(min(top_ys)), float(max(top_ys))),
                    reason="wall-top-asymmetric",
                    context={"spread_m": spread, "wall_top_count": len(top_ys)},
                    trace=HypothesisTrace(
                        stage="flat_ceilings",
                        rule="wall-top-spread-within-knee-delta",
                        inputs={
                            "room_id": room.identifier,
                            "spread_m": spread,
                            "threshold_m": KNEE_DELTA_M,
                        },
                        decision_reason=(
                            "Wall-top spread exceeds knee-delta; room likely has "
                            "asymmetric slope. Deferred to "
                            "slanted_roofs/fill_remaining."
                        ),
                    ),
                )
            )
            continue

        if spread > SCAN_NOISE_M:
            # Soft-flat check: small spread AND outlier wall is a minority of the
            # perimeter — treat as flat using the consensus (median) wall-top Y.
            out_frac = _outlier_fraction(room.walls, top_ys)
            if spread <= FLAT_CEILING_SPREAD_M and out_frac < OUTLIER_FRACTION_MAX:
                y = float(np.median(top_ys))
                ceilings.append(
                    V3FlatCeiling(
                        id=make_element_id(
                            building_uuid, "v3-flat-ceiling", room_eid_inner
                        ),
                        room_id=room.identifier,
                        footprint_xz=[
                            (float(c[0]), y, float(c[2])) for c in room.floor_polygon
                        ],
                        y=y,
                        over="room",
                        trace=HypothesisTrace(
                            stage="flat_ceilings",
                            rule="wall-tops-soft-flat",
                            inputs={
                                "room_id": room.identifier,
                                "spread_m": spread,
                                "outlier_fraction": round(out_frac, 3),
                                "y": y,
                            },
                            decision_reason=(
                                "Spread above scan-noise but below flat-ceiling "
                                "tolerance, "
                                "and outlier wall is a minority of the perimeter — "
                                "treating as flat using median wall-top Y."
                            ),
                        ),
                    )
                )
                continue

            unresolved.append(
                V3UnresolvedRegion(
                    id=make_element_id(
                        building_uuid, "v3-unresolved", f"ceiling-{room_eid_inner}"
                    ),
                    room_id=room.identifier,
                    footprint_xz=footprint,
                    y_range=(float(min(top_ys)), float(max(top_ys))),
                    reason="wall-top-spread-above-noise",
                    context={
                        "spread_m": spread,
                        "wall_top_count": len(top_ys),
                        "outlier_fraction": round(out_frac, 3),
                    },
                    trace=HypothesisTrace(
                        stage="flat_ceilings",
                        rule="wall-top-spread-within-scan-noise",
                        inputs={
                            "room_id": room.identifier,
                            "spread_m": spread,
                            "outlier_fraction": round(out_frac, 3),
                            "flat_ceiling_spread_m": FLAT_CEILING_SPREAD_M,
                            "outlier_fraction_max": OUTLIER_FRACTION_MAX,
                        },
                        decision_reason=(
                            "Wall tops not flat within soft-flat tolerance "
                            "(spread or outlier fraction too large); "
                            "deferred to slanted_roofs/fill_remaining."
                        ),
                    ),
                )
            )
            continue

        y = float(np.median(top_ys))
        ceilings.append(
            V3FlatCeiling(
                id=make_element_id(building_uuid, "v3-flat-ceiling", room_eid_inner),
                room_id=room.identifier,
                footprint_xz=[
                    (float(c[0]), y, float(c[2])) for c in room.floor_polygon
                ],
                y=y,
                over="room",
                trace=HypothesisTrace(
                    stage="flat_ceilings",
                    rule="wall-tops-flat-within-noise",
                    inputs={
                        "room_id": room.identifier,
                        "spread_m": spread,
                        "y": y,
                    },
                    decision_reason=(
                        "All wall tops within scan-noise of each other — "
                        "trivially flat ceiling."
                    ),
                ),
            )
        )

    return ceilings, unresolved
