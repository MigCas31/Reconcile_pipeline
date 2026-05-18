from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import TYPE_CHECKING

from reconcile_tiers._core.wall_axis import axis_misalignment_deg
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.geometry import (
    edge_azimuth_deg,
    edge_inclination_deg,
    wall_quads,
)
from reconcile_tiers.roof.roof import RoofSegment

if TYPE_CHECKING:
    from shapely.geometry import Polygon

MIN_SEG_LEN_M = 0.3
MIN_INCL_DEG = 5.0
MAX_INCL_DEG = 80.0

# Architectural prior (LAYER 1): segments whose azimuth (compass) is more than
# this many degrees off any wall-axis-aligned direction get dropped. Keeps
# cluster averages clean. Skipped when `wall_axis_math` is None (irregular
# building or toggle off).
SEGMENT_AXIS_TOL_DEG = 20.0

# Buffer applied to a wing polygon when filtering segments by it. Tolerates
# axis-snap noise at the wing boundary so segments on a wall that lies
# exactly on a shared wing edge aren't accidentally dropped.
WING_FILTER_BUFFER_M = 0.05


def collect_oblique_segments(
    model: BuildingModel,
    has_floor_above: Callable[[float, float, int], bool] | None = None,
    exclude_room_indices: set[int] | None = None,
    *,
    wall_axis_math: float | None = None,
    wing_polygon: Polygon | None = None,
) -> list[RoofSegment]:
    """Collect oblique segments from wall edges.

    When `wall_axis_math` is supplied (LAYER 1 of architectural priors),
    segments whose down-slope direction (compass) is more than
    `SEGMENT_AXIS_TOL_DEG` off any wall-axis-aligned direction are dropped.
    This prevents scan-noise segments from contaminating downstream cluster
    averages.

    When `wing_polygon` is supplied (Phase 5: per-part fan-out), only
    segments whose XZ midpoint lies inside the buffered wing polygon are
    returned. This lets the per-part pipeline build its own wing-local
    cluster pool without mixing segments from neighbouring building parts.
    """
    if wing_polygon is not None:
        wing_filter = wing_polygon.buffer(WING_FILTER_BUFFER_M)
    else:
        wing_filter = None
    segments: list[RoofSegment] = []
    for room in model.rooms:
        if exclude_room_indices and room.index in exclude_room_indices:
            continue
        for wall in room.walls_computed:
            for corners in wall_quads(wall):
                for idx, first in enumerate(corners):
                    second = corners[(idx + 1) % len(corners)]
                    a = first
                    b = second
                    if a[1] < b[1]:
                        a, b = b, a
                    dx = b[0] - a[0]
                    dy = b[1] - a[1]
                    dz = b[2] - a[2]
                    length = sqrt(dx * dx + dy * dy + dz * dz)
                    if length < MIN_SEG_LEN_M:
                        continue
                    incl = edge_inclination_deg(a, b)
                    if incl <= MIN_INCL_DEG or incl >= MAX_INCL_DEG:
                        continue
                    midpoint_x = (a[0] + b[0]) / 2.0
                    midpoint_z = (a[2] + b[2]) / 2.0
                    if has_floor_above is not None and has_floor_above(
                        midpoint_x, midpoint_z, room.story
                    ):
                        continue
                    if wing_filter is not None:
                        from shapely.geometry import Point as _Point

                        if not wing_filter.contains(_Point(midpoint_x, midpoint_z)):
                            continue
                    azimuth = edge_azimuth_deg(a, b)
                    if (
                        wall_axis_math is not None
                        and axis_misalignment_deg(azimuth, wall_axis_math)
                        > SEGMENT_AXIS_TOL_DEG
                    ):
                        continue
                    segments.append(
                        RoofSegment(
                            a=[float(a[0]), float(a[1]), float(a[2])],
                            b=[float(b[0]), float(b[1]), float(b[2])],
                            incl=incl,
                            azimuth=azimuth,
                            length=length,
                            story=room.story,
                            room_index=room.index,
                            wall_id=wall.id,
                        )
                    )
    return segments
