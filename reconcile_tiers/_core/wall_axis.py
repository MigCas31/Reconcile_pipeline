"""Building principal-axis estimation from wall geometry.

The principal axis is the dominant direction of the building's walls in the XZ
plane (math convention: angle from +X axis, in `[0, 90)` because perpendicular
wall families fold together mod 90). It anchors every architectural prior:
slope direction, eave direction, footprint regularization all reference the
same axis.

Two callers feed two different inputs:
- `BuildingModel.rooms[*].walls_merged[*].corners` (3D coords as nested lists)
- `tier_payload["rooms"][*]["walls"][*]["corners"]` (3D coords as `{x,y,z}` dicts)

Both reduce to a sequence of XZ corner sets per wall. The kernel
`principal_axis_and_coverage` operates on that sequence.

Rationale (also in `tracking_progress.md`): per-wall direction is computed
from the *largest-corner-pair span*, not the bounding-box dimensions, because
the bbox approach aliases perpendicular wall families into different mod-90
bins. See `tests/reconcile_tiers/roof/test_priors_at_source.py` for a
regression test on this.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import atan2, degrees, hypot

# Wall edges shorter than this are ignored — too noisy.
MIN_WALL_LENGTH_M = 0.2

# A wall direction (mod 90) within this many degrees of the dominant axis
# counts as "aligned" when computing coverage.
COVERAGE_TOL_DEG = 12.0

# Below this horizontal-normal magnitude a plane is effectively flat — no
# meaningful slope direction to snap.
MIN_HORIZONTAL_NORMAL = 1e-4

# Default snap tolerance for emission-time priors (Layer 3). Matches Layer 1
# (`SEGMENT_AXIS_TOL_DEG`) and Layer 2 (`CLUSTER_AXIS_SNAP_TOL_DEG`) so the
# whole stack uses one tolerance.
PLANE_AXIS_SNAP_TOL_DEG = 20.0


def _largest_pair_direction_xz(
    corners: Sequence[Sequence[float]],
) -> tuple[float, float, float] | None:
    """Return (dx, dz, length) of the longest XZ chord between any two corners.

    For a quad-shaped wall this is typically the diagonal between two
    diagonally-opposite corners — but since walls extrude vertically in Y, the
    XZ projection of all four corners forms a degenerate rectangle (effectively
    two distinct XZ points), so the "diagonal" reduces to the wall span. For
    non-quad walls this still picks the longest XZ extent.
    """
    n = len(corners)
    if n < 2:
        return None
    best = (0.0, 0.0, 0.0)
    for i in range(n):
        ax = float(corners[i][0])
        az = float(corners[i][2])
        for j in range(i + 1, n):
            bx = float(corners[j][0])
            bz = float(corners[j][2])
            dx = bx - ax
            dz = bz - az
            length = hypot(dx, dz)
            if length > best[2]:
                best = (dx, dz, length)
    if best[2] < 1e-9:
        return None
    return best


def _angle_mod90_delta(a: float, b: float) -> float:
    diff = abs((a - b) % 90.0)
    return min(diff, 90.0 - diff)


def principal_axis_and_coverage(
    wall_corner_sequences: Iterable[Sequence[Sequence[float]]],
) -> tuple[float, float] | None:
    """Compute the building's principal axis (math convention, mod 90) and the
    fraction of wall length that's within `COVERAGE_TOL_DEG` of `{axis, axis+90}`.

    Returns `None` if no walls are usable. Caller should gate on
    `coverage >= 0.70` before applying any axis-dependent prior.
    """
    bins: dict[int, float] = {}
    samples: list[tuple[float, float]] = []
    for corners in wall_corner_sequences:
        direction = _largest_pair_direction_xz(corners)
        if direction is None:
            continue
        dx, dz, length = direction
        if length < MIN_WALL_LENGTH_M:
            continue
        az = degrees(atan2(dz, dx)) % 90.0
        samples.append((az, length))
        bin_key = round(az / 2.0) % 45
        for offset, weight in ((0, 1.0), (1, 0.5), (-1, 0.5)):
            key = (bin_key + offset) % 45
            bins[key] = bins.get(key, 0.0) + weight * length
    if not bins:
        return None
    axis = float(max(bins.items(), key=lambda kv: kv[1])[0]) * 2.0
    total = sum(length for _, length in samples)
    aligned = sum(
        length
        for az, length in samples
        if _angle_mod90_delta(az, axis) <= COVERAGE_TOL_DEG
    )
    coverage = aligned / total if total > 0 else 0.0
    return axis, coverage


def nearest_axis_aligned_compass(
    slope_az_compass: float, wall_axis_math: float
) -> float:
    """Return the slope_az (compass) value closest to `slope_az_compass` such
    that the slope is axis-aligned with walls at `wall_axis_math`.

    Compass and math conventions are reflections (`compass = 90 - math`), so
    axis-aligned slope_az candidates are `{-axis + k*90}` mod 360, NOT
    `{axis + k*90}`. See `audit/roof_defects.py:_axis_misalignment_deg` for the
    cross-convention background.
    """
    candidates = [(-wall_axis_math + k) % 360.0 for k in (0.0, 90.0, 180.0, 270.0)]
    return min(candidates, key=lambda c: _circular_delta(slope_az_compass, c))


def _circular_delta(a: float, b: float) -> float:
    diff = abs((a - b) % 360.0)
    return min(diff, 360.0 - diff)


def axis_misalignment_deg(slope_az_compass: float, wall_axis_math: float) -> float:
    """Distance (deg, in [0, 45]) from axis-alignment between a slope and walls.

    Mirrors `audit/roof_defects.py:_axis_misalignment_deg` and lives here so
    non-audit callers can use it without depending on the audit module.
    """
    sum_mod_90 = (slope_az_compass + wall_axis_math) % 90.0
    return min(sum_mod_90, 90.0 - sum_mod_90)


# ---- thin wrappers for the two existing call sites --------------------------


def axis_from_building_model(model) -> tuple[float, float] | None:
    """Wrapper for `BuildingModel`-typed callers.

    Reads `model.rooms[*].walls_merged[*].corners` and feeds the kernel.
    """

    def _walls() -> Iterable[Sequence[Sequence[float]]]:
        for room in model.rooms:
            for wall in room.walls_merged:
                yield wall.corners

    return principal_axis_and_coverage(_walls())


def axis_from_payload(payload: dict) -> tuple[float, float] | None:
    """Wrapper for tier-payload-typed callers (audit module).

    Reads `payload["rooms"][*]["walls"][*]["corners"]` (corners as `{x,y,z}`
    dicts) and adapts to the kernel's `[x, y, z]` sequence input.
    """

    def _walls() -> Iterable[Sequence[Sequence[float]]]:
        for room in payload.get("rooms") or []:
            for wall in room.get("walls") or []:
                corners = wall.get("corners") or []
                yield [
                    [
                        float(c.get("x", 0.0)),
                        float(c.get("y", 0.0)),
                        float(c.get("z", 0.0)),
                    ]
                    for c in corners
                ]

    return principal_axis_and_coverage(_walls())


# ---- Layer 3 emission-time snap ---------------------------------------------


def snap_corners_and_plane_to_axis(
    corners: Sequence[Sequence[float]],
    plane,
    *,
    wall_axis_math: float,
    tol_deg: float = PLANE_AXIS_SNAP_TOL_DEG,
):
    """Rotate `plane`'s slope direction to the nearest axis-aligned target,
    preserving inclination magnitude and the corner-cloud centroid height.

    Returns `(snapped_corners, snapped_plane)` or `None` if the plane is
    near-flat, outside `tol_deg` of any axis-aligned target, or the snapped
    plane can't y-resolve a corner.

    XZ corner positions are preserved; only Y is recomputed via the snapped
    plane's `y_at(x, z)` so the surface stays anchored to its scanned
    footprint.
    """
    import math as _math

    from reconcile_tiers._core.plane import Plane

    a, b, c = float(plane.a), float(plane.b), float(plane.c)
    r = _math.hypot(a, c)
    if r < MIN_HORIZONTAL_NORMAL:
        return None  # near-flat; nothing to snap

    slope_az = _math.degrees(_math.atan2(-a, -c)) % 360.0
    if axis_misalignment_deg(slope_az, wall_axis_math) > tol_deg:
        return None

    target = nearest_axis_aligned_compass(slope_az, wall_axis_math)
    target_rad = _math.radians(target)
    a_new = -r * _math.sin(target_rad)
    c_new = -r * _math.cos(target_rad)

    n = len(corners)
    if n == 0:
        return None
    cx = sum(float(p[0]) for p in corners) / n
    cy = sum(float(p[1]) for p in corners) / n
    cz = sum(float(p[2]) for p in corners) / n
    d_new = a_new * cx + b * cy + c_new * cz

    snapped = Plane(a=a_new, b=b, c=c_new, d=d_new)
    new_corners: list[list[float]] = []
    for p in corners:
        x, _, z = float(p[0]), float(p[1]), float(p[2])
        y = snapped.y_at(x, z)
        if y is None:
            return None
        new_corners.append([x, y, z])
    return new_corners, snapped
