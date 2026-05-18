"""End-of-pipeline integrity checks for primitive-synthesised roof pieces.

The primitive synthesisers produce geometry that's correct *at synthesis
time* (verified by the per-primitive `_validate_synthesised_surfaces`
inside each constructor). But that geometry then flows through clipping,
painting, dormer reconstruction, and thermal envelope synthesis — any of
which can quietly truncate or distort it. This module catches that
distortion *after* the full payload is built.

Violations:
- `slope_truncated` — a primitive's emitted piece doesn't reach `ridge_y`
  (downstream clip cut it short — e.g. `_clip_piece_to_eave` against a
  wrong `attic_lid_y` for a vaulted gable room).
- `corners_off_plane` — corners deviate from the piece's declared plane
  more than tolerance (clipping produced non-planar quads).
- `slope_missing` — the primitive's wing has zero matching pieces in the
  payload (the synthesised surface got filtered out entirely).

In strict mode (`ARCHITECTURAL_PRIORS_STRICT=1`), violations raise.
Otherwise they're logged + appended to a violations sink.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from reconcile_tiers.roof_primitive.types import GableParams

# Tolerances are looser than synthesis-time checks: clipping legitimately
# trims small slivers and the painter snaps to a precision grid (1e-8 m).
PIECE_PLANARITY_TOL_M = 0.005  # 5 mm
RIDGE_REACH_TOL_M = 0.05  # 5 cm
SLOPE_AZ_MATCH_TOL_DEG = 5.0
INCL_MATCH_TOL_DEG = 5.0


@dataclass(frozen=True, slots=True)
class PrimitiveViolation:
    kind: str  # "slope_truncated" | "corners_off_plane" | "slope_missing"
    wing_index: int
    primitive: str  # "gable", ...
    detail: str
    locator_id: str | None = None


def validate_primitive_emissions(
    payload, primitive_records: list, *, strict: bool | None = None
) -> list[PrimitiveViolation]:
    """Walk `payload.ceiling`, match each piece to a primitive record, and
    return the list of integrity violations.

    `strict` defaults to env var `ARCHITECTURAL_PRIORS_STRICT == "1"`. When
    True, raises `PrimitiveIntegrityError` if any violations are found.
    """
    if strict is None:
        strict = os.environ.get("ARCHITECTURAL_PRIORS_STRICT") == "1"

    violations: list[PrimitiveViolation] = []
    for params in primitive_records:
        if isinstance(params, GableParams):
            violations.extend(_validate_gable(payload, params))
        # Future: dispatch on other primitive kinds.

    if strict and violations:
        msg = "primitive integrity violations:\n  " + "\n  ".join(
            f"[{v.kind}] wing {v.wing_index} ({v.primitive}): {v.detail}"
            + (f" locator={v.locator_id}" if v.locator_id else "")
            for v in violations
        )
        raise PrimitiveIntegrityError(msg)
    return violations


class PrimitiveIntegrityError(AssertionError):
    """Raised in strict mode when primitive emissions are corrupted by
    downstream clipping/painting."""


def _validate_gable(payload, params: GableParams) -> list[PrimitiveViolation]:
    """For each of the two slopes the gable should produce, find the
    corresponding piece(s) in `payload.ceiling` and check that:
    - the piece spans up to `ridge_y` (no downstream truncation);
    - corners are still coplanar.
    """
    expected_slopes = _expected_slope_directions(params)
    matches_per_slope: list[list[tuple]] = [[], []]
    for ce in payload.ceiling:
        a, b, c, _d = ce.plane.a, ce.plane.b, ce.plane.c, ce.plane.d
        # Primitive surfaces are oblique; ignore flat ceilings.
        horiz_norm = math.hypot(a, c)
        if horiz_norm < 1e-4:
            continue
        slope_az = math.degrees(math.atan2(-a, -c)) % 360.0
        incl = math.degrees(math.atan2(horiz_norm, abs(b)))
        for i, (exp_az, exp_incl) in enumerate(expected_slopes):
            if (
                _angle_diff_360(slope_az, exp_az) <= SLOPE_AZ_MATCH_TOL_DEG
                and abs(incl - exp_incl) <= INCL_MATCH_TOL_DEG
            ):
                matches_per_slope[i].append(ce)
                break

    violations: list[PrimitiveViolation] = []
    for i, matches in enumerate(matches_per_slope):
        if not matches:
            violations.append(
                PrimitiveViolation(
                    kind="slope_missing",
                    wing_index=params.wing_index,
                    primitive="gable",
                    detail=(
                        f"slope {i} (expected slope_az={expected_slopes[i][0]:.1f}°) "
                        f"has no emitted piece"
                    ),
                )
            )
            continue
        # Collect all corners across matching pieces — the painter may have
        # split one slope into multiple pieces. Across them, the y range
        # must reach ridge_y.
        max_y = max(c.y for ce in matches for c in ce.corners)
        min_y = min(c.y for ce in matches for c in ce.corners)
        if max_y < params.ridge_y - RIDGE_REACH_TOL_M:
            for ce in matches:
                violations.append(
                    PrimitiveViolation(
                        kind="slope_truncated",
                        wing_index=params.wing_index,
                        primitive="gable",
                        detail=(
                            f"slope {i} max_y={max_y:.3f} < "
                            f"ridge_y={params.ridge_y:.3f} "
                            f"(eave_y={params.eave_y:.3f}, span={max_y - min_y:.3f}m)"
                        ),
                        locator_id=ce.locator_id,
                    )
                )
        # Planarity per matching piece.
        for ce in matches:
            a, b, c, d = ce.plane.a, ce.plane.b, ce.plane.c, ce.plane.d
            max_res = max(abs(a * p.x + b * p.y + c * p.z - d) for p in ce.corners)
            if max_res > PIECE_PLANARITY_TOL_M:
                violations.append(
                    PrimitiveViolation(
                        kind="corners_off_plane",
                        wing_index=params.wing_index,
                        primitive="gable",
                        detail=f"max corner residual={max_res:.4f} m > "
                        f"tol={PIECE_PLANARITY_TOL_M}",
                        locator_id=ce.locator_id,
                    )
                )
    return violations


def _expected_slope_directions(
    params: GableParams,
) -> list[tuple[float, float]]:
    """Return [(slope_az_a, incl), (slope_az_b, incl)] — the two compass
    directions the gable's slopes face.

    For a gable with ridge running at `ridge_axis_math` (math convention),
    each slope faces perpendicular to that axis. Compass angle of the
    slope-down direction = `(90 - (ridge_axis_math + 90)) mod 360` for one
    side and that ± 180° for the other.
    """
    # Slope direction (math) is perpendicular to the ridge → ridge_axis + 90.
    # Compass = 90 - math.
    perp_math = (params.ridge_axis_math_deg + 90.0) % 180.0
    slope_az_a = (90.0 - perp_math) % 360.0
    slope_az_b = (slope_az_a + 180.0) % 360.0
    # Inclination from eave_y, ridge_y, half-width (perpendicular distance
    # from eave to ridge in XZ).
    a_mid = (
        (params.eave_a_xz[0][0] + params.eave_a_xz[1][0]) / 2.0,
        (params.eave_a_xz[0][1] + params.eave_a_xz[1][1]) / 2.0,
    )
    b_mid = (
        (params.eave_b_xz[0][0] + params.eave_b_xz[1][0]) / 2.0,
        (params.eave_b_xz[0][1] + params.eave_b_xz[1][1]) / 2.0,
    )
    half_width = math.hypot(b_mid[0] - a_mid[0], b_mid[1] - a_mid[1]) / 2.0
    rise = params.ridge_y - params.eave_y
    incl = math.degrees(math.atan2(rise, half_width)) if half_width > 1e-9 else 90.0
    return [(slope_az_a, incl), (slope_az_b, incl)]


def _angle_diff_360(a: float, b: float) -> float:
    diff = abs((a - b) % 360.0)
    return min(diff, 360.0 - diff)
