"""Type definitions for roof-part primitives.

A `RoofPartParams` instance is everything needed to synthesise the roof over a
building part. The primitives are deliberately small: flat parts, single
oblique parts, and composites such as gables that are just multiple oblique
parts with a shared ridge. Synthesisers emit surfaces from axis-snapped
parameters, not raw scan corner soup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from reconcile_tiers._core.plane import Plane


class PrimitiveKind(StrEnum):
    GABLE = "gable"
    OBLIQUE = "oblique"
    FLAT = "flat"


@dataclass(frozen=True, slots=True)
class GableParams:
    """A gable roof: two slopes meeting at a ridge.

    The ridge runs along the wing's *long* axis (parallel to the eaves).
    Slope direction is perpendicular to the ridge, in the math (atan2(dz,dx))
    convention so it matches the rest of the codebase's wall-axis math.

    `eave_a_xz` / `eave_b_xz` are the two eave lines (each a pair of XZ
    endpoints). `ridge_xz` is the ridge line (a pair of XZ endpoints) running
    midway between the eaves.
    """

    wing_index: int
    eave_y: float
    ridge_y: float
    ridge_axis_math_deg: float  # math convention; 0 = along +X, 90 = along +Z
    eave_a_xz: tuple[tuple[float, float], tuple[float, float]]
    eave_b_xz: tuple[tuple[float, float], tuple[float, float]]
    ridge_xz: tuple[tuple[float, float], tuple[float, float]]
    dominant_story: int


@dataclass(frozen=True, slots=True)
class ObliqueParams:
    """One axis-snapped oblique roof segment covering a wing footprint.

    The plane preserves the observed height and inclination, but its horizontal
    gradient is snapped to the wing's long or short axis. `slope_axis_math_deg`
    is the upslope/down-slope axis in math convention; the eave/ridge direction
    is perpendicular to it.
    """

    wing_index: int
    plane: Plane
    polygon_xz: tuple[tuple[float, float], ...]
    dominant_story: int
    slope_axis_math_deg: float
    eave_axis_math_deg: float
    incl_deg: float


# Backward-compatible public name while callers migrate from the old "shed"
# wording to the primitive vocabulary.
ShedParams = ObliqueParams


@dataclass(frozen=True, slots=True)
class FlatParams:
    """A flat roof/lid covering a wing footprint.

    Flat pieces are emitted by `build_flat_surfaces`; this primitive records
    that the wing is intentionally flat so the per-wing dispatch does not fall
    through to raw oblique synthesis.
    """

    wing_index: int
    y: float
    polygon_xz: tuple[tuple[float, float], ...]
    dominant_story: int


RoofPartParams = GableParams | ObliqueParams | FlatParams
