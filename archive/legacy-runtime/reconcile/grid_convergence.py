"""Grid convergence computation for true north → grid north correction.

Ported from web-main: amsterdam/libraries/appkit/src/geo/north-reference.ts
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GridNorthReference:
    crs: str
    central_meridian_deg: float
    model: Literal["transverse_mercator", "lambert_conformal_conic"]
    cone_constant: float | None = None  # only for lambert_conformal_conic


# Denmark: UTM zone 32N (EPSG:25832), central meridian 9°
EPSG_25832 = GridNorthReference(
    crs="EPSG:25832",
    central_meridian_deg=9.0,
    model="transverse_mercator",
)

# France: Lambert-93 (EPSG:2154), central meridian 3°
EPSG_2154 = GridNorthReference(
    crs="EPSG:2154",
    central_meridian_deg=3.0,
    model="lambert_conformal_conic",
    cone_constant=0.725607765053267,
)


def compute_grid_convergence_rad(
    lat_deg: float, lon_deg: float, ref: GridNorthReference
) -> float:
    """Compute grid convergence in radians.

    Grid convergence is the angle between true north and grid north at a given
    location. Positive means grid north is east of true north.

    Matches web-main computeGridConvergenceRad() in north-reference.ts.
    """
    lon_delta_rad = math.radians(lon_deg - ref.central_meridian_deg)

    if ref.model == "lambert_conformal_conic":
        assert ref.cone_constant is not None
        return lon_delta_rad * ref.cone_constant

    # Transverse Mercator (UTM)
    lat_rad = math.radians(lat_deg)
    return math.atan(math.tan(lon_delta_rad) * math.sin(lat_rad))


def compute_grid_convergence_deg(
    lat_deg: float, lon_deg: float, ref: GridNorthReference
) -> float:
    """Compute grid convergence in degrees."""
    return math.degrees(compute_grid_convergence_rad(lat_deg, lon_deg, ref))


def grid_convergence_for_country(
    country_code: str, lat_deg: float, lon_deg: float
) -> float:
    """Return grid convergence in degrees for a given country and location.

    Defaults to 0.0 for unknown countries.
    """
    refs = {
        "DK": EPSG_25832,
        "DE": EPSG_25832,  # Germany also uses UTM zone 32N
        "FR": EPSG_2154,
    }
    ref = refs.get(country_code.upper())
    if ref is None:
        return 0.0
    return compute_grid_convergence_deg(lat_deg, lon_deg, ref)
