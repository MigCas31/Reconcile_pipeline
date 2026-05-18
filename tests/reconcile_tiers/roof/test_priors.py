from __future__ import annotations

from math import cos, radians, sin

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.roof.priors import filter_obliques_by_roof_type
from reconcile_tiers.roof.roof import ObliqueSurface, RoofCluster


def _make_oblique(
    *,
    azimuth: float,
    incl: float,
    rect: tuple[float, float, float, float] = (0.0, 0.0, 4.0, 4.0),
    base_y: float = 2.0,
    dominant_story: int = 1,
) -> ObliqueSurface:
    """Construct a synthetic oblique whose plane normal corresponds to the
    given (azimuth, inclination) and whose footprint is `rect=(x0,z0,x1,z1)`.
    """
    inc_rad = radians(incl)
    az_rad = radians(azimuth)
    nx = sin(inc_rad) * sin(az_rad)
    ny = cos(inc_rad)
    nz = sin(inc_rad) * cos(az_rad)
    if ny < 0:
        nx, ny, nz = -nx, -ny, -nz
    x0, z0, x1, z1 = rect
    cx = (x0 + x1) / 2.0
    cz = (z0 + z1) / 2.0
    d = nx * cx + ny * base_y + nz * cz
    plane = Plane(a=nx, b=ny, c=nz, d=d)
    corners = []
    for x, z in [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]:
        y = (d - nx * x - nz * z) / ny
        corners.append([float(x), float(y), float(z)])
    cluster = RoofCluster(
        segments=[],
        avg_incl=float(incl),
        avg_azimuth=float(azimuth) % 360.0,
        ref_pt=[cx, base_y, cz],
    )
    return ObliqueSurface(
        corners=corners,
        plane=plane,
        cluster=cluster,
        dominant_story=dominant_story,
        ridge={},
    )


def test_gable_keeps_mirror_pair() -> None:
    obliques = [
        _make_oblique(azimuth=0.0, incl=30.0, rect=(0.0, 0.0, 6.0, 4.0)),
        _make_oblique(azimuth=180.0, incl=30.0, rect=(0.0, 4.0, 6.0, 8.0)),
    ]
    kept, dropped = filter_obliques_by_roof_type(obliques)
    assert len(kept) == 2
    assert dropped == []


def test_gable_drops_sticking_out_extra() -> None:
    obliques = [
        _make_oblique(azimuth=0.0, incl=30.0, rect=(0.0, 0.0, 6.0, 4.0)),
        _make_oblique(azimuth=180.0, incl=30.0, rect=(0.0, 4.0, 6.0, 8.0)),
        # An extra surface pointing east at a totally different pitch — no
        # mirror partner, should be dropped as "sticking out".
        _make_oblique(azimuth=90.0, incl=15.0, rect=(6.0, 2.0, 7.0, 4.0)),
    ]
    kept, dropped = filter_obliques_by_roof_type(obliques)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert dropped[0].cluster.avg_azimuth == 90.0


def test_gable_safeguard_keeps_all_when_no_partners() -> None:
    # Two same-direction slopes — neither has a mirror partner. Filter would
    # drop everything; never-strand safeguard keeps both.
    obliques = [
        _make_oblique(azimuth=170.0, incl=20.0, rect=(0.0, 0.0, 4.0, 4.0)),
        _make_oblique(azimuth=170.0, incl=22.0, rect=(0.0, 4.0, 4.0, 8.0)),
    ]
    kept, dropped = filter_obliques_by_roof_type(obliques)
    assert len(kept) == 2
    assert dropped == []


def test_min_obliques_skip_when_too_few_for_type() -> None:
    # Two obliques that look like a hip pair (perpendicular, similar pitch)
    # — but hip needs 4. Below the min, the filter skips entirely.
    obliques = [
        _make_oblique(azimuth=0.0, incl=45.0),
        _make_oblique(azimuth=90.0, incl=45.0),
    ]
    kept, _dropped = filter_obliques_by_roof_type(obliques)
    # Either kept everything because skipped, or dropped fewer than all
    # (never-strand). Either way both surfaces should survive.
    assert len(kept) == 2


def test_complex_keeps_large_singletons() -> None:
    # Complex roof with one well-supported pair and one large singleton with
    # no partner; the singleton survives because it's large enough to be a
    # genuine wing rather than a stranded outlier.
    obliques = [
        _make_oblique(azimuth=0.0, incl=30.0, rect=(0.0, 0.0, 6.0, 4.0)),
        _make_oblique(azimuth=180.0, incl=30.0, rect=(0.0, 4.0, 6.0, 8.0)),
        _make_oblique(
            azimuth=45.0, incl=25.0, rect=(10.0, 0.0, 16.0, 8.0)
        ),  # large separate wing
        # A small unpartnered one — should be dropped.
        _make_oblique(azimuth=270.0, incl=10.0, rect=(0.0, 0.0, 1.0, 1.0)),
    ]
    kept, dropped = filter_obliques_by_roof_type(obliques)
    azimuths_kept = sorted(o.cluster.avg_azimuth for o in kept)
    azimuths_dropped = sorted(o.cluster.avg_azimuth for o in dropped)
    assert 0.0 in azimuths_kept and 180.0 in azimuths_kept
    assert 270.0 in azimuths_dropped
    # 45° at 48 m² is large enough to survive as a complex wing.
    assert 45.0 in azimuths_kept


def test_complex_drops_large_singleton_inside_partner_field() -> None:
    # Same topology as above, but the singleton occupies the already-explained
    # paired roof field. It is visually discontinuous, not a separate wing.
    obliques = [
        _make_oblique(azimuth=0.0, incl=30.0, rect=(0.0, 0.0, 6.0, 4.0)),
        _make_oblique(azimuth=180.0, incl=30.0, rect=(0.0, 4.0, 6.0, 8.0)),
        _make_oblique(azimuth=45.0, incl=25.0, rect=(0.0, 0.0, 6.0, 8.0)),
    ]

    kept, dropped = filter_obliques_by_roof_type(obliques)

    assert sorted(o.cluster.avg_azimuth for o in kept) == [0.0, 180.0]
    assert [o.cluster.avg_azimuth for o in dropped] == [45.0]


def test_empty_input_returns_empty() -> None:
    kept, dropped = filter_obliques_by_roof_type([])
    assert kept == [] and dropped == []
