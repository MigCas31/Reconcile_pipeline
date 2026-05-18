from __future__ import annotations

from shapely.geometry import Polygon, box

from reconcile.wing_decomposition import decompose_to_wings


def test_decompose_to_wings_coalesces_crossing_mass_fragments() -> None:
    footprint = Polygon(
        [
            (-3.586, 0.263),
            (-3.726, 2.341),
            (-0.75, 2.541),
            (-0.716, 2.04),
            (-0.673, 2.043),
            (0.35, 2.11),
            (0.327, 2.489),
            (1.694, 2.738),
            (5.755, 2.978),
            (5.79, 2.384),
            (7.824, 2.518),
            (8.06, -1.044),
            (4.226, -1.297),
            (4.324, -2.777),
            (5.145, -2.708),
            (5.606, -8.255),
            (0.952, -8.642),
            (0.49, -3.086),
            (0.481, -2.974),
            (1.378, -2.899),
            (1.28, -1.416),
            (-3.404, -1.726),
            (-3.536, 0.266),
        ]
    )

    wings = decompose_to_wings(footprint)

    assert len(wings) == 2
    assert round(wings[0].area_m2, 1) == 50.1
    assert round(wings[1].area_m2, 1) == 26.3


def test_decompose_to_wings_preserves_simple_l_shape_split() -> None:
    footprint = box(0.0, 0.0, 8.0, 3.0).union(box(0.0, 0.0, 3.0, 8.0))

    wings = decompose_to_wings(footprint)

    assert len(wings) == 2
    assert [round(wing.area_m2, 1) for wing in wings] == [24.0, 15.0]
