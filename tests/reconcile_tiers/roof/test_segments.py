import pytest

from reconcile_tiers.roof.segments import collect_oblique_segments
from tests.reconcile_tiers.roof.helpers import make_gable_model, make_simple_slant_model


def test_collect_oblique_segments_filters_by_inclination_length_and_floor_above():
    model = make_gable_model()

    segments = collect_oblique_segments(model)

    assert len(segments) == 2
    assert {segment.room_index for segment in segments} == {0}
    assert all(segment.length >= 0.3 for segment in segments)
    assert all(segment.incl == pytest.approx(30.0, abs=1e-6) for segment in segments)

    blocked = collect_oblique_segments(
        model, has_floor_above=lambda _x, _z, _story: True
    )
    assert blocked == []


def test_collect_oblique_segments_respects_simple_slant_exclusion():
    model = make_simple_slant_model()

    assert collect_oblique_segments(model, exclude_room_indices={0}) == []


def test_collect_oblique_segments_filters_by_wing_polygon():
    """The gable model emits 2 segments, midpoints at z=0 and z=4 (both x=3).

    A wing polygon covering only z<2 keeps the z=0 segment; covering only
    z>2 keeps the z=4 segment; covering both returns both.
    """
    from shapely.geometry import Polygon

    model = make_gable_model()
    full = collect_oblique_segments(model)
    assert len(full) == 2

    south_half = Polygon([(0.0, -1.0), (6.0, -1.0), (6.0, 2.0), (0.0, 2.0)])
    north_half = Polygon([(0.0, 2.0), (6.0, 2.0), (6.0, 5.0), (0.0, 5.0)])
    full_wing = Polygon([(0.0, -1.0), (6.0, -1.0), (6.0, 5.0), (0.0, 5.0)])

    south = collect_oblique_segments(model, wing_polygon=south_half)
    north = collect_oblique_segments(model, wing_polygon=north_half)
    everything = collect_oblique_segments(model, wing_polygon=full_wing)

    assert len(south) == 1
    assert len(north) == 1
    # Together the partition recovers the full set.
    assert {round((s.a[2] + s.b[2]) / 2.0, 1) for s in south} == {0.0}
    assert {round((s.a[2] + s.b[2]) / 2.0, 1) for s in north} == {4.0}
    assert len(everything) == len(full)
