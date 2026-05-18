import pytest

from reconcile_tiers.roof.clustering import (
    MIN_CLUSTER_SIZE,
    cluster_oblique_segments,
)
from reconcile_tiers.roof.geometry import angle_diff_deg
from reconcile_tiers.roof.roof import RoofSegment


def _seg(azimuth: float, z: float) -> RoofSegment:
    return RoofSegment(
        a=[0.0, 2.0, z],
        b=[3.0, 3.732050807568877, z],
        incl=30.0,
        azimuth=azimuth,
        length=3.4641016151377544,
        story=0,
        room_index=0,
    )


def test_same_facing_segments_form_one_directional_cluster():
    clusters = cluster_oblique_segments([_seg(90.0, 0.0), _seg(92.0, 2.0)])

    assert len(clusters) == 1
    assert len(clusters[0].segments) == MIN_CLUSTER_SIZE
    assert clusters[0].avg_incl == pytest.approx(30.0)
    assert angle_diff_deg(clusters[0].avg_azimuth, 91.0) == pytest.approx(0.0, abs=0.1)


def test_opposing_gable_faces_remain_separate_directional_clusters():
    clusters = cluster_oblique_segments(
        [
            _seg(90.0, 0.0),
            _seg(91.0, 2.0),
            _seg(270.0, 4.0),
            _seg(271.0, 6.0),
        ]
    )

    assert len(clusters) == 2
    assert [len(cluster.segments) for cluster in clusters] == [
        MIN_CLUSTER_SIZE,
        MIN_CLUSTER_SIZE,
    ]
    assert angle_diff_deg(
        clusters[0].avg_azimuth, clusters[1].avg_azimuth
    ) == pytest.approx(180.0, abs=1.0)


def test_perpendicular_faces_do_not_cluster():
    clusters = cluster_oblique_segments([_seg(0.0, 0.0), _seg(90.0, 2.0)])

    assert clusters == []
