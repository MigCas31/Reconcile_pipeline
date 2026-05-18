from reconcile.extract3d import stitch


def test_candidate_segments_reject_true_crossings():
    accepted = [((0.0, 0.0), (1.0, 1.0), 0.0, 2.4)]
    candidate = [((0.0, 1.0), (1.0, 0.0), 0.0, 2.4)]
    assert stitch._candidate_segments_cross_existing(candidate, accepted)


def test_candidate_segments_allow_shared_corner_l_joints():
    accepted = [((0.0, 0.0), (1.0, 0.0), 0.0, 2.4)]
    candidate = [((1.0, 0.0), (1.0, 1.0), 0.0, 2.4)]
    assert not stitch._candidate_segments_cross_existing(candidate, accepted)


def test_candidate_segments_reject_collinear_overlap_without_shared_endpoint():
    accepted = [((0.0, 0.0), (1.0, 0.0), 0.0, 2.4)]
    candidate = [((0.2, 0.0), (0.8, 0.0), 0.0, 2.4)]
    assert stitch._candidate_segments_cross_existing(candidate, accepted)


def test_prune_crossing_vertical_stitches_keeps_non_crossing_set():
    walls = [
        {
            "type": "stitch",
            "corners": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 2.4, 1.0],
                [0.0, 2.4, 0.0],
            ],
        },
        {
            "type": "stitch",
            "corners": [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 2.4, 0.0],
                [0.0, 2.4, 1.0],
            ],
        },
        {
            "type": "stitch_floor",
            "corners": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.5]],
        },
    ]
    out = stitch.prune_crossing_vertical_stitches(walls)
    kept_stitch = [entry for entry in out if entry.get("type") == "stitch"]
    assert len(kept_stitch) == 1
    assert any(entry.get("type") == "stitch_floor" for entry in out)
