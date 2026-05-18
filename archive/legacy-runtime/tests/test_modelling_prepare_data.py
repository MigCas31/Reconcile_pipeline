from __future__ import annotations

import pandas as pd

from reconcile_v3.analysis.modelling import prepare_data


def test_prepare_data_excludes_offline_only_columns():
    df = pd.DataFrame(
        [
            {
                "building_uuid": "b1",
                "proposal_id": "p1",
                "label": "accepted",
                "heuristic_label": "accepted",
                "numeric_live_feature": 1.0,
                "piece_kind_is_room": True,
                "pred_gbm_proba": 0.9,
                "meta_isolation_forest_score": -0.1,
                "lbl_pitch_bucket_accept_rate": 0.4,
                "sib_cluster_accept_fraction": 0.7,
                "xm_heuristic_agrees_with_label": True,
                "trace_rule_accept_rate_prior": 0.5,
            },
            {
                "building_uuid": "b2",
                "proposal_id": "p2",
                "label": "rejected",
                "heuristic_label": "rejected",
                "numeric_live_feature": 2.0,
                "piece_kind_is_room": False,
                "pred_gbm_proba": 0.1,
                "meta_isolation_forest_score": -0.2,
                "lbl_pitch_bucket_accept_rate": 0.6,
                "sib_cluster_accept_fraction": 0.2,
                "xm_heuristic_agrees_with_label": True,
                "trace_rule_accept_rate_prior": 0.3,
            },
        ]
    )

    prepared = prepare_data(df)

    assert "numeric_live_feature" in prepared.feature_names
    assert "piece_kind_is_room" in prepared.feature_names
    assert "pred_gbm_proba" not in prepared.feature_names
    assert "meta_isolation_forest_score" not in prepared.feature_names
    assert "lbl_pitch_bucket_accept_rate" not in prepared.feature_names
    assert "sib_cluster_accept_fraction" not in prepared.feature_names
    assert "xm_heuristic_agrees_with_label" not in prepared.feature_names
    assert "trace_rule_accept_rate_prior" not in prepared.feature_names
