"""Offline-only model and training-time feature enrichers.

These features are available when rebuilding ``artifacts/features_expanded.parquet``
from labels plus model artifacts. They are not inference-time signals for new
buildings, so they do not belong in the online feature path.
"""

from __future__ import annotations

import itertools
import pickle

import numpy as np
import pandas as pd

_EPS = 1e-9


def _entropy(series: pd.Series) -> float | None:
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return None
    probs = counts.to_numpy(dtype=float)
    probs = probs / probs.sum()
    valid = probs > 0.0
    return float(-np.sum(probs[valid] * np.log(probs[valid])))


def _accept_fraction(labels: pd.Series) -> float | None:
    valid = labels.dropna()
    if valid.empty:
        return None
    return float((valid == "accepted").mean())


def _agreement_fraction(values: pd.Series, labels: pd.Series) -> float | None:
    mask = values.notna() & labels.notna()
    if not mask.any():
        return None
    return float((values[mask] == labels[mask]).mean())


def _attach_oof_predictions(
    df: pd.DataFrame,
    *,
    oof_predictions_path: str,
    reference_features_path: str | None,
) -> pd.DataFrame:
    oof = pd.read_parquet(oof_predictions_path)
    attached = None
    if len(oof) == len(df):
        if "building_uuid" not in oof.columns or oof["building_uuid"].astype(
            str
        ).equals(df["building_uuid"].astype(str)):
            attached = oof.reset_index(drop=True)
    if attached is None and reference_features_path:
        ref = pd.read_parquet(
            reference_features_path, columns=["building_uuid", "proposal_id"]
        )
        if len(ref) == len(oof):
            ref = ref.copy()
            ref["building_uuid"] = ref["building_uuid"].astype(str)
            ref["proposal_id"] = ref["proposal_id"].astype(str)
            ref["__row__"] = np.arange(len(ref))
            oof_join = oof.reset_index(drop=True).drop(
                columns=["building_uuid"], errors="ignore"
            )
            keyed = ref.merge(oof_join, left_on="__row__", right_index=True, how="left")
            lookup = keyed.set_index(["building_uuid", "proposal_id"])
            idx = pd.MultiIndex.from_frame(
                df[["building_uuid", "proposal_id"]].astype(str)
            )
            attached = lookup.reindex(idx).reset_index(drop=True)
    if attached is None:
        return df
    out = df.copy()
    out["pred_gbm_proba"] = attached["p_gbm_cal"].astype(float)
    out["pred_gbm_margin"] = out["pred_gbm_proba"] - 0.5
    out["pred_gbm_disagreement_with_tree"] = (
        attached["p_gbm_cal"].astype(float) - attached["p_tree"].astype(float)
    ).abs()
    out["label_is_near_decision_boundary"] = out["pred_gbm_margin"].abs() < 0.1
    return out


def _numeric_feature_matrix(
    df: pd.DataFrame, *, max_features: int = 32
) -> tuple[np.ndarray, list[str]]:
    numeric = df.select_dtypes(include=[np.number]).copy()
    drop_cols = {
        "label_is_accept",
        "pred_gbm_proba",
        "pred_gbm_margin",
        "pred_gbm_disagreement_with_tree",
        "pred_ensemble_stddev",
    }
    keep = [c for c in numeric.columns if c not in drop_cols]
    if not keep:
        return (np.empty((len(df), 0), dtype=float), [])
    numeric = numeric[keep]
    variances = (
        numeric.var(axis=0, skipna=True).fillna(0.0).sort_values(ascending=False)
    )
    chosen = variances.head(min(max_features, len(variances))).index.tolist()
    X = numeric[chosen].astype(float)
    X = X.fillna(X.median())
    arr = X.to_numpy(dtype=float)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std < _EPS] = 1.0
    arr = (arr - mean) / std
    return (arr, chosen)


def augment_dataframe(
    feat_df: pd.DataFrame,
    *,
    oof_predictions_path: str | None = None,
    gbm_pkl_path: str | None = None,
    reference_features_path: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``feat_df`` with offline-only columns added."""
    df = feat_df.copy()

    if oof_predictions_path:
        df = _attach_oof_predictions(
            df,
            oof_predictions_path=oof_predictions_path,
            reference_features_path=reference_features_path,
        )

    if gbm_pkl_path:
        with open(gbm_pkl_path, "rb") as f:
            bundle = pickle.load(f)
        feature_names = list(bundle["feature_names"])
        models = list(bundle["models"])
        if feature_names and models:
            X = df.reindex(columns=feature_names).astype("float32")
            probs = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
            df["pred_ensemble_stddev"] = probs.std(axis=1)
            booster = models[0].booster_
            contrib = booster.predict(X, pred_contrib=True)
            if contrib.ndim == 2 and contrib.shape[1] >= len(feature_names):
                contrib = contrib[:, : len(feature_names)]
                abs_contrib = np.abs(contrib)
                top_idx = abs_contrib.argmax(axis=1)
                df["pred_shap_value_top_feature"] = [feature_names[i] for i in top_idx]
                sums = abs_contrib.sum(axis=1, keepdims=True)
                probs_abs = np.divide(
                    abs_contrib, sums, out=np.zeros_like(abs_contrib), where=sums > _EPS
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    ent = -np.sum(
                        np.where(probs_abs > 0.0, probs_abs * np.log(probs_abs), 0.0),
                        axis=1,
                    )
                df["pred_shap_entropy"] = ent

    if "label" in df.columns:
        labels = df["label"]
        df["lbl_was_split_child"] = (
            df.get("is_split_child").fillna(False).astype(bool)
            if "is_split_child" in df.columns
            else False
        )
        df["lbl_labeler_id"] = df.get("labeler") if "labeler" in df.columns else None
        df["lbl_latency_ms"] = None
        df["lbl_session_id"] = None
        df["lbl_skip_count_before_decide"] = None
        df["lbl_skip_count_in_session"] = None
        df["lbl_viewer_camera_az_deg"] = None
        df["lbl_has_merge_event"] = (
            df.get("merge_mode").fillna(False).astype(bool)
            if "merge_mode" in df.columns
            else False
        )
        df["scan_age_days"] = None
        df["scan_roomplan_version"] = None

        if "proposal_id" in df.columns:
            ordered = df.assign(_order=np.arange(len(df)))
            if "ts" in ordered.columns:
                ordered = ordered.sort_values(
                    ["proposal_id", "ts", "_order"], kind="stable"
                )
            flip_score = pd.Series(1.0, index=ordered.index, dtype=float)
            for _, grp in ordered.groupby("proposal_id", sort=False):
                vals = grp["label"].dropna().tolist()
                if len(vals) >= 2:
                    flips = sum(1 for a, b in itertools.pairwise(vals) if a != b)
                    score = 1.0 - (flips / max(len(vals) - 1, 1))
                else:
                    score = 1.0
                flip_score.loc[grp.index] = score
            df["label_flip_stability_score"] = flip_score.sort_index().to_numpy()

        if "cluster_canonical_id" in df.columns:
            df["sib_cluster_count"] = df.groupby("cluster_canonical_id")[
                "proposal_id"
            ].transform("count")
            df["sib_cluster_accept_fraction"] = df.groupby("cluster_canonical_id")[
                "label"
            ].transform(_accept_fraction)
            df["sib_peer_label_entropy"] = df.groupby("cluster_canonical_id")[
                "label"
            ].transform(_entropy)
            df["sib_peer_label_disagreement_rate"] = 1.0 - df.groupby(
                "cluster_canonical_id"
            )["label"].transform(
                lambda s: (
                    float(s.value_counts(normalize=True, dropna=True).max())
                    if not s.dropna().empty
                    else np.nan
                )
            )

        if {"building_uuid", "part_index"}.issubset(df.columns):
            df["sib_part_accept_fraction"] = df.groupby(
                [
                    "building_uuid",
                    "part_index",
                ]
            )["label"].transform(_accept_fraction)

        if "building_uuid" in df.columns:
            df["sib_bld_accept_fraction"] = df.groupby("building_uuid")[
                "label"
            ].transform(_accept_fraction)
            df["bld_accept_rate_history"] = df["sib_bld_accept_fraction"]
            if "ts" in df.columns:
                ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
                temp = df.assign(_ts=ts, _row=np.arange(len(df)))
                temp = temp.sort_values(["building_uuid", "_ts", "_row"], kind="stable")
                before = temp.groupby("building_uuid").cumcount()
                sizes = temp.groupby("building_uuid")["proposal_id"].transform("count")
                temp["sib_labeled_before_me_count"] = before
                temp["sib_labeled_after_me_count"] = sizes - before - 1
                temp = temp.sort_index()
                df["sib_labeled_before_me_count"] = temp[
                    "sib_labeled_before_me_count"
                ].to_numpy()
                df["sib_labeled_after_me_count"] = temp[
                    "sib_labeled_after_me_count"
                ].to_numpy()
            else:
                df["sib_labeled_before_me_count"] = None
                df["sib_labeled_after_me_count"] = None

        if {"plane_azimuth_deg", "plane_incl_deg", "plane_d"}.issubset(df.columns):
            coplanar_key = list(
                zip(
                    df["building_uuid"].astype(str),
                    np.round(df["plane_azimuth_deg"].fillna(-999.0) / 5.0).astype(int),
                    np.round(df["plane_incl_deg"].fillna(-999.0) / 3.0).astype(int),
                    np.round(df["plane_d"].fillna(-999.0) / 0.25).astype(int),
                    strict=False,
                )
            )
            df["_coplanar_key"] = coplanar_key
            df["sib_coplanar_accept_fraction"] = df.groupby("_coplanar_key")[
                "label"
            ].transform(_accept_fraction)
            df.drop(columns=["_coplanar_key"], inplace=True)

        if "heuristic_label" in df.columns:
            df["xm_heuristic_agrees_with_label"] = df["heuristic_label"] == df["label"]
            if {"building_uuid", "part_index"}.issubset(df.columns):
                df["xm_heuristic_disagreement_rate_in_part"] = (
                    1.0
                    - df.groupby(["building_uuid", "part_index"])
                    .apply(
                        lambda g: _agreement_fraction(g["heuristic_label"], g["label"])
                    )
                    .reindex(
                        pd.MultiIndex.from_frame(df[["building_uuid", "part_index"]])
                    )
                    .to_numpy()
                )

        if "plane_incl_deg" in df.columns:
            pitch_bucket = (
                np.round(df["plane_incl_deg"].fillna(-999.0) / 5.0) * 5.0
            ).astype(str)
            df["lbl_pitch_bucket_accept_rate"] = labels.groupby(pitch_bucket).transform(
                _accept_fraction
            )

        if "plane_azimuth_deg" in df.columns:
            az_bucket = (
                np.round(df["plane_azimuth_deg"].fillna(-999.0) / 22.5) * 22.5
            ).astype(str)
            df["lbl_azimuth_bucket_accept_rate"] = labels.groupby(az_bucket).transform(
                _accept_fraction
            )

        if "derived_part_roof_family_guess" in df.columns:
            fam = df["derived_part_roof_family_guess"].fillna("unknown")
            df["lbl_family_accept_rate"] = labels.groupby(fam).transform(
                _accept_fraction
            )

        if "trace_rule" in df.columns:
            rules = df["trace_rule"].fillna("unknown")
            df["trace_rule_accept_rate_prior"] = labels.groupby(rules).transform(
                _accept_fraction
            )

    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

        X, _ = _numeric_feature_matrix(df)
        if X.shape[1] > 0 and len(df) >= 16:
            iso = IsolationForest(
                random_state=0, n_estimators=100, contamination="auto"
            )
            iso.fit(X)
            iso_scores = iso.score_samples(X)
            df["meta_isolation_forest_score"] = iso_scores

            lof = LocalOutlierFactor(n_neighbors=min(20, len(df) - 1), novelty=False)
            lof.fit_predict(X)
            lof_scores = -lof.negative_outlier_factor_
            df["meta_local_outlier_factor"] = lof_scores

            nn = NearestNeighbors(n_neighbors=min(11, len(df)))
            nn.fit(X)
            dists, inds = nn.kneighbors(X)
            if dists.shape[1] > 1:
                df["meta_knn_distance_mean_m"] = dists[:, 1:].mean(axis=1)
            else:
                df["meta_knn_distance_mean_m"] = dists.mean(axis=1)
            if "label" in df.columns:
                accept = (df["label"] == "accepted").to_numpy(dtype=float)
                neigh_accept = []
                for row_inds in inds:
                    use = row_inds[1:] if len(row_inds) > 1 else row_inds
                    neigh_accept.append(
                        float(np.mean(accept[use])) if len(use) else np.nan
                    )
                df["meta_knn_accept_fraction"] = neigh_accept

            kmeans = MiniBatchKMeans(
                n_clusters=min(12, max(2, len(df) // 50)),
                random_state=0,
                batch_size=min(1024, len(df)),
                n_init="auto",
            )
            df["meta_cluster_label_from_unsupervised"] = kmeans.fit_predict(X)

            iso_cut = np.nanpercentile(df["meta_isolation_forest_score"], 10)
            knn_cut = np.nanpercentile(df["meta_knn_distance_mean_m"], 90)
            df["meta_in_training_manifold"] = (
                df["meta_isolation_forest_score"] >= iso_cut
            )
            df["meta_low_density_region"] = df["meta_knn_distance_mean_m"] >= knn_cut
        else:
            for col in (
                "meta_isolation_forest_score",
                "meta_local_outlier_factor",
                "meta_knn_accept_fraction",
                "meta_knn_distance_mean_m",
                "meta_in_training_manifold",
                "meta_cluster_label_from_unsupervised",
                "meta_low_density_region",
            ):
                df[col] = None
    except Exception:
        for col in (
            "meta_isolation_forest_score",
            "meta_local_outlier_factor",
            "meta_knn_accept_fraction",
            "meta_knn_distance_mean_m",
            "meta_in_training_manifold",
            "meta_cluster_label_from_unsupervised",
            "meta_low_density_region",
        ):
            df[col] = None

    return df
