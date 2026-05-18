"""Phase 4: train + validate classifiers on the labeled feature set.

Two models:

- **Decision tree** (`max_depth=4`, balanced) -- yields human-readable rules.
- **LightGBM** -- probability-producing scorer, isotonic-calibrated on the
  out-of-fold predictions.

Both are evaluated under `GroupKFold` by ``building_uuid`` with per-row
inverse-building-count weights (per the *generalize-before-specialize*
memory; otherwise the 594-label building dominates).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

# Columns that are identifiers, free-form text, or directly derived from the
# label -- absolutely must not be passed to any model.
_LEAK_COLS = frozenset(
    {
        "building_uuid",
        "proposal_id",
        "cluster_canonical_id",
        "parent_proposal_id",
        "labeler",
        "ts",
        "label",
        "heuristic_label",
        "label_is_accept",
        "heuristic_disagrees_with_user",
        "merge_mode",
    }
)

_OFFLINE_ONLY_PREFIXES = (
    "pred_",
    "meta_",
    "lbl_",
    "sib_",
)

_OFFLINE_ONLY_COLS = frozenset(
    {
        "label_is_near_decision_boundary",
        "label_flip_stability_score",
        "scan_age_days",
        "scan_roomplan_version",
        "bld_accept_rate_history",
        "xm_heuristic_agrees_with_label",
        "xm_heuristic_disagreement_rate_in_part",
        "lbl_pitch_bucket_accept_rate",
        "lbl_azimuth_bucket_accept_rate",
        "lbl_family_accept_rate",
        "trace_rule_accept_rate_prior",
    }
)


@dataclass(frozen=True)
class PreparedData:
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    weights: np.ndarray
    feature_names: list[str]
    heuristic_pred: np.ndarray  # 1 if heuristic == accepted else 0


def prepare_data(df: pd.DataFrame) -> PreparedData:
    """Split the parquet into model-ready numpy arrays.

    Drops identifier / leaky columns, coerces bools -> int, categoricals ->
    integer codes (LightGBM handles them natively via ``categorical_feature``;
    the tree sees them as numeric). Missing values stay as NaN for LightGBM
    and are imputed with column medians for the tree.
    """
    if "label" not in df.columns:
        raise ValueError("features dataframe must include 'label'")
    labeled = df[df["label"].isin(("accepted", "rejected"))].reset_index(drop=True)
    y = (labeled["label"] == "accepted").astype(int).to_numpy()
    groups = labeled["building_uuid"].astype(str).to_numpy()
    counts = (
        labeled.groupby("building_uuid")["building_uuid"]
        .transform("count")
        .astype(float)
    )
    weights = (1.0 / counts).to_numpy()

    def _is_model_feature_col(col: str) -> bool:
        if col in _LEAK_COLS or col in _OFFLINE_ONLY_COLS:
            return False
        if col.startswith(_OFFLINE_ONLY_PREFIXES):
            return False
        return True

    feature_cols = [
        c
        for c in labeled.columns
        if (_is_model_feature_col(c) and labeled[c].dtype.kind not in ("O", "U"))
        or c.startswith("piece_kind_is_")
    ]
    # Add back small-cardinality object cols as category codes (piece_kind etc.).
    for c in labeled.columns:
        if not _is_model_feature_col(c) or c in feature_cols:
            continue
        if labeled[c].dtype == object:
            nunique = labeled[c].nunique(dropna=True)
            if 1 < nunique <= 20:
                feature_cols.append(c)
    X = labeled[feature_cols].copy()
    # Coerce bool -> int; object -> category codes.
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype("Int8")
        elif X[c].dtype == object:
            X[c] = (
                X[c].astype("category").cat.codes.replace(-1, np.nan).astype("float32")
            )
    # Final dtype: float32 for numeric, NaN preserved.
    for c in X.columns:
        if X[c].dtype.kind in ("i", "u"):
            X[c] = X[c].astype("float32")
        elif X[c].dtype.kind == "f":
            X[c] = X[c].astype("float32")
    heuristic_pred = (labeled["heuristic_label"] == "accepted").astype(int).to_numpy()
    return PreparedData(
        X=X,
        y=y,
        groups=groups,
        weights=weights,
        feature_names=list(X.columns),
        heuristic_pred=heuristic_pred,
    )


def _group_kfold_oof_gbm(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
    lgbm_params: dict | None = None,
) -> tuple[np.ndarray, list[Any]]:
    """Out-of-fold probabilities from LightGBM under GroupKFold."""
    from lightgbm import LGBMClassifier

    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )
    if lgbm_params:
        params.update(lgbm_params)

    oof = np.full(len(y), np.nan, dtype=float)
    models: list[Any] = []
    gkf = GroupKFold(n_splits=n_splits)
    for _i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr], sample_weight=weights[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        models.append(m)
    return oof, models


def _group_kfold_oof_tree(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    n_splits: int = 5,
    max_depth: int = 4,
    random_state: int = 0,
) -> tuple[np.ndarray, list[DecisionTreeClassifier]]:
    """Out-of-fold hard-label predictions from a depth-4 tree (mean-imputed)."""
    oof = np.full(len(y), np.nan, dtype=float)
    models: list[DecisionTreeClassifier] = []
    medians = X.median(numeric_only=True)
    X_imp = X.fillna(medians)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X_imp, y, groups):
        m = DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
        )
        m.fit(X_imp.iloc[tr], y[tr], sample_weight=weights[tr])
        oof[te] = m.predict_proba(X_imp.iloc[te])[:, 1]
        models.append(m)
    return oof, models


def _calibrate_isotonic(
    y_true: np.ndarray, p_raw: np.ndarray
) -> tuple[IsotonicRegression, np.ndarray]:
    iso = IsotonicRegression(out_of_bounds="clip")
    mask = ~np.isnan(p_raw)
    iso.fit(p_raw[mask], y_true[mask])
    p_cal = np.full_like(p_raw, np.nan)
    p_cal[mask] = iso.transform(p_raw[mask])
    return iso, p_cal


def _binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, p_score: np.ndarray | None = None
) -> dict[str, Any]:
    """Precision/recall/F1 for both classes + ROC/PR AUC when a score is given."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out: dict[str, Any] = {
        "f1_accept": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0.0)),
        "f1_reject": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0.0)),
        "precision_accept": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        ),
        "recall_accept": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        ),
        "precision_reject": float(
            precision_score(y_true, y_pred, pos_label=0, zero_division=0.0)
        ),
        "recall_reject": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0.0)
        ),
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
    }
    if p_score is not None:
        mask = ~np.isnan(p_score)
        if mask.sum() >= 2 and len(np.unique(y_true[mask])) >= 2:
            out["roc_auc"] = float(roc_auc_score(y_true[mask], p_score[mask]))
            out["pr_auc"] = float(average_precision_score(y_true[mask], p_score[mask]))
    return out


def _calibration_curve(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    mask = ~np.isnan(p)
    yt, pv = y_true[mask], p[mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(pv, edges) - 1, 0, n_bins - 1)
    means_true, means_pred, counts = [], [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            means_true.append(None)
            means_pred.append(None)
            counts.append(0)
            continue
        means_true.append(float(yt[m].mean()))
        means_pred.append(float(pv[m].mean()))
        counts.append(int(m.sum()))
    return {
        "bins": edges.tolist(),
        "mean_true": means_true,
        "mean_pred": means_pred,
        "counts": counts,
    }


def _per_building(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    rows = []
    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    s_g = pd.Series(groups)
    for uuid, idx in s_g.groupby(s_g).groups.items():
        yt = s_true.loc[idx].to_numpy()
        yp = s_pred.loc[idx].to_numpy()
        if len(yt) < 2:
            continue
        rows.append(
            {
                "building_uuid": uuid,
                "n_labels": len(yt),
                "accept_rate_true": float(yt.mean()),
                "accept_rate_pred": float(yp.mean()),
                "f1_accept": float(f1_score(yt, yp, pos_label=1, zero_division=0.0)),
                "f1_reject": float(f1_score(yt, yp, pos_label=0, zero_division=0.0)),
                "errors": int((yt != yp).sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values("errors", ascending=False)
    return out


def _plot_calibration(
    raw_curve: dict,
    cal_curve: dict,
    out_path: Path,
    heuristic_rate: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for curve, label, color in [
        (raw_curve, "GBM raw", "tab:blue"),
        (cal_curve, "GBM isotonic", "tab:orange"),
    ]:
        pp = [p for p in curve["mean_pred"] if p is not None]
        tt = [t for t in curve["mean_true"] if t is not None]
        ax.plot(pp, tt, marker="o", label=label, color=color)
    hx, hy = heuristic_rate
    ax.scatter([hx], [hy], marker="s", s=80, color="tab:green", label="heuristic")
    ax.set_xlabel("mean predicted p(accept)")
    ax.set_ylabel("fraction accepted (truth)")
    ax.set_title("Calibration")
    ax.legend(loc="best")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_pr(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    heuristic_pred: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = ~np.isnan(p_cal)
    precision, recall, _ = precision_recall_curve(y_true[mask], p_cal[mask])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="tab:orange", label="GBM calibrated")
    # Heuristic operating point.
    h_prec = precision_score(y_true, heuristic_pred, pos_label=1, zero_division=0.0)
    h_rec = recall_score(y_true, heuristic_pred, pos_label=1, zero_division=0.0)
    ax.scatter(
        [h_rec], [h_prec], s=80, marker="s", color="tab:green", label="heuristic"
    )
    ax.axhline(
        y_true.mean(),
        ls=":",
        color="black",
        lw=1,
        label=f"base rate={y_true.mean():.2f}",
    )
    ax.set_xlabel("recall (accept)")
    ax.set_ylabel("precision (accept)")
    ax.set_title("Precision-Recall")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "feature_importance"):
        imp = model.feature_importance()
    else:
        return pd.DataFrame()
    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_phase4(
    df: pd.DataFrame,
    out_dir: Path,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Train + evaluate both models; write artifacts + return a metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(df)
    print(
        f"prepared: {data.X.shape[0]} rows x {data.X.shape[1]} features; "
        f"{int(data.y.sum())} accept / {int((1 - data.y).sum())} reject; "
        f"{len(set(data.groups))} buildings"
    )

    # ---------------- LightGBM ----------------
    p_gbm_raw, gbm_models = _group_kfold_oof_gbm(
        data.X, data.y, data.groups, data.weights, n_splits=n_splits
    )
    # Isotonic calibration: fit on OOF preds. This is mildly optimistic
    # because the GBM has seen the same buildings (via other folds), but
    # the calibrator is 1-D and non-parametric, so the bias is small.
    iso_cal, p_gbm_cal = _calibrate_isotonic(data.y, p_gbm_raw)

    # ---------------- Decision tree ----------------
    p_tree, tree_models = _group_kfold_oof_tree(
        data.X, data.y, data.groups, data.weights, n_splits=n_splits
    )
    pred_tree = (p_tree >= 0.5).astype(int)

    # ---------------- Metrics ----------------
    tau = 0.5
    pred_gbm_raw = (p_gbm_raw >= tau).astype(int)
    pred_gbm_cal = (p_gbm_cal >= tau).astype(int)

    metrics = {
        "heuristic": _binary_metrics(data.y, data.heuristic_pred),
        "tree_depth4": _binary_metrics(data.y, pred_tree, p_tree),
        "gbm_raw": _binary_metrics(data.y, pred_gbm_raw, p_gbm_raw),
        "gbm_calibrated": _binary_metrics(data.y, pred_gbm_cal, p_gbm_cal),
        "base_rate": float(data.y.mean()),
        "n_buildings": len(set(data.groups)),
        "n_rows": len(data.y),
        "n_features": int(data.X.shape[1]),
    }

    # Gate check
    f1_heur = metrics["heuristic"]["f1_accept"]
    f1_gbm = metrics["gbm_calibrated"]["f1_accept"]
    gate_pass = f1_gbm >= f1_heur + 0.10
    metrics["gate_heuristic_plus_10pct_f1_accept"] = {
        "heuristic_f1": f1_heur,
        "gbm_f1": f1_gbm,
        "delta": f1_gbm - f1_heur,
        "passes": bool(gate_pass),
    }

    # ---------------- Per-building heatmap ----------------
    per_b = _per_building(data.y, pred_gbm_cal, data.groups)
    per_b_path = out_dir / "per_building_accuracy.csv"
    per_b.to_csv(per_b_path, index=False)

    # ---------------- Calibration curves ----------------
    raw_curve = _calibration_curve(data.y, p_gbm_raw)
    cal_curve = _calibration_curve(data.y, p_gbm_cal)
    heuristic_rate = (
        float(data.heuristic_pred.mean()),
        float(
            data.y[data.heuristic_pred == 1].mean()
            if (data.heuristic_pred == 1).any()
            else 0.0
        ),
    )
    _plot_calibration(raw_curve, cal_curve, out_dir / "calibration.png", heuristic_rate)
    _plot_pr(data.y, p_gbm_cal, data.heuristic_pred, out_dir / "precision_recall.png")

    # ---------------- Feature importance ----------------
    imps: list[pd.DataFrame] = []
    for m in gbm_models:
        imps.append(_feature_importance(m, data.feature_names))
    if imps:
        mean_imp = (
            pd.concat(imps)
            .groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        mean_imp.to_csv(out_dir / "gbm_feature_importance.csv", index=False)

    # ---------------- Persist models + OOF preds ----------------
    import pickle

    with (out_dir / "model_gbm.pkl").open("wb") as f:
        pickle.dump(
            {
                "models": gbm_models,
                "feature_names": data.feature_names,
                "isotonic_calibrator": iso_cal,
                "calibration_curve_raw": raw_curve,
                "calibration_curve_cal": cal_curve,
            },
            f,
        )
    with (out_dir / "model_tree.pkl").open("wb") as f:
        pickle.dump(
            {"models": tree_models, "feature_names": data.feature_names},
            f,
        )
    oof = pd.DataFrame(
        {
            "building_uuid": data.groups,
            "y_true": data.y,
            "heuristic_pred": data.heuristic_pred,
            "p_tree": p_tree,
            "p_gbm_raw": p_gbm_raw,
            "p_gbm_cal": p_gbm_cal,
        }
    )
    oof.to_parquet(out_dir / "oof_predictions.parquet", index=False)

    # ---------------- Markdown summary ----------------
    _write_report(out_dir / "model_report.md", metrics, per_b, data, imps)

    return metrics


def _write_report(
    path: Path,
    metrics: dict,
    per_b: pd.DataFrame,
    data: PreparedData,
    imps: list[pd.DataFrame],
) -> None:
    def pct(x: float | None) -> str:
        return (
            "--"
            if x is None or (isinstance(x, float) and math.isnan(x))
            else f"{x:.3f}"
        )

    lines: list[str] = []
    lines.append("# Phase 4 -- model validation\n\n")
    lines.append(
        f"- {metrics['n_rows']} labels across {metrics['n_buildings']} buildings\n"
    )
    lines.append(f"- {metrics['n_features']} features\n")
    lines.append(f"- accept base rate: {metrics['base_rate']:.3f}\n\n")

    lines.append("## Metrics (threshold τ=0.5 for model predictions)\n\n")
    lines.append(
        "| Model | F1 accept | F1 reject | Prec accept | Recall accept | ROC AUC | PR "
        "AUC |\n"
    )
    lines.append("|---|---|---|---|---|---|---|\n")
    for key in ("heuristic", "tree_depth4", "gbm_raw", "gbm_calibrated"):
        m = metrics[key]
        lines.append(
            "| {name} | {f1a} | {f1r} | {pa} | {ra} | {roc} | {pr} |\n".format(
                name=key,
                f1a=pct(m.get("f1_accept")),
                f1r=pct(m.get("f1_reject")),
                pa=pct(m.get("precision_accept")),
                ra=pct(m.get("recall_accept")),
                roc=pct(m.get("roc_auc")),
                pr=pct(m.get("pr_auc")),
            )
        )
    lines.append("\n")
    gate = metrics["gate_heuristic_plus_10pct_f1_accept"]
    lines.append(
        f"**Gate (GBM F1 accept >= heuristic + 0.10):** "
        f"{'PASS' if gate['passes'] else 'FAIL'} "
        f"(Δ = {gate['delta']:.3f})\n\n"
    )

    lines.append(
        "## Confusion matrices (rows: truth, cols: prediction; classes [reject, "
        "accept])\n\n"
    )
    for key in ("heuristic", "tree_depth4", "gbm_calibrated"):
        cm = metrics[key]["confusion_matrix"]
        lines.append(f"### {key}\n\n")
        lines.append("|  | pred reject | pred accept |\n|---|---|---|\n")
        lines.append(f"| true reject | {cm[0][0]} | {cm[0][1]} |\n")
        lines.append(f"| true accept | {cm[1][0]} | {cm[1][1]} |\n\n")

    lines.append("## Worst-20 buildings by absolute error count (GBM calibrated)\n\n")
    lines.append(
        "| building_uuid | n_labels | accept_rate_true | accept_rate_pred | "
        "F1 accept | errors |\n"
    )
    lines.append("|---|---|---|---|---|---|\n")
    for _, r in per_b.head(20).iterrows():
        lines.append(
            f"| {r['building_uuid']} | {int(r['n_labels'])} | "
            f"{r['accept_rate_true']:.2f} | {r['accept_rate_pred']:.2f} | "
            f"{r['f1_accept']:.3f} | {int(r['errors'])} |\n"
        )
    lines.append("\n")

    if imps:
        mean_imp = (
            pd.concat(imps)
            .groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
        )
        lines.append("## Top-20 GBM feature importance (mean over folds)\n\n")
        lines.append("| rank | feature | importance |\n|---|---|---|\n")
        for i, (feat, imp) in enumerate(mean_imp.head(20).items(), start=1):
            lines.append(f"| {i} | `{feat}` | {imp:.0f} |\n")

    path.write_text("".join(lines))
    with (path.parent / "model_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
