"""Phase 3: per-feature descriptive statistics + effect sizes.

Produces a ranking table comparing accepted vs rejected segments on each
numeric feature. Per feature:

    - accept / reject count
    - mean, std, median, p5, p95 for each class
    - Cohen's d (pooled-SD effect size, continuous)
    - Cramér's V (categorical, computed on boolean-like columns)
    - Mutual information with the label (sklearn)

Building-UUID inverse weighting is applied when computing effect sizes so
that a single building with 500 labels can't dominate the statistic -- per
the *generalize-before-specialize* memory, we want cohort-level signal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

# Columns that are identifiers or free-form text -- don't try to rank them.
_EXCLUDE_COLS = frozenset(
    {
        "building_uuid",
        "proposal_id",
        "cluster_canonical_id",
        "parent_proposal_id",
        "labeler",
        "ts",
        "label",
        "heuristic_label",
        "label_is_accept",  # perfect correlation with target
        "heuristic_disagrees_with_user",  # depends on label
    }
)


def _building_weights(df: pd.DataFrame) -> pd.Series:
    """Weight each row by ``1 / labels_from_this_building``."""
    counts = df.groupby("building_uuid")["building_uuid"].transform("count")
    return 1.0 / counts.astype(float)


def _cohens_d(
    x: np.ndarray,
    y: np.ndarray,
    wx: np.ndarray | None = None,
    wy: np.ndarray | None = None,
) -> float | None:
    x_mask = ~np.isnan(x)
    y_mask = ~np.isnan(y)
    xc = x[x_mask]
    yc = y[y_mask]
    wxc = wx[x_mask] if wx is not None else None
    wyc = wy[y_mask] if wy is not None else None
    if len(xc) < 2 or len(yc) < 2:
        return None
    mx = np.average(xc, weights=wxc) if wxc is not None else xc.mean()
    my = np.average(yc, weights=wyc) if wyc is not None else yc.mean()
    vx = xc.var(ddof=1) if wxc is None else np.average((xc - mx) ** 2, weights=wxc)
    vy = yc.var(ddof=1) if wyc is None else np.average((yc - my) ** 2, weights=wyc)
    pooled = math.sqrt((vx + vy) / 2.0)
    if pooled < 1e-9:
        return None
    return float((mx - my) / pooled)


def _cramers_v(series: pd.Series, labels: pd.Series) -> float | None:
    """Cramér's V on a 2-way contingency table (features x labels)."""
    tab = pd.crosstab(series, labels)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2 = 0.0
    n = tab.to_numpy().sum()
    if n == 0:
        return None
    row_tot = tab.sum(axis=1).to_numpy()
    col_tot = tab.sum(axis=0).to_numpy()
    expected = np.outer(row_tot, col_tot) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = (tab.to_numpy() - expected) ** 2
        ratio = np.where(expected > 0, diff / expected, 0.0)
        chi2 = float(ratio.sum())
    r, c = tab.shape
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return None
    return float(math.sqrt(chi2 / denom))


def _is_categorical(col: pd.Series) -> bool:
    if col.dtype == bool:
        return True
    if col.dtype == object:
        # small cardinality -> categorical; high -> free text, skip
        nunique = col.nunique(dropna=True)
        return 1 < nunique <= 20
    return False


def _is_continuous(col: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(col):
        return False
    if pd.api.types.is_numeric_dtype(col):
        return col.notna().sum() >= 20
    return False


def _mutual_info_numeric(
    df: pd.DataFrame, feature_cols: Sequence[str], y: pd.Series
) -> dict[str, float]:
    """MI for numeric features (missing values imputed with column median)."""
    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError:  # pragma: no cover
        return {}
    numeric_cols = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]
    if not numeric_cols:
        return {}
    X = df[numeric_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    # Drop any cols still containing NaN (all-NaN cols) and constants.
    keep = [c for c in numeric_cols if X[c].notna().all() and X[c].nunique() > 1]
    if not keep:
        return {}
    X = X[keep].to_numpy()
    mi = mutual_info_classif(X, y.to_numpy(), random_state=0)
    return dict(zip(keep, map(float, mi), strict=False))


def rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per feature with per-class stats, effect size, and MI."""
    if "label" not in df:
        raise ValueError("features dataframe must include 'label'")
    labels = df["label"]
    accept_mask = labels == "accepted"
    reject_mask = labels == "rejected"
    weights = _building_weights(df)
    w_acc = weights[accept_mask].to_numpy()
    w_rej = weights[reject_mask].to_numpy()

    # Drop redundant ``_count`` columns from per-prefix stats -- they all
    # collapse to the same cluster_member_count. Keep one canonical copy.
    candidate_cols = [
        c
        for c in df.columns
        if c not in _EXCLUDE_COLS
        and not (c.startswith("member_") and c.endswith("_count"))
    ]
    mi = _mutual_info_numeric(df, candidate_cols, labels)

    rows: list[dict] = []
    for col in candidate_cols:
        series = df[col]
        base = {
            "feature": col,
            "n_accept": int(accept_mask.sum()),
            "n_reject": int(reject_mask.sum()),
            "n_null": int(series.isna().sum()),
            "mutual_info": mi.get(col),
        }
        if _is_continuous(series):
            a = series[accept_mask].to_numpy(dtype=float)
            r = series[reject_mask].to_numpy(dtype=float)
            a_clean = a[~np.isnan(a)]
            r_clean = r[~np.isnan(r)]
            if len(a_clean) < 5 or len(r_clean) < 5:
                continue
            row = {
                **base,
                "kind": "continuous",
                "accept_mean": float(a_clean.mean()),
                "accept_std": float(a_clean.std(ddof=0)),
                "accept_median": float(np.median(a_clean)),
                "accept_p5": float(np.percentile(a_clean, 5)),
                "accept_p95": float(np.percentile(a_clean, 95)),
                "reject_mean": float(r_clean.mean()),
                "reject_std": float(r_clean.std(ddof=0)),
                "reject_median": float(np.median(r_clean)),
                "reject_p5": float(np.percentile(r_clean, 5)),
                "reject_p95": float(np.percentile(r_clean, 95)),
                "cohens_d": _cohens_d(a, r, w_acc, w_rej),
                "cramers_v": None,
            }
            rows.append(row)
        elif _is_categorical(series):
            v = _cramers_v(series, labels)
            rows.append(
                {
                    **base,
                    "kind": "categorical",
                    "accept_mean": None,
                    "accept_std": None,
                    "accept_median": None,
                    "accept_p5": None,
                    "accept_p95": None,
                    "reject_mean": None,
                    "reject_std": None,
                    "reject_median": None,
                    "reject_p5": None,
                    "reject_p95": None,
                    "cohens_d": None,
                    "cramers_v": v,
                }
            )
    out = pd.DataFrame(rows)
    # rank by absolute effect size (d for continuous, V for categorical) then MI
    eff = out["cohens_d"].abs().where(out["kind"] == "continuous", out["cramers_v"])
    out = out.assign(abs_effect=eff).sort_values(
        ["abs_effect", "mutual_info"], ascending=[False, False], na_position="last"
    )
    out["rank"] = range(1, len(out) + 1)
    return out


def plot_top_features(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    out_dir,
    *,
    top_n: int = 20,
) -> list[str]:
    """Write a histogram PNG per top-N feature; return list of paths."""
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    top = ranking.head(top_n)
    for _, row in top.iterrows():
        col = row["feature"]
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        series = df[col]
        if row["kind"] == "continuous":
            a = series[df["label"] == "accepted"].dropna()
            r = series[df["label"] == "rejected"].dropna()
            if len(a) < 5 or len(r) < 5:
                plt.close(fig)
                continue
            combined = pd.concat([a, r])
            lo, hi = combined.quantile(0.01), combined.quantile(0.99)
            ax.hist(
                a.clip(lo, hi),
                bins=40,
                alpha=0.55,
                label="accept",
                color="tab:green",
                density=True,
            )
            ax.hist(
                r.clip(lo, hi),
                bins=40,
                alpha=0.55,
                label="reject",
                color="tab:red",
                density=True,
            )
            ax.set_xlabel(col)
            ax.set_ylabel("density")
            ax.set_title(
                f"{col}\nd={row['cohens_d']:.3f}  MI={row['mutual_info']:.3f}"
                if row.get("mutual_info") is not None
                else f"{col}\nd={row['cohens_d']:.3f}"
            )
        else:
            ct = pd.crosstab(series.astype(str), df["label"], normalize="columns")
            ct.plot.bar(ax=ax, color={"accepted": "tab:green", "rejected": "tab:red"})
            ax.set_ylabel("fraction within class")
            ax.set_title(
                f"{col}\nV={row['cramers_v']:.3f}  MI={row['mutual_info']:.3f}"
                if row.get("mutual_info") is not None
                else f"{col}\nV={row['cramers_v']:.3f}"
            )
            ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="best")
        fig.tight_layout()
        # sanitize
        safe = col.replace("/", "_").replace(" ", "_")
        path = out_dir / f"{int(row['rank']):02d}_{safe}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(str(path))
    return paths
