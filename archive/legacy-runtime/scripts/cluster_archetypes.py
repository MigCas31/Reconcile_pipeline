"""Data-driven archetype clustering of V3 merged-roof-segment proposals.

Pipeline: numeric features -> median impute -> StandardScale -> PCA (95% var)
-> sklearn HDBSCAN. Per cluster we report size, accept rate among labeled
rows, the top distinguishing features (cluster-mean z-score vs grand mean),
and 5 exemplar ``proposal_id``s that the viewer can load directly.

Usage::

    python scripts/cluster_archetypes.py
    python scripts/cluster_archetypes.py --min-cluster-size 80

Output: ``reports/archetype_clusters.md`` plus a CSV mirror at
``reports/archetype_clusters.csv`` (one row per proposal with its cluster id).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
FEATURES_PATH = REPO / "artifacts" / "features_expanded.parquet"
REPORT_MD = REPO / "reports" / "archetype_clusters.md"
REPORT_CSV = REPO / "reports" / "archetype_clusters.csv"

DROP_COLS = {
    "building_uuid",
    "proposal_id",
    "cluster_canonical_id",
    "label",
    "heuristic_label",
    "labeler",
    "parent_proposal_id",
    "label_is_accept",
}


def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c not in DROP_COLS]
    X = df[keep].select_dtypes(include=[np.number]).copy()
    # Boolean columns come through as bool; cast for scaling.
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
    return X


def _top_distinguishing(
    X_std: np.ndarray,
    feature_names: list[str],
    members: np.ndarray,
    k: int = 8,
) -> list[tuple[str, float, float]]:
    """Return the top ``k`` features where the cluster mean differs most
    from the grand mean (in standardized units). We report ``(feature,
    cluster_mean_raw, z_vs_grand)`` -- z is how many grand-population SDs
    the cluster's mean sits away from zero (the scaler subtracted the
    grand mean already, so the cluster mean IS the z)."""
    if members.sum() == 0:
        return []
    cluster_mean = X_std[members].mean(axis=0)
    order = np.argsort(-np.abs(cluster_mean))[:k]
    return [
        (feature_names[i], float(cluster_mean[i]), float(cluster_mean[i]))
        for i in order
    ]


def _exemplars(df: pd.DataFrame, members: np.ndarray, n: int = 5) -> list[str]:
    pool = df.loc[members, "proposal_id"].tolist()
    if len(pool) <= n:
        return pool
    # Deterministic, but spread across the cluster (every Nth).
    step = max(1, len(pool) // n)
    return pool[::step][:n]


def cluster(
    features_path: Path,
    method: str,
    min_cluster_size: int,
    min_samples: int,
    pca_variance: float,
    n_clusters: int,
    cluster_selection_method: str = "eom",
) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_parquet(features_path)
    X = _select_numeric(df)
    feature_names = list(X.columns)
    print(f"loaded {len(df)} proposals with {len(feature_names)} numeric features")

    # Median impute -- robust to the long-tailed geometric features.
    X = X.fillna(X.median(numeric_only=True))
    X = X.replace([np.inf, -np.inf], 0.0)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X.values)

    pca = PCA(n_components=pca_variance, svd_solver="full")
    X_pca = pca.fit_transform(X_std)
    print(
        f"PCA -> {X_pca.shape[1]} components "
        f"(keep {pca_variance:.0%} var, "
        f"explained={pca.explained_variance_ratio_.sum():.3f})"
    )

    if method == "hdbscan":
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
        )
        labels = hdb.fit_predict(X_pca)
        n_noise = int((labels == -1).sum())
        n_found = int(labels.max() + 1) if labels.max() >= 0 else 0
        print(
            f"HDBSCAN -> {n_found} clusters + {n_noise} noise "
            f"({100 * n_noise / len(df):.1f}%)"
        )
    elif method == "kmeans":
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = km.fit_predict(X_pca)
        print(f"KMeans -> {n_clusters} clusters (inertia={km.inertia_:.1f})")
    else:
        raise ValueError(f"unknown method: {method}")

    df_out = df[["proposal_id", "building_uuid", "label", "heuristic_label"]].copy()
    df_out["cluster"] = labels

    # Build per-cluster summary.
    summaries: list[dict] = []
    for cid in sorted(set(labels)):
        members = labels == cid
        total = int(members.sum())
        cluster_labels = df.loc[members, "label"]
        n_accept = int((cluster_labels == "accepted").sum())
        n_reject = int((cluster_labels == "rejected").sum())
        n_labeled = n_accept + n_reject
        accept_rate = (n_accept / n_labeled) if n_labeled else float("nan")

        heur = df.loc[members, "heuristic_label"]
        heur_accept = int((heur == "accepted").sum())
        heur_total = int((heur.isin(["accepted", "rejected"])).sum())
        heur_accept_rate = (heur_accept / heur_total) if heur_total else float("nan")

        distinguishing = _top_distinguishing(X_std, feature_names, members, k=8)
        exemplars = _exemplars(df, members, n=5)

        # Original-feature means for the distinguishing features -- useful
        # for eyeballing whether "std=2.3" means "tiny segment" or "giant
        # segment" without having to invert the scaler in your head.
        raw_means: dict[str, float] = {}
        for name, _, _ in distinguishing:
            raw_means[name] = float(X.loc[members, name].mean())

        summaries.append(
            {
                "cluster": int(cid),
                "size": total,
                "n_labeled": n_labeled,
                "n_accept": n_accept,
                "n_reject": n_reject,
                "accept_rate": accept_rate,
                "heur_accept_rate": heur_accept_rate,
                "distinguishing": distinguishing,
                "raw_means": raw_means,
                "exemplars": exemplars,
            }
        )
    return df_out, summaries


def _fmt_rate(r: float) -> str:
    if r != r:  # NaN
        return "--"
    return f"{100 * r:.1f}%"


def write_report(
    summaries: list[dict],
    out_md: Path,
    out_csv: Path,
    df_out: pd.DataFrame,
    method: str,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    real = [s for s in summaries if s["cluster"] != -1]
    real.sort(
        key=lambda s: s["accept_rate"] if s["accept_rate"] == s["accept_rate"] else 2.0
    )
    noise = next((s for s in summaries if s["cluster"] == -1), None)

    lines = []
    lines.append("# Archetype clusters -- V3 merged roof segments\n")
    lines.append(
        f"{method.upper()} on a StandardScaled + PCA-reduced feature matrix. "
        "Clusters are sorted by human accept rate -- the lowest-accept "
        "clusters at the top are the strongest candidates for a classifier "
        "auto-reject rule or a systematic proposer bug; the highest-accept "
        "clusters at the bottom are candidate auto-accepts.\n"
    )

    total_labeled = sum(s["n_labeled"] for s in real)
    total_accept = sum(s["n_accept"] for s in real)
    lines.append(
        f"**Population.** {len(df_out)} proposals * "
        f"{len(real)} clusters * "
        f"{(noise['size'] if noise else 0)} noise points * "
        f"grand accept rate {100 * total_accept / max(1, total_labeled):.1f}%.\n"
    )

    # At-a-glance table -- scan all clusters without scrolling.
    lines.append("## At a glance\n")
    lines.append("| cluster | size | labeled | accept | heuristic | top feature |")
    lines.append("|---|---|---|---|---|---|")
    for s in real:
        top = s["distinguishing"][0] if s["distinguishing"] else ("--", 0.0, 0.0)
        top_str = f"`{top[0]}` ({top[1]:+.1f}sigma)"
        lines.append(
            f"| {s['cluster']} | {s['size']} | {s['n_labeled']} | "
            f"{_fmt_rate(s['accept_rate'])} | {_fmt_rate(s['heur_accept_rate'])} | "
            f"{top_str} |"
        )
    lines.append("")

    lines.append("## Clusters (sorted by accept rate, ascending)\n")
    for s in real:
        cid = s["cluster"]
        lines.append(
            f"### Cluster {cid} -- {s['size']} members * accept "
            f"{_fmt_rate(s['accept_rate'])} "
            f"({s['n_accept']}/{s['n_labeled']} labeled) * heuristic "
            f"{_fmt_rate(s['heur_accept_rate'])}\n"
        )
        lines.append(
            "**Most distinguishing features** (z vs grand mean, cluster raw mean):"
        )
        lines.append("")
        lines.append("| feature | z | raw mean |")
        lines.append("|---|---|---|")
        for name, z, _ in s["distinguishing"]:
            raw = s["raw_means"][name]
            lines.append(f"| `{name}` | {z:+.2f} | {raw:.3g} |")
        lines.append("")
        lines.append("**Exemplars** (paste into the viewer search bar):")
        for pid in s["exemplars"]:
            lines.append(f"- `{pid}`")
        lines.append("")

    if noise is not None and noise["size"] > 0:
        lines.append(
            f"## Noise bucket -- {noise['size']} members * accept "
            f"{_fmt_rate(noise['accept_rate'])} "
            f"({noise['n_accept']}/{noise['n_labeled']} labeled)\n"
        )
        lines.append(
            "These points didn't fit any dense cluster. A high accept rate "
            "here (vs the grand average) suggests under-clustering; a low "
            "rate suggests oddball rejects worth individual inspection.\n"
        )

    lines.append("## How to use this\n")
    lines.append(
        "1. Scan the top 3-5 clusters (lowest accept rate). If >95 % "
        "rejection and size > 100, that's a candidate auto-reject rule -- "
        "check the distinguishing features first, then verify 3 exemplars "
        "in the viewer to name the archetype.\n"
        "2. Scan the bottom 3-5 clusters (highest accept rate). Same logic "
        "in reverse: easy auto-accepts if they hold up under inspection.\n"
        "3. Any cluster where the heuristic accept rate diverges wildly "
        "from the human accept rate is a labeling-quality hotspot -- the "
        "heuristic is either systematically wrong or systematically right "
        "where the human wavered.\n"
    )

    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md} and {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(FEATURES_PATH))
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        default=80,
        help="HDBSCAN min_cluster_size; smaller -> more, finer archetypes",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="HDBSCAN min_samples; higher -> more conservative (more noise)",
    )
    ap.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="fraction of variance kept by PCA before clustering",
    )
    ap.add_argument("--output-md", default=str(REPORT_MD))
    ap.add_argument("--output-csv", default=str(REPORT_CSV))
    ap.add_argument(
        "--method",
        choices=["kmeans", "hdbscan"],
        default="kmeans",
        help="kmeans gives a fixed K with no noise; hdbscan finds density clusters",
    )
    ap.add_argument(
        "--n-clusters",
        type=int,
        default=20,
        help="KMeans cluster count (kmeans method only)",
    )
    ap.add_argument(
        "--selection",
        choices=["eom", "leaf"],
        default="leaf",
        help="HDBSCAN cluster_selection_method; leaf -> more, smaller clusters",
    )
    args = ap.parse_args()

    df_out, summaries = cluster(
        Path(args.features),
        method=args.method,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        pca_variance=args.pca_variance,
        n_clusters=args.n_clusters,
        cluster_selection_method=args.selection,
    )
    write_report(
        summaries, Path(args.output_md), Path(args.output_csv), df_out, args.method
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
