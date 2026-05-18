"""Render a static top-down figure for a given building's ceiling atoms, colored
by role and annotated with the "bisects gable" signature.

Writes a PNG to /tmp/flat_cap_cohort_<uuid>.png — use to eyeball whether the
flat caps my predicate flags are duplicating nearby sloped ceilings or are
legitimate attic soffits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon

RESULTS_PATH = Path("reconcile/roof_algorithms_py_results.json")

OVERLAP_FRACTION_MIN = 0.95
EAVE_MAX_ABOVE_CAP_M = 0.15
RIDGE_MIN_ABOVE_CAP_M = 0.40


def _poly_xz(corners):
    if not corners or len(corners) < 3:
        return None
    ring = [(float(c[0]), float(c[2])) for c in corners if len(c) >= 3]
    if len(ring) < 3:
        return None
    p = ShapelyPolygon(ring)
    if not p.is_valid:
        p = p.buffer(0)
    return p if (not p.is_empty and p.area > 0) else None


def _y_range(corners):
    ys = [float(c[1]) for c in corners if len(c) >= 3]
    return (min(ys), max(ys)) if ys else None


def bisects_gable(cap_poly, cap_y, oblique_records):
    best = None
    for ob in oblique_records:
        inter = cap_poly.intersection(ob["poly"])
        if inter.is_empty:
            continue
        ovr = inter.area / cap_poly.area
        if ovr < OVERLAP_FRACTION_MIN:
            continue
        eave = ob["y_min"] - cap_y
        ridge = ob["y_max"] - cap_y
        if eave > EAVE_MAX_ABOVE_CAP_M:
            continue
        if ridge < RIDGE_MIN_ABOVE_CAP_M:
            continue
        candidate = {"ovr": ovr, "eave": eave, "ridge": ridge}
        if best is None or candidate["ridge"] > best["ridge"]:
            best = candidate
    return best


def render(uuid: str, highlight_atom: str | None = None) -> Path:
    results = json.loads(RESULTS_PATH.read_text())
    b = results[uuid]
    oblique_surfaces = (b.get("roof_surfaces") or {}).get("oblique") or []
    oblique_records = []
    for s in oblique_surfaces:
        c = s.get("corners") or s.get("poly") or []
        p = _poly_xz(c)
        yr = _y_range(c)
        if p and yr:
            oblique_records.append({"poly": p, "y_min": yr[0], "y_max": yr[1]})

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax_f, ax_o = axes

    # flat partitions
    flat_cohort_count = 0
    flat_total_count = 0
    for partition in (b.get("ceiling_partitions") or {}).get("flat") or []:
        corners = partition.get("poly") or []
        cap_poly = _poly_xz(corners)
        if cap_poly is None:
            continue
        cap_y = float(partition.get("top_y_m") or 0.0)
        area = float(partition.get("area_m2") or 0.0)
        flat_total_count += 1
        signature = (
            bisects_gable(cap_poly, cap_y, oblique_records) if area >= 0.1 else None
        )
        if signature:
            flat_cohort_count += 1
        color = "#ef4444" if signature else "#60a5fa"
        alpha = 0.65 if signature else 0.35
        edge = "#b91c1c" if signature else "#1e40af"
        if partition.get("id") == highlight_atom:
            color = "#fbbf24"
            edge = "#000"
            alpha = 0.9
        coords = list(cap_poly.exterior.coords)
        patch = MplPolygon(
            coords,
            closed=True,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.5,
            alpha=alpha,
        )
        ax_f.add_patch(patch)
        cx, cy = cap_poly.centroid.x, cap_poly.centroid.y
        ax_f.text(
            cx,
            cy,
            f"{area:.1f}m²\ny={cap_y:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="#111",
        )

    # oblique partitions (ceiling side)
    for partition in (b.get("ceiling_partitions") or {}).get("oblique") or []:
        corners = partition.get("poly") or []
        p = _poly_xz(corners)
        if p is None:
            continue
        area = float(partition.get("area_m2") or 0.0)
        coords = list(p.exterior.coords)
        patch = MplPolygon(
            coords,
            closed=True,
            facecolor="#10b981",
            edgecolor="#065f46",
            linewidth=1.5,
            alpha=0.5,
        )
        ax_o.add_patch(patch)
        cx, cy = p.centroid.x, p.centroid.y
        ax_o.text(
            cx, cy, f"{area:.1f}m²", ha="center", va="center", fontsize=6, color="#111"
        )

    # oblique ROOF surfaces shown on both for context
    for ob in oblique_records:
        coords = list(ob["poly"].exterior.coords)
        for ax in (ax_f, ax_o):
            patch = MplPolygon(
                coords,
                closed=True,
                facecolor="none",
                edgecolor="#475569",
                linewidth=0.8,
                linestyle="--",
                alpha=0.8,
            )
            ax.add_patch(patch)

    for ax, title in [
        (
            ax_f,
            f"Flat ceiling partitions ({flat_cohort_count}/{flat_total_count} match "
            f"bisects-gable)",
        ),
        (ax_o, "Oblique ceiling partitions"),
    ]:
        ax.autoscale()
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(title, fontsize=10)

    legend_handles = [
        mpatches.Patch(
            color="#ef4444", label="flat cap: bisects-gable predicate match"
        ),
        mpatches.Patch(color="#60a5fa", label="flat cap: other"),
        mpatches.Patch(color="#10b981", label="oblique ceiling partition"),
        mpatches.Patch(
            facecolor="none", edgecolor="#475569", label="oblique roof surface (dashed)"
        ),
    ]
    if highlight_atom:
        legend_handles.append(
            mpatches.Patch(color="#fbbf24", label=f"user-reported: {highlight_atom}")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(f"Ceiling atoms (top-down) — {uuid}", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    out = Path(f"/tmp/flat_cap_cohort_{uuid[:8]}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("uuid")
    ap.add_argument("--highlight-atom", default=None)
    args = ap.parse_args()
    out = render(args.uuid, highlight_atom=args.highlight_atom)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
