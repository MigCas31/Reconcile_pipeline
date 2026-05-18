"""Coverage audit: what fraction of each building's footprint is covered
by the union of human-accepted V3 merged roof segments?

This is a classifier-independent audit. The classifier can only pick from
the proposer's output; no amount of scoring can recover coverage the
proposer never produced. If accepted proposals cover <X%% of the
footprint, a perfect classifier caps at ~X%% accuracy against ground truth.

Coverage here means XZ-plane area coverage: we take each accepted
segment's ``footprint_xz`` (already projected), union them, and divide by
the union of ``parts[*].footprint_xz`` (the true building silhouette).
Overlap between accepted segments is reported separately — high overlap
inflates raw covered area but doesn't add real footprint coverage.

Usage::

    python scripts/audit_coverage.py
    python scripts/audit_coverage.py --output reports/coverage_audit.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import ijson
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

REPO = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO / "reconcile" / "reconcile_v3_results.json"
LABELS_PATH = REPO / ".context" / "v3_roof_proposal_labels.jsonl"
DEFAULT_OUT = REPO / "reports" / "coverage_audit.csv"


def _load_last_write_labels(path: Path) -> dict[str, dict]:
    """Return the latest label per ``proposal_id`` (jsonl is append-only)."""
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("proposal_id")
            if not isinstance(pid, str):
                continue
            prev = out.get(pid)
            ts = r.get("ts") or ""
            if prev is None or ts >= (prev.get("ts") or ""):
                out[pid] = r
    return out


def _safe_poly(ring: list) -> Polygon | None:
    """Build a 2D polygon from either [x, z] pairs or [x, y, z] triples.

    The V3 pipeline stores ``footprint_xz`` as 3-tuples with y=0 despite
    the name, so we have to be defensive about both shapes.
    """
    if not ring or len(ring) < 3:
        return None
    try:
        sample = ring[0]
        if len(sample) >= 3:
            pts = [(float(p[0]), float(p[2])) for p in ring]
        else:
            pts = [(float(p[0]), float(p[1])) for p in ring]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area <= 1e-6:
            return None
        # make_valid can return GeometryCollection — keep the polygons only.
        if poly.geom_type == "Polygon":
            return poly
        if poly.geom_type == "MultiPolygon":
            return max(poly.geoms, key=lambda g: g.area)
        return None
    except Exception:
        return None


def _union(polys: list[Polygon]) -> Polygon | None:
    if not polys:
        return None
    try:
        u = unary_union(polys)
        if u.is_empty:
            return None
        return u
    except Exception:
        return None


def audit(results_path: Path, labels_path: Path) -> list[dict]:
    labels = _load_last_write_labels(labels_path)
    accepted_by_building: dict[str, set[str]] = {}
    rejected_by_building: dict[str, set[str]] = {}
    for pid, rec in labels.items():
        uuid = rec.get("building_uuid")
        if not isinstance(uuid, str):
            continue
        lbl = rec.get("label")
        if lbl == "accepted":
            accepted_by_building.setdefault(uuid, set()).add(pid)
        elif lbl == "rejected":
            rejected_by_building.setdefault(uuid, set()).add(pid)

    rows: list[dict] = []
    with results_path.open("rb") as f:
        for b in ijson.items(f, "item", use_float=True):
            uuid = b.get("building_uuid")
            if not isinstance(uuid, str):
                continue
            if uuid not in accepted_by_building and uuid not in rejected_by_building:
                continue  # not labeled

            # True footprint = union of parts[*].footprint_xz.
            part_polys: list[Polygon] = []
            for part in b.get("parts") or []:
                p = _safe_poly(part.get("footprint_xz") or [])
                if p is not None:
                    part_polys.append(p)
            footprint = _union(part_polys)
            if footprint is None or footprint.area <= 1e-6:
                continue

            accepts = accepted_by_building.get(uuid, set())
            rejects = rejected_by_building.get(uuid, set())
            accept_polys: list[Polygon] = []
            for seg in b.get("merged_roof_segments") or []:
                sid = seg.get("id")
                if sid in accepts:
                    p = _safe_poly(seg.get("footprint_xz") or [])
                    if p is not None:
                        # Clip accepted footprint to the building silhouette;
                        # otherwise over-reaching proposals get credit for
                        # covering thin air outside the building.
                        p = p.intersection(footprint)
                        if not p.is_empty and p.area > 1e-6:
                            accept_polys.append(p)

            covered = _union(accept_polys)
            covered_area = covered.area if covered is not None else 0.0

            # Pairwise overlap among accepts (how much accept area is
            # counted multiple times before union).
            raw_sum = sum(p.area for p in accept_polys)
            overlap_area = max(0.0, raw_sum - covered_area)

            rows.append(
                {
                    "building_uuid": uuid,
                    "address": b.get("address") or "",
                    "footprint_m2": float(footprint.area),
                    "covered_m2": float(covered_area),
                    "coverage_frac": float(covered_area / footprint.area),
                    "overlap_m2": float(overlap_area),
                    "overlap_frac_of_accepts": float(
                        overlap_area / raw_sum if raw_sum > 0 else 0.0
                    ),
                    "n_accepts": len(accepts),
                    "n_rejects": len(rejects),
                    "n_labels": len(accepts) + len(rejects),
                }
            )
    return rows


def summarize(rows: list[dict]) -> None:
    if not rows:
        print("no labeled buildings found in results JSON")
        return
    fracs = sorted(r["coverage_frac"] for r in rows)
    overlaps = sorted(r["overlap_frac_of_accepts"] for r in rows)

    def pct(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        idx = max(0, min(len(xs) - 1, round(q * (len(xs) - 1))))
        return xs[idx]

    n = len(rows)
    mean_frac = sum(fracs) / n
    print(f"\n=== coverage audit: {n} labeled buildings ===")
    print(f"coverage_frac  mean={mean_frac:.3f} median={statistics.median(fracs):.3f}")
    print(
        f"  p05={pct(fracs, 0.05):.3f}  p25={pct(fracs, 0.25):.3f}  "
        f"p75={pct(fracs, 0.75):.3f}  p95={pct(fracs, 0.95):.3f}"
    )
    below_50 = sum(1 for f in fracs if f < 0.5)
    below_80 = sum(1 for f in fracs if f < 0.8)
    at_or_above_95 = sum(1 for f in fracs if f >= 0.95)
    print(
        f"  <50%: {below_50} ({100 * below_50 / n:.1f}%)  "
        f"<80%: {below_80} ({100 * below_80 / n:.1f}%)  "
        f">=95%: {at_or_above_95} ({100 * at_or_above_95 / n:.1f}%)"
    )
    print(
        f"overlap_frac   mean={sum(overlaps) / n:.3f}  "
        f"median={statistics.median(overlaps):.3f}  "
        f"p95={pct(overlaps, 0.95):.3f}"
    )
    worst = sorted(rows, key=lambda r: r["coverage_frac"])[:5]
    print("\nworst coverage (for visual inspection in the viewer):")
    for r in worst:
        print(
            f"  {r['coverage_frac']:.2%}  {r['building_uuid']}  "
            f"footprint={r['footprint_m2']:.0f}m² covered={r['covered_m2']:.0f}m² "
            f"accepts={r['n_accepts']}/{r['n_labels']}  {r['address'][:50]}"
        )
    best = sorted(rows, key=lambda r: -r["coverage_frac"])[:3]
    print("\nbest coverage:")
    for r in best:
        print(
            f"  {r['coverage_frac']:.2%}  {r['building_uuid']}  "
            f"accepts={r['n_accepts']}/{r['n_labels']}  {r['address'][:50]}"
        )


def _write_csv(rows: list[dict], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "building_uuid",
        "address",
        "footprint_m2",
        "covered_m2",
        "coverage_frac",
        "overlap_m2",
        "overlap_frac_of_accepts",
        "n_accepts",
        "n_rejects",
        "n_labels",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        # Sort ascending by coverage so the worst cases surface first.
        for r in sorted(rows, key=lambda r: r["coverage_frac"]):
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS_PATH))
    ap.add_argument("--labels", default=str(LABELS_PATH))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    results_path = Path(args.results)
    labels_path = Path(args.labels)
    out_path = Path(args.output)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        return 2
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found", file=sys.stderr)
        return 2

    rows = audit(results_path, labels_path)
    summarize(rows)
    _write_csv(rows, out_path)
    print(f"\nwrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
