"""Audits 1 + 2: cohort with concave footprints + underbuilt roofs, and
per-signal availability across all buildings.

Reads `reconcile/roof_algorithms_py_results.json` once (streaming via ijson)
and `pipeline-outputs/<uuid>/reconciled.json` per building. Emits a CSV-ish
summary to stdout and a JSON detail file for follow-up analysis.

Signals measured per building:
  - footprint_concavity   : 1 - polygon_area / convex_hull_area (proxy for L/U/T)
  - oblique_surfaces      : count of committed oblique surfaces
  - oblique_azimuth_bins  : distinct 30 deg bins among oblique surfaces (azimuth mod
  180)
  - valid_clusters        : oblique clusters that survived clustering
  - dropped_clusters      : clusters that survived clustering but produced no surface
                            (proxy for the 2 m ridge-extent gate or downstream drops)
  - cluster_azimuth_bins  : distinct 30 deg bins among valid_clusters
  - ceiling_planes        : count of candidate ceiling planes
  - has_slanted_ceiling   : whether simple-slant or oblique ceiling fragments exist
  - stories               : story count from reconciled.json
  - story_polygons        : how many stories have non-trivial polygons
  - floor_delta_present   : whether lower-story polygon clearly differs from upper
  - segment_azimuth_bins  : distinct 30 deg bins among raw oblique segments

The "underbuilt" cohort is buildings where footprint_concavity is high but
oblique_surfaces is small or all surfaces share a single azimuth bin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ijson

REPO = Path(__file__).resolve().parent.parent
ROOF_RESULTS = REPO / "reconcile" / "roof_algorithms_py_results.json"
PIPELINE_OUT = REPO / "pipeline-outputs"
DETAIL_OUT = REPO / ".context" / "roof_signal_availability.json"

AZIMUTH_BIN_DEG = 30.0


def azimuth_bin(az: float) -> int:
    a = float(az) % 180.0
    return int(a // AZIMUTH_BIN_DEG)


def polygon_area_xz(corners: list) -> float:
    if not corners or len(corners) < 3:
        return 0.0
    a = 0.0
    n = len(corners)
    for i in range(n):
        x1, _, z1 = (
            corners[i][0],
            corners[i][1] if len(corners[i]) > 1 else 0.0,
            corners[i][-1],
        )
        x2, _, z2 = corners[(i + 1) % n][0], 0.0, corners[(i + 1) % n][-1]
        a += x1 * z2 - x2 * z1
    return abs(a) * 0.5


def convex_hull_xz(points: list) -> list:
    pts = sorted({(float(p[0]), float(p[-1])) for p in points})
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_area(pts: list) -> float:
    if len(pts) < 3:
        return 0.0
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, z1 = pts[i]
        x2, z2 = pts[(i + 1) % n]
        a += x1 * z2 - x2 * z1
    return abs(a) * 0.5


def concavity_ratio(footprint_corners: list) -> float:
    if not footprint_corners or len(footprint_corners) < 3:
        return 0.0
    pts = [(float(c[0]), float(c[-1])) for c in footprint_corners]
    poly_a = polygon_area_xz([[x, 0.0, z] for x, z in pts])
    hull = convex_hull_xz([[x, 0.0, z] for x, z in pts])
    h_a = hull_area(hull)
    if h_a <= 0:
        return 0.0
    return max(0.0, 1.0 - poly_a / h_a)


def extract_footprint(rec: dict) -> list:
    ceil = rec.get("ceiling") or {}
    fp = ceil.get("footprint")
    if isinstance(fp, list) and fp:
        return fp
    return []


def reconciled_for(uuid: str) -> dict | None:
    rj = PIPELINE_OUT / uuid / "reconciled.json"
    if not rj.exists():
        return None
    try:
        with rj.open() as f:
            return json.load(f)
    except Exception:
        return None


def story_signals(reconciled: dict | None) -> dict:
    if not reconciled:
        return {
            "stories": 0,
            "story_polygons_with_data": 0,
            "floor_delta_present": False,
            "floor_delta_max_pct": 0.0,
            "cross_floor_gap_count": 0,
            "cross_floor_gap_total_m2": 0.0,
        }
    sf = reconciled.get("story_footprints") or []
    areas = [float(s.get("area_m2") or 0.0) for s in sf if s.get("footprint_wkt")]
    polygons_with_data = sum(1 for s in sf if s.get("footprint_wkt"))
    delta_max_pct = 0.0
    if len([a for a in areas if a > 0]) >= 2:
        positive = [a for a in areas if a > 0]
        lo, hi = min(positive), max(positive)
        if hi > 0:
            delta_max_pct = (hi - lo) / hi
    gaps = reconciled.get("cross_floor_gaps") or []
    gap_total = sum(float(g.get("area_m2") or 0.0) for g in gaps)
    return {
        "stories": len(sf),
        "story_polygons_with_data": polygons_with_data,
        "floor_delta_present": delta_max_pct > 0.10 or gap_total > 1.0,
        "floor_delta_max_pct": round(delta_max_pct, 3),
        "cross_floor_gap_count": len(gaps),
        "cross_floor_gap_total_m2": round(gap_total, 2),
    }


def summarize(uuid: str, rec: dict) -> dict:
    rs = rec.get("roof_surfaces") or {}
    obliques = rs.get("oblique") or []
    flats = rs.get("flat") or []
    clusters = rec.get("valid_clusters") or []
    ceil = rec.get("ceiling") or {}
    planes = ceil.get("planes") or []
    simple_slant = ceil.get("simple_slant") or []
    ceil_oblique = ceil.get("oblique") or []
    segments = rec.get("segments") or []

    ob_az_bins: set[int] = set()
    ob_az_list = []
    for o in obliques:
        cl = o.get("cluster") or {}
        az = cl.get("avgAzimuth")
        if az is None:
            continue
        ob_az_bins.add(azimuth_bin(az))
        ob_az_list.append(round(float(az), 2))

    cl_az_bins: set[int] = set()
    cl_az_list = []
    for c in clusters:
        az = c.get("avgAzimuth")
        if az is None:
            continue
        cl_az_bins.add(azimuth_bin(az))
        cl_az_list.append(round(float(az), 2))

    seg_az_bins: set[int] = set()
    for s in segments:
        az = s.get("azimuth")
        if az is None:
            continue
        seg_az_bins.add(azimuth_bin(az))

    fp = extract_footprint(rec)
    concavity = concavity_ratio(fp) if fp else 0.0
    story_sig = story_signals(reconciled_for(uuid))

    out = {
        "uuid": uuid,
        "concavity": round(concavity, 3),
        "oblique_surfaces": len(obliques),
        "oblique_az_bins": len(ob_az_bins),
        "oblique_az": ob_az_list,
        "valid_clusters": len(clusters),
        "cluster_az_bins": len(cl_az_bins),
        "cluster_az": cl_az_list,
        "dropped_clusters": max(0, len(clusters) - len(obliques)),
        "flat_surfaces": len(flats),
        "ceiling_planes": len(planes),
        "simple_slant": len(simple_slant) if isinstance(simple_slant, list) else 0,
        "ceiling_oblique": len(ceil_oblique) if isinstance(ceil_oblique, list) else 0,
        "has_slanted_ceiling": bool(simple_slant) or bool(ceil_oblique),
        "segments": len(segments),
        "segment_az_bins": len(seg_az_bins),
    }
    out.update(story_sig)
    return out


def main() -> int:
    if not ROOF_RESULTS.exists():
        print(f"missing {ROOF_RESULTS}", file=sys.stderr)
        return 1
    DETAIL_OUT.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    print("processing... (streaming 347 MB)", file=sys.stderr)
    with ROOF_RESULTS.open("rb") as f:
        for uuid, rec in ijson.kvitems(f, "", use_float=True):
            try:
                summaries.append(summarize(uuid, rec))
            except Exception as e:
                print(f"warn {uuid}: {e}", file=sys.stderr)
            if len(summaries) % 25 == 0:
                print(f"  {len(summaries)} buildings", file=sys.stderr)

    print(f"total: {len(summaries)} buildings", file=sys.stderr)
    DETAIL_OUT.write_text(json.dumps(summaries, indent=1))
    print(f"detail written: {DETAIL_OUT}", file=sys.stderr)

    n = len(summaries)
    concave = [s for s in summaries if s["concavity"] >= 0.10]
    very_concave = [s for s in summaries if s["concavity"] >= 0.20]

    underbuilt = [
        s
        for s in summaries
        if s["concavity"] >= 0.10
        and (s["oblique_surfaces"] <= 1 or s["oblique_az_bins"] <= 1)
    ]
    underbuilt_strong = [
        s for s in summaries if s["concavity"] >= 0.20 and s["oblique_az_bins"] <= 1
    ]

    cluster_drop_pattern = [
        s for s in summaries if s["dropped_clusters"] > 0 and s["concavity"] >= 0.10
    ]

    print()
    print("=" * 70)
    print(f"AUDIT 1 — concave footprint + underbuilt roof cohort   (n={n})")
    print("=" * 70)
    print(f"concave (>=0.10): {len(concave)} ({100 * len(concave) / n:.1f}%)")
    print(
        f"very concave (>=0.20): {len(very_concave)} "
        f"({100 * len(very_concave) / n:.1f}%)"
    )
    print()
    print("UNDERBUILT (concave >=0.10 AND obliques<=1 OR single az bin):")
    print(f"  {len(underbuilt)} ({100 * len(underbuilt) / n:.1f}%)")
    print("UNDERBUILT STRONG (concave >=0.20 AND single az bin):")
    print(f"  {len(underbuilt_strong)} ({100 * len(underbuilt_strong) / n:.1f}%)")
    print()
    print("DROPPED CLUSTERS (cluster survived but no surface) on concave footprints:")
    print(f"  {len(cluster_drop_pattern)} ({100 * len(cluster_drop_pattern) / n:.1f}%)")

    print()
    print("=" * 70)
    print("AUDIT 2 — per-signal availability")
    print("=" * 70)
    sig_floor_delta = sum(1 for s in summaries if s["floor_delta_present"])
    sig_slanted_ceiling = sum(1 for s in summaries if s["has_slanted_ceiling"])
    sig_ceiling_planes = sum(1 for s in summaries if s["ceiling_planes"] > 0)
    sig_multi_story = sum(1 for s in summaries if s["stories"] >= 2)
    sig_cross_floor_gaps = sum(1 for s in summaries if s["cross_floor_gap_count"] > 0)
    sig_segments = sum(1 for s in summaries if s["segments"] > 0)
    sig_multi_az_clusters = sum(1 for s in summaries if s["cluster_az_bins"] >= 2)

    print(
        f"floor_delta_present:        {sig_floor_delta}/{n}  "
        f"({100 * sig_floor_delta / n:.1f}%)"
    )
    print(
        f"has_slanted_ceiling:        {sig_slanted_ceiling}/{n}  "
        f"({100 * sig_slanted_ceiling / n:.1f}%)"
    )
    print(
        f"ceiling_planes > 0:         {sig_ceiling_planes}/{n}  "
        f"({100 * sig_ceiling_planes / n:.1f}%)"
    )
    print(
        f"multi-story (>=2):          {sig_multi_story}/{n}  "
        f"({100 * sig_multi_story / n:.1f}%)"
    )
    print(
        f"cross_floor_gaps present:   {sig_cross_floor_gaps}/{n}  "
        f"({100 * sig_cross_floor_gaps / n:.1f}%)"
    )
    print(
        f"oblique segments present:   {sig_segments}/{n}  "
        f"({100 * sig_segments / n:.1f}%)"
    )
    print(
        f"multi-azimuth clusters:     {sig_multi_az_clusters}/{n}  "
        f"({100 * sig_multi_az_clusters / n:.1f}%)"
    )

    print()
    print("=" * 70)
    print("UNDERBUILT EXAMPLES (top 20 by concavity)")
    print("=" * 70)
    examples = sorted(underbuilt, key=lambda s: s["concavity"], reverse=True)[:20]
    for s in examples:
        print(
            f"  {s['uuid']}  concavity={s['concavity']:.2f}  "
            f"obliques={s['oblique_surfaces']}  ob_az_bins={s['oblique_az_bins']}  "
            f"valid_clusters={s['valid_clusters']}  "
            f"cluster_az_bins={s['cluster_az_bins']}  "
            f"dropped={s['dropped_clusters']}  "
            f"slanted_ceiling={s['has_slanted_ceiling']}  "
            f"floor_delta={s['floor_delta_present']}"
        )

    target = "74e87bcd-3989-4d5c-8f16-f7782dc3afbd"
    print()
    print("=" * 70)
    print(f"TARGET BUILDING {target}")
    print("=" * 70)
    for s in summaries:
        if s["uuid"] == target:
            for k, v in s.items():
                print(f"  {k}: {v}")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
