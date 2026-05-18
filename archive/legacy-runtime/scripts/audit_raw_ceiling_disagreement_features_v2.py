"""Follow-up to ``audit_raw_ceiling_disagreement_features.py`` — adds
room-size, top-floor / ground-floor, and inner / outer room attributes.

Attributes added per room:

* ``is_top_story``        — room.story == max(stories in this building).
* ``is_ground_story``     — story == 0 AND building has ≥ 2 stories (for
  single-story buildings the room is simultaneously "top" and "ground"; this
  flag only marks ground-with-upper).
* ``is_only_story``       — building has exactly one story.
* ``exterior_edge_frac``  — share of floor-polygon perimeter (XZ) that is
  NOT coincident with any other same-story room's floor polygon (small 0.10 m
  tolerance). 1.0 = island room, 0.0 = fully surrounded.
* ``perimeter_m``         — floor XZ perimeter.
* ``max_floor_dim_m``     — length of the longest edge of the axis-aligned
  bounding box in XZ.
* ``aspect_ratio``        — max_dim / min_dim of the AABB.
* Bucketed versions of ``floor_area_m2`` using tighter edges.

Reads disagreement classes from
``reports/raw_ceiling_pipeline_gap/per_plane.csv``. Outputs
``reports/raw_ceiling_disagreement_features_v2/`` with per-room CSV and
summary JSON.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
PER_PLANE_CSV = REPO / "reports" / "raw_ceiling_pipeline_gap" / "per_plane.csv"
OUT_DIR = REPO / "reports" / "raw_ceiling_disagreement_features_v2"

DISAGREE_CLASSES = {
    "oblique_wrong_orientation",
    "flat_covered_by_oblique",
    "oblique_covered_by_flat",
}
ADJ_TOL_M = 0.10


def _safe_polygon(fp):
    pts = [(p[0], p[2]) for p in fp]
    if len(pts) < 3:
        return None
    p = Polygon(pts)
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty or not p.is_valid:
        return None
    return p


def _perimeter_features(fp):
    pts = [(p[0], p[2]) for p in fp]
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    w, h = max(xs) - min(xs), max(zs) - min(zs)
    max_dim = max(w, h)
    min_dim = max(min(w, h), 1e-6)
    perim = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        perim += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    return perim, max_dim, max_dim / min_dim


def _exterior_fraction(fp, others_union) -> float:
    if others_union is None or others_union.is_empty:
        return 1.0
    pts = [(p[0], p[2]) for p in fp]
    if len(pts) < 3:
        return 0.0
    buf = others_union.buffer(ADJ_TOL_M)
    total = 0.0
    shared = 0.0
    for i in range(len(pts)):
        seg = LineString([pts[i], pts[(i + 1) % len(pts)]])
        length = seg.length
        if length < 1e-6:
            continue
        total += length
        try:
            inter = seg.intersection(buf)
        except Exception:
            continue
        if not inter.is_empty:
            shared += inter.length
    if total <= 0:
        return 1.0
    return max(0.0, 1.0 - shared / total)


def _load_per_plane():
    by_room = defaultdict(list)
    with PER_PLANE_CSV.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["uuid"], int(row["story"]), int(row["room_index"]))
            by_room[key].append(row)
    return by_room


def _build_rows():
    with BUILDINGS_PATH.open() as f:
        buildings = json.load(f)
    per_plane = _load_per_plane()

    rows: list[dict] = []
    for bldg in buildings:
        uuid = bldg["uuid"]
        rooms = bldg.get("rooms") or []
        if not rooms:
            continue
        stories = [int(r.get("story", 0)) for r in rooms]
        max_story = max(stories)
        min_story = min(stories)
        n_stories = max_story - min_story + 1

        # Build per-story union of other rooms' floor polygons.
        polys_by_story: dict[int, list[tuple[int, Polygon]]] = defaultdict(list)
        for ri, room in enumerate(rooms):
            poly = _safe_polygon(room.get("floor_polygon") or [])
            if poly is None:
                continue
            polys_by_story[int(room.get("story", 0))].append((ri, poly))

        for ri, room in enumerate(rooms):
            story = int(room.get("story", 0))
            key = (uuid, story, ri)
            if key not in per_plane:
                continue
            plane_rows = per_plane[key]
            fp = room.get("floor_polygon") or []
            if len(fp) < 3:
                continue

            others = [p for rr, p in polys_by_story.get(story, []) if rr != ri]
            if others:
                try:
                    others_union = unary_union([p.buffer(0) for p in others])
                except Exception:
                    others_union = None
            else:
                others_union = None
            ext_frac = _exterior_fraction(fp, others_union)

            perim, max_dim, aspect = _perimeter_features(fp)
            poly_self = _safe_polygon(fp)
            area = float(poly_self.area) if poly_self else 0.0

            n_disagree = sum(
                1 for r in plane_rows if r["classification"] in DISAGREE_CLASSES
            )
            n_planes = len(plane_rows)

            rows.append(
                {
                    "uuid": uuid,
                    "story": story,
                    "room_index": ri,
                    "n_planes_scored": n_planes,
                    "n_disagree": n_disagree,
                    "disagreement_rate": round(n_disagree / n_planes, 3)
                    if n_planes
                    else 0.0,
                    "building_n_stories": n_stories,
                    "is_top_story": story == max_story,
                    "is_ground_story": story == 0 and max_story > 0,
                    "is_only_story": n_stories == 1,
                    "exterior_edge_frac": round(ext_frac, 3),
                    "perimeter_m": round(perim, 2),
                    "max_floor_dim_m": round(max_dim, 2),
                    "floor_area_m2": round(area, 2),
                    "aspect_ratio": round(aspect, 2),
                }
            )

    return rows


def _bucket(value, edges, labels):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return labels[i]
    return labels[-1]


def _weighted_rate(rows):
    n = sum(r["n_planes_scored"] for r in rows)
    d = sum(r["n_disagree"] for r in rows)
    return d, n, (d / n if n else 0.0)


def _analyze(rows):
    corpus = _weighted_rate(rows)
    report = {
        "corpus": {
            "n_planes": corpus[1],
            "n_disagree": corpus[0],
            "rate": round(corpus[2], 4),
        },
        "bool_features": {},
        "numeric_features": {},
        "cross_features": {},
    }

    bool_feats = ["is_top_story", "is_ground_story", "is_only_story"]
    for feat in bool_feats:
        pos = [r for r in rows if r[feat]]
        neg = [r for r in rows if not r[feat]]
        dp, np_, rp = _weighted_rate(pos)
        dn, nn, rn = _weighted_rate(neg)
        report["bool_features"][feat] = {
            "true": {
                "rooms": len(pos),
                "planes": np_,
                "disagree": dp,
                "rate": round(rp, 4),
            },
            "false": {
                "rooms": len(neg),
                "planes": nn,
                "disagree": dn,
                "rate": round(rn, 4),
            },
            "relative_risk": round(rp / rn, 2) if rn else None,
        }

    num_specs = {
        "exterior_edge_frac": (
            [0, 0.2, 0.5, 0.8, 1.0001],
            ["<0.2_interior", "0.2-0.5", "0.5-0.8", "0.8-1.0_exterior"],
        ),
        "floor_area_m2": (
            [0, 5, 10, 20, 40, 99999],
            ["<5", "5-10", "10-20", "20-40", "40+"],
        ),
        "perimeter_m": (
            [0, 10, 15, 25, 40, 9999],
            ["<10", "10-15", "15-25", "25-40", "40+"],
        ),
        "max_floor_dim_m": (
            [0, 3, 5, 8, 12, 9999],
            ["<3", "3-5", "5-8", "8-12", "12+"],
        ),
        "aspect_ratio": (
            [1, 1.3, 2, 3, 999],
            ["<1.3_square", "1.3-2", "2-3", "3+_elongated"],
        ),
        "building_n_stories": ([1, 2, 3, 4, 999], ["1", "2", "3", "4+"]),
    }

    for feat, (edges, labels) in num_specs.items():
        buckets = defaultdict(list)
        for r in rows:
            buckets[_bucket(r[feat], edges, labels)].append(r)
        out = {}
        for label in labels:
            bucket = buckets.get(label, [])
            d, n, rate = _weighted_rate(bucket)
            out[label] = {
                "rooms": len(bucket),
                "planes": n,
                "disagree": d,
                "rate": round(rate, 4),
            }
        rates = [v["rate"] for v in out.values() if v["planes"] >= 20]
        out["_spread"] = round(max(rates) - min(rates), 4) if len(rates) >= 2 else 0.0
        report["numeric_features"][feat] = out

    # cross: exterior_frac x is_top_story
    edges = [0, 0.5, 0.8, 1.0001]
    lab = ["interior<0.5", "mixed0.5-0.8", "exterior>0.8"]
    cross = {}
    for r in rows:
        eb = _bucket(r["exterior_edge_frac"], edges, lab)
        tb = "top" if r["is_top_story"] else "not_top"
        cross.setdefault((eb, tb), []).append(r)
    cross_out = {}
    for (eb, tb), bucket in cross.items():
        d, n, rate = _weighted_rate(bucket)
        cross_out[f"{eb} & {tb}"] = {
            "rooms": len(bucket),
            "planes": n,
            "disagree": d,
            "rate": round(rate, 4),
        }
    report["cross_features"]["exterior_x_top_story"] = cross_out

    # cross: building_n_stories x is_top_story
    cross2 = {}
    for r in rows:
        ns = r["building_n_stories"]
        tb = "top" if r["is_top_story"] else "not_top"
        key = f"{ns}-story_{tb}"
        cross2.setdefault(key, []).append(r)
    cross2_out = {}
    for key, bucket in sorted(cross2.items()):
        d, n, rate = _weighted_rate(bucket)
        cross2_out[key] = {
            "rooms": len(bucket),
            "planes": n,
            "disagree": d,
            "rate": round(rate, 4),
        }
    report["cross_features"]["n_stories_x_top_story"] = cross2_out

    return report


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print(report):
    c = report["corpus"]
    print(f"Corpus: {c['n_planes']} planes, rate={c['rate']}\n")
    print("Bool features:")
    for feat, d in report["bool_features"].items():
        t, f = d["true"], d["false"]
        print(
            f"  {feat:<20} true: rooms={t['rooms']:>4} planes={t['planes']:>4} "
            f"rate={t['rate']:.3f}   "
            f"false: rooms={f['rooms']:>4} planes={f['planes']:>4} "
            f"rate={f['rate']:.3f}   RR={d['relative_risk']}"
        )
    print("\nNumeric features:")
    for feat, d in sorted(
        report["numeric_features"].items(), key=lambda kv: -kv[1].get("_spread", 0)
    ):
        print(f"  {feat} (spread={d.get('_spread', 0):.3f}):")
        for label, info in d.items():
            if label.startswith("_"):
                continue
            print(
                f"    {label:<24} rooms={info['rooms']:>4} planes={info['planes']:>4} "
                f"rate={info['rate']:.3f}"
            )
    print("\nCross features:")
    for name, buckets in report["cross_features"].items():
        print(f"  {name}:")
        for key, info in sorted(buckets.items()):
            print(
                f"    {key:<30} rooms={info['rooms']:>4} planes={info['planes']:>4} "
                f"rate={info['rate']:.3f}"
            )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _build_rows()
    _write_csv(OUT_DIR / "per_room.csv", rows)
    report = _analyze(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    _print(report)


if __name__ == "__main__":
    main()
