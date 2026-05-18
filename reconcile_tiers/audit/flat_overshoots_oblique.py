"""Diagnostic: FLAT_EMIT ceilings whose footprint covers area that, by the
geometry of *chosen* oblique planes nearby, should actually be sloped.

User-visible failure mode (example
``a8aca518-…::tier-ceiling-flat::14/1``): a FLAT_EMIT lid at ridge Y
extends past the gable; on the unmirrored side of the ridge the flat
blankets what should physically be the opposite slope of the gable.

Detection via chosen oblique plane extrapolation. For each FLAT_EMIT F at
Y = flat_y, polygon flat_xz:
1. For each chosen oblique O in the same payload (any story whose plane
   evaluated at the flat centroid gives Y near flat_y — i.e. the oblique's
   ridge passes through the flat's altitude):
2. Sample O's plane y over a grid inside flat_xz.
3. Where O's plane evaluates ``y_O > flat_y + clearance`` AND the sample
   point is OUTSIDE O's own XZ polygon, the flat is overshooting the
   oblique's ridge on the unmirrored side. The area there should be sloped
   (mirror of O across the ridge), not flat.

Outputs ``analysis_outputs/flat_overshoots_oblique_<ts>.csv`` with rows
per (flat, oblique, area-overshoot) triple.

Run::

    python -m reconcile_tiers.audit.flat_overshoots_oblique --all
    python -m reconcile_tiers.audit.flat_overshoots_oblique --uuid <uuid>
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from collections import Counter
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = WORKSPACE_ROOT / "pipeline-outputs"
OUT_DIR = WORKSPACE_ROOT / "analysis_outputs"

RIDGE_FLAT_TOL_M = 0.30
"""Centroid altitude tolerance: the oblique's plane must evaluate within this
of flat_y at the flat's centroid for the oblique's ridge to be "at the lid"."""

CLEARANCE_M = 0.30
"""Per-sample threshold: y_O(p) - flat_y must exceed this for the sample to
count as "the oblique would predict roof above the flat at p"."""

SAMPLE_STEP_M = 0.20
"""XZ sample step inside the flat polygon."""

MIN_FLAT_AREA_M2 = 0.5

MIN_OVERSHOOT_AREA_M2 = 0.30


def _xz_polygon(corners) -> Polygon | None:
    if len(corners) < 3:
        return None
    poly = Polygon([(c["x"], c["z"]) for c in corners])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly is None or poly.is_empty or poly.area <= 0:
        return None
    return poly


def _y_mean(corners) -> float | None:
    if not corners:
        return None
    return sum(float(c["y"]) for c in corners) / len(corners)


def _is_flat(piece: dict[str, Any]) -> bool:
    plane = piece.get("plane") or {}
    return abs(plane.get("b", 0.0)) >= 0.95 and (piece.get("source") == "flat_ceiling")


def _is_oblique(piece: dict[str, Any]) -> bool:
    plane = piece.get("plane") or {}
    b = plane.get("b", 1.0)
    src = (piece.get("source") or "").lower()
    if "oblique" in src:
        return True
    return abs(b) < 0.95


def _plane_y_at(plane: dict[str, Any], x: float, z: float) -> float | None:
    a = float(plane.get("a", 0.0))
    b = float(plane.get("b", 0.0))
    c = float(plane.get("c", 0.0))
    d = float(plane.get("d", 0.0))
    if abs(b) < 1e-9:
        return None
    return (d - a * x - c * z) / b


def _sample_grid(poly: Polygon, step: float) -> list[tuple[float, float]]:
    minx, minz, maxx, maxz = poly.bounds
    xs: list[float] = []
    zs: list[float] = []
    x = minx
    while x <= maxx:
        xs.append(x)
        x += step
    z = minz
    while z <= maxz:
        zs.append(z)
        z += step
    pts: list[tuple[float, float]] = []
    for x in xs:
        for z in zs:
            from shapely.geometry import Point

            if poly.contains(Point(x, z)):
                pts.append((x, z))
    return pts


def _classify_flat(
    flat: dict[str, Any],
    obliques: list[tuple[dict[str, Any], Polygon]],
) -> list[dict[str, Any]]:
    flat_poly = _xz_polygon(flat.get("corners") or [])
    if flat_poly is None or flat_poly.area < MIN_FLAT_AREA_M2:
        return [{"classification": "degenerate", "flat_area_xz_m2": 0.0}]
    flat_y = _y_mean(flat.get("corners") or [])
    if flat_y is None:
        return [{"classification": "degenerate"}]

    cx, cz = float(flat_poly.centroid.x), float(flat_poly.centroid.y)
    samples = _sample_grid(flat_poly, SAMPLE_STEP_M)
    if not samples:
        return [{"classification": "degenerate"}]
    sample_area = SAMPLE_STEP_M * SAMPLE_STEP_M

    rows: list[dict[str, Any]] = []
    any_ridge_oblique = False
    for ob, ob_poly in obliques:
        plane = ob.get("plane") or {}
        # "Ridge-at-lid" filter: the oblique's plane must evaluate within
        # RIDGE_FLAT_TOL_M of flat_y somewhere over the flat. Centroid is a
        # cheap proxy. Same b sign so we don't pair with floor-like planes.
        y_at_centroid = _plane_y_at(plane, cx, cz)
        if y_at_centroid is None:
            continue
        # The oblique's ridge altitude (highest y on the oblique) must be near
        # flat_y. We approximate this as max(y_at_centroid, max corner y).
        ob_y_max = max((float(c["y"]) for c in ob.get("corners") or []), default=None)
        if ob_y_max is None:
            continue
        if abs(ob_y_max - flat_y) > RIDGE_FLAT_TOL_M:
            continue
        any_ridge_oblique = True

        # Count flat samples that are (a) outside ob_poly and (b) where the
        # plane's extrapolated y > flat_y + clearance. These are "the oblique
        # would predict roof above the flat here, and there's no oblique
        # surface physically at this XZ" → unmirrored overshoot.
        from shapely.geometry import Point

        overshoot_samples = 0
        for x, z in samples:
            y_at = _plane_y_at(plane, x, z)
            if y_at is None:
                continue
            if y_at - flat_y < CLEARANCE_M:
                continue
            if ob_poly.contains(Point(x, z)):
                continue
            overshoot_samples += 1
        if overshoot_samples == 0:
            continue
        overshoot_area = overshoot_samples * sample_area
        if overshoot_area < MIN_OVERSHOOT_AREA_M2:
            continue
        rows.append(
            {
                "classification": "unmirrored_overshoot",
                "flat_area_xz_m2": float(flat_poly.area),
                "flat_y": flat_y,
                "oblique_locator_id": ob.get("locator_id"),
                "oblique_plane_b": plane.get("b"),
                "oblique_y_max": ob_y_max,
                "overshoot_area_m2": overshoot_area,
                "overshoot_ratio": overshoot_area / float(flat_poly.area),
            }
        )

    if rows:
        return rows
    if not any_ridge_oblique:
        return [
            {
                "classification": "no_ridge_oblique",
                "flat_area_xz_m2": float(flat_poly.area),
                "flat_y": flat_y,
            }
        ]
    return [
        {
            "classification": "ok",
            "flat_area_xz_m2": float(flat_poly.area),
            "flat_y": flat_y,
        }
    ]


def diagnose_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ceilings = payload.get("ceiling") or []
    flats = [c for c in ceilings if _is_flat(c)]
    obliques: list[tuple[dict[str, Any], Polygon]] = []
    for c in ceilings:
        if not _is_oblique(c):
            continue
        poly = _xz_polygon(c.get("corners") or [])
        if poly is None:
            continue
        obliques.append((c, poly))
    rows: list[dict[str, Any]] = []
    for flat in flats:
        for entry in _classify_flat(flat, obliques):
            entry["rule"] = "flat_overshoots_oblique"
            entry["locator_id"] = flat.get("locator_id")
            rows.append(entry)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--uuid", type=str)
    parser.add_argument("--root", type=Path, default=PIPELINE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.uuid:
        targets = [args.uuid]
    else:
        targets = sorted(p.parent.name for p in args.root.glob("*/tier_payload.json"))

    out_dir = args.out or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"flat_overshoots_oblique_{timestamp}.csv"

    counts: Counter[str] = Counter()
    by_uuid_overshoot: Counter[str] = Counter()
    by_uuid_area: dict[str, float] = {}
    rows_total = 0
    fields = [
        "uuid",
        "rule",
        "classification",
        "locator_id",
        "flat_area_xz_m2",
        "flat_y",
        "oblique_locator_id",
        "oblique_plane_b",
        "oblique_y_max",
        "overshoot_area_m2",
        "overshoot_ratio",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for uuid in targets:
            payload_path = args.root / uuid / "tier_payload.json"
            if not payload_path.exists():
                continue
            try:
                payload = json.loads(payload_path.read_text())
            except Exception:
                continue
            for row in diagnose_payload(payload):
                row["uuid"] = uuid
                writer.writerow(row)
                counts[row["classification"]] += 1
                if row["classification"] == "unmirrored_overshoot":
                    by_uuid_overshoot[uuid] += 1
                    by_uuid_area[uuid] = by_uuid_area.get(uuid, 0.0) + row.get(
                        "overshoot_area_m2", 0.0
                    )
                rows_total += 1

    print(f"\ncsv: {csv_path}")
    print(f"buildings: {len(targets)}, rows: {rows_total}\n")
    print(f"{'classification':<24} {'count':>6}")
    for cls, n in counts.most_common():
        print(f"{cls:<24} {n:>6}")
    if by_uuid_overshoot:
        print("\n=== top buildings by total overshoot area (m²) ===")
        ranked = sorted(by_uuid_area.items(), key=lambda kv: -kv[1])
        for uuid, total_area in ranked[:15]:
            n = by_uuid_overshoot[uuid]
            print(f"  {uuid[:8]} {n:>3} (flat,oblique) overshoots, {total_area:.1f} m²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
