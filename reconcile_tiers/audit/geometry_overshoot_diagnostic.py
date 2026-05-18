"""Plan H Phase 1: read-only diagnostic for geometry "sticking out" beyond
where it should be.

The Plan C audit fix uses the all-rooms buffer-union as the envelope. That's
right for catching pipeline output that's outside the BUILDING — but it's too
permissive for catching pieces that overshoot a SPECIFIC STORY. An oblique
that sits over a kicked-out wing on the wrong story is "in-envelope" globally
but "stuck out" locally.

This diagnostic computes per-piece XZ overshoot vs its **per-story** envelope
and classifies the cause:

- ``oblique_overshoot``  — oblique piece extends past its dominant story's
  footprint. Likely cause: ``roof/clipping.py`` clips against the global
  footprint, not the dominant story's.
- ``flat_overshoot``  — flat ceiling extends past its room's story envelope.
  Less common; usually indicates room/story misassignment.
- ``small_sliver``  — piece overshoots by <0.3 m^2 absolute and <8% of its
  area. Probably noise; report but separate.
- ``no_story_match``  — piece centroid fell outside every per-story envelope.
  Usually means the building has only one story and the per-story = global.
- ``in_envelope``  — passes (no overshoot beyond tolerance).

Run:

    python -m reconcile_tiers.audit.geometry_overshoot_diagnostic --all
    python -m reconcile_tiers.audit.geometry_overshoot_diagnostic --uuid <uuid>
    python -m reconcile_tiers.audit.geometry_overshoot_diagnostic --all --rating-cohort
    1,2

Outputs ``analysis_outputs/geometry_overshoot_<ts>.csv`` and a per-class
tally. Phase 2 (the fix) is gated on the tally — if ``oblique_overshoot``
dominates, we know to clip per-story in ``roof/clipping.py``.
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
from shapely.ops import unary_union

from reconcile_tiers.audit.rules import (
    _corners_xz,
    _room_floor_pieces,
    _safe_polygon,
    _y_range,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = WORKSPACE_ROOT / "pipeline-outputs"
OUT_DIR = WORKSPACE_ROOT / "analysis_outputs"

PER_STORY_BUFFER_M = 0.4
"""Soft buffer around each story's footprint. Pieces inside this band are not
counted as overshoot — they're at the eave/edge, where small overhangs are
expected."""

SMALL_SLIVER_AREA_M2 = 0.3
SMALL_SLIVER_RATIO = 0.08


def _per_story_envelopes(rooms: list[dict[str, Any]]) -> dict[int, Polygon]:
    by_story: dict[int, list[Polygon]] = {}
    for room in rooms:
        story = room.get("story")
        if story is None:
            continue
        for piece in _room_floor_pieces(room):
            corners = piece.get("corners") or []
            poly = _safe_polygon(_corners_xz(corners))
            if poly is None:
                continue
            by_story.setdefault(int(story), []).append(
                poly.buffer(PER_STORY_BUFFER_M, join_style="mitre")
            )
    out: dict[int, Polygon] = {}
    for story, polys in by_story.items():
        merged = unary_union(polys)
        if merged.is_empty:
            continue
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        out[story] = merged
    return out


def _piece_dominant_story(
    piece: dict[str, Any],
    bands: dict[int, tuple[float, float]],
) -> int | None:
    """Story whose Y-band contains the piece's Y midpoint."""
    cyr = _y_range(piece.get("corners") or [])
    if cyr is None:
        return None
    mid = 0.5 * (cyr[0] + cyr[1])
    for story in sorted(bands.keys()):
        lo, hi = bands[story]
        if lo - 0.5 <= mid <= hi + 0.5:
            return story
    return None


def _story_y_bands(rooms: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    bands: dict[int, list[tuple[float, float]]] = {}
    for room in rooms:
        story = room.get("story")
        if story is None:
            continue
        for wall in room.get("walls") or []:
            yr = _y_range(wall.get("corners") or [])
            if yr is None:
                continue
            bands.setdefault(int(story), []).append(yr)
    out: dict[int, tuple[float, float]] = {}
    for story, ranges in bands.items():
        if not ranges:
            continue
        out[story] = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    return out


def _classify_overshoot(
    piece: dict[str, Any],
    per_story_env: dict[int, Polygon],
    bands: dict[int, tuple[float, float]],
) -> tuple[str, dict[str, Any]]:
    corners = piece.get("corners") or []
    poly = _safe_polygon(_corners_xz(corners))
    if poly is None or poly.area <= 0:
        return ("degenerate_polygon", {})

    if not per_story_env:
        return ("no_story_match", {"piece_area_xz": float(poly.area)})

    story = _piece_dominant_story(piece, bands)
    envelope = per_story_env.get(story) if story is not None else None
    if envelope is None:
        # Fall back to the union of all stories — represents "the building".
        envelope = unary_union(list(per_story_env.values()))
        story = -1

    outside = poly.difference(envelope)
    outside_area = float(outside.area) if not outside.is_empty else 0.0
    ratio = outside_area / max(poly.area, 1e-9)
    evidence: dict[str, Any] = {
        "piece_area_xz": float(poly.area),
        "outside_area_xz": outside_area,
        "outside_ratio": ratio,
        "matched_story": story,
    }

    if outside_area <= 0.0:
        return ("in_envelope", evidence)
    if outside_area < SMALL_SLIVER_AREA_M2 and ratio < SMALL_SLIVER_RATIO:
        return ("small_sliver", evidence)

    source = (piece.get("source") or "").lower()
    plane = piece.get("plane") or {}
    plane_b = float(plane.get("b", 1.0))
    is_oblique = "oblique" in source or "merged_coplanar" in source or plane_b < 0.95
    return ("oblique_overshoot" if is_oblique else "flat_overshoot", evidence)


def diagnose_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = payload.get("rooms") or []
    per_story_env = _per_story_envelopes(rooms)
    bands = _story_y_bands(rooms)
    rows: list[dict[str, Any]] = []
    for piece in payload.get("ceiling") or []:
        classification, evidence = _classify_overshoot(piece, per_story_env, bands)
        if classification == "in_envelope":
            continue  # don't write 1000s of rows for the no-op case
        rows.append(
            {
                "rule": "geometry_overshoot",
                "locator_id": piece.get("locator_id"),
                "classification": classification,
                "source": piece.get("source"),
                "piece_area_xz_m2": evidence.get("piece_area_xz"),
                "outside_area_xz_m2": evidence.get("outside_area_xz"),
                "outside_ratio": evidence.get("outside_ratio"),
                "matched_story": evidence.get("matched_story"),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--uuid", type=str)
    parser.add_argument("--root", type=Path, default=PIPELINE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rating-cohort", type=str, default=None)
    args = parser.parse_args(argv)

    if args.uuid:
        targets = [args.uuid]
    else:
        targets = sorted(p.parent.name for p in args.root.glob("*/tier_payload.json"))
    if args.rating_cohort:
        ratings_path = WORKSPACE_ROOT / ".context" / "roof_ratings.json"
        if ratings_path.exists():
            ratings = json.loads(ratings_path.read_text())
            wanted = {s.strip() for s in args.rating_cohort.split(",") if s.strip()}
            targets = [
                u
                for u in targets
                if str((ratings.get(u) or {}).get("rating")) in wanted
            ]

    out_dir = args.out or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"geometry_overshoot_{timestamp}.csv"

    counts: Counter[tuple[str, str]] = Counter()
    rows_total = 0
    fields = [
        "uuid",
        "rule",
        "classification",
        "locator_id",
        "source",
        "piece_area_xz_m2",
        "outside_area_xz_m2",
        "outside_ratio",
        "matched_story",
    ]
    by_source = Counter()
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
                counts[(row["classification"], row.get("source") or "?")] += 1
                if row["classification"] in ("oblique_overshoot", "flat_overshoot"):
                    by_source[row.get("source") or "?"] += 1
                rows_total += 1

    print(f"\ncsv: {csv_path}")
    print(f"buildings: {len(targets)}, overshoot rows: {rows_total}\n")
    print(f"{'classification':<26} {'source':<24} {'count':>6}")
    for (cls, src), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{cls:<26} {src:<24} {n:>6}")
    if by_source:
        print("\n=== material overshoot by source ===")
        for src, n in by_source.most_common():
            print(f"  {src:<24} {n:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
