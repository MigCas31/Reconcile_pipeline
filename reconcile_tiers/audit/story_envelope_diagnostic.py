"""Read-only diagnostic for `out_of_envelope` and `ceiling_below_floor` flags.

For each flag, classify the upstream step likely responsible. Phase 1 of
Plan C — the source-step distribution decides what to fix in Phase 2.

Run:

    python -m reconcile_tiers.audit.story_envelope_diagnostic --all
    python -m reconcile_tiers.audit.story_envelope_diagnostic --uuid <uuid>

Outputs a CSV at ``analysis_outputs/story_envelope_diag_<ts>.csv`` and prints
a per-classification tally to stdout.

Background — there are **two** notions of "building envelope" in play:

1. ``audit/rules.py`` (``_building_envelope``): convex hull of **ground-story
   rooms only**, buffered.
2. ``roof/footprint.py`` (``build_building_footprint``): buffer-union of
   **all rooms across all stories**, then shrunk.

For multi-story buildings these disagree, so a piece can be flagged
``out_of_envelope`` by the audit while the pipeline considers it in-bounds.
The diagnostic separates that case (audit envelope is too tight) from real
pipeline overshoot.
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
    rule_ceiling_below_floor,
    rule_out_of_envelope,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = WORKSPACE_ROOT / "pipeline-outputs"
OUT_DIR = WORKSPACE_ROOT / "analysis_outputs"

# Slightly looser than the audit's buffer so we don't double-count buffer-edge
# pieces as overshoot.
PER_STORY_TOLERANCE_M = 0.4


def _per_story_envelopes(rooms: list[dict[str, Any]]) -> dict[int, Polygon]:
    """Per-story buffer-union of room floors, mirroring the pipeline's
    `build_building_footprint` shape but emitted per story so we can
    distinguish ground-only vs all-rooms envelope mismatches."""
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
                poly.buffer(0.3, join_style="mitre")
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


def _all_rooms_envelope(rooms: list[dict[str, Any]]) -> Polygon | None:
    """Pipeline-style envelope: union of all-rooms buffered floors."""
    polys = []
    for room in rooms:
        for piece in _room_floor_pieces(room):
            corners = piece.get("corners") or []
            poly = _safe_polygon(_corners_xz(corners))
            if poly is not None:
                polys.append(poly.buffer(0.3, join_style="mitre"))
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.is_empty:
        return None
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    return merged


def _piece_centroid_xz(corners) -> tuple[float, float] | None:
    poly = _safe_polygon(_corners_xz(corners))
    if poly is None:
        return None
    cx, cz = poly.representative_point().coords[0]
    return float(cx), float(cz)


def _classify_out_of_envelope(
    flag: dict[str, Any],
    payload: dict[str, Any],
    per_story_env: dict[int, Polygon],
    all_rooms_env: Polygon | None,
) -> str:
    """Decide whether the audit envelope is too tight, the pipeline clipped
    too aggressively, or something else entirely."""
    target_locator = flag.get("locator")
    target = _find_piece(target_locator, payload)
    if target is None:
        return "unknown_target"

    centroid = _piece_centroid_xz(target.get("corners") or [])
    if centroid is None:
        return "degenerate_polygon"
    point = (centroid[0], centroid[1])

    # If the piece centroid is inside the all-rooms-pipeline envelope but
    # outside the audit's ground-only envelope, the audit is the problem.
    inside_pipeline = all_rooms_env is not None and all_rooms_env.buffer(
        PER_STORY_TOLERANCE_M
    ).contains(_pt(point))
    if inside_pipeline:
        return "audit_envelope_too_tight"

    # Otherwise the piece really is outside even the all-rooms envelope.
    # Check if it sits over a non-ground story — then the issue is that the
    # pipeline never extended the footprint into that story.
    target_story = _piece_story(target, per_story_env)
    if target_story is not None and target_story > min(per_story_env.keys() or [0]):
        return "pipeline_per_story_missing"

    # Oblique pieces overshoot via roof clipping — different fix surface.
    source = (target.get("source") or "").lower()
    if "oblique" in source or target.get("plane", {}).get("b", 1.0) < 0.95:
        return "oblique_clip_overshoot"

    return "pipeline_clip_overshoot"


def _classify_ceiling_below_floor(
    flag: dict[str, Any],
    payload: dict[str, Any],
    rooms: list[dict[str, Any]],
) -> str:
    """Classify why the ceiling sits below the floor."""
    target = _find_piece(flag.get("locator"), payload)
    if target is None:
        return "unknown_target"

    cyr = _y_range(target.get("corners") or [])
    if cyr is None:
        return "degenerate_polygon"

    # Story Y-bands derived the same way the audit does.
    story_bands: dict[int, tuple[float, float]] = {}
    for room in rooms:
        story = room.get("story")
        if story is None:
            continue
        for wall in room.get("walls") or []:
            wyr = _y_range(wall.get("corners") or [])
            if wyr is None:
                continue
            existing = story_bands.get(int(story))
            if existing is None:
                story_bands[int(story)] = wyr
            else:
                story_bands[int(story)] = (
                    min(existing[0], wyr[0]),
                    max(existing[1], wyr[1]),
                )

    if not story_bands:
        return "no_story_bands"

    # Where does the ceiling Y-range actually fit?
    ceil_mid = 0.5 * (cyr[0] + cyr[1])
    matching_story = None
    for story, (y_lo, y_hi) in sorted(story_bands.items()):
        if y_lo - 0.1 <= ceil_mid <= y_hi + 0.1:
            matching_story = story
            break

    flag_room_locator = (flag.get("evidence") or {}).get("room_locator_id")
    flag_room_story: int | None = None
    if flag_room_locator:
        for room in rooms:
            if room.get("locator_id") == flag_room_locator:
                flag_room_story = room.get("story")
                break

    if matching_story is None:
        return "y_outside_all_bands"
    if flag_room_story is not None and matching_story != flag_room_story:
        return "story_assignment_wrong"
    if cyr[1] < cyr[0]:
        return "y_frame_inversion"
    return "ceiling_below_same_story_floor"


def _piece_story(
    piece: dict[str, Any], per_story_env: dict[int, Polygon]
) -> int | None:
    """Best-effort: which story's envelope contains this piece's centroid."""
    centroid = _piece_centroid_xz(piece.get("corners") or [])
    if centroid is None:
        return None
    pt = _pt(centroid)
    for story, env in sorted(per_story_env.items(), reverse=True):
        if env.contains(pt):
            return story
    return None


def _pt(xy: tuple[float, float]):
    from shapely.geometry import Point

    return Point(*xy)


def _find_piece(
    locator_id: str | None, payload: dict[str, Any]
) -> dict[str, Any] | None:
    if not locator_id:
        return None
    pools = [
        payload.get("ceiling") or [],
        payload.get("knee_walls") or [],
        payload.get("dormer_faces") or [],
        payload.get("gaps") or [],
    ]
    for pool in pools:
        for entry in pool:
            if entry.get("locator_id") == locator_id:
                return entry
    for room in payload.get("rooms") or []:
        for wall in room.get("walls") or []:
            if wall.get("locator_id") == locator_id:
                return wall
    return None


def diagnose_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = payload.get("rooms") or []
    per_story_env = _per_story_envelopes(rooms)
    all_rooms_env = _all_rooms_envelope(rooms)

    rows: list[dict[str, Any]] = []

    for flag in rule_out_of_envelope(payload):
        classification = _classify_out_of_envelope(
            flag, payload, per_story_env, all_rooms_env
        )
        rows.append(
            {
                "rule": "out_of_envelope",
                "locator_id": flag.get("locator"),
                "classification": classification,
                "category": (flag.get("evidence") or {}).get("category"),
                "outside_area_xz_m2": (flag.get("evidence") or {}).get(
                    "outside_area_xz_m2"
                ),
                "outside_ratio": (flag.get("evidence") or {}).get("outside_ratio"),
            }
        )

    for flag in rule_ceiling_below_floor(payload):
        classification = _classify_ceiling_below_floor(flag, payload, rooms)
        rows.append(
            {
                "rule": "ceiling_below_floor",
                "locator_id": flag.get("locator"),
                "classification": classification,
                "delta_below_m": (flag.get("evidence") or {}).get("delta_below_m"),
                "ceiling_y_range": json.dumps(
                    (flag.get("evidence") or {}).get("ceiling_y_range")
                ),
                "room_locator_id": (flag.get("evidence") or {}).get("room_locator_id"),
            }
        )

    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true", help="Scan every building under --root."
    )
    group.add_argument("--uuid", type=str, help="Diagnose a single building UUID.")
    parser.add_argument("--root", type=Path, default=PIPELINE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--rating-cohort",
        type=str,
        default=None,
        help="Comma-separated manual ratings to filter by (e.g. '1,2'); reads "
        ".context/roof_ratings.json.",
    )
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
            filtered: list[str] = []
            for uuid in targets:
                entry = ratings.get(uuid)
                rating = entry.get("rating") if isinstance(entry, dict) else None
                if str(rating) in wanted:
                    filtered.append(uuid)
            targets = filtered

    if not targets:
        print("no buildings to scan")
        return 0

    out_dir = args.out or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"story_envelope_diag_{timestamp}.csv"

    classification_counts: Counter[tuple[str, str]] = Counter()
    rows_total = 0
    fields = [
        "uuid",
        "rule",
        "classification",
        "locator_id",
        "category",
        "outside_area_xz_m2",
        "outside_ratio",
        "delta_below_m",
        "ceiling_y_range",
        "room_locator_id",
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
            except Exception as exc:
                print(f"warn: skip {uuid}: {exc}")
                continue
            rows = diagnose_payload(payload)
            for row in rows:
                row["uuid"] = uuid
                writer.writerow(row)
                classification_counts[(row["rule"], row["classification"])] += 1
                rows_total += 1

    print(f"\ncsv written: {csv_path}")
    print(f"buildings scanned: {len(targets)}")
    print(f"flags classified: {rows_total}\n")

    print(f"{'rule':<25} {'classification':<32} {'count':>8}")
    for (rule, cls), count in sorted(
        classification_counts.items(), key=lambda kv: (kv[0][0], -kv[1])
    ):
        print(f"{rule:<25} {cls:<32} {count:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
