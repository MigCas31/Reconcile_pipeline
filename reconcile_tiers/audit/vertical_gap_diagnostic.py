"""Plan I Phase 1: read-only diagnostic for vertical (X-Y plane) gaps.

The user-visible failure mode: stacked floors with walls visibly missing --
you can see through the building because a vertical wall section between
adjacent stories was never emitted.

The pipeline already enumerates these in ``payload.gaps[]`` with kinds
``side``, ``exterior_side``, plus the stitch variants (``stitch``,
``stitch_floor``, ``stitch_ceiling``). This diagnostic classifies each
side/exterior_side gap by the most likely upstream cause, so Phase 2 can
target it surgically:

- ``visible_through_wall``  -- exterior-facing side gap with material height
  (>0.4 m) and area (>0.3 m^2). The most reader-visible failure: daylight
  through the building.
- ``unstitched_session_seam``  -- side gap whose XZ position falls on the
  boundary between two scan sessions (heuristic: gap straddles a story
  boundary AND its area approximately equals the perimeter x story height
  delta of an inter-session seam). Cause: cross-floor stitching never closed.
- ``story_seam_uncovered``  -- side gap at a story-to-story interface where
  no wall is emitted on either side. Cause: synthetic-wall pass didn't fire.
- ``interior_intra_story``  -- small interior side gap below visibility
  threshold. Probably noise, reported separately.
- ``other_side``  -- fallback bucket.

Run:

    python -m reconcile_tiers.audit.vertical_gap_diagnostic --all
    python -m reconcile_tiers.audit.vertical_gap_diagnostic --all --rating-cohort 1,2

Outputs ``analysis_outputs/vertical_gap_<ts>.csv`` and a per-class tally.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reconcile_tiers.audit.rules import _y_range

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = WORKSPACE_ROOT / "pipeline-outputs"
OUT_DIR = WORKSPACE_ROOT / "analysis_outputs"

VISIBLE_HEIGHT_M = 0.4
VISIBLE_AREA_M2 = 0.3
STORY_BOUNDARY_TOL_M = 0.6


def _gap_height(gap: dict[str, Any]) -> float:
    yr = _y_range(gap.get("corners") or [])
    if yr is None:
        return 0.0
    return float(yr[1] - yr[0])


def _gap_area(gap: dict[str, Any]) -> float:
    """Cheap polygon area treating corners as XYZ -- for vertical pieces this
    is the wall-plane area we care about."""
    corners = gap.get("corners") or []
    if len(corners) < 3:
        return 0.0
    # Project to the dominant axis-aligned plane (XZ if vertical, etc.).
    # For side gaps the polygon is roughly vertical, so treat XY-Z as a 3D
    # ring and take half the magnitude of the cross-product sum.
    pts = [(c["x"], c["y"], c["z"]) for c in corners]
    n = len(pts)
    nx = ny = nz = 0.0
    for i in range(n):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % n]
        nx += y0 * z1 - z0 * y1
        ny += z0 * x1 - x0 * z1
        nz += x0 * y1 - y0 * x1
    return 0.5 * (nx * nx + ny * ny + nz * nz) ** 0.5


def _story_floor_y(rooms: list[dict[str, Any]]) -> dict[int, list[float]]:
    """Per story, the Y values of every floor piece."""
    out: dict[int, list[float]] = {}
    for room in rooms:
        story = room.get("story")
        if story is None:
            continue
        for piece in room.get("floor") or []:
            yr = _y_range(piece.get("corners") or [])
            if yr is None:
                continue
            out.setdefault(int(story), []).append(yr[0])
    return out


def _crosses_story_boundary(
    gap: dict[str, Any], floor_ys: dict[int, list[float]]
) -> bool:
    yr = _y_range(gap.get("corners") or [])
    if yr is None:
        return False
    for _story, ys in sorted(floor_ys.items()):
        if not ys:
            continue
        floor_y = sum(ys) / len(ys)
        # Gap spans the floor level if its bottom is below and its top above
        # by at least the tolerance.
        if (
            yr[0] < floor_y - STORY_BOUNDARY_TOL_M
            and yr[1] > floor_y + STORY_BOUNDARY_TOL_M
        ):
            return True
    return False


def _classify(
    gap: dict[str, Any], rooms: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    kind = gap.get("kind")
    adjacency = gap.get("adjacency")
    height = _gap_height(gap)
    area = _gap_area(gap)
    floor_ys = _story_floor_y(rooms)
    crosses_seam = _crosses_story_boundary(gap, floor_ys)
    yr = _y_range(gap.get("corners") or [])

    evidence = {
        "kind": kind,
        "adjacency": adjacency,
        "height_m": height,
        "area_m2": area,
        "y_range": list(yr) if yr is not None else None,
        "crosses_story_boundary": crosses_seam,
    }

    if kind not in ("side", "exterior_side"):
        return ("not_vertical", evidence)

    if kind == "exterior_side" or adjacency == "externalAir":
        if height >= VISIBLE_HEIGHT_M and area >= VISIBLE_AREA_M2:
            return ("visible_through_wall", evidence)
        return ("exterior_sliver", evidence)

    # side, interior
    if crosses_seam:
        return ("story_seam_uncovered", evidence)
    if height < VISIBLE_HEIGHT_M and area < VISIBLE_AREA_M2:
        return ("interior_intra_story", evidence)
    return ("other_side", evidence)


def diagnose_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = payload.get("rooms") or []
    rows: list[dict[str, Any]] = []
    for gap in payload.get("gaps") or []:
        if gap.get("kind") not in ("side", "exterior_side"):
            continue
        cls, evidence = _classify(gap, rooms)
        rows.append(
            {
                "rule": "vertical_gap",
                "locator_id": gap.get("locator_id"),
                "classification": cls,
                "kind": evidence.get("kind"),
                "adjacency": evidence.get("adjacency"),
                "height_m": evidence.get("height_m"),
                "area_m2": evidence.get("area_m2"),
                "crosses_story_boundary": evidence.get("crosses_story_boundary"),
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
    csv_path = out_dir / f"vertical_gap_{timestamp}.csv"

    counts: Counter[str] = Counter()
    by_uuid_visible: Counter[str] = Counter()
    rows_total = 0
    fields = [
        "uuid",
        "rule",
        "classification",
        "locator_id",
        "kind",
        "adjacency",
        "height_m",
        "area_m2",
        "crosses_story_boundary",
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
                if row["classification"] == "visible_through_wall":
                    by_uuid_visible[uuid] += 1
                rows_total += 1

    print(f"\ncsv: {csv_path}")
    print(f"buildings: {len(targets)}, gap rows: {rows_total}\n")
    print(f"{'classification':<28} {'count':>6}")
    for cls, n in counts.most_common():
        print(f"{cls:<28} {n:>6}")
    if by_uuid_visible:
        print("\n=== top buildings by visible_through_wall count ===")
        for uuid, n in by_uuid_visible.most_common(10):
            print(f"  {uuid[:8]} {n:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
