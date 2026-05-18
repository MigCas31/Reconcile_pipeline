"""Pre-export overlap audit for the calor seam.

The plan to feed tirana's tier_payload geometry into calor's
``HomeRoomByRoomInitializeV1`` request requires a "no overlap, single owner"
invariant per room. This module measures, across all corpus buildings, how
often the six known overlap classes occur today so we can size the resolution
work before writing the actual export adapter.

Usage:

    python -m reconcile_tiers.export.overlap_audit \\
        --root pipeline-outputs \\
        --variant tier_payload.json \\
        --out .context/overlap_audit.json

Per-building output is also printed to stdout when ``--verbose``.

Six classes (one count per building, plus a defect list):

    1. ceiling_spans_rooms        — single ceiling piece whose XZ overlaps >1 room floor
    2. ceiling_floor_dup          — ceiling at story N has a duplicate floor at story
    N+1
    3. shared_wall_unmerged       — pair of walls in different rooms sharing a plane and
    overlap
                                    (calor *wants* this; we count to know how many pairs
                                    to expect)
    4. floor_overlap              — pairwise floor XZ intersection on the same story
    above tol
    5. roof_vs_top_ceiling        — top-story flat ceiling under an oblique ceiling
    piece, same XZ
    6. gap_closure_overlap        — gap piece whose XZ intersects a real ceiling/floor
    piece

The audit is read-only: it never modifies tier_payload.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

# Tolerances mirror reconcile_tiers.payload.validate.
CEILING_OVERLAP_TOLERANCE_M2 = 1e-2
FLOOR_OVERLAP_TOLERANCE_M2 = 0.30
WALL_OVERLAP_TOLERANCE_M2 = 1.0
GAP_OVERLAP_TOLERANCE_M2 = 1e-2
PLANE_KEY_NORMAL_TOL = 1e-2
PLANE_KEY_OFFSET_TOL = 5e-2
INTER_STORY_Y_TOL_M = 0.30


def _xz_polygon(
    corners: list[dict[str, float]], holes: list[list[dict[str, float]]] | None = None
) -> Polygon | None:
    if len(corners) < 3:
        return None
    shell = [(c["x"], c["z"]) for c in corners]
    hole_rings: list[list[tuple[float, float]]] = []
    for hole in holes or []:
        if len(hole) >= 3:
            hole_rings.append([(c["x"], c["z"]) for c in hole])
    try:
        poly = Polygon(shell, hole_rings)
    except Exception:
        return None
    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
        except Exception:
            return None
    if poly.is_empty or not isinstance(poly, Polygon):
        return None
    return poly


def _safe_inter_area(a: BaseGeometry, b: BaseGeometry) -> float:
    try:
        return float(a.intersection(b).area)
    except Exception:
        return 0.0


def _plane_key(plane: dict[str, float]) -> tuple[int, int, int, int]:
    """Quantize plane (a,b,c,d) so coplanar pieces share a key."""
    return (
        round(plane["a"] / PLANE_KEY_NORMAL_TOL),
        round(plane["b"] / PLANE_KEY_NORMAL_TOL),
        round(plane["c"] / PLANE_KEY_NORMAL_TOL),
        round(plane["d"] / PLANE_KEY_OFFSET_TOL),
    )


def _plane_y_at(plane: dict[str, float], x: float, z: float) -> float | None:
    b = plane.get("b") or 0.0
    if abs(b) < 1e-6:
        return None
    return (plane["d"] - plane["a"] * x - plane["c"] * z) / b


def _is_oblique(plane: dict[str, float]) -> bool:
    """Plane normal points away from straight-up by more than ~5°."""
    b = abs(plane.get("b") or 0.0)
    return b < 0.996  # cos(5°) ~ 0.996


def _floor_y(corners: list[dict[str, float]]) -> float | None:
    if not corners:
        return None
    ys = [c["y"] for c in corners]
    return sum(ys) / len(ys)


@dataclass
class BuildingDefects:
    uuid: str
    n_rooms: int = 0
    n_ceilings: int = 0
    n_gaps: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[str]] = field(default_factory=dict)

    def add(self, cls: str, locator: str | None = None) -> None:
        self.counts[cls] = self.counts.get(cls, 0) + 1
        if locator and len(self.samples.setdefault(cls, [])) < 3:
            self.samples[cls].append(locator)


CHECK_CLASSES = (
    "ceiling_spans_rooms",
    "ceiling_floor_dup",
    "shared_wall_unmerged",
    "floor_overlap",
    "roof_vs_top_ceiling",
    "gap_closure_overlap",
)


def audit_payload(payload: dict[str, Any]) -> BuildingDefects:
    uuid = str(payload.get("uuid", "?"))
    rooms = payload.get("rooms") or []
    ceilings = payload.get("ceiling") or []
    gaps = payload.get("gaps") or []
    defects = BuildingDefects(
        uuid=uuid, n_rooms=len(rooms), n_ceilings=len(ceilings), n_gaps=len(gaps)
    )

    # Pre-compute per-room floor polygons (XZ) and per-room story.
    room_floors: list[Polygon | None] = []
    room_floor_y: list[float | None] = []
    room_story: list[int] = []
    for room in rooms:
        polys: list[Polygon] = []
        floor_ys: list[float] = []
        for lid in room.get("floor") or []:
            corners = lid.get("corners") or []
            poly = _xz_polygon(corners)
            if poly is not None:
                polys.append(poly)
            y = _floor_y(corners)
            if y is not None:
                floor_ys.append(y)
        if polys:
            merged: BaseGeometry = polys[0]
            for p in polys[1:]:
                try:
                    merged = merged.union(p)
                except Exception:
                    pass
            room_floors.append(merged if isinstance(merged, Polygon) else polys[0])
        else:
            room_floors.append(None)
        room_floor_y.append(sum(floor_ys) / len(floor_ys) if floor_ys else None)
        room_story.append(int(room.get("story", 0)))

    # Check 1: ceiling piece XZ spans >1 room floor (would have to be split per-room for
    # calor).
    for _c_idx, piece in enumerate(ceilings):
        poly = _xz_polygon(piece.get("corners") or [], piece.get("holes"))
        if poly is None:
            continue
        owners = 0
        for floor in room_floors:
            if floor is None:
                continue
            if _safe_inter_area(poly, floor) > CEILING_OVERLAP_TOLERANCE_M2:
                owners += 1
                if owners > 1:
                    break
        if owners > 1:
            defects.add("ceiling_spans_rooms", piece.get("locator_id"))

    # Check 2: ceiling at story N has a near-coplanar floor at story N+1 (same physical
    # surface).
    for _c_idx, piece in enumerate(ceilings):
        plane = piece.get("plane") or {}
        if not plane or _is_oblique(plane):
            continue  # only flat ceilings can plausibly equal a floor above
        c_poly = _xz_polygon(piece.get("corners") or [])
        if c_poly is None:
            continue
        # Estimate ceiling Y from plane at polygon centroid.
        cx, cz = c_poly.centroid.x, c_poly.centroid.y  # shapely XZ projection: y == z
        c_y = _plane_y_at(plane, cx, cz)
        if c_y is None:
            continue
        for r_idx, floor in enumerate(room_floors):
            if floor is None:
                continue
            f_y = room_floor_y[r_idx]
            if f_y is None or abs(f_y - c_y) > INTER_STORY_Y_TOL_M:
                continue
            if _safe_inter_area(c_poly, floor) > CEILING_OVERLAP_TOLERANCE_M2:
                defects.add("ceiling_floor_dup", piece.get("locator_id"))
                break

    # Check 3: shared-wall pairs between rooms (calor wants both faces; count pairs that
    # exist).
    # Bucket walls by parallel-plane direction (normalized normal, sign-canonical),
    # then count pairs whose plane-offset distance is within a wall-thickness band.
    # This is "candidate shared wall pairs", not a defect — it sizes how many
    # wall-pair openings calor would receive after the swap.
    from reconcile_tiers._core.plane import (
        fit_plane_any,  # local import to keep CLI import light
    )

    NORMAL_BUCKET = 2e-3
    THICKNESS_MAX_M = 0.60
    wall_records: list[tuple[tuple[int, int, int], float, int, str]] = []
    for r_idx, room in enumerate(rooms):
        for _w_idx, wall in enumerate(room.get("walls") or []):
            corners = wall.get("corners") or []
            if len(corners) < 3:
                continue
            plane = fit_plane_any([[c["x"], c["y"], c["z"]] for c in corners])
            if plane is None:
                continue
            a, b, c, d = plane
            # canonicalize sign so opposing wall normals land on the same key
            comps = (a, b, c)
            dom = max(range(3), key=lambda i: abs(comps[i]))
            if comps[dom] < 0.0:
                a, b, c, d = -a, -b, -c, -d
            key = (
                round(a / NORMAL_BUCKET),
                round(b / NORMAL_BUCKET),
                round(c / NORMAL_BUCKET),
            )
            wall_records.append((key, d, r_idx, wall.get("locator_id", "")))
    by_normal: dict[tuple[int, int, int], list[tuple[float, int, str]]] = {}
    for key, d, r_idx, loc in wall_records:
        by_normal.setdefault(key, []).append((d, r_idx, loc))
    for entries in by_normal.values():
        n = len(entries)
        for i in range(n):
            d_i, r_i, loc_i = entries[i]
            for j in range(i + 1, n):
                d_j, r_j, _ = entries[j]
                if r_i == r_j:
                    continue
                if abs(d_i - d_j) <= THICKNESS_MAX_M:
                    defects.add("shared_wall_unmerged", loc_i)

    # Check 4: pairwise floor overlap on same story.
    for i in range(len(rooms)):
        if room_floors[i] is None:
            continue
        for j in range(i + 1, len(rooms)):
            if room_floors[j] is None or room_story[i] != room_story[j]:
                continue
            inter = _safe_inter_area(room_floors[i], room_floors[j])
            if inter > FLOOR_OVERLAP_TOLERANCE_M2:
                # Skip nested (mezzanine) cases — calor handles those via adjacency.
                try:
                    if room_floors[i].covers(room_floors[j]) or room_floors[j].covers(
                        room_floors[i]
                    ):
                        continue
                except Exception:
                    pass
                loc_i = rooms[i].get("locator_id") or ""
                defects.add(
                    "floor_overlap", f"{loc_i}|story={room_story[i]}|area={inter:.2f}m2"
                )

    # Check 5: roof (oblique CeilingPiece) over a flat top-story ceiling at same XZ.
    flat_pieces: list[tuple[int, Polygon]] = []
    oblique_pieces: list[tuple[int, Polygon]] = []
    for c_idx, piece in enumerate(ceilings):
        plane = piece.get("plane") or {}
        poly = _xz_polygon(piece.get("corners") or [])
        if poly is None or not plane:
            continue
        if _is_oblique(plane):
            oblique_pieces.append((c_idx, poly))
        else:
            flat_pieces.append((c_idx, poly))
    for fi, fpoly in flat_pieces:
        for _oi, opoly in oblique_pieces:
            if _safe_inter_area(fpoly, opoly) > CEILING_OVERLAP_TOLERANCE_M2:
                loc = ceilings[fi].get("locator_id")
                defects.add("roof_vs_top_ceiling", loc)
                break

    # Check 6: gap closure overlapping real ceiling/floor pieces by *area*, not seam.
    # Gap pieces are designed to butt up against ceilings/floors, so they
    # touch real geometry along thin seams. We only flag overlap that exceeds
    # 10% of the gap polygon's area — anything below is seam contact, not duplication.
    GAP_OVERLAP_RATIO = 0.10
    pre_built_ceilings = [
        (idx, _xz_polygon(piece.get("corners") or []))
        for idx, piece in enumerate(ceilings)
    ]
    for _g_idx, gap in enumerate(gaps):
        kind = gap.get("kind", "")
        if "stitch" not in kind and "gap" not in kind:
            continue
        gpoly = _xz_polygon(gap.get("corners") or [])
        if gpoly is None or gpoly.area <= 0.0:
            continue
        threshold = max(GAP_OVERLAP_TOLERANCE_M2, GAP_OVERLAP_RATIO * gpoly.area)
        hit = False
        for _, cpoly in pre_built_ceilings:
            if cpoly is None:
                continue
            if _safe_inter_area(gpoly, cpoly) > threshold:
                hit = True
                break
        if not hit:
            for floor in room_floors:
                if floor is None:
                    continue
                if _safe_inter_area(gpoly, floor) > threshold:
                    hit = True
                    break
        if hit:
            defects.add("gap_closure_overlap", gap.get("locator_id"))

    # Ensure all classes appear in counts (for stable summary tables).
    for cls in CHECK_CLASSES:
        defects.counts.setdefault(cls, 0)
    return defects


def _iter_payload_paths(root: Path, variant: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.glob(f"*/{variant}") if p.is_file())


def run_audit(
    root: Path, variant: str, limit: int | None = None
) -> list[BuildingDefects]:
    paths = _iter_payload_paths(root, variant)
    if limit is not None:
        paths = paths[:limit]
    results: list[BuildingDefects] = []
    for path in paths:
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except Exception as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue
        results.append(audit_payload(payload))
    return results


def summarize(results: list[BuildingDefects]) -> dict[str, Any]:
    total = len(results)
    summary: dict[str, Any] = {
        "buildings_audited": total,
        "classes": {},
    }
    for cls in CHECK_CLASSES:
        n_buildings = sum(1 for r in results if r.counts.get(cls, 0) > 0)
        n_total = sum(r.counts.get(cls, 0) for r in results)
        summary["classes"][cls] = {
            "buildings_with_defect": n_buildings,
            "buildings_share": (n_buildings / total) if total else 0.0,
            "total_defects": n_total,
        }
    return summary


def _format_summary_table(summary: dict[str, Any]) -> str:
    total = summary["buildings_audited"]
    lines = [f"buildings audited: {total}", ""]
    lines.append(f"{'class':<28} {'buildings':>10} {'share':>8} {'defects':>10}")
    lines.append("-" * 60)
    for cls, stats in summary["classes"].items():
        lines.append(
            f"{cls:<28} {stats['buildings_with_defect']:>10} "
            f"{stats['buildings_share'] * 100:>7.1f}% {stats['total_defects']:>10}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="pipeline-outputs", type=Path)
    parser.add_argument("--variant", default="tier_payload.json")
    parser.add_argument(
        "--out", type=Path, default=None, help="Write per-building defects JSON"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    results = run_audit(args.root, args.variant, args.limit)
    summary = summarize(results)
    print(_format_summary_table(summary))

    if args.verbose:
        for r in results:
            nz = {k: v for k, v in r.counts.items() if v}
            if nz:
                print(f"  {r.uuid}: {nz}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "summary": summary,
                    "buildings": [
                        {
                            "uuid": r.uuid,
                            "n_rooms": r.n_rooms,
                            "n_ceilings": r.n_ceilings,
                            "n_gaps": r.n_gaps,
                            "counts": r.counts,
                            "samples": r.samples,
                        }
                        for r in results
                    ],
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.out} ({len(results)} buildings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
