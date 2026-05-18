from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.build import build_tier_payload
from reconcile_tiers.build_internals.io_index import list_tier_payload_uuids
from reconcile_tiers.payload.schema import payload_to_dict

DEFAULT_OUTPUT = Path(".context/removed-ceiling-fragments.json")
REPLACEMENT_OVERLAP_MIN_M2 = 0.005
REPLACEMENT_GOOD_OVERLAP_RATIO = 0.8
REPLACEMENT_COPLANAR_MAX_DELTA_M = 0.08
REPLACEMENT_STORY_MIN_OVERLAP_M2 = 0.01
REPLACEMENT_STORY_MIN_ROOM_HEIGHT_M = 1.0
REPLACEMENT_STORY_TOP_SLACK_M = 0.5
REPLACEMENT_STORY_MAX_ABOVE_TOP_M = 1.5
RAW_CEILING_INTERIOR_BOUNDARY_DISTANCE_M = None
RAW_CEILING_INTERIOR_FRAGMENT_MAX_AREA_M2 = None

INTERIOR_DROP_REASONS = {
    "interior_fragment",
    "interior_covered",
    "interior_fragment_owner",
    "interior_covered_owner",
    "interior_fragment_final",
    "interior_covered_final",
}


def _summarise_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def _normalise_drop(uuid: str, drop: dict[str, Any]) -> dict[str, Any] | None:
    corners = drop.get("corners")
    if not isinstance(corners, list) or len(corners) < 3:
        return None
    reason = str(drop.get("reason") or "")
    return {
        "uuid": uuid,
        "locator_id": str(drop.get("locator_id") or ""),
        "reason": reason,
        "drop_stage": str(drop.get("drop_stage") or "unknown"),
        "source": str(drop.get("source") or ""),
        "story": drop.get("story"),
        "area_xz_m2": float(drop.get("area_xz_m2") or 0.0),
        "coverage_ratio": drop.get("coverage_ratio"),
        "corners": corners,
        "plane": drop.get("plane"),
    }


def _xz_polygon(corners: list) -> Any | None:
    if not isinstance(corners, list) or len(corners) < 3:
        return None
    try:
        from shapely.geometry import Polygon

        poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 1e-9:
            return None
        return poly
    except Exception:
        return None


def _piece_corners(piece: dict[str, Any]) -> list[list[float]]:
    out: list[list[float]] = []
    for corner in piece.get("corners") or []:
        if isinstance(corner, dict):
            out.append([float(corner["x"]), float(corner["y"]), float(corner["z"])])
        elif isinstance(corner, (list, tuple)) and len(corner) >= 3:
            out.append([float(corner[0]), float(corner[1]), float(corner[2])])
    return out


def _room_entries(payload_dict: dict[str, Any]) -> list[dict[str, Any]]:
    from shapely.ops import unary_union

    entries: list[dict[str, Any]] = []
    for idx, room in enumerate(payload_dict.get("rooms") or []):
        floor_polys = []
        floor_ys = []
        for floor in room.get("floor") or []:
            corners = _piece_corners(floor)
            poly = _xz_polygon(corners)
            if poly is not None:
                floor_polys.append(poly)
            floor_ys.extend(float(corner[1]) for corner in corners)
        if not floor_polys:
            continue
        try:
            floor_poly = unary_union(floor_polys)
        except Exception:
            floor_poly = floor_polys[0]
        wall_ys = [
            float(corner[1])
            for wall in room.get("walls") or []
            for corner in _piece_corners(wall)
        ]
        if not wall_ys:
            continue
        entries.append(
            {
                "index": idx,
                "story": room.get("story"),
                "poly": floor_poly,
                "floor_y": sum(floor_ys) / len(floor_ys) if floor_ys else min(wall_ys),
                "top_y": max(wall_ys),
            }
        )
    return entries


def _plane_from_record(record: dict[str, Any] | None) -> Plane | None:
    if not isinstance(record, dict):
        return None
    try:
        return Plane(
            a=float(record["a"]),
            b=float(record["b"]),
            c=float(record["c"]),
            d=float(record["d"]),
        )
    except Exception:
        return None


def _fit_plane(corners: list[list[float]]) -> Plane | None:
    if len(corners) < 3:
        return None
    plane = Plane.fit(corners)
    return None if isinstance(plane, FitFailure) else plane


def _polygon_parts(geometry: Any) -> list[Any]:
    if geometry is None or geometry.is_empty:
        return []
    try:
        from shapely.geometry import Polygon
    except Exception:
        return []
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 1e-9 else []
    return [
        part
        for part in getattr(geometry, "geoms", [])
        if isinstance(part, Polygon) and part.area > 1e-9
    ]


def _corners_on_plane(poly: Any, plane: Plane | None) -> list[list[float]]:
    if plane is None or poly is None or poly.is_empty:
        return []
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    corners: list[list[float]] = []
    for x, z in coords:
        y = plane.y_at(float(x), float(z))
        if y is None or not math.isfinite(float(y)):
            return []
        corners.append([round(float(x), 4), round(float(y), 4), round(float(z), 4)])
    return corners


def _plane_delta_m(
    removed_plane: Plane | None,
    candidate_plane: Plane | None,
    overlap: Any,
) -> float | None:
    if (
        removed_plane is None
        or candidate_plane is None
        or overlap is None
        or overlap.is_empty
    ):
        return None
    samples: list[tuple[float, float]] = []
    try:
        point = overlap.representative_point()
        samples.append((float(point.x), float(point.y)))
    except Exception:
        pass
    for part in _polygon_parts(overlap):
        coords = list(part.exterior.coords)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        samples.extend((float(x), float(z)) for x, z in coords[:12])
    deltas: list[float] = []
    for x, z in samples:
        removed_y = removed_plane.y_at(x, z)
        candidate_y = candidate_plane.y_at(x, z)
        if removed_y is None or candidate_y is None:
            continue
        deltas.append(abs(float(removed_y) - float(candidate_y)))
    return max(deltas) if deltas else None


def _plane_y_values(plane: Plane | None, geom: Any) -> list[float]:
    if plane is None or geom is None or geom.is_empty:
        return []
    values: list[float] = []
    for x, z in _sample_xz_points(geom):
        y = plane.y_at(float(x), float(z))
        if y is None or not math.isfinite(float(y)):
            return []
        values.append(float(y))
    return values


def _sample_xz_points(geom: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if geom is None or geom.is_empty:
        return points
    try:
        point = geom.representative_point()
        points.append((float(point.x), float(point.y)))
    except Exception:
        pass
    for part in _polygon_parts(geom):
        coords = list(part.exterior.coords)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        points.extend((float(x), float(z)) for x, z in coords[:12])
    return points


def _candidate_stories(
    poly: Any,
    plane: Plane | None,
    rooms: list[dict[str, Any]],
) -> list[int]:
    stories: set[int] = set()
    for room in rooms:
        story = room.get("story")
        if not isinstance(story, int):
            continue
        try:
            overlap = poly.intersection(room["poly"])
        except Exception:
            continue
        if overlap.is_empty or overlap.area < REPLACEMENT_STORY_MIN_OVERLAP_M2:
            continue
        y_values = _plane_y_values(plane, overlap)
        if not y_values:
            continue
        floor_y = float(room["floor_y"])
        top_y = float(room["top_y"])
        if (
            min(y_values) >= floor_y + REPLACEMENT_STORY_MIN_ROOM_HEIGHT_M
            and max(y_values) >= top_y - REPLACEMENT_STORY_TOP_SLACK_M
            and min(y_values) <= top_y + REPLACEMENT_STORY_MAX_ABOVE_TOP_M
        ):
            stories.add(story)
    return sorted(stories)


def _overlap_patch_corners(
    overlap: Any, candidate_plane: Plane | None
) -> list[list[float]]:
    parts = _polygon_parts(overlap)
    if not parts:
        return []
    largest = max(parts, key=lambda part: float(part.area))
    return _corners_on_plane(largest, candidate_plane)


def _coverage_status(overlap_ratio: float, plane_delta_m: float | None) -> str:
    if overlap_ratio >= REPLACEMENT_GOOD_OVERLAP_RATIO and (
        plane_delta_m is not None and plane_delta_m <= REPLACEMENT_COPLANAR_MAX_DELTA_M
    ):
        return "same_plane_cover"
    if plane_delta_m is None:
        return "overlap_unknown_plane"
    if plane_delta_m > REPLACEMENT_COPLANAR_MAX_DELTA_M:
        return "overlap_not_coplanar"
    return "partial_same_plane_overlap"


def _replacement_candidates(payload_dict: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = _room_entries(payload_dict)
    candidates: list[dict[str, Any]] = []
    for piece in payload_dict.get("ceiling") or []:
        corners = _piece_corners(piece)
        poly = _xz_polygon(corners)
        plane = _fit_plane(corners)
        if poly is None:
            continue
        stories = _candidate_stories(poly, plane, rooms)
        candidates.append(
            {
                "locator_id": piece.get("locator_id") or "",
                "kind": "ceiling",
                "source": piece.get("source") or "",
                "story": stories[0] if len(stories) == 1 else None,
                "stories": stories,
                "area_xz_m2": float(poly.area),
                "corners": corners,
                "_plane": plane,
                "_poly": poly,
            }
        )
    for gap in payload_dict.get("gaps") or []:
        if gap.get("kind") not in {
            "gap_ceiling",
            "stitch_ceiling",
            "exterior_ceiling",
        }:
            continue
        corners = _piece_corners(gap)
        poly = _xz_polygon(corners)
        plane = _fit_plane(corners)
        if poly is None:
            continue
        stories = _candidate_stories(poly, plane, rooms)
        candidates.append(
            {
                "locator_id": gap.get("locator_id") or "",
                "kind": gap.get("kind") or "gap_ceiling",
                "source": gap.get("kind") or "gap_ceiling",
                "story": stories[0] if len(stories) == 1 else None,
                "stories": stories,
                "area_xz_m2": float(poly.area),
                "corners": corners,
                "_plane": plane,
                "_poly": poly,
            }
        )
    return candidates


def _attach_replacements(
    items: list[dict[str, Any]],
    payload_dict: dict[str, Any],
) -> None:
    candidates = _replacement_candidates(payload_dict)
    for item in items:
        removed_poly = _xz_polygon(item.get("corners") or [])
        removed_plane = _plane_from_record(item.get("plane")) or _fit_plane(
            item.get("corners") or []
        )
        if removed_poly is None:
            item["replacements"] = []
            item["coverage_candidates"] = []
            item["same_plane_cover_count"] = 0
            continue
        replacements = []
        for candidate in candidates:
            item_story = item.get("story")
            candidate_stories = candidate.get("stories") or []
            if isinstance(item_story, int) and item_story not in candidate_stories:
                continue
            poly = candidate["_poly"]
            try:
                overlap_area = float(removed_poly.intersection(poly).area)
                overlap = removed_poly.intersection(poly)
            except Exception:
                continue
            if overlap_area < REPLACEMENT_OVERLAP_MIN_M2:
                continue
            replacement = {
                k: v for k, v in candidate.items() if k not in {"_poly", "_plane"}
            }
            overlap_ratio = overlap_area / max(float(removed_poly.area), 1e-9)
            plane_delta = _plane_delta_m(
                removed_plane,
                candidate.get("_plane"),
                overlap,
            )
            replacement["overlap_xz_m2"] = round(overlap_area, 3)
            replacement["overlap_ratio"] = round(overlap_ratio, 3)
            replacement["full_area_ratio"] = round(
                float(poly.area) / max(float(removed_poly.area), 1e-9),
                3,
            )
            replacement["plane_delta_m"] = (
                round(float(plane_delta), 3) if plane_delta is not None else None
            )
            replacement["coverage_status"] = (
                "neighbor_plane_extension"
                if "tier-ceiling-neighbor-extension"
                in str(replacement.get("locator_id") or "")
                else _coverage_status(overlap_ratio, plane_delta)
            )
            replacement["overlap_corners"] = _overlap_patch_corners(
                overlap,
                candidate.get("_plane"),
            )
            replacements.append(replacement)
        replacements.sort(
            key=lambda row: (
                0 if row.get("coverage_status") == "same_plane_cover" else 1,
                -float(row.get("overlap_ratio") or 0.0),
                float(row.get("plane_delta_m") or 999.0),
                row.get("locator_id") or "",
            )
        )
        item["replacements"] = replacements[:8]
        item["coverage_candidates"] = item["replacements"]
        item["same_plane_cover_count"] = sum(
            1
            for replacement in item["replacements"]
            if replacement.get("coverage_status") == "same_plane_cover"
        )


def build_removed_ceiling_viewer_data(
    *,
    pipeline_dir: Path,
    scan_root: Path | None,
    output_path: Path,
    uuids: list[str] | None = None,
    limit: int | None = None,
    include_all_drops: bool = False,
) -> dict[str, Any]:
    selected = uuids if uuids is not None else list_tier_payload_uuids(pipeline_dir)
    if limit is not None:
        selected = selected[:limit]

    buildings: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total_reason_counts: Counter[str] = Counter()
    total_source_counts: Counter[str] = Counter()
    total_coverage_counts: Counter[str] = Counter()
    total_area = 0.0
    total_count = 0

    for idx, uuid in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {uuid}", flush=True)
        drops: list[dict[str, Any]] = []
        try:
            payload = build_tier_payload(
                uuid,
                pipeline_dir,
                scan_root,
                drops_sink=drops,
            )
        except Exception as exc:
            errors.append({"uuid": uuid, "error": f"{type(exc).__name__}: {exc}"})
            continue

        payload_dict = payload_to_dict(payload)
        raw_items = (
            drops
            if include_all_drops
            else [drop for drop in drops if drop.get("reason") in INTERIOR_DROP_REASONS]
        )
        items = [
            item
            for drop in raw_items
            if (item := _normalise_drop(uuid, drop)) is not None
        ]
        _attach_replacements(items, payload_dict)
        if not items:
            continue
        reason_counts = Counter(item["reason"] for item in items)
        source_counts = Counter(item["source"] or "unknown" for item in items)
        area = round(sum(float(item.get("area_xz_m2") or 0.0) for item in items), 3)
        total_count += len(items)
        total_area += area
        total_reason_counts.update(reason_counts)
        total_source_counts.update(source_counts)
        total_coverage_counts.update(
            str(replacement.get("coverage_status") or "unknown")
            for item in items
            for replacement in item.get("replacements") or []
        )
        buildings.append(
            {
                "uuid": uuid,
                "address": payload_dict.get("address"),
                "classification": payload_dict.get("classification") or {},
                "removed_count": len(items),
                "removed_area_xz_m2": area,
                "reason_counts": _summarise_counter(reason_counts),
                "source_counts": _summarise_counter(source_counts),
                "items": sorted(
                    items,
                    key=lambda item: (
                        str(item.get("story")),
                        -float(item.get("area_xz_m2") or 0.0),
                        item.get("locator_id") or "",
                    ),
                ),
            }
        )

    buildings.sort(
        key=lambda row: (
            -int(row["removed_count"]),
            -float(row["removed_area_xz_m2"]),
            row["uuid"],
        )
    )
    result = {
        "schema_version": 1,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "pipeline_dir": str(pipeline_dir),
        "scan_root": str(scan_root) if scan_root is not None else None,
        "include_all_drops": include_all_drops,
        "thresholds": {
            "interior_boundary_distance_m": RAW_CEILING_INTERIOR_BOUNDARY_DISTANCE_M,
            "interior_fragment_max_area_m2": RAW_CEILING_INTERIOR_FRAGMENT_MAX_AREA_M2,
        },
        "summary": {
            "buildings_requested": len(selected),
            "buildings_with_removed": len(buildings),
            "total_removed_count": total_count,
            "total_removed_area_xz_m2": round(total_area, 3),
            "reason_counts": _summarise_counter(total_reason_counts),
            "source_counts": _summarise_counter(total_source_counts),
            "coverage_candidate_counts": _summarise_counter(total_coverage_counts),
            "errors": len(errors),
        },
        "buildings": buildings,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the data file used by the removed-ceiling-fragments viewer."
    )
    parser.add_argument("--pipeline-dir", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument("--scan-root", type=Path, default=Path(".scan-cache"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--uuid", action="append", dest="uuids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-all-drops", action="store_true")
    args = parser.parse_args(argv)

    scan_root = args.scan_root if args.scan_root.exists() else None
    result = build_removed_ceiling_viewer_data(
        pipeline_dir=args.pipeline_dir,
        scan_root=scan_root,
        output_path=args.output,
        uuids=args.uuids,
        limit=args.limit,
        include_all_drops=args.include_all_drops,
    )
    summary = result["summary"]
    print(
        "Wrote "
        f"{args.output} with {summary['total_removed_count']} removed pieces "
        f"across {summary['buildings_with_removed']} buildings"
    )
    if summary["errors"]:
        print(f"Encountered {summary['errors']} build errors; see output JSON.")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
