"""Write reconciled building JSON output."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .cross_floor_gaps import CrossFloorGap, StoryFootprint
from .floor_gaps import WallThickness
from .models import Building, Floor, Surface, Vec3
from .validation import BuildingQualityReport


def _serialize_vec3(v: Vec3) -> list[float]:
    return [v.x, v.y, v.z]


def _serialize_surface(s: Surface) -> dict:
    return {
        "identifier": s.identifier,
        "category": s.category,
        "confidence": s.confidence,
        "dimensions": _serialize_vec3(s.dimensions),
        "transform": s.transform.to_flat(),
        "polygonCorners": [_serialize_vec3(c) for c in s.polygon_corners],
        "story": s.story,
        "completedEdges": s.completed_edges,
        "parentIdentifier": s.parent_identifier,
        "curve": s.curve,
    }


def _serialize_floor(f: Floor) -> dict:
    return {
        "identifier": f.identifier,
        "category": f.category,
        "confidence": f.confidence,
        "dimensions": _serialize_vec3(f.dimensions),
        "transform": f.transform.to_flat(),
        "polygonCorners": [_serialize_vec3(c) for c in f.polygon_corners],
        "story": f.story,
        "completedEdges": f.completed_edges,
        "parentIdentifier": f.parent_identifier,
        "curve": f.curve,
    }


def _serialize_room(room) -> dict:
    return {
        "story": room.story,
        "walls": [_serialize_surface(w) for w in room.walls],
        "doors": [_serialize_surface(d) for d in room.doors],
        "windows": [_serialize_surface(w) for w in room.windows],
        "floors": [_serialize_floor(f) for f in room.floors],
        "referenceOriginTransform": (
            room.reference_origin_transform.to_flat()
            if room.reference_origin_transform
            else None
        ),
    }


def write_reconciled(
    building: Building,
    wall_thicknesses: list[WallThickness],
    quality_report: BuildingQualityReport,
    output_path: Path,
    cross_floor_gaps: list[CrossFloorGap] | None = None,
    story_footprints: list[StoryFootprint] | None = None,
):
    """Write reconciled building model to JSON."""
    data = {
        "version": building.version,
        "reconciliation": {
            "timestamp": datetime.now(UTC).isoformat(),
            "classification": quality_report.classification,
            "wall_count_merged": quality_report.wall_count_merged,
            "wall_count_scan": quality_report.wall_count_scan,
            "wall_reduction_pct": round(quality_report.wall_reduction_pct, 1),
            "mean_width_delta_cm": round(quality_report.mean_width_delta_cm, 1),
            "mean_height_delta_cm": round(quality_report.mean_height_delta_cm, 1),
            "floor_gap_measurements": quality_report.floor_gap_count,
            "floor_gap_median_cm": round(quality_report.floor_gap_median_cm, 1),
            "scan_session_count": quality_report.scan_session_count,
            "median_wall_displacement_m": round(
                quality_report.median_wall_displacement_m, 2
            ),
            "max_wall_displacement_m": round(quality_report.max_wall_displacement_m, 2),
            "story_changes": quality_report.story_change_count,
            "story_change_ratio": round(quality_report.story_change_ratio, 3),
            "walls_with_large_displacement": (
                quality_report.walls_with_large_displacement
            ),
            "flags": quality_report.flags,
        },
        "walls": [_serialize_surface(w) for w in building.top_level_walls],
        "doors": [_serialize_surface(d) for d in building.top_level_doors],
        "windows": [_serialize_surface(w) for w in building.top_level_windows],
        "floors": [_serialize_floor(f) for f in building.top_level_floors],
        "rooms": [_serialize_room(r) for r in building.rooms],
        "wall_thicknesses": [
            {
                "room_a": wt.room_a_id,
                "room_b": wt.room_b_id,
                "thickness_cm": round(wt.thickness_cm, 1),
                "confidence": wt.confidence,
                "centerline_wkt": wt.centerline.wkt,
            }
            for wt in wall_thicknesses
        ],
        "cross_floor_gaps": [
            {
                "story": g.story,
                "reference_story": g.reference_story,
                "region_wkt": g.region_wkt,
                "area_m2": round(g.area_m2, 3),
                "compactness": round(g.compactness, 3),
                "perimeter_contact_pct": round(g.perimeter_contact_pct, 3),
                "confidence": g.confidence,
                "confidence_score": round(g.confidence_score, 2),
                "affected_room": g.affected_room_id,
                "centroid": [round(g.centroid_x, 3), round(g.centroid_z, 3)],
            }
            for g in (cross_floor_gaps or [])
        ],
        "story_footprints": [
            {
                "story": sf.story,
                "area_m2": round(sf.area_m2, 2),
                "room_count": sf.room_count,
                "footprint_wkt": sf.footprint_wkt,
            }
            for sf in (story_footprints or [])
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
