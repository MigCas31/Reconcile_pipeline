from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reconcile_tiers._core.newell import newell_normal
from reconcile_tiers.extract.geometry import (
    corners_to_world,
    hybrid_wall_corners,
    wall_world_corners,
)
from reconcile_tiers.extract.stories import compute_room_stories, is_split_level
from reconcile_tiers.ingest.merged import load_merged
from reconcile_tiers.ingest.room_transforms import (
    METHOD_RANK,
    RoomTransform,
    compute_room_transforms,
)
from reconcile_tiers.ingest.scan_cache import (
    find_scan_cache_dir,
    load_raw_ceilings,
    load_raw_rooms,
    load_scan_metadata,
    parse_address_from_scan_dir,
)


@dataclass(frozen=True, slots=True)
class ExtractedWall:
    id: str
    corners: list[list[float]]
    source: str
    descent_strip: list[list[list[float]]] | None = None
    uplift_strip: list[list[list[float]]] | None = None
    synthetic: bool = False


@dataclass(frozen=True, slots=True)
class ExtractedElement:
    id: str
    corners: list[list[float]]
    source: str
    parent_wall_id: str | None = None


@dataclass(frozen=True, slots=True)
class RawCeilingPlane:
    corners: list[list[float]]
    # 0..1 cleanliness signal computed in extract/ceilings.py:score_raw_ceiling_planes.
    # 0.0 means "unscored / unknown" — v2 selection treats it as below threshold.
    support_quality: float = 0.0
    raw_room_key: str | None = None
    raw_plane_index: int | None = None
    raw_source: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedGap:
    story: int
    type: str
    corners: list[list[float]]
    area_m2: float
    compactness: float
    confidence: str
    centroid: list[float]
    ceiling_corners: list[list[float]]
    room_index: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractedGapWall:
    id: str
    corners: list[list[float]]
    type: str
    story: int
    confidence: str


@dataclass(frozen=True, slots=True)
class ExtractedStitchWall:
    id: str
    corners: list[list[float]]
    type: str
    story: int
    room_index: int | None
    room_indices: list[int]


@dataclass(frozen=True, slots=True)
class ExteriorGapIndicator:
    story: int
    element_type: str
    element_id: str
    element_corners: list[list[float]]
    element_width_m: float
    wall_id: str
    wall_corners: list[list[float]]
    wall_distance_m: float
    angle_deg: float
    confidence: str
    parent_wall_id: str | None = None
    parent_wall_corners: list[list[float]] | None = None


@dataclass(frozen=True, slots=True)
class GapClosure:
    corners: list[list[float]]
    type: str
    indicator_element_id: str
    indicator_wall_id: str
    story: int


@dataclass(frozen=True, slots=True)
class ExtractedRoom:
    index: int
    story: int
    floor_polygon: list[list[float]]
    walls_merged: list[ExtractedWall]
    walls_computed: list[ExtractedWall]
    doors: list[ExtractedElement]
    windows: list[ExtractedElement]
    openings: list[ExtractedElement]
    storages: list[ExtractedElement]
    raw_ceiling_planes: list[RawCeilingPlane]
    raw_ceiling_source: str | None
    ceiling_polygon: list[list[float]]
    ceiling_type: str | None
    ceiling_eave_height: float | None
    ceiling_ridge_height: float | None
    synthetic_walls: list[ExtractedWall] = field(default_factory=list)
    heating: str | None = None
    # When the scan shows both a flat lid and one or more slanted patches in
    # the same room (knee-wall geometry), `ceiling_is_kinked` is True and the
    # polygons below carry the lid and slope footprints. `ceiling_slope_polygon`
    # keeps the dominant patch for older call sites; `ceiling_slope_polygons`
    # preserves true double-slanted rooms.
    ceiling_is_kinked: bool = False
    ceiling_flat_polygon: list[list[float]] = field(default_factory=list)
    ceiling_slope_polygon: list[list[float]] = field(default_factory=list)
    ceiling_slope_polygons: list[list[list[float]]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BuildingModel:
    uuid: str
    address: str | None
    stories_found: int
    split_level: bool
    rooms: list[ExtractedRoom]
    scan_rooms_found: int
    scan_rooms_transformed: int
    cross_floor_gaps: list[ExtractedGap] = field(default_factory=list)
    gap_walls: list[ExtractedGapWall] = field(default_factory=list)
    stitch_walls: list[ExtractedStitchWall] = field(default_factory=list)
    exterior_gap_indicators: list[ExteriorGapIndicator] = field(default_factory=list)
    gap_closures: list[GapClosure] = field(default_factory=list)


def _build_raw_wall_index(
    raw_rooms: list[tuple[str, dict]],
    transforms: dict[str, RoomTransform],
) -> dict[str, tuple[dict, RoomTransform]]:
    raw_wall_data: dict[str, tuple[dict, RoomTransform]] = {}
    for room_name, room_data in raw_rooms:
        transform = transforms.get(room_name)
        if transform is None:
            continue
        rank = METHOD_RANK.get(transform.method, 0)
        for wall in room_data.get("walls", []):
            wall_id = wall.get("identifier")
            if not wall_id:
                continue
            existing = raw_wall_data.get(wall_id)
            if existing is None or rank > METHOD_RANK.get(existing[1].method, 0):
                raw_wall_data[wall_id] = (wall, transform)
    return raw_wall_data


def _build_raw_item_index(
    raw_rooms: list[tuple[str, dict]],
    transforms: dict[str, RoomTransform],
    key: str,
) -> dict[str, tuple[dict, RoomTransform]]:
    raw_item_data: dict[str, tuple[dict, RoomTransform]] = {}
    for room_name, room_data in raw_rooms:
        transform = transforms.get(room_name)
        if transform is None:
            continue
        rank = METHOD_RANK.get(transform.method, 0)
        for item in room_data.get(key, []):
            item_id = item.get("identifier")
            if not item_id:
                continue
            existing = raw_item_data.get(item_id)
            if existing is None or rank > METHOD_RANK.get(existing[1].method, 0):
                raw_item_data[item_id] = (item, transform)
    return raw_item_data


def _raw_room_matches_merged(raw_room: dict, merged_room: dict) -> bool:
    raw_wall_ids = {
        wall.get("identifier")
        for wall in raw_room.get("walls", [])
        if wall.get("identifier")
    }
    merged_wall_ids = {
        wall.get("identifier")
        for wall in merged_room.get("walls", [])
        if wall.get("identifier")
    }
    return bool(raw_wall_ids & merged_wall_ids)


def _build_raw_to_merged_index(
    raw_rooms: list[tuple[str, dict]], merged_rooms: list[dict]
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for room_name, raw_room in raw_rooms:
        raw_wall_ids = {
            wall.get("identifier")
            for wall in raw_room.get("walls", [])
            if wall.get("identifier")
        }
        if not raw_wall_ids:
            continue
        best_idx = -1
        best_overlap = 0
        for idx, merged_room in enumerate(merged_rooms):
            merged_ids = {
                wall.get("identifier")
                for wall in merged_room.get("walls", [])
                if wall.get("identifier")
            }
            overlap = len(raw_wall_ids & merged_ids)
            if overlap > best_overlap:
                best_idx = idx
                best_overlap = overlap
        if best_idx >= 0:
            mapping[room_name] = best_idx
    return mapping


def _orient_walls_outward(
    walls: list[ExtractedWall], floor_polygon: list[list[float]]
) -> list[ExtractedWall]:
    if not floor_polygon:
        return walls
    centroid = np.mean(np.asarray(floor_polygon), axis=0)
    oriented: list[ExtractedWall] = []
    for wall in walls:
        corners = wall.corners
        if len(corners) < 3:
            oriented.append(wall)
            continue
        normal = np.array(newell_normal(corners), dtype=float)
        if float(normal @ normal) < 1e-10:
            oriented.append(wall)
            continue
        wall_center = np.mean(np.asarray(corners), axis=0)
        if float(normal @ (wall_center - centroid)) < 0.0:
            oriented.append(
                ExtractedWall(
                    id=wall.id, corners=list(reversed(corners)), source=wall.source
                )
            )
        else:
            oriented.append(wall)
    return oriented


def _extract_room_walls(
    merged_room: dict,
    floor_polygon: list[list[float]],
    raw_wall_data: dict[str, tuple[dict, RoomTransform]],
) -> tuple[list[ExtractedWall], list[ExtractedWall], set[str]]:
    walls_merged = [
        ExtractedWall(
            id=wall["identifier"],
            corners=wall_world_corners(wall),
            source="merged-room",
        )
        for wall in merged_room.get("walls", [])
    ]
    walls_computed: list[ExtractedWall] = []
    for wall in merged_room.get("walls", []):
        wall_id = wall["identifier"]
        raw_entry = raw_wall_data.get(wall_id)
        if raw_entry is None:
            walls_computed.append(
                ExtractedWall(
                    id=wall_id, corners=wall_world_corners(wall), source="merged-room"
                )
            )
            continue
        raw_wall, transform = raw_entry
        if transform.method == "floor-svd":
            walls_computed.append(
                ExtractedWall(
                    id=wall_id,
                    corners=transform.apply(wall_world_corners(raw_wall)),
                    source="scan-cache",
                )
            )
        else:
            floor_y = (
                float(np.mean([corner[1] for corner in floor_polygon]))
                if floor_polygon
                else None
            )
            walls_computed.append(
                ExtractedWall(
                    id=wall_id,
                    corners=hybrid_wall_corners(wall, raw_wall, floor_y=floor_y),
                    source="hybrid",
                )
            )
    return (
        _orient_walls_outward(walls_merged, floor_polygon),
        _orient_walls_outward(walls_computed, floor_polygon),
        {wall.id for wall in walls_computed},
    )


def _add_dedup_walls(
    walls_computed: list[ExtractedWall],
    added_ids: set[str],
    merged_room: dict,
    all_merged_rooms: list[dict],
    raw_rooms: list[tuple[str, dict]],
    transforms: dict[str, RoomTransform],
    global_added: set[str],
) -> list[ExtractedWall]:
    out = list(walls_computed)
    for room_name, raw_room in raw_rooms:
        transform = transforms.get(room_name)
        if (
            transform is None
            or transform.method != "floor-svd"
            or not _raw_room_matches_merged(raw_room, merged_room)
        ):
            continue
        for wall in raw_room.get("walls", []):
            wall_id = wall.get("identifier")
            if not wall_id or wall_id in added_ids or wall_id in global_added:
                continue
            in_any_merged = any(
                any(
                    merged_wall.get("identifier") == wall_id
                    for merged_wall in room.get("walls", [])
                )
                for room in all_merged_rooms
            )
            if in_any_merged:
                continue
            out.append(
                ExtractedWall(
                    id=wall_id,
                    corners=transform.apply(wall_world_corners(wall)),
                    source="scan-cache-dedup",
                )
            )
            added_ids.add(wall_id)
            global_added.add(wall_id)
    return out


def _opening_near_parent(
    opening_corners: list[list[float]],
    parent_id: str | None,
    walls_computed: list[ExtractedWall],
    max_dist: float = 1.5,
) -> bool:
    if not parent_id:
        return True
    parent = next((wall for wall in walls_computed if wall.id == parent_id), None)
    if parent is None:
        return True
    opening_center = np.mean(np.asarray(opening_corners), axis=0)
    parent_center = np.mean(np.asarray(parent.corners), axis=0)
    return float(np.linalg.norm(opening_center - parent_center)) < max_dist


def _extract_openings(
    key: str,
    merged_items: list[dict],
    raw_items_by_id: dict[str, tuple[dict, RoomTransform]],
    merged_ids: set[str],
    merged_room: dict,
    walls_computed: list[ExtractedWall],
    raw_rooms: list[tuple[str, dict]],
    transforms: dict[str, RoomTransform],
    global_added: set[str],
) -> list[ExtractedElement]:
    out: list[ExtractedElement] = []
    for item in merged_items:
        item_id = item["identifier"]
        parent_id = item.get("parentIdentifier")
        raw_entry = raw_items_by_id.get(item_id)
        if raw_entry is not None and raw_entry[1].method == "floor-svd":
            raw_item, transform = raw_entry
            parent_id = parent_id or raw_item.get("parentIdentifier")
            transformed = transform.apply(wall_world_corners(raw_item))
            if _opening_near_parent(transformed, parent_id, walls_computed):
                out.append(
                    ExtractedElement(
                        id=item_id,
                        corners=transformed,
                        source="scan-cache",
                        parent_wall_id=parent_id,
                    )
                )
                continue
        out.append(
            ExtractedElement(
                id=item_id,
                corners=wall_world_corners(item),
                source="merged-room",
                parent_wall_id=parent_id,
            )
        )

    added_ids = {item.id for item in out}
    for room_name, raw_room in raw_rooms:
        transform = transforms.get(room_name)
        if (
            transform is None
            or transform.method != "floor-svd"
            or not _raw_room_matches_merged(raw_room, merged_room)
        ):
            continue
        for item in raw_room.get(key, []):
            item_id = item.get("identifier")
            if (
                not item_id
                or item_id in added_ids
                or item_id in global_added
                or item_id in merged_ids
            ):
                continue
            transformed = transform.apply(wall_world_corners(item))
            if not _opening_near_parent(
                transformed, item.get("parentIdentifier"), walls_computed
            ):
                continue
            out.append(
                ExtractedElement(
                    id=item_id,
                    corners=transformed,
                    source="scan-cache-dedup",
                    parent_wall_id=item.get("parentIdentifier"),
                )
            )
            added_ids.add(item_id)
            global_added.add(item_id)
    return out


def _extract_storages(
    merged_room: dict,
    raw_rooms: list[tuple[str, dict]],
    transforms: dict[str, RoomTransform],
    global_added: set[str],
) -> list[ExtractedElement]:
    out: list[ExtractedElement] = []
    added: set[str] = set()
    for room_name, raw_room in raw_rooms:
        transform = transforms.get(room_name)
        if (
            transform is None
            or transform.method != "floor-svd"
            or not _raw_room_matches_merged(raw_room, merged_room)
        ):
            continue
        for item in raw_room.get("objects", []):
            category = item.get("category", {})
            if not (isinstance(category, dict) and "storage" in category):
                continue
            item_id = item.get("identifier")
            if not item_id or item_id in added or item_id in global_added:
                continue
            out.append(
                ExtractedElement(
                    id=item_id,
                    corners=transform.apply(wall_world_corners(item)),
                    source="scan-cache",
                )
            )
            added.add(item_id)
            global_added.add(item_id)
    return out


def _merged_ids(merged_rooms: list[dict], key: str) -> set[str]:
    ids: set[str] = set()
    for room in merged_rooms:
        for item in room.get(key, []):
            item_id = item.get("identifier")
            if item_id:
                ids.add(item_id)
    return ids


def _remap_raw_ceilings(
    raw_ceilings: dict[str, dict],
    raw_to_merged: dict[str, int],
    transforms: dict[str, RoomTransform],
) -> dict[int, tuple[list[RawCeilingPlane], str | None]]:
    by_merged: dict[int, tuple[list[RawCeilingPlane], str | None]] = {}
    for room_name, merged_idx in raw_to_merged.items():
        ceiling = raw_ceilings.get(room_name)
        transform = transforms.get(room_name)
        if ceiling is None or transform is None:
            continue
        remapped: list[RawCeilingPlane] = []
        raw_source = ceiling.get("source")
        for plane_index, plane in enumerate(ceiling.get("planes") or []):
            world_corners = corners_to_world(plane["corners_local"], plane["transform"])
            remapped_corners = [
                [
                    round(float(corner[0]), 4),
                    round(float(corner[1]), 4),
                    round(float(corner[2]), 4),
                ]
                for corner in transform.apply(world_corners)
            ]
            if len(remapped_corners) >= 3:
                remapped.append(
                    RawCeilingPlane(
                        corners=remapped_corners,
                        raw_room_key=room_name,
                        raw_plane_index=plane_index,
                        raw_source=str(raw_source) if raw_source is not None else None,
                    )
                )
        if not remapped:
            continue
        existing = by_merged.get(merged_idx)
        if existing is None:
            by_merged[merged_idx] = (remapped, ceiling.get("source"))
        else:
            existing_planes, existing_source = existing
            by_merged[merged_idx] = ([*existing_planes, *remapped], existing_source)
    return by_merged


def _wings_for_extension_synthesis(
    uuid: str, rooms: list[ExtractedRoom]
) -> list | None:
    """Compute wings for wing-aware extension synthesis when v2 is enabled.

    Returns None when `WING_DETECTION_V2` is not set, so legacy extension
    behaviour is preserved bit-for-bit. Returns None on any error so a
    wing-decomposition failure never breaks the extraction pipeline.
    """
    import os as _os

    if _os.environ.get("WING_DETECTION_V2") != "1":
        return None
    try:
        from shapely.geometry import Polygon as _ShPolygon

        from reconcile_tiers._core.room_graph import build_room_graph
        from reconcile_tiers._core.shapely2 import make_valid_polygon
        from reconcile_tiers._core.wing_decomposition_v2 import decompose_to_wings_v2
        from reconcile_tiers.roof.footprint import build_building_footprint

        class _PartialModelForWings:
            __slots__ = ("rooms", "uuid")

            def __init__(self, uuid_: str, rooms_: list[ExtractedRoom]) -> None:
                self.uuid = uuid_
                self.rooms = rooms_

        partial = _PartialModelForWings(uuid, rooms)
        footprint = build_building_footprint(partial)
        if footprint is None or len(footprint.polygon_xz) < 3:
            return None
        poly = make_valid_polygon(_ShPolygon(footprint.polygon_xz))
        if poly is None or poly.is_empty:
            return None
        graph = build_room_graph(rooms)
        return decompose_to_wings_v2(poly, room_graph=graph, rooms=rooms)
    except Exception:
        return None


def extract_building_model(
    uuid: str, pipeline_dir: Path | str, scan_cache_root: Path | str | None
) -> BuildingModel:
    merged = load_merged(uuid, pipeline_dir)
    assignment = compute_room_stories(merged)
    scan_dir = find_scan_cache_dir(uuid, scan_cache_root) if scan_cache_root else None
    raw_rooms = load_raw_rooms(scan_dir)
    raw_ceilings = load_raw_ceilings(scan_dir)
    transforms = compute_room_transforms(raw_rooms, merged)
    raw_wall_data = _build_raw_wall_index(raw_rooms, transforms)
    raw_doors = _build_raw_item_index(raw_rooms, transforms, "doors")
    raw_windows = _build_raw_item_index(raw_rooms, transforms, "windows")
    raw_openings = _build_raw_item_index(raw_rooms, transforms, "openings")

    global_dedup_added: set[str] = set()
    global_doors_added: set[str] = set()
    global_windows_added: set[str] = set()
    global_openings_added: set[str] = set()
    global_storages_added: set[str] = set()
    rooms: list[ExtractedRoom] = []
    merged_rooms = [room.data for room in merged.rooms]
    raw_to_merged = _build_raw_to_merged_index(raw_rooms, merged_rooms)
    scan_meta = load_scan_metadata(scan_dir)
    merged_heating: dict[int, str] = {}
    for room_key, merged_idx in raw_to_merged.items():
        ht = scan_meta.room_heating.get(room_key)
        if ht and merged_idx not in merged_heating:
            merged_heating[merged_idx] = ht
    ceilings_by_merged = _remap_raw_ceilings(raw_ceilings, raw_to_merged, transforms)
    merged_door_ids = _merged_ids(merged_rooms, "doors")
    merged_window_ids = _merged_ids(merged_rooms, "windows")
    merged_opening_ids = _merged_ids(merged_rooms, "openings")
    for idx, merged_room in enumerate(merged_rooms):
        floor_polygon = (
            assignment.floor_polygons[idx]
            if idx < len(assignment.floor_polygons)
            else []
        )
        walls_merged, walls_computed, added_ids = _extract_room_walls(
            merged_room, floor_polygon, raw_wall_data
        )
        walls_computed = _add_dedup_walls(
            walls_computed,
            added_ids,
            merged_room,
            merged_rooms,
            raw_rooms,
            transforms,
            global_dedup_added,
        )
        walls_computed = _orient_walls_outward(walls_computed, floor_polygon)
        doors = _extract_openings(
            "doors",
            merged_room.get("doors", []),
            raw_doors,
            merged_door_ids,
            merged_room,
            walls_computed,
            raw_rooms,
            transforms,
            global_doors_added,
        )
        windows = _extract_openings(
            "windows",
            merged_room.get("windows", []),
            raw_windows,
            merged_window_ids,
            merged_room,
            walls_computed,
            raw_rooms,
            transforms,
            global_windows_added,
        )
        openings = _extract_openings(
            "openings",
            merged_room.get("openings", []),
            raw_openings,
            merged_opening_ids,
            merged_room,
            walls_computed,
            raw_rooms,
            transforms,
            global_openings_added,
        )
        storages = _extract_storages(
            merged_room, raw_rooms, transforms, global_storages_added
        )
        raw_ceiling_planes, raw_ceiling_source = ceilings_by_merged.get(idx, ([], None))
        rooms.append(
            ExtractedRoom(
                index=idx,
                story=assignment.room_stories[idx]
                if idx < len(assignment.room_stories)
                else 0,
                floor_polygon=floor_polygon,
                walls_merged=walls_merged,
                walls_computed=walls_computed,
                doors=doors,
                windows=windows,
                openings=openings,
                storages=storages,
                raw_ceiling_planes=raw_ceiling_planes,
                raw_ceiling_source=raw_ceiling_source,
                ceiling_polygon=[],
                ceiling_type=None,
                ceiling_eave_height=None,
                ceiling_ridge_height=None,
                heating=merged_heating.get(idx),
            )
        )
    from reconcile_tiers.extract.ceilings import infer_ceilings
    from reconcile_tiers.extract.extension import (
        compute_descent_strips,
        compute_uplift_strips,
    )
    from reconcile_tiers.extract.exterior import (
        compute_gap_closures,
        detect_exterior_gap_indicators,
    )
    from reconcile_tiers.extract.gaps import (
        assign_gaps_to_rooms,
        compute_cross_floor_gaps,
        compute_gap_walls,
        story_y_map_from_rooms,
    )
    from reconcile_tiers.extract.height_align import align_room_heights
    from reconcile_tiers.extract.inferred_rooms import infer_enclosed_void_rooms
    from reconcile_tiers.extract.overlaps import (
        clip_floor_overlaps,
        clip_walls_to_story_bounds,
    )
    from reconcile_tiers.extract.stitches import stitch_wall_gaps
    from reconcile_tiers.extract.wall_pairs import compute_slab_walls

    rooms = clip_floor_overlaps(rooms)
    rooms, _height_alignment_metrics = align_room_heights(rooms)
    story_y_map = story_y_map_from_rooms(rooms)
    rooms, _wall_clip_metrics = clip_walls_to_story_bounds(rooms, story_y_map)
    wings_for_extensions = _wings_for_extension_synthesis(uuid, rooms)
    rooms = compute_descent_strips(rooms, wings=wings_for_extensions)
    rooms = compute_uplift_strips(rooms, wings=wings_for_extensions)
    rooms = infer_ceilings(rooms)
    pre_gap_rooms = rooms
    story_y_map = story_y_map_from_rooms(pre_gap_rooms)
    cross_floor_gaps = compute_cross_floor_gaps(rooms)
    rooms, cross_floor_gaps = assign_gaps_to_rooms(cross_floor_gaps, rooms)
    rooms_with_inferred = infer_enclosed_void_rooms(rooms)
    if len(rooms_with_inferred) != len(rooms):
        rooms = rooms_with_inferred
        pre_gap_rooms = rooms
        story_y_map = story_y_map_from_rooms(pre_gap_rooms)
        cross_floor_gaps = compute_cross_floor_gaps(rooms)
        rooms, cross_floor_gaps = assign_gaps_to_rooms(cross_floor_gaps, rooms)
    exterior_gap_indicators = detect_exterior_gap_indicators(rooms)
    gap_closures = compute_gap_closures(exterior_gap_indicators)
    # Cause-1 (inter-room wall-thickness slabs) — emit from facing wall pairs.
    # Every vertex of every emitted surface is a real wall endpoint, so no
    # shapely set-op artifacts can leak in. Cross-story drape still runs
    # below to update gap.corners on cross_floor_gaps for downstream
    # consumers; it intentionally emits no walls itself.
    gap_walls = compute_slab_walls(rooms)
    cross_story_gaps = [g for g in cross_floor_gaps if g.type == "cross_story"]
    other_gaps = [g for g in cross_floor_gaps if g.type != "cross_story"]
    _, draped_cross_story = compute_gap_walls(
        cross_story_gaps,
        rooms,
        story_y_map,
        pre_absorption_floor_polygons=[room.floor_polygon for room in pre_gap_rooms],
    )
    cross_floor_gaps = other_gaps + draped_cross_story
    stitch_walls = stitch_wall_gaps(rooms)

    return BuildingModel(
        uuid=uuid,
        address=parse_address_from_scan_dir(scan_dir) if scan_dir else None,
        stories_found=assignment.stories_found,
        split_level=is_split_level(assignment.floor_polygons, assignment.room_stories),
        rooms=rooms,
        cross_floor_gaps=cross_floor_gaps,
        gap_walls=gap_walls,
        stitch_walls=stitch_walls,
        exterior_gap_indicators=exterior_gap_indicators,
        gap_closures=gap_closures,
        scan_rooms_found=len(raw_rooms),
        scan_rooms_transformed=len(transforms),
    )
