"""High-level extraction pipeline for one building."""

import json
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon

from reconcile_v2.graph_builder import (
    build_topology_graph as build_enriched_topology_graph,
)

from .ceilings import (
    compute_story_wall_top_cohort,
    extend_wall_to_dominant,
    extend_wall_to_slab,
    find_best_slab_above,
    infer_ceilings,
    reassign_raw_ceiling_planes_spatially,
    should_extend_wall_to_dominant,
)
from .common import (
    clamp_opening_to_parent,
    corners_to_world,
    hybrid_wall_corners,
    wall_world_corners,
)
from .exterior import compute_gap_closures, detect_exterior_gap_indicators
from .gaps import assign_gaps_to_rooms, compute_cross_floor_gaps, compute_gap_walls
from .height_alignment import align_room_heights
from .lineage import (
    STEP_DEDUP_WALLS,
    STEP_EXTEND_WALL_DOMINANT,
    STEP_EXTEND_WALL_SLAB,
    STEP_EXTRACT_OPENINGS,
    STEP_EXTRACT_STORAGES,
    STEP_EXTRACT_WALLS,
    STEP_OPENING_CLAMPS,
    record,
)
from .overlaps import clip_floor_overlaps, clip_walls_to_story_bounds
from .scan_data import (
    build_merged_id_sets,
    build_raw_indices,
    build_raw_to_merged_index,
    compute_room_transforms,
    find_merged_path,
    find_scan_cache_dir,
    load_raw_ceilings,
    load_raw_rooms,
    parse_address_from_scan_dir,
)
from .stitch import stitch_wall_gaps


def _load_classification(merged_path):
    recon_path = merged_path.parent / "reconciled.json"
    if not recon_path.exists():
        return "UNKNOWN", 0
    with open(recon_path) as handle:
        reconciled = json.load(handle)
    meta = reconciled.get("reconciliation", {})
    return meta.get("classification", "UNKNOWN"), 0


def _compute_room_stories(merged):
    floor_ys = []
    for room in merged["rooms"]:
        if room.get("floors") and room["floors"][0].get("polygonCorners"):
            floor_polygon = corners_to_world(
                room["floors"][0]["polygonCorners"], room["floors"][0]["transform"]
            )
            floor_ys.append(float(np.mean([corner[1] for corner in floor_polygon])))
        else:
            floor_ys.append(0.0)

    sorted_ys = sorted(set(floor_ys))
    story_map = {}
    story = 0
    for idx, y_val in enumerate(sorted_ys):
        if idx > 0 and abs(y_val - sorted_ys[idx - 1]) > 1.0:
            story += 1
        story_map[y_val] = story

    room_stories = []
    for floor_y in floor_ys:
        closest_y = min(sorted_ys, key=lambda sy: abs(sy - floor_y))
        room_stories.append(story_map[closest_y])

    return room_stories, story + 1


def _is_split_level(rooms_out):
    """Flag buildings that contain a half-floor or atypically-spaced stories.

    A half-floor shows up as either a story with a single room (side-wing or
    mezzanine appendage), or a story-to-story floor-Y delta noticeably below a
    typical full-story height (~2.4 m). Used for downstream filtering only;
    the wall-extension fix itself does not depend on this flag.
    """
    rooms_per_story = {}
    for room in rooms_out:
        rooms_per_story[room["story"]] = rooms_per_story.get(room["story"], 0) + 1
    if len(rooms_per_story) < 2:
        return False
    if any(c == 1 for c in rooms_per_story.values()):
        return True
    floor_y_by_story = {}
    for room in rooms_out:
        fp = room.get("floor_polygon") or []
        if not fp:
            continue
        floor_y_by_story.setdefault(room["story"], []).append(
            float(np.mean([c[1] for c in fp]))
        )
    mean_y = {s: float(np.mean(ys)) for s, ys in floor_y_by_story.items() if ys}
    ordered = sorted(mean_y)
    deltas = [
        mean_y[ordered[i + 1]] - mean_y[ordered[i]] for i in range(len(ordered) - 1)
    ]
    return any(d < 2.0 for d in deltas)


def _raw_room_matches_merged(raw_room, merged_room):
    raw_wall_ids = {wall["identifier"] for wall in raw_room.get("walls", [])}
    merged_wall_ids = {wall["identifier"] for wall in merged_room.get("walls", [])}
    return bool(raw_wall_ids & merged_wall_ids)


def _opening_near_parent(opening_corners, parent_id, walls_computed, max_dist=1.5):
    parent_wall = next(
        (wall for wall in walls_computed if wall["id"] == parent_id), None
    )
    if parent_wall is None:
        return True
    oc = np.mean(opening_corners, axis=0)
    pc = np.mean(parent_wall["corners"], axis=0)
    return float(np.linalg.norm(oc - pc)) < max_dist


def _orient_walls_outward(walls: list[dict], floor_polygon: list) -> None:
    """Reverse winding of any wall whose normal points toward the room centroid.

    Uses Newell's method to compute the polygon normal, then flips corners
    in-place if the normal points inward (toward the room centroid).  This
    ensures downstream FrontSide renderers (tier-preview.js) see the correct
    face without needing to guess the right orientation at render time.
    """
    if not floor_polygon:
        return
    cx = sum(c[0] for c in floor_polygon) / len(floor_polygon)
    cy = sum(c[1] for c in floor_polygon) / len(floor_polygon)
    cz = sum(c[2] for c in floor_polygon) / len(floor_polygon)
    for wall in walls:
        corners = wall.get("corners")
        if not corners or len(corners) < 3:
            continue
        nx = ny = nz = 0.0
        n = len(corners)
        for k in range(n):
            a, b = corners[k], corners[(k + 1) % n]
            nx += (a[1] - b[1]) * (a[2] + b[2])
            ny += (a[2] - b[2]) * (a[0] + b[0])
            nz += (a[0] - b[0]) * (a[1] + b[1])
        if nx * nx + ny * ny + nz * nz < 1e-10:
            continue
        wlen = len(corners)
        wx = sum(c[0] for c in corners) / wlen
        wy = sum(c[1] for c in corners) / wlen
        wz = sum(c[2] for c in corners) / wlen
        if nx * (wx - cx) + ny * (wy - cy) + nz * (wz - cz) < 0:
            wall["corners"] = corners[::-1]


def _extract_room_walls(mr, raw_rooms, raw_transforms, raw_wall_data, merged_rooms):
    floor_polygon = []
    if mr.get("floors") and mr["floors"][0].get("polygonCorners"):
        floor_polygon = corners_to_world(
            mr["floors"][0]["polygonCorners"], mr["floors"][0]["transform"]
        )

    walls_merged = [
        {"corners": wall_world_corners(wall), "id": wall["identifier"]}
        for wall in mr.get("walls", [])
    ]

    walls_computed = []
    for wall in mr.get("walls", []):
        wall_id = wall["identifier"]
        if wall_id in raw_wall_data:
            raw_wall, rot, trans, method = raw_wall_data[wall_id]
            if method == "floor-svd":
                transformed = [
                    (rot @ np.array(corner) + trans).tolist()
                    for corner in wall_world_corners(raw_wall)
                ]
                entry = {
                    "corners": transformed,
                    "id": wall_id,
                    "source": "scan-cache",
                }
                record(entry, STEP_EXTRACT_WALLS, "created", "method=floor-svd")
                walls_computed.append(entry)
            else:
                floor_y = (
                    np.mean([corner[1] for corner in floor_polygon])
                    if floor_polygon
                    else None
                )
                corners = hybrid_wall_corners(wall, raw_wall, floor_y=floor_y)
                entry = {"corners": corners, "id": wall_id, "source": "hybrid"}
                record(entry, STEP_EXTRACT_WALLS, "created", "method=hybrid")
                walls_computed.append(entry)
        else:
            entry = {
                "corners": wall_world_corners(wall),
                "id": wall_id,
                "source": "merged-room",
            }
            record(entry, STEP_EXTRACT_WALLS, "created", "source=merged-room")
            walls_computed.append(entry)

    added_ids = {wall["id"] for wall in walls_computed}
    return floor_polygon, walls_merged, walls_computed, added_ids


def _add_dedup_walls(
    walls_computed, added_ids, mr, merged_rooms, raw_rooms, raw_transforms, global_state
):
    for room_name, raw_room in raw_rooms:
        if room_name not in raw_transforms:
            continue
        rot, trans, _residual, method = raw_transforms[room_name]
        if method != "floor-svd" or not _raw_room_matches_merged(raw_room, mr):
            continue

        for wall in raw_room.get("walls", []):
            wall_id = wall["identifier"]
            if wall_id in added_ids or wall_id in global_state["walls"]:
                continue
            in_any_merged = any(
                any(mw["identifier"] == wall_id for mw in room.get("walls", []))
                for room in merged_rooms
            )
            if in_any_merged:
                continue
            transformed = [
                (rot @ np.array(corner) + trans).tolist()
                for corner in wall_world_corners(wall)
            ]
            entry = {
                "corners": transformed,
                "id": wall_id,
                "source": "scan-cache-dedup",
            }
            record(entry, STEP_DEDUP_WALLS, "created", "scan-cache-dedup")
            walls_computed.append(entry)
            added_ids.add(wall_id)
            global_state["walls"].add(wall_id)


def _extract_openings(
    kind,
    merged_items,
    raw_items_by_id,
    merged_ids,
    mr,
    walls_computed,
    raw_rooms,
    raw_transforms,
    global_added,
):
    items_out = []
    for item in merged_items:
        item_id = item["identifier"]
        parent_id = item.get("parentIdentifier")
        if item_id in raw_items_by_id:
            raw_item, rot, trans, method = raw_items_by_id[item_id]
            if method == "floor-svd":
                transformed = [
                    (rot @ np.array(corner) + trans).tolist()
                    for corner in wall_world_corners(raw_item)
                ]
                if _opening_near_parent(transformed, parent_id, walls_computed):
                    entry = {
                        "corners": transformed,
                        "id": item_id,
                        "source": "scan-cache",
                    }
                    record(
                        entry,
                        STEP_EXTRACT_OPENINGS,
                        "created",
                        f"kind={kind}, method=floor-svd",
                    )
                    items_out.append(entry)
                else:
                    entry = {
                        "corners": wall_world_corners(item),
                        "id": item_id,
                        "source": "merged-room",
                    }
                    record(
                        entry,
                        STEP_EXTRACT_OPENINGS,
                        "created",
                        f"kind={kind}, source=merged-room",
                    )
                    items_out.append(entry)
            else:
                entry = {
                    "corners": wall_world_corners(item),
                    "id": item_id,
                    "source": "merged-room",
                }
                record(
                    entry,
                    STEP_EXTRACT_OPENINGS,
                    "created",
                    f"kind={kind}, source=merged-room",
                )
                items_out.append(entry)
        else:
            entry = {
                "corners": wall_world_corners(item),
                "id": item_id,
                "source": "merged-room",
            }
            record(
                entry,
                STEP_EXTRACT_OPENINGS,
                "created",
                f"kind={kind}, source=merged-room",
            )
            items_out.append(entry)

    added_ids = {item["id"] for item in items_out}
    for room_name, raw_room in raw_rooms:
        if room_name not in raw_transforms:
            continue
        rot, trans, _residual, method = raw_transforms[room_name]
        if method != "floor-svd" or not _raw_room_matches_merged(raw_room, mr):
            continue

        for item in raw_room.get(kind, []):
            item_id = item["identifier"]
            if item_id in added_ids or item_id in global_added or item_id in merged_ids:
                continue
            transformed = [
                (rot @ np.array(corner) + trans).tolist()
                for corner in wall_world_corners(item)
            ]
            if not _opening_near_parent(
                transformed, item.get("parentIdentifier"), walls_computed
            ):
                continue
            entry = {
                "corners": transformed,
                "id": item_id,
                "source": "scan-cache-dedup",
            }
            record(
                entry,
                STEP_EXTRACT_OPENINGS,
                "created",
                f"kind={kind}, scan-cache-dedup",
            )
            items_out.append(entry)
            added_ids.add(item_id)
            global_added.add(item_id)

    return items_out


def _extract_storages(mr, raw_rooms, raw_transforms, global_added):
    storages_out = []
    added = set()
    for room_name, raw_room in raw_rooms:
        if room_name not in raw_transforms:
            continue
        rot, trans, _residual, method = raw_transforms[room_name]
        if method != "floor-svd" or not _raw_room_matches_merged(raw_room, mr):
            continue
        for obj in raw_room.get("objects", []):
            category = obj.get("category", {})
            if not (isinstance(category, dict) and "storage" in category):
                continue
            storage_id = obj["identifier"]
            if storage_id in added or storage_id in global_added:
                continue
            transformed = [
                (rot @ np.array(corner) + trans).tolist()
                for corner in wall_world_corners(obj)
            ]
            entry = {
                "corners": transformed,
                "id": storage_id,
                "source": "scan-cache",
            }
            record(entry, STEP_EXTRACT_STORAGES, "created")
            storages_out.append(entry)
            added.add(storage_id)
            global_added.add(storage_id)
    return storages_out


def _build_parent_lookup(mr, raw_rooms):
    parent_lookup = {}
    for key in ["doors", "windows", "openings"]:
        for item in mr.get(key, []):
            parent_id = item.get("parentIdentifier")
            if parent_id:
                parent_lookup[item["identifier"]] = parent_id

    for _room_name, raw_room in raw_rooms:
        for key in ["doors", "windows", "openings"]:
            for item in raw_room.get(key, []):
                parent_id = item.get("parentIdentifier")
                if parent_id and item["identifier"] not in parent_lookup:
                    parent_lookup[item["identifier"]] = parent_id

    return parent_lookup


def _apply_opening_clamps(rooms_out):
    for room in rooms_out:
        parent_lookup = room.pop("_parent_lookup", {})
        wall_corners_by_id = {
            wall["id"]: wall["corners"] for wall in room["walls_computed"]
        }
        for wall in room["walls_merged"]:
            wall_corners_by_id.setdefault(wall["id"], wall["corners"])

        all_corners = [
            corner for wall in room["walls_computed"] for corner in wall["corners"]
        ]
        room_bbox = None
        if all_corners:
            arr = np.array(all_corners)
            room_bbox = (arr.min(axis=0), arr.max(axis=0))

        for opening_list in (room["doors"], room["windows"]):
            for opening in opening_list:
                old_corners = opening["corners"]
                parent_id = parent_lookup.get(opening["id"])
                if parent_id and parent_id in wall_corners_by_id:
                    opening["corners"] = clamp_opening_to_parent(
                        opening["corners"], wall_corners_by_id[parent_id]
                    )
                elif room_bbox is not None:
                    bbox_min, bbox_max = room_bbox
                    opening["corners"] = [
                        np.clip(corner, bbox_min, bbox_max).tolist()
                        for corner in opening["corners"]
                    ]
                if opening["corners"] is not old_corners:
                    record(
                        opening,
                        STEP_OPENING_CLAMPS,
                        "modified",
                        f"clamped to parent {parent_id}",
                    )


def _merge_ceilings_by_room(raw_ceilings, raw_to_merged, raw_transforms):
    """Group ceiling planes per merged-room index, remapping into merged space.

    For each raw room that matched a merged room (via wall-id overlap), take its
    ceiling planes, transform each to raw-session-world via the plane's own
    transform, then apply the SVD `(rot, trans)` computed for that raw room so
    the output sits in the merged-building coordinate frame. Multiple raw rooms
    mapping to the same merged room accumulate — each contributes its own
    plane set.
    """
    by_idx = {}
    for raw_name, merged_idx in raw_to_merged.items():
        ceiling = raw_ceilings.get(raw_name)
        if ceiling is None or not ceiling.get("planes"):
            continue
        svd = raw_transforms.get(raw_name)
        if svd is None:
            continue
        rot, trans, _residual, _method = svd
        remapped = []
        for plane in ceiling["planes"]:
            world_corners = corners_to_world(plane["corners_local"], plane["transform"])
            remapped_corners = [
                [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)]
                for c in (rot @ np.array(corner) + trans for corner in world_corners)
            ]
            if len(remapped_corners) >= 3:
                remapped.append({"corners": remapped_corners})
        if not remapped:
            continue
        existing = by_idx.get(merged_idx)
        if existing is None:
            by_idx[merged_idx] = {"planes": remapped, "source": ceiling.get("source")}
        else:
            existing["planes"].extend(remapped)
    return by_idx


def extract_building(uuid, pipeline_dir, scan_cache_root, load_topology_graph=True):
    merged_path = find_merged_path(uuid, pipeline_dir)
    if not merged_path:
        return None

    with open(merged_path) as handle:
        merged = json.load(handle)

    classification, stories_changed = _load_classification(merged_path)
    room_stories, stories_found = _compute_room_stories(merged)

    scan_dir = find_scan_cache_dir(uuid, scan_cache_root) if scan_cache_root else None
    raw_rooms = load_raw_rooms(scan_dir) if scan_dir else []
    raw_transforms = compute_room_transforms(raw_rooms, merged) if raw_rooms else {}
    topology_graph = None
    if scan_dir is not None and load_topology_graph:
        try:
            topology_graph = build_enriched_topology_graph(
                merged_path=merged_path,
                scan_dir=scan_dir,
                uuid=uuid,
            )
        except Exception:
            topology_graph = None

    raw_indices = build_raw_indices(raw_rooms, raw_transforms)
    merged_ids = build_merged_id_sets(merged)

    raw_ceilings = load_raw_ceilings(scan_dir) if scan_dir else {}
    raw_to_merged = build_raw_to_merged_index(raw_rooms, merged) if raw_rooms else {}
    ceilings_by_merged_idx = _merge_ceilings_by_room(
        raw_ceilings, raw_to_merged, raw_transforms
    )
    global_state = {
        "walls": set(),
        "doors": set(),
        "windows": set(),
        "openings": set(),
        "storages": set(),
    }

    rooms_out = []
    for room_index, merged_room in enumerate(merged["rooms"]):
        story = room_stories[room_index] if room_index < len(room_stories) else 0
        floor_polygon, walls_merged, walls_computed, added_wall_ids = (
            _extract_room_walls(
                merged_room,
                raw_rooms,
                raw_transforms,
                raw_indices["walls"],
                merged["rooms"],
            )
        )
        _add_dedup_walls(
            walls_computed,
            added_wall_ids,
            merged_room,
            merged["rooms"],
            raw_rooms,
            raw_transforms,
            global_state,
        )
        if floor_polygon:
            _orient_walls_outward(walls_computed, floor_polygon)
            _orient_walls_outward(walls_merged, floor_polygon)

        doors_out = _extract_openings(
            "doors",
            merged_room.get("doors", []),
            raw_indices["doors"],
            merged_ids["doors"],
            merged_room,
            walls_computed,
            raw_rooms,
            raw_transforms,
            global_state["doors"],
        )
        windows_out = _extract_openings(
            "windows",
            merged_room.get("windows", []),
            raw_indices["windows"],
            merged_ids["windows"],
            merged_room,
            walls_computed,
            raw_rooms,
            raw_transforms,
            global_state["windows"],
        )
        openings_out = _extract_openings(
            "openings",
            merged_room.get("openings", []),
            raw_indices["openings"],
            merged_ids["openings"],
            merged_room,
            walls_computed,
            raw_rooms,
            raw_transforms,
            global_state["openings"],
        )
        storages_out = _extract_storages(
            merged_room, raw_rooms, raw_transforms, global_state["storages"]
        )

        raw_ceiling = ceilings_by_merged_idx.get(room_index)
        raw_ceiling_planes = raw_ceiling["planes"] if raw_ceiling else []
        raw_ceiling_source = raw_ceiling["source"] if raw_ceiling else None

        rooms_out.append(
            {
                "story": story,
                "floor_polygon": floor_polygon,
                "walls_merged": walls_merged,
                "walls_computed": walls_computed,
                "doors": doors_out,
                "windows": windows_out,
                "openings": openings_out,
                "storages": storages_out,
                "raw_ceiling_planes": raw_ceiling_planes,
                "raw_ceiling_source": raw_ceiling_source,
                "_parent_lookup": _build_parent_lookup(merged_room, raw_rooms),
            }
        )

    floor_overlap_metrics = clip_floor_overlaps(rooms_out, graph=topology_graph)
    reassign_raw_ceiling_planes_spatially(rooms_out)
    height_alignment_metrics = align_room_heights(rooms_out)
    cross_floor_gaps = compute_cross_floor_gaps(rooms_out)

    story_slab_raw = defaultdict(list)
    for room in rooms_out:
        floor_polygon = room["floor_polygon"]
        if floor_polygon and len(floor_polygon) >= 3:
            story_slab_raw[room["story"]].append(
                (
                    float(np.mean([corner[0] for corner in floor_polygon])),
                    float(np.mean([corner[2] for corner in floor_polygon])),
                    float(np.mean([corner[1] for corner in floor_polygon])),
                )
            )
    for gap in cross_floor_gaps:
        if gap["type"] != "within_story":
            continue
        corners = gap["corners"]
        if len(corners) < 3:
            continue
        story_slab_raw[gap["story"]].append(
            (
                float(np.mean([c[0] for c in corners])),
                float(np.mean([c[2] for c in corners])),
                float(np.mean([c[1] for c in corners])),
            )
        )

    # story_y_map uses the largest cluster of floor Ys for wall clipping baseline.
    story_y_map = {}
    for story, entries in story_slab_raw.items():
        floor_ys = sorted(floor_y for _cx, _cz, floor_y in entries)
        # Find the largest cluster within 0.30m
        best_cluster = floor_ys
        if len(floor_ys) >= 2:
            clusters = []
            current = [floor_ys[0]]
            for fy in floor_ys[1:]:
                if fy - current[-1] <= 0.30:
                    current.append(fy)
                else:
                    clusters.append(current)
                    current = [fy]
            clusters.append(current)
            best_cluster = max(clusters, key=len)
        story_y_map[story] = float(np.median(best_cluster))

    wall_clip_metrics = clip_walls_to_story_bounds(rooms_out, story_y_map)

    # Build Shapely polygons per story for polygon-aware slab selection.
    # Rooms only — gap polygons tend to be too loose for accurate XZ proximity.
    story_slab_polys: dict[int, list[tuple[Polygon, float]]] = defaultdict(list)
    for room in rooms_out:
        fp = room.get("floor_polygon") or []
        if len(fp) >= 3:
            try:
                poly = Polygon([(c[0], c[2]) for c in fp])
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                continue
            if poly.is_empty or not poly.is_valid:
                continue
            fy = float(np.mean([c[1] for c in fp]))
            story_slab_polys[room["story"]].append((poly, fy))

    story_cohort_cache: dict[int, dict | None] = {}

    for room in rooms_out:
        # Consider slabs from every story strictly above this room's story.
        # Split-level buildings can have a half-floor wing at story+1 whose
        # footprint sits beside (not over) the wall — its slab then fails
        # find_best_slab_above's XZ-distance test or sits below the wall top.
        # Including story+2, +3, ... lets a wall still reach the next full
        # slab directly overhead. Sorted ascending so ties break toward the
        # lowest viable slab (physically correct).
        slabs_above = []
        for s in sorted(story_slab_polys):
            if s > room["story"]:
                slabs_above.extend(story_slab_polys[s])
        # Include same-story rooms significantly above this room (half-levels).
        floor_polygon = room.get("floor_polygon") or []
        if floor_polygon:
            room_floor_y = float(np.mean([c[1] for c in floor_polygon]))
            for poly, fy in story_slab_polys.get(room["story"], []):
                if fy > room_floor_y + 1.0:
                    slabs_above.append((poly, fy))
        story = room["story"]
        for wall in room["walls_computed"]:
            wc = wall["corners"]
            if not wc:
                wall["extension_strip"] = None
                continue
            wall_top_y = max(c[1] for c in wc)
            slab_y = find_best_slab_above(wc, wall_top_y, slabs_above, max_gap=0.80)
            if slab_y is not None:
                ext = extend_wall_to_slab(wc, slab_y)
                wall["extension_strip"] = (
                    None if ext is None else ext["extension_strip"]
                )
                if wall["extension_strip"] is not None:
                    record(
                        wall,
                        STEP_EXTEND_WALL_SLAB,
                        "modified",
                        f"slab_y={slab_y:.2f}",
                    )
                continue
            # No slab above: fall back to the dominant-height cohort heuristic.
            if story not in story_cohort_cache:
                story_cohort_cache[story] = compute_story_wall_top_cohort(
                    rooms_out, story
                )
            cohort = story_cohort_cache[story]
            target_y = should_extend_wall_to_dominant(wc, cohort)
            if target_y is None:
                wall["extension_strip"] = None
                continue
            ext = extend_wall_to_dominant(wc, target_y)
            wall["extension_strip"] = None if ext is None else ext["extension_strip"]
            if wall["extension_strip"] is not None:
                record(
                    wall,
                    STEP_EXTEND_WALL_DOMINANT,
                    "modified",
                    f"dom_y={target_y:.2f}, cov={cohort['coverage_frac']:.2f}",
                )

    infer_ceilings(rooms_out)
    exterior_indicators = detect_exterior_gap_indicators(rooms_out)
    # Pre-absorption floor polygon snapshot — `assign_gaps_to_rooms` mutates
    # `room["floor_polygon"]` to absorb within-story gaps, but `compute_gap_walls`
    # needs the scan-derived (pre-absorption) room boundary to decide which gap
    # edges coincide with real walls vs. which ones bound the wall-thickness
    # void that still needs synthetic wall quads.
    pre_absorption_floor_polygons = [
        list(room.get("floor_polygon") or []) for room in rooms_out
    ]
    assign_gaps_to_rooms(cross_floor_gaps, rooms_out, graph=topology_graph)
    gap_closures = compute_gap_closures(
        exterior_indicators, rooms_out, graph=topology_graph
    )
    gap_walls = compute_gap_walls(
        cross_floor_gaps,
        rooms_out,
        story_y_map,
        gap_closures,
        graph=topology_graph,
        pre_absorption_floor_polygons=pre_absorption_floor_polygons,
    )
    stitch_walls = stitch_wall_gaps(rooms_out)
    _apply_opening_clamps(rooms_out)

    computed_total = sum(len(room["walls_computed"]) for room in rooms_out)
    merged_total = sum(len(room["walls_merged"]) for room in rooms_out)
    scan_cache_count = sum(
        1
        for room in rooms_out
        for wall in room["walls_computed"]
        if wall.get("source") == "scan-cache"
    )
    address = parse_address_from_scan_dir(scan_dir) if scan_dir else None

    return {
        "uuid": uuid,
        "address": address,
        "classification": classification,
        "rooms": rooms_out,
        "stories_found": stories_found,
        "stories_changed": stories_changed,
        "split_level": _is_split_level(rooms_out),
        "computed_walls_total": computed_total,
        "merged_walls_total": merged_total,
        "scan_cache_walls": scan_cache_count,
        "scan_rooms_found": len(raw_rooms),
        "scan_rooms_transformed": len(raw_transforms),
        "cross_floor_gaps": cross_floor_gaps,
        "gap_walls": gap_walls,
        "stitch_walls": stitch_walls,
        "exterior_gap_indicators": exterior_indicators,
        "gap_closures": gap_closures,
        "topology_graph_loaded": topology_graph is not None,
        "overlap_metrics": {
            "floor_overlaps": floor_overlap_metrics,
            "floor_overlap_count": len(floor_overlap_metrics),
            "total_floor_overlap_area_m2": round(
                sum(m["overlap_area_m2"] for m in floor_overlap_metrics), 2
            ),
            "walls_removed_in_overlap": sum(
                m.get("walls_removed", 0) for m in floor_overlap_metrics
            ),
            "doors_transferred": sum(
                m.get("doors_transferred", 0) for m in floor_overlap_metrics
            ),
            "windows_transferred": sum(
                m.get("windows_transferred", 0) for m in floor_overlap_metrics
            ),
            "walls_clipped": wall_clip_metrics["walls_clipped"],
            "walls_checked": wall_clip_metrics["walls_checked"],
        },
        "height_alignment_metrics": height_alignment_metrics,
    }
