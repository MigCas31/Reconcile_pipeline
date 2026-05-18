"""Extract 3D geometry for the viewer from merged.json + scan-cache rooms."""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from reconcile.extract3d.builder import _is_split_level
from reconcile.extract3d.ceilings import (
    apply_flat_classification,
    build_sloped_ceiling_lookup,
    classify_should_be_flat,
    drop_noisy_raw_ceiling_planes,
    find_best_slab_above,
    sloped_ceiling_y_at,
)
from reconcile.extract3d.gaps import (
    earclip_2d,
)
from reconcile.extract3d.height_alignment import align_room_heights
from reconcile.extract3d.stitch import (
    stitch_wall_gaps as modular_stitch_wall_gaps,
)
from reconcile.gap_ids import stable_gap_wall_id


def parse_transform(flat):
    return np.array(flat).reshape(4, 4, order="F")


def corners_to_world(polygon_corners, transform_flat):
    T = parse_transform(transform_flat)
    return [(T @ np.array([*c, 1.0]))[:3].tolist() for c in polygon_corners]


def wall_world_corners(wall):
    """Get wall corners in world coordinates.
    Uses polygonCorners if available, otherwise builds a rectangle from dimensions +
    transform.
    """
    T = parse_transform(wall["transform"])
    if wall.get("polygonCorners") and len(wall["polygonCorners"]) >= 3:
        return [(T @ np.array([*c, 1.0]))[:3].tolist() for c in wall["polygonCorners"]]
    # Rectangle from dimensions
    w = wall["dimensions"][0] / 2
    h = wall["dimensions"][1] / 2
    local = [[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]]
    return [(T @ np.array([*c, 1.0]))[:3].tolist() for c in local]


def clamp_opening_to_parent(opening_corners, parent_corners):
    """Clamp door/window corners so they don't extend beyond their parent wall.

    Projects both sets of corners onto the wall plane, computes the parent's
    bounding box in 2D wall-local coordinates, clamps the opening corners,
    then reprojects back to 3D.
    """
    if len(parent_corners) < 3 or len(opening_corners) < 3:
        return opening_corners

    parent = np.array(parent_corners)
    opening = np.array(opening_corners)

    # Compute wall plane basis vectors
    # Use first two edges of parent wall to define the plane
    e1 = parent[1] - parent[0]
    e1_len = np.linalg.norm(e1)
    if e1_len < 1e-9:
        return opening_corners
    e1_norm = e1 / e1_len

    # Normal from cross product of first two edges
    e2_raw = parent[2] - parent[1]
    normal = np.cross(e1, e2_raw)
    n_len = np.linalg.norm(normal)
    if n_len < 1e-9:
        return opening_corners
    normal = normal / n_len

    # Second basis vector perpendicular to e1 and normal
    e2_norm = np.cross(normal, e1_norm)

    # Origin = centroid of parent wall
    origin = parent.mean(axis=0)

    # Project parent corners to 2D
    def to_2d(pts):
        rel = pts - origin
        return np.column_stack([rel @ e1_norm, rel @ e2_norm])

    parent_2d = to_2d(parent)
    opening_2d = to_2d(opening)

    # Bounding box of parent in 2D
    p_min = parent_2d.min(axis=0)
    p_max = parent_2d.max(axis=0)

    # Clamp opening corners to parent bounding box
    clamped_2d = np.clip(opening_2d, p_min, p_max)

    if np.allclose(clamped_2d, opening_2d, atol=1e-6):
        return opening_corners  # No clamping needed

    # Reproject to 3D: keep original depth (distance along normal)
    result = []
    for i in range(len(opening_corners)):
        # Depth along normal from origin
        rel = opening[i] - origin
        depth = rel @ normal
        # Reconstruct 3D point from clamped 2D + original depth
        pt = (
            origin
            + clamped_2d[i, 0] * e1_norm
            + clamped_2d[i, 1] * e2_norm
            + depth * normal
        )
        result.append(pt.tolist())

    return result


def hybrid_wall_corners(merged_wall, raw_wall, floor_y=None):
    """Hybrid: merged wall's transform + shape (
        polygonCorners if available,
        else raw dims,
    ).
    If floor_y is given, align wall bottom to floor."""
    T = parse_transform(merged_wall["transform"])
    # Use polygonCorners if available (preserves slanted/roof shapes)
    # Prefer merged (already in building space orientation), fallback to raw
    merged_pc = merged_wall.get("polygonCorners", [])
    raw_pc = raw_wall.get("polygonCorners", [])
    if len(merged_pc) >= 3:
        local = merged_pc
    elif len(raw_pc) >= 3:
        local = raw_pc
    else:
        w = raw_wall["dimensions"][0] / 2
        h = raw_wall["dimensions"][1] / 2
        local = [[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]]
    corners = [(T @ np.array([*c, 1.0]))[:3].tolist() for c in local]
    if floor_y is not None:
        current_bottom = min(c[1] for c in corners)
        dy = floor_y - current_bottom
        corners = [[c[0], c[1] + dy, c[2]] for c in corners]
    return corners


def compute_svd(src, dst):
    """Compute rigid transform (R, t) from src to dst. Returns R, t, max_residual_cm."""
    src_c, dst_c = src.mean(0), dst.mean(0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dst_c - R @ src_c
    res = np.max(np.linalg.norm(dst - (R @ src.T).T - t, axis=1)) * 100
    return R, t, res


def find_scan_cache_dir(uuid, scan_cache_root):
    """Find scan-cache directory for a building UUID."""
    for entry in os.listdir(scan_cache_root):
        if uuid in entry and os.path.isdir(scan_cache_root / entry):
            return scan_cache_root / entry
    return None


def parse_address_from_scan_dir(scan_dir):
    """Extract street address from scan-cache directory name."""
    import re

    name = (
        scan_dir.name if hasattr(scan_dir, "name") else os.path.basename(str(scan_dir))
    )
    m = re.search(
        r"scans_[^_]+_(.+?)_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_",
        name,
    )
    if not m:
        return None
    raw = m.group(1)
    # Double underscore = separator (comma), single underscore = space
    addr = raw.replace("__", ", ").replace("_", " ")
    return addr


def load_raw_rooms(scan_dir):
    """Load individual room JSONs from scan-cache directory."""
    rooms = []
    for f in sorted(os.listdir(scan_dir)):
        if not f.endswith(".json"):
            continue
        if f in ("data.json", "arworldmap.json"):
            continue
        if f.startswith("ceiling_") or f.startswith("merged_"):
            continue
        with open(scan_dir / f) as fh:
            rooms.append((f, json.load(fh)))
    return rooms


def load_raw_ceilings(scan_dir):
    """Load per-room ceiling plane sets from `ceiling_<room-id>.json` (raw variant).

    Each ceiling file can contain multiple plane entries under `walls` -- one
    flat + several sloped planes for vaulted/pitched rooms. All planes are
    returned; the caller is expected to apply the same per-room SVD
    `(rot, trans)` used for walls/openings to remap into merged-building space.

    Keyed by the raw-room filename (`<room-id>.json`) so it lines up with
    entries returned by `load_raw_rooms`.
    """
    ceilings = {}
    if scan_dir is None or not os.path.isdir(scan_dir):
        return ceilings
    prefix = "ceiling_"
    skip = ("ceiling_merged_", "ceiling_metadata_")
    for f in sorted(os.listdir(scan_dir)):
        if not f.startswith(prefix) or not f.endswith(".json"):
            continue
        if any(f.startswith(s) for s in skip):
            continue
        room_id = f[len(prefix) : -len(".json")]
        room_key = f"{room_id}.json"
        with open(scan_dir / f) as fh:
            data = json.load(fh)
        source = "scan"
        meta_path = scan_dir / f"ceiling_metadata_{room_id}.json"
        if meta_path.exists():
            try:
                with open(meta_path) as mh:
                    meta = json.load(mh)
                src_obj = meta.get("ceilingSource") or {}
                if isinstance(src_obj, dict) and src_obj:
                    source = next(iter(src_obj.keys()))
            except (json.JSONDecodeError, OSError):
                pass
        planes = []
        for wall in data.get("walls") or []:
            corners_local = wall.get("polygonCorners") or []
            transform = wall.get("transform")
            if len(corners_local) < 3 or transform is None:
                continue
            planes.append({"corners_local": corners_local, "transform": transform})
        ceilings[room_key] = {"planes": planes, "source": source}
    return ceilings


def build_raw_to_merged_index(raw_rooms, merged_data):
    """Map each raw room filename to its best-matching merged room index."""
    mapping = {}
    merged_rooms = merged_data.get("rooms", [])
    for rname, rdata in raw_rooms:
        raw_wall_ids = {w["identifier"] for w in rdata.get("walls", [])}
        if not raw_wall_ids:
            continue
        best_idx = -1
        best_overlap = 0
        for idx, mr in enumerate(merged_rooms):
            merged_ids = {w["identifier"] for w in mr.get("walls", [])}
            overlap = len(raw_wall_ids & merged_ids)
            if overlap > best_overlap:
                best_idx = idx
                best_overlap = overlap
        if best_idx >= 0:
            mapping[rname] = best_idx
    return mapping


def compute_room_transforms(raw_rooms, merged_data):
    """Compute SVD transforms from raw rooms to building space.

    Strategy (hybrid):
    1. SVD on floor polygon corners when corner counts match (0.00cm residual)
    2. SVD on shared wall center positions as fallback for multi-room scans
    """
    transforms = {}  # raw_filename -> (R, t, residual, method)

    # Build merged wall UUID -> wall data for wall-center fallback
    merged_wall_map = {}
    for mr in merged_data.get("rooms", []):
        for w in mr.get("walls", []):
            merged_wall_map[w["identifier"]] = w

    for rname, rdata in raw_rooms:
        raw_uuids = {w["identifier"] for w in rdata.get("walls", [])}
        if not raw_uuids:
            continue

        # Find best matching merged room by UUID overlap
        best_idx, best_overlap = -1, 0
        for i, mr in enumerate(merged_data["rooms"]):
            mr_uuids = {w["identifier"] for w in mr.get("walls", [])}
            overlap = len(raw_uuids & mr_uuids)
            if overlap > best_overlap:
                best_idx, best_overlap = i, overlap

        if best_idx < 0:
            continue

        mr = merged_data["rooms"][best_idx]

        # Strategy 1: SVD on floor polygon corners (preferred, 0.00cm residual)
        if (
            rdata.get("floors")
            and rdata["floors"][0].get("polygonCorners")
            and mr.get("floors")
            and mr["floors"][0].get("polygonCorners")
        ):
            raw_fc = np.array(
                corners_to_world(
                    rdata["floors"][0]["polygonCorners"],
                    rdata["floors"][0]["transform"],
                )
            )
            m_fc = np.array(
                corners_to_world(
                    mr["floors"][0]["polygonCorners"],
                    mr["floors"][0]["transform"],
                )
            )
            if len(raw_fc) == len(m_fc):
                R, t, res = compute_svd(raw_fc, m_fc)
                if res < 50.0:
                    transforms[rname] = (R, t, res, "floor-svd")
                    continue

        # Strategy 2: SVD on shared wall center positions (fallback for multi-room
        # scans)
        src_pts, dst_pts = [], []
        for rw in rdata.get("walls", []):
            if rw["identifier"] in merged_wall_map:
                mw = merged_wall_map[rw["identifier"]]
                src_pts.append(parse_transform(rw["transform"])[:3, 3])
                dst_pts.append(parse_transform(mw["transform"])[:3, 3])

        if len(src_pts) >= 3:
            src_arr = np.array(src_pts)
            dst_arr = np.array(dst_pts)
            R, t, res = compute_svd(src_arr, dst_arr)
            if res < 200.0:  # higher threshold -- wall centers shift during merge
                transforms[rname] = (R, t, res, "wall-center-svd")

    return transforms


def _floor_polygon_to_shapely(floor_polygon_3d):
    """Convert 3D floor polygon [[x,y,z],...] to 2D Shapely Polygon on XZ plane."""
    if not floor_polygon_3d or len(floor_polygon_3d) < 3:
        return None
    coords = [(c[0], c[2]) for c in floor_polygon_3d]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.area < 0.01:
            return None
        return poly
    except Exception:
        return None


def _decompose_polys(geom):
    """Extract all Polygon objects from a geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def _element_xz_midpoint(corners):
    """Return (x, z) midpoint of a wall/door/window from its 3D corners."""
    if not corners or len(corners) < 2:
        return None
    x = np.mean([c[0] for c in corners])
    z = np.mean([c[2] for c in corners])
    return (x, z)


def _elements_in_overlap(elements, overlap_poly, buffer=0.15):
    """Split elements into (inside_overlap, outside_overlap) based on XZ midpoint."""
    buffered = overlap_poly.buffer(buffer)
    inside, outside = [], []
    for el in elements:
        mid = _element_xz_midpoint(el.get("corners", []))
        if mid and buffered.contains(Point(mid)):
            inside.append(el)
        else:
            outside.append(el)
    return inside, outside


def _orient_walls_outward(walls: list[dict], floor_polygon: list) -> None:
    """Reverse winding of any wall whose normal points toward the room centroid.

    Newell's method for the 3D polygon normal; flips corners in-place when the
    normal points inward.  Mirrors the same function in extract3d/builder.py.
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


def _wall_xz_normal(corners):
    """Compute the XZ-plane normal of a wall from its corners."""
    if len(corners) < 2:
        return None
    dx = corners[1][0] - corners[0][0]
    dz = corners[1][2] - corners[0][2]
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return None
    return (-dz / length, dx / length)


def _wall_base_segment_xz(corners):
    if len(corners) < 2:
        return None
    start = (float(corners[0][0]), float(corners[0][2]))
    end = (float(corners[1][0]), float(corners[1][2]))
    if math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-6:
        return None
    return LineString([start, end])


def _wall_segment_overlap_with_region(wall, region_poly, buffer=0.15):
    if region_poly is None or region_poly.is_empty:
        return 0.0, None
    segment = _wall_base_segment_xz(wall.get("corners", []))
    if segment is None:
        return 0.0, None
    clipped = segment.intersection(region_poly.buffer(buffer))
    if clipped.is_empty:
        return 0.0, None
    return float(clipped.length), clipped


def _project_line_interval(line, origin, direction):
    coords = []
    if getattr(line, "geom_type", "") == "LineString":
        coords = list(line.coords)
    else:
        stack = list(getattr(line, "geoms", []))
        while stack:
            geom = stack.pop()
            geom_type = getattr(geom, "geom_type", "")
            if geom_type == "LineString":
                coords.extend(list(geom.coords))
            else:
                stack.extend(list(getattr(geom, "geoms", [])))
    if not coords:
        raise ValueError("line geometry has no coordinate sequence")
    ts = [
        (coord[0] - origin[0]) * direction[0] + (coord[1] - origin[1]) * direction[1]
        for coord in coords
    ]
    return min(ts), max(ts)


def _winner_wall_covers_overlap_segment(
    wall,
    winner_walls,
    overlap_poly,
    *,
    buffer=0.15,
    max_offset_m=0.2,
    max_angle_deg=12.0,
    min_overlap_fraction=0.35,
):
    wall_segment = _wall_base_segment_xz(wall.get("corners", []))
    if wall_segment is None:
        return True, {"reason": "invalid_loser_segment"}

    overlap_len, overlap_segment = _wall_segment_overlap_with_region(
        wall, overlap_poly, buffer=buffer
    )
    if overlap_segment is None or overlap_len <= 1e-6:
        return False, {
            "reason": "no_overlap_segment",
            "overlap_length_m": round(overlap_len, 6),
        }

    direction = np.array(wall_segment.coords[1]) - np.array(wall_segment.coords[0])
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= 1e-6:
        return True, {"reason": "degenerate_loser_segment"}
    unit_dir = direction / segment_length
    origin = np.array(wall_segment.coords[0], dtype=float)
    try:
        target_start, target_end = _project_line_interval(
            overlap_segment, origin, unit_dir
        )
    except ValueError:
        return False, {
            "reason": "invalid_overlap_segment",
            "overlap_length_m": round(overlap_len, 6),
        }
    target_length = max(target_end - target_start, 0.0)
    if target_length <= 1e-6:
        return False, {
            "reason": "zero_target_interval",
            "overlap_length_m": round(overlap_len, 6),
        }

    best = {
        "reason": "no_winner_match",
        "overlap_length_m": round(overlap_len, 6),
        "best_fraction": 0.0,
        "best_offset_m": None,
        "best_angle_deg": None,
    }
    loser_normal = _wall_xz_normal(wall.get("corners", []))

    for winner_wall in winner_walls:
        winner_segment = _wall_base_segment_xz(winner_wall.get("corners", []))
        if winner_segment is None:
            continue
        winner_normal = _wall_xz_normal(winner_wall.get("corners", []))
        if loser_normal is None or winner_normal is None:
            continue
        dot = abs(
            loser_normal[0] * winner_normal[0] + loser_normal[1] * winner_normal[1]
        )
        angle_deg = math.degrees(math.acos(min(dot, 1.0)))
        if angle_deg > max_angle_deg:
            continue
        offset_m = float(winner_segment.distance(overlap_segment))
        if offset_m > max_offset_m:
            continue

        try:
            winner_start, winner_end = _project_line_interval(
                winner_segment, origin, unit_dir
            )
        except ValueError:
            continue
        projected_overlap = max(
            0.0, min(target_end, winner_end) - max(target_start, winner_start)
        )
        overlap_fraction = (
            projected_overlap / target_length if target_length > 1e-6 else 0.0
        )
        if overlap_fraction > best["best_fraction"]:
            best = {
                "reason": "winner_match",
                "overlap_length_m": round(overlap_len, 6),
                "best_fraction": round(overlap_fraction, 6),
                "best_offset_m": round(offset_m, 6),
                "best_angle_deg": round(angle_deg, 6),
            }
        if overlap_fraction >= min_overlap_fraction:
            return True, best

    return False, best


def _reassign_raw_ceiling_planes_spatially(rooms_out):
    """Move each raw ceiling plane to the room whose floor it actually sits over.

    See reconcile/extract3d/ceilings.py::reassign_raw_ceiling_planes_spatially
    for rationale -- this is the inline twin used by the CLI pipeline.
    """
    room_polys = []
    for room in rooms_out:
        fp = room.get("floor_polygon") or []
        if len(fp) < 3:
            room_polys.append(None)
            continue
        try:
            poly = Polygon([(c[0], c[2]) for c in fp])
            if not poly.is_valid:
                poly = poly.buffer(0)
            room_polys.append(poly if poly.is_valid and not poly.is_empty else None)
        except Exception:
            room_polys.append(None)

    reassignments = [[] for _ in rooms_out]
    for src_idx, room in enumerate(rooms_out):
        src_story = room.get("story", 0)
        for plane in room.get("raw_ceiling_planes") or []:
            corners = plane.get("corners") or []
            if len(corners) < 3:
                reassignments[src_idx].append(plane)
                continue
            xs = [c[0] for c in corners]
            zs = [c[2] for c in corners]
            centroid = Point(sum(xs) / len(xs), sum(zs) / len(zs))
            target_idx = None
            for ridx, poly in enumerate(room_polys):
                if poly is None or rooms_out[ridx].get("story", 0) != src_story:
                    continue
                if poly.contains(centroid):
                    target_idx = ridx
                    break
            if target_idx is None:
                try:
                    plane_poly = Polygon([(c[0], c[2]) for c in corners])
                    if not plane_poly.is_valid:
                        plane_poly = plane_poly.buffer(0)
                except Exception:
                    plane_poly = None
                if (
                    plane_poly is not None
                    and plane_poly.is_valid
                    and not plane_poly.is_empty
                ):
                    best_area = 0.0
                    for ridx, poly in enumerate(room_polys):
                        if poly is None or rooms_out[ridx].get("story", 0) != src_story:
                            continue
                        try:
                            inter = poly.intersection(plane_poly).area
                        except Exception:
                            inter = 0.0
                        if inter > best_area:
                            best_area = inter
                            target_idx = ridx
            if target_idx is None:
                target_idx = src_idx
            reassignments[target_idx].append(plane)

    for ridx, room in enumerate(rooms_out):
        room["raw_ceiling_planes"] = reassignments[ridx]


def _clip_floor_overlaps(rooms_out):
    """Clip overlapping floor polygons within each story. Larger rooms win.

    Also removes walls in the overlap zone from the clipped room, and
    transfers any doors/windows from those walls to the winning room so
    openings are preserved.
    """
    MIN_OVERLAP = 0.01  # m^2 - skip sliver overlaps
    MAX_HALF_FLOOR = (
        0.50  # rooms with floor Y this far from story median are half-floors
    )

    # Collect all rooms with valid floor polygons
    story_entries_raw = defaultdict(list)
    for ri, room in enumerate(rooms_out):
        poly = _floor_polygon_to_shapely(room["floor_polygon"])
        if poly and poly.area > 0.01:
            ys = [c[1] for c in room["floor_polygon"]]
            floor_y = float(np.mean(ys))
            story_entries_raw[room["story"]].append((ri, poly, poly.area, floor_y))

    # Filter out half-floor rooms (stairwell landings, mezzanines)
    story_entries = defaultdict(list)
    for story, entries in story_entries_raw.items():
        if len(entries) < 2:
            continue
        median_y = float(np.median([fy for _, _, _, fy in entries]))
        for ri, poly, area, floor_y in entries:
            if abs(floor_y - median_y) <= MAX_HALF_FLOOR:
                story_entries[story].append((ri, poly, area, floor_y))

    metrics = []
    # Track claim order so we can find the winning room for door/window transfer
    story_claim_order = defaultdict(list)  # story -> [(room_index, poly)]

    for story, entries in story_entries.items():
        entries.sort(key=lambda e: e[2], reverse=True)  # largest first
        claimed = None

        for ri, poly, orig_area, floor_y in entries:
            if claimed is None:
                claimed = poly
                story_claim_order[story].append((ri, poly))
                continue

            overlap = poly.intersection(claimed)
            if overlap.area < MIN_OVERLAP:
                claimed = make_valid(unary_union([claimed, poly]))
                story_claim_order[story].append((ri, poly))
                continue

            clipped = make_valid(poly.difference(claimed))

            # If clipping produced MultiPolygon, take the largest component
            parts = _decompose_polys(clipped)
            if len(parts) > 1:
                clipped = max(parts, key=lambda p: p.area)
            elif len(parts) == 1:
                clipped = parts[0]
            else:
                clipped = Polygon()

            # --- Remove walls in the overlap zone from the clipped room ---
            # Only remove walls that are "covered" by a wall in the winning
            # room (shared internal walls).  External walls that have no
            # nearby parallel counterpart in the winner are kept.
            overlap_for_test = make_valid(unary_union(_decompose_polys(overlap)))
            removed_region = make_valid(poly.difference(clipped))

            candidate_removed_walls = []
            kept_walls = []
            wall_decisions = []
            for wall in rooms_out[ri]["walls_computed"]:
                overlap_len, _ = _wall_segment_overlap_with_region(
                    wall, removed_region, buffer=0.15
                )
                if overlap_len > 1e-6:
                    candidate_removed_walls.append(wall)
                    wall_decisions.append(
                        {
                            "wall_id": wall.get("id"),
                            "candidate_overlap_length_m": round(overlap_len, 6),
                            "decision": "candidate",
                        }
                    )
                else:
                    kept_walls.append(wall)
            removed_doors, kept_doors = _elements_in_overlap(
                rooms_out[ri].get("doors", []), overlap_for_test
            )
            removed_windows, kept_windows = _elements_in_overlap(
                rooms_out[ri].get("windows", []), overlap_for_test
            )

            # Collect all walls from winning rooms that overlap this region
            winner_walls = []
            for winner_ri, winner_poly in story_claim_order[story]:
                if winner_poly.intersects(removed_region.buffer(0.15)):
                    winner_walls.extend(rooms_out[winner_ri]["walls_computed"])

            # Keep external walls (not covered by any winner wall)
            removed_walls = []
            for w in candidate_removed_walls:
                covered, detail = _winner_wall_covers_overlap_segment(
                    w, winner_walls, removed_region
                )
                if covered:
                    removed_walls.append(w)
                    wall_decisions.append(
                        {
                            "wall_id": w.get("id"),
                            "decision": "removed",
                            **detail,
                        }
                    )
                else:
                    kept_walls.append(w)
                    wall_decisions.append(
                        {
                            "wall_id": w.get("id"),
                            "decision": "kept",
                            **detail,
                        }
                    )

            rooms_out[ri]["walls_computed"] = kept_walls
            rooms_out[ri]["walls_removed_overlap"] = removed_walls
            rooms_out[ri]["doors"] = kept_doors
            rooms_out[ri]["windows"] = kept_windows

            # Transfer doors/windows to the winning room that owns the overlap
            if removed_doors or removed_windows:
                for winner_ri, winner_poly in story_claim_order[story]:
                    if not winner_poly.intersects(overlap_for_test):
                        continue
                    winner = rooms_out[winner_ri]
                    for d in removed_doors:
                        mid = _element_xz_midpoint(d.get("corners", []))
                        if mid and winner_poly.buffer(0.15).contains(Point(mid)):
                            existing = {x["id"] for x in winner.get("doors", [])}
                            if d["id"] not in existing:
                                winner.setdefault("doors", []).append(d)
                    for w in removed_windows:
                        mid = _element_xz_midpoint(w.get("corners", []))
                        if mid and winner_poly.buffer(0.15).contains(Point(mid)):
                            existing = {x["id"] for x in winner.get("windows", [])}
                            if w["id"] not in existing:
                                winner.setdefault("windows", []).append(w)

            metrics.append(
                {
                    "room_index": ri,
                    "story": story,
                    "original_area_m2": round(orig_area, 3),
                    "clipped_area_m2": round(clipped.area, 3),
                    "overlap_area_m2": round(overlap.area, 3),
                    "walls_removed": len(removed_walls),
                    "wall_decisions": wall_decisions,
                    "doors_transferred": len(removed_doors),
                    "windows_transferred": len(removed_windows),
                }
            )

            # Store original for viewer ghost
            rooms_out[ri]["floor_polygon_original"] = list(
                rooms_out[ri]["floor_polygon"]
            )

            if clipped.area > MIN_OVERLAP:
                coords_2d = list(clipped.exterior.coords)[:-1]
                rooms_out[ri]["floor_polygon"] = [
                    [c[0], floor_y, c[1]] for c in coords_2d
                ]
                rooms_out[ri]["floor_clipped"] = True

                # Overlap region for visualization
                overlap_parts = _decompose_polys(overlap)
                if overlap_parts:
                    biggest_overlap = max(overlap_parts, key=lambda p: p.area)
                    oc = list(biggest_overlap.exterior.coords)[:-1]
                    rooms_out[ri]["floor_overlap_region"] = [
                        [c[0], floor_y, c[1]] for c in oc
                    ]
            else:
                rooms_out[ri]["floor_polygon"] = []
                rooms_out[ri]["floor_clipped"] = True

            if clipped.area > MIN_OVERLAP:
                claimed = make_valid(unary_union([claimed, clipped]))
            story_claim_order[story].append(
                (
                    ri,
                    clipped if clipped.area > MIN_OVERLAP else Polygon(),
                )
            )

    return metrics


def _clip_walls_to_story_bounds(rooms_out, story_y_map):
    """Clip wall quads to their story's Y range so staircase walls don't overlap.

    Skips half-floor rooms (stairwell landings, mezzanines) whose floor Y
    deviates more than 0.50m from the story median.

    Detects split-level houses where inter-story gaps are much smaller than
    actual wall heights, and skips ceiling clipping for those stories.
    """
    TOP_EPSILON = 0.05  # allow walls to touch slab within 5 cm
    BOTTOM_TOL = 0.30  # allow 30 cm below floor (baseboard/foundation)
    MAX_HALF_FLOOR = 0.50
    MIN_STORY_RATIO = 0.75  # ceiling gap must be >= 75% of median wall height

    sorted_stories = sorted(story_y_map.keys())

    # Compute median wall height per story to detect split-level situations
    story_wall_heights = defaultdict(list)
    for room in rooms_out:
        for w in room.get("walls_computed", []):
            corners = w.get("corners", [])
            if len(corners) >= 3:
                ys = [c[1] for c in corners]
                h = max(ys) - min(ys)
                if h > 0.1:
                    story_wall_heights[room["story"]].append(h)

    story_median_wall_h = {}
    for story, heights in story_wall_heights.items():
        if heights:
            story_median_wall_h[story] = float(np.median(heights))

    story_bounds = {}
    for i, story in enumerate(sorted_stories):
        y_floor = story_y_map[story]
        y_ceiling = (
            story_y_map[sorted_stories[i + 1]] if i + 1 < len(sorted_stories) else None
        )

        # Split-level detection: if the gap to the next story is much smaller
        # than the actual wall heights, skip that ceiling (look further up)
        if y_ceiling is not None and story in story_median_wall_h:
            gap = y_ceiling - y_floor
            median_h = story_median_wall_h[story]
            if gap < MIN_STORY_RATIO * median_h:
                # Try the story after next
                y_ceiling = None
                for j in range(i + 2, len(sorted_stories)):
                    candidate = story_y_map[sorted_stories[j]]
                    if (candidate - y_floor) >= MIN_STORY_RATIO * median_h:
                        y_ceiling = candidate
                        break

        story_bounds[story] = (y_floor, y_ceiling)

    walls_clipped = 0
    walls_checked = 0

    for room in rooms_out:
        story = room["story"]
        if story not in story_bounds:
            continue

        # Skip half-floor rooms (stairwell landings, mezzanines)
        fp = room["floor_polygon"]
        room_floor_y = None
        if fp and len(fp) >= 3:
            room_floor_y = float(np.mean([c[1] for c in fp]))
            if abs(room_floor_y - story_y_map[story]) > MAX_HALF_FLOOR:
                continue

        y_floor, y_ceiling = story_bounds[story]

        # Split-level opt-out: if the room's own floor polygon agrees with its
        # walls' minimum y, the room is an extension at its own elevation (e.g.
        # sunroom or garage step-down). Trust the room over the story aggregate
        # for the bottom-clip baseline. Ceiling baseline stays on the story.
        effective_floor_y = y_floor
        if room_floor_y is not None:
            wall_bottoms = [
                min(c[1] for c in w["corners"])
                for w in room["walls_computed"]
                if len(w.get("corners") or []) >= 3
            ]
            if wall_bottoms and abs(room_floor_y - min(wall_bottoms)) <= 0.10:
                effective_floor_y = room_floor_y

        for w in room["walls_computed"]:
            walls_checked += 1
            corners = w["corners"]
            if len(corners) < 3:
                continue

            ys = [c[1] for c in corners]
            wall_min_y = min(ys)
            wall_max_y = max(ys)

            need_clip = False
            # Check top
            if y_ceiling is not None and wall_max_y > y_ceiling + TOP_EPSILON:
                need_clip = True
            # Check bottom
            if wall_min_y < effective_floor_y - BOTTOM_TOL:
                need_clip = True

            if not need_clip:
                continue

            new_corners = [list(c) for c in corners]

            if y_ceiling is not None:
                for c in new_corners:
                    if c[1] > y_ceiling:
                        c[1] = y_ceiling

            if wall_min_y < effective_floor_y - BOTTOM_TOL:
                for c in new_corners:
                    if c[1] < effective_floor_y:
                        c[1] = effective_floor_y

            w["corners_original"] = corners
            w["corners"] = new_corners
            w["wall_clipped"] = True
            walls_clipped += 1

    return {"walls_clipped": walls_clipped, "walls_checked": walls_checked}


def _compute_cross_floor_gaps(rooms_out):
    """Detect gaps between floor polygons: within-story and cross-story.

    Phase 1: Morphological close -> enclosed voids within a story
    Phase 2: Pairwise buffer-intersect -> strips between adjacent rooms (clipped to
    footprint)
    Phase 3: Cross-story differences -> areas covered by other floors but not this one
             (closets, thick walls, scanning differences)
    """
    from shapely import STRtree

    WALL_HALF = 0.25  # half-wall buffer for morphological close (Phase 1)
    PAIR_HALF = 0.50  # half-wall buffer for pairwise intersect (Phase 2)
    MAX_GAP = 1.00  # max gap width for pairwise check
    MIN_AREA = 0.005  # minimum gap area in m^2
    MAX_HALF_FLOOR = 0.50  # max floor Y deviation from story median before
    # treating a room as a half-floor (mezzanine/stairwell)

    story_rooms_raw = defaultdict(list)  # story -> [(poly, floor_y)]

    for _ri, room in enumerate(rooms_out):
        story = room["story"]
        poly = _floor_polygon_to_shapely(room["floor_polygon"])
        if poly is not None and poly.is_valid and poly.area > 0.01:
            ys = [c[1] for c in room["floor_polygon"]]
            story_rooms_raw[story].append((poly, float(np.mean(ys))))

    # Filter out half-floor rooms (floor Y far from story median)
    story_rooms = defaultdict(list)
    story_floor_ys = defaultdict(list)
    for story, entries in story_rooms_raw.items():
        floor_ys_all = [fy for _, fy in entries]
        median_y = float(np.median(floor_ys_all))
        for poly, fy in entries:
            if abs(fy - median_y) <= MAX_HALF_FLOOR:
                story_rooms[story].append(poly)
                story_floor_ys[story].append(fy)

    # Build per-story footprints
    story_footprints = {}
    story_y_map = {}
    for story, polys in sorted(story_rooms.items()):
        fp = make_valid(unary_union(polys))
        if fp.area > 0.01:
            story_footprints[story] = fp
            story_y_map[story] = float(np.mean(story_floor_ys[story]))

    # Identify half-floor stories: those sandwiched between two other
    # stories with BOTH adjacent Δy < 1.50 m. Half-floors have their own
    # walls + slabs, so cross-storey gap lids over their XZ would slice
    # through the existing room envelope. We subtract `half_floor_fp`
    # from each full story's `missing` below.
    HALF_FLOOR_DY = 1.50
    stories_by_y = sorted(story_y_map.keys(), key=lambda s: story_y_map[s])
    half_floor_stories: set[int] = set()
    for i in range(1, len(stories_by_y) - 1):
        s = stories_by_y[i]
        dy_below = abs(story_y_map[s] - story_y_map[stories_by_y[i - 1]])
        dy_above = abs(story_y_map[s] - story_y_map[stories_by_y[i + 1]])
        if dy_below < HALF_FLOOR_DY and dy_above < HALF_FLOOR_DY:
            half_floor_stories.add(s)
    if half_floor_stories:
        half_floor_fp = make_valid(
            unary_union([story_footprints[s] for s in half_floor_stories])
        )
    else:
        half_floor_fp = Polygon()

    gaps = []

    def _emit_single_gap(part, story, floor_y, gap_type):
        area = part.area
        compactness = 4 * math.pi * area / (part.length**2) if part.length > 0 else 0
        if compactness < 0.15:
            confidence = "high"
        elif compactness < 0.3:
            confidence = "medium"
        else:
            confidence = "low"
        coords_2d = list(part.exterior.coords)
        corners_3d = [[c[0], floor_y, c[1]] for c in coords_2d]
        centroid = part.centroid
        gaps.append(
            {
                "story": story,
                "type": gap_type,
                "corners": corners_3d,
                "area_m2": round(area, 3),
                "compactness": round(compactness, 3),
                "confidence": confidence,
                "centroid": [round(centroid.x, 3), floor_y, round(centroid.y, 3)],
            }
        )

    def _emit_gaps(regions, story, floor_y, gap_type, clip_to=None):
        for region in regions:
            if clip_to is not None:
                try:
                    region = make_valid(region.intersection(clip_to))
                except Exception:
                    continue
            for part in _decompose_polys(region):
                if part.area < MIN_AREA:
                    continue
                _emit_single_gap(part, story, floor_y, gap_type)

    for story, polys in sorted(story_rooms.items()):
        if len(polys) < 2:
            continue

        footprint = story_footprints[story]
        floor_y = story_y_map[story]
        morph_gap_parts = []
        hole_gap_parts = []

        # Phase 1: Morphological close -> enclosed voids (tight buffer)
        closed = make_valid(
            footprint.buffer(WALL_HALF, join_style=2).buffer(-WALL_HALF, join_style=2)
        )
        morph_gaps = make_valid(closed.difference(footprint))
        morph_gap_parts.extend(_decompose_polys(morph_gaps))

        # Interior holes in the closed polygon are gaps fully enclosed by
        # the building footprint (e.g. thick walls, closets, fireplaces).
        # These must NOT be clipped back to `closed` -- they live inside its
        # holes and would be erased.
        for poly_part in _decompose_polys(closed):
            for interior in poly_part.interiors:
                hole = Polygon(interior)
                if hole.is_valid and hole.area > MIN_AREA:
                    hole_gap_parts.append(hole)

        # Phase 2: Pairwise buffer-intersect for adjacent room gaps (wider buffer)
        tree = STRtree(polys)
        buffered = [
            p.buffer(PAIR_HALF, join_style=2) for p in polys
        ]  # mitre join: square corners, no rounded bulges
        pair_gap_parts = []
        seen_pairs = set()

        for i, _poly_i in enumerate(polys):
            candidates = tree.query(buffered[i])
            for j in candidates:
                if j <= i:
                    continue
                pair_key = (i, j)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if polys[i].distance(polys[j]) > MAX_GAP:
                    continue

                try:
                    intersection = buffered[i].intersection(buffered[j])
                    gap = make_valid(intersection.difference(footprint))
                    pair_gap_parts.extend(_decompose_polys(gap))
                except Exception:
                    continue

        # Wider clipping envelope for pairwise gaps (they may extend beyond
        # the tight morphological close)
        wide_closed = make_valid(
            footprint.buffer(PAIR_HALF, join_style=2).buffer(-PAIR_HALF, join_style=2)
        )

        all_phase1_parts = morph_gap_parts + hole_gap_parts

        # Phase 1's `close.difference(footprint)` can produce ONE ring-
        # shaped polygon threading every inter-room void when rooms are
        # tightly packed (9-room hallway buildings). Three.js ear-clips
        # that ring into building-spanning spike triangles because a thin
        # concave ring isn't one of its strong cases. To keep the
        # geometry intact but rendering-safe, cut the morph-gap output
        # into per-pair segments: intersect with each adjacent pair's
        # buffered neighborhood. Each segment is a simple strip between
        # exactly two rooms -- no ring, no snake, triangulates cleanly.
        morph_union = (
            make_valid(unary_union(morph_gap_parts)) if morph_gap_parts else None
        )
        if morph_union is not None and not morph_union.is_empty:
            pair_neighborhoods = []
            for i, _poly_i in enumerate(polys):
                for j in range(i + 1, len(polys)):
                    if polys[i].distance(polys[j]) > MAX_GAP:
                        continue
                    try:
                        nbhd = buffered[i].intersection(buffered[j])
                    except Exception:
                        continue
                    if not nbhd.is_empty:
                        pair_neighborhoods.append(nbhd)
            covered = []
            for nbhd in pair_neighborhoods:
                try:
                    chunk = make_valid(morph_union.intersection(nbhd))
                except Exception:
                    continue
                if chunk.is_empty or chunk.area < MIN_AREA:
                    continue
                covered.append(chunk)
                _emit_gaps([chunk], story, floor_y, "within_story", clip_to=closed)
            # Anything in the morph gap not covered by any pair
            # neighborhood (e.g. concavities under a single room's
            # protrusion) still needs to go out -- emit per decomposed part.
            if covered:
                try:
                    leftover = make_valid(morph_union.difference(unary_union(covered)))
                except Exception:
                    leftover = None
                if leftover is not None and not leftover.is_empty:
                    for part in _decompose_polys(leftover):
                        if part.area >= MIN_AREA:
                            _emit_gaps(
                                [part], story, floor_y, "within_story", clip_to=closed
                            )
            else:
                for part in morph_gap_parts:
                    _emit_gaps([part], story, floor_y, "within_story", clip_to=closed)

        # Interior hole gaps (already bounded by `closed`, no clipping).
        for part in hole_gap_parts:
            _emit_gaps([part], story, floor_y, "within_story")

        # Precompute Phase 1 coverage once so each Phase 2 pair gap can
        # subtract it without recomputing per iteration.
        phase1_cover = (
            make_valid(unary_union(all_phase1_parts)) if all_phase1_parts else None
        )

        for pair_gap in pair_gap_parts:
            if phase1_cover is not None:
                try:
                    pair_gap = make_valid(pair_gap.difference(phase1_cover))
                except Exception:
                    pass
            if pair_gap.is_empty:
                continue
            _emit_gaps([pair_gap], story, floor_y, "within_story", clip_to=wide_closed)

    # Phase 3: Cross-story differences
    # For each story, areas covered by ANY other story but NOT this one.
    # Same snake-decomposition trick as Phase 1: intersect the missing
    # region with each room-from-another-story's buffered footprint so
    # each emitted chunk corresponds to a single neighboring room rather
    # than a ring threading the whole outline.
    sorted_stories = sorted(story_footprints.keys())
    if len(sorted_stories) >= 2:
        all_footprints = [story_footprints[s] for s in sorted_stories]
        full_envelope = make_valid(unary_union(all_footprints))

        for story in sorted_stories:
            fp = story_footprints[story]
            floor_y = story_y_map[story]
            try:
                missing = make_valid(full_envelope.difference(fp))
                if not half_floor_fp.is_empty and story not in half_floor_stories:
                    missing = make_valid(missing.difference(half_floor_fp))
            except Exception:
                continue
            if missing.is_empty:
                continue
            other_room_buffs = [
                p.buffer(PAIR_HALF, join_style=2)
                for other, polys_other in story_rooms.items()
                if other != story
                for p in polys_other
            ]
            covered = []
            covered_union = None
            for nbhd in other_room_buffs:
                try:
                    chunk = make_valid(missing.intersection(nbhd))
                    if covered_union is not None:
                        chunk = make_valid(chunk.difference(covered_union))
                except Exception:
                    continue
                if chunk.is_empty or chunk.area < MIN_AREA:
                    continue
                covered.append(chunk)
                covered_union = (
                    chunk
                    if covered_union is None
                    else make_valid(unary_union([covered_union, chunk]))
                )
                _emit_gaps([chunk], story, floor_y, "cross_story")
            if covered:
                try:
                    leftover = make_valid(missing.difference(covered_union))
                except Exception:
                    leftover = None
                if leftover is not None and not leftover.is_empty:
                    for part in _decompose_polys(leftover):
                        if part.area >= MIN_AREA:
                            _emit_gaps([part], story, floor_y, "cross_story")
            else:
                _emit_gaps(_decompose_polys(missing), story, floor_y, "cross_story")

    return gaps


def _assign_gaps_to_rooms(gaps, rooms_out):
    """Assign within-story gaps to their nearest room and expand floor polygons.

    Each gap polygon is unioned into the nearest room's floor_polygon so the
    ceiling pipeline (build_flat_ceilings) covers the previously-unclaimed area.
    Also stores ``room_index`` on the gap dict for downstream propagation to
    gap walls.
    """
    # Build Shapely polygons for each room's floor
    room_shapely = []
    for ri, room in enumerate(rooms_out):
        poly = _floor_polygon_to_shapely(room.get("floor_polygon", []))
        room_shapely.append((ri, poly))

    for gap in gaps:
        if gap["type"] != "within_story":
            continue
        corners_3d = gap["corners"]
        if len(corners_3d) < 3:
            continue

        gap_poly = _floor_polygon_to_shapely(corners_3d)
        if gap_poly is None:
            continue

        story = gap["story"]
        gap_centroid = gap_poly.centroid

        # Find nearest room on the same story by Shapely distance
        best_ri = None
        best_dist = float("inf")
        for ri, rpoly in room_shapely:
            if rpoly is None or rooms_out[ri]["story"] != story:
                continue
            d = rpoly.distance(gap_centroid)
            if d < best_dist:
                best_dist = d
                best_ri = ri

        if best_ri is None:
            continue

        gap["room_index"] = best_ri

        # Compute ceiling_corners: gap polygon raised to the assigned room's
        # median wall-top Y so the viewer renders gap thermal ceilings at the
        # correct height instead of at floor level.
        assigned_room = rooms_out[best_ri]
        wall_top_ys = []
        for w in assigned_room.get("walls_computed") or assigned_room.get(
            "walls_merged", []
        ):
            cs = w.get("corners", [])
            if cs:
                wall_top_ys.append(max(c[1] for c in cs))
        if wall_top_ys:
            ceiling_y = round(float(np.median(wall_top_ys)), 4)
        else:
            ceiling_y = corners_3d[0][1]  # fallback: floor Y
        gap["ceiling_corners"] = [
            [round(c[0], 4), ceiling_y, round(c[2], 4)] for c in corners_3d
        ]

        # Expand that room's floor polygon to include the gap
        room = assigned_room
        room_poly = room_shapely[best_ri][1]
        if room_poly is None:
            continue

        floor_y = (
            room["floor_polygon"][0][1] if room["floor_polygon"] else corners_3d[0][1]
        )
        merged = make_valid(unary_union([room_poly, gap_poly]))

        # Extract the largest polygon from the union result
        if isinstance(merged, MultiPolygon):
            merged = max(merged.geoms, key=lambda g: g.area)

        # Convert back to 3D floor polygon
        coords_2d = list(merged.exterior.coords)
        if coords_2d and coords_2d[0] == coords_2d[-1]:
            coords_2d = coords_2d[:-1]
        room["floor_polygon"] = [
            [round(c[0], 4), floor_y, round(c[1], 4)] for c in coords_2d
        ]

        # Update the cached Shapely polygon for subsequent gap assignments
        room_shapely[best_ri] = (best_ri, merged)


def _compute_gap_walls(
    gaps,
    rooms_out,
    story_y_map,
    gap_closures=None,
    pre_absorption_floor_polygons=None,
):
    """Create wall quads along each edge of cross-floor gap polygons.

    Per-vertex Y interpolation from nearest wall edges, with filters:
    - Only walls with height >= 0.5m (skip slabs/ceilings)
    - Only walls whose bottom Y is within 0.75m of the gap's floor Y
    - Includes wall extensions and exterior gap closures as wall sources

    `pre_absorption_floor_polygons` mirrors
    ``reconcile.extract3d.gaps.compute_gap_walls`` -- see that function for
    why the pre-absorption boundary is required here.
    """
    DEFAULT_WALL_HEIGHT = 2.50
    MIN_WALL_HEIGHT = 0.5
    MAX_SNAP_DIST = 1.0
    MAX_Y_DIST = 0.75
    MIN_SNAP_DIST = 1e-6

    def _add_wall_edge(story, wc, room_index=None, wall_id=None):
        """Add a wall polygon (4+ corners) to the story_walls index.

        Uses corners[0]->corners[1] as the bottom edge (standard RoomPlan
        ordering: BL, BR, ...).  Identifies top corners by Y-midpoint
        and builds a top contour profile parameterised by t along the
        bottom edge, so pentagonal/slanted walls contribute their ridge.
        """
        if len(wc) < 4:
            return
        ys = [c[1] for c in wc]
        h = max(ys) - min(ys)
        if h < MIN_WALL_HEIGHT:
            return
        # Bottom edge from first two corners (BL->BR in RoomPlan convention)
        p0_xz = np.array([wc[0][0], wc[0][2]])
        p1_xz = np.array([wc[1][0], wc[1][2]])
        edge = p1_xz - p0_xz
        elen = np.linalg.norm(edge)
        if elen < 1e-6:
            return
        ybot_avg = (wc[0][1] + wc[1][1]) / 2
        # Top corners: everything above the Y-midpoint
        mid_y = (max(ys) + min(ys)) / 2.0
        top_cs = [c for c in wc if c[1] > mid_y - 0.01]
        if len(top_cs) < 2:
            # Fallback: use corners[2] and corners[3] as flat top
            top_cs = [wc[3], wc[2]]
        # Build top-contour profile: list of (t, y) pairs sorted by t
        top_profile = []
        for c in top_cs:
            cxz = np.array([c[0], c[2]])
            t = float(np.clip(np.dot(cxz - p0_xz, edge) / (elen**2), 0, 1))
            top_profile.append((t, c[1]))
        top_profile.sort(key=lambda p: p[0])
        edge_unit = edge / elen
        story_walls[story].append(
            {
                "corners": wc,
                "p0_xz": p0_xz,
                "edge": edge,
                "edge_unit": edge_unit,
                "elen": elen,
                "ybot_avg": ybot_avg,
                "top_profile": top_profile,
                "room_index": room_index,
                "wall_id": wall_id,
            }
        )

    # Collect wall edges per story, using extended top Y when wall has extensions
    story_walls = defaultdict(list)
    for room in rooms_out:
        story = room["story"]
        for w in room["walls_computed"]:
            wc = w["corners"]
            if len(wc) < 4:
                continue
            # Build effective corners: include wall extensions (raised top to slab)
            ext = w.get("extension_strip")
            if ext and len(ext) >= 1:
                # ext is a list of quads; find max Y across all quad corners
                ext_top_y = max(c[1] for quad in ext for c in quad)
                ec = [list(c) for c in wc]
                ys = [c[1] for c in wc]
                mid_y = (max(ys) + min(ys)) / 2.0
                # Raise all top corners to ext_top_y
                for i, c in enumerate(ec):
                    if c[1] > mid_y - 0.01:
                        ec[i] = [c[0], ext_top_y, c[2]]
            else:
                ec = wc
            _add_wall_edge(
                story, ec, room_index=room.get("room_index"), wall_id=w.get("id")
            )

    # Include exterior gap closure side walls as wall sources
    for gc in gap_closures or []:
        if gc.get("type") == "side" and len(gc.get("corners", [])) >= 4:
            _add_wall_edge(gc["story"], gc["corners"])

    # Fallback ceiling Y per story
    sorted_stories = sorted(story_y_map.keys())
    ceiling_y_map = {}
    for i, story in enumerate(sorted_stories):
        if i + 1 < len(sorted_stories):
            ceiling_y_map[story] = story_y_map[sorted_stories[i + 1]]
        else:
            heights = []
            for wall in story_walls.get(story, []):
                ys = [c[1] for c in wall["corners"]]
                heights.append(max(ys) - min(ys))
            median_h = float(np.median(heights)) if heights else DEFAULT_WALL_HEIGHT
            ceiling_y_map[story] = story_y_map[story] + median_h

    sloped_ceilings = build_sloped_ceiling_lookup(rooms_out)

    def _clamp_to_sloped_ceiling(xz_pt, story, floor_y, ytop):
        slant_y = sloped_ceiling_y_at(sloped_ceilings, xz_pt, story)
        if slant_y is None or slant_y >= ytop:
            return ytop
        return max(slant_y, floor_y + MIN_WALL_HEIGHT)

    def _interp_top_profile(top_profile, t):
        """Interpolate Y along a wall's top contour at parameter t."""
        if len(top_profile) == 1:
            return top_profile[0][1]
        # Clamp to profile range
        if t <= top_profile[0][0]:
            return top_profile[0][1]
        if t >= top_profile[-1][0]:
            return top_profile[-1][1]
        # Find bracketing segment
        for k in range(len(top_profile) - 1):
            t0, y0 = top_profile[k]
            t1, y1 = top_profile[k + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return y0 + frac * (y1 - y0)
        return top_profile[-1][1]

    def _project_to_wall_line(xz_pt, wall):
        rel = xz_pt - wall["p0_xz"]
        t_raw = float(np.dot(rel, wall["edge_unit"]))
        proj = wall["p0_xz"] + t_raw * wall["edge_unit"]
        dist = float(np.linalg.norm(xz_pt - proj))
        t_profile = float(np.clip(t_raw / wall["elen"], 0.0, 1.0))
        return proj, dist, t_profile

    def _snap_vertex_y(xz_pt, story, floor_y):
        """Find nearest wall edge on this floor and interpolate bottom/top Y.

        Uses the full top-contour profile so pentagonal/slanted walls
        contribute their ridge height at the correct parametric position.
        """
        fallback_top = ceiling_y_map.get(story, floor_y + DEFAULT_WALL_HEIGHT)
        best_dist = MAX_SNAP_DIST
        best_ybot = floor_y
        best_ytop = fallback_top

        for wall in story_walls.get(story, []):
            # Only walls on this floor level
            if abs(wall["ybot_avg"] - floor_y) > MAX_Y_DIST:
                continue
            _proj, dist, t = _project_to_wall_line(xz_pt, wall)
            if dist < best_dist:
                best_dist = dist
                best_ybot = float(wall["ybot_avg"])
                best_ytop = _interp_top_profile(wall["top_profile"], t)

        best_ytop = _clamp_to_sloped_ceiling(xz_pt, story, floor_y, best_ytop)
        return best_ybot, best_ytop

    def _pick_support_wall(xz_pt, prev_xz, next_xz, story, floor_y):
        tangent = None
        edge_ref = next_xz - prev_xz
        edge_len = float(np.linalg.norm(edge_ref))
        if edge_len > MIN_SNAP_DIST:
            tangent = edge_ref / edge_len
        best = None
        for wall in story_walls.get(story, []):
            if abs(wall["ybot_avg"] - floor_y) > MAX_Y_DIST:
                continue
            proj, dist, t_profile = _project_to_wall_line(xz_pt, wall)
            if dist > MAX_SNAP_DIST:
                continue
            cos_parallel = 1.0
            if tangent is not None:
                cos_parallel = abs(float(np.dot(tangent, wall["edge_unit"])))
            score = dist + 0.35 * (1.0 - cos_parallel)
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "wall": wall,
                    "proj": proj,
                    "dist": dist,
                    "t_profile": t_profile,
                }
        return best

    def _build_snapped_vertices(edge_verts, story, floor_y):
        snapped = []
        n_edges = len(edge_verts)
        for i, c in enumerate(edge_verts):
            prev_c = edge_verts[(i - 1) % n_edges]
            next_c = edge_verts[(i + 1) % n_edges]
            xz = np.array([c[0], c[2]], dtype=float)
            prev_xz = np.array([prev_c[0], prev_c[2]], dtype=float)
            next_xz = np.array([next_c[0], next_c[2]], dtype=float)
            picked = _pick_support_wall(xz, prev_xz, next_xz, story, floor_y)
            if picked is None:
                ybot, ytop = _snap_vertex_y(xz, story, floor_y)
                snapped.append({"xz": xz, "ybot": ybot, "ytop": ytop, "wall": None})
                continue
            ybot = float(picked["wall"]["ybot_avg"])
            ytop = _interp_top_profile(
                picked["wall"]["top_profile"], picked["t_profile"]
            )
            ytop = _clamp_to_sloped_ceiling(picked["proj"], story, floor_y, ytop)
            snapped.append(
                {
                    "xz": picked["proj"],
                    "ybot": ybot,
                    "ytop": ytop,
                    "wall": picked["wall"],
                }
            )

        coords = [(s["xz"][0], s["xz"][1]) for s in snapped]
        try:
            poly = Polygon(coords)
            if (not poly.is_valid) or poly.area <= 1e-6:
                raise ValueError("invalid snapped polygon")
        except Exception:
            snapped = []
            for c in edge_verts:
                xz = np.array([c[0], c[2]], dtype=float)
                ybot, ytop = _snap_vertex_y(xz, story, floor_y)
                snapped.append({"xz": xz, "ybot": ybot, "ytop": ytop, "wall": None})
        return snapped

    # Per-story room-floor union, used to decide where synthetic gap caps
    # would duplicate a real room floor. Prefer the pre-absorption snapshot
    # when the caller supplies it -- see sibling pre-compute in
    # reconcile.extract3d.gaps.compute_gap_walls for the rationale.
    from collections import defaultdict as _dd

    _rooms_by_story = _dd(list)
    for _idx, _room in enumerate(rooms_out):
        _s = _room.get("story")
        if _s is None:
            continue
        if pre_absorption_floor_polygons is not None and _idx < len(
            pre_absorption_floor_polygons
        ):
            _fp_source = pre_absorption_floor_polygons[_idx]
        else:
            _fp_source = _room.get("floor_polygon")
        _rp = _floor_polygon_to_shapely(_fp_source)
        if _rp is None or not _rp.is_valid or _rp.area <= 0.0:
            continue
        _rooms_by_story[_s].append(_rp)
    story_room_union = {}
    story_room_boundary = {}
    for _s, _polys in _rooms_by_story.items():
        try:
            _u = make_valid(unary_union(_polys))
        except Exception:
            _u = None
        story_room_union[_s] = _u
        try:
            story_room_boundary[_s] = (
                _u.boundary if (_u is not None and not _u.is_empty) else None
            )
        except Exception:
            story_room_boundary[_s] = None

    walls = []
    for gap in gaps:
        gap_type = gap["type"]
        if gap_type == "cross_story":
            corners_3d = gap.get("corners") or []
            if len(corners_3d) < 3:
                continue
            story = gap["story"]
            below = story - 1
            if below not in story_y_map:
                continue
            closed = len(corners_3d) >= 4 and corners_3d[0] == corners_3d[-1]
            edge_verts = corners_3d[:-1] if closed else corners_3d
            if len(edge_verts) < 3:
                continue
            snapped_below = _build_snapped_vertices(
                edge_verts, below, story_y_map[below]
            )
            draped = [
                [float(s["xz"][0]), float(s["ytop"]), float(s["xz"][1])]
                for s in snapped_below
            ]
            new_corners = [list(p) for p in draped]
            if closed:
                new_corners.append(list(draped[0]))
            gap["corners"] = new_corners
            gap["ceiling_corners"] = [list(p) for p in draped]
            centroid = gap.get("centroid")
            if isinstance(centroid, list) and len(centroid) == 3:
                centroid[1] = float(np.mean([p[1] for p in draped]))
            continue
        if gap_type != "within_story":
            continue
        corners_3d = gap["corners"]
        if len(corners_3d) < 3:
            continue
        story = gap["story"]
        floor_y = corners_3d[0][1]

        if len(corners_3d) >= 4 and corners_3d[0] == corners_3d[-1]:
            edge_verts = corners_3d[:-1]
        else:
            edge_verts = corners_3d
        if len(edge_verts) < 3:
            continue

        snapped = _build_snapped_vertices(edge_verts, story, floor_y)
        vertex_ys = [(s["ybot"], s["ytop"]) for s in snapped]

        # Use a consistent floor Y (max of per-vertex bottoms) so the
        # base stays level, but let the top follow adjacent wall heights.
        gap_floor_y = max(yb for yb, _ in vertex_ys)

        # Raise the gap floor polygon to match
        for c in corners_3d:
            c[1] = gap_floor_y
        gap["centroid"][1] = gap_floor_y

        # Deferred import so module-level dead-import pruning doesn't strip
        # these helpers (they're only used inside this loop).
        from reconcile.extract3d.gaps import (
            _edge_on_room_boundary,
            _piece_index,
            _ytop_at_xz,
        )

        # Clip the snapped polygon by the room-floor union for this story.
        # Mirrors reconcile.extract3d.gaps.compute_gap_walls -- see that
        # function for the rationale.
        snapped_xz_2d = [(float(s["xz"][0]), float(s["xz"][1])) for s in snapped]
        try:
            snapped_poly = Polygon(snapped_xz_2d)
            if not snapped_poly.is_valid:
                snapped_poly = make_valid(snapped_poly)
        except Exception:
            snapped_poly = None

        room_union = story_room_union.get(story)
        room_boundary = story_room_boundary.get(story)
        pieces_for_caps = [snapped]
        if (
            room_union is not None
            and not room_union.is_empty
            and snapped_poly is not None
            and not snapped_poly.is_empty
        ):
            try:
                clipped = make_valid(snapped_poly.difference(room_union))
            except Exception:
                clipped = None
            if clipped is None or clipped.is_empty:
                pieces_for_caps = []
            elif abs(snapped_poly.area - clipped.area) > 1e-6:
                all_pieces = sorted(
                    (p for p in _decompose_polys(clipped) if p.area > 0.01),
                    key=lambda p: -p.area,
                )
                pieces_for_caps = []
                for piece in all_pieces:
                    piece_coords = list(piece.exterior.coords)
                    if piece_coords and piece_coords[0] == piece_coords[-1]:
                        piece_coords = piece_coords[:-1]
                    if len(piece_coords) < 3:
                        continue
                    ps = []
                    for x, z in piece_coords:
                        ytop = _ytop_at_xz((x, z), snapped)
                        ps.append(
                            {
                                "xz": np.array([x, z], dtype=float),
                                "ybot": gap_floor_y,
                                "ytop": ytop,
                                "wall": None,
                            }
                        )
                    pieces_for_caps.append(ps)

        # Side walls always trace the original snapped polygon -- see sibling
        # implementation in reconcile.extract3d.gaps.compute_gap_walls.
        n_edges_sides = len(snapped)
        for ei in range(n_edges_sides):
            j = (ei + 1) % n_edges_sides
            c0 = snapped[ei]["xz"]
            c1 = snapped[j]["xz"]
            if _edge_on_room_boundary(c0, c1, room_boundary):
                continue
            ytop0 = float(snapped[ei]["ytop"])
            ytop1 = float(snapped[j]["ytop"])
            gw = {
                "id": stable_gap_wall_id(gap, gap["type"], "edge", ei),
                "corners": [
                    [float(c0[0]), gap_floor_y, float(c0[1])],
                    [float(c1[0]), gap_floor_y, float(c1[1])],
                    [float(c1[0]), ytop1, float(c1[1])],
                    [float(c0[0]), ytop0, float(c0[1])],
                ],
                "type": gap["type"],
                "story": story,
                "confidence": gap["confidence"],
            }
            if "room_index" in gap:
                gw["room_index"] = gap["room_index"]
            walls.append(gw)

        n_pieces = len(pieces_for_caps)
        for piece_idx, piece_snapped in enumerate(pieces_for_caps):
            n_edges_p = len(piece_snapped)
            if n_edges_p < 3:
                continue
            walls.append(
                {
                    "id": stable_gap_wall_id(
                        gap,
                        "gap_floor",
                        "polygon",
                        _piece_index(piece_idx, n_pieces, None),
                    ),
                    "corners": [
                        [float(v["xz"][0]), gap_floor_y, float(v["xz"][1])]
                        for v in piece_snapped
                    ],
                    "type": "gap_floor",
                    "story": story,
                    "confidence": gap["confidence"],
                    **(
                        {"room_index": gap["room_index"]} if "room_index" in gap else {}
                    ),
                }
            )
            xz_2d = [(float(v["xz"][0]), float(v["xz"][1])) for v in piece_snapped]
            ti = 0
            for ia, ib, ic in earclip_2d(xz_2d):
                va, vb, vc = piece_snapped[ia], piece_snapped[ib], piece_snapped[ic]
                x0, z0 = va["xz"]
                x1, z1 = vb["xz"]
                x2, z2 = vc["xz"]
                area_xz = abs((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) * 0.5
                if area_xz < 1e-5:
                    continue
                walls.append(
                    {
                        "id": stable_gap_wall_id(
                            gap,
                            "gap_ceiling",
                            "tri",
                            _piece_index(piece_idx, n_pieces, ti),
                        ),
                        "corners": [
                            [float(va["xz"][0]), float(va["ytop"]), float(va["xz"][1])],
                            [float(vb["xz"][0]), float(vb["ytop"]), float(vb["xz"][1])],
                            [float(vc["xz"][0]), float(vc["ytop"]), float(vc["xz"][1])],
                        ],
                        "type": "gap_ceiling",
                        "story": story,
                        "confidence": gap["confidence"],
                        **(
                            {"room_index": gap["room_index"]}
                            if "room_index" in gap
                            else {}
                        ),
                    }
                )
                ti += 1

    return walls


def _extend_wall_to_slab(corners, slab_y_above, epsilon=0.05, max_gap=0.80):
    """Extend wall top corners upward to reach slab_y_above.

    Identifies top vs bottom corners by Y value (not index), since RoomPlan
    polygon winding order varies depending on wall orientation. The top half
    of corners (by Y) are the ones that get extended to the slab.

    For slanted walls (flat-slant-flat), some top corners may already reach
    the slab while others don't -- only the ones below get raised.

    Returns dict with extension_strip, or None if no extension is needed.
    """
    if len(corners) < 3:
        return None

    ys = [c[1] for c in corners]
    max_y = max(ys)
    min_y = min(ys)
    wall_height = max_y - min_y
    if wall_height < 0.1:
        return None  # degenerate wall

    # Identify top corners: those in the upper half of the wall's Y range
    # Use a threshold at 60% of the height from bottom to catch slanted
    # corners that dip slightly below mid-height
    y_thresh = min_y + wall_height * 0.4
    top_indices = [i for i, y in enumerate(ys) if y > y_thresh]
    if not top_indices:
        return None

    # Find which top corners actually need extending (below slab - epsilon)
    need_ext = [i for i in top_indices if ys[i] < slab_y_above - epsilon]
    if not need_ext:
        return None  # all top corners already reach the slab

    # Half-floor check: if even the highest top corner is >max_gap below slab
    max_top_y = max(ys[i] for i in top_indices)
    if slab_y_above - max_top_y > max_gap:
        return None

    # Sort top indices by polygon order for proper strip winding
    top_indices_sorted = sorted(top_indices)

    # Extended corners: copy, raise only those top corners below the slab
    extended = [list(c) for c in corners]
    need_ext_set = set(need_ext)
    for i in need_ext:
        extended[i][1] = slab_y_above

    # Build extension as a list of quads between consecutive top corner pairs.
    # Each quad is planar, avoiding rendering issues with non-planar polygons
    # on walls with non-even tops (slanted, flat-slant-flat, etc.)
    extension_strips = []
    for k in range(len(top_indices_sorted) - 1):
        i0 = top_indices_sorted[k]
        i1 = top_indices_sorted[k + 1]
        orig_y0 = corners[i0][1]
        orig_y1 = corners[i1][1]
        ext_y0 = slab_y_above if i0 in need_ext_set else orig_y0
        ext_y1 = slab_y_above if i1 in need_ext_set else orig_y1
        # Skip if both corners are already at slab (no area)
        if abs(ext_y0 - orig_y0) < 1e-4 and abs(ext_y1 - orig_y1) < 1e-4:
            continue
        quad = [
            [corners[i0][0], orig_y0, corners[i0][2]],
            [corners[i1][0], orig_y1, corners[i1][2]],
            [corners[i1][0], ext_y1, corners[i1][2]],
            [corners[i0][0], ext_y0, corners[i0][2]],
        ]
        extension_strips.append(quad)

    if not extension_strips:
        return None

    return {
        "extended_corners": extended,
        "extension_strip": extension_strips,
    }


def _infer_ceilings(rooms_out):
    """Infer ceiling polygons.

    Two stages:

    1. Wall-top consensus classifier (`classify_should_be_flat`) marks rooms
       whose walls cluster at one height as flat, drops `raw_ceiling_planes`
       inconsistent with that consensus (the noMesh ingest stores wall
       polygons as ceiling planes -- those reach floor-level y), and stamps
       a horizontal ceiling polygon. Independent of the noisy raw planes,
       so noise can't contaminate the classification.
    2. Slant-detection fallback for rooms not flagged flat -- chains wall top
       edges in floor-polygon order to build a sloped ceiling contour.
    """
    SLANT_THRESH = 0.15
    SLOPE_THRESH = 0.15
    XZ_MATCH_TOL = 0.20

    if not rooms_out:
        return

    classifications = classify_should_be_flat(rooms_out)
    drop_noisy_raw_ceiling_planes(rooms_out, classifications)
    apply_flat_classification(rooms_out, classifications)

    for ri, room in enumerate(rooms_out):
        if classifications.get(ri, (False, None))[0]:
            continue
        fp = room.get("floor_polygon", [])
        walls = room.get("walls_computed") or room.get("walls_merged", [])
        if len(fp) < 3 or not walls:
            continue

        has_slant = False
        for w in walls:
            corners = w["corners"]
            if len(corners) < 3:
                continue
            ys = [c[1] for c in corners]
            mid_y = (max(ys) + min(ys)) / 2.0
            top_cs = [c for c in corners if c[1] > mid_y - 0.01]
            if len(top_cs) >= 2:
                top_range = max(c[1] for c in top_cs) - min(c[1] for c in top_cs)
                if top_range > SLANT_THRESH:
                    has_slant = True
                    break

        if not has_slant:
            continue

        ceiling_polygon = _build_ceiling_from_wall_tops(fp, walls, XZ_MATCH_TOL)

        if ceiling_polygon is None or len(ceiling_polygon) < 3:
            continue

        ceiling_ys = [p[1] for p in ceiling_polygon]
        y_range_ceil = max(ceiling_ys) - min(ceiling_ys)
        ceiling_type = "flat" if y_range_ceil < SLOPE_THRESH else "sloped"

        room["ceiling_polygon"] = ceiling_polygon
        room["ceiling_type"] = ceiling_type
        room["ceiling_ridge_height"] = round(max(ceiling_ys), 4)
        room["ceiling_eave_height"] = round(min(ceiling_ys), 4)


def _build_ceiling_from_wall_tops(fp, walls, xz_tol):
    """Chain wall top edges in floor-polygon order to build ceiling polygon.

    For each floor polygon edge, find the wall whose bottom corners match,
    then emit that wall's top corners in the correct direction.
    """
    n = len(fp)

    # Pre-compute wall info: bottom corners, top corners
    wall_info = []
    for w in walls:
        corners = w["corners"]
        if len(corners) < 3:
            continue
        ys = [c[1] for c in corners]
        mid_y = (max(ys) + min(ys)) / 2.0
        bot_corners = [c for c in corners if c[1] <= mid_y + 0.01]
        top_corners = [c for c in corners if c[1] > mid_y - 0.01]
        if len(bot_corners) < 2 or len(top_corners) < 2:
            continue
        wall_info.append({"bot": bot_corners, "top": top_corners})

    if not wall_info:
        return None

    def xz_dist(a, b):
        return math.hypot(a[0] - b[0], a[2] - b[2])

    def find_nearest_bot(corner, bot_list):
        best_d, best_c = float("inf"), None
        for bc in bot_list:
            d = xz_dist(corner, bc)
            if d < best_d:
                best_d, best_c = d, bc
        return best_d, best_c

    # For each floor polygon edge, find the best matching wall
    edge_walls = [None] * n
    used = set()
    for ei in range(n):
        v0 = fp[ei]
        v1 = fp[(ei + 1) % n]
        best_score = float("inf")
        best_wi = None
        for wi, wi_info in enumerate(wall_info):
            if wi in used:
                continue
            d0, _ = find_nearest_bot(v0, wi_info["bot"])
            d1, _ = find_nearest_bot(v1, wi_info["bot"])
            score = d0 + d1
            if score < best_score:
                best_score = score
                best_wi = wi
        if best_wi is not None and best_score < xz_tol * 2:
            edge_walls[ei] = best_wi
            used.add(best_wi)

    # Chain top corners in floor polygon order
    ceiling_pts = []
    for ei in range(n):
        wi = edge_walls[ei]
        if wi is None:
            continue

        v0 = fp[ei]
        v1 = fp[(ei + 1) % n]
        top_corners = wall_info[wi]["top"]
        bot_corners = wall_info[wi]["bot"]

        # Determine which bot corner is closer to v0 vs v1
        d0_list = [(xz_dist(v0, bc), bc) for bc in bot_corners]
        d0_list.sort()
        start_bot = d0_list[0][1]

        d1_list = [(xz_dist(v1, bc), bc) for bc in bot_corners]
        d1_list.sort()
        end_bot = d1_list[0][1]

        # For each bot corner, find the matching top corner (same XZ, higher Y)
        def find_top_at_xz(bot_c, top_list):
            best_d, best_t = float("inf"), None
            for tc in top_list:
                d = xz_dist(bot_c, tc)
                if d < best_d:
                    best_d, best_t = d, tc
            return best_t

        start_top = find_top_at_xz(start_bot, top_corners)
        end_top = find_top_at_xz(end_bot, top_corners)

        if start_top is None or end_top is None:
            continue

        # Collect top corners for this wall in order from start to end.
        if len(top_corners) <= 2:
            ordered_top = [start_top, end_top]
        else:
            # Find interior top corners (not at start/end bot positions)
            interior = []
            for tc in top_corners:
                d_start = xz_dist(tc, start_bot)
                d_end = xz_dist(tc, end_bot)
                if d_start > xz_tol and d_end > xz_tol:
                    interior.append(tc)

            # Order interior points by projection along edge direction
            edge_dir = np.array([v1[0] - v0[0], v1[2] - v0[2]])
            edge_len = np.linalg.norm(edge_dir)
            if edge_len > 1e-6:
                edge_unit = edge_dir / edge_len
                interior.sort(
                    key=lambda c: np.dot(
                        np.array([c[0] - v0[0], c[2] - v0[2]]), edge_unit
                    )
                )

            ordered_top = [start_top, *interior, end_top]

        # Emit ceiling points, deduplicating consecutive identical XZ
        for tc in ordered_top:
            pt = [round(tc[0], 4), round(tc[1], 4), round(tc[2], 4)]
            if ceiling_pts:
                last = ceiling_pts[-1]
                if abs(pt[0] - last[0]) < 0.01 and abs(pt[2] - last[2]) < 0.01:
                    # Same XZ -- keep the higher Y
                    if pt[1] > last[1]:
                        ceiling_pts[-1] = pt
                    continue
            ceiling_pts.append(pt)

    # Deduplicate first/last if they match in XZ
    if len(ceiling_pts) >= 2:
        first, last = ceiling_pts[0], ceiling_pts[-1]
        if abs(first[0] - last[0]) < 0.01 and abs(first[2] - last[2]) < 0.01:
            if last[1] > first[1]:
                ceiling_pts[0] = last
            ceiling_pts.pop()

    return ceiling_pts if len(ceiling_pts) >= 3 else None


def _flat_ceiling_fallback(rooms, slab_y):
    """Emit flat ceiling at the median wall-top height."""
    wall_tops = []
    for r in rooms:
        for w in r.get("walls_computed") or r.get("walls_merged", []):
            ys = [c[1] for c in w["corners"]]
            if ys:
                wall_tops.append(max(ys))
    if not wall_tops:
        return
    ceil_y = round(float(np.median(wall_tops)), 4)
    for room in rooms:
        fp = room.get("floor_polygon", [])
        if len(fp) < 3:
            continue
        room["ceiling_polygon"] = [[round(v[0], 4), ceil_y, round(v[2], 4)] for v in fp]
        room["ceiling_type"] = "flat"
        room["ceiling_ridge_height"] = ceil_y
        room["ceiling_eave_height"] = ceil_y


def _wall_xz_length(corners):
    """Compute the XZ-plane length of a wall/element from its corners."""
    if len(corners) < 2:
        return 0.0
    c = np.array(corners)
    # Use first two corners (bottom edge)
    dx = c[1][0] - c[0][0]
    dz = c[1][2] - c[0][2]
    return math.hypot(dx, dz)


def _canonicalize_wall_quad(corners):
    """Normalize a wall with arbitrary corner count/order to a 4-corner
    quad ``[bl, br, tr, tl]`` whose bottom edge is the wall's length axis.

    Canonical 4-corner walls round-trip unchanged; walls with 5+ corners
    or non-canonical ordering are reduced to a quad suitable for the
    gap-closure construction that assumes ``corners[0..1]`` is the
    bottom edge.
    """
    if len(corners) < 4:
        return [list(c) for c in corners]
    arr = [list(c) for c in corners]
    ys = [c[1] for c in arr]
    if max(ys) - min(ys) < 1e-3:
        return arr
    mid_y = (max(ys) + min(ys)) / 2.0
    bot_cs = [c for c in arr if c[1] < mid_y + 0.01]
    top_cs = [c for c in arr if c[1] > mid_y - 0.01]
    if len(bot_cs) < 2 or len(top_cs) < 2:
        return arr

    bl, br = bot_cs[0], bot_cs[1]
    best_d2 = -1.0
    for i in range(len(bot_cs)):
        for j in range(i + 1, len(bot_cs)):
            dx = bot_cs[i][0] - bot_cs[j][0]
            dz = bot_cs[i][2] - bot_cs[j][2]
            d2 = dx * dx + dz * dz
            if d2 > best_d2:
                best_d2 = d2
                bl, br = bot_cs[i], bot_cs[j]

    ref_dx = arr[1][0] - arr[0][0]
    ref_dz = arr[1][2] - arr[0][2]
    if (br[0] - bl[0]) * ref_dx + (br[2] - bl[2]) * ref_dz < 0:
        bl, br = br, bl

    def _nearest(x, z, candidates):
        best = candidates[0]
        best_d2 = float("inf")
        for c in candidates:
            dx = c[0] - x
            dz = c[2] - z
            d2 = dx * dx + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best = c
        return best

    tl = _nearest(bl[0], bl[2], top_cs)
    tr = _nearest(br[0], br[2], top_cs)
    return [list(bl), list(br), list(tr), list(tl)]


def _wall_top_bottom_profiles(corners, axis2d, origin_xz):
    """Project a wall polygon to (t, y) and split into top/bottom polylines.

    `t` = signed projection of corner XZ onto ``axis2d`` (relative to
    ``origin_xz``). The polygon's vertex order is partitioned into two
    monotonic-in-t runs; vertical (zero-dt) edges are absorbed into the
    surrounding run. The run with higher mean y becomes the top profile.

    Each profile is returned as a list of (t, y) sorted by t and
    deduplicated per t -- top keeps the maximum y at each t, bottom keeps
    the minimum. Returns ([], []) for degenerate inputs.

    This preserves kinks in the upper outline (e.g. an attic wall whose
    top follows a sloped roof eave) that a Y-mid split would discard.
    """
    if len(corners) < 3:
        return [], []
    pts = []
    for c in corners:
        dx = float(c[0]) - float(origin_xz[0])
        dz = float(c[2]) - float(origin_xz[1])
        t = dx * axis2d[0] + dz * axis2d[1]
        pts.append((t, float(c[1])))
    n = len(pts)

    eps = 1e-9
    t_min = min(p[0] for p in pts)
    t_max = max(p[0] for p in pts)
    if t_max - t_min < 1e-6:
        return [], []

    runs = []  # list of (sign, list of (t, y))
    current = [pts[0]]
    current_sign = 0
    for i in range(1, n + 1):
        a = pts[(i - 1) % n]
        b = pts[i % n]
        dt = b[0] - a[0]
        if abs(dt) < eps:
            current.append(b)
            continue
        sign = 1 if dt > 0 else -1
        if current_sign == 0 or current_sign == sign:
            current.append(b)
            current_sign = sign
        else:
            runs.append((current_sign, current))
            current = [a, b]
            current_sign = sign
    if current:
        runs.append((current_sign, current))
    if len(runs) >= 2 and runs[0][0] == runs[-1][0]:
        merged = runs[-1][1] + runs[0][1]
        runs = [(runs[0][0], merged), *runs[1:-1]]

    fwd = next((r for s, r in runs if s == 1), None)
    bwd = next((r for s, r in runs if s == -1), None)
    if not fwd or not bwd:
        return [], []

    def _consolidate(profile, take_max):
        s = sorted(profile, key=lambda p: p[0])
        out = []
        for t, y in s:
            if out and abs(out[-1][0] - t) < 1e-6:
                out[-1] = (
                    out[-1][0],
                    max(out[-1][1], y) if take_max else min(out[-1][1], y),
                )
            else:
                out.append((t, y))
        return out

    fwd_my = sum(p[1] for p in fwd) / len(fwd)
    bwd_my = sum(p[1] for p in bwd) / len(bwd)
    if fwd_my >= bwd_my:
        top_raw, bot_raw = fwd, bwd
    else:
        top_raw, bot_raw = bwd, fwd
    return _consolidate(top_raw, take_max=True), _consolidate(bot_raw, take_max=False)


def _interp_profile_y(profile, t):
    """Piecewise-linear interpolation of y at parameter t along profile.

    Profile is a list of (t, y) sorted by t. Out-of-range t clamps to ends.
    """
    if not profile:
        return 0.0
    if t <= profile[0][0]:
        return profile[0][1]
    if t >= profile[-1][0]:
        return profile[-1][1]
    for k in range(len(profile) - 1):
        t0, y0 = profile[k]
        t1, y1 = profile[k + 1]
        if t0 <= t <= t1:
            span = t1 - t0
            if span < 1e-9:
                return y0
            return y0 + (t - t0) / span * (y1 - y0)
    return profile[-1][1]


def _detect_exterior_gap_indicators(rooms_out):
    """Detect gaps in front of doors/openings/storages with a parallel wall on the
    other side.

    For each door/opening/storage wider than MIN_WIDTH:
    - Compute XZ normal from corner cross product
    - Search both directions for a parallel wall (angle < MAX_ANGLE) within
    MIN_DIST-MAX_DIST
    - Wall must be at least as wide as the element (0.9x tolerance)
    - For storages: only match walls from a *different* room (same-room wall is just the
    backing wall)
    - Keep the closest matching wall, assign confidence
    """
    MIN_WIDTH = 0.50
    MIN_DIST = 0.20
    MAX_DIST = 1.0
    MAX_ANGLE = 15.0
    WIDTH_TOLERANCE = 0.9  # wall must be >= 90% of element width

    # Collect all walls per story, tagged with room index
    story_walls = defaultdict(list)  # story -> [(wall, room_idx)]
    for ri, room in enumerate(rooms_out):
        for w in room["walls_computed"]:
            story_walls[room["story"]].append((w, ri))

    indicators = []

    def _find_parallel_wall(corners, elem_width, story, room_idx, require_other_room):
        """Search for a parallel wall across a gap from an element."""
        c = np.array(corners)
        e1 = c[1] - c[0]
        e2 = c[2] - c[1]
        normal = np.cross(e1, e2)
        normal_xz = np.array([normal[0], normal[2]])
        nlen = np.linalg.norm(normal_xz)
        if nlen < 1e-6:
            return None
        normal_xz = normal_xz / nlen

        elem_cx = float(np.mean([co[0] for co in corners]))
        elem_cz = float(np.mean([co[2] for co in corners]))

        best_match = None
        best_dist = float("inf")

        for sign in [1.0, -1.0]:
            direction = normal_xz * sign

            for w, w_room_idx in story_walls.get(story, []):
                if require_other_room and w_room_idx == room_idx:
                    continue

                wc = w["corners"]
                if len(wc) < 3:
                    continue

                wall_width = _wall_xz_length(wc)
                if wall_width < elem_width * WIDTH_TOLERANCE:
                    continue

                wcx = float(np.mean([co[0] for co in wc]))
                wcz = float(np.mean([co[2] for co in wc]))

                to_wall = np.array([wcx - elem_cx, wcz - elem_cz])
                dist_along = float(np.dot(to_wall, direction))

                if dist_along < MIN_DIST or dist_along > MAX_DIST:
                    continue

                perp = float(
                    np.abs(np.dot(to_wall, np.array([-direction[1], direction[0]])))
                )
                if perp > max(elem_width, wall_width) * 0.5:
                    continue

                wc_arr = np.array(wc)
                we1 = wc_arr[1] - wc_arr[0]
                we2 = wc_arr[2] - wc_arr[1]
                wnormal = np.cross(we1, we2)
                wnormal_xz = np.array([wnormal[0], wnormal[2]])
                wnlen = np.linalg.norm(wnormal_xz)
                if wnlen < 1e-6:
                    continue
                wnormal_xz = wnormal_xz / wnlen

                cos_angle = abs(float(np.dot(normal_xz, wnormal_xz)))
                cos_angle = min(cos_angle, 1.0)
                angle_deg = math.degrees(math.acos(cos_angle))

                if angle_deg > MAX_ANGLE:
                    continue

                if dist_along < best_dist:
                    best_dist = dist_along
                    best_match = (w, wc, dist_along, angle_deg)

        return best_match

    for ri, room in enumerate(rooms_out):
        story = room["story"]

        # Doors and openings: match any wall on the same story
        for elem_type, elems in [
            ("door", room.get("doors", [])),
            ("opening", room.get("openings", [])),
        ]:
            for elem in elems:
                corners = elem.get("corners", [])
                if len(corners) < 3:
                    continue
                elem_width = _wall_xz_length(corners)
                if elem_width < MIN_WIDTH:
                    continue
                match = _find_parallel_wall(
                    corners, elem_width, story, ri, require_other_room=False
                )
                if match:
                    w, wc, dist_along, angle_deg = match
                    indicators.append(
                        {
                            "story": story,
                            "element_type": elem_type,
                            "element_id": elem["id"],
                            "element_corners": corners,
                            "element_width_m": round(elem_width, 2),
                            "wall_id": w["id"],
                            "wall_corners": wc,
                            "wall_distance_m": round(dist_along, 2),
                            "angle_deg": round(angle_deg, 1),
                            "confidence": (
                                "high"
                                if angle_deg < 5 and dist_along < 0.3
                                else "medium"
                                if angle_deg < 10 and dist_along < 0.6
                                else "low"
                            ),
                        }
                    )

        # Storages: only if flush against a same-room wall (<5cm), and matched wall is
        # from another room
        STORAGE_WALL_PROXIMITY = 0.05
        room_wall_ids = {w["id"] for w in room["walls_computed"]}
        for s in room.get("storages", []):
            corners = s.get("corners", [])
            if len(corners) < 3:
                continue
            elem_width = _wall_xz_length(corners)
            if elem_width < MIN_WIDTH:
                continue
            # Check if storage is flush against any same-room wall
            sc = np.array(corners)
            s_cx = float(sc[:, 0].mean())
            s_cz = float(sc[:, 2].mean())
            flush = False
            for w, _w_ri in story_walls.get(story, []):
                if w["id"] not in room_wall_ids:
                    continue
                wc = w["corners"]
                if len(wc) < 2:
                    continue
                # Distance from storage center to wall line (XZ plane)
                w0 = np.array([wc[0][0], wc[0][2]])
                w1 = np.array([wc[1][0], wc[1][2]])
                sp = np.array([s_cx, s_cz])
                edge = w1 - w0
                elen = np.linalg.norm(edge)
                if elen < 1e-6:
                    continue
                t = float(np.dot(sp - w0, edge)) / (elen * elen)
                t = max(0.0, min(1.0, t))
                closest = w0 + t * edge
                dist = float(np.linalg.norm(sp - closest))
                if (
                    dist
                    < STORAGE_WALL_PROXIMITY
                    + elem_width * 0.5
                    + _wall_xz_length(wc) * 0.0
                ):
                    # More precise: perpendicular distance from storage center to wall
                    # line
                    perp_dist = (
                        float(abs(edge[0] * (sp - w0)[1] - edge[1] * (sp - w0)[0]))
                        / elen
                    )
                    if perp_dist < STORAGE_WALL_PROXIMITY:
                        flush = True
                        break
            if not flush:
                continue
            match = _find_parallel_wall(
                corners, elem_width, story, ri, require_other_room=True
            )
            if match:
                w, wc, dist_along, angle_deg = match
                indicators.append(
                    {
                        "story": story,
                        "element_type": "storage",
                        "element_id": s["id"],
                        "element_corners": corners,
                        "element_width_m": round(elem_width, 2),
                        "wall_id": w["id"],
                        "wall_corners": wc,
                        "wall_distance_m": round(dist_along, 2),
                        "angle_deg": round(angle_deg, 1),
                        "confidence": (
                            "high"
                            if angle_deg < 5 and dist_along < 0.3
                            else "medium"
                            if angle_deg < 10 and dist_along < 0.6
                            else "low"
                        ),
                    }
                )

    return indicators


def _compute_gap_closures(indicators, rooms_out):
    """Create wall/floor/ceiling quads that close the gaps between parallel walls.

    For each indicator, finds the parent wall of the element, computes the
    overlap region between the parent wall and the parallel wall in XZ,
    then creates side-wall, floor, and ceiling quads closing the gap.
    """
    closures = []

    # Build wall lookup: wall_id -> corners (for finding parent walls)
    # Also build element->parent mapping from all rooms
    all_walls_by_id = {}
    element_parent_wall = {}  # element_id -> wall_id
    for room in rooms_out:
        for w in room["walls_computed"]:
            all_walls_by_id[w["id"]] = w["corners"]
        parent_lookup = room.get("_parent_lookup", {})
        for elem_list in [
            room.get("doors", []),
            room.get("openings", []),
            room.get("storages", []),
        ]:
            for elem in elem_list:
                pid = parent_lookup.get(elem["id"])
                if pid:
                    element_parent_wall[elem["id"]] = pid

    for ind in indicators:
        wc_raw = ind["wall_corners"]
        if len(wc_raw) < 4:
            continue
        wc = np.array(_canonicalize_wall_quad(wc_raw))

        # Find the parent wall of the element; fall back to element corners
        parent_id = element_parent_wall.get(ind["element_id"])
        if parent_id and parent_id in all_walls_by_id:
            pc_raw = all_walls_by_id[parent_id]
        else:
            pc_raw = ind["element_corners"]
        if len(pc_raw) < 4:
            continue
        pc = np.array(_canonicalize_wall_quad(pc_raw))

        # Wall direction: along the bottom edge of the parallel wall (XZ plane)
        wall_dir = wc[1] - wc[0]
        wall_dir_xz = np.array([wall_dir[0], 0.0, wall_dir[2]])
        wall_len = np.linalg.norm(wall_dir_xz)
        if wall_len < 1e-6:
            continue
        axis = wall_dir_xz / wall_len

        # Project all corners onto the shared axis (XZ)
        origin = np.array([wc[0][0], 0.0, wc[0][2]])
        axis2d = np.array([axis[0], axis[2]])

        def proj_xz(pts, origin=origin, axis2d=axis2d):
            return [
                float(
                    np.dot(
                        np.array([p[0], p[2]]) - np.array([origin[0], origin[2]]),
                        axis2d,
                    )
                )
                for p in pts
            ]

        p_proj = proj_xz(pc)  # parent wall projection
        w_proj = proj_xz(wc)  # parallel wall projection

        p_min, p_max = min(p_proj), max(p_proj)
        w_min, w_max = min(w_proj), max(w_proj)

        # Overlap extent along the axis (full parallel region)
        ov_min = max(p_min, w_min)
        ov_max = min(p_max, w_max)
        if ov_max - ov_min < 0.05:
            continue

        # Wall XZ edges (for XZ projection of overlap endpoints onto each
        # wall's bottom edge). Y comes from the polyline profiles below so
        # that kinked walls (e.g. attic walls bounded by a sloped roof eave)
        # are followed faithfully instead of collapsed to a flat top.
        p0_xz = np.array([pc[0][0], pc[0][2]])
        p1_xz = np.array([pc[1][0], pc[1][2]])
        p_edge = p1_xz - p0_xz
        p_elen = np.linalg.norm(p_edge)

        w0_xz = np.array([wc[0][0], wc[0][2]])
        w1_xz = np.array([wc[1][0], wc[1][2]])
        w_edge = w1_xz - w0_xz
        w_elen = np.linalg.norm(w_edge)

        def _edge_t(xz_pt, p0, edge, elen):
            """Compute clamped parameter t along an edge for a given XZ point."""
            if elen > 1e-6:
                return float(np.clip(np.dot(xz_pt - p0, edge) / (elen**2), 0, 1))
            return 0.0

        origin_xz = (float(origin[0]), float(origin[2]))
        p_top_profile, p_bot_profile = _wall_top_bottom_profiles(
            pc_raw, axis2d, origin_xz
        )
        w_top_profile, w_bot_profile = _wall_top_bottom_profiles(
            wc_raw, axis2d, origin_xz
        )
        if not (p_top_profile and p_bot_profile and w_top_profile and w_bot_profile):
            continue  # Degenerate wall (zero-length axis or non-monotonic-in-t)

        # Compute XZ positions and interpolated Y at left and right overlap edges
        side_pts = []  # [(p_xz, p_ybot, p_ytop, w_xz, w_ybot, w_ytop)]
        for t_val in [ov_min, ov_max]:
            xz_pt = np.array([origin[0], origin[2]]) + axis2d * t_val

            p_t = _edge_t(xz_pt, p0_xz, p_edge, p_elen)
            p_xz = p0_xz + p_t * p_edge if p_elen > 1e-6 else p0_xz
            p_ybot = _interp_profile_y(p_bot_profile, t_val)
            p_ytop = _interp_profile_y(p_top_profile, t_val)

            w_t = _edge_t(xz_pt, w0_xz, w_edge, w_elen)
            w_xz = w0_xz + w_t * w_edge if w_elen > 1e-6 else w0_xz
            w_ybot = _interp_profile_y(w_bot_profile, t_val)
            w_ytop = _interp_profile_y(w_top_profile, t_val)

            side_pts.append((p_xz, p_ybot, p_ytop, w_xz, w_ybot, w_ytop))

            # Side wall quad: follows the slope of both walls
            closures.append(
                {
                    "corners": [
                        [float(p_xz[0]), p_ybot, float(p_xz[1])],
                        [float(w_xz[0]), w_ybot, float(w_xz[1])],
                        [float(w_xz[0]), w_ytop, float(w_xz[1])],
                        [float(p_xz[0]), p_ytop, float(p_xz[1])],
                    ],
                    "type": "side",
                    "indicator_element_id": ind["element_id"],
                    "indicator_wall_id": ind["wall_id"],
                    "story": ind["story"],
                }
            )

        # Floor quad: follows the bottom edge slope of both walls
        L, R = side_pts
        p_xz_l, p_ybot_l, p_ytop_l, w_xz_l, w_ybot_l, w_ytop_l = L
        p_xz_r, p_ybot_r, p_ytop_r, w_xz_r, w_ybot_r, w_ytop_r = R
        closures.append(
            {
                "corners": [
                    [float(p_xz_l[0]), p_ybot_l, float(p_xz_l[1])],
                    [float(p_xz_r[0]), p_ybot_r, float(p_xz_r[1])],
                    [float(w_xz_r[0]), w_ybot_r, float(w_xz_r[1])],
                    [float(w_xz_l[0]), w_ybot_l, float(w_xz_l[1])],
                ],
                "type": "floor",
                "indicator_element_id": ind["element_id"],
                "indicator_wall_id": ind["wall_id"],
                "story": ind["story"],
            }
        )

        # Ceiling quad: follows the top edge slope of both walls
        closures.append(
            {
                "corners": [
                    [float(p_xz_l[0]), p_ytop_l, float(p_xz_l[1])],
                    [float(p_xz_r[0]), p_ytop_r, float(p_xz_r[1])],
                    [float(w_xz_r[0]), w_ytop_r, float(w_xz_r[1])],
                    [float(w_xz_l[0]), w_ytop_l, float(w_xz_l[1])],
                ],
                "type": "ceiling",
                "indicator_element_id": ind["element_id"],
                "indicator_wall_id": ind["wall_id"],
                "story": ind["story"],
            }
        )

    return closures


def _stitch_wall_gaps(rooms_out):
    """Delegate to the shared modular stitch implementation."""
    return modular_stitch_wall_gaps(rooms_out)


def extract_building(uuid, pipeline_dir, scan_cache_root):
    """Extract 3D data for one building."""
    # Find merged.json
    merged_path = None
    for entry in os.listdir(pipeline_dir):
        if entry.startswith(uuid) and os.path.isdir(pipeline_dir / entry):
            mp = pipeline_dir / entry / "merged.json"
            if mp.exists():
                merged_path = mp
                break

    if not merged_path:
        return None

    with open(merged_path) as f:
        merged = json.load(f)

    # Load reconciled.json for classification if available
    recon_path = merged_path.parent / "reconciled.json"
    classification = "UNKNOWN"
    stories_changed = 0
    if recon_path.exists():
        with open(recon_path) as f:
            recon = json.load(f)
        meta = recon.get("reconciliation", {})
        classification = meta.get("classification", "UNKNOWN")

    # Story fix: cluster floor Y positions
    floor_ys = []
    for mr in merged["rooms"]:
        if mr.get("floors") and mr["floors"][0].get("polygonCorners"):
            fc = corners_to_world(
                mr["floors"][0]["polygonCorners"],
                mr["floors"][0]["transform"],
            )
            mean_y = np.mean([c[1] for c in fc])
            floor_ys.append(mean_y)
        else:
            floor_ys.append(0.0)

    # Cluster stories by Y gap > 1.0m
    sorted_ys = sorted(set(floor_ys))
    story_map = {}
    current_story = 0
    for i, y in enumerate(sorted_ys):
        if i > 0 and abs(y - sorted_ys[i - 1]) > 1.0:
            current_story += 1
        story_map[y] = current_story

    stories_found = current_story + 1

    # Assign stories to rooms
    room_stories = []
    for fy in floor_ys:
        closest_y = min(sorted_ys, key=lambda sy: abs(sy - fy))
        room_stories.append(story_map[closest_y])

    # Try loading raw scan-cache rooms
    scan_dir = find_scan_cache_dir(uuid, scan_cache_root) if scan_cache_root else None
    raw_rooms = load_raw_rooms(scan_dir) if scan_dir else []
    raw_transforms = compute_room_transforms(raw_rooms, merged) if raw_rooms else {}

    # Raw per-room ceilings -- multi-plane (flat + sloped pieces for vaulted
    # rooms). Remapped via the same per-room SVD used for walls so they land
    # in merged-building space.
    raw_ceilings = load_raw_ceilings(scan_dir) if scan_dir else {}
    raw_to_merged = build_raw_to_merged_index(raw_rooms, merged) if raw_rooms else {}
    ceilings_by_merged_idx = {}
    for rname, merged_idx in raw_to_merged.items():
        ceiling = raw_ceilings.get(rname)
        if ceiling is None or not ceiling.get("planes"):
            continue
        svd = raw_transforms.get(rname)
        if svd is None:
            continue
        R_room, t_room, _res, _method = svd
        remapped = []
        for plane in ceiling["planes"]:
            world_corners = corners_to_world(plane["corners_local"], plane["transform"])
            remapped_corners = [
                [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)]
                for c in (R_room @ np.array(cc) + t_room for cc in world_corners)
            ]
            if len(remapped_corners) >= 3:
                remapped.append({"corners": remapped_corners})
        if not remapped:
            continue
        existing = ceilings_by_merged_idx.get(merged_idx)
        if existing is None:
            ceilings_by_merged_idx[merged_idx] = {
                "planes": remapped,
                "source": ceiling.get("source"),
            }
        else:
            existing["planes"].extend(remapped)

    # Build wall UUID -> raw room + transform mapping
    raw_wall_data = {}  # wall_uuid -> (wall_data, R, t, method)
    raw_door_data = {}  # door_uuid -> (door_data, R, t, method)
    raw_window_data = {}  # window_uuid -> (window_data, R, t, method)
    raw_opening_data = {}  # opening_uuid -> (opening_data, R, t, method)
    raw_storage_data = {}  # storage_uuid -> (object_data, R, t, method)
    # Prefer floor-svd transforms over wall-center-svd when the same element
    # appears in multiple raw rooms (floor-svd is higher quality)
    _METHOD_RANK = {"floor-svd": 2, "hybrid": 1, "wall-center-svd": 0}
    for rname, rdata in raw_rooms:
        if rname not in raw_transforms:
            continue
        R, t, _res, method = raw_transforms[rname]
        rank = _METHOD_RANK.get(method, 0)
        for w in rdata.get("walls", []):
            wid = w["identifier"]
            if wid not in raw_wall_data or rank > _METHOD_RANK.get(
                raw_wall_data[wid][3], 0
            ):
                raw_wall_data[wid] = (w, R, t, method)
        for d in rdata.get("doors", []):
            did = d["identifier"]
            if did not in raw_door_data or rank > _METHOD_RANK.get(
                raw_door_data[did][3], 0
            ):
                raw_door_data[did] = (d, R, t, method)
        for w in rdata.get("windows", []):
            wid = w["identifier"]
            if wid not in raw_window_data or rank > _METHOD_RANK.get(
                raw_window_data[wid][3], 0
            ):
                raw_window_data[wid] = (w, R, t, method)
        for o in rdata.get("openings", []):
            oid = o["identifier"]
            if oid not in raw_opening_data or rank > _METHOD_RANK.get(
                raw_opening_data[oid][3], 0
            ):
                raw_opening_data[oid] = (o, R, t, method)
        for obj in rdata.get("objects", []):
            cat = obj.get("category", {})
            if isinstance(cat, dict) and "storage" in cat:
                sid = obj["identifier"]
                if sid not in raw_storage_data or rank > _METHOD_RANK.get(
                    raw_storage_data[sid][3], 0
                ):
                    raw_storage_data[sid] = (obj, R, t, method)

    # Track globally which deduped items have been added (avoid duplicates across rooms)
    global_dedup_added = set()
    global_dedup_doors_added = set()
    global_dedup_windows_added = set()
    global_dedup_openings_added = set()
    global_dedup_storages_added = set()

    # Build merged door/window/opening UUID sets for dedup detection
    merged_door_ids = set()
    merged_window_ids = set()
    merged_opening_ids = set()
    merged_storage_ids = set()
    for omr in merged["rooms"]:
        for d in omr.get("doors", []):
            merged_door_ids.add(d["identifier"])
        for w in omr.get("windows", []):
            merged_window_ids.add(w["identifier"])
        for o in omr.get("openings", []):
            merged_opening_ids.add(o["identifier"])
        for obj in omr.get("objects", []):
            cat = obj.get("category", {})
            if isinstance(cat, dict) and "storage" in cat:
                merged_storage_ids.add(obj["identifier"])

    # Extract rooms
    rooms_out = []
    for ri, mr in enumerate(merged["rooms"]):
        story = room_stories[ri] if ri < len(room_stories) else 0

        # Floor polygon (from merged room, already in building space)
        floor_polygon = []
        if mr.get("floors") and mr["floors"][0].get("polygonCorners"):
            floor_polygon = corners_to_world(
                mr["floors"][0]["polygonCorners"],
                mr["floors"][0]["transform"],
            )

        # Merged walls (top-level deduplicated) -- find walls belonging to this room
        # Actually, room-level walls are already per-room. Use top-level for merged
        # view.
        walls_merged = []
        for w in mr.get("walls", []):
            corners = wall_world_corners(w)
            walls_merged.append({"corners": corners, "id": w["identifier"]})

        # Build merged wall UUID map for hybrid lookups
        merged_wall_by_id = {}
        for omr in merged["rooms"]:
            for mw in omr.get("walls", []):
                merged_wall_by_id[mw["identifier"]] = mw

        # Computed walls: best-available geometry from scan-cache, hybrid, or merged
        # fallback
        walls_computed = []
        for w in mr.get("walls", []):
            wid = w["identifier"]
            if wid in raw_wall_data:
                raw_w, R, t_vec, method = raw_wall_data[wid]
                if method == "floor-svd":
                    # High-quality transform -- use full SVD on raw wall
                    raw_corners = wall_world_corners(raw_w)
                    transformed = [
                        (R @ np.array(c) + t_vec).tolist() for c in raw_corners
                    ]
                    walls_computed.append(
                        {
                            "corners": transformed,
                            "id": wid,
                            "source": "scan-cache",
                        }
                    )
                else:
                    # Wall-center SVD (noisy) -- hybrid: merged position + raw
                    # dimensions
                    fy = (
                        np.mean([c[1] for c in floor_polygon])
                        if floor_polygon
                        else None
                    )
                    corners = hybrid_wall_corners(w, raw_w, floor_y=fy)
                    walls_computed.append(
                        {
                            "corners": corners,
                            "id": wid,
                            "source": "hybrid",
                        }
                    )
            else:
                # No raw data -- use merged room wall as-is
                corners = wall_world_corners(w)
                walls_computed.append(
                    {
                        "corners": corners,
                        "id": wid,
                        "source": "merged-room",
                    }
                )

        # Also add walls that are in raw rooms but NOT in any merged room (dropped by
        # dedup)
        # These are the "other side" of shared walls -- add each deduped wall only once
        added_uuids = {w["id"] for w in walls_computed}
        for rname, rdata in raw_rooms:
            if rname not in raw_transforms:
                continue
            R, t_vec, _res, method = raw_transforms[rname]
            # Only add deduped walls from high-quality (floor-svd) transforms
            if method != "floor-svd":
                continue
            # Check if this raw room maps to this merged room
            raw_uuids = {w["identifier"] for w in rdata.get("walls", [])}
            mr_uuids = {w["identifier"] for w in mr.get("walls", [])}
            if not (raw_uuids & mr_uuids):
                continue
            for w in rdata.get("walls", []):
                wid = w["identifier"]
                if wid in added_uuids or wid in global_dedup_added:
                    continue
                # Check if this wall is NOT in any merged room (it was deduped)
                in_any_merged = any(
                    any(mw["identifier"] == wid for mw in omr.get("walls", []))
                    for omr in merged["rooms"]
                )
                if in_any_merged:
                    continue  # This wall belongs to another merged room

                raw_corners = wall_world_corners(w)
                transformed = [(R @ np.array(c) + t_vec).tolist() for c in raw_corners]
                walls_computed.append(
                    {
                        "corners": transformed,
                        "id": wid,
                        "source": "scan-cache-dedup",
                    }
                )
                added_uuids.add(wid)
                global_dedup_added.add(wid)

        if floor_polygon:
            _orient_walls_outward(walls_computed, floor_polygon)
            _orient_walls_outward(walls_merged, floor_polygon)

        # Helper: check if opening corners are near their parent wall
        # Returns True if opening center is within max_dist of parent wall center
        def _opening_near_parent(
            opening_corners, parent_id, max_dist=1.5, walls_computed=walls_computed
        ):
            parent_wall = next(
                (w for w in walls_computed if w["id"] == parent_id), None
            )
            if parent_wall is None:
                return True  # Can't check, assume OK
            oc = np.mean(opening_corners, axis=0)
            pc = np.mean(parent_wall["corners"], axis=0)
            return float(np.linalg.norm(oc - pc)) < max_dist

        # Doors: extract from merged room, use raw scan transform if available
        doors_out = []
        for d in mr.get("doors", []):
            did = d["identifier"]
            parent_id = d.get("parentIdentifier")
            if did in raw_door_data:
                raw_d, R, t_vec, method = raw_door_data[did]
                if method == "floor-svd":
                    raw_corners = wall_world_corners(raw_d)
                    transformed = [
                        (R @ np.array(c) + t_vec).tolist() for c in raw_corners
                    ]
                    if _opening_near_parent(transformed, parent_id):
                        doors_out.append(
                            {
                                "corners": transformed,
                                "id": did,
                                "source": "scan-cache",
                            }
                        )
                    else:
                        corners = wall_world_corners(d)
                        doors_out.append(
                            {
                                "corners": corners,
                                "id": did,
                                "source": "merged-room",
                            }
                        )
                else:
                    corners = wall_world_corners(d)
                    doors_out.append(
                        {
                            "corners": corners,
                            "id": did,
                            "source": "merged-room",
                        }
                    )
            else:
                corners = wall_world_corners(d)
                doors_out.append(
                    {
                        "corners": corners,
                        "id": did,
                        "source": "merged-room",
                    }
                )

        # Also add doors from raw rooms that were dropped during merge (dedup)
        added_door_ids = {d["id"] for d in doors_out}
        for rname, rdata in raw_rooms:
            if rname not in raw_transforms:
                continue
            R, t_vec, _res, method = raw_transforms[rname]
            if method != "floor-svd":
                continue
            raw_uuids = {w["identifier"] for w in rdata.get("walls", [])}
            mr_uuids = {w["identifier"] for w in mr.get("walls", [])}
            if not (raw_uuids & mr_uuids):
                continue
            for d in rdata.get("doors", []):
                did = d["identifier"]
                if (
                    did in added_door_ids
                    or did in global_dedup_doors_added
                    or did in merged_door_ids
                ):
                    continue
                raw_corners = wall_world_corners(d)
                transformed = [(R @ np.array(c) + t_vec).tolist() for c in raw_corners]
                parent_id = d.get("parentIdentifier")
                if not _opening_near_parent(transformed, parent_id):
                    continue  # Skip dedup doors displaced from parent wall
                doors_out.append(
                    {
                        "corners": transformed,
                        "id": did,
                        "source": "scan-cache-dedup",
                    }
                )
                added_door_ids.add(did)
                global_dedup_doors_added.add(did)

        # Windows: extract from merged room, use raw scan transform if available
        windows_out = []
        for w in mr.get("windows", []):
            wid = w["identifier"]
            parent_id = w.get("parentIdentifier")
            if wid in raw_window_data:
                raw_w, R, t_vec, method = raw_window_data[wid]
                if method == "floor-svd":
                    raw_corners = wall_world_corners(raw_w)
                    transformed = [
                        (R @ np.array(c) + t_vec).tolist() for c in raw_corners
                    ]
                    if _opening_near_parent(transformed, parent_id):
                        windows_out.append(
                            {
                                "corners": transformed,
                                "id": wid,
                                "source": "scan-cache",
                            }
                        )
                    else:
                        corners = wall_world_corners(w)
                        windows_out.append(
                            {
                                "corners": corners,
                                "id": wid,
                                "source": "merged-room",
                            }
                        )
                else:
                    corners = wall_world_corners(w)
                    windows_out.append(
                        {
                            "corners": corners,
                            "id": wid,
                            "source": "merged-room",
                        }
                    )
            else:
                corners = wall_world_corners(w)
                windows_out.append(
                    {
                        "corners": corners,
                        "id": wid,
                        "source": "merged-room",
                    }
                )

        # Also add windows from raw rooms that were dropped during merge (dedup)
        added_window_ids = {w["id"] for w in windows_out}
        for rname, rdata in raw_rooms:
            if rname not in raw_transforms:
                continue
            R, t_vec, _res, method = raw_transforms[rname]
            if method != "floor-svd":
                continue
            raw_uuids = {w["identifier"] for w in rdata.get("walls", [])}
            mr_uuids = {w["identifier"] for w in mr.get("walls", [])}
            if not (raw_uuids & mr_uuids):
                continue
            for w in rdata.get("windows", []):
                wid = w["identifier"]
                if (
                    wid in added_window_ids
                    or wid in global_dedup_windows_added
                    or wid in merged_window_ids
                ):
                    continue
                raw_corners = wall_world_corners(w)
                transformed = [(R @ np.array(c) + t_vec).tolist() for c in raw_corners]
                parent_id = w.get("parentIdentifier")
                if not _opening_near_parent(transformed, parent_id):
                    continue  # Skip dedup windows displaced from parent wall
                windows_out.append(
                    {
                        "corners": transformed,
                        "id": wid,
                        "source": "scan-cache-dedup",
                    }
                )
                added_window_ids.add(wid)
                global_dedup_windows_added.add(wid)

        # Openings (same extraction pattern as doors)
        openings_out = []
        for o in mr.get("openings", []):
            oid = o["identifier"]
            parent_id = o.get("parentIdentifier")
            if oid in raw_opening_data:
                raw_o, R, t_vec, method = raw_opening_data[oid]
                if method == "floor-svd":
                    raw_corners = wall_world_corners(raw_o)
                    transformed = [
                        (R @ np.array(c) + t_vec).tolist() for c in raw_corners
                    ]
                    if _opening_near_parent(transformed, parent_id):
                        openings_out.append(
                            {
                                "corners": transformed,
                                "id": oid,
                                "source": "scan-cache",
                            }
                        )
                    else:
                        corners = wall_world_corners(o)
                        openings_out.append(
                            {
                                "corners": corners,
                                "id": oid,
                                "source": "merged-room",
                            }
                        )
                else:
                    corners = wall_world_corners(o)
                    openings_out.append(
                        {
                            "corners": corners,
                            "id": oid,
                            "source": "merged-room",
                        }
                    )
            else:
                corners = wall_world_corners(o)
                openings_out.append(
                    {
                        "corners": corners,
                        "id": oid,
                        "source": "merged-room",
                    }
                )

        # Also add openings from raw rooms dropped during merge (dedup)
        added_opening_ids = {o["id"] for o in openings_out}
        for rname, rdata in raw_rooms:
            if rname not in raw_transforms:
                continue
            R, t_vec, _res, method = raw_transforms[rname]
            if method != "floor-svd":
                continue
            raw_uuids = {w["identifier"] for w in rdata.get("walls", [])}
            mr_uuids = {w["identifier"] for w in mr.get("walls", [])}
            if not (raw_uuids & mr_uuids):
                continue
            for o in rdata.get("openings", []):
                oid = o["identifier"]
                if (
                    oid in added_opening_ids
                    or oid in global_dedup_openings_added
                    or oid in merged_opening_ids
                ):
                    continue
                raw_corners = wall_world_corners(o)
                transformed = [(R @ np.array(c) + t_vec).tolist() for c in raw_corners]
                parent_id = o.get("parentIdentifier")
                if not _opening_near_parent(transformed, parent_id):
                    continue
                openings_out.append(
                    {
                        "corners": transformed,
                        "id": oid,
                        "source": "scan-cache-dedup",
                    }
                )
                added_opening_ids.add(oid)
                global_dedup_openings_added.add(oid)

        # Storage objects: synthesize corners from dimensions + transform (no
        # polygonCorners)
        storages_out = []
        added_storage_ids = set()
        for rname, rdata in raw_rooms:
            if rname not in raw_transforms:
                continue
            R, t_vec, _res, method = raw_transforms[rname]
            if method != "floor-svd":
                continue
            raw_uuids = {w["identifier"] for w in rdata.get("walls", [])}
            mr_uuids = {w["identifier"] for w in mr.get("walls", [])}
            if not (raw_uuids & mr_uuids):
                continue
            for obj in rdata.get("objects", []):
                cat = obj.get("category", {})
                if not (isinstance(cat, dict) and "storage" in cat):
                    continue
                sid = obj["identifier"]
                if sid in added_storage_ids or sid in global_dedup_storages_added:
                    continue
                # Synthesize rectangle corners from dimensions + transform
                corners = wall_world_corners(
                    obj
                )  # uses dimensions fallback since no polygonCorners
                transformed = [(R @ np.array(c) + t_vec).tolist() for c in corners]
                storages_out.append(
                    {
                        "corners": transformed,
                        "id": sid,
                        "source": "scan-cache",
                    }
                )
                added_storage_ids.add(sid)
                global_dedup_storages_added.add(sid)

        # Build parent lookup for door/window/opening clamping (applied after wall
        # post-processing)
        parent_lookup = {}
        for d in mr.get("doors", []):
            pid = d.get("parentIdentifier")
            if pid:
                parent_lookup[d["identifier"]] = pid
        for w in mr.get("windows", []):
            pid = w.get("parentIdentifier")
            if pid:
                parent_lookup[w["identifier"]] = pid
        for o in mr.get("openings", []):
            pid = o.get("parentIdentifier")
            if pid:
                parent_lookup[o["identifier"]] = pid
        for _rname, rdata in raw_rooms:
            for d in rdata.get("doors", []):
                pid = d.get("parentIdentifier")
                if pid and d["identifier"] not in parent_lookup:
                    parent_lookup[d["identifier"]] = pid
            for w in rdata.get("windows", []):
                pid = w.get("parentIdentifier")
                if pid and w["identifier"] not in parent_lookup:
                    parent_lookup[w["identifier"]] = pid
            for o in rdata.get("openings", []):
                pid = o.get("parentIdentifier")
                if pid and o["identifier"] not in parent_lookup:
                    parent_lookup[o["identifier"]] = pid

        raw_ceiling = ceilings_by_merged_idx.get(ri)
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
                "_parent_lookup": parent_lookup,
            }
        )

    # Clip overlapping floor polygons within each story
    floor_overlap_metrics = _clip_floor_overlaps(rooms_out)
    _reassign_raw_ceiling_planes_spatially(rooms_out)
    height_alignment_metrics = align_room_heights(rooms_out)

    # Extend raw walls upward to close vertical gaps between floors.
    # Build per-story slab index: story -> list of (shapely_polygon_xz, floor_y)
    # plus a matching centroid index for the legacy half-floor median filter.
    MAX_HALF_FLOOR = 0.50
    story_slab_raw = defaultdict(list)  # story -> [(polygon_xz, floor_y)]
    story_slab_centroid_ys = defaultdict(list)
    for room in rooms_out:
        fp = room["floor_polygon"]
        if fp and len(fp) >= 3:
            cy = float(np.mean([c[1] for c in fp]))
            try:
                poly = Polygon([(c[0], c[2]) for c in fp])
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                continue
            if poly.is_empty or not poly.is_valid:
                continue
            story_slab_raw[room["story"]].append((poly, cy))
            story_slab_centroid_ys[room["story"]].append(cy)

    # Build story_y_map from the post-filter Y cluster (used by wall clipping).
    # Keep all slab polygons in story_slabs -- the polygon-aware picker naturally
    # handles half-level rooms via distance, so no need to exclude them.
    story_slabs = dict(story_slab_raw)
    story_y_map = {}
    for story, ys in story_slab_centroid_ys.items():
        median_y = float(np.median(ys))
        kept = [y for y in ys if abs(y - median_y) <= MAX_HALF_FLOOR]
        story_y_map[story] = float(np.median(kept)) if kept else median_y
    wall_clip_metrics = _clip_walls_to_story_bounds(rooms_out, story_y_map)

    for room in rooms_out:
        # Consider slabs from every story strictly above this room's story.
        # Split-level buildings can have a half-floor wing at story+1 whose
        # footprint sits beside (not over) the wall -- its slab then fails
        # find_best_slab_above's XZ-distance test or sits below the wall top.
        # Including story+2, +3, ... lets a wall still reach the next full
        # slab directly overhead. Sorted ascending so ties break toward the
        # lowest viable slab (physically correct).
        slabs_above = []
        for s in sorted(story_slabs):
            if s > room["story"]:
                slabs_above.extend(story_slabs[s])
        for w in room["walls_computed"]:
            wc = w["corners"]
            if not wc:
                w["extension_strip"] = None
                continue
            wall_top_y = max(c[1] for c in wc)
            slab_y = find_best_slab_above(wc, wall_top_y, slabs_above, max_gap=0.80)
            if slab_y is None:
                w["extension_strip"] = None
                continue
            result = _extend_wall_to_slab(wc, slab_y)
            if result is None:
                w["extension_strip"] = None
            else:
                w["extension_strip"] = result["extension_strip"]

    # Infer ceiling polygons from roof plane fitting
    _infer_ceilings(rooms_out)

    # Cross-floor gap detection from floor polygons
    cross_floor_gaps_out = _compute_cross_floor_gaps(rooms_out)

    # Snapshot pre-absorption floor polygons -- see compute_gap_walls docstring
    # in reconcile.extract3d.gaps for why this matters.
    pre_absorption_floor_polygons = [
        list(room.get("floor_polygon") or []) for room in rooms_out
    ]

    # Assign within-story gaps to nearest rooms and expand floor polygons
    _assign_gaps_to_rooms(cross_floor_gaps_out, rooms_out)

    # Detect exterior gap indicators from doors/openings
    exterior_gap_indicators = _detect_exterior_gap_indicators(rooms_out)

    # Compute gap closure geometry (side walls that close detected gaps)
    gap_closures = _compute_gap_closures(exterior_gap_indicators, rooms_out)

    # Create wall quads along gap edges (uses room walls + gap closures for Y snap)
    gap_walls = _compute_gap_walls(
        cross_floor_gaps_out,
        rooms_out,
        story_y_map,
        gap_closures,
        pre_absorption_floor_polygons=pre_absorption_floor_polygons,
    )

    # Stitch disconnected wall endpoints within each room
    stitch_walls = _stitch_wall_gaps(rooms_out)

    # Clamp doors/windows to their parent wall bounds (after all wall post-processing)
    for room in rooms_out:
        parent_lookup = room.pop("_parent_lookup", {})
        wall_corners_by_id = {w["id"]: w["corners"] for w in room["walls_computed"]}
        for w in room["walls_merged"]:
            if w["id"] not in wall_corners_by_id:
                wall_corners_by_id[w["id"]] = w["corners"]

        # Room-level bounding box as fallback for openings without parent match
        all_wc = [c for w in room["walls_computed"] for c in w["corners"]]
        room_bbox = None
        if all_wc:
            arr = np.array(all_wc)
            room_bbox = (arr.min(axis=0), arr.max(axis=0))

        for opening_list in (room["doors"], room["windows"]):
            for opening in opening_list:
                pid = parent_lookup.get(opening["id"])
                if pid and pid in wall_corners_by_id:
                    opening["corners"] = clamp_opening_to_parent(
                        opening["corners"], wall_corners_by_id[pid]
                    )
                elif room_bbox is not None:
                    bbox_min, bbox_max = room_bbox
                    opening["corners"] = [
                        np.clip(c, bbox_min, bbox_max).tolist()
                        for c in opening["corners"]
                    ]

    computed_total = sum(len(r["walls_computed"]) for r in rooms_out)
    merged_total = sum(len(r["walls_merged"]) for r in rooms_out)
    scan_cache_count = sum(
        1
        for r in rooms_out
        for w in r["walls_computed"]
        if w.get("source") == "scan-cache"
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
        "cross_floor_gaps": cross_floor_gaps_out,
        "gap_walls": gap_walls,
        "stitch_walls": stitch_walls,
        "exterior_gap_indicators": exterior_gap_indicators,
        "gap_closures": gap_closures,
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


def main():
    pipeline_dir = Path("pipeline-outputs")
    scan_cache_root = Path(".scan-cache")

    # Select buildings to extract: CLI args or all from pipeline-outputs
    if len(sys.argv) > 1:
        uuids = sys.argv[1:]
    else:
        uuids = sorted(
            entry.name
            for entry in pipeline_dir.iterdir()
            if entry.is_dir() and (entry / "merged.json").exists()
        )

    results = []
    for uuid in uuids:
        print(f"Extracting {uuid}...")
        try:
            result = extract_building(uuid, pipeline_dir, scan_cache_root)
        except Exception as exc:
            print(f"  FAILED ({type(exc).__name__}: {exc})")
            continue
        if result:
            results.append(result)
            computed_total = result["computed_walls_total"]
            merged_total = result["merged_walls_total"]
            scan_cache = result["scan_cache_walls"]
            print(
                f"  {len(result['rooms'])} rooms, {result['stories_found']} stories, "
                f"{computed_total} computed walls ({scan_cache} from scan-cache), "
                f"{merged_total} merged walls, "
                f"{result['scan_rooms_found']} scan rooms "
                f"({result['scan_rooms_transformed']} transformed)"
            )
        else:
            print("  SKIPPED (no merged.json)")

    out_path = Path("reconcile/buildings_3d.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"\nWrote {len(results)} buildings to {out_path}")

    _run_roof_pipeline(results, pipeline_dir, scan_cache_root)


def _run_roof_pipeline(
    buildings: list[dict], pipeline_dir: Path, scan_cache_root: Path
) -> None:
    from reconcile.roof_algorithms_py import run_roof_algorithms
    from reconcile_v2.graph_builder import build_topology_graph

    roof_out_path = Path("reconcile/roof_algorithms_py_results.json")
    try:
        with open(roof_out_path) as f:
            roof_results: dict = json.load(f)
    except FileNotFoundError:
        roof_results = {}

    failures = 0
    for bldg in buildings:
        uuid = bldg["uuid"]
        merged_path = pipeline_dir / uuid / "merged.json"
        try:
            graph = build_topology_graph(
                merged_path=merged_path,
                scan_dir=scan_cache_root,
                uuid=uuid,
            )
            roof_results[uuid] = run_roof_algorithms(bldg, graph=graph)
            print(f"  [roof] ok {uuid}")
        except Exception as exc:
            failures += 1
            print(f"  [roof] error {uuid}: {exc}", file=sys.stderr)

    tmp_path = roof_out_path.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(roof_results), encoding="utf-8")
    tmp_path.replace(roof_out_path)
    print(
        f"Wrote roof results for {len(buildings) - failures}/{len(buildings)} "
        f"buildings to {roof_out_path}"
    )


if __name__ == "__main__":
    main()
