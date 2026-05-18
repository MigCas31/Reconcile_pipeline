"""Detect dormer front walls protruding through oblique roof surfaces
and group them with adjacent cheek walls."""

from __future__ import annotations

from math import atan2, sqrt

from shapely.geometry import Point, Polygon

from .math_utils import angle_diff, plane_normal, plane_y_at

# Tolerances
RIDGE_DOT_MAX = 0.35  # ~20° — wall normal must be roughly perpendicular to ridge
PROTRUSION_THRESH = 0.10  # metres above/below roof plane
MAX_BOTTOM_ABOVE_ROOF = 1.5  # metres — max distance wall bottom can be above roof
MAX_WIDTH_RATIO = 0.70  # dormer width < 70% of ridge span
MAX_HEIGHT_RATIO = 0.75  # dormer wall height < 75% of room's tallest wall
CORNER_MATCH_TOL = 0.15  # metres — shared-corner matching tolerance
CHEEK_ANGLE_TOL = 30.0  # degrees — cheek azimuth must be ~90° off front wall


def _wall_normal_xz(corners: list) -> tuple[float, float]:
    """Return the XZ unit normal of a wall quad.

    Uses the bottom two corners (lowest Y) to define the wall direction,
    then returns the perpendicular in the XZ plane.
    """
    pts = sorted(corners, key=lambda c: c[1])  # sort by Y ascending
    b0, b1 = pts[0], pts[1]
    dx = b1[0] - b0[0]
    dz = b1[2] - b0[2]
    length = sqrt(dx * dx + dz * dz)
    if length < 1e-6:
        return 0.0, 0.0
    return -dz / length, dx / length


def _wall_azimuth(corners: list) -> float:
    """Return azimuth (degrees, 0-360) of a wall's outward normal in XZ."""
    nx, nz = _wall_normal_xz(corners)
    if abs(nx) < 1e-9 and abs(nz) < 1e-9:
        return 0.0
    return atan2(nx, nz) * 180.0 / 3.141592653589793 % 360.0


def _wall_width_along_ridge(corners: list, ridge_x: float, ridge_z: float) -> float:
    """Project wall corners onto the ridge direction and return span."""
    projs = [(c[0] * ridge_x + c[2] * ridge_z) for c in corners]
    return max(projs) - min(projs)


def _point_near_roof_xz(
    px: float, pz: float, surface: dict, margin: float = 2.0
) -> bool:
    """Check if (px, pz) is within the XZ bounding box of the roof surface."""
    corners = surface["corners"]
    xs = [c[0] for c in corners]
    zs = [c[2] for c in corners]
    return (
        min(xs) - margin <= px <= max(xs) + margin
        and min(zs) - margin <= pz <= max(zs) + margin
    )


def detect_dormers(
    bldg: dict,
    oblique_roof_surfaces: list[dict],
    story_floor_y: dict,
    roof_coverage_graph: dict | None = None,
    top_boundary_graph: dict | None = None,
    knee_walls: list[dict] | None = None,
    roof_evidence_graph: dict | None = None,
) -> list[dict]:
    """Detect dormer front walls that protrude through oblique roof surfaces.

    Returns a list of dormer descriptors, each containing:
      - front_wall: {"id", "corners", "room_index"}
      - roof_surface_index: int
      - roof_plane: {"n": {x,y,z}, "ref": {x,y,z}}
    """
    dormers: list[dict] = []
    sloped_rooms_by_hypothesis: dict[str, set[int]] = {}
    subparts_by_id = {
        str(subpart["id"]): subpart
        for subpart in ((roof_coverage_graph or {}).get("subparts") or [])
        if isinstance(subpart, dict)
    }
    subpart_evidence = (roof_evidence_graph or {}).get("subpart_evidence") or {}
    atom_subpart_membership = (roof_coverage_graph or {}).get(
        "atom_subpart_membership"
    ) or {}
    covered_polys_by_hypothesis_room: dict[tuple[str, int], list[dict]] = {}
    for atom_id, region in (
        (roof_coverage_graph or {}).get("atom_regions") or {}
    ).items():
        coverage = ((roof_coverage_graph or {}).get("atom_coverage") or {}).get(
            atom_id
        ) or {}
        hypothesis_id = coverage.get("sloped_hypothesis_id")
        state = str(coverage.get("sloped_state", "none"))
        room_index = region.get("room_index")
        polygon_xz = region.get("polygon_xz") or []
        if (
            not hypothesis_id
            or state not in {"confirmed", "partial"}
            or not isinstance(room_index, int)
        ):
            continue
        if len(polygon_xz) < 3:
            continue
        try:
            poly = Polygon([(float(p[0]), float(p[1])) for p in polygon_xz])
        except Exception:
            continue
        if poly.is_empty or not poly.is_valid or poly.area <= 0.01:
            continue
        subpart_ids = [
            str(subpart_id) for subpart_id in atom_subpart_membership.get(atom_id, [])
        ]
        semantic_kinds = sorted(
            {
                str(subparts_by_id[subpart_id].get("semantic_kind", "slope_run"))
                for subpart_id in subpart_ids
                if subpart_id in subparts_by_id
            }
        )
        covered_polys_by_hypothesis_room.setdefault(
            (str(hypothesis_id), room_index), []
        ).append(
            {
                "poly": poly,
                "subpart_ids": subpart_ids,
                "semantic_kinds": semantic_kinds,
                "support_evidence_score": max(
                    int(
                        (subpart_evidence.get(subpart_id) or {}).get(
                            "evidence_score", 0
                        )
                        or 0
                    )
                    for subpart_id in subpart_ids
                )
                if subpart_ids
                else 0,
            }
        )

    knee_walls_by_room: dict[int, list[dict]] = {}
    for knee_wall in knee_walls or []:
        room_index = knee_wall.get("room_index")
        if isinstance(room_index, int):
            knee_walls_by_room.setdefault(room_index, []).append(knee_wall)

    if top_boundary_graph is not None:
        for node in top_boundary_graph.get("nodes") or []:
            if node.get("type") != "TopBoundaryAtom":
                continue
            if node.get("role") != "sloped_ceiling":
                continue
            hypothesis_id = node.get("roof_hypothesis_id")
            room_index = node.get("room_index")
            if not hypothesis_id or not isinstance(room_index, int):
                continue
            sloped_rooms_by_hypothesis.setdefault(str(hypothesis_id), set()).add(
                room_index
            )

    for surf_idx, surface in enumerate(oblique_roof_surfaces):
        cluster = surface["cluster"]
        ridge_x = surface["ridge"]["x"]
        ridge_z = surface["ridge"]["z"]
        ridge_span = surface["ridge"]["max"] - surface["ridge"]["min"]
        normal = plane_normal(cluster["avgAzimuth"], cluster["avgIncl"])
        center = surface["center"]
        plane = {"n": normal, "ref": center}
        dominant_story = surface["dominant_story"]
        allowed_room_indices = sloped_rooms_by_hypothesis.get(
            str(surface.get("roof_hypothesis_id"))
        )
        surface_hypothesis_id = str(surface.get("roof_hypothesis_id"))

        for room_idx, room in enumerate(bldg.get("rooms", [])):
            if room.get("story") != dominant_story:
                continue
            if (
                allowed_room_indices is not None
                and room_idx not in allowed_room_indices
            ):
                continue

            # Compute the tallest wall in this room for height-ratio filtering.
            room_max_wall_h = 0.0
            for _w in room.get("walls_merged", []):
                _wc = _w.get("corners", [])
                if len(_wc) >= 3:
                    _h = max(c[1] for c in _wc) - min(c[1] for c in _wc)
                    if _h > room_max_wall_h:
                        room_max_wall_h = _h

            # Check both computed and merged walls; track seen IDs
            # to avoid duplicates.  Computed walls are iterated first
            # because they better represent individual wall segments
            # (merged walls may combine adjacent segments, inflating height).
            seen_ids: set[str] = set()
            all_walls = list(room.get("walls_computed", []))
            all_walls.extend(room.get("walls_merged", []))

            for wall in all_walls:
                wid = wall.get("id", "")
                if wid in seen_ids:
                    continue
                seen_ids.add(wid)

                corners = wall.get("corners", [])
                if len(corners) < 3:
                    continue

                # 1. Orientation: wall normal must be ~perpendicular to ridge
                wnx, wnz = _wall_normal_xz(corners)
                dot_ridge = abs(wnx * ridge_x + wnz * ridge_z)
                if dot_ridge > RIDGE_DOT_MAX:
                    continue

                # 2. Protrusion: wall top must be above the roof plane.
                # Accept walls that straddle the roof (classic case) OR
                # walls entirely above the roof if their bottom is within
                # MAX_BOTTOM_ABOVE_ROOF of the plane (short dormer walls).
                roof_ys = [plane_y_at(plane, c[0], c[2]) for c in corners]
                above = [
                    c[1] > ry + PROTRUSION_THRESH
                    for c, ry in zip(corners, roof_ys, strict=False)
                ]
                if not any(above):
                    continue
                if all(above):
                    # All corners above — only accept if bottom is close
                    # to the roof plane (short dormer sitting on top).
                    min_gap = min(
                        c[1] - ry for c, ry in zip(corners, roof_ys, strict=False)
                    )
                    if min_gap > MAX_BOTTOM_ABOVE_ROOF:
                        continue

                # 3. Size: dormer is smaller than the roof
                wall_width = _wall_width_along_ridge(corners, ridge_x, ridge_z)
                if ridge_span > 0 and wall_width > ridge_span * MAX_WIDTH_RATIO:
                    continue

                # 4. Height: dormer wall must be shorter than the room's
                #    tallest wall — full-height exterior walls are not dormers.
                wall_height = max(c[1] for c in corners) - min(c[1] for c in corners)
                if (
                    room_max_wall_h > 0
                    and wall_height > room_max_wall_h * MAX_HEIGHT_RATIO
                ):
                    continue

                # 5. Proximity: wall center within roof XZ bounds
                wcx = sum(c[0] for c in corners) / len(corners)
                wcz = sum(c[2] for c in corners) / len(corners)
                if not _point_near_roof_xz(wcx, wcz, surface):
                    continue
                covered_regions = covered_polys_by_hypothesis_room.get(
                    (surface_hypothesis_id, room_idx), []
                )
                if covered_regions:
                    pt = Point(wcx, wcz)
                    matching_regions = [
                        region
                        for region in covered_regions
                        if region["poly"].buffer(0.1).contains(pt)
                    ]
                    if not matching_regions:
                        continue
                    if all(
                        region["semantic_kinds"] == ["l_t_branch"]
                        and int(region.get("support_evidence_score", 0) or 0) < 3
                        for region in matching_regions
                    ):
                        continue
                if _is_knee_wall_front(wall, knee_walls_by_room.get(room_idx, [])):
                    continue

                dormers.append(
                    {
                        "front_wall": {
                            "id": wid,
                            "corners": corners,
                            "room_index": room_idx,
                        },
                        "roof_surface_index": surf_idx,
                        "roof_plane": plane,
                    }
                )

    return dormers


def _is_knee_wall_front(
    wall: dict,
    knee_walls: list[dict],
    max_dist: float = 0.25,
    max_angle_deg: float = 20.0,
) -> bool:
    corners = wall.get("corners", [])
    if len(corners) < 2:
        return False
    wall_mid = element_midpoint_xz(corners)
    wall_az = _wall_azimuth(corners)
    if wall_mid is None:
        return False
    for knee_wall in knee_walls:
        knee_corners = knee_wall.get("corners", [])
        if len(knee_corners) < 2:
            continue
        knee_mid = element_midpoint_xz(knee_corners)
        if knee_mid is None:
            continue
        dist = sqrt((wall_mid[0] - knee_mid[0]) ** 2 + (wall_mid[1] - knee_mid[1]) ** 2)
        if dist > max_dist:
            continue
        knee_az = _wall_azimuth(knee_corners)
        if angle_diff(wall_az, knee_az) <= max_angle_deg:
            return True
    return False


def element_midpoint_xz(corners: list) -> tuple[float, float] | None:
    if not corners:
        return None
    return (
        sum(float(c[0]) for c in corners) / len(corners),
        sum(float(c[2]) for c in corners) / len(corners),
    )


def _corners_share_vertex(
    corners_a: list, corners_b: list, tol: float = CORNER_MATCH_TOL
) -> list[tuple[int, int]]:
    """Return pairs of indices (i_a, i_b) where corners nearly coincide."""
    matches = []
    for ia, ca in enumerate(corners_a):
        for ib, cb in enumerate(corners_b):
            dx = ca[0] - cb[0]
            dy = ca[1] - cb[1]
            dz = ca[2] - cb[2]
            if sqrt(dx * dx + dy * dy + dz * dz) < tol:
                matches.append((ia, ib))
    return matches


def group_dormer_walls(front_wall: dict, room_walls: list[dict]) -> dict:
    """Starting from a dormer front wall, trace adjacent cheek walls.

    Returns {"front": wall, "left_cheek": wall|None, "right_cheek": wall|None}.
    Left/right is determined by which side of the front wall the cheek connects to.
    """
    front_corners = front_wall["corners"]
    front_az = _wall_azimuth(front_corners)

    # Identify bottom-left and bottom-right of front wall along ridge direction.
    # The two bottom corners define the wall's horizontal extent.
    sorted_by_y = sorted(range(len(front_corners)), key=lambda i: front_corners[i][1])
    bottom_indices = sorted_by_y[:2]

    # Determine left/right based on XZ ordering relative to wall normal
    b0, b1 = front_corners[bottom_indices[0]], front_corners[bottom_indices[1]]
    wnx, wnz = _wall_normal_xz(front_corners)
    # Cross product: if (b1 - b0) cross normal > 0, b0 is "left"
    cross = (b1[0] - b0[0]) * wnz - (b1[2] - b0[2]) * wnx
    if cross >= 0:
        left_bottom_idx, right_bottom_idx = bottom_indices[0], bottom_indices[1]
    else:
        left_bottom_idx, right_bottom_idx = bottom_indices[1], bottom_indices[0]

    # Find the top corners above each bottom corner
    left_top_idx = _find_top_corner(front_corners, left_bottom_idx)
    right_top_idx = _find_top_corner(front_corners, right_bottom_idx)

    left_indices = {left_bottom_idx, left_top_idx}
    right_indices = {right_bottom_idx, right_top_idx}

    left_cheek = None
    right_cheek = None

    for wall in room_walls:
        if wall["id"] == front_wall["id"]:
            continue
        wc = wall.get("corners", [])
        if len(wc) < 3:
            continue

        # Check azimuth: cheeks should be ~90° off from front wall
        waz = _wall_azimuth(wc)
        diff = angle_diff(front_az, waz)
        if abs(diff - 90.0) > CHEEK_ANGLE_TOL:
            continue

        # Check shared corners
        matches = _corners_share_vertex(front_corners, wc)
        if not matches:
            continue

        front_matched = {m[0] for m in matches}

        if front_matched & left_indices and left_cheek is None:
            left_cheek = wall
        elif front_matched & right_indices and right_cheek is None:
            right_cheek = wall

    return {
        "front": front_wall,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
    }


def _find_top_corner(corners: list, bottom_idx: int) -> int:
    """Find the corner directly above bottom_idx (closest in XZ, highest Y)."""
    bc = corners[bottom_idx]
    best_idx = bottom_idx
    best_score = -1e9
    for i, c in enumerate(corners):
        if i == bottom_idx:
            continue
        dx = c[0] - bc[0]
        dz = c[2] - bc[2]
        xz_dist = sqrt(dx * dx + dz * dz)
        # Prefer high Y, low XZ distance
        if xz_dist < 0.5 and c[1] > best_score:
            best_score = c[1]
            best_idx = i
    return best_idx
