"""Build dormer cheek, header, and roof-cutout geometry.

Given a dormer group (front wall + optional existing cheeks) and the
oblique roof surface it penetrates, compute:
  - Trimmed / extended / generated cheek surfaces
  - A horizontal header surface (if not already covered by a ceiling)
  - A cutout quadrilateral on the roof plane
"""

from __future__ import annotations

from math import cos, sin, sqrt

from .math_utils import plane_normal, plane_y_at


def build_dormer_geometry(
    dormer_group: dict,
    roof_surface: dict,
    room_walls: list[dict] | None = None,
) -> dict:
    """Build complete dormer geometry from a dormer group and roof surface.

    Parameters
    ----------
    dormer_group : dict
        {"front": wall, "left_cheek": wall|None, "right_cheek": wall|None}
    roof_surface : dict
        An oblique roof surface with cluster, center, ridge, corners.
    room_walls : list[dict] | None
        All walls in the room, used for overlap clipping.

    Returns
    -------
    dict with keys:
        cheeks: list of {"corners": [...], "source": str}
        header: {"corners": [...], "source": str}
        cutout_corners: list of 4 points on the roof plane
    """
    front = dormer_group["front"]
    front_corners = front["corners"]
    cluster = roof_surface["cluster"]
    center = roof_surface["center"]
    normal = plane_normal(cluster["avgAzimuth"], cluster["avgIncl"])
    plane = {"n": normal, "ref": center}

    # Compute slope direction (downhill) and ridge direction in 3D
    azi_rad = cluster["avgAzimuth"] * 3.141592653589793 / 180.0
    incl_rad = cluster["avgIncl"] * 3.141592653589793 / 180.0
    slope_x = sin(azi_rad) * cos(incl_rad)
    -sin(incl_rad)
    slope_z = cos(azi_rad) * cos(incl_rad)
    ridge_x = roof_surface["ridge"]["x"]
    ridge_z = roof_surface["ridge"]["z"]

    # Dormer top Y = max Y of front wall
    dormer_top_y = max(c[1] for c in front_corners)

    # The dormer width is defined by the TOP edge of the front wall — the
    # part that actually protrudes above the roof.  Take the two highest-Y
    # corners that are distinct in XZ.
    sorted_by_y_desc = sorted(front_corners, key=lambda c: -c[1])
    top_corners = [sorted_by_y_desc[0]]
    for c in sorted_by_y_desc[1:]:
        dx = c[0] - top_corners[0][0]
        dz = c[2] - top_corners[0][2]
        if sqrt(dx * dx + dz * dz) > 0.1:
            top_corners.append(c)
            break
    if len(top_corners) < 2:
        # Fallback: two highest-Y corners
        top_corners = sorted_by_y_desc[:2]

    # Order along ridge direction to get left/right
    def ridge_proj(c):
        return c[0] * ridge_x + c[2] * ridge_z

    top_corners.sort(key=ridge_proj)
    front_left = top_corners[0]
    front_right = top_corners[1]

    # Roof height at front left/right
    roof_y_left = plane_y_at(plane, front_left[0], front_left[2])
    roof_y_right = plane_y_at(plane, front_right[0], front_right[2])

    # Points where front wall meets roof plane (bottom of dormer on roof)
    front_left_on_roof = (front_left[0], roof_y_left, front_left[2])
    front_right_on_roof = (front_right[0], roof_y_right, front_right[2])

    # Top of front wall at left/right
    front_left_top = (front_left[0], dormer_top_y, front_left[2])
    front_right_top = (front_right[0], dormer_top_y, front_right[2])

    # Back of dormer: walk uphill along slope until roof height = dormer_top_y
    # to find the MAX possible depth.  Then use a fraction of that depth so
    # the back edge has non-zero height — producing a trapezoid, not a triangle.
    max_back_left = _walk_to_roof_height(
        front_left, dormer_top_y, plane, slope_x, slope_z
    )
    max_back_right = _walk_to_roof_height(
        front_right, dormer_top_y, plane, slope_x, slope_z
    )

    if max_back_left is None and max_back_right is None:
        # Degenerate — roof doesn't reach dormer top on either side
        return {
            "cheeks": [],
            "header": None,
            "cutout_corners": [],
        }

    # If one side fails, mirror the depth from the successful side
    if max_back_left is None:
        max_back_left = _mirror_back_point(
            front_left, front_right, max_back_right, slope_x, slope_z, plane
        )
    elif max_back_right is None:
        max_back_right = _mirror_back_point(
            front_right, front_left, max_back_left, slope_x, slope_z, plane
        )

    # Use 70% of max depth so the back edge has non-zero height,
    # producing visible trapezoidal cheeks instead of collapsed triangles.
    DEPTH_FRACTION = 0.70

    back_left = _interpolate_back(front_left, max_back_left, DEPTH_FRACTION)
    back_right = _interpolate_back(front_right, max_back_right, DEPTH_FRACTION)

    # Back points: bottom on roof plane, top at dormer_top_y
    back_left_roof_y = plane_y_at(plane, back_left[0], back_left[2])
    back_right_roof_y = plane_y_at(plane, back_right[0], back_right[2])
    back_left_on_roof = (back_left[0], back_left_roof_y, back_left[2])
    back_right_on_roof = (back_right[0], back_right_roof_y, back_right[2])

    # Back top points at dormer_top_y (distinct from back_on_roof)
    back_left_top = (back_left[0], dormer_top_y, back_left[2])
    back_right_top = (back_right[0], dormer_top_y, back_right[2])

    # Cutout quad on the roof plane (4 points)
    cutout_corners = [
        front_left_on_roof,
        front_right_on_roof,
        back_right_on_roof,
        back_left_on_roof,
    ]

    # Collect room walls for overlap clipping (exclude front wall and
    # detected cheek walls — those are handled via existing_wall param).
    front_id = front.get("id", "")
    left_cheek_wall = dormer_group.get("left_cheek")
    right_cheek_wall = dormer_group.get("right_cheek")
    exclude_ids = {front_id}
    if left_cheek_wall:
        exclude_ids.add(left_cheek_wall.get("id", ""))
    if right_cheek_wall:
        exclude_ids.add(right_cheek_wall.get("id", ""))
    clip_walls = [
        w
        for w in (room_walls or [])
        if w.get("id", "") not in exclude_ids and len(w.get("corners", [])) >= 3
    ]

    # Build cheeks — now proper trapezoids since back_on_roof.y < back_top.y
    left_cheek = _build_cheek(
        front_on_roof=front_left_on_roof,
        front_top=front_left_top,
        back_on_roof=back_left_on_roof,
        back_top=back_left_top,
        existing_wall=left_cheek_wall,
        plane=plane,
        dormer_top_y=dormer_top_y,
        clip_walls=clip_walls,
    )
    right_cheek = _build_cheek(
        front_on_roof=front_right_on_roof,
        front_top=front_right_top,
        back_on_roof=back_right_on_roof,
        back_top=back_right_top,
        existing_wall=right_cheek_wall,
        plane=plane,
        dormer_top_y=dormer_top_y,
        clip_walls=clip_walls,
    )

    cheeks = [left_cheek, right_cheek]

    # Build header
    header = _build_header(
        front_left_top=front_left_top,
        front_right_top=front_right_top,
        back_left_top=back_left_top,
        back_right_top=back_right_top,
        dormer_top_y=dormer_top_y,
    )

    return {
        "cheeks": cheeks,
        "header": header,
        "cutout_corners": cutout_corners,
    }


def _interpolate_back(
    front: tuple | list,
    max_back: tuple | list,
    frac: float,
) -> tuple:
    """Interpolate between front and max_back by fraction frac (0..1)."""
    return (
        front[0] + frac * (max_back[0] - front[0]),
        front[1] + frac * (max_back[1] - front[1]),
        front[2] + frac * (max_back[2] - front[2]),
    )


def _walk_to_roof_height(
    start_xz: tuple | list,
    target_y: float,
    plane: dict,
    slope_x: float,
    slope_z: float,
    max_t: float = 20.0,
) -> tuple | None:
    """Walk uphill (opposite slope direction) from start until roof = target_y.

    Returns (x, target_y, z) or None if the roof never reaches target_y.
    """
    n = plane["n"]
    plane["ref"]
    sx, sz = start_xz[0], start_xz[2]

    # plane_y_at(x, z) = ref_y - (nx*(x - ref_x) + nz*(z - ref_z)) / ny
    # We walk: x(t) = sx - slope_x * t, z(t) = sz - slope_z * t
    # Substituting:
    #   y(t) = ref_y - (nx*(sx - slope_x*t - ref_x) + nz*(sz - slope_z*t - ref_z)) / ny
    #   y(t) = y(0) + t * (nx*slope_x + nz*slope_z) / ny
    # Solve y(t) = target_y:
    ny = n["y"]
    if abs(ny) < 1e-9:
        return None

    y0 = plane_y_at(plane, sx, sz)
    rate = (n["x"] * slope_x + n["z"] * slope_z) / ny

    if abs(rate) < 1e-9:
        return None  # roof doesn't change height in slope direction

    t = (target_y - y0) / rate
    if t < 0 or t > max_t:
        return None

    return (sx - slope_x * t, target_y, sz - slope_z * t)


def _build_cheek(
    front_on_roof: tuple,
    front_top: tuple,
    back_on_roof: tuple,
    back_top: tuple,
    existing_wall: dict | None,
    plane: dict,
    dormer_top_y: float,
    clip_walls: list[dict] | None = None,
) -> dict:
    """Build a dormer cheek surface, clipped against existing walls.

    The ideal cheek is a trapezoid:
        front_top ---- back_top     (horizontal at dormer_top_y)
            |              |
        front_on_roof -- back_on_roof  (on sloped roof plane)

    The ideal cheek is always generated first, then clipped against any
    existing walls (including the detected cheek wall) to prevent overlap.
    """
    ideal_corners = [front_on_roof, back_on_roof, back_top, front_top]

    # Collect all wall corner lists to clip against
    walls_to_clip: list[list] = []
    if existing_wall is not None:
        ec = existing_wall.get("corners", [])
        if len(ec) >= 3:
            walls_to_clip.append(ec)
    for w in clip_walls or []:
        walls_to_clip.append(w.get("corners", w) if isinstance(w, dict) else w)

    if not walls_to_clip:
        return {"corners": ideal_corners, "source": "generated"}

    clipped = _clip_cheek_against_walls(
        ideal_corners, walls_to_clip, front_on_roof, back_on_roof
    )

    if not clipped or len(clipped) < 3:
        # Fully covered by existing walls — no additional cheek needed
        return {"corners": [], "source": "clipped-empty"}

    return {"corners": clipped, "source": "generated"}


def _build_header(
    front_left_top: tuple,
    front_right_top: tuple,
    back_left_top: tuple,
    back_right_top: tuple,
    dormer_top_y: float,
) -> dict:
    """Build a horizontal header surface at dormer_top_y."""
    header_corners = [
        front_left_top,
        front_right_top,
        back_right_top,
        back_left_top,
    ]
    return {"corners": header_corners, "source": "generated"}


def _mirror_back_point(
    fail_front: tuple | list,
    ok_front: tuple | list,
    ok_back: tuple | list,
    slope_x: float,
    slope_z: float,
    plane: dict,
) -> tuple:
    """Compute a back point for a failing side by mirroring slope distance.

    Walks the same slope-direction distance from fail_front as the
    successful side walked from ok_front to ok_back.
    """
    dx_ok = ok_back[0] - ok_front[0]
    dz_ok = ok_back[2] - ok_front[2]
    # The walk is: x(t) = start - slope_x * t.  Recover t from the successful side.
    slope_len_sq = slope_x * slope_x + slope_z * slope_z
    if slope_len_sq < 1e-12:
        return ok_back  # degenerate — return the other side's back as fallback
    t = -(dx_ok * slope_x + dz_ok * slope_z) / slope_len_sq
    x = fail_front[0] - slope_x * t
    z = fail_front[2] - slope_z * t
    y = plane_y_at(plane, x, z)
    return (x, y, z)


def _clip_cheek_against_walls(
    ideal_corners: list,
    wall_corner_lists: list[list],
    front: tuple,
    back: tuple,
) -> list:
    """Clip ideal cheek polygon against overlapping walls using 2D projection.

    Projects both the cheek and walls into the cheek's vertical plane
    (u = depth along front→back, v = Y), performs a Shapely polygon
    difference, and converts back to 3D.

    Returns the clipped 3D corner list, or [] if fully covered.
    """
    dx = back[0] - front[0]
    dz = back[2] - front[2]
    depth = sqrt(dx * dx + dz * dz)
    if depth < 0.01:
        return list(ideal_corners)

    dir_x = dx / depth
    dir_z = dz / depth
    # Cheek plane normal in XZ (perpendicular to depth direction)
    normal_x = -dir_z
    normal_z = dir_x

    def to_uv(corners):
        return [
            ((c[0] - front[0]) * dir_x + (c[2] - front[2]) * dir_z, c[1])
            for c in corners
        ]

    def from_uv(pts):
        return [(front[0] + u * dir_x, v, front[2] + u * dir_z) for u, v in pts]

    # Filter walls that lie roughly in the cheek's vertical plane
    coplanar_walls = []
    for wc in wall_corner_lists:
        if len(wc) < 3:
            continue
        # Check signed distance of each corner to cheek plane
        dists = [
            (c[0] - front[0]) * normal_x + (c[2] - front[2]) * normal_z for c in wc
        ]
        if not all(abs(d) < 0.3 for d in dists):
            continue
        # Check overlap in depth direction
        projs = [(c[0] - front[0]) * dir_x + (c[2] - front[2]) * dir_z for c in wc]
        if max(projs) < -0.1 or min(projs) > depth + 0.1:
            continue
        coplanar_walls.append(wc)

    if not coplanar_walls:
        return list(ideal_corners)

    try:
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.ops import unary_union

        ideal_2d = to_uv(ideal_corners)
        ideal_poly = Polygon(ideal_2d)
        if not ideal_poly.is_valid:
            ideal_poly = ideal_poly.buffer(0)
        if ideal_poly.is_empty or ideal_poly.area < 0.01:
            return list(ideal_corners)

        wall_polys = []
        for wc in coplanar_walls:
            w_2d = to_uv(wc)
            wp = Polygon(w_2d)
            if not wp.is_valid:
                wp = wp.buffer(0)
            if not wp.is_empty and wp.area > 0.01:
                wall_polys.append(wp)

        if not wall_polys:
            return list(ideal_corners)

        wall_union = unary_union(wall_polys)
        result = ideal_poly.difference(wall_union)

        if result.is_empty or result.area < 0.01:
            return []

        if isinstance(result, MultiPolygon):
            result = max(result.geoms, key=lambda g: g.area)

        coords = list(result.exterior.coords)[:-1]
        return from_uv(coords)
    except Exception:
        return list(ideal_corners)  # fallback: no clipping on error
