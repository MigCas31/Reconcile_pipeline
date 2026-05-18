from __future__ import annotations

from .math_utils import angle_diff, clip_poly_by_ridge


def _per_plane_footprint(
    plane: dict,
    all_rooms: list[dict],
    buffer: float = 1.0,
    wing_polygons: list | None = None,
) -> list[tuple[float, float]] | None:
    """Compute a tight footprint from the rooms that contributed segments.

    Returns a 2-D polygon (list of (x, z) tuples) covering the source rooms
    buffered by *buffer* metres.  Unlike expanding to include entire adjacent
    rooms (which can pull in long bridging rooms), this approach grows the
    source room geometry uniformly, naturally covering nearby areas without
    inheriting distant room extents.

    When ``wing_polygons`` is provided, the result is intersected with the
    wing rectangle(s) that contain the plane's scan evidence, preventing a
    cluster's footprint from bleeding into a perpendicular wing it doesn't
    physically belong to (per `feedback_synthesis_should_follow_scan`) while
    preserving roof planes that are actually observed across multiple wings.

    Returns ``None`` if Shapely is unavailable or the result is degenerate.
    """
    room_indices: set[int] = set(plane.get("room_indices", []))
    if not room_indices or not all_rooms:
        return None

    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return None

    polys = []
    for ri in room_indices:
        if ri >= len(all_rooms):
            continue
        fp = all_rooms[ri].get("floor_polygon", [])
        if not fp or len(fp) < 3:
            continue
        ring = [(p[0], p[2]) for p in fp]
        try:
            p = Polygon(ring).buffer(buffer, join_style="mitre")
            if p.is_valid and not p.is_empty:
                polys.append(p)
        except Exception:
            continue

    if not polys:
        return None

    merged = unary_union(polys)
    if merged.is_empty:
        return None
    # Roofs span the full convex extent of contributing rooms — don't
    # let L-shaped or concave room footprints hollow out the ceiling.
    merged = merged.convex_hull

    if wing_polygons:
        wing_constraint = _wing_constraint_for_plane(plane, all_rooms, wing_polygons)
        if wing_constraint is not None:
            try:
                clipped_wing = merged.intersection(wing_constraint)
                if not clipped_wing.is_empty and clipped_wing.geom_type in (
                    "Polygon",
                    "MultiPolygon",
                ):
                    if clipped_wing.geom_type == "MultiPolygon":
                        clipped_wing = max(clipped_wing.geoms, key=lambda g: g.area)
                    if clipped_wing.area > 0:
                        merged = clipped_wing
            except Exception:
                pass

    coords = list(merged.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    return [(x, z) for x, z in coords] if len(coords) >= 3 else None


def _wing_constraint_for_plane(plane: dict, all_rooms: list[dict], wing_polygons: list):
    """Return the wing polygon constraint for this plane, or None.

    Segment midpoints are concrete physical points (slanted wall edges) that
    locate the slope evidence.  A single-wing cluster is clipped to that wing;
    a cluster observed in multiple wings gets the convex hull of those wings
    so the roof plane can continue across the building instead of collapsing
    to the wing containing the average midpoint. Falls back to the old centroid
    ownership rule only when segment evidence is unavailable.
    """
    try:
        from shapely.geometry import Point, Polygon
        from shapely.ops import unary_union
    except ImportError:
        return None

    cluster = plane.get("cl") or {}
    segs = cluster.get("segs") or []
    evidence_points: list[Point] = []
    if segs:
        for seg in segs:
            a = seg.get("a")
            b = seg.get("b")
            if not a or not b:
                continue
            evidence_points.append(
                Point(
                    (float(a[0]) + float(b[0])) / 2.0,
                    (float(a[2]) + float(b[2])) / 2.0,
                )
            )

    if evidence_points:
        matched_indices: set[int] = set()
        for point in evidence_points:
            containing_indices = [
                index
                for index, wing in enumerate(wing_polygons)
                if wing is not None
                and not wing.is_empty
                and (wing.covers(point) or wing.distance(point) <= 0.5)
            ]
            if containing_indices:
                matched_indices.update(containing_indices)
        if len(matched_indices) == 1:
            return wing_polygons[next(iter(matched_indices))]
        if len(matched_indices) > 1:
            selected = [
                wing_polygons[index]
                for index in sorted(matched_indices)
                if wing_polygons[index] is not None
                and not wing_polygons[index].is_empty
            ]
            if selected:
                return unary_union(selected).convex_hull

    centroid: Point | None = None
    if evidence_points:
        centroid = Point(
            sum(point.x for point in evidence_points) / len(evidence_points),
            sum(point.y for point in evidence_points) / len(evidence_points),
        )

    if centroid is None and all_rooms:
        seed_indices = plane.get("seed_room_indices") or plane.get("room_indices") or []
        cx_sum = 0.0
        cz_sum = 0.0
        n = 0
        for ri in seed_indices:
            if ri >= len(all_rooms):
                continue
            fp = all_rooms[ri].get("floor_polygon") or []
            if len(fp) < 3:
                continue
            cx_sum += sum(p[0] for p in fp) / len(fp)
            cz_sum += sum(p[2] for p in fp) / len(fp)
            n += 1
        if n > 0:
            centroid = Point(cx_sum / n, cz_sum / n)

    if centroid is None:
        return None

    inside: Polygon | None = None
    for wing in wing_polygons:
        if wing is None or wing.is_empty:
            continue
        if wing.contains(centroid):
            inside = wing
            break
    if inside is not None:
        return inside
    best: Polygon | None = None
    best_dist = float("inf")
    for wing in wing_polygons:
        if wing is None or wing.is_empty:
            continue
        d = wing.distance(centroid)
        if d < best_dist:
            best_dist = d
            best = wing
    return best


def _intersect_footprints(
    fp_a: list[tuple[float, float]], fp_b: list[tuple[float, float]]
) -> list[tuple[float, float]] | None:
    """Return the intersection of two 2-D footprint polygons via Shapely."""
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return None

    try:
        pa = Polygon(fp_a)
        pb = Polygon(fp_b)
        inter = pa.intersection(pb)
        if inter.is_empty or inter.geom_type not in ("Polygon", "MultiPolygon"):
            return None
        if inter.geom_type == "MultiPolygon":
            inter = max(inter.geoms, key=lambda g: g.area)
        coords = list(inter.exterior.coords)
        if coords and coords[-1] == coords[0]:
            coords = coords[:-1]
        return [(x, z) for x, z in coords] if len(coords) >= 3 else None
    except Exception:
        return None


def _expand_footprint_with_plane_rooms(
    building_footprint: list,
    plane: dict,
    all_rooms: list[dict] | None,
) -> list[tuple[float, float]]:
    """Union ``building_footprint`` with this plane's room floor polygons.

    The global footprint is built from exposed rooms only and can legitimately
    miss a lower-story room with sloped-ceiling evidence (the regression the
    2026-04-25 change was working around). Including the plane's actual room
    polygons — no buffer, no convex hull — keeps that signal while still
    constraining the plane to scanned room geometry.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return [(float(p[0]), float(p[1])) for p in building_footprint]

    polys = []
    try:
        base = Polygon([(float(p[0]), float(p[1])) for p in building_footprint])
        if base.is_valid and not base.is_empty:
            polys.append(base)
    except Exception:
        pass

    for ri in plane.get("room_indices") or []:
        if all_rooms is None or ri >= len(all_rooms):
            continue
        fp = all_rooms[ri].get("floor_polygon") or []
        if len(fp) < 3:
            continue
        try:
            poly = Polygon([(float(p[0]), float(p[2])) for p in fp])
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
        except Exception:
            continue

    if not polys:
        return [(float(p[0]), float(p[1])) for p in building_footprint]

    merged = unary_union(polys)
    if merged.is_empty:
        return [(float(p[0]), float(p[1])) for p in building_footprint]
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)

    coords = list(merged.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    return [(float(x), float(z)) for x, z in coords]


def build_initial_plane_clips(
    *,
    ceiling_planes: list,
    building_footprint: list,
    exposed_rooms: list,
    all_rooms: list[dict] | None = None,
    wing_polygons: list | None = None,
) -> list:
    plane_clipped = []

    # Slope-direction margin for flat-ceiling cross-checks.
    _SLOPE_MARGIN = 1.5

    flat_ceil_polys = [
        [(p[0], p[2]) for p in er["fp"]]
        for er in exposed_rooms
        if (er["wallTopY"] - er["wallTopMin"]) < 0.3
    ]

    for pi, plane in enumerate(ceiling_planes):
        ridge_min = plane["minRidge"]
        ridge_max = plane["maxRidge"]

        expanded = True
        while expanded:
            expanded = False
            for flat_poly in flat_ceil_polys:
                projs = [
                    (pt[0] - plane["ref"]["x"]) * plane["ridgeX"]
                    + (pt[1] - plane["ref"]["z"]) * plane["ridgeZ"]
                    for pt in flat_poly
                ]
                min_p, max_p = min(projs), max(projs)
                if max_p >= ridge_min - 1.0 and min_p <= ridge_max + 1.0:
                    # Check slope-direction overlap to prevent chaining
                    # across building sections that are offset perpendicular
                    # to the ridge (e.g. L-shaped extensions).
                    slope_projs = [
                        (pt[0] - plane["ref"]["x"]) * plane["slopeX"]
                        + (pt[1] - plane["ref"]["z"]) * plane["slopeZ"]
                        for pt in flat_poly
                    ]
                    s_min, s_max = min(slope_projs), max(slope_projs)
                    if (
                        s_max < plane["minSlope"] - _SLOPE_MARGIN
                        or s_min > plane["maxSlope"] + _SLOPE_MARGIN
                    ):
                        continue

                    if min_p < ridge_min:
                        ridge_min = min_p
                        expanded = True
                    if max_p > ridge_max:
                        ridge_max = max_p
                        expanded = True

        # Room-based expansion fallback: when no flat ceilings are available,
        # extend ridge bounds to cover exposed rooms whose centroids fall
        # within the plane's slope-direction extent.
        if not flat_ceil_polys:
            room_expanded = True
            while room_expanded:
                room_expanded = False
                for er in exposed_rooms:
                    slope_proj = (er["fcx"] - plane["ref"]["x"]) * plane["slopeX"] + (
                        er["fcz"] - plane["ref"]["z"]
                    ) * plane["slopeZ"]
                    if (
                        slope_proj < plane["minSlope"] - _SLOPE_MARGIN
                        or slope_proj > plane["maxSlope"] + _SLOPE_MARGIN
                    ):
                        continue
                    ridge_proj = (er["fcx"] - plane["ref"]["x"]) * plane["ridgeX"] + (
                        er["fcz"] - plane["ref"]["z"]
                    ) * plane["ridgeZ"]
                    if ridge_min - 2.0 <= ridge_proj <= ridge_max + 2.0:
                        if ridge_proj - 1.0 < ridge_min:
                            ridge_min = ridge_proj - 1.0
                            room_expanded = True
                        if ridge_proj + 1.0 > ridge_max:
                            ridge_max = ridge_proj + 1.0
                            room_expanded = True

        # Per-plane footprint: constrain the plane's polygon to actual
        # scanned building geometry. The constraint is the global footprint
        # expanded with this plane's room floor polygons (raw, no buffer, no
        # convex hull) — that preserves the lower-story-room-with-slope
        # signal the 2026-04-25 change cared about while still keeping the
        # plane inside the scanned envelope. The 1 m-buffered convex hull
        # (`plane_fp`) is then intersected with that envelope, so it can
        # tighten but not extend past the walls.
        envelope_fp = (
            _expand_footprint_with_plane_rooms(building_footprint, plane, all_rooms)
            if all_rooms
            else list(building_footprint)
        )
        effective_fp = envelope_fp
        if all_rooms:
            plane_fp = _per_plane_footprint(
                plane, all_rooms, wing_polygons=wing_polygons
            )
            if plane_fp:
                clipped_fp = _intersect_footprints(envelope_fp, plane_fp)
                if clipped_fp:
                    effective_fp = clipped_fp

        clipped = list(effective_fp)
        clipped = clip_poly_by_ridge(
            clipped,
            plane["ridgeX"],
            plane["ridgeZ"],
            plane["ref"]["x"],
            plane["ref"]["z"],
            ridge_min,
            True,
        )
        clipped = clip_poly_by_ridge(
            clipped,
            plane["ridgeX"],
            plane["ridgeZ"],
            plane["ref"]["x"],
            plane["ref"]["z"],
            ridge_max,
            False,
        )

        # For isolated planes (no opposing plane), also clip along slope direction
        # to prevent the ceiling from extending beyond the room's XZ bounds.
        has_opposing = False
        for pj, other in enumerate(ceiling_planes):
            if pj == pi:
                continue
            if plane["dominantStory"] != other["dominantStory"]:
                continue
            azi_diff = angle_diff(plane["cl"]["avgAzimuth"], other["cl"]["avgAzimuth"])
            if 140.0 <= azi_diff <= 220.0:
                has_opposing = True
                break

        if not has_opposing:
            # Compute contributing rooms' ridge/slope bounds so margins
            # never push the ceiling beyond the actual room footprint.
            room_ridge_min = float("-inf")
            room_ridge_max = float("inf")
            room_slope_min = float("-inf")
            room_slope_max = float("inf")
            room_indices = plane.get("room_indices", [])
            if all_rooms and room_indices:
                r_mins, r_maxs, s_mins, s_maxs = [], [], [], []
                for ri in room_indices:
                    if ri >= len(all_rooms):
                        continue
                    fp = all_rooms[ri].get("floor_polygon", [])
                    if not fp:
                        continue
                    for p in fp:
                        r_proj = (p[0] - plane["ref"]["x"]) * plane["ridgeX"] + (
                            p[2] - plane["ref"]["z"]
                        ) * plane["ridgeZ"]
                        s_proj = (p[0] - plane["ref"]["x"]) * plane["slopeX"] + (
                            p[2] - plane["ref"]["z"]
                        ) * plane["slopeZ"]
                        r_mins.append(r_proj)
                        r_maxs.append(r_proj)
                        s_mins.append(s_proj)
                        s_maxs.append(s_proj)
                if r_mins:
                    room_ridge_min = min(r_mins)
                    room_ridge_max = max(r_maxs)
                    room_slope_min = min(s_mins)
                    room_slope_max = max(s_maxs)

            slope_min = max(plane["minSlope"] - 0.5, room_slope_min)
            slope_max = min(plane["maxSlope"] + 0.5, room_slope_max)
            clipped = clip_poly_by_ridge(
                clipped,
                plane["slopeX"],
                plane["slopeZ"],
                plane["ref"]["x"],
                plane["ref"]["z"],
                slope_min,
                True,
            )
            clipped = clip_poly_by_ridge(
                clipped,
                plane["slopeX"],
                plane["slopeZ"],
                plane["ref"]["x"],
                plane["ref"]["z"],
                slope_max,
                False,
            )
            seg_ridge_min = max(plane["minRidge"] - 0.5, room_ridge_min)
            seg_ridge_max = min(plane["maxRidge"] + 0.5, room_ridge_max)
            if seg_ridge_min > ridge_min:
                clipped = clip_poly_by_ridge(
                    clipped,
                    plane["ridgeX"],
                    plane["ridgeZ"],
                    plane["ref"]["x"],
                    plane["ref"]["z"],
                    seg_ridge_min,
                    True,
                )
            if seg_ridge_max < ridge_max:
                clipped = clip_poly_by_ridge(
                    clipped,
                    plane["ridgeX"],
                    plane["ridgeZ"],
                    plane["ref"]["x"],
                    plane["ref"]["z"],
                    seg_ridge_max,
                    False,
                )

        plane_clipped.append(
            {
                "clipped": clipped,
                "ridgeMin": ridge_min,
                "ridgeMax": ridge_max,
            }
        )

    return plane_clipped
