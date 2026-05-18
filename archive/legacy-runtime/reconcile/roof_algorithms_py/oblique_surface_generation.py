from __future__ import annotations

from .math_utils import clip_by_max_y, plane_y_at

# A typical Danish eave overhangs the wall top by up to ~0.5 m. Anything more
# than that is the polygon extending past the actual scanned walls and the
# plane equation projecting that (x,z) to a y far below where the eave sits.
_EAVE_OVERHANG_M = 0.5


def _clip_above_floor_y(
    poly: list[tuple[float, float, float]], floor_y: float
) -> list[tuple[float, float, float]]:
    """Sutherland-Hodgman clip: keep the portion of *poly* at or above *floor_y*."""
    out: list[tuple[float, float, float]] = []
    n = len(poly)
    for k in range(n):
        curr, nxt = poly[k], poly[(k + 1) % n]
        c_in, n_in = curr[1] >= floor_y, nxt[1] >= floor_y
        if c_in:
            out.append(curr)
        if c_in != n_in:
            t = (floor_y - curr[1]) / (nxt[1] - curr[1])
            out.append(
                (
                    curr[0] + t * (nxt[0] - curr[0]),
                    floor_y,
                    curr[2] + t * (nxt[2] - curr[2]),
                )
            )
    return out


def _wall_top_y_for_plane(plane: dict, all_rooms: list[dict] | None) -> float | None:
    """Eave-level wall-top y across the plane's contributing rooms.

    For an attic room the slanted-ceiling walls span from floor up to the
    ridge, so per-wall ``max(corner.y)`` ranges from the eave height
    (long-side walls) to the ridge height (gable-end walls). The eave is
    the *minimum* across per-wall tops — the lowest place a wall's roofline
    reaches. Anchoring the polygon at that height keeps the long-side eave
    on the wall and lets the gable triangles extend upward to the ridge.
    """
    if not all_rooms:
        return None
    room_indices = plane.get("room_indices") or plane.get("seed_room_indices") or []
    per_wall_tops: list[float] = []
    for ri in room_indices:
        if ri >= len(all_rooms):
            continue
        room = all_rooms[ri]
        for wall in room.get("walls_computed") or []:
            corner_ys: list[float] = []
            for c in wall.get("corners") or []:
                try:
                    corner_ys.append(float(c[1]))
                except (TypeError, ValueError, IndexError):
                    continue
            if corner_ys:
                per_wall_tops.append(max(corner_ys))
    return min(per_wall_tops) if per_wall_tops else None


def build_oblique_roof_surfaces(
    ceiling_planes: list,
    plane_clipped: list,
    plane_max_y: list,
    story_floor_y: dict,
    *,
    all_rooms: list[dict] | None = None,
) -> list[dict]:
    """Build 3D oblique roof surfaces from ceiling-plane clipping results.

    Uses the ceiling pipeline's smart 2D clipping (footprint narrowing,
    L-junction expansion, opposing plane cuts, height caps) and projects
    the clipped polygons onto the 3D plane to produce roof surface dicts
    compatible with dormer detection.
    """
    surfaces: list[dict] = []

    for pi, plane in enumerate(ceiling_planes):
        clipped_2d = plane_clipped[pi]["clipped"]
        if len(clipped_2d) < 3:
            continue

        # Project 2D (x, z) to 3D (x, y, z) via the plane equation
        corners = [(p[0], plane_y_at(plane, p[0], p[1]), p[1]) for p in clipped_2d]

        # Clip below floor level
        floor_y = story_floor_y.get(plane["dominantStory"])
        if floor_y is not None:
            corners = _clip_above_floor_y(corners, floor_y)
            if len(corners) < 3:
                continue

        # Clip by height cap (floor above on multi-story buildings)
        if plane_max_y[pi] != float("inf"):
            corners = clip_by_max_y(corners, plane_max_y[pi])
            if len(corners) < 3:
                continue

        surfaces.append(
            {
                "cluster": plane["cl"],
                "dominant_story": plane["dominantStory"],
                "corners": corners,
                "center": {
                    "x": plane["ref"]["x"],
                    "z": plane["ref"]["z"],
                    "y": plane["ref"]["y"],
                },
                "ridge": {
                    "x": plane["ridgeX"],
                    "z": plane["ridgeZ"],
                    "min": plane_clipped[pi]["ridgeMin"],
                    "max": plane_clipped[pi]["ridgeMax"],
                },
            }
        )

    return surfaces
