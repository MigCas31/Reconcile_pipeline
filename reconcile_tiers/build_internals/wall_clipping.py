"""Wall-to-gable-roof clipping helpers used by `reconcile_tiers.build`.

Clips top-story walls down to the underside of the selected gable obliques,
with a per-room sanity gate that skips clipping if it would amputate a wall
cutout (door / window).
"""

from __future__ import annotations

from dataclasses import replace

from reconcile_tiers.assemble.walls_to_rooms import reclip_cutouts_to_wall
from reconcile_tiers.build_internals.constants import ROOF_WALL_CLIP_EPS_M
from reconcile_tiers.build_internals.gable_selection import (
    _clip_polygon_below_roof_plane,
    _select_gable_obliques_for_room,
)
from reconcile_tiers.build_internals.wings import (
    _compute_wings,
    _wing_polygon_for_room,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import Room, Vec3, Wall
from reconcile_tiers.roof.roof import RoofModel


def _wall_cutout_top_y(wall: Wall) -> float | None:
    """Highest cutout (door/window) top y in this wall, or None if the wall
    has no cutouts. Cutouts are scan-derived openings -- physically present
    in the wall -- so the wall must reach at least that y.
    """
    ymax: float | None = None
    for cut in wall.cutouts or ():
        for corner in cut.corners or ():
            cy = float(corner.y)
            if ymax is None or cy > ymax:
                ymax = cy
    return ymax


def _gable_clip_invalidates_cutout(wall: Wall, obliques) -> bool:
    """True if clipping this wall against `obliques` would push the wall
    top below one of its own cutouts (or collapse the wall while a cutout
    still demands it survive).

    Used as a room-level sanity gate: a habitable room with through-walls
    and windows cannot have a gable plane physically cutting through it
    unless the openings are dormers, and the dormer reconstruction is a
    separate concept tracked in `dormer_faces`.
    """
    cutout_top = _wall_cutout_top_y(wall)
    if cutout_top is None:
        return False
    corners = [[corner.x, corner.y, corner.z] for corner in wall.corners]
    for oblique in obliques:
        corners = _clip_polygon_below_roof_plane(corners, oblique.plane)
        if len(corners) < 3:
            return True
    clipped_top = max(point[1] for point in corners)
    return clipped_top < cutout_top - ROOF_WALL_CLIP_EPS_M


def _clip_wall_to_gable_roof(wall: Wall, obliques) -> Wall | None:
    corners = [[corner.x, corner.y, corner.z] for corner in wall.corners]
    all_inside = True
    for oblique in obliques:
        plane = oblique.plane
        a, b, c, d = float(plane.a), float(plane.b), float(plane.c), float(plane.d)
        if any(
            a * point[0] + b * point[1] + c * point[2] - d > ROOF_WALL_CLIP_EPS_M
            for point in corners
        ):
            all_inside = False
            break
    if all_inside:
        return wall
    for oblique in obliques:
        corners = _clip_polygon_below_roof_plane(corners, oblique.plane)
        if len(corners) < 3:
            return None
    new_cutouts = reclip_cutouts_to_wall(corners, wall.cutouts)
    return replace(
        wall,
        corners=[Vec3(x=x, y=y, z=z) for x, y, z in corners],
        cutouts=new_cutouts,
    )


def _clip_top_story_walls_to_gable_roof(
    rooms: list[Room],
    model: BuildingModel,
    roof: RoofModel,
) -> list[Room]:
    if roof.kinks.ridge_y is None:
        return rooms
    top_story = max((room.story for room in model.rooms), default=0)
    model_rooms_by_index = {room.index: room for room in model.rooms}
    wings = _compute_wings(model)
    out: list[Room] = []
    for room in rooms:
        try:
            room_index = int(room.locator_id.rsplit("::tier-room::", 1)[1])
        except (IndexError, ValueError):
            out.append(room)
            continue
        model_room = model_rooms_by_index.get(room_index)
        if model_room is None or model_room.story != top_story:
            out.append(room)
            continue
        if len(model_room.floor_polygon) < 3:
            out.append(room)
            continue

        wing_poly = _wing_polygon_for_room(model_room, wings) if wings else None
        selected = _select_gable_obliques_for_room(
            model_room, model, roof, wing_poly=wing_poly
        )
        if selected is None:
            out.append(room)
            continue
        selected_obliques, _oblique_union = selected

        # Per-room sanity gate. If clipping any wall against the gable
        # would amputate one of its own cutouts (a door, a window), the
        # gable plane is not physically over this room *unless* a dormer
        # exists to reconcile the slope with the openings. A real
        # gable+dormer room can have cutouts that the algebraic plane
        # would chop, because dormer reconstruction restores the windows
        # in dormer_faces; that evidence is the `roof.dormer_candidates`
        # for this room. Without it, the consistent reading is "this
        # room is not under the gable," and the walls stay unclipped.
        room_has_dormer = any(
            getattr(d, "room_index", None) == model_room.index
            for d in (roof.dormer_candidates or ())
        )
        if not room_has_dormer and any(
            _gable_clip_invalidates_cutout(wall, selected_obliques)
            for wall in room.walls
        ):
            out.append(room)
            continue

        clipped_walls: list[Wall] = []
        for wall in room.walls:
            clipped = _clip_wall_to_gable_roof(wall, selected_obliques)
            if clipped is not None:
                clipped_walls.append(clipped)
        out.append(replace(room, walls=clipped_walls))
    return out
