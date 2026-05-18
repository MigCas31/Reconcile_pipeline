"""Tests for reconcile.extract3d.overlaps.clip_walls_to_story_bounds.

Focus is the split-level / extension room case — a room whose own floor polygon
sits below the story aggregate but whose walls correctly meet that lower floor.
Without the per-room baseline, all walls get incorrectly raised to the story
floor and float above the room's floor polygon.
"""

from __future__ import annotations

from reconcile.extract3d.overlaps import clip_walls_to_story_bounds


def _rect_floor(x0, z0, x1, z1, y):
    return [
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ]


def _wall(wall_id, x0, z0, x1, z1, y_bottom, y_top):
    return {
        "id": wall_id,
        "corners": [
            [x0, y_bottom, z0],
            [x1, y_bottom, z1],
            [x1, y_top, z1],
            [x0, y_top, z0],
        ],
    }


def _room(floor_y, walls, *, story=0, x0=0.0, z0=0.0, x1=4.0, z1=3.0):
    return {
        "story": story,
        "floor_polygon": _rect_floor(x0, z0, x1, z1, floor_y),
        "walls_computed": walls,
    }


def test_over_extended_wall_is_clipped_to_story_floor() -> None:
    """Floor polygon sits at the story floor but walls dangle 0.40 m below it.
    Room is not self-consistent, so the clipper falls back to the story floor
    and raises the wall bottoms to y=0.
    """
    story_floor_y = 0.0
    walls = [
        _wall("w0", 0.0, 0.0, 4.0, 0.0, y_bottom=-0.40, y_top=2.40),
        _wall("w1", 4.0, 0.0, 4.0, 3.0, y_bottom=-0.40, y_top=2.40),
        _wall("w2", 4.0, 3.0, 0.0, 3.0, y_bottom=-0.40, y_top=2.40),
        _wall("w3", 0.0, 3.0, 0.0, 0.0, y_bottom=-0.40, y_top=2.40),
    ]
    rooms = [_room(story_floor_y, walls)]
    story_y_map = {0: story_floor_y}

    metrics = clip_walls_to_story_bounds(rooms, story_y_map)
    assert metrics["walls_clipped"] == 4
    for w in rooms[0]["walls_computed"]:
        ys = [c[1] for c in w["corners"]]
        assert min(ys) == story_floor_y
        assert w["wall_clipped"] is True


def test_split_level_extension_preserves_walls() -> None:
    """Room floor sits 0.34 m below story aggregate; walls match that floor.

    Mirrors the Frodesdalsvej 8 room 0:7 case. The story aggregate is driven
    by the main rooms; the extension is an outlier. With the per-room baseline,
    the extension's walls must NOT be clipped.
    """
    story_floor_y = 0.0
    ext_floor_y = -0.34
    ext_walls = [
        _wall("ew0", 0.0, 0.0, 4.0, 0.0, y_bottom=ext_floor_y, y_top=2.40),
        _wall("ew1", 4.0, 0.0, 4.0, 3.0, y_bottom=ext_floor_y, y_top=2.40),
        _wall("ew2", 4.0, 3.0, 0.0, 3.0, y_bottom=ext_floor_y, y_top=2.40),
        _wall("ew3", 0.0, 3.0, 0.0, 0.0, y_bottom=ext_floor_y, y_top=2.40),
    ]
    ext_room = _room(ext_floor_y, ext_walls)

    main_walls = [
        _wall("mw0", 10.0, 0.0, 14.0, 0.0, y_bottom=story_floor_y, y_top=2.40),
    ]
    main_room = _room(story_floor_y, main_walls, x0=10.0, x1=14.0)

    rooms = [main_room, ext_room]
    story_y_map = {0: story_floor_y}

    metrics = clip_walls_to_story_bounds(rooms, story_y_map)
    assert metrics["walls_clipped"] == 0
    for w in ext_room["walls_computed"]:
        assert "wall_clipped" not in w
        ys = [c[1] for c in w["corners"]]
        assert min(ys) == ext_floor_y


def test_errant_wall_in_otherwise_consistent_room_still_clipped() -> None:
    """One wall dangles far below the room's own floor — that wall must still
    be clipped (the room's self-consistency check uses min over all walls, so
    a single errant wall breaks the opt-out and falls back to the story floor).
    """
    story_floor_y = 0.0
    walls = [
        _wall("w0", 0.0, 0.0, 4.0, 0.0, y_bottom=0.00, y_top=2.40),
        _wall("w1", 4.0, 0.0, 4.0, 3.0, y_bottom=0.00, y_top=2.40),
        _wall("w2", 4.0, 3.0, 0.0, 3.0, y_bottom=0.00, y_top=2.40),
        _wall("w_errant", 0.0, 3.0, 0.0, 0.0, y_bottom=-0.80, y_top=2.40),
    ]
    rooms = [_room(story_floor_y, walls)]
    story_y_map = {0: story_floor_y}

    metrics = clip_walls_to_story_bounds(rooms, story_y_map)
    assert metrics["walls_clipped"] == 1
    errant = rooms[0]["walls_computed"][3]
    assert errant["wall_clipped"] is True
    assert min(c[1] for c in errant["corners"]) == story_floor_y


def test_ceiling_clip_still_works_under_per_room_opt_out() -> None:
    """Even when the room opts into the per-room floor baseline, walls that
    extend above the story ceiling must still be clipped to the ceiling.
    """
    story_floor_y_0 = 0.0
    story_floor_y_1 = 3.0
    ext_floor_y = -0.20
    ext_walls = [
        _wall("ew0", 0.0, 0.0, 4.0, 0.0, y_bottom=ext_floor_y, y_top=3.50),
        _wall("ew1", 4.0, 0.0, 4.0, 3.0, y_bottom=ext_floor_y, y_top=3.50),
        _wall("ew2", 4.0, 3.0, 0.0, 3.0, y_bottom=ext_floor_y, y_top=3.50),
        _wall("ew3", 0.0, 3.0, 0.0, 0.0, y_bottom=ext_floor_y, y_top=3.50),
    ]
    ext_room = _room(ext_floor_y, ext_walls, story=0)
    upstairs_walls = [
        _wall("uw0", 10.0, 0.0, 14.0, 0.0, y_bottom=story_floor_y_1, y_top=5.40),
    ]
    upstairs = _room(story_floor_y_1, upstairs_walls, story=1, x0=10.0, x1=14.0)

    rooms = [ext_room, upstairs]
    story_y_map = {0: story_floor_y_0, 1: story_floor_y_1}

    metrics = clip_walls_to_story_bounds(rooms, story_y_map)
    assert metrics["walls_clipped"] == 4
    for w in ext_room["walls_computed"]:
        ys = [c[1] for c in w["corners"]]
        assert min(ys) == ext_floor_y
        assert max(ys) == story_floor_y_1
        assert w["wall_clipped"] is True
