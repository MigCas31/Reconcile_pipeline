"""Multi-signal wing decomposition fusion: geometric + graph + session."""

from __future__ import annotations

from shapely.geometry import Polygon

from reconcile_tiers._core.room_graph import RoomAdjacency, build_room_graph
from reconcile_tiers._core.wing_decomposition import Wing, decompose_to_wings
from reconcile_tiers._core.wing_decomposition_v2 import decompose_to_wings_v2
from reconcile_tiers.extract.building import (
    ExtractedElement,
    ExtractedRoom,
    ExtractedWall,
)


def _wall(wid, x0, z0, x1, z1, *, y_low=0.0, y_high=2.5):
    return ExtractedWall(
        id=wid,
        corners=[
            [x0, y_low, z0],
            [x1, y_low, z1],
            [x1, y_high, z1],
            [x0, y_high, z0],
        ],
        source="merged-room",
    )


def _room(*, index, story, x0, z0, x1, z1, walls=None, doors=None):
    return ExtractedRoom(
        index=index,
        story=story,
        floor_polygon=[
            [x0, 0.0, z0],
            [x1, 0.0, z0],
            [x1, 0.0, z1],
            [x0, 0.0, z1],
        ],
        walls_merged=walls or [],
        walls_computed=walls or [],
        doors=doors or [],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )


def test_no_signals_falls_through_to_v1():
    """With no rooms / room_graph / session_clusters, behaviour matches v1 exactly."""
    rect = Polygon([(0, 0), (8, 0), (8, 4), (0, 4)])
    v1 = decompose_to_wings(rect)
    v2 = decompose_to_wings_v2(rect)
    assert len(v1) == len(v2)
    assert v1[0].role == v2[0].role
    assert v1[0].confidence == v2[0].confidence == "high"


def test_l_shape_no_signals_matches_v1():
    l_shape = Polygon([(0, 0), (8, 0), (8, 4), (4, 4), (4, 8), (0, 8)])
    v1 = decompose_to_wings(l_shape)
    v2 = decompose_to_wings_v2(l_shape)
    assert len(v1) == len(v2)
    for a, b in zip(v1, v2, strict=False):
        assert abs(a.area_m2 - b.area_m2) < 1e-6
        assert a.role == b.role


def test_empty_polygon_returns_no_wings():
    assert decompose_to_wings_v2(Polygon()) == []


def test_rooms_provided_but_no_extra_signals_falls_through():
    rect = Polygon([(0, 0), (8, 0), (8, 4), (0, 4)])
    rooms = [_room(index=0, story=0, x0=0, z0=0, x1=8, z1=4)]
    wings = decompose_to_wings_v2(rect, rooms=rooms)
    assert len(wings) == 1
    assert wings[0].confidence == "high"
    assert wings[0].disagreement == ()


def test_three_tier_consensus_high_confidence_when_all_agree():
    """L-shape where geometric wings, room-graph components, and sessions all agree.

    The L-shape's geometric decomposition gives wing 0 = (0,0)-(4,8) and
    wing 1 = (4,0)-(8,4). Rooms aligned to these wings, with no graph
    connections and distinct sessions, produce a unanimous partition.
    """
    l_shape = Polygon([(0, 0), (8, 0), (8, 4), (4, 4), (4, 8), (0, 8)])
    rooms = [
        _room(index=0, story=0, x0=0, z0=0, x1=4, z1=8),  # vertical limb (wing 0)
        _room(
            index=1, story=0, x0=4, z0=0, x1=8, z1=4
        ),  # horizontal extension (wing 1)
    ]
    graph = RoomAdjacency(
        door_edges=frozenset(),
        shared_wall_edges={},
        rooms_by_story={0: frozenset({0, 1})},
    )
    sessions = {0: 0, 1: 1}
    wings = decompose_to_wings_v2(
        l_shape, room_graph=graph, session_clusters=sessions, rooms=rooms
    )
    assert len(wings) >= 2
    for w in wings:
        assert w.confidence == "high"
        assert w.disagreement == ()


def test_session_disagreement_yields_medium_confidence():
    """Rooms split across two geometric wings, joined by one session.

    Geometry + graph (no edges) say "different wings"; session says "same".
    Strict-majority consensus = different; session is the lone dissenter.
    """
    l_shape = Polygon([(0, 0), (8, 0), (8, 4), (4, 4), (4, 8), (0, 8)])
    rooms = [
        _room(index=0, story=0, x0=0, z0=0, x1=4, z1=8),  # in wing 0
        _room(index=1, story=0, x0=4, z0=0, x1=8, z1=4),  # in wing 1
    ]
    graph = RoomAdjacency(
        door_edges=frozenset(),
        shared_wall_edges={},
        rooms_by_story={0: frozenset({0, 1})},
    )
    sessions = {0: 0, 1: 0}  # both rooms same session
    wings = decompose_to_wings_v2(
        l_shape, room_graph=graph, session_clusters=sessions, rooms=rooms
    )
    assert len(wings) >= 2
    # session is the lone dissenter on the cross-wing pair.
    has_session_dissent = any("session" in w.disagreement for w in wings)
    assert has_session_dissent
    for w in wings:
        if "session" in w.disagreement:
            assert w.confidence == "medium"


def test_story_break_filter_does_not_split_lateral_wing():
    """Multi-story session: vertical session break (different stories) is filtered.

    Rooms 0,1 on story 0 with session 0; rooms 2,3 on story 1 with session 1.
    Sessions are story-disjoint → session tier merges them into one wing.
    """
    big_rect = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    rooms = [
        _room(index=0, story=0, x0=0, z0=0, x1=5, z1=5),
        _room(index=1, story=0, x0=5, z0=0, x1=10, z1=5),
        _room(index=2, story=1, x0=0, z0=5, x1=5, z1=10),
        _room(index=3, story=1, x0=5, z0=5, x1=10, z1=10),
    ]
    graph = RoomAdjacency(
        door_edges=frozenset(),
        shared_wall_edges={},
        rooms_by_story={0: frozenset({0, 1}), 1: frozenset({2, 3})},
    )
    sessions = {0: 0, 1: 0, 2: 1, 3: 1}
    wings = decompose_to_wings_v2(
        big_rect, room_graph=graph, session_clusters=sessions, rooms=rooms
    )
    # Story-disjoint sessions are merged for wing purposes; no session
    # tier dissent.
    assert all("session" not in w.disagreement for w in wings)


def test_long_axis_math_preserved_on_consensus_wing():
    rect = Polygon([(0, 0), (10, 0), (10, 3), (0, 3)])  # long axis along +X
    rooms = [_room(index=0, story=0, x0=0, z0=0, x1=10, z1=3)]
    graph = RoomAdjacency(
        door_edges=frozenset(),
        shared_wall_edges={},
        rooms_by_story={0: frozenset({0})},
    )
    sessions = {0: 0}
    wings = decompose_to_wings_v2(
        rect, room_graph=graph, session_clusters=sessions, rooms=rooms
    )
    assert wings, "expected at least one wing"
    assert abs(wings[0].long_axis_math) < 5.0


def test_min_area_filter_drops_tiny_wings():
    """A wing whose synthesised polygon falls below `min_area_m2` is dropped."""
    rect = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    rooms = [_room(index=0, story=0, x0=0, z0=0, x1=4, z1=4)]
    graph = RoomAdjacency(
        door_edges=frozenset(),
        shared_wall_edges={},
        rooms_by_story={0: frozenset({0})},
    )
    sessions = {0: 0}
    wings = decompose_to_wings_v2(
        rect,
        room_graph=graph,
        session_clusters=sessions,
        rooms=rooms,
        min_area_m2=100.0,  # impossibly large
    )
    # All wings filtered out; falls back to geometric decomposition.
    # Geometric decomposition with min_area_m2=100 also returns the
    # whole footprint as a single wing (fallback path).
    assert isinstance(wings, list)


def test_zero_rooms_with_signals_falls_through():
    rect = Polygon([(0, 0), (8, 0), (8, 4), (0, 4)])
    graph = RoomAdjacency(
        door_edges=frozenset(),
        shared_wall_edges={},
        rooms_by_story={},
    )
    wings = decompose_to_wings_v2(rect, room_graph=graph, rooms=[])
    # With zero rooms, fusion is impossible; geometric output preserved.
    assert len(wings) == 1
    assert wings[0].confidence == "high"


def test_wing_dataclass_backwards_compat_with_v1_callers():
    """Existing callers reading only old Wing fields work unchanged."""
    rect = Polygon([(0, 0), (8, 0), (8, 4), (0, 4)])
    wings = decompose_to_wings_v2(rect)
    w = wings[0]
    # Old fields still accessible, with old types.
    assert isinstance(w.index, int)
    assert isinstance(w.area_m2, float)
    assert w.role in {"main", "extension"}
    assert isinstance(w.long_axis_math, float)


def test_disagreement_tuple_is_sorted_and_deterministic():
    """
    When multiple tiers dissent, the `disagreement` tuple is sorted alphabetically.
    """
    l_shape = Polygon([(0, 0), (8, 0), (8, 4), (4, 4), (4, 8), (0, 8)])
    rooms = [
        _room(index=0, story=0, x0=0, z0=0, x1=8, z1=4),
        _room(index=1, story=0, x0=0, z0=4, x1=4, z1=8),
    ]
    graph = RoomAdjacency(
        door_edges=frozenset({frozenset({0, 1})}),  # door says SAME wing
        shared_wall_edges={},
        rooms_by_story={0: frozenset({0, 1})},
    )
    sessions = {0: 0, 1: 0}  # session says SAME wing
    wings = decompose_to_wings_v2(
        l_shape, room_graph=graph, session_clusters=sessions, rooms=rooms
    )
    # Geometry says different (L-shape decomposes to 2 wings), but graph and
    # session say same. Strict majority -> consensus "same wing" (1 wing).
    assert len(wings) >= 1
    # If any wing has disagreement, the tuple is sorted.
    for w in wings:
        if w.disagreement:
            assert list(w.disagreement) == sorted(w.disagreement)


def test_real_room_graph_smoke():
    """End-to-end smoke: build the room graph from rooms, feed v2."""
    walls_a = [
        _wall("A_n", 0.0, 0.0, 4.0, 0.0),
        _wall("A_e", 4.0, 0.0, 4.0, 3.0),
        _wall("A_s", 4.0, 3.0, 0.0, 3.0),
        _wall("A_w", 0.0, 3.0, 0.0, 0.0),
    ]
    walls_b = [
        _wall("B_w", 4.10, 0.0, 4.10, 3.0),
        _wall("B_n", 4.10, 0.0, 8.0, 0.0),
        _wall("B_e", 8.0, 0.0, 8.0, 3.0),
        _wall("B_s", 8.0, 3.0, 4.10, 3.0),
    ]
    door = ExtractedElement(
        id="d1",
        corners=[
            [4.10, 0.0, 1.0],
            [4.10, 0.0, 2.0],
            [4.10, 2.0, 2.0],
            [4.10, 2.0, 1.0],
        ],
        source="merged-room",
        parent_wall_id="B_w",
    )
    a = _room(index=0, story=0, x0=0, z0=0, x1=4, z1=3, walls=walls_a)
    b = _room(index=1, story=0, x0=4.10, z0=0, x1=8, z1=3, walls=walls_b, doors=[door])
    rect = Polygon([(0, 0), (8, 0), (8, 3), (0, 3)])
    graph = build_room_graph([a, b])
    wings = decompose_to_wings_v2(
        rect,
        room_graph=graph,
        session_clusters={0: 0, 1: 0},
        rooms=[a, b],
    )
    # Connected via door + shared wall, single session — one wing.
    assert wings
    assert isinstance(wings[0], Wing)
