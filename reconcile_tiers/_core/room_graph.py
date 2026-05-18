"""Room-adjacency graph: rooms as nodes, doors and shared walls as edges.

A wing is, definitionally, a region whose only connections to the rest of
the building are 1-2 doorways and a small number of shared walls -- i.e.,
an articulation vertex/edge in the room graph. Wing detection therefore
becomes a graph-cut problem on this structure rather than a pure
2D-geometry problem on the footprint polygon.

Edges come from two existing pieces of infrastructure:

  - `extract/wall_pairs.find_facing_pairs(rooms)` discovers geometrically
    facing wall pairs across rooms. Each `SlabQuad` carries the two
    rooms' indices; aggregating slabs by room-pair gives shared-wall
    edges weighted by total overlap length.

  - For each door (`ExtractedElement` with `parent_wall_id`), the door
    connects its owning room to whichever room has a wall facing the
    door's parent wall. We resolve "which room?" by looking up the
    parent wall in the slab-pair index built above.

Graph construction is per-story by default (wings are lateral, not
vertical), but the `RoomAdjacency` struct also exposes a flat view for
callers that want it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from reconcile_tiers.extract.building import ExtractedRoom
from reconcile_tiers.extract.wall_pairs import SlabQuad, find_facing_pairs


def _edge_key(a: int, b: int) -> frozenset[int]:
    return frozenset({a, b})


@dataclass(frozen=True, slots=True)
class RoomAdjacency:
    """Room-connectivity graph.

    Edges are unordered pairs encoded as `frozenset[int]` of room indices.
    A self-edge (single-element frozenset) never appears.
    """

    door_edges: frozenset[frozenset[int]]
    shared_wall_edges: dict[frozenset[int], float]
    rooms_by_story: dict[int, frozenset[int]] = field(default_factory=dict)

    def neighbours(self, room_index: int) -> set[int]:
        """All rooms connected to `room_index` by either a door or a shared wall."""
        out: set[int] = set()
        for edge in self.door_edges:
            if room_index in edge:
                out.update(edge - {room_index})
        for edge in self.shared_wall_edges:
            if room_index in edge:
                out.update(edge - {room_index})
        return out

    def edges(self) -> set[frozenset[int]]:
        """Union of door and shared-wall edges (unweighted)."""
        return set(self.door_edges) | set(self.shared_wall_edges)


def _index_walls_to_facing_rooms(
    slabs: Sequence[SlabQuad],
) -> dict[tuple[int, int], set[int]]:
    """
    Map (room_index, wall_index) -> set of room indices on the other side of any facing
    pair.
    """
    out: dict[tuple[int, int], set[int]] = {}
    for slab in slabs:
        a_key = (slab.wall_a.room_index, slab.wall_a.wall_index)
        b_key = (slab.wall_b.room_index, slab.wall_b.wall_index)
        if (
            slab.wall_a.wall_index >= 0
        ):  # negative index = synthesised ceiling-edge run, skip
            out.setdefault(a_key, set()).add(slab.wall_b.room_index)
        if slab.wall_b.wall_index >= 0:
            out.setdefault(b_key, set()).add(slab.wall_a.room_index)
    return out


def _shared_wall_edges_from_slabs(
    slabs: Sequence[SlabQuad],
) -> dict[frozenset[int], float]:
    """Aggregate slab overlap lengths per room-pair edge.

    Two facing wall runs produce a slab whose tangent extent is the
    overlap length on `wall_a`'s line (`abs(t_along(a1) - t_along(a0))`,
    which equals the overlap parameter range from `find_facing_pairs`).
    Summing this across all slabs between the same two rooms gives the
    total shared-wall extent -- a natural edge weight.
    """
    edges: dict[frozenset[int], float] = {}
    for slab in slabs:
        a, b = slab.wall_a.room_index, slab.wall_b.room_index
        if a == b:
            continue
        ax, az = slab.a0
        bx, bz = slab.a1
        overlap = ((bx - ax) ** 2 + (bz - az) ** 2) ** 0.5
        if overlap <= 0.0:
            continue
        key = _edge_key(a, b)
        edges[key] = edges.get(key, 0.0) + float(overlap)
    return edges


def _resolve_door_edges(
    rooms: Sequence[ExtractedRoom],
    wall_to_facing_rooms: dict[tuple[int, int], set[int]],
) -> set[frozenset[int]]:
    """For each door, find the room across the door's parent wall."""
    edges: set[frozenset[int]] = set()
    for room_idx, room in enumerate(rooms):
        # Build wall_id -> wall_index map for this room (to locate the door's parent
        # wall).
        wall_idx_by_id: dict[str, int] = {}
        for widx, wall in enumerate(room.walls_computed):
            if wall.id and wall.id not in wall_idx_by_id:
                wall_idx_by_id[wall.id] = widx
        for door in room.doors:
            parent_id = door.parent_wall_id
            if not parent_id:
                continue
            wall_idx = wall_idx_by_id.get(parent_id)
            if wall_idx is None:
                continue
            partners = wall_to_facing_rooms.get((room_idx, wall_idx), set())
            for partner in partners:
                if partner == room_idx:
                    continue
                edges.add(_edge_key(room_idx, partner))
    return edges


def build_room_graph(rooms: Sequence[ExtractedRoom]) -> RoomAdjacency:
    """Construct the per-story room-adjacency graph.

    Reuses `extract/wall_pairs.find_facing_pairs` for the geometric
    pairing and the door-pair resolution; this module's only added value
    is aggregation by room-pair edge plus door-to-room indexing.
    """
    if not rooms:
        return RoomAdjacency(
            door_edges=frozenset(),
            shared_wall_edges={},
            rooms_by_story={},
        )

    slabs = list(find_facing_pairs(rooms))
    wall_to_facing = _index_walls_to_facing_rooms(slabs)
    shared = _shared_wall_edges_from_slabs(slabs)
    door_edges = frozenset(_resolve_door_edges(rooms, wall_to_facing))

    rooms_by_story: dict[int, set[int]] = {}
    for idx, room in enumerate(rooms):
        rooms_by_story.setdefault(int(room.story), set()).add(idx)

    return RoomAdjacency(
        door_edges=door_edges,
        shared_wall_edges=shared,
        rooms_by_story={
            story: frozenset(members) for story, members in rooms_by_story.items()
        },
    )
