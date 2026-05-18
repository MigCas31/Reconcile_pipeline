"""Multi-signal wing decomposition: geometry + graph + ARKit session.

Three independent partition modalities are computed and fused into a single
consensus wing partition. Each modality answers "which rooms belong to the
same wing?" using different evidence, and where they agree the consensus
is high-confidence; where they disagree the disagreement is recorded on
the resulting `Wing`s rather than silently resolved.

  - **Geometric tier** (always available): the v1 grid-decomposition of
    the synthesised footprint polygon. Each room is assigned to the wing
    polygon with which its floor polygon overlaps most.

  - **Graph tier** (when `room_graph` and `rooms` provided): articulation
    analysis on the door-graph. An articulation room (a corridor between
    two wings) is removed; the remaining connected components are
    candidate wings; the articulation room is reassigned to whichever
    component it is most strongly connected to.

  - **Session tier** (when `session_clusters` and `rooms` provided):
    rooms grouped by ARKit `referenceOriginTransform` yaw cluster. Cross-
    story session breaks are filtered out — a stair-induced session
    restart is a vertical break, not a wing boundary.

When two tiers are unavailable the third is returned as-is. When only the
geometric tier is available the function reduces to v1's
`decompose_to_wings`. The pure-function signature `decompose_to_wings_v2(
footprint, **signals)` is preserved so callers can plug in tiers
incrementally.
"""

from __future__ import annotations

from collections.abc import Sequence

from shapely.geometry import Polygon

from reconcile_tiers._core.room_graph import RoomAdjacency
from reconcile_tiers._core.wing_decomposition import (
    _MIN_RECT_AREA_M2,
    Wing,
    _long_axis_math_deg,
    decompose_to_wings,
    wing_polygon_for_room,
)

# Buffer/shrink used when synthesising a wing polygon from a room cluster.
# Mirrors the constants in roof/footprint.py:build_building_footprint so
# the resulting wing geometry has the same scan-noise tolerance as the
# building footprint we already trust.
_WING_FROM_ROOMS_BUFFER_M = 0.30
_WING_FROM_ROOMS_SHRINK_M = 0.30


def _room_xz_polygon(room) -> Polygon | None:
    if len(room.floor_polygon) < 3:
        return None
    coords = [(float(p[0]), float(p[2])) for p in room.floor_polygon]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = Polygon(coords)
    except Exception:
        return None
    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
        except Exception:
            return None
    if poly.is_empty:
        return None
    return poly


def _geometric_partition(
    rooms: Sequence,
    geom_wings: list[Wing],
) -> dict[int, int]:
    """Map room_index → geometric wing index via `wing_polygon_for_room`."""
    if not geom_wings:
        return {idx: 0 for idx in range(len(rooms))}
    if len(geom_wings) == 1:
        return {idx: 0 for idx in range(len(rooms))}
    out: dict[int, int] = {}
    for idx, room in enumerate(rooms):
        wing_poly = wing_polygon_for_room(room, geom_wings)
        label = 0
        if wing_poly is not None:
            for w in geom_wings:
                if w.polygon is wing_poly:
                    label = w.index
                    break
        out[idx] = label
    return out


def _articulation_points(
    nodes: list[int],
    adjacency: dict[int, set[int]],
) -> set[int]:
    """Tarjan-style articulation-point detection on an undirected graph.

    Returns the set of nodes whose removal would disconnect the graph (or
    increase the number of connected components). Linear in |V| + |E|.
    """
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    articulations: set[int] = set()
    counter = [0]

    def dfs(u: int) -> None:
        # Iterative DFS to avoid recursion-limit issues on deep graphs.
        stack: list[tuple[int, iter]] = [(u, iter(adjacency.get(u, ())))]
        disc[u] = low[u] = counter[0]
        counter[0] += 1
        parent[u] = None
        children_of_root = 0
        while stack:
            node, it = stack[-1]
            try:
                v = next(it)
            except StopIteration:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[node])
                    if parent[p] is not None and low[node] >= disc[p]:
                        articulations.add(p)
                continue
            if v not in disc:
                parent[v] = node
                disc[v] = low[v] = counter[0]
                counter[0] += 1
                stack.append((v, iter(adjacency.get(v, ()))))
                if node == u:
                    children_of_root += 1
            elif v != parent.get(node):
                low[node] = min(low[node], disc[v])
        if children_of_root >= 2:
            articulations.add(u)

    for node in nodes:
        if node not in disc:
            dfs(node)
    return articulations


def _graph_partition(
    rooms: Sequence,
    room_graph: RoomAdjacency,
) -> dict[int, int]:
    """Door-graph articulation partition: rooms that aren't articulation
    points fall into connected components when articulations are removed.
    Articulation rooms are reassigned to whichever component they are
    most strongly connected to (via shared-wall length, then door count).
    """
    nodes = list(range(len(rooms)))
    if not nodes:
        return {}

    # Build adjacency from door + shared-wall edges (treat both as
    # equivalent for connectivity; weights are used only for articulation
    # tie-breaking below).
    adjacency: dict[int, set[int]] = {idx: set() for idx in nodes}
    for edge in room_graph.door_edges:
        if len(edge) != 2:
            continue
        a, b = tuple(edge)
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    for edge in room_graph.shared_wall_edges:
        if len(edge) != 2:
            continue
        a, b = tuple(edge)
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)

    arts = _articulation_points(nodes, adjacency)

    # Connected components of (graph minus articulations).
    label: dict[int, int] = {}
    next_label = 0
    for start in nodes:
        if start in arts or start in label:
            continue
        # BFS from start, ignoring articulation neighbours as bridges
        # (we still traverse non-articulation neighbours).
        queue = [start]
        label[start] = next_label
        while queue:
            u = queue.pop()
            for v in adjacency.get(u, ()):
                if v in arts or v in label:
                    continue
                label[v] = next_label
                queue.append(v)
        next_label += 1

    if next_label == 0:
        # Every room is an articulation (e.g., a single-room or trivial
        # graph). Fall back to one wing.
        return {idx: 0 for idx in nodes}

    # Assign articulation rooms to whichever component-neighbour they
    # share the most weight with.
    for art in arts:
        best_label: int | None = None
        best_weight = -1.0
        for neighbour in adjacency.get(art, ()):
            if neighbour in arts or neighbour not in label:
                continue
            edge = frozenset({art, neighbour})
            w = float(room_graph.shared_wall_edges.get(edge, 0.0))
            if edge in room_graph.door_edges:
                w += 1.0  # presence-of-door bonus
            if w > best_weight:
                best_weight = w
                best_label = label[neighbour]
        if best_label is None:
            # Articulation with no non-articulation neighbour — its own
            # component.
            best_label = next_label
            next_label += 1
        label[art] = best_label

    # Any room not yet labelled (isolates) gets its own component.
    for idx in nodes:
        if idx not in label:
            label[idx] = next_label
            next_label += 1
    return label


def _session_partition(
    rooms: Sequence,
    session_clusters: dict[int, int],
    room_graph: RoomAdjacency | None,
) -> dict[int, int]:
    """Sessions grouped into wings via cross-session adjacency, with
    cross-story session breaks filtered out.

    A session whose rooms are entirely on a different story than another
    session's rooms is treated as the same wing (session restart was
    forced by the staircase, not by an architectural break). Sessions
    that share at least one story and have any cross-session adjacency
    edge in `room_graph` are merged into the same wing.
    """
    nodes = list(range(len(rooms)))
    if not nodes:
        return {}
    # Map session_id → set of rooms, and session_id → set of stories.
    session_rooms: dict[int, set[int]] = {}
    session_stories: dict[int, set[int]] = {}
    for idx in nodes:
        sid = session_clusters.get(idx)
        if sid is None:
            continue
        session_rooms.setdefault(sid, set()).add(idx)
        session_stories.setdefault(sid, set()).add(int(rooms[idx].story))

    if not session_rooms:
        return {idx: 0 for idx in nodes}

    sessions = sorted(session_rooms.keys())
    # Union-find over sessions: merge whenever (a) any room-graph edge
    # crosses sessions while at least one story is shared (lateral
    # adjacency on the same level), or (b) sessions are story-disjoint
    # (vertical break, not architectural).
    parent: dict[int, int] = {sid: sid for sid in sessions}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Story-disjoint merge.
    for i, sa in enumerate(sessions):
        for sb in sessions[i + 1 :]:
            if not (session_stories[sa] & session_stories[sb]):
                union(sa, sb)

    # Adjacency-driven merge (only when at least one story is shared).
    if room_graph is not None:
        for edge in room_graph.door_edges | set(room_graph.shared_wall_edges.keys()):
            if len(edge) != 2:
                continue
            a, b = tuple(edge)
            sa = session_clusters.get(a)
            sb = session_clusters.get(b)
            if sa is None or sb is None or sa == sb:
                continue
            if session_stories[sa] & session_stories[sb]:
                union(sa, sb)

    # Assign labels to merged session-groups deterministically (by min sid).
    group_of: dict[int, int] = {}
    label_of_group: dict[int, int] = {}
    next_label = 0
    for sid in sessions:
        root = find(sid)
        group_of[sid] = root
    # Order groups by smallest member so labels are stable.
    sorted_groups = sorted({root for root in group_of.values()})
    for root in sorted_groups:
        label_of_group[root] = next_label
        next_label += 1

    out: dict[int, int] = {}
    for idx in nodes:
        sid = session_clusters.get(idx)
        if sid is None:
            out[idx] = next_label
            next_label += 1
        else:
            out[idx] = label_of_group[group_of[sid]]
    return out


def _same_wing_relation(
    partition: dict[int, int], rooms_count: int
) -> dict[frozenset[int], bool]:
    """For each unordered pair (i, j), record whether `partition[i] == partition[j]`."""
    rel: dict[frozenset[int], bool] = {}
    for i in range(rooms_count):
        for j in range(i + 1, rooms_count):
            rel[frozenset({i, j})] = partition.get(i) == partition.get(j)
    return rel


def _consensus_partition(
    rooms_count: int,
    partitions: list[tuple[str, dict[int, int]]],
) -> tuple[dict[int, int], dict[int, set[str]]]:
    """Strict-majority "same wing" voting.

    For each unordered room-pair (i, j), count tiers that say `same_wing`.
    A pair is consensus-same when the count is strictly greater than half
    the number of available tiers (so 2-of-3, both-of-2, or 1-of-1).
    Connected components on the consensus-same relation give the
    final per-room labelling. Per-room disagreement set lists the tiers
    whose pairwise vote dissented from consensus on majority of pairs
    involving that room.
    """
    n_tiers = len(partitions)
    if n_tiers == 0:
        return {idx: 0 for idx in range(rooms_count)}, {
            idx: set() for idx in range(rooms_count)
        }

    # Pairwise same-wing votes per tier.
    rels = [(name, _same_wing_relation(part, rooms_count)) for name, part in partitions]

    threshold = n_tiers // 2 + 1  # strict majority
    consensus_same: dict[frozenset[int], bool] = {}
    pair_disagreement: dict[frozenset[int], set[str]] = {}
    for i in range(rooms_count):
        for j in range(i + 1, rooms_count):
            key = frozenset({i, j})
            votes_same = sum(1 for _, rel in rels if rel.get(key, False))
            consensus = votes_same >= threshold
            consensus_same[key] = consensus
            dissenters: set[str] = set()
            for name, rel in rels:
                if rel.get(key, False) != consensus:
                    dissenters.add(name)
            if dissenters:
                pair_disagreement[key] = dissenters

    # Connected components on the consensus-same relation.
    parent = list(range(rooms_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair, same in consensus_same.items():
        if same:
            a, b = tuple(pair)
            union(a, b)

    root_to_label: dict[int, int] = {}
    next_label = 0
    out_partition: dict[int, int] = {}
    for idx in range(rooms_count):
        root = find(idx)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        out_partition[idx] = root_to_label[root]

    # Aggregate per-room dissent counts.
    per_room_dissent: dict[int, set[str]] = {idx: set() for idx in range(rooms_count)}
    per_room_pair_count: dict[int, int] = {idx: 0 for idx in range(rooms_count)}
    per_room_dissent_count: dict[int, dict[str, int]] = {
        idx: {} for idx in range(rooms_count)
    }
    for pair, dissenters in pair_disagreement.items():
        a, b = tuple(pair)
        per_room_pair_count[a] += 1
        per_room_pair_count[b] += 1
        for d in dissenters:
            per_room_dissent_count[a][d] = per_room_dissent_count[a].get(d, 0) + 1
            per_room_dissent_count[b][d] = per_room_dissent_count[b].get(d, 0) + 1

    for idx in range(rooms_count):
        for tier, count in per_room_dissent_count[idx].items():
            # Tier consistently dissented on this room: > half of its pairs.
            if count * 2 > max(per_room_pair_count[idx], 1):
                per_room_dissent[idx].add(tier)
    return out_partition, per_room_dissent


def _wing_polygon_from_rooms(
    room_indices: Sequence[int],
    rooms: Sequence,
    geom_wings: list[Wing],
) -> Polygon | None:
    """Synthesise a wing polygon from a cluster of rooms.

    Mirrors `roof/footprint.build_building_footprint`: buffer floor
    polygons, union, shrink. Falls back to the matching geometric-wing
    polygon when the buffered union is empty.
    """
    polys: list[Polygon] = []
    for ridx in room_indices:
        if 0 <= ridx < len(rooms):
            poly = _room_xz_polygon(rooms[ridx])
            if poly is not None:
                polys.append(poly)
    if not polys:
        return None
    buffered = [p.buffer(_WING_FROM_ROOMS_BUFFER_M, join_style=2) for p in polys]
    union = buffered[0]
    for p in buffered[1:]:
        union = union.union(p)
    shrunk = union.buffer(-_WING_FROM_ROOMS_SHRINK_M, join_style=2)
    if shrunk.is_empty:
        # Fallback: pick the geom_wing with greatest overlap with the union.
        best = None
        best_area = 0.0
        for w in geom_wings:
            inter = w.polygon.intersection(union)
            if inter.area > best_area:
                best_area = inter.area
                best = w.polygon
        return best
    if shrunk.geom_type == "MultiPolygon":
        shrunk = max(shrunk.geoms, key=lambda g: g.area)
    if isinstance(shrunk, Polygon) and shrunk.area > 0:
        return shrunk
    return None


def _wing_role(index: int, rank_by_area: list[int]) -> str:
    return "main" if index == rank_by_area[0] else "extension"


def _wings_from_consensus(
    consensus: dict[int, int],
    per_room_dissent: dict[int, set[str]],
    rooms: Sequence,
    geom_wings: list[Wing],
    *,
    n_tiers: int,
    min_area_m2: float,
) -> list[Wing]:
    """Materialise consensus into a list of `Wing`s with confidence + disagreement."""
    if not consensus:
        return []
    groups: dict[int, list[int]] = {}
    for room_idx, label in consensus.items():
        groups.setdefault(label, []).append(room_idx)

    candidates: list[tuple[Polygon, list[int], set[str]]] = []
    for _label, members in groups.items():
        poly = _wing_polygon_from_rooms(members, rooms, geom_wings)
        if poly is None or poly.area < min_area_m2:
            continue
        wing_dissenters: set[str] = set()
        for ridx in members:
            wing_dissenters.update(per_room_dissent.get(ridx, set()))
        candidates.append((poly, members, wing_dissenters))

    if not candidates:
        return geom_wings

    candidates.sort(key=lambda triple: -triple[0].area)
    rank_by_area = list(range(len(candidates)))

    out: list[Wing] = []
    for new_idx, (poly, _members, dissenters) in enumerate(candidates):
        # Confidence rule (per the plan):
        #   no dissent across this wing → "high"
        #   one tier dissented → "medium"
        #   two or more tiers dissented (only possible with >= 3 tiers) → "low"
        if not dissenters:
            confidence = "high"
        elif len(dissenters) <= 1 and n_tiers >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        out.append(
            Wing(
                index=new_idx,
                polygon=poly,
                area_m2=float(poly.area),
                role="main" if new_idx == rank_by_area[0] else "extension",
                long_axis_math=_long_axis_math_deg(poly),
                confidence=confidence,
                disagreement=tuple(sorted(dissenters)),
            )
        )
    return out


def decompose_to_wings_v2(
    footprint: Polygon,
    *,
    room_graph: RoomAdjacency | None = None,
    session_clusters: dict[int, int] | None = None,
    rooms: Sequence | None = None,
    roof_cluster_azimuths: list[float] | None = None,
    min_area_m2: float = _MIN_RECT_AREA_M2,
) -> list[Wing]:
    """Multi-signal wing decomposition.

    Always runs the geometric tier (v1's `decompose_to_wings`). When
    `rooms` is provided, additionally runs the graph and session tiers
    where their inputs are available, then fuses by strict-majority
    same-wing voting. With all signal kwargs `None`, behaviour reduces
    exactly to `decompose_to_wings`.
    """
    if footprint.is_empty or footprint.area <= 0:
        return []

    geom_wings = decompose_to_wings(footprint, min_area_m2=min_area_m2)

    if rooms is None:
        return geom_wings
    if room_graph is None and session_clusters is None:
        return geom_wings

    rooms_count = len(rooms)
    if rooms_count == 0:
        return geom_wings

    partitions: list[tuple[str, dict[int, int]]] = []
    geom_part = _geometric_partition(rooms, geom_wings)
    partitions.append(("geometric", geom_part))

    if room_graph is not None:
        partitions.append(("graph", _graph_partition(rooms, room_graph)))
    if session_clusters is not None:
        partitions.append(
            ("session", _session_partition(rooms, session_clusters, room_graph))
        )

    if len(partitions) == 1:
        return geom_wings

    consensus, per_room_dissent = _consensus_partition(rooms_count, partitions)
    return _wings_from_consensus(
        consensus,
        per_room_dissent,
        rooms,
        geom_wings,
        n_tiers=len(partitions),
        min_area_m2=min_area_m2,
    )


_WINGS_V2_CACHE: dict[str, list[Wing]] = {}


def compute_wings_for_model_v2(model) -> list[Wing]:
    """Multi-signal wing decomposition for a `BuildingModel`, cached per uuid.

    Wraps `decompose_to_wings_v2` by deriving `room_graph` and
    `session_clusters` from the model's rooms and merged-doc data, then
    invoking the multi-signal entry point.
    """
    cached = _WINGS_V2_CACHE.get(model.uuid)
    if cached is not None:
        return cached

    from reconcile_tiers._core.shapely2 import make_valid_polygon
    from reconcile_tiers.roof.footprint import build_building_footprint

    footprint = build_building_footprint(model)
    if footprint is None or len(footprint.polygon_xz) < 3:
        _WINGS_V2_CACHE[model.uuid] = []
        return []
    poly = make_valid_polygon(Polygon(footprint.polygon_xz))
    if poly is None or poly.is_empty:
        _WINGS_V2_CACHE[model.uuid] = []
        return []

    rooms = list(model.rooms)
    room_graph = _build_room_graph_safe(rooms)
    session_clusters = _build_session_clusters_for_model(model)

    wings = decompose_to_wings_v2(
        poly,
        room_graph=room_graph,
        session_clusters=session_clusters,
        rooms=rooms,
    )
    _WINGS_V2_CACHE[model.uuid] = wings
    return wings


def _build_room_graph_safe(rooms: Sequence) -> RoomAdjacency | None:
    """Best-effort room-graph construction; None if upstream geometry missing."""
    try:
        from reconcile_tiers._core.room_graph import build_room_graph

        return build_room_graph(rooms)
    except Exception:
        return None


def _build_session_clusters_for_model(model) -> dict[int, int] | None:
    """Best-effort session clustering by parsing this model's merged.json.

    Returns None when the merged doc cannot be located (e.g., synthetic
    test models with no `pipeline-outputs/<uuid>` directory). The session
    tier is then simply not run.
    """
    try:
        import os

        from reconcile_tiers._core.session_clusters import (
            cluster_merged_rooms_by_session,
        )
        from reconcile_tiers.ingest.merged import load_merged

        pipeline_dir = os.environ.get(
            "PIPELINE_OUTPUTS_DIR",
            os.path.join(os.path.dirname(__file__), "..", "..", "pipeline-outputs"),
        )
        merged_doc = load_merged(model.uuid, pipeline_dir)
        # Map MergedRoom.index → session_id; assume MergedRoom.index aligns
        # with ExtractedRoom.index (they're both positional on the merged
        # rooms list).
        return cluster_merged_rooms_by_session(merged_doc.rooms)
    except Exception:
        return None
