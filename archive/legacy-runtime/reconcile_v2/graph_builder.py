"""Graph enrichment layer for topology-v2.

Builds on the existing V2 graph artifact and adds ontology-oriented 3D cell
complex data, semantic relations, and validation diagnostics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from shapely.geometry import Polygon

from . import builder as base_builder
from .cell_decomposition import CellComplex, build_cell_complex
from .graph_validation import validate_graph
from .models import GraphEdge, GraphNode, TopologyGraph


def _stable_hash(parts: list[str], length: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:length]


def _edge_id(edge_type: str, from_id: str, to_id: str, salt: str = "") -> str:
    return f"edge:{edge_type.lower()}:{_stable_hash([edge_type, from_id, to_id, salt])}"


def _put_edge(edge_map: dict[str, GraphEdge], edge: GraphEdge) -> None:
    if edge.id not in edge_map:
        edge_map[edge.id] = edge


def _relation_rank(state: str) -> int:
    return {"weak": 0, "partial": 1, "confirmed": 2}.get(str(state), 1)


def _merge_relation_state(existing: str | None, derived: str) -> str:
    if not existing:
        return derived
    return existing if _relation_rank(existing) <= _relation_rank(derived) else derived


def _put_relation_edge(edge_map: dict[str, GraphEdge], edge: GraphEdge) -> None:
    for existing in edge_map.values():
        if (
            existing.type == edge.type
            and existing.from_id == edge.from_id
            and existing.to_id == edge.to_id
        ):
            merged = {**(existing.evidence or {})}
            merged.update(edge.evidence or {})
            existing_state = (existing.evidence or {}).get("relation_state")
            new_state = (edge.evidence or {}).get("relation_state")
            if existing_state is not None and new_state is not None:
                merged["relation_state"] = _merge_relation_state(
                    str(existing_state), str(new_state)
                )
            existing_support = (existing.evidence or {}).get("support_ratio")
            new_support = (edge.evidence or {}).get("support_ratio")
            if existing_support is not None and new_support is not None:
                merged["support_ratio"] = min(
                    float(existing_support), float(new_support)
                )
            existing.evidence = merged
            return
    _put_edge(edge_map, edge)


def _story_stats_from_cells(
    cell_complex: CellComplex,
) -> dict[int, dict[str, float | int]]:
    stats: dict[int, dict[str, float | int]] = {}
    unique_rooms_by_story: dict[int, set[str]] = {}
    for cell in cell_complex.cells:
        if cell.kind != "room" or not isinstance(cell.story, int):
            continue
        entry = stats.setdefault(
            cell.story,
            {
                "floor_y": cell.bbox_xyz[1],
                "ceiling_y": cell.bbox_xyz[4],
                "room_count": 0,
            },
        )
        entry["floor_y"] = min(float(entry["floor_y"]), cell.bbox_xyz[1])
        entry["ceiling_y"] = max(float(entry["ceiling_y"]), cell.bbox_xyz[4])
        if cell.source_id:
            unique_rooms_by_story.setdefault(cell.story, set()).add(cell.source_id)
    for story, entry in stats.items():
        entry["room_count"] = len(unique_rooms_by_story.get(story, set()))
        entry["height_m"] = float(entry["ceiling_y"]) - float(entry["floor_y"])
        entry["story_index"] = story
    return stats


def _classify_surface_roles(graph: TopologyGraph) -> None:
    exposed_rooms = {
        edge.from_id
        for edge in graph.edges
        if edge.type == "EXPOSES_TO" and edge.from_id.startswith("room:")
    }
    container_count: dict[str, int] = {}
    for edge in graph.edges:
        if (
            edge.type == "CONTAINS"
            and edge.to_id.startswith("surface:")
            and edge.from_id.startswith("room:")
        ):
            container_count[edge.to_id] = container_count.get(edge.to_id, 0) + 1

    for node in graph.nodes:
        if node.type != "Surface":
            continue
        props = node.properties or {}
        kind = props.get("surface_kind")
        if kind == "floor":
            props["surface_role"] = "floor_slab"
        elif kind == "wall":
            if container_count.get(node.id, 0) > 1:
                props["surface_role"] = "interior_wall"
            else:
                parent_rooms = [
                    edge.from_id
                    for edge in graph.edges
                    if edge.type == "CONTAINS"
                    and edge.to_id == node.id
                    and edge.from_id.startswith("room:")
                ]
                props["surface_role"] = (
                    "exterior_wall"
                    if any(room_id in exposed_rooms for room_id in parent_rooms)
                    else "interior_wall"
                )
        elif kind == "opening":
            props["surface_role"] = "virtual"
        node.properties = props


def _bbox_gap(a: list[float] | None, b: list[float] | None) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 4 or len(b) != 4:
        return float("inf")
    ax0, az0, ax1, az1 = [float(v) for v in a]
    bx0, bz0, bx1, bz1 = [float(v) for v in b]
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dz = max(bz0 - az1, az0 - bz1, 0.0)
    if dx == 0.0 and dz == 0.0:
        return 0.0
    return (dx * dx + dz * dz) ** 0.5


def _emit_surface_connectivity(graph: TopologyGraph) -> None:
    edges = {edge.id: edge for edge in graph.edges}
    room_adjacency = {
        (edge.from_id, edge.to_id)
        for edge in graph.edges
        if edge.type == "ADJACENT_TO"
        and edge.from_id.startswith("room:")
        and edge.to_id.startswith("room:")
    }
    room_surfaces: dict[str, list[GraphNode]] = {}
    for edge in graph.edges:
        if (
            edge.type != "CONTAINS"
            or not edge.from_id.startswith("room:")
            or not edge.to_id.startswith("surface:")
        ):
            continue
        surface = graph.get_node(edge.to_id)
        if surface is None or surface.type != "Surface":
            continue
        if (surface.properties or {}).get("surface_kind") != "wall":
            continue
        room_surfaces.setdefault(edge.from_id, []).append(surface)

    for room_a, surfaces_a in room_surfaces.items():
        for room_b, surfaces_b in room_surfaces.items():
            if room_a == room_b or (room_a, room_b) not in room_adjacency:
                continue
            for surface_a in surfaces_a:
                for surface_b in surfaces_b:
                    gap = _bbox_gap(surface_a.bbox_xz, surface_b.bbox_xz)
                    if gap > 0.2:
                        continue
                    for left, right in (
                        (surface_a.id, surface_b.id),
                        (surface_b.id, surface_a.id),
                    ):
                        _put_edge(
                            edges,
                            GraphEdge(
                                id=_edge_id("CONNECTS_TO", left, right),
                                type="CONNECTS_TO",
                                from_id=left,
                                to_id=right,
                                source="graph_builder",
                                evidence={
                                    "bbox_gap_m": round(gap, 6),
                                    "junction_type": "BUTT",
                                },
                            ),
                        )

    graph.edges = sorted(edges.values(), key=lambda edge: edge.id)
    graph.refresh()


def _emit_interference_edges(graph: TopologyGraph) -> None:
    edges = {edge.id: edge for edge in graph.edges}
    rooms = [
        node
        for node in graph.nodes
        if node.type == "Room" and isinstance(node.story, int)
    ]
    room_floor_polygons = (graph.geometry_index or {}).get("room_floor_polygons", {})
    for i, left in enumerate(rooms):
        left_wkt = room_floor_polygons.get(left.id)
        if not isinstance(left_wkt, str) or not left_wkt:
            continue
        try:
            left_poly = Polygon()
            from shapely import wkt as shapely_wkt

            left_poly = shapely_wkt.loads(left_wkt)
        except Exception:
            continue
        for right in rooms[i + 1 :]:
            if right.story != left.story:
                continue
            right_wkt = room_floor_polygons.get(right.id)
            if not isinstance(right_wkt, str) or not right_wkt:
                continue
            try:
                from shapely import wkt as shapely_wkt

                right_poly = shapely_wkt.loads(right_wkt)
            except Exception:
                continue
            overlap_area = left_poly.intersection(right_poly).area
            if overlap_area <= 0.0:
                continue
            if overlap_area < 0.01:
                continue
            for from_id, to_id in ((left.id, right.id), (right.id, left.id)):
                _put_edge(
                    edges,
                    GraphEdge(
                        id=_edge_id("INTERFERES", from_id, to_id),
                        type="INTERFERES",
                        from_id=from_id,
                        to_id=to_id,
                        source="graph_builder",
                        evidence={"overlap_area_m2": round(overlap_area, 6)},
                    ),
                )
    graph.edges = sorted(edges.values(), key=lambda edge: edge.id)
    graph.refresh()


def _annotate_partial_relation_states(graph: TopologyGraph) -> None:
    room_nodes = {node.id: node for node in graph.nodes if node.type == "Room"}
    for edge in graph.edges:
        if edge.type not in {"ADJACENT_TO", "ABOVE", "BELOW", "EXPOSES_TO"}:
            continue
        evidence = dict(edge.evidence or {})
        existing_state = evidence.get("relation_state")
        existing_support = evidence.get("support_ratio")
        if edge.type in {"ABOVE", "BELOW"}:
            lower_id = edge.to_id if edge.type == "ABOVE" else edge.from_id
            upper_id = edge.from_id if edge.type == "ABOVE" else edge.to_id
            lower = room_nodes.get(lower_id)
            upper = room_nodes.get(upper_id)
            shared_area = float(evidence.get("shared_face_area", 0.0))
            if shared_area <= 0.0 and existing_state is not None:
                edge.evidence = evidence
                continue
            lower_area = (
                float((lower.properties or {}).get("area_m2", 0.0)) if lower else 0.0
            )
            upper_area = (
                float((upper.properties or {}).get("area_m2", 0.0)) if upper else 0.0
            )
            lower_ratio = shared_area / lower_area if lower_area > 1e-6 else 0.0
            upper_ratio = shared_area / upper_area if upper_area > 1e-6 else 0.0
            support_ratio = (
                min(lower_ratio, upper_ratio)
                if lower_ratio and upper_ratio
                else max(lower_ratio, upper_ratio)
            )
            derived_support = round(support_ratio, 6)
            derived_state = (
                "confirmed"
                if support_ratio >= 0.35
                else ("partial" if support_ratio >= 0.05 else "weak")
            )
            evidence["support_ratio"] = (
                min(float(existing_support), derived_support)
                if existing_support is not None
                else derived_support
            )
            evidence["relation_state"] = (
                _merge_relation_state(str(existing_state), derived_state)
                if existing_state is not None
                else derived_state
            )
        elif edge.type == "EXPOSES_TO":
            room = room_nodes.get(edge.from_id)
            shared_area = float(evidence.get("shared_face_area", 0.0))
            if shared_area <= 0.0 and existing_state is not None:
                edge.evidence = evidence
                continue
            room_area = (
                float((room.properties or {}).get("area_m2", 0.0)) if room else 0.0
            )
            room_height = 0.0
            if room is not None:
                floor_y = float((room.properties or {}).get("floor_height_y", 0.0))
                ceil_y = float((room.properties or {}).get("ceiling_height_y", floor_y))
                room_height = max(ceil_y - floor_y, 0.0)
            side_span = shared_area / room_height if room_height > 1e-6 else 0.0
            span_ratio = (
                side_span / max(room_area**0.5, 1e-6) if room_area > 1e-6 else 0.0
            )
            derived_support = round(span_ratio, 6)
            derived_state = (
                "confirmed"
                if span_ratio >= 0.35
                else ("partial" if span_ratio >= 0.05 else "weak")
            )
            evidence["support_ratio"] = (
                min(float(existing_support), derived_support)
                if existing_support is not None
                else derived_support
            )
            evidence["relation_state"] = (
                _merge_relation_state(str(existing_state), derived_state)
                if existing_state is not None
                else derived_state
            )
        elif edge.type == "ADJACENT_TO":
            left = room_nodes.get(edge.from_id)
            right = room_nodes.get(edge.to_id)
            shared_area = float(evidence.get("shared_face_area", 0.0))
            if shared_area <= 0.0 and existing_state is not None:
                edge.evidence = evidence
                continue
            left_height = 0.0
            right_height = 0.0
            left_area = (
                float((left.properties or {}).get("area_m2", 0.0)) if left else 0.0
            )
            right_area = (
                float((right.properties or {}).get("area_m2", 0.0)) if right else 0.0
            )
            if left is not None:
                left_height = max(
                    float((left.properties or {}).get("ceiling_height_y", 0.0))
                    - float((left.properties or {}).get("floor_height_y", 0.0)),
                    0.0,
                )
            if right is not None:
                right_height = max(
                    float((right.properties or {}).get("ceiling_height_y", 0.0))
                    - float((right.properties or {}).get("floor_height_y", 0.0)),
                    0.0,
                )
            width = shared_area / max(
                min(v for v in [left_height, right_height] if v > 1e-6)
                if any(v > 1e-6 for v in [left_height, right_height])
                else 1.0,
                1e-6,
            )
            support_ratio = width / max(
                min(v**0.5 for v in [left_area, right_area] if v > 1e-6)
                if any(v > 1e-6 for v in [left_area, right_area])
                else 1.0,
                1e-6,
            )
            derived_support = round(support_ratio, 6)
            derived_state = (
                "confirmed"
                if support_ratio >= 0.35
                else ("partial" if support_ratio >= 0.05 else "weak")
            )
            evidence["support_ratio"] = (
                min(float(existing_support), derived_support)
                if existing_support is not None
                else derived_support
            )
            evidence["relation_state"] = (
                _merge_relation_state(str(existing_state), derived_state)
                if existing_state is not None
                else derived_state
            )
        edge.evidence = evidence


def _enrich_with_cells(graph: TopologyGraph, cell_complex: CellComplex) -> None:
    nodes = {node.id: node for node in graph.nodes}
    edges = {edge.id: edge for edge in graph.edges}
    gap_source_by_cell_id: dict[str, str] = {}

    story_node_id_by_story: dict[int, str] = {
        node.story: node.id
        for node in nodes.values()
        if node.type == "Story" and isinstance(node.story, int)
    }

    for cell in cell_complex.cells:
        node = GraphNode(
            id=cell.id,
            type="Cell",
            story=cell.story,
            source="cell_decomposition",
            source_ids=[cell.source_id] if cell.source_id else [],
            properties={
                "cell_kind": cell.kind,
                "centroid_xyz": cell.centroid.as_list(),
                "bbox_xyz": [round(v, 6) for v in cell.bbox_xyz],
                "volume_m3": round(cell.volume_m3, 6),
                **cell.properties,
            },
            metrics={"face_count": len(cell.faces)},
        )
        nodes[node.id] = node

        if cell.kind == "gap":
            source_gap_id = (
                cell.source_id
                if isinstance(cell.source_id, str) and cell.source_id.startswith("gap:")
                else None
            )
            if source_gap_id is None:
                source_gap_id = f"gap:derived:{
                    _stable_hash(
                        [cell.id, str(cell.story), str(cell.bbox_xyz)],
                        20,
                    )
                }"
                if source_gap_id not in nodes:
                    height = max(float(cell.bbox_xyz[4]) - float(cell.bbox_xyz[1]), 0.0)
                    area_m2 = 0.0
                    region_wkt = None
                    footprint = (cell.properties or {}).get("xz_footprint")
                    if isinstance(footprint, list) and len(footprint) >= 3:
                        try:
                            poly = Polygon(
                                [(float(p[0]), float(p[1])) for p in footprint]
                            )
                            area_m2 = float(poly.area)
                            region_wkt = poly.wkt
                        except Exception:
                            area_m2 = 0.0
                            region_wkt = None
                    nodes[source_gap_id] = GraphNode(
                        id=source_gap_id,
                        type="Gap",
                        story=cell.story,
                        confidence=0.65,
                        source="cell_decomposition",
                        source_ids=[cell.id],
                        properties={
                            "gap_kind": (cell.properties or {}).get(
                                "gap_kind", "intra_story"
                            ),
                            "relation_state": "confirmed",
                            "derived": True,
                            "derived_enclosed_void": True,
                            "region_wkt": region_wkt,
                        },
                        metrics={
                            "area_m2": round(area_m2, 6),
                            "height_m": round(height, 6),
                            "volume_m3": round(float(cell.volume_m3), 6),
                        },
                    )
                    if isinstance(cell.story, int):
                        story_node_id = story_node_id_by_story.get(cell.story)
                        if story_node_id:
                            _put_edge(
                                edges,
                                GraphEdge(
                                    id=_edge_id(
                                        "HAS_GAP",
                                        story_node_id,
                                        source_gap_id,
                                        "derived-cell",
                                    ),
                                    type="HAS_GAP",
                                    from_id=story_node_id,
                                    to_id=source_gap_id,
                                    source="cell_decomposition",
                                    confidence=0.65,
                                    evidence={
                                        "derived_from_cell": cell.id,
                                        "kind": "enclosed_void",
                                    },
                                ),
                            )
            gap_source_by_cell_id[cell.id] = source_gap_id
            _put_edge(
                edges,
                GraphEdge(
                    id=_edge_id("CONTAINS", source_gap_id, cell.id, "cell"),
                    type="CONTAINS",
                    from_id=source_gap_id,
                    to_id=cell.id,
                    source="graph_builder",
                ),
            )

        if cell.source_id and cell.source_id in nodes:
            source_node = nodes[cell.source_id]
            _put_edge(
                edges,
                GraphEdge(
                    id=_edge_id("CONTAINS", source_node.id, cell.id, "cell"),
                    type="CONTAINS",
                    from_id=source_node.id,
                    to_id=cell.id,
                    source="graph_builder",
                ),
            )
            if source_node.type == "Room":
                for boundary_edge in graph.edges_from(source_node.id, "CONTAINS"):
                    if not boundary_edge.to_id.startswith("boundary:"):
                        continue
                    _put_edge(
                        edges,
                        GraphEdge(
                            id=_edge_id(
                                "BOUNDED_BY", source_node.id, boundary_edge.to_id
                            ),
                            type="BOUNDED_BY",
                            from_id=source_node.id,
                            to_id=boundary_edge.to_id,
                            source="graph_builder",
                        ),
                    )

    for adjacency in cell_complex.adjacency:
        a = nodes.get(adjacency.from_cell_id)
        b = nodes.get(adjacency.to_cell_id)
        if a is None or b is None:
            continue
        a_source = (a.source_ids or [None])[0]
        b_source = (b.source_ids or [None])[0]
        a_kind = (a.properties or {}).get("cell_kind")
        b_kind = (b.properties or {}).get("cell_kind")
        a_gap_source = gap_source_by_cell_id.get(a.id) if a_kind == "gap" else None
        b_gap_source = gap_source_by_cell_id.get(b.id) if b_kind == "gap" else None

        if adjacency.relation == "adjacent_to":
            if (
                a_source
                and b_source
                and a_source.startswith("room:")
                and b_source.startswith("room:")
            ):
                for left, right in ((a_source, b_source), (b_source, a_source)):
                    _put_relation_edge(
                        edges,
                        GraphEdge(
                            id=_edge_id("ADJACENT_TO", left, right, "cell-complex"),
                            type="ADJACENT_TO",
                            from_id=left,
                            to_id=right,
                            source="cell_decomposition",
                            evidence={
                                "shared_face_area": round(
                                    adjacency.shared_face_area, 6
                                ),
                                "axis": adjacency.axis,
                            },
                        ),
                    )
            elif a_kind == "outside" and b_source and b_source.startswith("room:"):
                _put_relation_edge(
                    edges,
                    GraphEdge(
                        id=_edge_id("EXPOSES_TO", b_source, a.id),
                        type="EXPOSES_TO",
                        from_id=b_source,
                        to_id=a.id,
                        source="cell_decomposition",
                        evidence={
                            "shared_face_area": round(adjacency.shared_face_area, 6)
                        },
                    ),
                )
            elif b_kind == "outside" and a_source and a_source.startswith("room:"):
                _put_relation_edge(
                    edges,
                    GraphEdge(
                        id=_edge_id("EXPOSES_TO", a_source, b.id),
                        type="EXPOSES_TO",
                        from_id=a_source,
                        to_id=b.id,
                        source="cell_decomposition",
                        evidence={
                            "shared_face_area": round(adjacency.shared_face_area, 6)
                        },
                    ),
                )
            elif (
                a_kind == "gap"
                and b_source
                and b_source.startswith("room:")
                and a_gap_source
            ):
                _put_edge(
                    edges,
                    GraphEdge(
                        id=_edge_id("BOUNDED_BY", a_gap_source, b_source),
                        type="BOUNDED_BY",
                        from_id=a_gap_source,
                        to_id=b_source,
                        source="cell_decomposition",
                    ),
                )
            elif (
                b_kind == "gap"
                and a_source
                and a_source.startswith("room:")
                and b_gap_source
            ):
                _put_edge(
                    edges,
                    GraphEdge(
                        id=_edge_id("BOUNDED_BY", b_gap_source, a_source),
                        type="BOUNDED_BY",
                        from_id=b_gap_source,
                        to_id=a_source,
                        source="cell_decomposition",
                    ),
                )
        elif adjacency.relation in {"above", "below"} and a_source and b_source:
            if a_source.startswith("room:") and b_source.startswith("room:"):
                if adjacency.relation == "below":
                    lower, upper = a_source, b_source
                else:
                    lower, upper = b_source, a_source
                _put_relation_edge(
                    edges,
                    GraphEdge(
                        id=_edge_id("ABOVE", upper, lower, "cell-complex"),
                        type="ABOVE",
                        from_id=upper,
                        to_id=lower,
                        source="cell_decomposition",
                        evidence={
                            "shared_face_area": round(adjacency.shared_face_area, 6)
                        },
                    ),
                )
                _put_relation_edge(
                    edges,
                    GraphEdge(
                        id=_edge_id("BELOW", lower, upper, "cell-complex"),
                        type="BELOW",
                        from_id=lower,
                        to_id=upper,
                        source="cell_decomposition",
                        evidence={
                            "shared_face_area": round(adjacency.shared_face_area, 6)
                        },
                    ),
                )

    graph.nodes = sorted(nodes.values(), key=lambda node: node.id)
    graph.edges = sorted(edges.values(), key=lambda edge: edge.id)
    graph.refresh()


def _apply_story_and_room_properties(
    graph: TopologyGraph, cell_complex: CellComplex
) -> None:
    story_stats = _story_stats_from_cells(cell_complex)
    room_nodes = {node.id: node for node in graph.nodes if node.type == "Room"}

    highest_story = max(story_stats) if story_stats else None
    for node in graph.nodes:
        if node.type == "Story" and isinstance(node.story, int):
            stats = story_stats.get(node.story, {})
            node.properties.update(stats)
        elif node.type == "Room" and isinstance(node.story, int):
            stats = story_stats.get(node.story, {})
            if stats:
                node.properties["floor_height_y"] = stats.get("floor_y")
                node.properties["ceiling_height_y"] = stats.get("ceiling_y")
                node.properties["is_ground_story"] = node.story == min(story_stats)
                node.properties["is_top_story"] = node.story == highest_story
                room_cells = [
                    cell
                    for cell in graph.nodes_by_type("Cell")
                    if (cell.properties or {}).get("cell_kind") == "room"
                    and (cell.source_ids or [None])[0] == node.id
                ]
                if room_cells:
                    area = sum(
                        Polygon(
                            [
                                (float(p[0]), float(p[1]))
                                for p in (cell.properties or {}).get("xz_footprint", [])
                            ]
                        ).area
                        for cell in room_cells
                        if isinstance((cell.properties or {}).get("xz_footprint"), list)
                        and len((cell.properties or {}).get("xz_footprint")) >= 3
                    )
                    volume = sum(
                        float((cell.properties or {}).get("volume_m3", 0.0))
                        for cell in room_cells
                    )
                    node.properties["area_m2"] = round(area, 6)
                    node.properties["volume_m3"] = round(volume, 6)

    for room_id, room_node in room_nodes.items():
        room_node.properties["adjacent_room_count"] = len(
            [
                edge
                for edge in graph.edges_from(room_id, "ADJACENT_TO")
                if edge.to_id.startswith("room:")
            ]
        )


def build_topology_graph(
    merged_path: Path,
    scan_dir: Path,
    uuid: str = "",
    pipeline_version: str = "v2.1.0",
) -> TopologyGraph:
    graph = base_builder.build_topology_graph(
        merged_path=merged_path,
        scan_dir=scan_dir,
        uuid=uuid,
        pipeline_version=pipeline_version,
    )

    cell_complex = build_cell_complex(graph)
    _enrich_with_cells(graph, cell_complex)
    _apply_story_and_room_properties(graph, cell_complex)
    _annotate_partial_relation_states(graph)
    _classify_surface_roles(graph)
    _emit_surface_connectivity(graph)
    _emit_interference_edges(graph)

    diagnostics = validate_graph(graph)
    graph.geometry_index["cell_complex"] = cell_complex.as_dict()
    graph.quality["cell_complex"] = cell_complex.metadata
    graph.quality["graph_validation"] = {
        "issue_count": len(diagnostics),
        "issues": [issue.as_dict() for issue in diagnostics],
    }
    graph.quality["counts"] = {
        **graph.quality.get("counts", {}),
        "cells": len(cell_complex.cells),
    }
    graph.metadata["pipeline_version"] = pipeline_version
    graph.refresh()
    return graph
