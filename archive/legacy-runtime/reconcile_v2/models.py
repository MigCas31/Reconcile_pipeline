"""Data models for topology V2 graph artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    id: str
    type: str
    story: int | None = None
    confidence: float | None = None
    source: str | None = None
    source_ids: list[str] = field(default_factory=list)
    transform_ref: str | None = None
    bbox_xz: list[float] | None = None  # [min_x, min_z, max_x, max_z]
    metrics: dict[str, Any] = field(default_factory=dict)
    legacy_refs: dict[str, str] = field(default_factory=dict)
    ifc_class: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    id: str
    type: str
    from_id: str
    to_id: str
    confidence: float | None = None
    source: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    ifc_relation: str | None = None


@dataclass
class TopologyGraph:
    version: str
    metadata: dict[str, Any]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    geometry_index: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    _node_index: dict[str, GraphNode] = field(init=False, repr=False)
    _edge_from_index: dict[str, list[GraphEdge]] = field(init=False, repr=False)
    _edge_to_index: dict[str, list[GraphEdge]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._node_index = {node.id: node for node in self.nodes}
        self._edge_from_index = {}
        self._edge_to_index = {}
        for edge in self.edges:
            self._edge_from_index.setdefault(edge.from_id, []).append(edge)
            self._edge_to_index.setdefault(edge.to_id, []).append(edge)

    def refresh(self) -> None:
        self._rebuild_indexes()

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._node_index.get(node_id)

    def nodes_by_type(self, node_type: str) -> list[GraphNode]:
        return [node for node in self.nodes if node.type == node_type]

    def edges_from(self, node_id: str, edge_type: str | None = None) -> list[GraphEdge]:
        edges = list(self._edge_from_index.get(node_id, []))
        if edge_type is None:
            return edges
        return [edge for edge in edges if edge.type == edge_type]

    def edges_to(self, node_id: str, edge_type: str | None = None) -> list[GraphEdge]:
        edges = list(self._edge_to_index.get(node_id, []))
        if edge_type is None:
            return edges
        return [edge for edge in edges if edge.type == edge_type]

    def neighbors(self, node_id: str, edge_type: str) -> list[GraphNode]:
        out: list[GraphNode] = []
        for edge in self.edges_from(node_id, edge_type):
            neighbor = self.get_node(edge.to_id)
            if neighbor is not None:
                out.append(neighbor)
        for edge in self.edges_to(node_id, edge_type):
            neighbor = self.get_node(edge.from_id)
            if neighbor is not None:
                out.append(neighbor)
        seen: set[str] = set()
        deduped: list[GraphNode] = []
        for node in out:
            if node.id in seen:
                continue
            seen.add(node.id)
            deduped.append(node)
        return deduped

    def parent(self, node_id: str) -> GraphNode | None:
        for edge in self.edges_to(node_id, "CONTAINS"):
            node = self.get_node(edge.from_id)
            if node is not None:
                return node
        return None

    def children(self, node_id: str) -> list[GraphNode]:
        out: list[GraphNode] = []
        for edge in self.edges_from(node_id, "CONTAINS"):
            node = self.get_node(edge.to_id)
            if node is not None:
                out.append(node)
        return out

    def find_path(self, from_id: str, to_id: str) -> list[GraphEdge] | None:
        if from_id == to_id:
            return []
        queue: list[tuple[str, list[GraphEdge]]] = [(from_id, [])]
        seen = {from_id}
        while queue:
            current, path = queue.pop(0)
            for edge in self.edges_from(current):
                if edge.to_id in seen:
                    continue
                next_path = [*path, edge]
                if edge.to_id == to_id:
                    return next_path
                seen.add(edge.to_id)
                queue.append((edge.to_id, next_path))
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metadata": self.metadata,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "story": n.story,
                    "confidence": n.confidence,
                    "source": n.source,
                    "source_ids": n.source_ids,
                    "transform_ref": n.transform_ref,
                    "bbox_xz": n.bbox_xz,
                    "metrics": n.metrics,
                    "legacy_refs": n.legacy_refs,
                    "ifc_class": n.ifc_class,
                    "properties": n.properties,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "type": e.type,
                    "from": e.from_id,
                    "to": e.to_id,
                    "confidence": e.confidence,
                    "source": e.source,
                    "evidence": e.evidence,
                    "ifc_relation": e.ifc_relation,
                }
                for e in self.edges
            ],
            "geometry_index": self.geometry_index,
            "quality": self.quality,
        }
