"""Validation helpers for enriched topology graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import TopologyGraph


@dataclass
class GraphDiagnostic:
    code: str
    severity: str
    message: str
    node_id: str | None = None
    edge_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
            "details": self.details,
        }


def validate_graph(graph: TopologyGraph) -> list[GraphDiagnostic]:
    diagnostics: list[GraphDiagnostic] = []

    for room in graph.nodes_by_type("Room"):
        bounded_by = graph.edges_from(room.id, "BOUNDED_BY")
        if bounded_by and len(bounded_by) < 3:
            diagnostics.append(
                GraphDiagnostic(
                    code="ROOM_TOO_FEW_BOUNDARIES",
                    severity="warning",
                    message="Room has fewer than three BOUNDED_BY edges.",
                    node_id=room.id,
                    details={"bounded_by_count": len(bounded_by)},
                )
            )

    adjacency_pairs = {
        (edge.from_id, edge.to_id): edge
        for edge in graph.edges
        if edge.type == "ADJACENT_TO"
    }
    for edge in [edge for edge in graph.edges if edge.type == "ADJACENT_TO"]:
        if (edge.to_id, edge.from_id) not in adjacency_pairs:
            diagnostics.append(
                GraphDiagnostic(
                    code="ASYMMETRIC_ADJACENCY",
                    severity="warning",
                    message="ADJACENT_TO edge is missing its reverse direction.",
                    edge_id=edge.id,
                    details={"from": edge.from_id, "to": edge.to_id},
                )
            )

    above_pairs = {
        (edge.from_id, edge.to_id): edge for edge in graph.edges if edge.type == "ABOVE"
    }
    below_pairs = {
        (edge.from_id, edge.to_id): edge for edge in graph.edges if edge.type == "BELOW"
    }
    for edge in [edge for edge in graph.edges if edge.type == "ABOVE"]:
        if edge.from_id == edge.to_id:
            diagnostics.append(
                GraphDiagnostic(
                    code="SELF_ABOVE",
                    severity="error",
                    message="Node cannot be ABOVE itself.",
                    edge_id=edge.id,
                )
            )
        if (edge.to_id, edge.from_id) not in below_pairs:
            diagnostics.append(
                GraphDiagnostic(
                    code="ABOVE_BELOW_MISMATCH",
                    severity="warning",
                    message="ABOVE edge is missing matching BELOW edge.",
                    edge_id=edge.id,
                    details={"from": edge.from_id, "to": edge.to_id},
                )
            )
        if (edge.to_id, edge.from_id) in above_pairs:
            diagnostics.append(
                GraphDiagnostic(
                    code="ABOVE_CYCLE",
                    severity="error",
                    message="ABOVE relation is cyclic.",
                    edge_id=edge.id,
                    details={"from": edge.from_id, "to": edge.to_id},
                )
            )

    for node in graph.nodes:
        thickness = (node.properties or {}).get("thickness_m")
        if thickness is not None and float(thickness) < 0:
            diagnostics.append(
                GraphDiagnostic(
                    code="NEGATIVE_THICKNESS",
                    severity="error",
                    message="Negative thickness is invalid.",
                    node_id=node.id,
                    details={"thickness_m": thickness},
                )
            )
        floor_y = (node.properties or {}).get("floor_height_y")
        ceil_y = (node.properties or {}).get("ceiling_height_y")
        if (
            floor_y is not None
            and ceil_y is not None
            and float(ceil_y) < float(floor_y)
        ):
            diagnostics.append(
                GraphDiagnostic(
                    code="CEILING_BELOW_FLOOR",
                    severity="error",
                    message="Ceiling height is below floor height.",
                    node_id=node.id,
                    details={"floor_height_y": floor_y, "ceiling_height_y": ceil_y},
                )
            )

    for gap in graph.nodes_by_type("Gap"):
        bounded = graph.edges_from(gap.id, "BOUNDED_BY") + graph.edges_to(
            gap.id, "BOUNDED_BY"
        )
        room_refs = {
            edge.to_id if edge.from_id == gap.id else edge.from_id
            for edge in bounded
            if (edge.to_id.startswith("room:") or edge.from_id.startswith("room:"))
        }
        if bounded and len(room_refs) < 1:
            diagnostics.append(
                GraphDiagnostic(
                    code="GAP_WITHOUT_ROOM",
                    severity="warning",
                    message="Gap is not connected to any room via BOUNDED_BY.",
                    node_id=gap.id,
                )
            )

    return diagnostics
