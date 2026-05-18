"""Output and validation helpers for topology V2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ifc_mapping import ConformanceReport
from .models import TopologyGraph


def _load_schema() -> dict[str, Any]:
    schema_path = Path(__file__).parent / "schema" / "topology-v2.schema.json"
    return json.loads(schema_path.read_text())


def validate_against_schema(payload: dict[str, Any]) -> list[str]:
    """Validate output against JSON schema if jsonschema is available.

    Falls back to minimal structural checks when dependency is unavailable.
    """
    errors: list[str] = []
    schema = _load_schema()

    try:
        import jsonschema  # type: ignore

        validator = jsonschema.Draft202012Validator(schema)
        for err in validator.iter_errors(payload):
            path = "/".join(str(x) for x in err.path)
            errors.append(f"{path}: {err.message}")
        return errors
    except Exception:
        # Minimal required checks
        for key in schema.get("required", []):
            if key not in payload:
                errors.append(f"missing required key: {key}")
        if "nodes" in payload and not isinstance(payload["nodes"], list):
            errors.append("nodes must be a list")
        if "edges" in payload and not isinstance(payload["edges"], list):
            errors.append("edges must be a list")
        return errors


def _pct(num: int, total: int) -> str:
    val = (num / total * 100) if total else 100
    return f"{val:.1f}%"


def build_qa_report(
    graph: TopologyGraph,
    conformance: ConformanceReport,
    schema_errors: list[str],
) -> dict[str, Any]:
    nodes = graph.nodes
    edges = graph.edges

    rooms = [n for n in nodes if n.type == "Room"]
    stories = [n for n in nodes if n.type == "Story"]
    surfaces = [n for n in nodes if n.type == "Surface"]

    # every room has exactly one parent story
    room_parent_count = {r.id: 0 for r in rooms}
    for e in edges:
        if e.type == "CONTAINS" and e.to_id in room_parent_count:
            room_parent_count[e.to_id] += 1
    room_parent_ok = sum(1 for c in room_parent_count.values() if c == 1)

    # surface linkage: linked to room OR explicitly top-level
    surface_room_linked = set()
    for e in edges:
        if (
            e.type == "CONTAINS"
            and e.to_id.startswith("surface:")
            and e.from_id.startswith("room:")
        ):
            surface_room_linked.add(e.to_id)
    linked_or_top_level = sum(
        1
        for s in surfaces
        if s.id in surface_room_linked or bool(s.properties.get("top_level"))
    )

    acceptance = {
        "rooms_single_parent_story": {
            "target": "100%",
            "actual": f"{(room_parent_ok / len(rooms) * 100) if rooms else 100:.1f}%",
            "pass": room_parent_ok == len(rooms),
        },
        "room_story_orphans": {
            "target": 0,
            "actual": conformance.checks.get("orphans", {}).get("orphans", 0),
            "pass": conformance.checks.get("orphans", {}).get("orphans", 0) == 0,
        },
        "surface_linkage": {
            "target": ">=95%",
            "actual": _pct(linked_or_top_level, len(surfaces)),
            "pass": (linked_or_top_level / len(surfaces) if surfaces else 1.0) >= 0.95,
        },
        "qa_report_generated": {
            "target": True,
            "actual": True,
            "pass": True,
        },
    }

    stitched = (graph.geometry_index or {}).get("stitched_geometry", {})
    stitched_stats = stitched.get("stats", {})
    stitched_validation = stitched.get("validation", {})

    return {
        "version": graph.version,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "room_count": len(rooms),
            "story_count": len(stories),
            "surface_count": len(surfaces),
            "schema_valid": len(schema_errors) == 0,
            "ifc_conformance_passed": conformance.passed,
        },
        "schema": {
            "errors": schema_errors,
        },
        "ifc_mapping": {
            "mapping_table": {
                "Room": "IfcSpace",
                "Story": "IfcBuildingStorey",
                "Surface.wall": "IfcWall",
                "Surface.door": "IfcDoor",
                "Surface.window": "IfcWindow",
                "Surface.opening": "IfcOpeningElement",
                "Surface.floor": "IfcSlab",
                "CONTAINS": "IfcRelContainedInSpatialStructure",
                "Boundary": "IfcRelSpaceBoundary",
            },
            "conformance": conformance.as_dict(),
        },
        "stitched_geometry": {
            "stats": stitched_stats,
            "validation": stitched_validation,
        },
        "acceptance_thresholds": acceptance,
    }


def write_topology_outputs(
    graph: TopologyGraph,
    output_v2_path: Path,
    qa_output_path: Path,
    conformance: ConformanceReport,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = graph.as_dict()
    schema_errors = validate_against_schema(payload)
    qa = build_qa_report(graph, conformance, schema_errors)

    output_v2_path.parent.mkdir(parents=True, exist_ok=True)
    qa_output_path.parent.mkdir(parents=True, exist_ok=True)

    output_v2_path.write_text(json.dumps(payload, indent=2))
    qa_output_path.write_text(json.dumps(qa, indent=2))

    return payload, qa
