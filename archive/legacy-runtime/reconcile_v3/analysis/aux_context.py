"""Optional per-building auxiliary context for ontology / V2 / cross-modal features.

This cache is intentionally separate from ``v3_context`` because it is much
more expensive to build: it runs the topology graph and ontology summary
builders over ``pipeline-outputs/<uuid>/merged.json`` plus scan-cache data.

Callers opt into it explicitly when they need ontology / V2 / cross-modal
features. The lightweight V3-only feature path stays unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reconcile_v2.graph_builder import build_topology_graph

_SCHEMA_VERSION = 1


def _room_key(room_index: int) -> str:
    return f"room:{room_index}"


def _pick(d: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


def _surface_record(surface: dict[str, Any]) -> dict[str, Any]:
    return _pick(
        surface,
        (
            "id",
            "category",
            "source_kind",
            "source_id",
            "room_id",
            "room_index",
            "story",
            "part_id",
            "roof_hypothesis_id",
            "cell_id",
            "cell_kind",
            "corners",
        ),
    )


def _roof_surface_record(surface: dict[str, Any]) -> dict[str, Any]:
    record = _pick(
        surface,
        (
            "kind",
            "surface_kind",
            "story",
            "dominant_story",
            "y",
            "corners",
            "roof_hypothesis_id",
            "roof_hypothesis_support_score",
            "flat_role",
            "flat_role_reason",
            "room_index",
            "graph_room_id",
        ),
    )
    cluster = surface.get("cluster") or {}
    if isinstance(cluster, dict):
        if cluster.get("avgAzimuth") is not None:
            record["avg_azimuth_deg"] = cluster.get("avgAzimuth")
        if cluster.get("avgIncl") is not None:
            record["avg_incl_deg"] = cluster.get("avgIncl")
    return record


def _cell_record(cell: dict[str, Any]) -> dict[str, Any]:
    return _pick(
        cell,
        (
            "id",
            "kind",
            "story",
            "source_id",
            "bbox_xyz",
            "centroid",
            "volume_m3",
            "properties",
            "faces",
        ),
    )


def _roof_cell_record(cell: dict[str, Any]) -> dict[str, Any]:
    return _pick(
        cell,
        (
            "id",
            "cell_kind",
            "story",
            "room_id",
            "room_index",
            "part_id",
            "base_atom_id",
            "roof_hypothesis_id",
            "roof_surface_kind",
            "volume_m3",
            "centroid_xyz",
            "bbox_xyz",
            "faces",
        ),
    )


def build_for_uuid(uuid: str, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Build the expensive auxiliary context for a single building UUID."""
    import reconcile.viewer_server as vs

    pipeline_root = Path(vs.PIPELINE_ROOT)
    scan_cache_root = Path(vs.SCAN_CACHE_ROOT)
    merged_path = pipeline_root / uuid / "merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"No merged.json for {uuid}")

    graph = build_topology_graph(
        merged_path=merged_path, scan_dir=scan_cache_root, uuid=uuid
    )
    building = vs.extract_building(
        uuid=uuid,
        pipeline_dir=pipeline_root,
        scan_cache_root=scan_cache_root,
        load_topology_graph=False,
    )
    if not building:
        raise RuntimeError(f"extract_building returned no building for {uuid}")
    roof = vs.run_roof_algorithms(building, graph=graph)
    topology_cell_complex = (graph.geometry_index or {}).get("cell_complex") or {}
    summary, part_graph_room_ids = vs._build_ontology_summary(
        uuid=uuid,
        roof=roof,
        topology_cell_complex=topology_cell_complex,
        building=building,
    )
    parts = vs._build_ontology_part_payloads(
        uuid=uuid,
        summary=summary,
        part_graph_room_ids=part_graph_room_ids,
        topology_cell_complex=topology_cell_complex,
        roof_cell_complex=roof.get("roof_cell_complex") or {},
        occupied_room_cell_complex=roof.get("occupied_room_cell_complex") or {},
        building=building,
        roof=roof,
    )
    full_model = parts.get(vs.FULL_BUILDING_PART_ID) or {}

    topology = {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "quality": graph.quality,
        "surface_nodes": [
            {
                "id": node.id,
                "ifc_class": node.ifc_class,
                "source": node.source,
                "source_ids": list(node.source_ids or []),
                "legacy_refs": node.legacy_refs or {},
                "properties": node.properties or {},
            }
            for node in graph.nodes
            if node.type == "Surface"
        ],
        "adjacency_edges": [
            {
                "id": edge.id,
                "type": edge.type,
                "from_id": edge.from_id,
                "to_id": edge.to_id,
                "evidence": edge.evidence or {},
            }
            for edge in graph.edges
            if edge.type in {"ADJACENT_TO", "ABOVE", "BELOW"}
        ],
        "cell_complex_cells": [
            _cell_record(cell) for cell in (topology_cell_complex.get("cells") or [])
        ],
    }
    ontology = {
        "summary_metadata": summary.get("metadata") or {},
        "building_parts": summary.get("building_parts") or [],
        "coverage_subparts": summary.get("coverage_subparts") or [],
        "semantic_atoms": summary.get("semantic_atoms") or [],
        "room_summaries": summary.get("room_summaries") or {},
        "roof_coverage_metadata": summary.get("roof_coverage_metadata") or {},
        "top_boundary_metadata": summary.get("top_boundary_metadata") or {},
        "roof_evidence_metadata": summary.get("roof_evidence_metadata") or {},
        "full_model_metadata": full_model.get("metadata") or {},
        "full_model_roof_cells": [
            _roof_cell_record(cell) for cell in (full_model.get("roof_cells") or [])
        ],
        "full_model_knee_walls": full_model.get("knee_walls") or [],
        "full_model_renderable_surfaces": [
            _surface_record(surface)
            for surface in (full_model.get("renderable_surfaces") or [])
        ],
        "roof_surfaces_oblique": [
            _roof_surface_record(surface)
            for surface in ((roof.get("roof_surfaces") or {}).get("oblique") or [])
        ],
        "roof_surfaces_flat": [
            _roof_surface_record(surface)
            for surface in ((roof.get("roof_surfaces") or {}).get("flat") or [])
        ],
    }
    return {
        "aux_schema_version": _SCHEMA_VERSION,
        "building_uuid": uuid,
        "topology": topology,
        "ontology": ontology,
    }


def _cache_is_compatible(ctx: dict[str, dict]) -> bool:
    if not ctx:
        return False
    try:
        sample = next(iter(ctx.values()))
    except StopIteration:
        return False
    return (
        isinstance(sample, dict)
        and sample.get("aux_schema_version") == _SCHEMA_VERSION
        and "topology" in sample
        and "ontology" in sample
    )


def load_or_build(
    cache_path: Path,
    *,
    only_uuids: set[str],
    rebuild: bool = False,
) -> dict[str, dict]:
    """Load the auxiliary cache, building missing UUIDs on demand."""
    ctx: dict[str, dict] = {}
    if not rebuild and cache_path.exists():
        with cache_path.open() as f:
            loaded = json.load(f)
        if _cache_is_compatible(loaded):
            ctx = loaded
    missing = sorted(uuid for uuid in only_uuids if uuid not in ctx)
    if missing:
        for uuid in missing:
            ctx[uuid] = build_for_uuid(uuid)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(ctx, f)
    return {uuid: ctx[uuid] for uuid in only_uuids if uuid in ctx}
