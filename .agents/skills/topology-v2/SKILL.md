---
name: topology-v2
description: >
  Use when working on reconcile_v2/, the topology graph, wall thickness
  inference, IFC mapping, geometry stitching, or the V2 CLI pipeline.
---

# Topology V2 Pipeline

## How to Work on Topology

1. **Understand the graph model** — this is not geometry code; it's a graph/topology layer on top of geometry. Read `models.py` first to understand GraphNode, GraphEdge, and TopologyGraph before touching anything.
2. **Check IFC standards** — IFC (Industry Foundation Classes) is an ISO standard for BIM. Search [buildingSMART IFC docs](https://standards.buildingsmart.org/IFC/) when adding or modifying IFC mappings. Our mappings must align with the standard.
3. **Consult Shapely docs** — `stitch_geometry.py` relies heavily on Shapely for 2D polygon operations. Check [Shapely docs](https://shapely.readthedocs.io/) before writing custom polygon logic.
4. **Keep IDs deterministic** — the builder generates UUIDs via stable hashing. Any change to the hashing inputs will change all IDs across all buildings. This breaks regression tests and downstream consumers.
5. **Run the conformance check** — after any change, verify `ConformanceReport.issues` is empty (or only contains expected issues). IFC conformance is a quality gate.
6. **Check calor for the consumer perspective** — the Go backend consumes topology output. Changes to the output schema may break calor's processing.

## Purpose

V2 builds a graph-based topology from merged building data. Distinct from V1 (geometry extraction): V2 focuses on room adjacency, wall inference, and IFC conformance.

### Why Topology Matters

Geometry tells you *where* things are. Topology tells you *how things connect*. For energy assessments, knowing that Room A and Room B share a 20cm exterior wall matters more than knowing their exact 3D coordinates — because that wall is where heat escapes.

- **Adjacency** = which rooms share walls. This determines heat transfer paths.
- **Wall thickness** = the gap between two rooms' floor edges where they meet. Thicker walls have different thermal properties.
- **IFC conformance** = the topology must export to IFC (the building industry standard) so other tools can consume it. Each room becomes an IfcSpace, each wall an IfcWall.

## Pipeline Flow

```
Building data
  -> builder.py       (create TopologyGraph: nodes + edges)
  -> topology.py      (infer intra-story adjacency)
  -> wall_thickness_inference.py  (sample edge distances -> statistics)
  -> stitch_geometry.py           (coplanar clustering + 2D union via Shapely)
  -> ifc_mapping.py   (conformance check against IFC classes)
  -> output.py        (JSON schema validation + write)
```

Entry point: `reconcile_v2/cli.py` -> `build_and_write_v2()`

## Core Models (`reconcile_v2/models.py`)

| Class | Key Fields | Purpose |
|-------|-----------|---------|
| `GraphNode` | id, type, story, metadata | Room/space in topology graph |
| `GraphEdge` | source, target, type, metadata | Adjacency or wall relationship |
| `TopologyGraph` | nodes, edges, quality_metrics | Complete building topology |

## Module Reference

| Module | Key Export | Purpose |
|--------|----------|---------|
| `builder.py` | — | Creates nodes/edges from Building/Surface/Room data. Deterministic IDs via stable hashing. |
| `topology.py` | `infer_intra_story_adjacency()` | Detects wall thickness from floor-gap evidence. Uses `IntraStoryAdjacency`, `GapRecord` dataclasses. |
| `wall_thickness_inference.py` | `InferredWallThickness` | Samples edge distances between adjacent rooms. Computes p05/p50/p95/std with confidence scoring. |
| `ifc_mapping.py` | `ConformanceReport` | Maps domain objects to IFC classes. Validates alignment with issue tracking. |
| `stitch_geometry.py` | — | Coplanar clustering by story/kind/plane similarity. 2D union in plane space using Shapely. Shared vertex indexing. |
| `output.py` | — | JSON schema validation via `jsonschema` with fallback to minimal structural checks. |

## IFC Mapping

| Domain Object | IFC Class |
|--------------|-----------|
| Room | IfcSpace |
| Wall | IfcWall |
| Story | IfcBuildingStorey |
| Building | IfcBuilding |

`ConformanceReport` tracks issues — check `.issues` list after validation.

## Key Patterns

- **Deterministic IDs**: Builder generates UUIDs via stable hashing — same input always produces same output. Essential for diffing and regression testing.
- **Confidence scoring**: Wall thickness inference returns confidence alongside statistics. Low confidence = insufficient edge samples.
- **Shapely for 2D union**: `stitch_geometry.py` projects 3D surfaces to 2D plane space, performs union via Shapely, then re-projects to 3D with shared vertex indexing.
- **Schema validation with fallback**: `output.py` tries `jsonschema` library first, falls back to minimal structural checks if library unavailable.

## Gotchas

- `build_and_write_v2()` is the canonical entry — don't call modules individually unless testing
- Wall thickness uses percentile statistics (p05/p50/p95), not simple mean — resistant to outliers
- Stitch geometry depends on Shapely v2 — polygon operations may differ from v1
- IFC conformance is a check, not a transform — it reports issues but doesn't fix them
