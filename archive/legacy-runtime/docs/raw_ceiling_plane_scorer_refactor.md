# Raw Ceiling Scorer Reset: Logic Spec + Clean V2 Rewrite

## Purpose And Non-Goals

### Purpose
- Replace the monolithic scorer with a modular, relation-first V2.
- Keep diagnostics + viewer overlay as the primary product surface.
- Make every ridge/eave ownership/final-layer decision explainable via explicit spatial relations.

### Non-goals
- Do not promote V2 into production roof selection in `reconcile/roof_algorithms_py`.
- Do not preserve every legacy heuristic branch when equivalent relation-based behavior exists.

## Input Sources And Required Fields

### Input files
- `reconcile/buildings_3d.json`
- `reconcile/roof_algorithms_py_results.json`
- `reports/ridge_eave_scores_20260420/scores.json`
- `reconcile/reconcile_v3_results.json`

### Required source fields
- Building geometry: `rooms[*].story`, `rooms[*].floor_polygon`, room walls/raw ceilings.
- Roof results: `ceiling.planes`, `roof_surfaces.oblique`, `building_part_graph.room_membership`, `building_part_graph.hypothesis_membership`.
- Ridge/eave scores: `plane_groups[*]`, `candidates[*]`, selection flags and plane coefficients.
- V3 segments: `merged_roof_segments[*].footprint_xz`, segment features and snapshots.

## Current V1 Pipeline Map (Ordered By `score_buildings`)

1. Collect story envelopes/gaps.
2. Collect targets (`candidate_oblique`, `committed_oblique`) and selected ridge/eave targets.
3. Build ridge/eave diagnostics + anchor masks.
4. Collect raw evidence (planes -> promote -> edges -> conflicts).
5. Build eave chains and score target-chain support.
6. Build split pieces (supported + residual).
7. Trim/resolve pieces with ownership/mirror/face-run logic.
8. Compute target scores and summary payload.
9. Emit reports (`per_target`, `per_story`, `plane_extent_splits`, summary).

## Function-Cluster Map (V1 Concerns)

- Geometry primitives and polygon conversion helpers.
- Raw plane trust + promotion.
- Edge labeling + chain connectivity.
- Split generation and geometric trimming.
- Ownership/precedence/redundancy arbitration.
- Final-layer classification.
- Reporting/CSV/summary emission.

## Data Contracts Emitted (Current + V2)

### Required outputs
- `per_target.json`
- `plane_extent_splits.json`
- `summary.json`

### Added in V2
- `plane_relations.json`

### Canonical split-piece fields (minimum)
- `uuid`, `story`, `piece_id`, `target_element_id`, `target_kind`, `piece_role`
- `support_score`, `final_layer`, `final_layer_reason`
- `chain_ids`, `corners`, `holes`

## Complexity Hotspots In V1

- Multi-pass ownership + precedence rewrites on row dictionaries.
- Mirror-partner pruning with context-dependent fallbacks.
- Face-run overlap rewriting and post-hoc demotion logic.
- Cross-cutting use of thresholds in multiple stages.

## Keep / Delete / Deferred

### Keep
- Raw trust scoring + low-trust promotion concepts.
- Chain-derived support for split generation.
- Building-part-aware ownership semantics.
- Viewer-facing split payload shape.

### Delete (core path)
- Face-run dual systems as mandatory baseline.
- Implicit row-order competition logic.
- Post-hoc final-layer cascades not backed by explicit pair relations.

### Deferred
- Optional post-processor for advanced same-face reconciliation.
- Full parity comparator automation across all legacy detail columns.

## V2 Architecture (Stage-by-stage)

1. `targets.py`: collect normalized hypotheses.
2. `raw_evidence.py`: collect/promote raw planes, edges, conflicts.
3. `chains.py`: chain construction + facade continuity.
4. `support_scoring.py`: independent target-chain support scores.
5. `splitter.py`: supported/residual partition only.
6. `relation_context.py`: per-target `PlaneContext`.
7. `plane_relations.py`: pairwise `PlaneRelation` classification.
8. `ownership.py`: relation-backed cover/redundancy annotation.
9. `layer_policy.py`: deterministic final/candidate policy.
10. `reporting.py`: output serialization + relation sidecar.
11. `runner.py`: orchestration only.
12. `cli.py`: I/O and execution mode.

## Public API / Interface

- `score_building_v2(...) -> BuildingResult`
- `score_corpus_v2(...) -> CorpusResult`
- Backward compatibility alias: `score_buildings_v2(...)`

## Migration And Compatibility Plan

1. Keep V1 script as entrypoint and add `--engine v1|v2|shadow`.
2. `shadow` runs both engines and writes side-by-side outputs + diff metadata.
3. Keep output path conventions stable; V2 writes the same core artifacts.
4. Adapt consumers to canonical fields only (`viewer`, `element_locator`, analysis scripts) before removing legacy extras.

## Open Risks

- Main-part inference from part membership count can be ambiguous in tied distributions.
- Relation thresholds may underfit unusual roof-part geometries.
- V2 may intentionally drop legacy heuristic artifacts that some debug flows relied on.

## Validation Plan

- Unit tests for:
  - anchoring metrics,
  - part attribution,
  - relation-kind classification,
  - final-layer policy constraints.
- Regression UUID checks for `117d...`, `c87c...`, `e015...`.
- Shadow diff on representative cohorts plus full-corpus smoke run.
- Compare schema and high-level counts between V1 outputs and V2 outputs.
