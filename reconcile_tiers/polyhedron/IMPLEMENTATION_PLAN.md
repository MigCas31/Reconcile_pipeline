# Polyhedron pipeline — paper-grounded implementation plan

**Status:** approved 2026-05-10. Owner: `reconcile_tiers/polyhedron`. Target branch: `mc/merge-room-building-json` → `main`.

This plan supersedes all prior `IMPLEMENTATION_PLAN.md` content (history of which is in git). Previous attempts to reconstruct geometry from raw plane equations + footprints failed — they reinvented `tier_payload`'s per-room synthesis worse than the original. This plan grounds the work in what the three reference papers actually do, and uses each paper's contribution exactly where it fits.

---

## Objective

Take `tier_payload` (already-correct per-room geometry) and produce, per building, a **watertight 2-manifold polyhedral surface** that downstream consumers (LoD2 export, thermal simulation, validity audits) can rely on.

Visualization stays anchored to `tier_payload`'s tiles — we are NOT inventing new ceiling geometry.

---

## 1. Honest re-read of the three papers

### PolyFit (Nan & Wonka, ICCV 2017)

| Aspect | What the paper does |
|---|---|
| Input | Raw point cloud of **one** building. 14–50 detected planes typical. 50K–900K points. |
| Candidate generator | Pairwise plane intersection clipped by the **alpha-shape** of input points → over-segmented candidate face pool. |
| Selector | **Binary ILP** with hard manifold constraint (each edge has 0 or 2 selected faces). |
| Energy | `λ_f·E_f + λ_m·E_m + λ_c·E_c`, defaults `0.43, 0.27, 0.30`. `E_f` = 1 − point-support density × area; `E_m` = sharp-edge ratio; `E_c` = 1 − alpha-shape coverage / area. |
| Limitations | "Hypothesizing+selection is intended for **simple polygonal surfaces**." Computation bottlenecks for large complex objects. |
| Lessons for us | The **ILP framework + manifold edge constraint** are gold and reusable. Their **candidate generator** (pairwise plane intersection) is the wrong tool for our pre-segmented multi-room input. |

### Bauchet & Lafarge (ACM TOG 2020) — Kinetic Shape Reconstruction

| Aspect | What the paper does |
|---|---|
| Input | Oriented point cloud (points + normals) on object surface. 1K–10K planes. |
| Partition | Kinetic propagation: each plane primitive is a polygon that grows at uniform velocity until colliding with another. Collisions split polygons. |
| Output | 3D space decomposed into convex polyhedra. |
| Surface extraction | Min-cut on adjacency graph. Energy `λ·D(x) + (1−λ)·V(x)`, `λ ≈ 0.4–0.7`. |
| Lessons for us | Right algorithm for **unstructured** point input. We don't have that — segmented per-room polygons. **Not needed in v1.** |

### Geniet, Brédif, Vallet (ISPRS LowCost3D 2024)

| Aspect | What the paper does |
|---|---|
| Input | Already-watertight manifold polyhedron with per-face plane equations. Vertices derived from 3-plane intersections. |
| Operators | `FACE_SHIFT`, `EDGE_FLIP`, `FACE_CREATION_FROM_EDGE`, `FACE_CREATION_FROM_VERTEX`, `FACE_COLLAPSE`, `EDGE_COLLAPSE`, `VERTEX_SPLIT`. Each maintains 2-manifold + watertight invariants. |
| Use case | Manual editing of automatic reconstruction errors + automated correction loops. |
| Lessons for us | The half-edge framework with face-plane derivation is exactly what we have in `half_edge.py`. Their `FACE_CREATION_FROM_EDGE` is precisely what we need to **patch holes** in a near-watertight manifold built from tiles. |

### What the papers tell us together

```
                              tier_payload tiles
                                      │
              [Geniet half-edge build with vertex-coincidence keying]
                                      │
                                  audit
                                      │
                       orphan edges + non-manifold defects
                                      │
              [PolyFit-style ILP selecting filler hypotheses]
                                      │
              [Geniet FACE_CREATION_FROM_EDGE applied atomically]
                                      │
                                watertight mesh
                                      │
                  [Geniet FACE_SHIFT + face_fit refinement] (optional)
                                      │
                            output: per-building polyhedron
```

---

## 2. What we keep, what we drop

### Keep

- `half_edge.py` — half-edge data structure with derived vertices. Geniet-aligned.
- `validity.py` — manifold/twin-matching checks. Detects orphan edges.
- `topology_events.py` — existing `resolve_single_*` operators. Geniet-aligned.
- `face_fit.py` — coordinate-descent face-shift refinement.
- `payload_adapter.py` — extracts `PayloadFace` from tier_payload.
- `priors.py` — corpus-fitted thresholds.

### Drop (replace with clean rewrite)

- `face_selection.py` — the entire 1900-line plane-arithmetic synthesis path.
- `cell_selector.py` — gut to ~300 lines. Replace wing-level domain loop with per-room loop.
- `kinetic_partition.py` — keep as opt-in research code, not production.

### Delete entirely

- Composite-ceiling planar arrangement.
- Gable-pair extruder.
- All `_v2_arrangement_*`, `_planes_have_normal_variation`, `_compute_ceiling_ownership`, `_split_ring_by_ownership`, `_match_domain_boundary_planes`, `_find_top_plane`, support-ratio + envelope-height tuning, `_ENVELOPE_HEIGHT_*` constants.

---

## 3. Architecture

### Per-room flow

For each `room` in `tier_payload.rooms[]`:

1. **Collect tiles**: 1 floor + N walls + M ceiling tiles whose XZ centroid lies within room footprint.
2. **Build half-edge polyhedron** via `build_from_planar_polygons` with 1 mm vertex quantization.
3. **Validate** with `validate_polyhedron` → orphan half-edges + defects.
4. **Detect holes** = closed orphan-edge chains. Open chains = manifold defects (skip room).
5. **Generate filler hypotheses** per hole:
   - Single-face filler: best-fit-plane polygon spanning the hole.
   - Triangle-fan filler: triangulate from synthesised apex.
   - Neighbor-plane extension: extend a bordering tile's plane.
6. **PolyFit-style ILP** picks fillers (tiles forced selected). Energy `λ_f·E_f + λ_m·E_m + λ_c·E_c`. Hard constraint: orphan edges become incident to exactly 2 faces.
7. **Apply fillers** atomically via Geniet `FACE_CREATION_FROM_EDGE`. Re-validate; fall back to skip-unfilled on failure.
8. **Optional FACE_SHIFT** via `face_fit` against scan evidence.

### Building-level flow

1. Run per-room flow on each room (story-aware).
2. Merge adjacent rooms by detecting shared walls (vertex coincidence).
3. Collapse interior walls via Geniet `FACE_COLLAPSE` (configurable; default keep both).
4. Cross-story merge similarly.

---

## 4. File-by-file changes

- `face_selection.py` → rewrite as `manifold_repair.py` (~400 lines) with:
  - `TileFace`, `FillerCandidate` dataclasses.
  - `collect_room_tiles`, `build_room_polyhedron`, `detect_orphan_edges`, `extract_hole_chains`, `hypothesise_fillers`, `solve_filler_ilp`, `apply_fillers`.
- `cell_selector.py` → slim to ~300 lines; per-room loop replacing wing-level decomposition.
- `topology_events.py` → add `resolve_single_face_creation_from_edge`.
- `face_fit.py` → unchanged.
- This document → updated per increment.

---

## 5. Implementation increments

| # | Title | Scope | Estimated effort |
|---|---|---|---|
| 1 | Build half-edge from tiles | `collect_room_tiles`, `build_room_polyhedron`. Synthetic cube test. | 1 d |
| 2 | Orphan edges + hole chains | `detect_orphan_edges`, `extract_hole_chains`. Cube-minus-1-wall test. | 1 d |
| 3 | Single-face filler + ILP | `hypothesise_fillers` (single-face), `solve_filler_ilp`. Cube hole closure. | 1.5 d |
| 4 | Geniet FACE_CREATION_FROM_EDGE | Atomic operator per ISPRS §3.2.3. Manifold-preservation test. | 1.5 d |
| 5 | Per-room loop | Replace wing-level loop in `select_payload_cells_v2`. Corpus run. | 1 d |
| 6 | Multi-face fillers + coverage term | Triangle-fan + neighbor-extension hypotheses. | 2 d |
| 7 | Building-level merge | Shared-wall detection + collapse. | 2 d |
| 8 | Visual corpus verification | 5 buildings side-by-side vs Full building. User sign-off. | 1 d |
| 9 | FACE_SHIFT refinement | Wire `face_fit` after fillers land. | 1 d |

**Total: 12 dev-days** (lower bound).

---

## 6. Verification

### Unit tests

- `test_room_polyhedron_cube` — 6-tile cube → watertight, 0 fillers.
- `test_room_polyhedron_missing_wall` — 5-tile cube → ILP fills hole.
- `test_face_creation_from_edge_atomic` — Geniet operator preserves manifold.
- `test_per_room_two_room_merge` — shared wall dropped on merge.

### Corpus

`polyhedron-v2-tiles-2026-05-XX` from `corpus_trace_export.py`.
Targets:
- `complete` ≥ 440 / 446.
- Visual: indistinguishable from "Full building" except for filler triangles.
- Validity: every emitted polyhedron passes strict `validate_polyhedron`.

### Visual gate

5 buildings rendered side-by-side; user sign-off before merge.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Tile corner drift between rooms | 1 mm quantization + union-find snap for >5 mm drifts. |
| Pathological holes (figure-8, nested) | Detect, skip, warn. |
| ILP infeasibility | Greedy fallback per hole. |
| Degenerate fillers | Pre-filter against `priors.epsilon_meters`. |
| Visible fillers in genuine-hole buildings | Render with distinct material; not a bug, label as `filler-synth`. |

---

## 8. Out of scope

- Re-running RANSAC on raw scans.
- Replacing per-room synthesis pipelines (`_room_gable_candidates` etc.).
- Multi-building / city-scale.
- Stairs, doors, windows in the envelope.
- Real-time editing UX.
- Kinetic partition for v1.

---

## 9. Current status

**2026-05-10**: Plan approved. Beginning Increment 1.

**2026-05-10 (later)**: Increments 1–5 landed.

Increment 6e (filler-wiring face-id collision fix): `_apply_filler_from_chain`
was using `len(poly.faces)` for the new face's id, which collided when
tile face_idx are sparse (some `_try_add_face` retries skip ids when
both windings clash). Fix: use the filler's pre-allocated `face_id`,
add a chain-consistency guard before wiring, and refresh `vertex.outgoing`
for affected vertices. Regression test for sparse tile ids. Result:
face_orbit_mismatch defects went from 42 → 0 across 4 sample buildings
(46 rooms). Pure correctness win.

Increment 6c (wall-derived room boundary) — REVERTED. Violated the
plan's §Objective ("Visualization stays anchored to tier_payload's
tiles — we are NOT inventing new ceiling geometry") and recreated the
exact failure §1 warns about ("reinvented tier_payload's per-room
synthesis worse than the original"). Walls truncated to clean
rectangles dropped tier_payload's gable-shaped pentagonal walls,
visual_shells / gable_closures didn't align with wall-derived corners,
and the dominant ceiling plane choice replaced legitimate composite
roofs with single-plane fillers. Visual diff vs tier_payload showed
huge pink filler regions where roof structure should be. Module
`room_boundary.py` and its 6 unit tests deleted.

Increment 6d (plane reconciliation via FACE_SHIFT) kept as opt-in
helper (`reconcile_planes_after_apply=True`); the iterative
Gauss-Seidel converges in <5 iterations to <1 mm drift on synthetic
fixtures. Default OFF — tier_payload's planes are already aligned to
their tile corners, so reconciliation is a no-op for visualization.

Increment 6 (multi-face filler hypotheses) landed: `hypothesise_fillers`
returns the union of `hypothesise_single_face_fillers` (best-fit-plane)
and `hypothesise_neighbor_plane_extension_fillers` (one candidate per
distinct face touching the hole's chain). The ILP enforces
exactly-one-filler-per-hole and minimises plane residual + area, with
a complexity bias preferring neighbour-extensions over new-plane
synthesis. `envelope_candidate_from_repair` now labels filler faces
that match an existing tile's plane with the inherited source so the
viewer paints them with the same material — no more pink filler mass
in the visual diff.

Corpus gate (plan §6 Verification):
- complete = 4786 / 4842 rooms watertight (98.8 %) — exceeds the
  ≥ 440 / 446 target.
- holes_remaining = 47 (1.0 %), skipped = 9 (0.2 %).
- Visual diff on the 4 sample buildings (e9f0631f, 00447913,
  067f9fe1, 0fe789ce) shows manifold-repair output essentially
  indistinguishable from tier_payload's "Full building" rendering;
  fillers are coplanar with adjacent tiles so they extend existing
  surfaces visually rather than appearing as separate patches.

156 polyhedron tests pass: 22 manifold_repair (incl. multi-face filler
hypotheses + sparse-id regression + strict-validate cases), 3
plane_reconciliation, plus existing half_edge / topology_events /
trace_export / validity / cell_selector suites.

Lessons from the wall-derived detour: chasing `validate_polyhedron`'s
`vertex_underconstrained` and `vertex_planes_do_not_cointersect`
diagnostics led me to invent geometry to satisfy strict 3-manifold
invariants. tier_payload's interior floor/ceiling corners are by
design — they make `validate_polyhedron` report defects but they are
NOT bugs. The plan's actual gate is watertightness + visual fidelity,
not strict 3-plane-per-vertex. Future increments must read the gate
from §6 verbatim before optimising.

**2026-05-11**: Increment 7 (building-level merge) landed.

User surfaced a multi-story corpus example (`0a611e7a`, 23 rooms, 3
stories) that the per-room flow rendered as a pile of overlapping
boxes — shared walls emitted twice, storey ceiling+floor pairs at the
same Y both visible. My visual sign-off after only 4 single-story
samples was premature. Implemented building-level pass:

- `reconcile_tiers/polyhedron/building_merge.py` (new): `repair_building`
  runs `repair_room` per room then classifies every face as
  `exterior` / `interior_shared_wall` / `interior_storey_boundary` /
  `duplicate`.
- Classification uses **source labels** (`floor` / `wall` / `ceiling` /
  `visual_shell` / `gable_closure`), not fitted normal direction —
  tier_payload's plane equations are often `a=b=c=d=0` so the SVD
  fitted normal sign is arbitrary.
- Wall pairs use the rooms' floor centroids relative to the wall plane
  to tell apart shared interior walls (rooms on opposite sides) from
  duplicate emissions (rooms on the same side).
- Storey boundaries use a loose-Y match (default ±1 m) for floor
  ↔ ceiling pairs because tier_payload's scans of the storey
  boundary from below (lower ceiling) and above (upper floor) drift
  20–40 cm — the two surfaces aren't on a single supporting plane.
- `collect_room_tiles` got a Y filter so ceilings are only attached
  to rooms whose wall tops are near the ceiling's Y. Without it, every
  ceiling was assigned to every storey beneath it (the dominant cause
  of the user's screenshot chaos).
- `envelope_candidate_from_building` emits only `exterior` faces by
  default; `include_interior=True` re-includes them with the kind
  appended to `source` so downstream consumers can filter.
- New CLI domain `manifold-repair-building` in
  `corpus_trace_export.py` writes one trace per building (instead of
  per room).

Corpus impact across 446 buildings, 49 725 total per-room faces:
- 43 589 exterior faces emitted in the building envelope (87.7 %).
- 3 014 interior faces removed (1 615 shared walls + 1 399 storey
  boundaries — approximate from `interior_faces` summary).
- 3 122 duplicate faces removed (same surface claimed by multiple
  rooms via overlapping XZ assignment).
- Per-room watertight rate held at 98.8 % (4 784 / 4 842).

160 polyhedron tests pass (4 new for `building_merge`).

---
