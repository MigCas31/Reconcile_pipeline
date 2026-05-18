# Roof Algorithms Inspection + Python Conversion Task List

## Goal
Convert the modular roof/ceiling pipeline currently implemented in JavaScript (`reconcile/roof-algorithms/*.js`) into Python with explicit, small, reorderable steps.

## Algorithm Inspection Summary

### 1) Story/floor indexing
- Build per-story floor polygon lookup (`floors_by_story`).
- Compute ordered story list (`all_stories`).
- Provide `has_floor_above(x, z, story)` using point-in-polygon in XZ.
- Compute story metadata:
  - `story_floor_polys`
  - `story_floor_y` (min floor Y per story)
  - `story_max_y` (max wall-top Y per story)
- Compute building Y bounds (`bldg_min_y`, `bldg_max_y`).

### 2) Oblique roof segment extraction
- Read wall polygons (`corners`) and extension strips as candidate quads.
- Extract edge segments from each quad.
- Keep sloped edges only (`5° < inclination < 80°`).
- Remove short edges (`len < 0.3m`) and edges with floor above.

### 3) Oblique cluster detection
- Greedy clustering by:
  - azimuth similarity (< 30°)
  - inclination similarity (< 15°)
  - coplanarity tolerance (`0.5m` projected distance to cluster plane)
- Keep clusters with at least 2 segments.

### 4) Oblique roof plane synthesis
- For each valid oblique cluster:
  - derive slope/ridge directions from azimuth/inclination
  - compute ridge extent from segment projections
  - skip narrow planes (`ridge span < 2m`)
  - construct initial 3D quad over slope direction
  - clip by max Y (`bldg_max_y`)
  - clip against same-story floor polygons when roof cuts below floor (dormer/bay guard)
- Output polygons and source segments (rendering-independent).

### 5) Flat roof surface synthesis
- Intermediate exposed floors:
  - rooms with no floor above
  - width threshold (`>= 2m` in X or Z)
  - create padded bbox flat surfaces
- Top-story wall-top flat edges:
  - detect near-horizontal wall-top edges
  - cluster by height (`0.15m` tolerance)
  - keep clusters with >=2 segments
  - create padded bbox flat surfaces

### 6) Ceiling plane construction
- Build exposed rooms (no floor above).
- Flat ceiling condition:
  - wall-top spread < `0.3m`
- Build oblique ceiling plane descriptors from oblique clusters:
  - normal
  - reference point
  - ridge/slope axes
  - ridge/slope extents
  - dominant story

### 7) Building footprint selection
- Compute top exposed story.
- Compare top vs lower stories (down to 2 levels) to detect footprint overhang.
- Build convex hull footprint in XZ using selected points.

### 8) Ceiling clipping passes
- Ridge clipping to plane extent.
- Ridge expansion pass 1:
  - absorb adjacent/overlapping flat ceiling polygons.
- Ridge expansion pass 2 (gable-gap repair):
  - if plane covers >=60% footprint span, borrow adjacent rooms from opposing gable plane (max 2 rooms).
- L-junction synthesis:
  - detect near-perpendicular plane pairs
  - test ridge-line intersection near footprint
  - synthesize extension patches clipped by valley half-plane
- Upper-story cap pass:
  - compute per-plane max Y cap from overlapping upper floors/walls
- Opposing gable cross-plane clip:
  - clip plane pairs with azimuth difference ~180°.

### 9) Final ceiling polygons
- Flat ceilings per exposed room.
- Oblique ceilings from clipped 2D polygons + plane Y evaluation.
- Floor-level clipping (cannot drop below own story floor).
- Upper-story cap clipping.
- Add synthesized L-junction patches.

## Python Conversion Task List

- [ ] Create Python package directory for roof algorithms.
- [ ] Add shared geometry/math utilities (angle diff, polygon tests, clipping, hull, plane math).
- [ ] Port story index + building/stories stats step.
- [ ] Port segment extraction step.
- [ ] Port oblique clustering step.
- [ ] Port oblique roof plane polygon generation step.
- [ ] Port flat roof surface generation step.
- [ ] Port exposed-room and ceiling-plane derivation step.
- [ ] Port building footprint derivation step.
- [ ] Port ceiling clipping passes (ridge expansion, gable repair, L-junction, caps, cross-plane).
- [ ] Port final ceiling polygon generation step.
- [ ] Add orchestrator `run_roof_algorithms(building)` returning serializable outputs.
- [ ] Add `__init__.py` exports.
- [ ] Run syntax/compile checks for all new Python files.
