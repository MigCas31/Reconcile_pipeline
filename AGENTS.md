# AGENTS.md

AI guidance for the tirana (look-ma-no-hands) repo.

## Quick Context

3D building geometry reconciliation toolkit for **Lun Energy / Plans**. Building surveyors 3D-scan homes with an iOS app (Apple RoomPlan), this repo processes multi-session scan data into coherent building models with roof detection, topology graphs, and 3D visualization.

## Tech Stack

| Technology | Usage |
|-----------|-------|
| Python 3.11+ | All processing — extraction, reconciliation, roof pipeline, topology |
| numpy | Geometry operations, vector math, transforms |
| dataclasses | All domain models (Vec3, Transform, GraphNode, etc.) |
| Shapely v2 | 2D polygon union, coplanar stitching (reconcile_v2) |
| Three.js | 3D building viewer (vanilla JS, no framework) |
| MapLibre | Orthophoto satellite overlay in viewer |

## Commands

| Command | Purpose |
|---------|---------|
| `python -m pytest tests/` | Run tests |
| `python reconcile/extract_3d.py` | V1 3D extraction pipeline |
| `python -m reconcile_v2.cli` | V2 topology pipeline |
| `python -m reconcile_tiers.server` | Tier viewer on :8080 |

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `reconcile/` | V1 extraction — extract_3d, builder, cross-floor gaps |
| `reconcile/extract3d/` | Modular extraction (ceilings, exterior, gaps, overlaps, stitch) |
| `reconcile/roof_algorithms_py/` | 9-step roof detection pipeline |
| `reconcile/viewer-modules/` | Three.js viewer JS modules |
| `reconcile_v2/` | V2 topology — graph adjacency, wall thickness, IFC mapping |
| `pipeline-outputs/` | ~200 UUID-named building output directories |
| `tests/` | pytest tests |

## Domain Model

| Entity | Description |
|--------|-------------|
| Building | Top-level container with stories and metadata |
| Story | Floor/level of a building |
| Room | Individual space with walls and floor polygon |
| Wall | Wall element with corner polygons and extension strips |
| Segment | Section of wall with azimuth, inclination, endpoints |
| Vec3 | x, y, z coordinate (dataclass in `reconcile/models.py`) |
| Transform | 4x4 matrix for coordinate transforms |
| TopologyGraph | Graph of GraphNode + GraphEdge (reconcile_v2) |

## Element IDs (Shareable Locators)

Every rendered element has a shareable ID: `<building_uuid>::<kind>::<id>`. Right-click any element in the viewer to copy its ID. Paste into the search bar to jump back to it.

**When you receive an element ID** (e.g. `a6cb04fa-e84a-4641-a667-b4dd05dd7d41::floor::0:4`):
1. Parse it: `parse_element_id(token)` returns `(building_uuid, kind, element_id)`
2. Resolve it: `find_element(buildings, token)` returns the element dict with corners, json_path, story, etc.
3. CLI: `python -m reconcile.element_locator --element-id "<token>"`

| Kind | ID format | Scope |
|------|-----------|-------|
| `wall-merged` | wall id | room |
| `wall-computed` | wall id | room |
| `wall-extension` | wall id | room |
| `wall-clipped-original` | wall id | room |
| `door` | door id | room |
| `window` | window id | room |
| `floor` | `<story>:<room_index>` | room |
| `floor-overlap` | `<story>:<room_index>` | room |
| `gap-cross-story` | gap id | building |
| `gap-within-story` | gap id | building |
| `wall-stitch` | stitch id | building |
| `gap-wall` | gap wall id | building |
| `exterior-gap-element` | indicator id | building |
| `exterior-gap-wall` | indicator id | building |
| `gap-closure` | closure id | building |
| `roof-oblique` | `oblique:<index>` | building |
| `roof-flat` | `flat:<index>` | building |
| `ceiling-flat` | `ceiling-flat:<index>` | building |
| `ceiling-oblique` | `ceiling-oblique:<index>` | building |
| `ceiling-simple-slant` | `ceiling-slant:<index>` | building |
| `ceiling-raw` | `<story>:<room_index>:<plane_index>` | room |

**Key files**: `reconcile/element_locator.py` (backend), `reconcile/viewer-main.js` (frontend: `makeElementUid`, `parseElementUid`, `attachLocator`, `selectElementByUid`)

## Think in Buildings, Not Just Code

This codebase models **real physical buildings**. Every data structure maps to something you can touch. Code decisions should be grounded in how buildings actually work.

### Physical Reality Drives the Code

- **Gravity is vertical (Y-up)** — floors are horizontal, walls are vertical (or near-vertical). Roofs slope to shed water. If your geometry produces a wall that leans 45 degrees or a floor that isn't flat, something is wrong.
- **Buildings have stories** — each story has a floor, walls, and a ceiling. The ceiling of one story is (approximately) the floor of the next. Cross-floor gaps happen because scans don't align perfectly.
- **Walls have thickness** — they're not infinitely thin planes. Two adjacent rooms share a wall, and the gap between their floor edges reveals the wall thickness. This is what `wall_thickness_inference.py` measures.
- **Roofs exist because of weather** — they slope (oblique) to drain rain, or they're flat. Roof geometry follows from structural engineering: ridge lines, hip lines, valleys. Our roof pipeline detects these from the inclination and azimuth of scanned wall segments.
- **Rooms are bounded volumes** — a room has a floor polygon, walls around the perimeter, and a ceiling above. When we say "exposed room" we mean a room with no floor above it (i.e., it's on the top story or has a roof above).
- **Buildings sit on the ground** — the building footprint is the 2D outline where the building meets the earth. We derive this from the convex hull of exposed room floor polygons.
- **Doors and windows are holes in walls** — they have positions and dimensions within the wall plane. They affect thermal calculations downstream.

### Why This Matters for Code

When you encounter a geometry bug or design decision, ask: *"What would this look like in a real building?"*

- If a roof surface extends beyond the building footprint → the clipping failed (roofs don't float in air)
- If two rooms overlap in 3D space → the stitching has a bug (rooms don't physically overlap)
- If wall thickness is negative → the adjacency inference is wrong (walls have positive thickness)
- If a ceiling plane is below the floor → the height cap computation failed (ceilings are above floors)
- If the azimuth of a roof segment doesn't match the building orientation → check coordinate transforms (buildings face consistent directions)

### Human-First Troubleshooting

When debugging why an element was classified as X instead of Y, or why it was included/excluded from a set, do **not** jump straight to threshold tweaking or heuristic patching. Start with the human-visible reality:

- Ask what a human sees immediately that the algorithm missed: shape, support, height, continuity, adjacency, footprint containment, roof coverage, or room/story context.
- State the physical reason the element "obviously" belongs to class X or not-Y before inspecting the code path.
- Use heuristics as things to audit, not truths to trust. The goal is to explain the gap between human perception and machine classification.
- Prefer fixes that model the missing physical signal or decision boundary more directly, rather than stacking another special case on top.

### Architectural Vocabulary

Use these terms correctly — they map to specific code concepts:

| Term | Physical meaning | Code location |
|------|-----------------|---------------|
| Story/storey | A horizontal level (ground floor, 1st floor, etc.) | `story_index.py`, Story in models |
| Footprint | 2D outline of building at ground level | `footprint_derivation.py` |
| Ridge | Highest line where two roof slopes meet | `clip_poly_by_ridge()` |
| Hip | Sloped edge where two roof faces meet at an angle | Detected via oblique clustering |
| Eave | Lower edge of roof where it overhangs the wall | Bottom of oblique surface candidates |
| Azimuth | Compass direction a surface faces (0-360 degrees) | Segment azimuth field |
| Inclination | Angle from horizontal (0 = flat, 90 = vertical) | Segment incl field |
| Oblique | Neither horizontal nor vertical (a sloped surface) | 5 deg < incl < 80 deg filter |
| Adjacent | Rooms that share a wall | `infer_intra_story_adjacency()` |
| IFC | Industry Foundation Classes — the ISO standard for BIM data exchange | `ifc_mapping.py` |

## How to Work in This Codebase

### Research Before Implementing

This is a **computational geometry** codebase — algorithms here have known correct solutions in academic literature and open-source libraries. Before writing new geometry code:

1. **Search online** for how others have solved the problem (Sutherland-Hodgman, convex hull algorithms, plane intersection, etc.). Don't reinvent.
2. **Check numpy/Shapely docs** — many operations already exist as library functions.
3. **Read the existing code first** — we likely already have a utility for what you need (check `math_utils.py`, `models.py`).
4. **Check the calor backend** (`../calor` or `github.com/lun-energy/calor`) — the Go backend may have a reference implementation.
5. **Check web-main** (`.context/web-main-latest/`) — the TypeScript frontend may have a parallel implementation to stay consistent with.

### Code Principles

- **Models**: dataclasses, never plain dicts for domain objects
- **Geometry**: numpy for vector math, Y-up coordinate system
- **Type hints**: preferred on all public functions
- **Dependencies**: minimal — numpy, Shapely, jsonschema. No heavy frameworks.
- **Correctness over cleverness** — geometry bugs are subtle and hard to catch visually. Prefer well-known algorithms with references over custom solutions.
- **Test with real buildings** — run the viewer on pipeline-outputs/ to visually verify geometry changes. Numbers alone don't catch spatial bugs.

### When Modifying Algorithms

- **Understand the full pipeline** before changing a step — changes cascade (see roof-pipeline skill)
- **Don't change thresholds** unless you've tested on multiple buildings from `pipeline-outputs/`
- **Keep parity** with calor/web-main where both codebases implement the same logic (e.g., grid convergence)
- **Visualize your changes** — use the viewer to confirm geometry looks correct, not just that tests pass

## Progress Tracking

Every time you modify files in `reconcile/`, `reconcile_v2/`, or `reconcile_v3/`:

1. **Update `tracking_progress.md`** at the repo root with a new entry
2. Each entry must include:
   - **Date** (YYYY-MM-DD)
   - **What changed** — which files/modules were modified and what the change does
   - **Why** — the motivation: what problem it solves, what hypothesis it tests, or what goal it advances
   - **Result** — outcome if known (test results, corpus metrics, visual verification)
3. Append to the end of the file (chronological order, newest last)
4. Include enough detail that a reader can understand the problem, what was tried, and why — document failed approaches and pivots, not just successes

## Gotchas

**CRITICAL**: Azimuth filtering uses **180** degree threshold, NOT 90 degrees. The 90-degree range caused false clips in production.

- `DATAFORDELEREN_API_KEY` env var required for viewer orthophotos (fallback: GCP Secret Manager)
- Coordinate systems: UTM32N (EPSG:25832) for metric, WGS84 for GPS. Always apply grid convergence.
- `reconcile/cli_v2.py` is just a shim — delegates to `reconcile_v2.cli`
- Roof pipeline steps MUST run in order — each depends on previous output

## Skills

| Skill | When to use |
|-------|-------------|
| [Geometry](/.agents/skills/geometry/SKILL.md) | Vec3, transforms, coordinate systems, numpy math |
| [Extraction Pipeline](/.agents/skills/extraction-pipeline/SKILL.md) | extract_3d, builder pattern, cross-floor gaps |
| [Roof Pipeline](/.agents/skills/roof-pipeline/SKILL.md) | Roof detection, ceiling planes, oblique surfaces |
| [Viewer](/.agents/skills/viewer/SKILL.md) | Three.js viewer, orthophotos, MapLibre |
| [Topology V2](/.agents/skills/topology-v2/SKILL.md) | Graph topology, wall thickness, IFC mapping |
| [Testing](/.agents/skills/testing/SKILL.md) | pytest, fixtures, test coverage |
| [Danish Geodata](/.agents/skills/danish-geodata/SKILL.md) | Datafordeler API, building footprints, WMTS |
| [Run & Verify](/.agents/skills/run-and-verify/SKILL.md) | End-to-end pipeline runs, server restart, browser verification |
| [Debug Element](/.agents/skills/debug-element/SKILL.md) | Triage a shareable element ID (legacy, ontology-*, or tier-*), trace to pipeline + thresholds, propose root cause |
| [Inspect Building](/.agents/skills/inspect-building/SKILL.md) | **Run first** when the user reports an issue with a building or element (anything wrong/weird/missing/broken) or asks for the full picture of a UUID. Builds a one-folder report with screenshots, LoD2 realism flags, and tier_payload defect audit so you have context before forming hypotheses. Hand off to Debug Element for single-element root cause. |

### Using Skills

Before writing code that matches a skill's domain:

1. **Read the skill file** and any linked source files
2. **Apply the patterns exactly** — don't improvise when a pattern exists
3. **Check constants and thresholds** — many have been tuned through production use

## References

| Resource | When to consult |
|----------|----------------|
| `../calor` / [github.com/lun-energy/calor](https://github.com/lun-energy/calor) | Backend reference implementations, API shapes |
| `.context/web-main-latest/` | Frontend TypeScript parallel implementations |
| [Three.js docs](https://threejs.org/docs/) | Viewer development — check docs before writing custom geometry/material code |
| [MapLibre docs](https://maplibre.org/maplibre-gl-js/docs/) | Map overlay development |
| [Shapely docs](https://shapely.readthedocs.io/) | Polygon operations — check if Shapely already has what you need |
| [numpy docs](https://numpy.org/doc/) | Vector/matrix operations |
| [Datafordeler API docs](https://datafordeler.dk/) | Danish geodata API reference |
| `pipeline-outputs/` | Real building data for visual testing |

---

## Universal Coding Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Source: [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills), derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876). These apply on top of the project-specific guidance above.

**Tradeoff:** these bias toward caution over speed. For trivial tasks (typo fixes, obvious one-liners), use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
