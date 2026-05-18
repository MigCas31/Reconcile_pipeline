---
name: roof-pipeline
description: >
  Use when working on roof detection, ceiling planes, oblique surface generation,
  segment clustering, or any file in reconcile/roof_algorithms_py/.
---

# Roof Detection Pipeline

## How to Work on Roofs

1. **Understand the full 9-step pipeline before changing anything** — this is the most sensitive code in the repo. Every step feeds the next.
2. **Search academic literature** — roof detection from LiDAR/3D scans is a well-studied problem. Search for "roof plane detection", "RANSAC roof segmentation", "building roof reconstruction" before inventing new approaches.
3. **Never change thresholds blindly** — every constant (MIN_SEG_LEN, COPLANAR_TOL, etc.) was tuned against real Danish buildings. Test on at least 5 buildings from `pipeline-outputs/` before changing a value.
4. **Visualize intermediate results** — the pipeline returns all intermediate data. Use the viewer's `renderRoofFromPythonResult()` to see what each step produces. If you only check the final output, you'll miss where things went wrong.
5. **Check the viewer's roof rendering** — `viewer-modules/roof-python.js` shows how roof output is consumed. Changes to output format will break the viewer.

## Physical Intuition

Roofs exist to protect buildings from weather. Their geometry follows structural and drainage constraints:

- **Oblique (sloped) roofs** shed water. Each face has an azimuth (compass direction it faces) and an inclination (steepness). Two faces meeting at the top form a **ridge**. Two faces meeting at a sloped edge form a **hip**.
- **Flat roofs** are nearly horizontal (inclination < 5 degrees). They still have slight slopes for drainage but we treat them as flat.
- **The pipeline detects roofs from wall scans** — RoomPlan captures wall geometry including sloped attic walls. We cluster these by orientation to find roof planes, then clip them to the building's footprint.
- **Why segments, not surfaces**: the scanner gives us wall segments (edges with azimuth + inclination), not ready-made roof surfaces. We must reconstruct surfaces by grouping co-planar segments and extending them.
- **Why clipping matters**: raw roof planes extend infinitely. We clip them by the building footprint (roofs don't extend beyond walls), by height caps (roofs don't go below floors), and by opposing planes (where two roof faces meet, one must stop).

When debugging: if a roof looks physically impossible (floating in air, extending underground, passing through walls), the bug is in the clipping or plane generation, not the clustering.

## Golden Rules

1. Pipeline steps MUST run in order — each depends on previous output
2. Never change filtering thresholds without understanding the cascade effect
3. Clipping is 4-stage — skipping a stage causes geometry artifacts
4. The 180-degree azimuth filter is intentional — do NOT reduce to 90 degrees

## Entry Point

`reconcile/roof_algorithms_py/pipeline.py` -> `run_roof_algorithms(bldg)` returns dict with all intermediate results and final surfaces.

Exported from `__init__.py`: `run_roof_algorithms`, `ROOF_PIPELINE_STEPS`.

## 9-Step Pipeline

| Step | File | Function | Purpose |
|------|------|----------|---------|
| 1 | `story_index.py` | `compute_building_y_bounds()` | Story hierarchy + building Y bounds |
| 2 | `segment_collection.py` | `collect_oblique_segments()` | Extract oblique wall segments (5 deg < incl < 80 deg, len >= 0.3) |
| 3 | `oblique_clustering.py` | `cluster_oblique_segments()` | Group by circular mean azimuth/inclination |
| 4 | `oblique_surface_generation.py` | `build_oblique_roof_surfaces()` | Generate candidate planes from clusters |
| 5 | `flat_surface_generation.py` | `build_flat_roof_surfaces()` | Flat roofs from exposed rooms (intermediate + top story) |
| 6 | `ceiling_plane_generation.py` | `collect_exposed_rooms()` + `build_ceiling_planes()` | Identify exposed rooms, build ceiling plane definitions |
| 7 | `footprint_derivation.py` | `build_building_footprint()` | Convex hull from exposed room floor polygons |
| 8 | `ceiling_plane_clipping.py` | `clip_ceiling_planes()` | 4-stage clipping orchestrator |
| 9 | `ceiling_surface_generation.py` | `build_flat_ceilings()` + `build_oblique_ceilings()` | Final ceiling/roof geometry |

## Critical Constants

| Constant | Value | File | Purpose |
|----------|-------|------|---------|
| MIN_SEG_LEN | 0.3 | segment_collection.py | Minimum segment length (meters) |
| COPLANAR_TOL | 0.5 deg | oblique_clustering.py | Angular tolerance for clustering |
| MIN_CLUSTER_SIZE | 2 | oblique_clustering.py | Minimum segments per cluster |
| PLANE_HEIGHT | 10.0 | roof_oblique_candidates.py | Vertical extent of candidate plane |
| Y-cluster tolerance | 0.15 | roof_flat_geometry.py | Flat segment Y-value grouping |
| Flat ceiling variance | 0.3 | ceiling_surface_generation.py | Max wallTopY - wallTopMin for flat |
| Min ridge span | 2.0 | roof_oblique_candidates.py | Below this -> skip candidate |
| Min bbox dimension | 2.0 | roof_flat_intermediate.py | Below this -> skip flat surface |
| Inclination range | 5 deg - 80 deg | segment_collection.py | Oblique wall filter |

## 4-Stage Clipping (Step 8)

`clip_ceiling_planes()` orchestrates:

1. **`build_initial_plane_clips()`** — Clip each plane's 2D footprint to building footprint boundary
2. **`build_junction_patches()`** — Generate perpendicular intersection patches where oblique planes meet at ridges
3. **`compute_plane_height_caps()`** — Compute max Y from walls above, building envelope, opposing planes
4. **`apply_opposing_plane_cuts()`** — Mutual clipping of opposite-facing planes. **Modifies `plane_clipped` in-place.**

## Filtering Cascade

Segments pass through multiple filters. Understanding this cascade is essential for debugging:

```
all segments
  -> inclination filter (5 deg - 80 deg)
  -> length filter (>= 0.3)
  -> floor-above filter (skip if floor above)
  -> cluster (circular mean, 0.5 deg tolerance)
  -> cluster size filter (>= 2)
  -> ridge span filter (>= 2.0)
  -> candidate generation
  -> corner count filter (>= 3 corners)
  -> story clipping
  -> corner count filter again (>= 3)
  -> final surfaces
```

## Key Algorithms

- **Oblique clustering**: Uses sin/cos accumulation for circular mean of azimuth angles (handles 0/360 wraparound)
- **Candidate generation**: Projects cluster center + slope/ridge vectors into 3D plane, clips by building bounds
- **Convex hull**: Andrew's monotone chain on {x, z} points for building footprint
- **Ceiling clipping**: Sutherland-Hodgman variants for half-plane, ridge-line, and max-Y clipping

## Roof Element IDs

Roof and ceiling surfaces have shareable element IDs (right-click in viewer to copy). Roof data is embedded in each building dict in `buildings_3d.json` under `roof_surfaces` and `ceiling` keys.

| Kind | ID format | Data path |
|------|-----------|-----------|
| `roof-oblique` | `oblique:<index>` | `roof_surfaces.oblique[i]` |
| `roof-flat` | `flat:<index>` | `roof_surfaces.flat[i]` |
| `ceiling-flat` | `ceiling-flat:<index>` | `ceiling.flat[i]` |
| `ceiling-oblique` | `ceiling-oblique:<index>` | `ceiling.oblique[i]` |
| `ceiling-simple-slant` | `ceiling-slant:<index>` | `ceiling.simple_slant[i]` |

Resolve with: `python -m reconcile.element_locator --element-id "<uuid>::roof-oblique::oblique:0"`

Index-based IDs are stable within a single pipeline run but shift if surfaces are added/removed.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No roof surfaces generated | All segments filtered out | Check inclination range, segment lengths |
| Missing roof faces | Cluster too small (< 2 segments) | Check MIN_CLUSTER_SIZE, COPLANAR_TOL |
| Roof extends beyond building | Footprint clipping failed | Check `build_building_footprint()` output |
| Intersecting ceiling planes | Opposing cuts incomplete | Verify `apply_opposing_plane_cuts()` ran |
| Flat roof missing | Room marked as having floor above | Check `has_floor_above` closure |
