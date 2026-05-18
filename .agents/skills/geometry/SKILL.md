---
name: building-geometry
description: >
  Use when working with Vec3, Transform, coordinate transforms (UTM/WGS84),
  geometric math, numpy vector operations, point-in-polygon tests, plane
  equations, or clipping algorithms in this codebase.
---

# Building Geometry

## How to Work with Geometry

1. **Search online first** — most geometry problems (clipping, intersection, projection, hull) have well-known algorithms. Search for the algorithm name + "python" or "numpy" before writing custom code.
2. **Check `math_utils.py` before writing new utility functions** — we likely already have what you need. Don't duplicate.
3. **Check Shapely** — for 2D polygon operations (union, intersection, buffer, contains), Shapely almost always has a correct, tested implementation. Use it instead of rolling your own.
4. **Check numpy** — vectorized operations are preferred over Python loops for geometry. Search numpy docs for the operation you need.
5. **Validate visually** — use the viewer (`viewer_server.py`) to check geometry output. Numerical correctness alone doesn't guarantee spatial correctness.
6. **Keep parity with web-main** — `grid_convergence.py` was ported from TypeScript. If you change it here, the same change must happen in web-main (or verify they still agree via `tests/test_grid_convergence.py`).

## Core Types

Defined in `reconcile/models.py`:

| Type | Fields | Purpose |
|------|--------|---------|
| `Vec3` | x, y, z | 3D coordinate (dataclass) |
| `Transform` | 4x4 matrix | Coordinate transform |

Coordinate convention: **Y-up** (Y is vertical height).

## Coordinate Systems

| System | EPSG | Usage |
|--------|------|-------|
| UTM32N | 25832 | Metric calculations, Datafordeler API queries |
| WGS84 | 4326 | GPS input, lat/lon display |

Grid convergence (`reconcile/grid_convergence.py`):
- `compute_grid_convergence_rad(lat, lon, projection)` — true north to grid north angle
- `GridNorthReference` dataclass — stores convergence result
- Supports UTM (Denmark) and Lambert (France) projections
- Ported from web-main's `north-reference.ts`

## Math Utilities

All in `reconcile/roof_algorithms_py/math_utils.py`:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `angle_diff` | `(a, b) -> float` | Shortest angular distance mod 360 degrees |
| `point_in_poly_xz` | `(px, pz, poly) -> bool` | Ray-casting point-in-polygon on XZ plane |
| `point_in_poly_2d` | `(px, pz, poly) -> bool` | Ray-casting for Point2 tuples |
| `plane_normal` | `(azimuth_deg, incl_deg) -> dict` | Normal vector {x, y, z} from angles |
| `seg_midpoint` | `(seg) -> dict` | 3D midpoint of segment with keys "a", "b" |
| `plane_y_at` | `(plane, x, z) -> float` | Evaluate plane equation: y = ref.y - (n.x*(x-ref.x) + n.z*(z-ref.z))/n.y |
| `clip_by_max_y` | `(poly, max_y) -> list[Point3]` | Sutherland-Hodgman clip by Y ceiling |
| `clip_by_half_plane_xz` | `(poly, ox, oz, nx, nz) -> list[Point3]` | Half-plane clip in 3D using XZ normal |
| `clip_poly_by_ridge` | `(poly, r_dir_x, r_dir_z, ref_x, ref_z, bound, keep_above) -> list[Point2]` | Ridge-line clip in 2D |
| `clip_poly_by_half_plane_2d` | `(poly, dx, dz, offset) -> list[Point2]` | Half-plane clip in 2D |
| `convex_hull_2d` | `(points) -> list[dict]` | Andrew's monotone chain on {x, z} dicts |
| `point_near_footprint` | `(px, pz, footprint, margin, fn) -> bool` | Point within polygon or within margin of edges |

## Type Aliases

```python
Point2 = tuple[float, float]  # (x, z) in XZ plane
Point3 = tuple[float, float, float]  # (x, y, z)
```

## Physical Conventions

- **Y-up**: Y is the vertical axis (height). X and Z form the horizontal ground plane. This matches architectural convention where "up" matters most.
- **Azimuth**: measured in degrees, 0-360, representing compass direction. A wall facing north has azimuth ~0/360. A wall facing east has azimuth ~90. Two walls can face opposite directions (differ by ~180 degrees) and still be parallel — this is why we use a 180-degree filter, not 90.
- **Inclination**: angle from horizontal. 0 = perfectly flat (floor/ceiling). 90 = perfectly vertical (wall). Between 5-80 degrees = oblique (roof slope).
- **Segments have two endpoints** (a, b) where a is higher (greater Y). The segment's azimuth is the direction the surface faces, not the direction the segment runs.

## CRITICAL: Azimuth Rule

**NEVER** use 90 degrees for azimuth filtering. The correct threshold is **180 degrees**.

The 90-degree range was tested and caused false clips in production. This applies to any code that filters or compares azimuth angles.

## Anti-Patterns

| Wrong | Correct | Why |
|-------|---------|-----|
| Euclidean distance on lat/lon | Convert to UTM first | Lat/lon are angular, not metric |
| 90-degree azimuth filter | 180-degree azimuth filter | 90 degrees causes false clips |
| Skip grid convergence for small areas | Always apply correction | Error accumulates at building scale |
| Modify clipping polygon in-place | Return new polygon | Sutherland-Hodgman produces new vertex list |
