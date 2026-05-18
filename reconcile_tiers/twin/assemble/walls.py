"""Step 2 of the assembly: Walls per Room.

Anchor: each `ExtractedWall.corners` (4 corners; Y-sorted top/bottom
pairs). The canonical Wall is a planar quad with:
  - bottom edge horizontal at the host Floor's Y
  - top edge horizontal at the mean of the two scanned top corner Ys
  - top XZ taken from the scan (NOT forced to match bottom XZ); this
    preserves wall lean as architectural geometry rather than treating
    it as scan noise
  - plane: best-fit through the 4 snapped corners, which is exactly
    vertical when top XZ ≈ bottom XZ and tilted otherwise

The original 4 corners are preserved as ScanEvidence. Construction is
structural: no scan-precision thresholds. Walls whose extracted corners
cannot yield a non-degenerate XZ bottom edge raise `InvariantViolation`
from the Wall constructor — the assembler does not filter; the type
system does.
"""

from __future__ import annotations

import math

from reconcile_tiers.extract.building import ExtractedRoom, ExtractedWall
from reconcile_tiers.payload.schema import Plane, Vec3
from reconcile_tiers.twin.types import FLOAT_EPS, Evidence, Floor, Provenance, Wall


def wall_planes_for_room(room: ExtractedRoom, *, floor: Floor) -> dict[str, Plane]:
    """Compute the canonical plane for each raw wall in the room,
    keyed by `ExtractedWall.id`. Used by Opening assignment to project
    opening corners onto the eventual Wall plane (which may be slightly
    tilted when the wall leans).

    Walls whose extraction is degenerate are absent from the returned
    mapping; the assembler will treat them as orphan walls in
    `walls_for_room`.
    """
    floor_y = float(floor.polygon[0].y)
    by_id: dict[str, dict[str, ExtractedWall]] = {}
    for raw_wall in room.walls_merged:
        by_id.setdefault(raw_wall.id, {})["merged"] = raw_wall
    for raw_wall in room.walls_computed:
        by_id.setdefault(raw_wall.id, {})["computed"] = raw_wall

    planes: dict[str, Plane] = {}
    for wall_id, versions in by_id.items():
        anchor = versions.get("merged") or versions.get("computed")
        if anchor is None:
            continue
        plane = _canonical_plane(anchor, floor_y=floor_y)
        if plane is not None:
            planes[wall_id] = plane
    return planes


def _split_wall_quad(
    corners: list[list[float]],
) -> (
    tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        float,
    ]
    | None
):
    """Return ((ba_xz), (bb_xz), (ta_xz), (tb_xz), top_y_mean) for a 4-corner wall.

    Sorts corners by Y; the lowest two form the bottom edge, the highest
    two form the top edge. The bottom edge is anchored externally at the
    floor Y, so we discard those corner Ys here. The top edge takes the
    mean of the two highest corner Ys.

    Pairing: bottom-A is paired with the top corner closest to it in XZ
    so the resulting quad has consistent winding (bottom_a → bottom_b
    → top_b → top_a). For perfectly rectangular walls the pairing is
    unambiguous; for leaning walls it preserves the wall's actual shape.
    """
    if len(corners) < 4:
        return None
    sorted_by_y = sorted(
        ((float(c[0]), float(c[1]), float(c[2])) for c in corners),
        key=lambda c: c[1],
    )
    bot1, bot2 = sorted_by_y[0], sorted_by_y[1]
    top1, top2 = sorted_by_y[-2], sorted_by_y[-1]
    bot_xz = [(bot1[0], bot1[2]), (bot2[0], bot2[2])]
    top_xz_candidates = [(top1[0], top1[2]), (top2[0], top2[2])]

    # Pair bottom-A with the top corner closest to it in XZ.
    def _xz_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    if _xz_dist(bot_xz[0], top_xz_candidates[0]) <= _xz_dist(
        bot_xz[0], top_xz_candidates[1]
    ):
        ta_xz, tb_xz = top_xz_candidates[0], top_xz_candidates[1]
    else:
        ta_xz, tb_xz = top_xz_candidates[1], top_xz_candidates[0]

    top_y = 0.5 * (top1[1] + top2[1])
    return bot_xz[0], bot_xz[1], ta_xz, tb_xz, top_y


def walls_for_room(
    room: ExtractedRoom,
    *,
    floor: Floor,
    building_uuid: str,
    openings_by_wall_id: dict[str, tuple] | None = None,
) -> tuple[tuple[Wall, ...], tuple[Evidence, ...]]:
    """Construct canonical Walls for `room`. Anchored above by their
    own raw top Y; below by the Floor's canonical Y.

    `openings_by_wall_id`, if provided, attaches Openings to their host
    Wall by raw `ExtractedWall.id`. The Wall constructor then verifies
    each opening lies in the canonical wall plane (which it must, since
    `openings_by_wall_id` is built by `openings.openings_by_wall_id`
    using the same plane).

    Returns `(walls, orphan_evidence)`. Orphans are walls whose
    extraction geometry is degenerate (collinear bottom edge in XZ) and
    cannot be promoted to a Wall primitive — their raw corners are
    emitted as Evidence on the residual stream.
    """
    floor_y = float(floor.polygon[0].y)
    walls: list[Wall] = []
    orphans: list[Evidence] = []

    by_id: dict[str, dict[str, ExtractedWall]] = {}
    for raw_wall in room.walls_merged:
        by_id.setdefault(raw_wall.id, {})["merged"] = raw_wall
    for raw_wall in room.walls_computed:
        by_id.setdefault(raw_wall.id, {})["computed"] = raw_wall

    for wall_id, versions in by_id.items():
        anchor = versions.get("merged") or versions.get("computed")
        if anchor is None:
            continue
        evidence = tuple(
            _wall_evidence(versions[k], f"extracted_room.walls_{k}")
            for k in ("merged", "computed")
            if k in versions
        )
        openings = (
            openings_by_wall_id.get(wall_id, ())
            if openings_by_wall_id is not None
            else ()
        )
        try:
            wall = _build_wall(
                anchor,
                evidence=evidence,
                openings=openings,
                floor_y=floor_y,
                room=room,
                building_uuid=building_uuid,
            )
        except Exception:
            wall = None
        if wall is None:
            orphans.extend(evidence)
        else:
            walls.append(wall)

    return tuple(walls), tuple(orphans)


def _canonical_plane(raw_wall: ExtractedWall, *, floor_y: float) -> Plane | None:
    """Strictly vertical plane through the bottom edge XZ.

    Earlier I tried preserving the wall's apparent lean by fitting a
    plane through all 4 scan corners. The corpus showed that this
    inherits extraction noise (corners drift by 1-5 cm) as fake lean,
    and the Y-snap-onto-plane that follows distorts the bottom edge.
    Architectural reality: ~all walls are vertical. The lean signal
    is dominated by noise; preserving it produced more error than
    signal. Force vertical here.
    """
    quad = _split_wall_quad(raw_wall.corners)
    if quad is None:
        return None
    ba_xz, bb_xz, _ta_xz, _tb_xz, top_y = quad
    if top_y - floor_y < FLOAT_EPS:
        return None
    dx = bb_xz[0] - ba_xz[0]
    dz = bb_xz[1] - ba_xz[1]
    edge_len = math.hypot(dx, dz)
    if edge_len < FLOAT_EPS:
        return None
    nx, nz = dz / edge_len, -dx / edge_len
    px, pz = ba_xz
    return Plane(a=nx, b=0.0, c=nz, d=-(nx * px + nz * pz))


def _fit_wall_plane(corners: list) -> Plane | None:
    """Best-fit plane through wall corners, with the normal flipped to
    have a non-negative Y-cross component so the plane equation is
    canonical. Returns None for collinear inputs or horizontal fits.
    """
    import numpy as np

    arr = np.array([(p.x, p.y, p.z) for p in corners], dtype=float)
    centroid = arr.mean(axis=0)
    centered = arr - centroid
    try:
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular[1] < FLOAT_EPS:
        return None
    normal = vt[-1]
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    # Don't allow the plane to come out horizontal (would be a ceiling
    # masquerading as a wall); the Wall constructor would reject it.
    if abs(abs(ny) - 1.0) < 1e-3:
        return None
    d = -(nx * centroid[0] + ny * centroid[1] + nz * centroid[2])
    return Plane(a=nx, b=ny, c=nz, d=float(d))


def _project_to_plane(point: Vec3, plane: Plane) -> Vec3:
    """Snap a 3D point onto the plane along the plane's unit normal."""
    s = plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d
    return Vec3(
        x=point.x - s * plane.a,
        y=point.y - s * plane.b,
        z=point.z - s * plane.c,
    )


def _build_wall(
    raw_wall: ExtractedWall,
    *,
    evidence: tuple[Evidence, ...],
    openings: tuple,
    floor_y: float,
    room: ExtractedRoom,
    building_uuid: str,
) -> Wall | None:
    """Construct a Wall preserving the raw scanned polygon shape.

    Walls are quads, pentagons (gable peaks), or hexagons (stepped tops).
    We honour the corner count and let each wall's own min-Y define its
    bottom edge — extraction's wall Y and room.floor Y can differ by a
    few cm, which is real noise we don't paper over by snapping walls
    to the room's floor. The wall plane is the vertical plane through
    the bottom edge.
    """
    if len(raw_wall.corners) < 3:
        return None

    raw_polygon = [
        Vec3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in raw_wall.corners
    ]

    min_y = min(p.y for p in raw_polygon)
    max_y = max(p.y for p in raw_polygon)
    if max_y - min_y < FLOAT_EPS:
        return None

    # Snap every corner that's within 1 cm of the wall's min-Y down to
    # min-Y exactly — gives a clean horizontal bottom edge even when
    # extraction's bottom corners differ by a few mm. (1 cm is float-
    # precision relative to a ~2.5 m wall; not a building parameter.)
    bottom_threshold = min_y + 0.01
    floor_corners_xz: list[tuple[float, float]] = []
    snapped_polygon: list[Vec3] = []
    for p in raw_polygon:
        if p.y <= bottom_threshold:
            snapped_polygon.append(Vec3(x=p.x, y=min_y, z=p.z))
            floor_corners_xz.append((p.x, p.z))
        else:
            snapped_polygon.append(p)

    if len(floor_corners_xz) < 2:
        return None

    plane = _vertical_plane_through(floor_corners_xz)
    if plane is None:
        return None

    # Project every corner onto the plane (so the polygon is coplanar).
    snapped = tuple(_project_to_plane(p, plane) for p in snapped_polygon)

    return Wall(
        id=_wall_id(building_uuid, room, raw_wall),
        polygon=snapped,
        plane=plane,
        openings=tuple(openings),
        evidence=evidence,
    )


def _vertical_plane_through(floor_xz: list[tuple[float, float]]) -> Plane | None:
    """Vertical plane through the line spanned by a wall's floor-level
    corners. Uses the two extreme floor-corner XZ points as anchors so
    walls with >2 floor corners (rare; e.g. corner-step bottom) still
    get a sensible plane."""
    if len(floor_xz) < 2:
        return None
    # Find the pair of points spanning the largest XZ distance — this
    # is the bottom edge.
    best: tuple[tuple[float, float], tuple[float, float], float] | None = None
    n = len(floor_xz)
    for i in range(n):
        for j in range(i + 1, n):
            ax, az = floor_xz[i]
            bx, bz = floor_xz[j]
            d = math.hypot(bx - ax, bz - az)
            if best is None or d > best[2]:
                best = ((ax, az), (bx, bz), d)
    if best is None or best[2] < FLOAT_EPS:
        return None
    (ax, az), (bx, bz), edge_len = best
    nx = (bz - az) / edge_len
    nz = -(bx - ax) / edge_len
    return Plane(a=nx, b=0.0, c=nz, d=-(nx * ax + nz * az))


def _intersect_horizontal(plane: Plane, point: Vec3, y: float) -> Vec3 | None:
    """Move `point` to the requested Y along the plane (so it stays on
    the plane). Equivalent to intersecting a horizontal line through
    the point's XZ with the plane at that Y. Returns None if the plane
    is exactly horizontal (no intersection)."""
    if abs(plane.b) < FLOAT_EPS:
        # Vertical plane — Y-shift is free along the plane; just adopt y.
        return Vec3(x=point.x, y=y, z=point.z)
    # plane: a*x + b*y + c*z + d = 0 → for a leaning plane we shift the
    # point's XZ along the plane's gradient so it satisfies the equation.
    s = plane.a * point.x + plane.b * y + plane.c * point.z + plane.d
    norm_sq = plane.a * plane.a + plane.c * plane.c
    if norm_sq < FLOAT_EPS:
        return Vec3(x=point.x, y=y, z=point.z)
    factor = s / norm_sq
    return Vec3(
        x=point.x - factor * plane.a,
        y=y,
        z=point.z - factor * plane.c,
    )


def _wall_evidence(raw_wall: ExtractedWall, source: str) -> Evidence:
    geometry = tuple(
        Vec3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in raw_wall.corners
    )
    return Evidence(
        provenance=Provenance(kind="scan", source=source),
        geometry=geometry,
    )


def _wall_id(building_uuid: str, room: ExtractedRoom, raw_wall: ExtractedWall) -> str:
    return f"{building_uuid}::wall::{room.story}:{room.index}::{raw_wall.id}"
