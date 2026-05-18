#!/usr/bin/env python3
"""Ridge/eave topology scorer for Phase A candidate faces.

Scores operate at the **plane-group** level, not the individual intersected
segment level. Candidates that share the same plane equation (same V3
merged_roof_segment cluster, just clipped against different neighbors) are
grouped; the group's **union footprint** is what we score against.

For every plane-group we find the best opposing partner plane-group
(azimuth Δ ≈ 180°) and score the pair on mirror-parity checks:

* ``horizontality``       — ridge line (plane ∩ plane) should be level
* ``azimuth_opposition``  — downslope directions should be exactly antiparallel
* ``inclination_match``   — the two planes should share the same pitch
* ``eave_height_parity``  — their lowest edges should sit at the same Y

These are physical parity checks on the plane equations; no shape-based
side assignment or eave-perimeter calculation is needed, so Phase A's
ridge extrapolation does not perturb them.

Each candidate inherits the best pair-score of its plane-group, so the
viewer can keep its per-candidate overlay.

Dormers land in their own plane-group (different plane coefficients from
the main roof), so they don't muddy the main-roof scoring.

Usage
-----
    python scripts/score_candidates_ridge_eave.py \
        --candidates reports/candidate_faces_20260419/candidates.json \
        --out reports/ridge_eave_scores_20260420/scores.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Scoring knobs. Only AZIMUTH_PAIR_TOL_DEG and HORIZONTALITY_SIGMA_DEG are
# physical constants (plane normal precision, ridge-level tolerance from
# scan/construction noise). Everything else is derived from scan shape.
AZIMUTH_PAIR_TOL_DEG = 30.0
# Pair-eligibility distance: allows two plane-group footprints to sit apart by
# a small amount (e.g. a shed-extension separated from the main roof by a
# connect-buffer split, or by a narrow gap of unsupported scan). 2.5 m is
# wide enough to catch realistic extensions but still rules out unrelated
# roof fragments on opposite ends of a building.
MIN_FOOTPRINT_OVERLAP_OR_ADJ_M = 2.5
MIN_CANDIDATE_AREA_M2 = 0.25
# Physical soft tolerances (ridge parity, plane pitch, azimuth opposition,
# eave elevation). These are the only hardcoded sigmas — they encode real
# scan/construction noise, not shape heuristics.
HORIZONTALITY_SIGMA_DEG = 3.0  # ridge line inclination
AZIMUTH_OPPOSITION_SIGMA_DEG = 15.0  # deviation from pure 180° opposition
INCLINATION_MATCH_SIGMA_DEG = 12.0  # plane-pitch mismatch — wide enough
# to admit asymmetric structures (shed
# dormers, half-hips) where the two
# sides of a ridge have genuinely
# different pitches; still kills
# cross-element pairs (e.g. a roof
# plane mis-pairing with a wall).
EAVE_HEIGHT_PARITY_SIGMA_M = 2.0  # eave-Y mismatch — accepts asymmetric
# gables where one side meets the
# wall at a different elevation than
# the other (stepped roofs, shed
# dormers over main gable).
CREATOR_EAVE_PROXIMITY_SIGMA_M = 1.5  # provenance tie-break: prefer mirror
# partners whose source candidate
# footprints cluster near the eave they
# imply, especially on extensions where
# global parity alone is ambiguous.
PAIR_SCORE_TIE_EPS = 1e-6
# Exterior gate: a candidate's physical (unextended) scan must reach within
# this tolerance of the top-story wall tops to count as a rain-facing roof
# surface. Segments that only live below the wall envelope are interior
# geometry (attic vault faces, lower-floor scan noise) and cannot form
# valid gable pairs — their plane equations, when extrapolated, produce
# spurious mirror matches against real roof planes.
EXTERIOR_SCAN_TOL_M = 0.5
# Plane-group selection threshold: a plane-group is kept as part of the
# final envelope iff its best mirror-pair score reaches this. Plane-groups
# that don't pair with anything (best_score is None) fail the gate. Set
# below the geometric-mean "RED" elbow observed in the corpus so that
# plausible single-slope outshuts aren't dropped; high enough that shallow
# false-positive planes competing for the same footprint as a real green
# plane lose. Planes that pass the gate have their footprints unioned into
# the per-building envelope.
SELECTION_SCORE_THRESHOLD = 0.30
PLANE_KEY_DECIMALS = 3  # rounding for plane-group identity

# Physical eave detection from top-story wall tops. A plane is considered
# eave-resting on a wall when its predicted y at the wall-top xz position
# matches the wall-top y within this tolerance. 0.3 m covers scan noise and
# the fact that roof planes often overhang the wall by a small amount.
WALL_TOP_MATCH_TOL_M = 0.3
# Corners are "top" corners if their y is within this epsilon of the wall's
# max y. Handles minor wobble in scan-derived wall corners.
WALL_TOP_EPS_M = 0.1

# Connected-component splitting within a plane-group. A plane may cover
# multiple disjoint building parts (two wings of an L-shape that share the
# same pitch/azimuth, or a main slope plus a detached shed-dormer on the same
# plane). We buffer each member footprint by this amount before taking the
# union, so near-touching segments (scan noise / small gaps) stay merged, but
# truly disjoint parts get split into separate plane-subgroups and scored
# independently. Members are re-associated to the subgroup they overlap most.
PLANE_GROUP_CONNECT_BUFFER_M = 0.3
MIN_SUBGROUP_AREA_M2 = 1.5  # drop tiny stray components

# Morphological-closing radius for the plane-group union. Fills interior
# concavities (scan-patchy notches, diagonal cuts where a scanned wall under
# the roof split the candidate footprint, thin between-room strips) up to
# ~2·C in width without extending the outer silhouette. Clipped to the
# per-building room-footprint union so closing never invents area outside
# the physical building. 0.75 m covers the within-story gap strips from
# cross_floor_gaps AND the ~1 m diagonal cuts seen on e0155eef-… story-2
# walls, while staying well below a full room dimension to avoid hijacking
# neighbouring rooms into the plane-group extent.
PLANE_GROUP_CLOSING_M = 0.75

# Maximum distance the plane-group's XZ extent is allowed to flood outward
# into the building envelope. Closing fills interior notches but cannot
# GROW the outer silhouette — cross_story gaps like 117d172e-…::cross_story:
# high:10 extend beyond the candidate footprint by more than the closing
# radius, so the plane-group's outer edge stops short of the building
# envelope. Flood-fill unions in the connected subset of envelope_fp
# within this distance of the plane-group's closed union, which claims
# cross_story gaps, overhangs, and lower-story extensions that sit
# physically below the same roof plane. The distance cap also enforces
# the user's "not completely free floating for more than X meters" rule:
# detached outbuildings or wings further than FLOOD_REACH_M from any
# candidate are not absorbed.
PLANE_GROUP_FLOOD_REACH_M = 2.0


# ---------------------------------------------------------------------------
# Plane / vector utilities
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _plane_normal(plane: list[float]) -> np.ndarray:
    """Return normal (a,b,c) with b>0 (roof-up)."""
    a, b, c, _ = (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))
    n = np.array([a, b, c], dtype=float)
    if n[1] < 0:
        n = -n
    return _unit(n)


def _downslope_xz(plane: list[float]) -> np.ndarray:
    """Unit 2D direction (dx, dz) of steepest descent in xz.

    With ``y = -(a·x + c·z + d)/b`` and b>0, ``∇y = (-a/b, -c/b)``, so the
    downslope vector (direction where y decreases) is ``(a, c)`` normalized.
    """
    n = _plane_normal(plane)
    a, _, c = n
    d2 = np.array([a, c], dtype=float)
    return _unit(d2)


def _canonical_plane(plane: list[float]) -> tuple[float, float, float, float]:
    """Canonicalize (a,b,c,d) so b>0 (roof-up)."""
    a, b, c, d = (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))
    if b < 0:
        a, b, c, d = -a, -b, -c, -d
    return (a, b, c, d)


def _plane_key(plane: list[float]) -> tuple[float, float, float, float]:
    a, b, c, d = _canonical_plane(plane)
    q = PLANE_KEY_DECIMALS
    return (round(a, q), round(b, q), round(c, q), round(d, q))


def _plane_id(building_uuid: str, key: tuple) -> str:
    """Short deterministic id for a plane-group (used as pair a_id/b_id)."""
    h = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=6).hexdigest()
    return f"{building_uuid}::plane-group::{h}"


def _plane_intersection_line_3d(p1: list[float], p2: list[float]):
    """Return (point, direction) of the intersection line of two planes, or None.

    ``direction`` is unit. ``point`` is any point on the line.
    """
    n1 = _plane_normal(p1)
    n2 = _plane_normal(p2)
    d = np.cross(n1, n2)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:
        return None
    d = d / dn
    d1 = float(p1[3])
    d2 = float(p2[3])
    A = np.vstack([n1, n2, d])
    b = np.array([-d1, -d2, 0.0], dtype=float)
    try:
        pt = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return pt, d


def _ridge_inclination_deg(direction: np.ndarray) -> float:
    """Angle between the ridge direction and the horizontal xz plane."""
    dy = abs(float(direction[1]))
    horiz = math.sqrt(float(direction[0]) ** 2 + float(direction[2]) ** 2)
    if horiz < 1e-9:
        return 90.0
    return math.degrees(math.atan2(dy, horiz))


def _plane_inclination_deg(plane: list[float]) -> float:
    """Angle of the plane itself from horizontal (0 = flat, 90 = vertical).

    With b>0 normal, cos(plane_incl) = n_y = b/|n|.
    """
    n = _plane_normal(plane)
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(float(n[1]))))))


def _azimuth_deviation_deg(ds_a: np.ndarray, ds_b: np.ndarray) -> float:
    """Angle between ds_a and the *opposite* of ds_b, in degrees.

    Returns 0 for a perfect 180°-apart pair (ds_a = -ds_b) and 180 for
    a parallel pair (ds_a = ds_b). 90 for perpendicular (hip) pairs.
    """
    neg_b = -ds_b
    cos_t = float(np.dot(ds_a, neg_b))
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


def _plane_y_at(plane: list[float], xz: np.ndarray) -> np.ndarray:
    """Plane's y value at each (x, z) — uses y = -(a x + c z + d)/b."""
    a, b, c, d = (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))
    if b < 0:
        a, b, c, d = -a, -b, -c, -d
    return -(xz[:, 0] * a + xz[:, 1] * c + d) / b


def _plane_eave_y(plane: list[float], poly: ShapelyPolygon) -> float | None:
    """Lowest y where the plane meets the polygon's exterior (eave elevation).

    Fallback when no physical wall-top data is available. Unreliable for
    ridge-extrapolated plane unions because min-Y moves with the plane's
    extrapolated extent rather than staying at the real eave.
    """
    coords = _coords_from_union(poly)
    if coords is None or len(coords) == 0:
        return None
    ys = _plane_y_at(plane, coords)
    return float(ys.min())


def _plane_eave_y_clusters(
    plane: list[float],
    wall_tops: np.ndarray | None,
    partner_plane: list[float] | None = None,
) -> list[tuple[float, int]]:
    """Return clustered (eave_y, count) candidates where the plane rests on walls.

    A corner ``(x, y, z)`` counts as an eave anchor for ``plane`` iff the
    plane's predicted y at ``(x, z)`` agrees with the corner's y within
    ``WALL_TOP_MATCH_TOL_M``. That alone would also match **ridge corners**
    (e.g. a gable pentagon's apex), where both mirror planes pass through the
    same point at the same elevation. When ``partner_plane`` is supplied,
    corners where the partner *also* agrees are excluded as ridge corners —
    the eave is by definition the edge where only one of the two planes
    rests.

    A plane that extrapolates across the whole scan may rest on walls at
    multiple distinct elevations (L- and U-shapes with wings at different
    eave heights). Each distinct elevation becomes a cluster; the pair
    scorer picks the cluster shared with the partner.

    Adjacent y's are merged when they're within ``WALL_TOP_EPS_M`` (scan
    noise). Returns the list sorted by descending count, then ascending y.
    Empty if the plane doesn't rest on any non-ridge wall corner.
    """
    if wall_tops is None or len(wall_tops) == 0:
        return []
    xz = wall_tops[:, [0, 2]]
    y_pred = _plane_y_at(plane, xz)
    y_real = wall_tops[:, 1]
    mask = np.abs(y_pred - y_real) <= WALL_TOP_MATCH_TOL_M
    if partner_plane is not None:
        y_partner = _plane_y_at(partner_plane, xz)
        ridge_mask = np.abs(y_partner - y_real) <= WALL_TOP_MATCH_TOL_M
        mask = mask & ~ridge_mask
    matched = y_real[mask]
    if len(matched) == 0:
        return []
    ys = np.sort(matched)
    clusters: list[list[float]] = [[float(ys[0])]]
    for y in ys[1:]:
        if float(y) - clusters[-1][-1] <= WALL_TOP_EPS_M:
            clusters[-1].append(float(y))
        else:
            clusters.append([float(y)])
    out = [(float(np.mean(c)), len(c)) for c in clusters]
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def _resolve_eave_pair(
    clusters_a: list[tuple[float, int]], clusters_b: list[tuple[float, int]]
) -> tuple[float, float] | None:
    """Pick the (eave_a, eave_b) pair with the smallest |Δ|.

    Tie-break by total support (higher combined cluster count preferred).
    Returns None if either plane has no clusters.
    """
    if not clusters_a or not clusters_b:
        return None
    best_key: tuple[float, int] | None = None
    best_pair: tuple[float, float] | None = None
    for y_a, c_a in clusters_a:
        for y_b, c_b in clusters_b:
            key = (abs(y_a - y_b), -(c_a + c_b))
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (y_a, y_b)
    return best_pair


def _extract_top_story_wall_tops(building: dict) -> np.ndarray:
    """Collect every top-story walls_merged corner as an Nx3 array.

    All corners are returned — no pre-filtering for "top" or "eave" corners.
    The per-plane match tolerance (``WALL_TOP_MATCH_TOL_M``) in
    :func:`_plane_eave_y_clusters` naturally excludes base-of-wall corners
    (they're ~2-3 m below any realistic roof plane) and, together with the
    partner-plane ridge check, excludes apex / ridge-line corners. This
    keeps the extractor shape-agnostic — it works for rectangle, pentagon
    (symmetric and asymmetric gables), hexagon, and octagon walls without
    per-shape rules.
    """
    rooms = building.get("rooms") or []
    stories = [r.get("story", 0) for r in rooms]
    if not stories:
        return np.zeros((0, 3), dtype=float)
    top = max(stories)
    pts: list[list[float]] = []
    for r in rooms:
        if r.get("story") != top:
            continue
        for w in r.get("walls_merged") or []:
            corners = w.get("corners") or []
            if len(corners) > 1 and corners[0] == corners[-1]:
                corners = corners[:-1]
            for c in corners:
                if len(c) < 3:
                    continue
                pts.append([float(c[0]), float(c[1]), float(c[2])])
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.array(pts, dtype=float)


def _top_story_wall_top_y(wall_corners: np.ndarray) -> float | None:
    """Max Y across top-story wall_merged corners — the building's wall-top envelope.

    Candidates whose physical (unextended) scan y_max sits well below this
    value are interior geometry, not rain-facing roof surfaces.
    """
    if wall_corners is None or len(wall_corners) == 0:
        return None
    return float(wall_corners[:, 1].max())


def _load_scan_y_max_by_parent(v3_path: Path) -> dict[str, float]:
    """Map merged_roof_segment id → max Y of its scan corners.

    The ``corners`` array on merged_roof_segments comes directly from the
    V3 geometric step and is the best available proxy for the physical
    vertical extent of the scanned surface. Phase A extrapolation expands
    this laterally but the corners themselves remain scan-anchored.
    """
    if not v3_path.exists():
        return {}
    with v3_path.open() as handle:
        v3 = json.load(handle)
    buildings = v3 if isinstance(v3, list) else [v3]
    out: dict[str, float] = {}
    for b in buildings:
        for seg in b.get("merged_roof_segments") or []:
            sid = seg.get("id")
            if not sid:
                continue
            corners = seg.get("corners") or []
            ys = [float(c[1]) for c in corners if len(c) >= 3]
            if ys:
                out[sid] = max(ys)
    return out


def _plane_ridge_y(plane: list[float], poly: ShapelyPolygon) -> float | None:
    """Highest y where the plane meets the polygon's exterior (ridge elevation)."""
    coords = _coords_from_union(poly)
    if coords is None or len(coords) == 0:
        return None
    ys = _plane_y_at(plane, coords)
    return float(ys.max())


# ---------------------------------------------------------------------------
# OBB (oriented bounding box) in xz along a given axis
# ---------------------------------------------------------------------------


def _obb_aligned(points_xz: np.ndarray, long_axis: np.ndarray) -> dict:
    long_axis = _unit(long_axis)
    perp = np.array([-long_axis[1], long_axis[0]], dtype=float)
    t_long = points_xz @ long_axis
    t_perp = points_xz @ perp
    min_l, max_l = float(t_long.min()), float(t_long.max())
    min_p, max_p = float(t_perp.min()), float(t_perp.max())
    cx = 0.5 * (min_l + max_l)
    cp = 0.5 * (min_p + max_p)
    center = cx * long_axis + cp * perp
    half_l = 0.5 * (max_l - min_l)
    half_p = 0.5 * (max_p - min_p)
    corners = np.array(
        [
            center + (-half_l) * long_axis + (-half_p) * perp,
            center + (half_l) * long_axis + (-half_p) * perp,
            center + (half_l) * long_axis + (half_p) * perp,
            center + (-half_l) * long_axis + (half_p) * perp,
        ]
    )
    return {
        "center": center,
        "long": long_axis,
        "perp": perp,
        "half_long": half_l,
        "half_perp": half_p,
        "corners": corners,
    }


def _plane_group_eave_segment(
    points_xz: np.ndarray, downslope_xz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Eave segment for a plane-group: long edge of its OBB on the downslope side.

    Long axis for the OBB is the eave direction = perpendicular to downslope.
    Use the union footprint points — this gives the full plane extent, not
    just one intersected fragment.
    """
    eave_dir = np.array([-downslope_xz[1], downslope_xz[0]], dtype=float)
    obb = _obb_aligned(points_xz, eave_dir)
    c = obb["center"]
    hl = obb["half_long"]
    hp = obb["half_perp"]
    eave_center = c + hp * downslope_xz
    p0 = eave_center - hl * eave_dir
    p1 = eave_center + hl * eave_dir
    return p0, p1


def _seg_length(p0: np.ndarray, p1: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p0))


def _angle_between_dirs_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = _unit(a)
    b = _unit(b)
    cos_t = abs(float(np.dot(a, b)))
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


def _clip_line_to_obb(
    pt3: np.ndarray, dir3: np.ndarray, obb: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    d_xz = np.array([dir3[0], dir3[2]], dtype=float)
    dn = float(np.linalg.norm(d_xz))
    if dn < 1e-9:
        return None
    d_xz /= dn
    scale = 1.0 / dn
    dir3_xz1 = dir3 * scale
    pt_xz = np.array([pt3[0], pt3[2]], dtype=float)
    long_axis = obb["long"]
    perp = obb["perp"]
    t_long_0 = float(np.dot(pt_xz - obb["center"], long_axis))
    t_perp_0 = float(np.dot(pt_xz - obb["center"], perp))
    v_long = float(np.dot(d_xz, long_axis))
    v_perp = float(np.dot(d_xz, perp))

    t_min, t_max = -1e9, 1e9

    def _clip(v: float, t0: float, half: float) -> tuple[float, float] | None:
        nonlocal t_min, t_max
        if abs(v) < 1e-9:
            if abs(t0) > half + 1e-6:
                return None
            return t_min, t_max
        t_lo = (-half - t0) / v
        t_hi = (half - t0) / v
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        return max(t_min, t_lo), min(t_max, t_hi)

    clip_long = _clip(v_long, t_long_0, obb["half_long"])
    if clip_long is None:
        return None
    t_min, t_max = clip_long
    clip_perp = _clip(v_perp, t_perp_0, obb["half_perp"])
    if clip_perp is None:
        return None
    t_min, t_max = clip_perp
    if t_max <= t_min + 1e-6:
        return None
    p_start = pt3 + t_min * dir3_xz1
    p_end = pt3 + t_max * dir3_xz1
    return p_start, p_end


# ---------------------------------------------------------------------------
# Plane-group construction
# ---------------------------------------------------------------------------


def _coords_from_union(poly) -> np.ndarray | None:
    """Exterior coords of a (Multi)Polygon, flattened for OBB computation."""
    if poly is None or poly.is_empty:
        return None
    if hasattr(poly, "exterior"):
        return np.array(list(poly.exterior.coords)[:-1], dtype=float)
    pts: list[list[float]] = []
    for geom in getattr(poly, "geoms", [poly]):
        if hasattr(geom, "exterior"):
            pts.extend(list(geom.exterior.coords)[:-1])
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=float)


def _building_fp_union(bldg: dict | None) -> ShapelyPolygon | None:
    """Union of all room floor polygons (XZ) for a buildings_3d.json entry.

    Serves as the outer clip for morphological-closing widening of
    plane-group unions (see ``PLANE_GROUP_CLOSING_M``): closing can fill
    interior notches of the plane-group but the result is then intersected
    with this union so closing never invents area outside the physical
    building. Every story's rooms contribute — a plane-group's XZ extent
    can overlap rooms on any story since roofs sit above the full stack.
    """
    if not isinstance(bldg, dict):
        return None
    polys: list[ShapelyPolygon] = []
    for r in bldg.get("rooms") or []:
        fp = r.get("floor_polygon") or []
        if len(fp) < 3:
            continue
        ring = [(float(c[0]), float(c[2])) for c in fp if len(c) >= 3]
        if len(ring) < 3:
            continue
        try:
            p = ShapelyPolygon(ring)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty or p.area < 1e-4:
                continue
            polys.append(p)
        except Exception:
            continue
    if not polys:
        return None
    try:
        u = unary_union(polys)
    except Exception:
        return None
    if u.is_empty:
        return None
    return u


def _building_envelope_fp(bldg: dict | None) -> ShapelyPolygon | None:
    """Maximum XZ extent a plane-group may flood-fill into.

    ``_building_fp_union`` gives the room-only union (used to CLIP closing
    so morphological operations can't invent outside-the-building area).
    The envelope additionally folds in every ``cross_floor_gap`` polygon
    (both ``within_story`` and ``cross_story``). Those gaps represent
    between-room or between-story areas that physically sit beneath the
    roof — a plane-group may claim them, limited by the flood-reach cap.
    """
    if not isinstance(bldg, dict):
        return None
    polys: list[ShapelyPolygon] = []
    for r in bldg.get("rooms") or []:
        fp = r.get("floor_polygon") or []
        if len(fp) < 3:
            continue
        ring = [(float(c[0]), float(c[2])) for c in fp if len(c) >= 3]
        if len(ring) < 3:
            continue
        try:
            p = ShapelyPolygon(ring)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty or p.area < 1e-4:
                continue
            polys.append(p)
        except Exception:
            continue
    for g in bldg.get("cross_floor_gaps") or []:
        corners = g.get("corners") or []
        if len(corners) < 3:
            continue
        ring = [(float(c[0]), float(c[2])) for c in corners if len(c) >= 3]
        if len(ring) < 3:
            continue
        try:
            p = ShapelyPolygon(ring)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty or p.area < 1e-4:
                continue
            polys.append(p)
        except Exception:
            continue
    if not polys:
        return None
    try:
        u = unary_union(polys)
    except Exception:
        return None
    if u.is_empty:
        return None
    return u


def _build_plane_groups(
    building_uuid: str,
    cands: list[dict],
    scan_poly: ShapelyPolygon | None = None,
    building_fp: ShapelyPolygon | None = None,
    envelope_fp: ShapelyPolygon | None = None,
) -> list[dict]:
    """Group candidates by canonical plane equation AND spatial connectivity.

    Two-step grouping:
    1. Bucket candidates by canonical plane-coefficient key. Members share
       the same physical slope (same pitch, azimuth, altitude).
    2. Within a bucket, union member footprints with a small buffer and split
       into connected components. Each component becomes its own plane-subgroup.
       This handles buildings where a single plane covers multiple disjoint
       parts (L/T-shape with matching-azimuth wings, detached shed-dormers on
       the same plane, etc.) — each part gets its own OBB and medial axis.

    Returns a list of plane-subgroup dicts:
        id          : synthetic subgroup id (deterministic hash)
        key         : (plane_key, component_index)
        plane       : a representative plane [a,b,c,d] (from largest-area member)
        rep_id      : representative candidate id (largest-area member)
        members     : list of candidate dicts in this subgroup
        member_ids  : list of candidate ids
        union       : Shapely polygon (exterior union of member footprints)
        union_coords: Nx2 numpy array of exterior points (for OBB)
        total_area  : union area (real extent; not member-sum)
        max_support : max ``support_m2`` across members
        azimuth_deg : representative azimuth
        inclination_deg : representative inclination
        n_components_in_parent : number of components the parent plane split into
    """
    buckets: dict[tuple, list[tuple[dict, ShapelyPolygon]]] = {}
    for c in cands:
        plane = c.get("plane")
        if not isinstance(plane, list) or len(plane) != 4:
            continue
        if c.get("area_m2", 0.0) < MIN_CANDIDATE_AREA_M2:
            continue
        fp = c.get("footprint_xz")
        if not fp or len(fp) < 3:
            continue
        try:
            poly = ShapelyPolygon(fp)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
        except Exception:
            continue
        buckets.setdefault(_plane_key(plane), []).append((c, poly))

    groups: list[dict] = []
    for key, entries in buckets.items():
        member_polys = [p for _, p in entries]
        try:
            buffered = unary_union(
                [p.buffer(PLANE_GROUP_CONNECT_BUFFER_M) for p in member_polys]
            )
        except Exception:
            continue
        if buffered.is_empty:
            continue
        components = list(buffered.geoms) if hasattr(buffered, "geoms") else [buffered]

        for comp_idx, comp in enumerate(components):
            comp_members: list[dict] = []
            comp_polys: list[ShapelyPolygon] = []
            for m, p in entries:
                try:
                    inter = p.intersection(comp)
                except Exception:
                    continue
                if inter.is_empty:
                    continue
                # Assign member to component where ≥50% of its area lies.
                if inter.area >= 0.5 * p.area:
                    comp_members.append(m)
                    comp_polys.append(p)
            if not comp_polys:
                continue
            try:
                real_union = unary_union(comp_polys)
            except Exception:
                continue
            if real_union.is_empty:
                continue
            # Clip to the scan footprint so Phase A's ridge-extrapolation
            # cannot push plane extents beyond the real building. Without this
            # the OBB can be 50% oversized, which crushes centering and
            # eave_footprint scores.
            if scan_poly is not None and not scan_poly.is_empty:
                try:
                    clipped = real_union.intersection(scan_poly)
                except Exception:
                    clipped = real_union
                if not clipped.is_empty and clipped.area >= MIN_SUBGROUP_AREA_M2:
                    real_union = clipped
            # Fill interior concavities via morphological closing, clipped to
            # the building's room-footprint union. Candidate footprints are
            # per-segment/per-room and scan coverage is never perfect, so the
            # raw union acquires two kinds of holes that the physical roof
            # plane does not have: (a) thin strips between adjacent room
            # slabs (the ``cross_floor_gaps[type=within_story]`` case on
            # 38f71f1d-…), and (b) diagonal cuts where a scanned wall under
            # the roof split the candidate footprint but the roof itself
            # extends over the wall (the e0155eef-…::wall-computed::E2E5…
            # case). Closing at 2·C ≈ 1.5 m bridges both without extending
            # the outer silhouette; intersecting with ``building_fp`` keeps
            # anything closing might have smoothed beyond the building back
            # inside it. Must run AFTER the scan clip because scan_poly
            # derives from candidates and does not include those holes —
            # an earlier clip would immediately trim what we just filled.
            if building_fp is not None and not building_fp.is_empty:
                try:
                    closed = real_union.buffer(PLANE_GROUP_CLOSING_M).buffer(
                        -PLANE_GROUP_CLOSING_M
                    )
                    addition = closed.intersection(building_fp)
                    if not addition.is_empty and addition.area > 1e-3:
                        real_union = unary_union([real_union, addition])
                except Exception:
                    pass
            # Flood-fill into the building envelope (rooms + cross_floor_gaps)
            # capped at PLANE_GROUP_FLOOD_REACH_M. Closing fills interior
            # notches but can't grow the outer silhouette, so cross_story
            # extensions like 117d172e-…::cross_story:high:10 stay uncovered.
            # Flood-fill takes real_union.buffer(X) ∩ envelope_fp and keeps
            # only the connected components that touch real_union — so
            # disconnected wings and detached outbuildings don't get absorbed,
            # and even within one connected envelope the expansion stops X m
            # from the current plane-group (the "not free-floating more than
            # X meters" rule).
            if envelope_fp is not None and not envelope_fp.is_empty:
                try:
                    buffered = real_union.buffer(PLANE_GROUP_FLOOD_REACH_M)
                    reach = buffered.intersection(envelope_fp)
                    if not reach.is_empty:
                        comps = (
                            list(reach.geoms) if hasattr(reach, "geoms") else [reach]
                        )
                        touching = [c for c in comps if c.intersects(real_union)]
                        if touching:
                            real_union = unary_union([real_union, *touching])
                except Exception:
                    pass
            if real_union.is_empty or real_union.area < MIN_SUBGROUP_AREA_M2:
                continue
            coords = _coords_from_union(real_union)
            if coords is None:
                continue

            rep = max(comp_members, key=lambda m: m.get("area_m2", 0.0))
            sub_key = (key, comp_idx)
            groups.append(
                {
                    "id": _plane_id(building_uuid, sub_key),
                    "key": sub_key,
                    "plane": rep["plane"],
                    "rep_id": rep["id"],
                    "members": comp_members,
                    "member_ids": [m["id"] for m in comp_members],
                    "union": real_union,
                    "union_coords": coords,
                    "total_area": round(float(real_union.area), 3),
                    "max_support": round(
                        max(
                            float(m.get("support_m2", 0.0) or 0.0) for m in comp_members
                        ),
                        3,
                    ),
                    "azimuth_deg": rep.get("azimuth_deg"),
                    "inclination_deg": rep.get("inclination_deg"),
                    "n_components_in_parent": len(components),
                }
            )

    return groups


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _ridge_segment_on_scan(
    plane_a: list[float], plane_b: list[float], scan_poly: ShapelyPolygon
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Project plane∩plane to XZ, clip to scan. Return (p0_3d, p1_3d, dir3).

    Endpoints are returned in 3D (y from the 3D ridge line, xz from scan clip).
    """
    line = _plane_intersection_line_3d(plane_a, plane_b)
    if line is None:
        return None
    pt3, dir3 = line
    dir_xz = np.array([dir3[0], dir3[2]], dtype=float)
    n = float(np.linalg.norm(dir_xz))
    if n < 1e-9:
        return None
    dir_xz_u = dir_xz / n
    # Extend far and clip to scan (work in 2D, lift y back via parametric).
    span = 500.0
    p_xz = np.array([pt3[0], pt3[2]], dtype=float)
    seg2d = ShapelyLineString(
        [
            tuple(p_xz - span * dir_xz_u),
            tuple(p_xz + span * dir_xz_u),
        ]
    ).intersection(scan_poly)
    if seg2d.is_empty:
        return None
    if hasattr(seg2d, "geoms"):
        # concave scan can cut ridge into pieces — pick longest
        seg2d = max(seg2d.geoms, key=lambda g: getattr(g, "length", 0.0))
    if not hasattr(seg2d, "length") or seg2d.length < 1e-3:
        return None
    coords = list(seg2d.coords)
    a2 = np.array(coords[0], dtype=float)
    b2 = np.array(coords[-1], dtype=float)
    # Parametric lift to 3D using dir3 / dir_xz length ratio.
    t_a = float(np.dot(a2 - p_xz, dir_xz_u)) / n  # scalar along dir3
    t_b = float(np.dot(b2 - p_xz, dir_xz_u)) / n
    p3a = np.array(pt3, dtype=float) + t_a * np.array(dir3, dtype=float)
    p3b = np.array(pt3, dtype=float) + t_b * np.array(dir3, dtype=float)
    return p3a, p3b, np.array(dir3, dtype=float)


def _split_scan_by_ridge(
    scan_poly: ShapelyPolygon, ridge_xz_a: np.ndarray, ridge_xz_b: np.ndarray
) -> list[ShapelyPolygon]:
    """Split scan into polygons by the ridge line (extended).

    Uses a thin slicing polygon (shapely doesn't have a robust line-polygon
    split for arbitrary cases). We build an extended line and use difference
    with an infinitesimally thin rotated rectangle. Practically: extend the
    ridge far past the scan bounds, buffer by a tiny epsilon to get a slicer,
    subtract from scan; the result is ≥2 polygons when the ridge crosses.
    """
    dir_xz = ridge_xz_b - ridge_xz_a
    n = float(np.linalg.norm(dir_xz))
    if n < 1e-9:
        return [scan_poly]
    u = dir_xz / n
    span = 1000.0
    p0 = ridge_xz_a - span * u
    p1 = ridge_xz_b + span * u
    from shapely.ops import split as shp_split

    try:
        result = shp_split(scan_poly, ShapelyLineString([tuple(p0), tuple(p1)]))
    except Exception:
        return [scan_poly]
    pieces = (
        [g for g in result.geoms if g.area > 1e-6]
        if hasattr(result, "geoms")
        else [result]
    )
    return pieces


def _assign_sides(
    pieces: list[ShapelyPolygon],
    ridge_mid_xz: np.ndarray,
    ds_a: np.ndarray,
    ds_b: np.ndarray,
) -> tuple[ShapelyPolygon | None, ShapelyPolygon | None]:
    """Partition all pieces between plane A and plane B by centroid·downslope.

    Every piece gets assigned to the side whose downslope direction its
    centroid (relative to the ridge midpoint) projects onto more strongly.
    The returned ``side_a`` / ``side_b`` polygons are the unions of all
    pieces on each side — this handles pair domains that split into 3+
    pieces when a ridge is tangent to a disjoint lobe (e.g. an extension
    whose plane was extended to meet the ridge).
    """
    a_pieces: list[ShapelyPolygon] = []
    b_pieces: list[ShapelyPolygon] = []
    for piece in pieces:
        c = piece.centroid
        offset = np.array([c.x - ridge_mid_xz[0], c.y - ridge_mid_xz[1]], dtype=float)
        proj_a = float(np.dot(offset, ds_a))
        proj_b = float(np.dot(offset, ds_b))
        if proj_a >= proj_b:
            a_pieces.append(piece)
        else:
            b_pieces.append(piece)
    side_a = unary_union(a_pieces) if a_pieces else None
    side_b = unary_union(b_pieces) if b_pieces else None
    return side_a, side_b


def _eave_geometry(
    piece: ShapelyPolygon, ridge_line: ShapelyLineString, tol: float
) -> tuple[float, ShapelyLineString | None]:
    """Eave = exterior of piece minus the portion coincident with the ridge.

    Returns (eave_length, eave_geometry_or_none).
    """
    try:
        exterior = ShapelyLineString(list(piece.exterior.coords))
    except Exception:
        return 0.0, None
    try:
        not_ridge = exterior.difference(ridge_line.buffer(tol, cap_style=2))
    except Exception:
        return float(exterior.length), exterior
    return float(not_ridge.length), not_ridge


def _candidate_member_polygon(member: dict) -> ShapelyPolygon | None:
    fp = member.get("footprint_xz")
    if not fp or len(fp) < 3:
        return None
    try:
        poly = ShapelyPolygon(fp)
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        return None
    if poly.is_empty or poly.area < 1e-6:
        return None
    return poly


def _creator_eave_proximity_score(
    plane_group: dict,
    eave_geom: ShapelyLineString | None,
) -> float:
    """How strongly the plane-group's source candidates cluster near its eave.

    Uses candidate-footprint centroids as a proxy for the oblique segments that
    created the plane-group. This is intentionally local: when one large slope
    can mirror multiple opposite-side groups, the partner whose creator
    footprints concentrate near the implied eave should win the tie.
    """
    if eave_geom is None or eave_geom.is_empty:
        return 0.0
    weighted_score = 0.0
    total_weight = 0.0
    for member in plane_group.get("members") or []:
        poly = _candidate_member_polygon(member)
        if poly is None:
            continue
        centroid = poly.centroid
        dist_m = float(eave_geom.distance(centroid))
        proximity = math.exp(-((dist_m / CREATOR_EAVE_PROXIMITY_SIGMA_M) ** 2))
        weight = max(
            float(member.get("support_m2") or 0.0),
            float(member.get("area_m2") or 0.0),
            float(poly.area),
            1e-6,
        )
        weighted_score += weight * proximity
        total_weight += weight
    if total_weight <= 1e-9:
        return 0.0
    return weighted_score / total_weight


def _pair_rank_key(pair: dict) -> tuple[float, float]:
    return (
        float(pair.get("score", pair.get("best_score")) or 0.0),
        float(pair.get("creator_eave_proximity") or 0.0),
    )


def _pair_beats(previous: dict | None, candidate: dict) -> bool:
    if previous is None:
        return True
    prev_score = float(previous.get("score", previous.get("best_score")) or 0.0)
    cand_score = float(candidate.get("score", candidate.get("best_score")) or 0.0)
    if cand_score > prev_score + PAIR_SCORE_TIE_EPS:
        return True
    if prev_score > cand_score + PAIR_SCORE_TIE_EPS:
        return False
    prev_creator = float(previous.get("creator_eave_proximity") or 0.0)
    cand_creator = float(candidate.get("creator_eave_proximity") or 0.0)
    return cand_creator > prev_creator + PAIR_SCORE_TIE_EPS


def _score_plane_pair(
    pg_a: dict,
    pg_b: dict,
    scan_poly: ShapelyPolygon | None,
    wall_tops: np.ndarray | None = None,
) -> dict | None:
    """Mirror-based score for a plane pair.

    A gable ridge is formed by two planes whose (azimuth, inclination, eave
    elevation) vectors mirror each other:

      * ``azimuth_opposition`` — downslope directions are ~180° apart
      * ``inclination_match``  — the two planes share the same pitch
      * ``horizontality``      — their intersection line is level (ridge)
      * ``eave_height_parity`` — their low edges sit at the same Y

    These are physical parity checks on the plane equations; no shape-based
    side assignment or eave perimeter calculation is needed. Phase A's ridge
    extrapolation does not perturb these quantities.
    """
    plane_a = pg_a["plane"]
    plane_b = pg_b["plane"]

    ds_a = _downslope_xz(plane_a)
    ds_b = _downslope_xz(plane_b)

    # Fast reject: downslope vectors must be close to antiparallel.
    if float(np.dot(ds_a, ds_b)) > -math.cos(math.radians(AZIMUTH_PAIR_TOL_DEG)):
        return None

    poly_a: ShapelyPolygon = pg_a["union"]
    poly_b: ShapelyPolygon = pg_b["union"]
    try:
        if poly_a.distance(poly_b) > MIN_FOOTPRINT_OVERLAP_OR_ADJ_M:
            return None
    except Exception:
        return None

    # Ridge clipped into the scan (or pair domain if scan doesn't cover the pair).
    try:
        pair_domain = unary_union([poly_a, poly_b])
    except Exception:
        return None
    if pair_domain.is_empty:
        return None
    clip_domain = (
        scan_poly
        if (
            scan_poly is not None
            and not scan_poly.is_empty
            and scan_poly.intersects(pair_domain)
        )
        else pair_domain
    )
    ridge = _ridge_segment_on_scan(plane_a, plane_b, clip_domain)
    if ridge is None:
        return None
    r0_3d, r1_3d, dir3 = ridge
    r0_xz = np.array([r0_3d[0], r0_3d[2]], dtype=float)
    r1_xz = np.array([r1_3d[0], r1_3d[2]], dtype=float)
    ridge_len = float(np.linalg.norm(r1_xz - r0_xz))
    if ridge_len < 1e-3:
        return None

    # --- horizontality: ridge line level in 3D
    ridge_incl_deg = _ridge_inclination_deg(dir3)
    horizontality = math.exp(-((ridge_incl_deg / HORIZONTALITY_SIGMA_DEG) ** 2))

    # --- azimuth_opposition: ds_a and -ds_b should coincide
    az_dev_deg = _azimuth_deviation_deg(ds_a, ds_b)
    azimuth_opposition = math.exp(-((az_dev_deg / AZIMUTH_OPPOSITION_SIGMA_DEG) ** 2))

    # --- inclination_match: same pitch
    incl_a = _plane_inclination_deg(plane_a)
    incl_b = _plane_inclination_deg(plane_b)
    incl_diff_deg = abs(incl_a - incl_b)
    inclination_match = math.exp(-((incl_diff_deg / INCLINATION_MATCH_SIGMA_DEG) ** 2))

    # --- eave_height_parity: physical eave Y on each plane should match.
    # A plane may rest on walls at multiple distinct elevations (when its
    # extrapolated extent crosses walls at different top heights). Cluster
    # matched wall-top y's per plane, then pick the cluster pair with the
    # smallest |Δ| — i.e. the interpretation under which the two planes
    # genuinely share an eave. Fall back to union min-Y when either plane
    # has no wall-top matches.
    clusters_a = _plane_eave_y_clusters(plane_a, wall_tops, partner_plane=plane_b)
    clusters_b = _plane_eave_y_clusters(plane_b, wall_tops, partner_plane=plane_a)
    resolved = _resolve_eave_pair(clusters_a, clusters_b)
    if resolved is not None:
        eave_a_y, eave_b_y = resolved
        eave_a_source = "wall_tops"
        eave_b_source = "wall_tops"
    else:
        eave_a_y = _plane_eave_y(plane_a, poly_a)
        eave_b_y = _plane_eave_y(plane_b, poly_b)
        eave_a_source = "union_min_y"
        eave_b_source = "union_min_y"
    if eave_a_y is None or eave_b_y is None:
        eave_height_parity = 0.0
        eave_diff_m: float | None = None
    else:
        eave_diff_m = abs(eave_a_y - eave_b_y)
        eave_height_parity = math.exp(
            -((eave_diff_m / EAVE_HEIGHT_PARITY_SIGMA_M) ** 2)
        )

    # --- ridge_reach_gap: diagnostic only (not used in score).
    # We tried treating asymmetric scan tops as a "one plane doesn't reach the
    # ridge" signal, but corpus-wide testing (user feedback on c87c1e25,
    # e0155eef, 16784bad) showed gable scans frequently have 3+ m asymmetry
    # from occlusion / partial coverage alone. The check produced too many
    # false positives on real gables, so it's retained only as metadata.
    scan_top_a = pg_a.get("scan_y_top")
    scan_top_b = pg_b.get("scan_y_top")
    if scan_top_a is None or scan_top_b is None:
        ridge_reach_gap_m: float | None = None
    else:
        ridge_reach_gap_m = abs(float(scan_top_a) - float(scan_top_b))

    components = {
        "horizontality": round(horizontality, 4),
        "azimuth_opposition": round(azimuth_opposition, 4),
        "inclination_match": round(inclination_match, 4),
        "eave_height_parity": round(eave_height_parity, 4),
    }
    eps = 1e-6
    score = (
        max(horizontality, eps)
        * max(azimuth_opposition, eps)
        * max(inclination_match, eps)
        * max(eave_height_parity, eps)
    ) ** (1.0 / 4.0)

    # For the viewer: split the pair domain by the ridge so the UI can color
    # each side. Side assignment is not used for scoring; it's purely
    # presentational and robust to pieces the ridge doesn't split (e.g. a
    # ridge tangent to an extension lobe).
    ridge_mid_xz = 0.5 * (r0_xz + r1_xz)
    ridge_line_xz = ShapelyLineString([tuple(r0_xz), tuple(r1_xz)])
    pieces = _split_scan_by_ridge(pair_domain, r0_xz, r1_xz)
    sig_pieces = [p for p in pieces if p.area >= 0.02 * pair_domain.area]
    side_a, side_b = (
        _assign_sides(sig_pieces, ridge_mid_xz, ds_a, ds_b)
        if sig_pieces
        else (None, None)
    )

    def _polygon_coords(g) -> list[list[float]]:
        if g is None or g.is_empty:
            return []
        out: list[list[float]] = []
        for gg in getattr(g, "geoms", [g]):
            if hasattr(gg, "exterior"):
                out.append([[float(x), float(y)] for x, y in gg.exterior.coords])
        return out

    eave_tol = ridge_len * 0.01 + 0.05
    eave_a_len, eave_a_geom = (
        _eave_geometry(side_a, ridge_line_xz, eave_tol) if side_a else (0.0, None)
    )
    eave_b_len, eave_b_geom = (
        _eave_geometry(side_b, ridge_line_xz, eave_tol) if side_b else (0.0, None)
    )
    creator_eave_proximity_a = _creator_eave_proximity_score(pg_a, eave_a_geom)
    creator_eave_proximity_b = _creator_eave_proximity_score(pg_b, eave_b_geom)
    creator_eave_proximity = 0.5 * (creator_eave_proximity_a + creator_eave_proximity_b)

    def _line_coords(g) -> list[list[float]]:
        if g is None or g.is_empty:
            return []
        out: list[list[float]] = []
        for gg in getattr(g, "geoms", [g]):
            if hasattr(gg, "coords"):
                out.append([[float(x), float(y)] for x, y in gg.coords])
        return out

    return {
        "a_id": pg_a["rep_id"],
        "b_id": pg_b["rep_id"],
        "a_plane_group_id": pg_a["id"],
        "b_plane_group_id": pg_b["id"],
        "a_member_ids": pg_a["member_ids"],
        "b_member_ids": pg_b["member_ids"],
        "score": round(score, 4),
        "components": components,
        "azimuth_deviation_deg": round(az_dev_deg, 3),
        "inclination_diff_deg": round(incl_diff_deg, 3),
        "eave_y_diff_m": None if eave_diff_m is None else round(eave_diff_m, 3),
        "ridge_reach_gap_m": None
        if ridge_reach_gap_m is None
        else round(ridge_reach_gap_m, 3),
        "scan_y_max_a": None
        if pg_a.get("scan_y_max") is None
        else round(float(pg_a["scan_y_max"]), 3),
        "scan_y_max_b": None
        if pg_b.get("scan_y_max") is None
        else round(float(pg_b["scan_y_max"]), 3),
        "scan_y_top_a": None if scan_top_a is None else round(float(scan_top_a), 3),
        "scan_y_top_b": None if scan_top_b is None else round(float(scan_top_b), 3),
        "eave_a_y": None if eave_a_y is None else round(eave_a_y, 3),
        "eave_b_y": None if eave_b_y is None else round(eave_b_y, 3),
        "eave_a_source": eave_a_source,
        "eave_b_source": eave_b_source,
        "ridge_xz": [
            [float(r0_xz[0]), float(r0_xz[1])],
            [float(r1_xz[0]), float(r1_xz[1])],
        ],
        "ridge_y": [float(r0_3d[1]), float(r1_3d[1])],
        "ridge_inclination_deg": round(ridge_incl_deg, 3),
        "side_a_xz": _polygon_coords(side_a),
        "side_b_xz": _polygon_coords(side_b),
        "side_a_area": round(float(side_a.area), 3) if side_a else 0.0,
        "side_b_area": round(float(side_b.area), 3) if side_b else 0.0,
        "eave_a_xz": _line_coords(eave_a_geom),
        "eave_b_xz": _line_coords(eave_b_geom),
        "eave_a_length": round(eave_a_len, 3),
        "eave_b_length": round(eave_b_len, 3),
        "creator_eave_proximity_a": round(creator_eave_proximity_a, 4),
        "creator_eave_proximity_b": round(creator_eave_proximity_b, 4),
        "creator_eave_proximity": round(creator_eave_proximity, 4),
    }


def _score_building(
    entry: dict,
    wall_tops_by_uuid: dict[str, np.ndarray] | None = None,
    scan_y_max_by_parent: dict[str, float] | None = None,
    building_fp_by_uuid: dict[str, ShapelyPolygon] | None = None,
    envelope_fp_by_uuid: dict[str, ShapelyPolygon] | None = None,
) -> dict:
    uuid = entry["building_uuid"]
    fp_ring = entry.get("scan_footprint_xz")
    scan_poly: ShapelyPolygon | None = None
    if fp_ring and len(fp_ring) >= 3:
        try:
            scan_poly = ShapelyPolygon(fp_ring)
            if not scan_poly.is_valid:
                scan_poly = scan_poly.buffer(0)
            if scan_poly.is_empty:
                scan_poly = None
        except Exception:
            scan_poly = None

    wall_tops = (wall_tops_by_uuid or {}).get(uuid)

    cands = entry.get("candidates") or []
    n_before_exterior_gate = len(cands)
    n_filtered_interior = 0
    # Cubic-envelope exterior gate applied at the PLANE-GROUP level.
    # Candidates with the same plane equation get bucketed together by
    # _build_plane_groups; a plane-group is exterior iff at least ONE of
    # its members has physical scan reaching the top-story wall tops. This
    # way a plane that has some pieces scanned at roof level and other
    # pieces scanned at cellar level (common for gable ends where one side
    # is visible and the other is partially interior) still counts as a
    # roof plane, and all its pieces — including the cellar-parented ones
    # — are kept.
    wall_top_y = _top_story_wall_top_y(wall_tops) if wall_tops is not None else None
    building_fp = (building_fp_by_uuid or {}).get(uuid)
    envelope_fp = (envelope_fp_by_uuid or {}).get(uuid)
    plane_groups = _build_plane_groups(uuid, cands, scan_poly, building_fp, envelope_fp)
    # Stamp two scan-Y statistics onto each plane-group, derived from the
    # per-parent scan_y_max values (max Y across a parent merged_roof_segment's
    # corners):
    #   * ``scan_y_max``      — hard max across parents. Used by the exterior
    #                           gate: a plane-group is "rain-facing" iff AT
    #                           LEAST ONE of its parents reaches the top-story
    #                           wall tops.
    #   * ``scan_y_top``      — robust top estimate (median of top-3 per-parent
    #                           maxes, falling back to the max when fewer than
    #                           3 parents). Used by ridge_reach_parity: a
    #                           chimney / dormer protrusion on an otherwise
    #                           ~10 m roof can pull the hard max up to 16 m,
    #                           which kills a valid gable pair; the median of
    #                           the top-3 parents ignores the single outlier.
    for pg in plane_groups:
        member_parent_ids = [
            m.get("parent_segment_id") for m in pg.get("members") or []
        ]
        if scan_y_max_by_parent:
            parent_maxes = [
                scan_y_max_by_parent.get(pid) for pid in member_parent_ids if pid
            ]
            parent_maxes = sorted(
                (v for v in parent_maxes if v is not None), reverse=True
            )
        else:
            parent_maxes = []
        pg["scan_y_max"] = parent_maxes[0] if parent_maxes else None
        if len(parent_maxes) >= 3:
            pg["scan_y_top"] = parent_maxes[1]  # median of top-3
        elif parent_maxes:
            pg["scan_y_top"] = parent_maxes[0]
        else:
            pg["scan_y_top"] = None
    if wall_top_y is not None and scan_y_max_by_parent and plane_groups:
        # Exterior gate at the PLANE-COEFFICIENT level, not per connected
        # component. The connected-components split (see _build_plane_groups)
        # can carve a single real plane into multiple disjoint plane-groups
        # when the building has separate wings on the same slope (e.g. a main
        # house + a detached east wing whose roof shares the main plane's
        # pitch/azimuth). If the scan reached roof-top elevation on the main
        # cluster, the east wing's cluster is still a real roof plane — even
        # if its own parents never reached wall-top. Filter interior only
        # when the entire plane (all components combined) stays below the
        # gate.
        gate_threshold = wall_top_y - EXTERIOR_SCAN_TOL_M
        plane_passes: dict[tuple, bool] = {}
        for pg in plane_groups:
            plane_coeffs = pg["key"][0]
            max_scan_y = pg.get("scan_y_max")
            passes = max_scan_y is not None and max_scan_y >= gate_threshold
            plane_passes[plane_coeffs] = plane_passes.get(plane_coeffs, False) or passes

        # Symmetry credit: a plane that fails the direct gate still counts as
        # a real roof plane if it has a structural mirror partner (antiparallel
        # downslope + adjacent footprint) whose plane passes the gate. Physical
        # rationale: a gable has two slopes by construction; if the scan saw
        # one slope at roof height, the opposing slope exists even when the
        # scan only captured it from below wall-tops (surveyor path occluded,
        # neighbouring structure blocking the view, etc.).
        passing_groups = [
            pg for pg in plane_groups if plane_passes.get(pg["key"][0], False)
        ]
        az_tol_cos = -math.cos(math.radians(AZIMUTH_PAIR_TOL_DEG))
        for pg in plane_groups:
            coeffs = pg["key"][0]
            if plane_passes.get(coeffs, False):
                continue
            ds_a = _downslope_xz(pg["plane"])
            poly_a: ShapelyPolygon = pg["union"]
            for pg_b in passing_groups:
                ds_b = _downslope_xz(pg_b["plane"])
                if float(np.dot(ds_a, ds_b)) > az_tol_cos:
                    continue
                try:
                    if poly_a.distance(pg_b["union"]) > MIN_FOOTPRINT_OVERLAP_OR_ADJ_M:
                        continue
                except Exception:
                    continue
                plane_passes[coeffs] = True
                break

        kept_groups: list[dict] = []
        for pg in plane_groups:
            plane_coeffs = pg["key"][0]
            if plane_passes.get(plane_coeffs, False):
                kept_groups.append(pg)
            else:
                n_filtered_interior += len(pg.get("members") or [])
        plane_groups = kept_groups

    pairs: list[dict] = []
    for i, pg_a in enumerate(plane_groups):
        for pg_b in plane_groups[i + 1 :]:
            p = _score_plane_pair(pg_a, pg_b, scan_poly, wall_tops)
            if p is not None:
                pairs.append(p)

    # Plane-group-level best partner. ``best_ridge_xz`` keeps the XZ endpoints
    # of the ridge from this pair so the summary can split the plane-group
    # union at it (physical side vs above-ridge extrapolation).
    best_per_group: dict[str, dict] = {}
    for p in pairs:
        for side_pg, other_pg, _side_rep, other_rep in (
            ("a_plane_group_id", "b_plane_group_id", "a_id", "b_id"),
            ("b_plane_group_id", "a_plane_group_id", "b_id", "a_id"),
        ):
            pgid = p[side_pg]
            prev = best_per_group.get(pgid)
            if _pair_beats(prev, p):
                best_per_group[pgid] = {
                    "best_partner_plane_group_id": p[other_pg],
                    "best_partner_id": p[other_rep],
                    "best_score": p["score"],
                    "best_components": p["components"],
                    "best_ridge_xz": p.get("ridge_xz"),
                    "creator_eave_proximity": p.get("creator_eave_proximity"),
                }

    # Lookup: candidate id -> plane-group id
    cand_to_group: dict[str, str] = {}
    group_by_id: dict[str, dict] = {pg["id"]: pg for pg in plane_groups}
    for pg in plane_groups:
        for cid in pg["member_ids"]:
            cand_to_group[cid] = pg["id"]

    # Plane-group selection: a plane-group survives if either
    #   (a) its best mirror-pair score clears SELECTION_SCORE_THRESHOLD, or
    #   (b) it has no structurally valid mirror partner at all (unpaired),
    #       having already passed the exterior gate above — i.e. it is a real
    #       rain-facing roof plane whose would-be partner was filtered out as
    #       interior (shed dormers, one-sided wings, half-hips where the
    #       opposite slope is hidden behind an adjacent structure).
    # Plane-groups that HAVE at least one structurally valid pair but score
    # below the threshold stay red: the mirror-parity check rejected them.
    paired_pgids: set[str] = set(best_per_group.keys())
    all_pgids: set[str] = {pg["id"] for pg in plane_groups}
    unpaired_pgids: set[str] = all_pgids - paired_pgids
    selected_pgids: set[str] = {
        pgid
        for pgid, best in best_per_group.items()
        if best.get("best_score", 0.0) >= SELECTION_SCORE_THRESHOLD
    } | unpaired_pgids

    scored_candidates = []
    for c in cands:
        pgid = cand_to_group.get(c["id"])
        best = best_per_group.get(pgid) if pgid else None
        scored_candidates.append(
            {
                "id": c["id"],
                "parent_segment_id": c.get("parent_segment_id"),
                "area_m2": c.get("area_m2"),
                "azimuth_deg": c.get("azimuth_deg"),
                "inclination_deg": c.get("inclination_deg"),
                "extended": c.get("extended"),
                "plane_group_id": pgid,
                "is_plane_group_rep": bool(
                    pgid and group_by_id[pgid]["rep_id"] == c["id"]
                ),
                "best_partner_id": best.get("best_partner_id") if best else None,
                "best_partner_plane_group_id": best.get("best_partner_plane_group_id")
                if best
                else None,
                "best_score": best.get("best_score") if best else None,
                "best_components": best.get("best_components") if best else None,
                "selected": bool(pgid in selected_pgids),
            }
        )

    # Per-building envelope = union of selected plane-groups' footprints,
    # clipped to scan footprint. Viewer renders this as the "surviving"
    # roof surface with interior/unmatched planes suppressed.
    envelope_xz: list[list[float]] | None = None
    envelope_area_m2: float = 0.0
    selected_polys = [
        pg["union"]
        for pg in plane_groups
        if pg["id"] in selected_pgids and pg.get("union") is not None
    ]
    if selected_polys:
        env = unary_union(selected_polys)
        if scan_poly is not None and not scan_poly.is_empty:
            env = env.intersection(scan_poly)
        if not env.is_empty:
            envelope_area_m2 = float(env.area)
            ring_coords = _coords_from_union(env)
            if ring_coords is not None and len(ring_coords) >= 3:
                envelope_xz = [[float(x), float(z)] for x, z in ring_coords]

    # Emit per-plane-group union polygon (exterior ring in XZ) and the
    # representative plane equation, so the viewer can render ONE unified
    # face per selected plane-group instead of fragmented Phase A pieces.
    # Pieces are split by intersections with every other candidate plane —
    # including dropped red ones — so unfragmenting the surviving plane
    # recovers the footprint that clipping had taken from it.
    def _pg_union_ring(pg: dict) -> list[list[float]] | None:
        poly = pg.get("union")
        if poly is None or poly.is_empty:
            return None
        ring = _coords_from_union(poly)
        if ring is None or len(ring) < 3:
            return None
        return [[float(x), float(z)] for x, z in ring]

    def _poly_to_rings(poly) -> list[list[list[float]]]:
        if poly is None or poly.is_empty:
            return []
        rings: list[list[list[float]]] = []
        for geom in getattr(poly, "geoms", [poly]):
            if not hasattr(geom, "exterior"):
                continue
            coords = list(geom.exterior.coords)
            if len(coords) < 4:
                continue
            rings.append([[float(x), float(z)] for x, z in coords[:-1]])
        return rings

    def _split_pg_union_at_ridge(
        pg: dict, ridge_xz: list[list[float]] | None
    ) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
        """Split plane-group union at the ridge; return (below, above) rings.

        ``below_ridge`` = pieces on the plane's downslope side of the ridge
        (physical roof surface). ``above_ridge`` = pieces on the upslope side,
        i.e. Phase A's extrapolation past the ridge into the partner plane's
        domain (the viewer paints these light blue).
        """
        poly = pg.get("union")
        plane = pg.get("plane")
        if poly is None or poly.is_empty or not plane or not ridge_xz:
            return [], []
        try:
            r0 = np.array(ridge_xz[0], dtype=float)
            r1 = np.array(ridge_xz[1], dtype=float)
        except Exception:
            return [], []
        if float(np.linalg.norm(r1 - r0)) < 1e-6:
            return [], []
        pieces = _split_scan_by_ridge(poly, r0, r1)
        if len(pieces) <= 1:
            return [], []
        ds = _downslope_xz(plane)
        ridge_mid = 0.5 * (r0 + r1)
        below: list[ShapelyPolygon] = []
        above: list[ShapelyPolygon] = []
        for piece in pieces:
            if piece.is_empty:
                continue
            c = piece.centroid
            offset = np.array([c.x - ridge_mid[0], c.y - ridge_mid[1]], dtype=float)
            if float(np.dot(offset, ds)) >= 0.0:
                below.append(piece)
            else:
                above.append(piece)
        below_poly = unary_union(below) if below else None
        above_poly = unary_union(above) if above else None
        return _poly_to_rings(below_poly), _poly_to_rings(above_poly)

    plane_group_summary = []
    for pg in plane_groups:
        best = best_per_group.get(pg["id"], {})
        ridge_xz = best.get("best_ridge_xz") if pg["id"] in selected_pgids else None
        below_rings, above_rings = _split_pg_union_at_ridge(pg, ridge_xz)
        plane_group_summary.append(
            {
                "id": pg["id"],
                "rep_id": pg["rep_id"],
                "member_ids": pg["member_ids"],
                "total_area": pg["total_area"],
                "max_support": pg["max_support"],
                "azimuth_deg": pg["azimuth_deg"],
                "inclination_deg": pg["inclination_deg"],
                "scan_y_max": pg.get("scan_y_max"),
                "scan_y_top": pg.get("scan_y_top"),
                "n_components_in_parent": pg.get("n_components_in_parent", 1),
                "best_partner_plane_group_id": best.get("best_partner_plane_group_id"),
                "best_score": best.get("best_score"),
                "best_components": best.get("best_components"),
                "best_ridge_xz": best.get("best_ridge_xz"),
                "creator_eave_proximity": best.get("creator_eave_proximity"),
                "selected": bool(pg["id"] in selected_pgids),
                "plane": pg.get("plane"),
                "union_xz": _pg_union_ring(pg),
                "union_below_ridge_xz": below_rings,
                "union_above_ridge_xz": above_rings,
            }
        )

    return {
        "building_uuid": uuid,
        "has_scan_footprint": scan_poly is not None,
        "wall_top_y": wall_top_y,
        "n_candidates_input": n_before_exterior_gate,
        "n_filtered_interior": n_filtered_interior,
        "n_candidates": len(
            [c for c in cands if c.get("area_m2", 0.0) >= MIN_CANDIDATE_AREA_M2]
        ),
        "n_plane_groups": len(plane_groups),
        "n_plane_groups_selected": len(selected_pgids),
        "n_pairs": len(pairs),
        "envelope_xz": envelope_xz,
        "envelope_area_m2": envelope_area_m2,
        "candidates": scored_candidates,
        "plane_groups": plane_group_summary,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default="reports/candidate_faces_20260419/candidates.json",
        type=Path,
    )
    ap.add_argument(
        "--buildings",
        default="reconcile/buildings_3d.json",
        type=Path,
        help="buildings_3d.json — top-story walls_merged provide physical eave "
        "elevations",
    )
    ap.add_argument(
        "--v3-results",
        default="reconcile/reconcile_v3_results.json",
        type=Path,
        help="reconcile_v3_results.json — merged_roof_segments provide physical scan Y "
        "extents",
    )
    ap.add_argument(
        "--out",
        default="reports/ridge_eave_scores_20260420/scores.json",
        type=Path,
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    wall_tops_by_uuid: dict[str, np.ndarray] = {}
    building_fp_by_uuid: dict[str, ShapelyPolygon] = {}
    envelope_fp_by_uuid: dict[str, ShapelyPolygon] = {}
    if args.buildings.exists():
        print(f"Loading {args.buildings}...")
        with args.buildings.open() as handle:
            buildings_3d = json.load(handle)
        for b in buildings_3d:
            uuid = b.get("uuid")
            if not uuid:
                continue
            wall_tops_by_uuid[uuid] = _extract_top_story_wall_tops(b)
            fp = _building_fp_union(b)
            if fp is not None:
                building_fp_by_uuid[uuid] = fp
            env = _building_envelope_fp(b)
            if env is not None:
                envelope_fp_by_uuid[uuid] = env
        n_with = sum(1 for a in wall_tops_by_uuid.values() if len(a) >= 2)
        print(
            f"  {n_with}/{len(wall_tops_by_uuid)} buildings have ≥2 top-story wall tops"
        )
        print(
            f"  {len(building_fp_by_uuid)}/{len(wall_tops_by_uuid)} buildings have "
            f"room-floor footprint unions"
        )
        print(
            f"  {len(envelope_fp_by_uuid)}/{len(wall_tops_by_uuid)} buildings have "
            f"envelope (rooms + gaps) unions"
        )
    else:
        print(f"WARNING: {args.buildings} not found; falling back to union min-Y eaves")

    scan_y_max_by_parent: dict[str, float] = {}
    if args.v3_results.exists():
        print(f"Loading {args.v3_results}...")
        scan_y_max_by_parent = _load_scan_y_max_by_parent(args.v3_results)
        print(f"  {len(scan_y_max_by_parent)} merged_roof_segment scan-Y extents")
    else:
        print(
            f"WARNING: {args.v3_results} not found; exterior-gate filter disabled "
            "— interior-plane candidates will not be dropped"
        )

    print(f"Loading {args.candidates}...")
    import ijson

    t0 = time.time()
    results: list[dict] = []
    n_scored = 0
    n_total_pairs = 0
    n_total_planes = 0

    with args.candidates.open("rb") as handle:
        for entry in ijson.items(handle, "item", use_float=True):
            if args.limit and n_scored >= args.limit:
                break
            try:
                out = _score_building(
                    entry,
                    wall_tops_by_uuid,
                    scan_y_max_by_parent,
                    building_fp_by_uuid,
                    envelope_fp_by_uuid,
                )
            except Exception as exc:
                print(
                    f"  error {entry.get('building_uuid')}: {type(exc).__name__}: {exc}"
                )
                continue
            results.append(out)
            n_scored += 1
            n_total_pairs += out["n_pairs"]
            n_total_planes += out["n_plane_groups"]
            if n_scored % 25 == 0:
                print(
                    f"  {n_scored} buildings "
                    f"({n_total_planes} planes, {n_total_pairs} pairs, "
                    f"{time.time() - t0:.1f}s)"
                )

    elapsed = time.time() - t0
    print(
        f"\nScored {n_scored} buildings, {n_total_planes} planes, "
        f"{n_total_pairs} pairs in {elapsed:.1f}s."
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidates_source": str(args.candidates),
        "scoring": {
            "level": "plane-subgroup",
            "PLANE_KEY_DECIMALS": PLANE_KEY_DECIMALS,
            "PLANE_GROUP_CONNECT_BUFFER_M": PLANE_GROUP_CONNECT_BUFFER_M,
            "PLANE_GROUP_CLOSING_M": PLANE_GROUP_CLOSING_M,
            "PLANE_GROUP_FLOOD_REACH_M": PLANE_GROUP_FLOOD_REACH_M,
            "MIN_SUBGROUP_AREA_M2": MIN_SUBGROUP_AREA_M2,
            "AZIMUTH_PAIR_TOL_DEG": AZIMUTH_PAIR_TOL_DEG,
            "MIN_FOOTPRINT_OVERLAP_OR_ADJ_M": MIN_FOOTPRINT_OVERLAP_OR_ADJ_M,
            "HORIZONTALITY_SIGMA_DEG": HORIZONTALITY_SIGMA_DEG,
            "aggregation": "gm_of_4_mirror_parity",
            "AZIMUTH_OPPOSITION_SIGMA_DEG": AZIMUTH_OPPOSITION_SIGMA_DEG,
            "INCLINATION_MATCH_SIGMA_DEG": INCLINATION_MATCH_SIGMA_DEG,
            "EAVE_HEIGHT_PARITY_SIGMA_M": EAVE_HEIGHT_PARITY_SIGMA_M,
            "WALL_TOP_MATCH_TOL_M": WALL_TOP_MATCH_TOL_M,
            "WALL_TOP_EPS_M": WALL_TOP_EPS_M,
            "EXTERIOR_SCAN_TOL_M": EXTERIOR_SCAN_TOL_M,
            "SELECTION_SCORE_THRESHOLD": SELECTION_SCORE_THRESHOLD,
            "eave_y_source": "top_story_wall_tops_with_union_min_y_fallback",
            "exterior_gate": "scan_y_max >= top_story_wall_top_y - EXTERIOR_SCAN_TOL_M",
            "selection_rule": "plane_group.best_score >= SELECTION_SCORE_THRESHOLD",
            "components": [
                "horizontality",
                "azimuth_opposition",
                "inclination_match",
                "eave_height_parity",
                "ridge_reach_parity",
            ],
        },
        "buildings": results,
    }
    with args.out.open("w") as handle:
        json.dump(payload, handle)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
