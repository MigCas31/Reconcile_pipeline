"""Bands 2+4 advanced feature computers.

Finer per-segment features that need either

    - the per-building wall index (Band 2: source-wall aggregates), or
    - shape/moment/normal math that's too niche for the baseline expander
      (Band 4: Hu moments, Fourier descriptors, inertia ellipse, radial
      signature, spherical variance / Fisher kappa).

All functions are pure and return flat ``dict[str, float|int|bool|None]``.
Missing inputs produce Nones; no exceptions leak out.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..audit import note_geom_skip

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _polygon_xz(corners_xyz: Sequence[Sequence[float]]) -> Polygon | None:
    if not corners_xyz or len(corners_xyz) < 3:
        return None
    try:
        p = Polygon([(float(c[0]), float(c[2])) for c in corners_xyz])
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            return None
        return p
    except Exception:
        return None


def _polygon_2d(points: Sequence[Sequence[float]]) -> Polygon | None:
    if not points or len(points) < 3:
        return None
    try:
        p = Polygon([(float(q[0]), float(q[1])) for q in points])
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            return None
        return p
    except Exception:
        return None


def _stats(values: Sequence[float], prefix: str) -> dict[str, float | None]:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not xs:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_range": None,
            f"{prefix}_median": None,
        }
    arr = np.asarray(xs, dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=0)),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_range": float(arr.max() - arr.min()),
        f"{prefix}_median": float(np.median(arr)),
    }


def _azimuth_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return float(d if d <= 180.0 else 360.0 - d)


def _wall_azimuth_incl_length_top_bottom(
    corners: Sequence[Sequence[float]],
) -> dict[str, float | None]:
    """
    Summarize a wall polygon: azimuth of its vertical plane, tilt, length, y range.
    """
    if not corners or len(corners) < 3:
        return {
            "azimuth_deg": None,
            "incl_deg": None,
            "length_m": None,
            "top_y_m": None,
            "bottom_y_m": None,
            "centroid_x": None,
            "centroid_z": None,
        }
    pts = np.asarray(corners, dtype=float)
    ys = pts[:, 1]
    top_y = float(ys.max())
    bot_y = float(ys.min())
    # XZ spread tells us the wall's horizontal azimuth. Take the principal
    # axis of the XZ point cloud — robust to the variable vertex count and
    # to walls that aren't axis-aligned rectangles.
    xz = pts[:, [0, 2]] - pts[:, [0, 2]].mean(axis=0)
    try:
        cov = np.cov(xz.T)
        _eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, -1]  # largest-eigenvalue axis
    except Exception:
        principal = None
    if principal is None or not np.all(np.isfinite(principal)):
        az = None
    else:
        # atan2(x, z) convention: 0 = +Z, 90 = +X.
        ang = math.degrees(math.atan2(principal[0], principal[1]))
        # Fold to [0, 180): walls are bidirectional.
        if ang < 0:
            ang += 180.0
        if ang >= 180.0:
            ang -= 180.0
        az = float(ang)
    # Length = largest pairwise distance in XZ.
    n = len(pts)
    max_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(math.hypot(pts[i, 0] - pts[j, 0], pts[i, 2] - pts[j, 2]))
            if d > max_d:
                max_d = d
    # Inclination: compute best-fit plane normal via SVD and measure tilt
    # from vertical (90° = truly vertical wall).
    centered = pts - pts.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        nl = float(np.linalg.norm(normal))
        if nl > _EPS:
            normal = normal / nl
            # Tilt from horizontal = acos(|ny|). Wall tilt from vertical is
            # (90 - that) so we report |90 - acos(|ny|)|.
            incl_from_horiz = math.degrees(
                math.acos(max(-1.0, min(1.0, abs(float(normal[1])))))
            )
            wall_tilt_from_vertical = float(abs(90.0 - incl_from_horiz))
        else:
            wall_tilt_from_vertical = None
    except Exception:
        wall_tilt_from_vertical = None
    cx, cz = pts[:, 0].mean(), pts[:, 2].mean()
    return {
        "azimuth_deg": az,
        "incl_deg": wall_tilt_from_vertical,
        "length_m": float(max_d),
        "top_y_m": top_y,
        "bottom_y_m": bot_y,
        "centroid_x": float(cx),
        "centroid_z": float(cz),
    }


# ---------------------------------------------------------------------------
# Category K — Band 2 source-wall aggregates
# ---------------------------------------------------------------------------


_WALL_FEATURE_EMPTY = {
    "swall_resolved_count": 0,
    "swall_resolution_ratio": None,
    "swall_length_mean": None,
    "swall_length_std": None,
    "swall_length_min": None,
    "swall_length_max": None,
    "swall_length_total": None,
    "swall_top_y_mean": None,
    "swall_top_y_std": None,
    "swall_top_y_range": None,
    "swall_bottom_y_mean": None,
    "swall_bottom_y_std": None,
    "swall_azimuth_mean": None,
    "swall_azimuth_std": None,
    "swall_azimuth_max_diff_deg": None,
    "swall_incl_mean_deg": None,
    "swall_incl_std_deg": None,
    "swall_incl_max_deg": None,
    "swall_incl_is_nonvertical_fraction": None,
    "swall_centroid_to_seg_mean_m": None,
    "swall_centroid_to_seg_max_m": None,
    "swall_fraction_walls_merged": None,
    "swall_fraction_same_room_as_slab": None,
    "swall_fraction_on_footprint_boundary": None,
    "swall_stories_touched": None,
    "swall_unique_stories": 0,
}


def source_wall_features(
    cluster_members: Sequence[dict] | None,
    wall_index: dict[str, dict[str, Any]] | None,
    building_boundary_xz: Sequence[Sequence[float]] | None,
    segment_poly: Polygon | None,
) -> dict[str, Any]:
    """Aggregate stats over the walls that seeded this merged segment."""
    if not cluster_members or not wall_index:
        return dict(_WALL_FEATURE_EMPTY)
    resolved: list[dict[str, Any]] = []
    total_members = 0
    for m in cluster_members:
        wid = m.get("source_wall_id")
        if not wid:
            continue
        total_members += 1
        info = wall_index.get(wid)
        if info is None:
            continue
        wf = _wall_azimuth_incl_length_top_bottom(info.get("corners") or [])
        wf["wall_id"] = wid
        wf["room_id"] = info.get("room_id")
        wf["kind"] = info.get("kind")
        wf["story"] = info.get("story")
        wf["slab_room_id"] = m.get("slab_room_id")
        resolved.append(wf)
    out: dict[str, Any] = dict(_WALL_FEATURE_EMPTY)
    out["swall_resolved_count"] = len(resolved)
    out["swall_resolution_ratio"] = (
        float(len(resolved) / total_members) if total_members else None
    )
    if not resolved:
        return out
    lengths = [w["length_m"] for w in resolved if w.get("length_m") is not None]
    top_ys = [w["top_y_m"] for w in resolved if w.get("top_y_m") is not None]
    bot_ys = [w["bottom_y_m"] for w in resolved if w.get("bottom_y_m") is not None]
    azs = [w["azimuth_deg"] for w in resolved if w.get("azimuth_deg") is not None]
    incls = [w["incl_deg"] for w in resolved if w.get("incl_deg") is not None]
    if lengths:
        arr = np.asarray(lengths, dtype=float)
        out["swall_length_mean"] = float(arr.mean())
        out["swall_length_std"] = float(arr.std(ddof=0))
        out["swall_length_min"] = float(arr.min())
        out["swall_length_max"] = float(arr.max())
        out["swall_length_total"] = float(arr.sum())
    if top_ys:
        arr = np.asarray(top_ys, dtype=float)
        out["swall_top_y_mean"] = float(arr.mean())
        out["swall_top_y_std"] = float(arr.std(ddof=0))
        out["swall_top_y_range"] = float(arr.max() - arr.min())
    if bot_ys:
        arr = np.asarray(bot_ys, dtype=float)
        out["swall_bottom_y_mean"] = float(arr.mean())
        out["swall_bottom_y_std"] = float(arr.std(ddof=0))
    if azs:
        # Circular stats on [0, 180) (walls are undirected). Double the
        # angle to put it on [0, 360) before averaging.
        rad2 = np.radians([2.0 * a for a in azs])
        c, s = float(np.mean(np.cos(rad2))), float(np.mean(np.sin(rad2)))
        mean2 = math.degrees(math.atan2(s, c))
        if mean2 < 0:
            mean2 += 360.0
        out["swall_azimuth_mean"] = float(mean2 / 2.0)
        r = math.sqrt(c * c + s * s)
        out["swall_azimuth_std"] = float(
            math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(r, _EPS))))) / 2.0
        )
        # Pairwise max azimuth diff in the undirected [0, 180) space.
        max_diff = 0.0
        for i in range(len(azs)):
            for j in range(i + 1, len(azs)):
                d = _azimuth_diff_deg(azs[i], azs[j])
                # Fold to [0, 90] since walls are bidirectional.
                if d > 90.0:
                    d = 180.0 - d
                if d > max_diff:
                    max_diff = d
        out["swall_azimuth_max_diff_deg"] = float(max_diff)
    if incls:
        arr = np.asarray(incls, dtype=float)
        out["swall_incl_mean_deg"] = float(arr.mean())
        out["swall_incl_std_deg"] = float(arr.std(ddof=0))
        out["swall_incl_max_deg"] = float(arr.max())
        # "Nonvertical" = tilted more than 2° from vertical.
        out["swall_incl_is_nonvertical_fraction"] = float(np.mean(arr > 2.0))

    # Distance from wall centroid to the merged segment centroid (XZ).
    if segment_poly is not None:
        cxcz = np.asarray(
            [
                [w["centroid_x"], w["centroid_z"]]
                for w in resolved
                if w.get("centroid_x") is not None
            ],
            dtype=float,
        )
        if len(cxcz):
            seg_c = np.asarray([segment_poly.centroid.x, segment_poly.centroid.y])
            dists = np.linalg.norm(cxcz - seg_c, axis=1)
            out["swall_centroid_to_seg_mean_m"] = float(dists.mean())
            out["swall_centroid_to_seg_max_m"] = float(dists.max())

    kinds = [w.get("kind") for w in resolved]
    out["swall_fraction_walls_merged"] = (
        float(sum(1 for k in kinds if k == "walls_merged") / len(kinds))
        if kinds
        else None
    )

    # How many share a room with the slab (the thing the roof caps).
    slabs = [w.get("slab_room_id") for w in resolved]
    room_ids = [w.get("room_id") for w in resolved]
    if slabs and room_ids:
        same = sum(
            1 for s, r in zip(slabs, room_ids, strict=False) if s and r and s == r
        )
        out["swall_fraction_same_room_as_slab"] = float(same / len(resolved))

    # Stories touched by the source walls.
    stories = {w.get("story") for w in resolved if w.get("story") is not None}
    out["swall_unique_stories"] = len(stories)
    out["swall_stories_touched"] = float(len(stories)) if stories else 0.0

    # Wall centroids inside vs outside building footprint boundary.
    boundary = _polygon_2d(building_boundary_xz or [])
    if boundary is not None:
        try:
            boundary_line = boundary.boundary
            dists = []
            for w in resolved:
                cx, cz = w.get("centroid_x"), w.get("centroid_z")
                if cx is None or cz is None:
                    continue
                from shapely.geometry import Point

                dists.append(float(Point(cx, cz).distance(boundary_line)))
            if dists:
                on_boundary = sum(1 for d in dists if d < 0.25)
                out["swall_fraction_on_footprint_boundary"] = float(
                    on_boundary / len(dists)
                )
        except Exception as exc:
            note_geom_skip(exc, "advanced_features.swall_fraction")
    return out


# ---------------------------------------------------------------------------
# Category L — Band 4 shape descriptors
# ---------------------------------------------------------------------------


def _raw_moments(xy: np.ndarray, p: int, q: int) -> float:
    """Polygon raw moment m_{pq} via the shoelace form."""
    n = len(xy)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        # Raw moment integrand for shoelace polygon. See
        # Steger (1996), "On the calculation of arbitrary moments of polygons".
        s += cross * _moment_integrand(x1, y1, x2, y2, p, q)
    return float(s / _moment_norm(p, q))


def _moment_integrand(
    x1: float, y1: float, x2: float, y2: float, p: int, q: int
) -> float:
    # Steger's closed form for the specific (p, q) we use stays cheap up to
    # fourth order. Fall back to a numerical line integral if we ever need
    # higher orders.
    if (p, q) == (0, 0):
        return 1.0
    if (p, q) == (1, 0):
        return (x1 + x2) / 2.0
    if (p, q) == (0, 1):
        return (y1 + y2) / 2.0
    if (p, q) == (2, 0):
        return (x1 * x1 + x1 * x2 + x2 * x2) / 3.0
    if (p, q) == (0, 2):
        return (y1 * y1 + y1 * y2 + y2 * y2) / 3.0
    if (p, q) == (1, 1):
        return (2 * x1 * y1 + x1 * y2 + x2 * y1 + 2 * x2 * y2) / 6.0
    if (p, q) == (3, 0):
        return (x1**3 + x1 * x1 * x2 + x1 * x2 * x2 + x2**3) / 4.0
    if (p, q) == (0, 3):
        return (y1**3 + y1 * y1 * y2 + y1 * y2 * y2 + y2**3) / 4.0
    if (p, q) == (2, 1):
        return (
            3 * x1 * x1 * y1
            + x1 * x1 * y2
            + 2 * x1 * x2 * y1
            + 2 * x1 * x2 * y2
            + x2 * x2 * y1
            + 3 * x2 * x2 * y2
        ) / 12.0
    if (p, q) == (1, 2):
        return (
            3 * y1 * y1 * x1
            + y1 * y1 * x2
            + 2 * y1 * y2 * x1
            + 2 * y1 * y2 * x2
            + y2 * y2 * x1
            + 3 * y2 * y2 * x2
        ) / 12.0
    # Fall back: trapezoidal rule on 8 samples per edge — still exact enough
    # for our orders but we shouldn't hit it.
    samples = 16
    t = np.linspace(0.0, 1.0, samples)
    xs = x1 + (x2 - x1) * t
    ys = y1 + (y2 - y1) * t
    return float(np.mean((xs**p) * (ys**q)))


def _moment_norm(p: int, q: int) -> float:
    # Shoelace gives 2A for moment (0,0). For general (p, q) the normalizer
    # is (p + q + 2)(p + q + 1)/… but Steger's formulas are already
    # integrated. We absorb the factor-of-2 for (0,0) only.
    return 2.0 if (p, q) == (0, 0) else 1.0


def _central_moments(xy: np.ndarray) -> dict[tuple[int, int], float]:
    m00 = _raw_moments(xy, 0, 0)
    if abs(m00) < _EPS:
        return {}
    m10 = _raw_moments(xy, 1, 0)
    m01 = _raw_moments(xy, 0, 1)
    cx = m10 / m00
    cy = m01 / m00
    translated = xy - np.array([cx, cy])
    mu: dict[tuple[int, int], float] = {(0, 0): m00}
    for p, q in [
        (2, 0),
        (0, 2),
        (1, 1),
        (3, 0),
        (0, 3),
        (2, 1),
        (1, 2),
    ]:
        mu[(p, q)] = _raw_moments(translated, p, q)
    return mu


def _hu_moments(mu: dict[tuple[int, int], float]) -> list[float | None]:
    """Seven Hu moments from the normalized central moments."""
    m00 = mu.get((0, 0), 0.0)
    if abs(m00) < _EPS:
        return [None] * 7
    # Normalized central moments η_{pq} = μ_{pq} / |m00|^(1 + (p+q)/2).
    # Use |m00| — raw polygon moments are signed by winding order, but Hu
    # moments are defined on positive area.
    m00_abs = abs(m00)
    eta: dict[tuple[int, int], float] = {}
    for (p, q), v in mu.items():
        if (p, q) == (0, 0):
            continue
        eta[(p, q)] = v / (m00_abs ** (1.0 + (p + q) / 2.0))
    try:
        n20 = eta[(2, 0)]
        n02 = eta[(0, 2)]
        n11 = eta[(1, 1)]
        n30 = eta[(3, 0)]
        n03 = eta[(0, 3)]
        n21 = eta[(2, 1)]
        n12 = eta[(1, 2)]
    except KeyError:
        return [None] * 7
    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4.0 * n11**2
    h3 = (n30 - 3.0 * n12) ** 2 + (3.0 * n21 - n03) ** 2
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = (n30 - 3.0 * n12) * (n30 + n12) * (
        (n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2
    ) + (3.0 * n21 - n03) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4.0 * n11 * (
        n30 + n12
    ) * (n21 + n03)
    h7 = (3.0 * n21 - n03) * (n30 + n12) * (
        (n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2
    ) - (n30 - 3.0 * n12) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    # Log-scaled signed Hu moments are easier to compare across shapes.
    return [_signed_log(x) for x in (h1, h2, h3, h4, h5, h6, h7)]


def _signed_log(x: float) -> float | None:
    if not np.isfinite(x):
        return None
    if abs(x) < _EPS:
        return 0.0
    return float(math.copysign(math.log10(abs(x)), x))


def _inertia_ellipse(mu: dict[tuple[int, int], float]) -> dict[str, float | None]:
    m00 = mu.get((0, 0), 0.0)
    if abs(m00) < _EPS:
        return {
            "inertia_major_axis_m": None,
            "inertia_minor_axis_m": None,
            "inertia_eccentricity": None,
            "inertia_orientation_deg": None,
        }
    m20 = mu.get((2, 0), 0.0) / m00
    m02 = mu.get((0, 2), 0.0) / m00
    m11 = mu.get((1, 1), 0.0) / m00
    disc = math.sqrt(max(0.0, (m20 - m02) ** 2 + 4.0 * m11**2))
    # Semi-axes of the equivalent ellipse with the same area + inertia.
    lam1 = (m20 + m02 + disc) / 2.0
    lam2 = (m20 + m02 - disc) / 2.0
    if lam1 < 0 or lam2 < 0:
        return {
            "inertia_major_axis_m": None,
            "inertia_minor_axis_m": None,
            "inertia_eccentricity": None,
            "inertia_orientation_deg": None,
        }
    major = 2.0 * math.sqrt(max(lam1, lam2))
    minor = 2.0 * math.sqrt(max(min(lam1, lam2), 0.0))
    ecc = None
    if major > _EPS:
        ecc = float(math.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))
    orientation = math.degrees(0.5 * math.atan2(2.0 * m11, max(m20 - m02, _EPS)))
    return {
        "inertia_major_axis_m": float(major),
        "inertia_minor_axis_m": float(minor),
        "inertia_eccentricity": ecc,
        "inertia_orientation_deg": float(orientation),
    }


def _fourier_descriptors(xy: np.ndarray, n_descriptors: int = 8) -> list[float | None]:
    """Magnitudes of the first ``n_descriptors`` Fourier coefficients of the
    complex boundary signal, normalized by the magnitude of the 1st
    non-DC coefficient. Returns signed log to keep the feature range tame."""
    if len(xy) < 4:
        return [None] * n_descriptors
    samples = 64
    # Resample uniformly along the closed boundary.
    closed = np.vstack([xy, xy[:1]])
    seg_lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < _EPS:
        return [None] * n_descriptors
    t = np.linspace(0.0, total, samples, endpoint=False)
    resampled = np.empty((samples, 2))
    for i, ti in enumerate(t):
        idx = np.searchsorted(cum, ti, side="right") - 1
        idx = max(0, min(idx, len(seg_lens) - 1))
        seg = seg_lens[idx]
        if seg < _EPS:
            resampled[i] = closed[idx]
            continue
        alpha = (ti - cum[idx]) / seg
        resampled[i] = closed[idx] + alpha * (closed[idx + 1] - closed[idx])
    signal = resampled[:, 0] + 1j * resampled[:, 1]
    signal -= signal.mean()  # translation invariance
    coeffs = np.fft.fft(signal)
    mags = np.abs(coeffs)
    if mags[1] < _EPS:
        return [None] * n_descriptors
    normed = mags / mags[1]  # scale-invariant
    out: list[float | None] = []
    for k in range(1, n_descriptors + 1):
        if k < len(normed):
            out.append(_signed_log(float(normed[k])))
        else:
            out.append(None)
    return out


def _radial_signature(xy: np.ndarray, n_samples: int = 16) -> list[float]:
    """Sample the distance from centroid to boundary around the polygon."""
    if len(xy) < 3:
        return [0.0] * n_samples
    cx = float(xy[:, 0].mean())
    cy = float(xy[:, 1].mean())
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return [0.0] * n_samples
    out = []
    # Large enough to leave the polygon for any reasonable roof.
    R = float(
        max(
            np.ptp(xy[:, 0]),
            np.ptp(xy[:, 1]),
            1.0,
        )
        * 2.0
    )
    from shapely.geometry import LineString

    boundary = poly.boundary
    for i in range(n_samples):
        theta = 2.0 * math.pi * i / n_samples
        ex, ey = cx + R * math.cos(theta), cy + R * math.sin(theta)
        try:
            ray = LineString([(cx, cy), (ex, ey)])
            hit = ray.intersection(boundary)
            if hit.is_empty:
                out.append(0.0)
                continue
            # Take the nearest point on the intersection (ray may graze
            # multiple edges).
            if hit.geom_type == "Point":
                dx = hit.x - cx
                dy = hit.y - cy
            elif hasattr(hit, "geoms"):
                best = None
                for g in hit.geoms:
                    for pt in g.coords if hasattr(g, "coords") else [(g.x, g.y)]:
                        d = math.hypot(pt[0] - cx, pt[1] - cy)
                        if best is None or d < best[0]:
                            best = (d, pt)
                if best is None:
                    out.append(0.0)
                    continue
                dx = best[1][0] - cx
                dy = best[1][1] - cy
            else:
                coords = list(hit.coords)
                pt = min(coords, key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
                dx = pt[0] - cx
                dy = pt[1] - cy
            out.append(math.hypot(dx, dy))
        except Exception:
            out.append(0.0)
    return out


def shape_descriptor_features(
    corners_xyz: Sequence[Sequence[float]],
) -> dict[str, Any]:
    poly = _polygon_xz(corners_xyz)
    if poly is None:
        out: dict[str, Any] = {f"hu_log_{i + 1}": None for i in range(7)}
        out.update({f"fourier_log_{i + 1}": None for i in range(8)})
        out.update(
            {
                "inertia_major_axis_m": None,
                "inertia_minor_axis_m": None,
                "inertia_eccentricity": None,
                "inertia_orientation_deg": None,
                "reock_compactness": None,
                "schwartzberg_compactness": None,
                "polsby_popper": None,
                "radial_mean_m": None,
                "radial_std_m": None,
                "radial_cv": None,
                "radial_min_m": None,
                "radial_max_m": None,
            }
        )
        return out
    xy = np.asarray([(float(c[0]), float(c[2])) for c in corners_xyz], dtype=float)
    # Hu moments (log-scaled, signed).
    mu = _central_moments(xy)
    hu = _hu_moments(mu)
    # Fourier descriptors (log-scaled magnitudes, normalized).
    fd = _fourier_descriptors(xy, n_descriptors=8)
    # Inertia ellipse.
    ine = _inertia_ellipse(mu)
    # Reock: area / (π r²) where r = min enclosing circle radius.
    # Approximate min enclosing circle via the convex hull's farthest-pair.
    hull = poly.convex_hull
    if hasattr(hull, "exterior"):
        coords = np.asarray(hull.exterior.coords)[:-1]
        if len(coords):
            center = coords.mean(axis=0)
            r_enc = float(np.linalg.norm(coords - center, axis=1).max())
            reock = (
                float(poly.area) / (math.pi * r_enc * r_enc) if r_enc > _EPS else None
            )
        else:
            reock = None
    else:
        reock = None
    # Schwartzberg = perimeter / circumference of equivalent-area circle.
    equiv_r = math.sqrt(poly.area / math.pi) if poly.area > 0 else 0.0
    schwartzberg = (
        (2.0 * math.pi * equiv_r) / float(poly.length) if poly.length > _EPS else None
    )
    # Polsby-Popper = 4π A / P² — same family as compactness but keep it
    # explicitly (some literature uses one, some the other).
    polsby = (
        4.0 * math.pi * float(poly.area) / (float(poly.length) ** 2)
        if poly.length > _EPS
        else None
    )
    radial = _radial_signature(xy, n_samples=16)
    if radial:
        r_arr = np.asarray(radial, dtype=float)
        r_mean = float(r_arr.mean())
        r_std = float(r_arr.std(ddof=0))
        r_min = float(r_arr.min())
        r_max = float(r_arr.max())
        r_cv = float(r_std / r_mean) if r_mean > _EPS else None
    else:
        r_mean = r_std = r_min = r_max = r_cv = None
    out: dict[str, Any] = {}
    for i, v in enumerate(hu):
        out[f"hu_log_{i + 1}"] = v
    for i, v in enumerate(fd):
        out[f"fourier_log_{i + 1}"] = v
    out.update(ine)
    out["reock_compactness"] = reock
    out["schwartzberg_compactness"] = schwartzberg
    out["polsby_popper"] = polsby
    out["radial_mean_m"] = r_mean
    out["radial_std_m"] = r_std
    out["radial_cv"] = r_cv
    out["radial_min_m"] = r_min
    out["radial_max_m"] = r_max
    return out


# ---------------------------------------------------------------------------
# Category M — 3D vertex / edge geometry
# ---------------------------------------------------------------------------


def vertex3d_features(
    corners_xyz: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Interior vertex angles, turning angles, edge-direction entropy, sharp corners."""
    out: dict[str, Any] = {
        "vangle_mean_deg": None,
        "vangle_std_deg": None,
        "vangle_min_deg": None,
        "vangle_max_deg": None,
        "vangle_sum_defect_deg": None,
        "turning_angle_mean_deg": None,
        "turning_angle_std_deg": None,
        "turning_angle_abs_sum_deg": None,
        "sharp_corner_count": None,
        "edge_direction_entropy": None,
        "bbox3d_volume_m3": None,
        "bbox3d_diagonal_m": None,
    }
    if not corners_xyz or len(corners_xyz) < 3:
        return out
    pts = np.asarray(corners_xyz, dtype=float)
    n = len(pts)
    angles: list[float] = []
    turns: list[float] = []
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        v1 = a - b
        v2 = c - b
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < _EPS or l2 < _EPS:
            continue
        cos_t = float(np.dot(v1, v2) / (l1 * l2))
        cos_t = max(-1.0, min(1.0, cos_t))
        interior = math.degrees(math.acos(cos_t))
        angles.append(interior)
        turns.append(180.0 - interior)
    if angles:
        arr = np.asarray(angles, dtype=float)
        out["vangle_mean_deg"] = float(arr.mean())
        out["vangle_std_deg"] = float(arr.std(ddof=0))
        out["vangle_min_deg"] = float(arr.min())
        out["vangle_max_deg"] = float(arr.max())
        out["vangle_sum_defect_deg"] = float(360.0 - abs(180.0 * (n - 2) - arr.sum()))
    if turns:
        arr = np.asarray(turns, dtype=float)
        out["turning_angle_mean_deg"] = float(arr.mean())
        out["turning_angle_std_deg"] = float(arr.std(ddof=0))
        out["turning_angle_abs_sum_deg"] = float(np.sum(np.abs(arr)))
        out["sharp_corner_count"] = int(np.sum(arr > 60.0))
    # Edge-direction entropy (8 azimuth bins in XZ).
    if n >= 2:
        bins = np.zeros(8)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            dx = float(b[0] - a[0])
            dz = float(b[2] - a[2])
            az = math.degrees(math.atan2(dx, dz))
            # Fold to [0, 180) (edges are undirected).
            if az < 0:
                az += 180.0
            if az >= 180.0:
                az -= 180.0
            bins[int((az / 180.0) * 8) % 8] += 1.0
        total = bins.sum()
        if total > 0:
            p = bins / total
            # Shannon entropy in nats, normalized to [0, 1] by log(8).
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0))
            out["edge_direction_entropy"] = float(ent / math.log(8.0))
    # 3D bbox.
    if n >= 1:
        bx = pts[:, 0]
        by = pts[:, 1]
        bz = pts[:, 2]
        vol = float(
            (bx.max() - bx.min()) * (by.max() - by.min()) * (bz.max() - bz.min())
        )
        diag = float(
            math.sqrt(
                (bx.max() - bx.min()) ** 2
                + (by.max() - by.min()) ** 2
                + (bz.max() - bz.min()) ** 2
            )
        )
        out["bbox3d_volume_m3"] = vol
        out["bbox3d_diagonal_m"] = diag
    return out


# ---------------------------------------------------------------------------
# Category N — cluster normal statistics
# ---------------------------------------------------------------------------


def normal_stats_features(cluster_members: Sequence[dict] | None) -> dict[str, Any]:
    """Directional stats on unit normals of cluster members."""
    out: dict[str, Any] = {
        "normals_spherical_variance": None,
        "normals_mean_resultant_length": None,
        "normals_fisher_kappa": None,
        "normals_d_entropy": None,
        "normals_pairwise_cos_min": None,
        "normals_pairwise_cos_mean": None,
        "member_room_entropy": None,
        "member_wall_entropy": None,
    }
    if not cluster_members:
        return out
    ns = []
    ds = []
    for m in cluster_members:
        p = m.get("plane")
        if not p or len(p) < 3:
            continue
        nv = np.asarray(p[:3], dtype=float)
        norm = float(np.linalg.norm(nv))
        if norm < _EPS:
            continue
        ns.append(nv / norm)
        if len(p) >= 4:
            ds.append(float(p[3]) / norm)
    if ns:
        arr = np.stack(ns)
        N = len(arr)
        mean_vec = arr.mean(axis=0)
        R = float(np.linalg.norm(mean_vec))
        mean_resultant = R  # in [0, 1]
        out["normals_mean_resultant_length"] = mean_resultant
        out["normals_spherical_variance"] = float(1.0 - mean_resultant)
        # Fisher-Watson kappa estimator for 3D unit vectors:
        #   κ ≈ (3 R̄ - R̄³) / (1 - R̄²).
        if mean_resultant < 1.0 - _EPS:
            kappa = (3.0 * mean_resultant - mean_resultant**3) / (
                1.0 - mean_resultant**2
            )
            out["normals_fisher_kappa"] = float(kappa)
        if N >= 2:
            dots = arr @ arr.T
            off = dots[~np.eye(N, dtype=bool)]
            abs_off = np.abs(off)
            out["normals_pairwise_cos_min"] = float(abs_off.min())
            out["normals_pairwise_cos_mean"] = float(abs_off.mean())
    if ds and len(ds) >= 2:
        # Binned entropy on d (10 bins across observed range).
        arr = np.asarray(ds, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi - lo > _EPS:
            bins = np.linspace(lo, hi, 11)
            hist, _ = np.histogram(arr, bins=bins)
            p = hist.astype(float) / float(hist.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0))
            out["normals_d_entropy"] = float(ent / math.log(10.0))
        else:
            out["normals_d_entropy"] = 0.0

    # Membership entropies (how fragmented the rooms/walls are).
    def _entropy(counts: dict[str, int]) -> float | None:
        total = sum(counts.values())
        if total == 0:
            return None
        p = np.asarray(list(counts.values()), dtype=float) / float(total)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0))
        # Normalize by log(#distinct) to keep in [0, 1].
        denom = math.log(len(counts)) if len(counts) > 1 else 1.0
        return float(ent / denom) if denom > 0 else 0.0

    room_counts: dict[str, int] = {}
    wall_counts: dict[str, int] = {}
    for m in cluster_members:
        r = m.get("source_room_id")
        w = m.get("source_wall_id")
        if r:
            room_counts[r] = room_counts.get(r, 0) + 1
        if w:
            wall_counts[w] = wall_counts.get(w, 0) + 1
    out["member_room_entropy"] = _entropy(room_counts)
    out["member_wall_entropy"] = _entropy(wall_counts)
    return out


# ---------------------------------------------------------------------------
# Category O — IoU / overlap with neighbors
# ---------------------------------------------------------------------------


def iou_features(
    segment_poly: Polygon | None,
    building_boundary_xz: Sequence[Sequence[float]] | None,
    building_context: dict | None,
) -> dict[str, Any]:
    """IoU with building footprint, part footprint, nearest final roof."""
    out: dict[str, Any] = {
        "iou_building_footprint": None,
        "cover_of_building_footprint": None,
        "iou_part_footprint": None,
        "cover_of_part_footprint": None,
        "iou_nearest_final_roof": None,
        "cover_of_nearest_final_roof": None,
        "final_roof_count_overlapping": 0,
        "final_roof_union_area_m2": None,
    }
    if segment_poly is None:
        return out
    s_area = float(segment_poly.area)

    def _iou(p: Polygon | None) -> tuple[float | None, float | None]:
        if p is None or p.is_empty:
            return (None, None)
        try:
            inter = segment_poly.intersection(p)
            inter_a = float(inter.area) if not inter.is_empty else 0.0
            union = segment_poly.union(p)
            union_a = float(union.area) if not union.is_empty else 0.0
            iou = inter_a / union_a if union_a > _EPS else None
            cover = inter_a / s_area if s_area > _EPS else None
            return (iou, cover)
        except Exception:
            return (None, None)

    boundary = _polygon_2d(building_boundary_xz or [])
    iou_b, cov_b = _iou(boundary)
    out["iou_building_footprint"] = iou_b
    out["cover_of_building_footprint"] = cov_b

    if building_context:
        parts = building_context.get("parts") or []
        centroid = segment_poly.centroid
        best_part_poly = None
        best_part_area = -1.0
        for p in parts:
            fp = p.get("footprint_xz") or []
            if not fp:
                continue
            first = fp[0]
            try:
                if len(first) == 3:
                    poly = Polygon([(float(c[0]), float(c[2])) for c in fp])
                else:
                    poly = Polygon([(float(c[0]), float(c[1])) for c in fp])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty or poly.area <= 0:
                    continue
            except Exception as exc:
                note_geom_skip(exc, "advanced_features.part_poly")
                continue
            if poly.contains(centroid) and poly.area > best_part_area:
                best_part_poly = poly
                best_part_area = poly.area
        iou_p, cov_p = _iou(best_part_poly)
        out["iou_part_footprint"] = iou_p
        out["cover_of_part_footprint"] = cov_p

        # Final slanted roofs — find best-overlap one.
        slanted = building_context.get("slanted_roofs") or []
        best_iou = None
        best_cov = None
        overlaps = 0
        union_polys = []
        for sr in slanted:
            c = sr.get("corners") or []
            if not c:
                continue
            sr_poly = _polygon_xz(c)
            if sr_poly is None:
                continue
            try:
                inter = segment_poly.intersection(sr_poly)
                if inter.is_empty:
                    continue
                inter_a = float(inter.area)
                if inter_a <= 0:
                    continue
                overlaps += 1
                union_polys.append(sr_poly)
                union = segment_poly.union(sr_poly)
                iou = inter_a / float(union.area) if union.area > _EPS else None
                cov = inter_a / s_area if s_area > _EPS else None
                if best_iou is None or (iou is not None and iou > best_iou):
                    best_iou = iou
                    best_cov = cov
            except Exception as exc:
                note_geom_skip(exc, "advanced_features.iou_final_roof")
                continue
        out["iou_nearest_final_roof"] = best_iou
        out["cover_of_nearest_final_roof"] = best_cov
        out["final_roof_count_overlapping"] = int(overlaps)
        if union_polys:
            try:
                u = unary_union(union_polys)
                out["final_roof_union_area_m2"] = float(u.area)
            except Exception:
                out["final_roof_union_area_m2"] = float(
                    sum(p.area for p in union_polys)
                )
    return out


# ---------------------------------------------------------------------------
# Top-level assembly (called from scripts/analyze_labels.py)
# ---------------------------------------------------------------------------


def advanced_features(
    record: dict,
    *,
    wall_index: dict[str, dict[str, Any]] | None,
    building_context: dict | None,
) -> dict[str, Any]:
    corners = record.get("segment_corners_xyz") or []
    boundary = record.get("building_boundary_xz") or []
    cluster_members = record.get("cluster_members") or []
    seg_poly = _polygon_xz(corners)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out: dict[str, Any] = {}
        out.update(
            source_wall_features(cluster_members, wall_index, boundary, seg_poly)
        )
        out.update(shape_descriptor_features(corners))
        out.update(vertex3d_features(corners))
        out.update(normal_stats_features(cluster_members))
        out.update(iou_features(seg_poly, boundary, building_context))
    return out
