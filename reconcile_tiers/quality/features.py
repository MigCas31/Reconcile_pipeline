"""Feature extraction for the roof_quality predictor.

Reads only the runtime ``TierPayload`` — same module is called from the build
pipeline (``score_building``) and from the offline fitter
(``fit_calibration.py``) to enforce no train/serve skew.

The chosen features are the ones that correlate with the rater's manual labels
on the corpus of 212 buildings. Tier classification dominates; ceiling
fragmentation, computed-oblique inflation, and the three watertightness
metrics (``overlap_ratio``, ``stitch_gap_count``, ``max_pair_normal_delta_deg``)
discriminate where tier alone saturates.
"""

from __future__ import annotations

import math

from reconcile_tiers.payload.schema import (
    CeilingSource,
    GapKind,
    GapScope,
    TierPayload,
    Vec3,
)

MAX_PIECES_FOR_PAIRS = 80
"""Cap pair-wise normal-delta computation. On a pathological 200-piece building
the O(k^2) loop would dominate runtime; ``ceiling_piece_count`` already
saturates the bad-quality signal at that scale, so the pair metric loses no
information when capped."""

_STITCH_KINDS = {GapKind.STITCH, GapKind.STITCH_FLOOR, GapKind.STITCH_CEIL}


def extract_features(payload: TierPayload) -> dict[str, float]:
    """Return a flat feature dict for `payload`.

    All values are floats — categorical signals (tier 1..8) are one-hot
    encoded as ``tier_1`` .. ``tier_8`` columns. The fitter consumes this
    same dict at training time.
    """
    cls = payload.classification
    ceiling = payload.ceiling
    pieces_for_geom = _largest_pieces_by_area(ceiling, MAX_PIECES_FOR_PAIRS)

    features: dict[str, float] = {}
    for tier_idx in range(1, 9):
        features[f"tier_{tier_idx}"] = 1.0 if cls.tier == tier_idx else 0.0
    features["n_stories"] = float(cls.n_stories)
    features["n_rooms"] = float(cls.n_rooms)

    features["ceiling_piece_count"] = float(len(ceiling))
    features["log1p_ceiling_piece_count"] = math.log1p(len(ceiling))
    features["computed_oblique_count"] = float(
        sum(1 for piece in ceiling if piece.source == CeilingSource.COMPUTED_OBLIQUE)
    )
    features["dormer_face_count"] = float(len(payload.dormer_faces))
    features["cross_floor_gap_count"] = float(
        sum(1 for gap in payload.gaps if gap.scope == GapScope.INTER_STORY)
    )
    features["stitch_gap_count"] = float(
        sum(1 for gap in payload.gaps if gap.kind in _STITCH_KINDS)
    )
    features["overlap_ratio"] = _overlap_ratio(ceiling)
    features["max_pair_normal_delta_deg"] = _max_pair_normal_delta_deg(pieces_for_geom)
    features["support_quality_p10"] = _percentile(
        [piece.support_quality for piece in ceiling],
        0.1,
        default=1.0,
    )
    return features


def feature_names() -> list[str]:
    """Stable ordering of feature columns. The fitter and runtime must agree."""
    return [
        *(f"tier_{i}" for i in range(1, 9)),
        "n_stories",
        "n_rooms",
        "ceiling_piece_count",
        "log1p_ceiling_piece_count",
        "computed_oblique_count",
        "dormer_face_count",
        "cross_floor_gap_count",
        "stitch_gap_count",
        "overlap_ratio",
        "max_pair_normal_delta_deg",
        "support_quality_p10",
    ]


def _xz_area(corners: list[Vec3]) -> float:
    if len(corners) < 3:
        return 0.0
    n = len(corners)
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += corners[i].x * corners[j].z - corners[j].x * corners[i].z
    return abs(total) * 0.5


def _xz_aabb(corners: list[Vec3]) -> tuple[float, float, float, float] | None:
    if not corners:
        return None
    xs = [c.x for c in corners]
    zs = [c.z for c in corners]
    return min(xs), max(xs), min(zs), max(zs)


def _aabb_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax0, ax1, az0, az1 = a
    bx0, bx1, bz0, bz1 = b
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oz = max(0.0, min(az1, bz1) - max(az0, bz0))
    return ox * oz


def _overlap_ratio(pieces: list) -> float:
    """Sum of per-piece XZ-area divided by the area of the bounding box that
    encloses every piece. >1 means pieces stack on each other in plan view —
    the visual signature of fragmentation in rated-1-2 buildings."""
    if not pieces:
        return 0.0
    total_area = 0.0
    xs: list[float] = []
    zs: list[float] = []
    for piece in pieces:
        total_area += _xz_area(piece.corners)
        for corner in piece.corners:
            xs.append(corner.x)
            zs.append(corner.z)
    if not xs:
        return 0.0
    bbox = (max(xs) - min(xs)) * (max(zs) - min(zs))
    if bbox <= 1e-9:
        return 0.0
    return total_area / bbox


def _max_pair_normal_delta_deg(pieces: list) -> float:
    """Maximum normal-angle Δ across XZ-overlapping piece pairs.

    >70° flags stuck-on perpendicular pieces (chimneys, dormers misplaced as
    walls). Clean gables peak at 30-50° (the actual ridge angle).
    """
    if len(pieces) < 2:
        return 0.0
    aabbs = [_xz_aabb(piece.corners) for piece in pieces]
    areas = [_xz_area(piece.corners) for piece in pieces]
    best = 0.0
    for i in range(len(pieces)):
        if aabbs[i] is None or areas[i] <= 0.0:
            continue
        for j in range(i + 1, len(pieces)):
            if aabbs[j] is None or areas[j] <= 0.0:
                continue
            overlap = _aabb_overlap(aabbs[i], aabbs[j])
            if overlap < 0.3 * min(areas[i], areas[j]):
                continue
            delta = _normal_delta_deg(pieces[i].plane, pieces[j].plane)
            if delta > best:
                best = delta
    return min(best, 90.0)


def _normal_delta_deg(plane_a, plane_b) -> float:
    n1 = (plane_a.a, plane_a.b, plane_a.c)
    n2 = (plane_b.a, plane_b.b, plane_b.c)
    norm1 = math.sqrt(sum(x * x for x in n1)) + 1e-12
    norm2 = math.sqrt(sum(x * x for x in n2)) + 1e-12
    dot = sum(a * b for a, b in zip(n1, n2, strict=False)) / (norm1 * norm2)
    dot = max(-1.0, min(1.0, abs(dot)))
    return math.degrees(math.acos(dot))


def _percentile(values: list[float], q: float, *, default: float) -> float:
    if not values:
        return default
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _largest_pieces_by_area(pieces: list, cap: int) -> list:
    if len(pieces) <= cap:
        return list(pieces)
    indexed = [(i, _xz_area(piece.corners)) for i, piece in enumerate(pieces)]
    indexed.sort(key=lambda item: item[1], reverse=True)
    keep = sorted(idx for idx, _area in indexed[:cap])
    return [pieces[idx] for idx in keep]
