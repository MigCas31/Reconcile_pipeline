"""Stair signal collection and room-pair scoring.

Two scan signals bias which `(L, U)` room pair carries the stair:
- Signal A: RoomPlan `stairs` objects (oriented bounding boxes from the
  scan's stairs metadata).
- Signal B: raw-ceiling / upper-floor overlap polygons.

Plus building-code-only candidates for adjacent rooms with no signal.

Extracted from `stairs.py`; `reconstruct_stairs` calls these as the first
phase of the pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import Point

# ─────────────── signal collection ───────────────


def _collect_signal_a(merged: dict[str, Any], story_y: dict[int, float]):
    """RoomPlan `stairs` OOBBs whose Y span straddles a story boundary."""
    from reconcile_tiers.assemble.stairs import _Signal

    out: list = []
    for o in merged.get("objects") or []:
        if not isinstance(o, dict):
            continue
        cat = o.get("category")
        if not (isinstance(cat, dict) and "stairs" in cat):
            continue
        transform = o.get("transform")
        dims = o.get("dimensions")
        if not (
            isinstance(transform, list)
            and len(transform) == 16
            and isinstance(dims, list)
            and len(dims) == 3
        ):
            continue
        T = np.array(transform, dtype=float).reshape(4, 4).T
        center = T[:3, 3]
        local_z = T[:3, 2]
        h = float(dims[1])
        y_lo = float(center[1] - h / 2)
        y_hi = float(center[1] + h / 2)
        # Find a story-pair (N, N+1) whose floor Ys straddle [y_lo, y_hi]
        sorted_st = sorted(story_y.keys())
        match = None
        for i in range(len(sorted_st) - 1):
            n, n1 = sorted_st[i], sorted_st[i + 1]
            yn, yn1 = story_y[n], story_y[n1]
            # OOBB connects if its Y-range overlaps the [yn, yn1] interval
            if y_lo - 0.5 <= yn1 and y_hi + 0.5 >= yn:
                match = (n, n1)
                break
        if match is None:
            continue
        out.append(
            _Signal(
                kind="A",
                centroid_xz=(float(center[0]), float(center[2])),
                heading_rad=float(math.atan2(local_z[0], local_z[2])),
                void_polygon=None,
                base_story=match[0],
                top_story=match[1],
                fragment_id=str(o.get("identifier") or ""),
            )
        )
    return out


def _collect_signal_b(ceiling_payload: list[dict[str, Any]], rooms: list):
    """Raw-ceiling polygons on story-N rooms whose XZ extent overlaps any
    story-(N+1) floor."""
    from reconcile_tiers.assemble.stairs import _polygon_xz, _Signal

    out: list = []
    by_idx = {r.room_index: r for r in rooms}
    for entry in ceiling_payload:
        if not (isinstance(entry, dict) and entry.get("source") == "raw_scan"):
            continue
        loc = entry.get("locator_id") or ""
        try:
            ri = int(loc.split("::")[-1].split(":")[0])
        except (ValueError, IndexError):
            continue
        lower = by_idx.get(ri)
        if lower is None:
            continue
        rc_poly = _polygon_xz(entry.get("corners") or [])
        if rc_poly is None or rc_poly.area < 1e-6:
            continue
        for upper in rooms:
            if upper.story != lower.story + 1:
                continue
            inter = rc_poly.intersection(upper.floor_xz)
            if inter.is_empty or inter.area < 0.05:
                continue
            inter_poly = (
                inter
                if inter.geom_type == "Polygon"
                else max(inter.geoms, key=lambda g: g.area)
            )
            cx, cz = inter_poly.centroid.x, inter_poly.centroid.y
            out.append(
                _Signal(
                    kind="B",
                    centroid_xz=(float(cx), float(cz)),
                    heading_rad=None,
                    void_polygon=inter_poly,
                    base_story=lower.story,
                    top_story=upper.story,
                    fragment_id=loc,
                )
            )
    return out


def _index_portals(merged: dict[str, Any]) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    for key in ("doors", "openings"):
        for o in merged.get(key) or []:
            if not isinstance(o, dict):
                continue
            transform = o.get("transform")
            if not (isinstance(transform, list) and len(transform) == 16):
                continue
            T = np.array(transform, dtype=float).reshape(4, 4).T
            xz = np.array([T[0, 3], T[2, 3]])
            story = o.get("story")
            if story is None:
                continue
            out.setdefault(int(story), []).append(xz)
    return out


# ─────────────── (L, U) connection scoring ───────────────


def _score_room_pairs(
    lowers: list,
    uppers: list,
    signals_a: list,
    signals_b: list,
) -> list:
    """Score every cross-story (L, U) pair. Each scan signal is attributed to
    its single best-matching pair (the one whose lower OR upper room sits
    closest to the signal's centroid / void); other pairs don't share it.
    Without that attribution, the same OOBB centroid would inflate the score
    of many pairs and we'd emit duplicate stairs.
    """
    from reconcile_tiers.assemble.stairs import (
        W_FOOTPRINT_OVERLAP,
        W_SIGNAL_A,
        W_SIGNAL_B,
        _Candidate,
    )

    if not lowers or not uppers:
        return []
    raw: list = []
    for L in lowers:
        for U in uppers:
            try:
                overlap = L.floor_xz.intersection(U.floor_xz).area
            except Exception:
                overlap = 0.0
            xz_dist = L.floor_xz.distance(U.floor_xz)
            # Keep any (L, U) within 5 m XZ — distant rooms aren't realistic
            # stair endpoints. Overlap is the strongest signal but we don't
            # require it (basement under partial footprint, etc.).
            if overlap < 0.10 and xz_dist > 5.0:
                continue
            raw.append((L, U, overlap))

    # Attribute each Signal A to its closest (L, U) pair only.
    a_assigned: dict = {}
    for idx, s in enumerate(signals_a):
        best = None
        best_score = math.inf
        for L, U, _ in raw:
            if s.base_story != L.story or s.top_story != U.story:
                continue
            pt = Point(s.centroid_xz)
            d = min(L.floor_xz.distance(pt), U.floor_xz.distance(pt))
            if d < best_score:
                best_score = d
                best = (L, U)
        if best is not None and best_score < 2.0:
            a_assigned[idx] = best
    # Attribute each Signal B similarly (use void polygon centroid).
    b_assigned: dict = {}
    for idx, s in enumerate(signals_b):
        if s.void_polygon is None:
            continue
        best = None
        best_score = math.inf
        for L, U, _ in raw:
            if s.base_story != L.story or s.top_story != U.story:
                continue
            inter = s.void_polygon.intersection(U.floor_xz)
            if inter.is_empty:
                continue
            d = -inter.area  # larger overlap → better (more negative)
            if d < best_score:
                best_score = d
                best = (L, U)
        if best is not None:
            b_assigned[idx] = best

    out: list[_Candidate] = []
    for L, U, overlap in raw:
        footprint_score = overlap / max(L.floor_xz.area, U.floor_xz.area, 1e-6)
        sigs_a = [
            signals_a[idx]
            for idx, pair in a_assigned.items()
            if pair is not None
            and pair[0].room_index == L.room_index
            and pair[1].room_index == U.room_index
        ]
        sigs_b = [
            signals_b[idx]
            for idx, pair in b_assigned.items()
            if pair is not None
            and pair[0].room_index == L.room_index
            and pair[1].room_index == U.room_index
        ]
        score = (
            W_FOOTPRINT_OVERLAP * footprint_score
            + W_SIGNAL_A * (1.0 if sigs_a else 0.0)
            + W_SIGNAL_B * (1.0 if sigs_b else 0.0)
        )
        out.append(
            _Candidate(
                base_story=L.story,
                top_story=U.story,
                lower=L,
                upper=U,
                signals_a=sigs_a,
                signals_b=sigs_b,
                connection_score=score,
            )
        )
    out.sort(key=lambda c: -c.connection_score)
    return out


# ─────────────── try-harder ladder per candidate ───────────────
