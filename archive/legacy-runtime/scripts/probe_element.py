"""Quick geometric probes for a single element ID -- stick-out, neighbors,
missing-at-point.

Used by the ``debug-element`` skill immediately after ``element_locator --trace``
to answer the three questions a user most often asks:

* Is this element sticking out beyond the building footprint?
* What is around it — which other atoms share the room/story/cluster, and how close?
* Is its stored role (``flat`` / ``sloped``) consistent with its actual inclination
  and with the roof surface overhead?
* Is its vertical level an outlier vs sibling floors/ceilings in the same room?
* (Missing segment) What, if anything, does the pipeline produce near a 3D point I
  can see in the viewer? Helps the user point at a hole in the model instead of an ID.

Two modes::

    python -m scripts.probe_element --element-id "<uuid>::<kind>::<id>" [--radius 3.0]
    python -m scripts.probe_element --uuid <uuid> --point X,Y,Z [--radius 3.0]

All output is JSON on stdout so the skill can ``jq`` it. Add ``--human`` for a
compact text summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from shapely.geometry import Polygon
from shapely.validation import make_valid

from reconcile.element_locator import (
    find_element,
    is_ontology_kind,
    is_tier_kind,
    parse_element_id,
)

# --- geometry helpers ------------------------------------------------------


def _poly_xz(corners_3d: list[list[float]]) -> list[tuple[float, float]]:
    """Project a 3D poly (y-up) to the horizontal (x, z) plane."""
    return [(float(p[0]), float(p[2])) for p in corners_3d if len(p) >= 3]


def _centroid_3d(corners_3d: list[list[float]]) -> tuple[float, float, float]:
    if not corners_3d:
        return (0.0, 0.0, 0.0)
    n = len(corners_3d)
    x = sum(float(p[0]) for p in corners_3d) / n
    y = sum(float(p[1]) for p in corners_3d) / n
    z = sum(float(p[2]) for p in corners_3d) / n
    return (x, y, z)


def _distance_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _safe_polygon(points_2d: list[tuple[float, float]]) -> Polygon | None:
    if len(points_2d) < 3:
        return None
    poly = Polygon(points_2d)
    if not poly.is_valid:
        fixed = make_valid(poly)
        # ``make_valid`` can return a MultiPolygon / GeometryCollection;
        # use the largest-area Polygon if present.
        if hasattr(fixed, "geoms"):
            polys = [g for g in fixed.geoms if isinstance(g, Polygon)]
            poly = max(polys, key=lambda p: p.area) if polys else poly
        elif isinstance(fixed, Polygon):
            poly = fixed
    return poly if poly.is_valid and not poly.is_empty else None


def _normalize_corners(corners: list) -> list[list[float]]:
    normalized: list[list[float]] = []
    for point in corners or []:
        if isinstance(point, dict):
            if {"x", "y", "z"}.issubset(point):
                normalized.append(
                    [
                        float(point["x"]),
                        float(point["y"]),
                        float(point["z"]),
                    ]
                )
            continue
        if len(point) >= 3:
            normalized.append([float(point[0]), float(point[1]), float(point[2])])
    return normalized


# --- flatness / role probe --------------------------------------------------

# Same gate `roof_partitioning.py` uses to split flat vs oblique ceiling atoms.
FLAT_MAX_DEG = 5.0
SLOPED_MIN_DEG = 5.0


def _surface_normal(corners_3d: list[list[float]]) -> tuple[float, float, float] | None:
    """Unit normal from the first three corners (right-hand rule)."""
    if len(corners_3d) < 3:
        return None
    p0 = [float(v) for v in corners_3d[0][:3]]
    p1 = [float(v) for v in corners_3d[1][:3]]
    p2 = [float(v) for v in corners_3d[2][:3]]
    ax, ay, az = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    bx, by, bz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag < 1e-9:
        return None
    return (nx / mag, ny / mag, nz / mag)


def _inclination_deg(normal: tuple[float, float, float] | None) -> float | None:
    """Angle from horizontal in degrees (0 = flat, 90 = vertical)."""
    if normal is None:
        return None
    ny = min(1.0, max(-1.0, abs(float(normal[1]))))
    return math.degrees(math.acos(ny))


def _role_from_kind(atom_kind: str | None, outer_kind: str) -> str:
    """Collapse any kind label into ``flat`` / ``sloped`` / ``unknown``."""
    k = (atom_kind or "").lower()
    if k == "flat":
        return "flat"
    if k in {"oblique", "sloped", "clipped", "simple_slant"}:
        return "sloped"
    o = (outer_kind or "").lower()
    if "flat" in o:
        return "flat"
    if "oblique" in o or "slant" in o:
        return "sloped"
    return "unknown"


def check_flatness(element: dict, roof_results_for_uuid: dict) -> dict:
    """Compare the element's measured inclination against its stored role and
    the roof surface directly overhead.

    Verdicts:
      * ``consistent_flat`` / ``consistent_sloped`` — stored role matches geometry.
      * ``stored_flat_but_tilted`` — the atom is labelled flat but its polygon
        normal tilts more than ``SLOPED_MIN_DEG`` from horizontal.
      * ``stored_sloped_but_horizontal`` — labelled sloped but effectively flat.
      * ``insufficient_data`` — missing corners or the kind couldn't be
        classified into flat/sloped.
    """
    corners = element.get("corners") or []
    outer_kind = element.get("kind") or ""
    atom_kind = element.get("atom_kind")
    stored_role = _role_from_kind(atom_kind, outer_kind)

    normal = _surface_normal(corners)
    inclination = _inclination_deg(normal)

    verdict = "insufficient_data"
    if inclination is not None and stored_role != "unknown":
        if stored_role == "flat" and inclination > SLOPED_MIN_DEG:
            verdict = "stored_flat_but_tilted"
        elif stored_role == "sloped" and inclination < FLAT_MAX_DEG:
            verdict = "stored_sloped_but_horizontal"
        elif stored_role == "flat":
            verdict = "consistent_flat"
        else:
            verdict = "consistent_sloped"

    # Roof surfaces that cover the element's 2D footprint. Several may stack
    # vertically (attic floor + sloped roof + flat roof); we need the one
    # closest in y to the element to cross-check the stored role.
    elem_centroid_y = _centroid_3d(corners)[1] if corners else None
    overhead: list[dict] = []
    target_poly = _safe_polygon(_poly_xz(corners))
    rs = roof_results_for_uuid.get("roof_surfaces", {}) or {}
    for subkind in ("oblique", "flat"):
        for rec in rs.get(subkind, []) or []:
            rec_corners = rec.get("corners") or []
            if len(rec_corners) < 3 or target_poly is None:
                continue
            rec_poly = _safe_polygon(_poly_xz(rec_corners))
            if rec_poly is None:
                continue
            overlap = float(target_poly.intersection(rec_poly).area)
            if overlap <= 0:
                continue
            n = _surface_normal(rec_corners)
            incl = _inclination_deg(n)
            rec_y = _centroid_3d(rec_corners)[1]
            overhead.append(
                {
                    "roof_kind": subkind,
                    "roof_inclination_deg": (
                        round(incl, 2) if incl is not None else None
                    ),
                    "xz_overlap_m2": round(overlap, 4),
                    "roof_centroid_y": round(rec_y, 4),
                    "vertical_delta_m": (
                        round(rec_y - elem_centroid_y, 4)
                        if elem_centroid_y is not None
                        else None
                    ),
                }
            )
    overhead.sort(
        key=lambda r: (
            abs(r["vertical_delta_m"]) if r["vertical_delta_m"] is not None else 1e9
        )
    )

    cross_checks: list[str] = []
    # Only cross-check when the primary measured-vs-stored signal is already
    # ambiguous or flagged. If the atom's own normal clearly matches its stored
    # role (``consistent_*``), don't second-guess it against a stacked roof
    # surface — that produces noise when attic/roof planes overlay a ceiling.
    nearest = overhead[0] if overhead else None
    primary_unclear = verdict in {
        "stored_flat_but_tilted",
        "stored_sloped_but_horizontal",
        "insufficient_data",
    }
    if (
        primary_unclear
        and nearest is not None
        and nearest.get("vertical_delta_m") is not None
        and abs(nearest["vertical_delta_m"]) < 1.0
    ):
        roof_incl = nearest.get("roof_inclination_deg") or 0.0
        if (
            stored_role == "flat"
            and nearest["roof_kind"] == "oblique"
            and roof_incl > 10
        ):
            cross_checks.append(
                f"nearest overhead roof is oblique @ {roof_incl}° "
                f"(xz overlap {nearest['xz_overlap_m2']} m², "
                f"dy={nearest['vertical_delta_m']}m)"
                " — corroborates sloped"
            )
        if stored_role == "sloped" and nearest["roof_kind"] == "flat" and roof_incl < 3:
            cross_checks.append(
                f"nearest overhead roof is flat @ {roof_incl}° "
                f"(xz overlap {nearest['xz_overlap_m2']} m², "
                f"dy={nearest['vertical_delta_m']}m)"
                " — corroborates flat"
            )

    return {
        "stored_role": stored_role,
        "stored_kind": atom_kind or outer_kind,
        "measured_inclination_deg": (
            round(inclination, 2) if inclination is not None else None
        ),
        "thresholds_deg": {"flat_max": FLAT_MAX_DEG, "sloped_min": SLOPED_MIN_DEG},
        "verdict": verdict,
        "overhead_roof_surfaces": overhead[:5],
        "cross_checks": cross_checks,
    }


# --- room-level probe -------------------------------------------------------

# Siblings within this band are treated as belonging to the same ceiling level.
ROOM_LEVEL_TOLERANCE_M = 0.3


def _y_bounds(corners_3d: list[list[float]]) -> tuple[float, float, float] | None:
    if not corners_3d:
        return None
    ys = [float(c[1]) for c in corners_3d if len(c) >= 2]
    if not ys:
        return None
    return (min(ys), max(ys), sum(ys) / len(ys))


def check_level(element: dict, roof_results_for_uuid: dict) -> dict:
    """Flag a floor/ceiling element whose vertical position is an outlier vs
    sibling partitions in the same (room_index, story).

    Catches cases like "ceiling partition 2 m above where the room's other
    ceilings sit" — i.e. the element was assigned to the wrong story or got
    its y from the wrong source geometry.
    """
    corners = element.get("corners") or []
    bounds = _y_bounds(corners)
    if bounds is None:
        return {"available": False, "reason": "element has no corners"}
    elem_min, elem_max, elem_mean = bounds

    room_index = element.get("room_index")
    story = element.get("story")
    elem_id = element.get("id")
    if room_index is None:
        return {
            "available": False,
            "reason": "element has no room_index — level check is per-room",
            "element_y_mean": round(elem_mean, 4),
        }

    siblings: list[dict] = []
    cp = roof_results_for_uuid.get("ceiling_partitions", {}) or {}
    for subkind in ("oblique", "flat"):
        for rec in cp.get(subkind, []) or []:
            if rec.get("id") == elem_id:
                continue
            if rec.get("room_index") != room_index:
                continue
            if story is not None and rec.get("story") != story:
                continue
            sib_bounds = _y_bounds(rec.get("poly") or [])
            if sib_bounds is None:
                continue
            sib_min, sib_max, sib_mean = sib_bounds
            siblings.append(
                {
                    "id": rec.get("id"),
                    "kind": rec.get("kind"),
                    "y_min": round(sib_min, 4),
                    "y_max": round(sib_max, 4),
                    "y_mean": round(sib_mean, 4),
                }
            )

    if not siblings:
        return {
            "available": True,
            "verdict": "no_siblings",
            "element_y_range": [round(elem_min, 4), round(elem_max, 4)],
            "element_y_mean": round(elem_mean, 4),
            "room_index": room_index,
            "story": story,
            "sibling_count": 0,
        }

    sib_y_min = min(s["y_min"] for s in siblings)
    sib_y_max = max(s["y_max"] for s in siblings)
    sib_y_mean = sum(s["y_mean"] for s in siblings) / len(siblings)
    delta_mean = elem_mean - sib_y_mean

    tol = ROOM_LEVEL_TOLERANCE_M
    above_range = elem_min > sib_y_max + tol
    below_range = elem_max < sib_y_min - tol
    if above_range:
        verdict = "above_room_ceiling_cohort"
    elif below_range:
        verdict = "below_room_ceiling_cohort"
    elif abs(delta_mean) > tol:
        verdict = "borderline_level"
    else:
        verdict = "consistent_level"

    return {
        "available": True,
        "verdict": verdict,
        "tolerance_m": tol,
        "element_y_range": [round(elem_min, 4), round(elem_max, 4)],
        "element_y_mean": round(elem_mean, 4),
        "room_index": room_index,
        "story": story,
        "sibling_count": len(siblings),
        "sibling_y_range": [round(sib_y_min, 4), round(sib_y_max, 4)],
        "sibling_y_mean": round(sib_y_mean, 4),
        "delta_from_sibling_mean_m": round(delta_mean, 4),
        "siblings": siblings[:5],
    }


# --- stick-out probe --------------------------------------------------------


def compute_stickout(
    element_corners_3d: list[list[float]],
    footprint_xz: list[list[float]],
) -> dict:
    """Measure how much the element extends beyond the building footprint.

    Returns per-vertex overhang + signed-area ratios.
    """
    if not element_corners_3d or not footprint_xz:
        return {"available": False, "reason": "missing element or footprint"}

    elem_xz = _poly_xz(element_corners_3d)
    fp_xz = [(float(p[0]), float(p[1])) for p in footprint_xz]
    elem_poly = _safe_polygon(elem_xz)
    fp_poly = _safe_polygon(fp_xz)
    if elem_poly is None or fp_poly is None:
        return {"available": False, "reason": "invalid polygon(s)"}

    inside = elem_poly.intersection(fp_poly).area
    outside = elem_poly.difference(fp_poly).area
    total = elem_poly.area

    # Per-vertex signed distance to the footprint boundary; positive == outside.
    vertex_overhangs: list[dict] = []
    for x, z in elem_xz:
        from shapely.geometry import Point

        point = Point(x, z)
        d_boundary = point.distance(fp_poly.boundary)
        outside_flag = not fp_poly.covers(point)
        vertex_overhangs.append(
            {
                "x": x,
                "z": z,
                "distance_to_boundary_m": round(float(d_boundary), 4),
                "outside_footprint": bool(outside_flag),
            }
        )

    max_overhang = max(
        (
            v["distance_to_boundary_m"]
            for v in vertex_overhangs
            if v["outside_footprint"]
        ),
        default=0.0,
    )

    verdict = "inside"
    if outside / total > 0.5:
        verdict = "mostly_outside"
    elif outside / total > 0.05:
        verdict = "partial_stickout"
    elif any(v["outside_footprint"] for v in vertex_overhangs):
        verdict = "edge_overhang"

    return {
        "available": True,
        "verdict": verdict,
        "area_total_m2": round(total, 4),
        "area_inside_footprint_m2": round(inside, 4),
        "area_outside_footprint_m2": round(outside, 4),
        "outside_ratio": round(outside / total if total else 0.0, 4),
        "max_overhang_m": round(max_overhang, 4),
        "overhanging_vertex_count": sum(
            1 for v in vertex_overhangs if v["outside_footprint"]
        ),
        "vertex_overhangs": vertex_overhangs,
    }


# --- neighborhood probe -----------------------------------------------------


def _atom_records(roof_results_for_uuid: dict) -> list[dict]:
    """Flatten atom-bearing collections into a common record shape."""
    records: list[dict] = []
    cp = roof_results_for_uuid.get("ceiling_partitions", {}) or {}
    for subkind in ("oblique", "flat"):
        for item in cp.get(subkind, []) or []:
            records.append(
                {
                    "id": item.get("id"),
                    "source": f"ceiling_partitions.{subkind}",
                    "category_hint": f"room_ceiling_{subkind}",
                    "corners": item.get("poly") or [],
                    "room_index": item.get("room_index"),
                    "story": item.get("story"),
                    "roof_hypothesis_id": item.get("roof_hypothesis_id"),
                    "kind": item.get("kind"),
                }
            )
    for item in roof_results_for_uuid.get("knee_walls", []) or []:
        records.append(
            {
                "id": item.get("id"),
                "source": "knee_walls",
                "category_hint": "knee_wall",
                "corners": item.get("corners") or [],
                "room_index": item.get("room_index"),
                "story": None,
                "roof_hypothesis_id": None,
                "kind": "knee_wall",
            }
        )
    rs = roof_results_for_uuid.get("roof_surfaces", {}) or {}
    for subkind in ("oblique", "flat"):
        for idx, item in enumerate(rs.get(subkind, []) or []):
            records.append(
                {
                    "id": item.get("boundary_face_id")
                    or f"roof_surfaces.{subkind}[{idx}]",
                    "source": f"roof_surfaces.{subkind}",
                    "category_hint": "exterior_roof",
                    "corners": item.get("corners") or [],
                    "room_index": item.get("room_index"),
                    "story": item.get("story") or item.get("dominant_story"),
                    "roof_hypothesis_id": item.get("roof_hypothesis_id"),
                    "kind": subkind,
                }
            )
    return records


def find_neighbors(
    target: dict,
    roof_results_for_uuid: dict,
    radius_m: float,
) -> list[dict]:
    """Return atoms within ``radius_m`` (horizontal centroid distance) of ``target``.

    Sorted by centroid distance ascending; the target itself is excluded.
    """
    target_corners = target.get("corners") or []
    if not target_corners:
        return []
    target_centroid = _centroid_3d(target_corners)
    target_id = target.get("id")
    target_xz = (target_centroid[0], target_centroid[2])
    target_poly = _safe_polygon(_poly_xz(target_corners))

    results: list[dict] = []
    for rec in _atom_records(roof_results_for_uuid):
        if rec["id"] == target_id:
            continue
        corners = rec.get("corners") or []
        if len(corners) < 3:
            continue
        c = _centroid_3d(corners)
        d_centroid = _distance_2d(target_xz, (c[0], c[2]))
        if d_centroid > radius_m:
            continue

        overlap_area = 0.0
        poly_min_gap_m: float | None = None
        other_poly = _safe_polygon(_poly_xz(corners))
        if target_poly is not None and other_poly is not None:
            overlap_area = float(target_poly.intersection(other_poly).area)
            try:
                poly_min_gap_m = float(target_poly.distance(other_poly))
            except Exception:
                poly_min_gap_m = None

        vertical_delta_m = c[1] - target_centroid[1]

        results.append(
            {
                "id": rec["id"],
                "source": rec["source"],
                "category_hint": rec["category_hint"],
                "kind": rec["kind"],
                "room_index": rec["room_index"],
                "story": rec["story"],
                "roof_hypothesis_id": rec["roof_hypothesis_id"],
                "centroid_distance_m": round(d_centroid, 4),
                "poly_min_gap_m": (
                    round(poly_min_gap_m, 4) if poly_min_gap_m is not None else None
                ),
                "overlap_area_m2": round(overlap_area, 4),
                "vertical_delta_m": round(vertical_delta_m, 4),
                "same_room": rec["room_index"] == target.get("room_index"),
                "same_hypothesis": (
                    rec["roof_hypothesis_id"] is not None
                    and rec["roof_hypothesis_id"] == target.get("roof_hypothesis_id")
                ),
            }
        )
    results.sort(key=lambda r: r["centroid_distance_m"])
    return results


# --- missing-at-point probe ------------------------------------------------


def find_near_point(
    roof_results_for_uuid: dict,
    point_xyz: tuple[float, float, float],
    radius_m: float,
) -> dict:
    """Report atoms near a 3D point — useful when the user sees a hole in the model.

    The horizontal (x, z) plane is used for distance; ``vertical_delta_m`` reports
    the y-offset of each neighbor's centroid vs. the given point.
    """
    px, py, pz = point_xyz
    target_xz = (px, pz)

    from shapely.geometry import Point

    probe_point = Point(px, pz)

    containing: list[dict] = []
    nearby: list[dict] = []

    for rec in _atom_records(roof_results_for_uuid):
        corners = rec.get("corners") or []
        if len(corners) < 3:
            continue
        poly = _safe_polygon(_poly_xz(corners))
        c = _centroid_3d(corners)
        d_centroid = _distance_2d(target_xz, (c[0], c[2]))
        d_poly = float(poly.distance(probe_point)) if poly is not None else None
        rec_summary = {
            "id": rec["id"],
            "source": rec["source"],
            "category_hint": rec["category_hint"],
            "kind": rec["kind"],
            "room_index": rec["room_index"],
            "story": rec["story"],
            "centroid_distance_m": round(d_centroid, 4),
            "poly_distance_m": round(d_poly, 4) if d_poly is not None else None,
            "vertical_delta_m": round(c[1] - py, 4),
        }
        if poly is not None and poly.covers(probe_point):
            containing.append(rec_summary)
        elif d_centroid <= radius_m:
            nearby.append(rec_summary)

    nearby.sort(key=lambda r: r["centroid_distance_m"])
    containing.sort(key=lambda r: abs(r["vertical_delta_m"]))

    return {
        "point": {"x": px, "y": py, "z": pz},
        "radius_m": radius_m,
        "containing": containing,  # atoms whose horizontal footprint covers this point
        "nearby": nearby,  # atoms within the radius but not covering it
        "likely_hole": not containing and not nearby,
    }


# --- entrypoint ------------------------------------------------------------


def _summarize_human(result: dict) -> str:
    lines: list[str] = []
    mode = result.get("mode")
    lines.append(f"# probe_element ({mode})")
    if mode == "element":
        elem = result.get("element", {})
        lines.append(
            f"kind={elem.get('kind')}  "
            f"source_id={elem.get('source_id', elem.get('id'))}"
        )
        lines.append(f"room={elem.get('room_index')}  story={elem.get('story')}")
        parent = elem.get("parent_cell") or {}
        if parent:
            lines.append(
                f"parent_cell: {parent.get('id')}  "
                f"cell_kind={parent.get('cell_kind')}  "
                f"roof_hyp={parent.get('roof_hypothesis_id')}  "
                f"base_atom={parent.get('base_atom_id')}"
            )
        so = result.get("stickout", {})
        if so.get("available"):
            lines.append(
                f"stick-out: verdict={so['verdict']}  "
                f"outside_ratio={so['outside_ratio']}  "
                f"max_overhang_m={so['max_overhang_m']}  "
                f"overhanging_vertices={so['overhanging_vertex_count']}"
            )
        else:
            lines.append(f"stick-out: unavailable ({so.get('reason')})")
        fl = result.get("flatness", {}) or {}
        if fl:
            lines.append(
                f"flatness: verdict={fl.get('verdict')}  "
                f"measured={fl.get('measured_inclination_deg')}°  "
                f"stored={fl.get('stored_role')} (kind={fl.get('stored_kind')})"
            )
            for cc in fl.get("cross_checks") or []:
                lines.append(f"  [cross-check] {cc}")
            for o in (fl.get("overhead_roof_surfaces") or [])[:3]:
                lines.append(
                    f"  overhead: {o['roof_kind']} roof "
                    f"incl={o['roof_inclination_deg']}° "
                    f"overlap={o['xz_overlap_m2']}m² y={o['roof_centroid_y']}"
                )
        lv = result.get("level", {}) or {}
        if lv.get("available"):
            lines.append(
                f"level: verdict={lv.get('verdict')}  "
                f"elem_y={lv.get('element_y_mean')} "
                f"(range {lv.get('element_y_range')})  "
                f"room={lv.get('room_index')} story={lv.get('story')}"
            )
            if lv.get("sibling_count"):
                lines.append(
                    f"  siblings: n={lv['sibling_count']}  "
                    f"y_mean={lv.get('sibling_y_mean')} "
                    f"range={lv.get('sibling_y_range')}  "
                    f"Δfrom_mean={lv.get('delta_from_sibling_mean_m')}m"
                )
        elif lv:
            lines.append(f"level: unavailable ({lv.get('reason')})")
        n = result.get("neighbors", [])
        lines.append(f"neighbors within {result['radius_m']}m: {len(n)}")
        for r in n[:8]:
            lines.append(
                f"  - {r['id']}  d={r['centroid_distance_m']}m  "
                f"dy={r['vertical_delta_m']}m  "
                f"source={r['source']}  same_room={r['same_room']}  "
                f"same_hyp={r['same_hypothesis']}"
            )
    else:
        pt = result.get("point", {})
        lines.append(
            f"point=({pt.get('x')}, {pt.get('y')}, {pt.get('z')})  "
            f"radius_m={result.get('radius_m')}"
        )
        lines.append(
            f"containing={len(result.get('containing', []))}  "
            f"nearby={len(result.get('nearby', []))}"
        )
        for r in result.get("containing", [])[:8]:
            lines.append(
                f"  [contains] {r['id']}  dy={r['vertical_delta_m']}m  "
                f"source={r['source']}"
            )
        for r in result.get("nearby", [])[:8]:
            lines.append(
                f"  [nearby]   {r['id']}  d={r['centroid_distance_m']}m  "
                f"dy={r['vertical_delta_m']}m  source={r['source']}"
            )
        if result.get("likely_hole"):
            lines.append("LIKELY HOLE: no atoms cover or neighbor this point.")
    return "\n".join(lines)


def _resolve_element(
    token: str,
    roof_results: dict,
    buildings_path: Path | None,
    raw_ceiling_plane_splits: dict | None = None,
    pipeline_dir: Path | None = None,
) -> dict:
    """Locator → canonical {id, corners, room_index, story, roof_hypothesis_id}."""
    parsed = parse_element_id(token)
    if is_tier_kind(parsed.kind):
        if pipeline_dir is None:
            raise SystemExit(
                f"Tier kind '{parsed.kind}' needs --pipeline-dir to load "
                f"tier_payload.json"
            )
        payload_path = pipeline_dir / parsed.building_uuid / "tier_payload.json"
        if not payload_path.exists():
            raise SystemExit(f"Tier payload not found: {payload_path}")
        payload = json.loads(payload_path.read_text())
        resolution = find_element(
            [],
            token,
            tier_payloads={parsed.building_uuid: payload},
        )
        elem = resolution.get("element") or {}
        corners = _normalize_corners(
            elem.get("corners")
            or elem.get("poly")
            or elem.get("element_corners")
            or elem.get("wall_corners")
            or []
        )
        return {
            "id": resolution.get("id"),
            "kind": resolution.get("kind"),
            "atom_kind": elem.get("kind"),
            "category": None,
            "corners": corners,
            "room_index": resolution.get("room_index"),
            "story": resolution.get("story"),
            "roof_hypothesis_id": None,
        }

    if is_ontology_kind(parsed.kind):
        resolution = find_element(
            [],
            token,
            roof_results=roof_results,
            raw_ceiling_plane_splits=raw_ceiling_plane_splits,
        )
        atom = resolution.get("atom") or {}
        corners = _normalize_corners(atom.get("poly") or atom.get("corners") or [])
        # Cell/face composites: the face dict carries geometry but not
        # room/story/hypothesis — those live on the parent cell.
        parent = resolution.get("parent_cell") or {}
        face_meta = atom.get("metadata") or {}
        # For arr-faces, ``kind`` is "top"/"side"/"bottom" (geometric role),
        # while ``source_kind`` carries the semantic flat/oblique label the
        # flatness probe expects. Prefer source_kind when present.
        atom_kind = (
            atom.get("source_kind")
            or atom.get("kind")
            or parent.get("roof_surface_kind")
        )
        return {
            "id": resolution.get("source_id"),
            "kind": resolution.get("kind"),
            "atom_kind": atom_kind,
            "category": resolution.get("category"),
            "corners": corners,
            "room_index": atom.get("room_index") or parent.get("room_index"),
            "story": atom.get("story") or parent.get("story"),
            "roof_hypothesis_id": (
                atom.get("roof_hypothesis_id")
                or face_meta.get("roof_hypothesis_id")
                or parent.get("roof_hypothesis_id")
            ),
            "parent_cell": parent or None,
        }
    # Legacy path — need buildings_3d.json
    if buildings_path is None or not buildings_path.exists():
        raise SystemExit(
            f"Legacy kind '{parsed.kind}' needs --buildings-json; {buildings_path} not "
            f"found"
        )
    buildings = json.loads(buildings_path.read_text())
    resolution = find_element(
        buildings,
        token,
        roof_results=roof_results,
        raw_ceiling_plane_splits=raw_ceiling_plane_splits,
    )
    elem = resolution.get("element") or {}
    corners = _normalize_corners(
        elem.get("corners")
        or elem.get("poly")
        or elem.get("element_corners")
        or elem.get("wall_corners")
        or []
    )
    return {
        "id": resolution.get("id"),
        "kind": resolution.get("kind"),
        "atom_kind": elem.get("kind"),
        "category": None,
        "corners": corners,
        "room_index": resolution.get("room_index"),
        "story": resolution.get("story"),
        "roof_hypothesis_id": None,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--element-id", help="Resolve an element and probe it")
    parser.add_argument("--uuid", help="Building UUID (required for --point)")
    parser.add_argument(
        "--point",
        help="3D point 'x,y,z' (y is up) for missing-at-point probe",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.0,
        help="Horizontal search radius in metres (default 3.0)",
    )
    parser.add_argument(
        "--roof-results",
        type=Path,
        default=Path("reconcile/roof_algorithms_py_results.json"),
    )
    parser.add_argument(
        "--buildings-json",
        type=Path,
        default=Path("reconcile/buildings_3d.json"),
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path("pipeline-outputs"),
        help="Directory containing <uuid>/tier_payload.json for tier-* locators",
    )
    parser.add_argument(
        "--raw-ceiling-plane-splits",
        type=Path,
        default=Path("reports/raw_ceiling_plane_scorer/plane_extent_splits.json"),
    )
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args()

    if not args.element_id and not (args.uuid and args.point):
        parser.error("Pass either --element-id OR (--uuid and --point)")

    roof_results_all = json.loads(args.roof_results.read_text())
    raw_ceiling_plane_splits = None
    if args.raw_ceiling_plane_splits.exists():
        raw_ceiling_plane_splits = json.loads(args.raw_ceiling_plane_splits.read_text())

    if args.element_id:
        parsed = parse_element_id(args.element_id)
        per_uuid = roof_results_all.get(parsed.building_uuid)
        if per_uuid is None:
            raise SystemExit(f"Building {parsed.building_uuid} not in roof results")
        element = _resolve_element(
            args.element_id,
            roof_results_all,
            args.buildings_json,
            raw_ceiling_plane_splits=raw_ceiling_plane_splits,
            pipeline_dir=args.pipeline_dir,
        )
        footprint = (per_uuid.get("ceiling", {}) or {}).get("footprint") or []
        stickout = compute_stickout(element["corners"], footprint)
        neighbors = find_neighbors(element, per_uuid, args.radius)
        flatness = check_flatness(element, per_uuid)
        level = check_level(element, per_uuid)
        result = {
            "mode": "element",
            "element": element,
            "footprint_vertex_count": len(footprint),
            "radius_m": args.radius,
            "stickout": stickout,
            "flatness": flatness,
            "level": level,
            "neighbors": neighbors,
        }
    else:
        per_uuid = roof_results_all.get(args.uuid)
        if per_uuid is None:
            raise SystemExit(f"Building {args.uuid} not in roof results")
        try:
            x, y, z = (float(v) for v in args.point.split(","))
        except ValueError as err:
            raise SystemExit(f"--point must be 'x,y,z' (got: {args.point!r})") from err
        near = find_near_point(per_uuid, (x, y, z), args.radius)
        result = {"mode": "point", **near}

    if args.human:
        print(_summarize_human(result))
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
