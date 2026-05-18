"""Phase B.3 -- geometric realism scorer.

Compares the BIP solver's selected envelope against the V3 full-model
reference (``slanted_roofs`` U ``flat_ceilings``) per building, all in
the XZ plane.

Signal we want:

* ``iou_xz``        Intersection-over-union of the two polygon unions.
                    1.0 = identical footprints, 0.0 = disjoint.
* ``coverage_ref``  Fraction of the reference envelope captured by the
                    selection. High coverage means the BIP didn't miss
                    big pieces of the V3 envelope.
* ``over_coverage`` Selection area outside the reference, normalised by
                    reference area. High = BIP reaches beyond V3 (e.g.
                    aggressive ridge extension).
* ``azimuth_mismatch_frac``
                    Of the selected faces, the fraction whose azimuth
                    disagrees with the nearest V3 reference face by
                    >30 deg. Catches cases where BIP selects a plane on
                    the wrong slope side.

These are the inner-loop signals for B.5 hyperparameter search. The B.4
screenshot harness is the outer check on the same trials.

Implementation notes
--------------------
* ``V3SlantedRoof.corners`` is a list of (x, y, z) -- 3D points on the
  slanted plane. Project to XZ by dropping y.
* ``V3FlatCeiling.footprint_xz`` is also named ``_xz`` but actually
  carries 3-tuples (x, y, z); project the same way.
* Shapely ``unary_union`` tolerates overlapping polygons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

Point2D = tuple[float, float]

_AZIMUTH_MISMATCH_DEG = 30.0
_IOU_HIGH_QUALITY = 0.9


@dataclass
class RealismScore:
    building_uuid: str
    iou_xz: float | None  # None if reference is missing
    coverage_ref: float
    over_coverage: float
    azimuth_mismatch_frac: float
    ref_area_m2: float
    sel_area_m2: float
    n_selected: int
    n_reference: int
    note: str = ""  # e.g. "no reference envelope" or "selection empty"


def _poly_from_ring(ring: list) -> Polygon | None:
    pts = []
    for pt in ring or []:
        if not isinstance(pt, (list, tuple)):
            continue
        if len(pt) >= 3:
            pts.append((float(pt[0]), float(pt[2])))
        elif len(pt) == 2:
            pts.append((float(pt[0]), float(pt[1])))
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid or poly.area <= 0:
        return None
    return poly


def _azimuth_diff_deg(a: float, b: float) -> float:
    raw = abs(a - b) % 360.0
    return min(raw, 360.0 - raw)


def _polys_from_reference(full_model: dict) -> list[Polygon]:
    """Project V3 ``slanted_roofs`` corners and ``flat_ceilings`` footprints
    into XZ polygons. Malformed entries are skipped silently.
    """
    polys: list[Polygon] = []
    for roof in full_model.get("slanted_roofs") or []:
        poly = _poly_from_ring(roof.get("corners"))
        if poly is not None:
            polys.append(poly)
    for ceil in full_model.get("flat_ceilings") or []:
        poly = _poly_from_ring(ceil.get("footprint_xz"))
        if poly is not None:
            polys.append(poly)
    return polys


def _ref_azimuths(full_model: dict) -> list[tuple[Point2D, float]]:
    """Nearest-neighbour lookup structure for azimuth comparison.
    Returns ``(centroid_xz, azimuth_deg)`` pairs for V3 slanted roofs.
    Flat ceilings have no azimuth -- they're excluded from the azimuth
    mismatch check.
    """
    out: list[tuple[Point2D, float]] = []
    for roof in full_model.get("slanted_roofs") or []:
        poly = _poly_from_ring(roof.get("corners"))
        if poly is None:
            continue
        az = roof.get("azimuth_deg")
        if az is None:
            plane = roof.get("plane") or []
            if len(plane) >= 3:
                a, _b, c, *_rest = plane
                az = math.degrees(math.atan2(-a, -c)) % 360.0
        if az is None:
            continue
        cx, cz = poly.centroid.x, poly.centroid.y
        out.append(((float(cx), float(cz)), float(az)))
    return out


def score_building(
    building_uuid: str,
    selected_faces: list[dict],
    full_model: dict | None,
) -> RealismScore:
    """Score one building's BIP selection vs. its V3 full-model reference.

    ``selected_faces`` is the Phase A candidate-face dict shape -- what
    ``/reconstruction`` returns in ``selected_faces``.
    ``full_model`` is the payload from ``/ontology-artifacts?view=full-model``
    (or equivalent in-process dict).
    """
    sel_polys = [
        p
        for p in (_poly_from_ring(f.get("footprint_xz")) for f in selected_faces)
        if p is not None
    ]
    sel_union = unary_union(sel_polys) if sel_polys else None
    sel_area = float(sel_union.area) if sel_union else 0.0

    if not full_model:
        return RealismScore(
            building_uuid=building_uuid,
            iou_xz=None,
            coverage_ref=0.0,
            over_coverage=0.0,
            azimuth_mismatch_frac=0.0,
            ref_area_m2=0.0,
            sel_area_m2=sel_area,
            n_selected=len(sel_polys),
            n_reference=0,
            note="no full_model payload",
        )

    ref_polys = _polys_from_reference(full_model)
    ref_union = unary_union(ref_polys) if ref_polys else None
    ref_area = float(ref_union.area) if ref_union else 0.0

    if ref_union is None or ref_area <= 0:
        return RealismScore(
            building_uuid=building_uuid,
            iou_xz=None,
            coverage_ref=0.0,
            over_coverage=0.0,
            azimuth_mismatch_frac=0.0,
            ref_area_m2=0.0,
            sel_area_m2=sel_area,
            n_selected=len(sel_polys),
            n_reference=0,
            note="no reference envelope",
        )

    if sel_union is None or sel_area <= 0:
        return RealismScore(
            building_uuid=building_uuid,
            iou_xz=0.0,
            coverage_ref=0.0,
            over_coverage=0.0,
            azimuth_mismatch_frac=0.0,
            ref_area_m2=ref_area,
            sel_area_m2=0.0,
            n_selected=0,
            n_reference=len(ref_polys),
            note="selection empty",
        )

    inter_area = float(sel_union.intersection(ref_union).area)
    union_area = float(sel_union.union(ref_union).area)
    iou = inter_area / union_area if union_area > 0 else 0.0
    coverage_ref = inter_area / ref_area if ref_area > 0 else 0.0
    over_area = max(sel_area - inter_area, 0.0)
    over_coverage = over_area / ref_area if ref_area > 0 else 0.0

    # Azimuth mismatch -- only meaningful when the selection carries
    # azimuth and we have slanted-roof references to compare against.
    ref_az = _ref_azimuths(full_model)
    n_az_checks = 0
    n_az_mismatch = 0
    for face in selected_faces:
        face_az = face.get("azimuth_deg")
        poly = _poly_from_ring(face.get("footprint_xz"))
        if face_az is None or poly is None or not ref_az:
            continue
        cx, cz = float(poly.centroid.x), float(poly.centroid.y)
        # Nearest reference centroid in XZ.
        best_d2 = float("inf")
        best_az = None
        for (rx, rz), raz in ref_az:
            d2 = (cx - rx) ** 2 + (cz - rz) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_az = raz
        if best_az is None:
            continue
        n_az_checks += 1
        if _azimuth_diff_deg(float(face_az), best_az) > _AZIMUTH_MISMATCH_DEG:
            n_az_mismatch += 1
    az_mismatch_frac = (n_az_mismatch / n_az_checks) if n_az_checks else 0.0

    return RealismScore(
        building_uuid=building_uuid,
        iou_xz=iou,
        coverage_ref=coverage_ref,
        over_coverage=over_coverage,
        azimuth_mismatch_frac=az_mismatch_frac,
        ref_area_m2=ref_area,
        sel_area_m2=sel_area,
        n_selected=len(sel_polys),
        n_reference=len(ref_polys),
    )


def aggregate_scores(scores: list[RealismScore]) -> dict[str, Any]:
    """Aggregate across a list of buildings. Excludes ``iou_xz=None``
    (buildings without a reference envelope) from mean/median IoU."""
    ious = [s.iou_xz for s in scores if s.iou_xz is not None]
    cov = [s.coverage_ref for s in scores]
    over = [s.over_coverage for s in scores]
    az_mm = [s.azimuth_mismatch_frac for s in scores]

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _median(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    return {
        "n_buildings": len(scores),
        "n_with_reference": len(ious),
        "mean_iou": _mean(ious),
        "median_iou": _median(ious),
        "frac_iou_ge_0_9": (
            sum(1 for i in ious if i >= _IOU_HIGH_QUALITY) / len(ious) if ious else 0.0
        ),
        "mean_coverage_ref": _mean(cov),
        "mean_over_coverage": _mean(over),
        "mean_az_mismatch_frac": _mean(az_mm),
    }
