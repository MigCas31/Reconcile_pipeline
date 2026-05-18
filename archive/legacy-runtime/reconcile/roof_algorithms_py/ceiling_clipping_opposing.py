from __future__ import annotations

from .math_utils import angle_diff, clip_poly_by_half_plane_2d


def _overlap_lower_envelope(poly_i, poly_j, dx, dz, offset):
    """Clip two 2D polygons to keep only the lower surface in their overlap.

    For the region where poly_i and poly_j overlap, keeps the portion where
    each respective plane is lower.  Non-overlapping portions are preserved
    unchanged.

    Returns (new_poly_i, new_poly_j) as coordinate lists.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from shapely.validation import make_valid
    except ImportError:
        return list(poly_i), list(poly_j)

    def to_shapely(poly):
        if len(poly) < 3:
            return Polygon()
        p = Polygon(poly)
        if not p.is_valid:
            p = make_valid(p)
        return p if p.geom_type == "Polygon" else Polygon()

    def from_shapely(geom):
        if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
            return []
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        coords = list(geom.exterior.coords)
        if coords and coords[-1] == coords[0]:
            coords = coords[:-1]
        return [(c[0], c[1]) for c in coords]

    shape_i = to_shapely(poly_i)
    shape_j = to_shapely(poly_j)
    if shape_i.is_empty or shape_j.is_empty:
        return list(poly_i), list(poly_j)

    overlap = shape_i.intersection(shape_j)
    if overlap.is_empty or overlap.area < 1e-6:
        return list(poly_i), list(poly_j)

    overlap_coords = from_shapely(overlap)
    if len(overlap_coords) < 3:
        return list(poly_i), list(poly_j)

    overlap_i_lower = clip_poly_by_half_plane_2d(overlap_coords, dx, dz, offset)
    overlap_j_lower = clip_poly_by_half_plane_2d(overlap_coords, -dx, -dz, -offset)

    # result_i = (original_i - overlap) U (overlap where i is lower)
    non_overlap_i = shape_i.difference(overlap)
    if overlap_i_lower and len(overlap_i_lower) >= 3:
        kept_i = unary_union([non_overlap_i, to_shapely(overlap_i_lower)])
    else:
        kept_i = non_overlap_i

    non_overlap_j = shape_j.difference(overlap)
    if overlap_j_lower and len(overlap_j_lower) >= 3:
        kept_j = unary_union([non_overlap_j, to_shapely(overlap_j_lower)])
    else:
        kept_j = non_overlap_j

    result_i = from_shapely(kept_i)
    result_j = from_shapely(kept_j)

    return (
        result_i if len(result_i) >= 3 else list(poly_i),
        result_j if len(result_j) >= 3 else list(poly_j),
    )


def apply_lower_envelope_cuts(*, ceiling_planes: list, plane_clipped: list) -> None:
    """Clip overlapping ceiling-plane pairs so each keeps only where it is lower.

    Two strategies based on azimuth difference:

    - **Opposing planes (140-180 deg)**: full half-plane cut along the equal-height
      line.  This correctly splits the roof along the ridge where two slopes
      face each other.
    - **All other overlapping pairs**: overlap-only lower-envelope cut.  Within
      the 2D overlap region, keep whichever surface is lower; preserve
      non-overlapping portions unchanged.  This handles L-junctions (60-140 deg),
      near-parallel planes at different inclinations, and any other overlap.
    """
    for i, pi in enumerate(ceiling_planes):
        if len(plane_clipped[i]["clipped"]) < 3:
            continue
        ai, bi = pi["n"]["x"] / pi["n"]["y"], pi["n"]["z"] / pi["n"]["y"]
        ci = pi["ref"]["y"] + ai * pi["ref"]["x"] + bi * pi["ref"]["z"]

        for j in range(i + 1, len(ceiling_planes)):
            pj = ceiling_planes[j]
            if len(plane_clipped[j]["clipped"]) < 3:
                continue
            if pi["dominantStory"] != pj["dominantStory"]:
                continue
            ad = angle_diff(pi["cl"]["avgAzimuth"], pj["cl"]["avgAzimuth"])

            aj, bj = pj["n"]["x"] / pj["n"]["y"], pj["n"]["z"] / pj["n"]["y"]
            cj = pj["ref"]["y"] + aj * pj["ref"]["x"] + bj * pj["ref"]["z"]
            dx, dz, offset = aj - ai, bj - bi, ci - cj
            if abs(dx) < 1e-9 and abs(dz) < 1e-9:
                continue

            if ad >= 140.0:
                # Opposing planes: full half-plane cut along the ridge
                plane_clipped[i]["clipped"] = clip_poly_by_half_plane_2d(
                    plane_clipped[i]["clipped"], dx, dz, offset
                )
                plane_clipped[j]["clipped"] = clip_poly_by_half_plane_2d(
                    plane_clipped[j]["clipped"], -dx, -dz, -offset
                )
            else:
                # L-junction planes (60-140 deg): overlap-only lower envelope
                new_i, new_j = _overlap_lower_envelope(
                    plane_clipped[i]["clipped"],
                    plane_clipped[j]["clipped"],
                    dx,
                    dz,
                    offset,
                )
                plane_clipped[i]["clipped"] = new_i
                plane_clipped[j]["clipped"] = new_j
