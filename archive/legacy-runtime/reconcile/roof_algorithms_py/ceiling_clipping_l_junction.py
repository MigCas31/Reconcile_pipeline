from __future__ import annotations

from .math_utils import (
    angle_diff,
    clip_poly_by_ridge,
    point_near_footprint,
    ridge_line_intersection_2d,
)


def detect_and_expand_l_junctions(
    *,
    ceiling_planes: list,
    plane_clipped: list,
    building_footprint: list,
    point_in_poly_2d,
) -> list[dict]:
    """Detect L-junction pairs and extend their ceiling planes through the junction.

    For perpendicular plane pairs whose ridge lines intersect inside the
    building footprint, extends each plane's ridge bounds through the junction
    area.  The opposing-plane cuts (run later) handle the ridge-line split
    within each wing; in the junction overlap area both wings' ceilings
    coexist and the viewer shows whichever is lower.

    Modifies *plane_clipped* in place.  Returns a list of detected junction
    records for diagnostics.
    """
    l_junctions: list[dict] = []

    for i, pi in enumerate(ceiling_planes):
        if len(plane_clipped[i]["clipped"]) < 3:
            continue
        for j in range(i + 1, len(ceiling_planes)):
            pj = ceiling_planes[j]
            if len(plane_clipped[j]["clipped"]) < 3:
                continue
            if pi["dominantStory"] != pj["dominantStory"]:
                continue

            azi_diff = angle_diff(pi["cl"]["avgAzimuth"], pj["cl"]["avgAzimuth"])
            if azi_diff < 60.0 or azi_diff > 120.0:
                continue

            # --- Phase A: verify junction is inside building footprint ---
            junction = ridge_line_intersection_2d(pi, pj)
            if not junction:
                continue
            if not point_near_footprint(
                junction["x"],
                junction["z"],
                building_footprint,
                1.0,
                point_in_poly_2d,
            ):
                continue

            l_junctions.append({"plane_a": i, "plane_b": j, "junction_pt": junction})

            # --- Phase B: extend ridge bounds through junction ---
            for idx, plane in ((i, pi), (j, pj)):
                proj_r = (junction["x"] - plane["ref"]["x"]) * plane["ridgeX"] + (
                    junction["z"] - plane["ref"]["z"]
                ) * plane["ridgeZ"]

                cur_min = plane_clipped[idx]["ridgeMin"]
                cur_max = plane_clipped[idx]["ridgeMax"]
                new_min = min(cur_min, proj_r - 1.0)
                new_max = max(cur_max, proj_r + 1.0)

                if new_min < cur_min or new_max > cur_max:
                    # Re-clip the building footprint with expanded ridge bounds.
                    new_poly = list(building_footprint)
                    new_poly = clip_poly_by_ridge(
                        new_poly,
                        plane["ridgeX"],
                        plane["ridgeZ"],
                        plane["ref"]["x"],
                        plane["ref"]["z"],
                        new_min,
                        True,
                    )
                    new_poly = clip_poly_by_ridge(
                        new_poly,
                        plane["ridgeX"],
                        plane["ridgeZ"],
                        plane["ref"]["x"],
                        plane["ref"]["z"],
                        new_max,
                        False,
                    )
                    if len(new_poly) >= 3:
                        plane_clipped[idx]["clipped"] = new_poly
                        plane_clipped[idx]["ridgeMin"] = new_min
                        plane_clipped[idx]["ridgeMax"] = new_max

    return l_junctions
