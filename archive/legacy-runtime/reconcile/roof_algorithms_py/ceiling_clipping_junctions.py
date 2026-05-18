from __future__ import annotations

from .math_utils import (
    angle_diff,
    clip_poly_by_half_plane_2d,
    clip_poly_by_ridge,
    point_near_footprint,
)


def _ridge_line_intersection_2d(pi: dict, pj: dict):
    dx = pj["ref"]["x"] - pi["ref"]["x"]
    dz = pj["ref"]["z"] - pi["ref"]["z"]
    det = pi["ridgeX"] * pj["ridgeZ"] - pi["ridgeZ"] * pj["ridgeX"]
    if abs(det) < 1e-9:
        return None
    t = (dx * pj["ridgeZ"] - dz * pj["ridgeX"]) / det
    return {
        "x": pi["ref"]["x"] + t * pi["ridgeX"],
        "z": pi["ref"]["z"] + t * pi["ridgeZ"],
    }


def build_junction_patches(
    *,
    ceiling_planes: list,
    plane_clipped: list,
    building_footprint: list,
    point_in_poly_2d,
) -> list:
    junction_patches = []

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

            junction = _ridge_line_intersection_2d(pi, pj)
            if not junction:
                continue
            if not point_near_footprint(
                junction["x"], junction["z"], building_footprint, 1.0, point_in_poly_2d
            ):
                continue

            to_j_x, to_j_z = (
                pj["ref"]["x"] - pi["ref"]["x"],
                pj["ref"]["z"] - pi["ref"]["z"],
            )
            dot_i = pi["slopeX"] * to_j_x + pi["slopeZ"] * to_j_z
            to_i_x, to_i_z = (
                pi["ref"]["x"] - pj["ref"]["x"],
                pi["ref"]["z"] - pj["ref"]["z"],
            )
            dot_j = pj["slopeX"] * to_i_x + pj["slopeZ"] * to_i_z
            if dot_i <= 0.0 or dot_j <= 0.0:
                continue

            ai, bi = pi["n"]["x"] / pi["n"]["y"], pi["n"]["z"] / pi["n"]["y"]
            ci = pi["ref"]["y"] + ai * pi["ref"]["x"] + bi * pi["ref"]["z"]
            aj, bj = pj["n"]["x"] / pj["n"]["y"], pj["n"]["z"] / pj["n"]["y"]
            cj = pj["ref"]["y"] + aj * pj["ref"]["x"] + bj * pj["ref"]["z"]
            vdx, vdz, voffset = aj - ai, bj - bi, ci - cj
            if abs(vdx) < 1e-9 and abs(vdz) < 1e-9:
                continue

            for idx, plane, sign in ((i, pi, 1.0), (j, pj, -1.0)):
                proj_r = (junction["x"] - plane["ref"]["x"]) * plane["ridgeX"] + (
                    junction["z"] - plane["ref"]["z"]
                ) * plane["ridgeZ"]
                cur_min, cur_max = (
                    plane_clipped[idx]["ridgeMin"],
                    plane_clipped[idx]["ridgeMax"],
                )

                if proj_r > cur_max:
                    ext_min, ext_max = cur_max, proj_r + 2.0
                elif proj_r < cur_min:
                    ext_min, ext_max = proj_r - 2.0, cur_min
                else:
                    continue

                ext = list(building_footprint)
                ext = clip_poly_by_ridge(
                    ext,
                    plane["ridgeX"],
                    plane["ridgeZ"],
                    plane["ref"]["x"],
                    plane["ref"]["z"],
                    ext_min,
                    True,
                )
                ext = clip_poly_by_ridge(
                    ext,
                    plane["ridgeX"],
                    plane["ridgeZ"],
                    plane["ref"]["x"],
                    plane["ref"]["z"],
                    ext_max,
                    False,
                )
                if len(ext) < 3:
                    continue
                ext = clip_poly_by_half_plane_2d(
                    ext, sign * vdx, sign * vdz, sign * voffset
                )
                if len(ext) < 3:
                    continue
                junction_patches.append(
                    {
                        "poly2d": ext,
                        "planeIdx": idx,
                        "plane": plane,
                    }
                )

    return junction_patches
