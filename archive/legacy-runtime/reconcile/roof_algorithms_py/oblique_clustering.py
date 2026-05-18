from __future__ import annotations

from math import atan2, cos, sin, sqrt

from .math_utils import angle_diff, plane_normal, seg_midpoint

COPLANAR_TOL = 0.5
MIN_CLUSTER_SIZE = 2


def cluster_oblique_segments(segments: list[dict]) -> list[dict]:
    clusters: list[dict] = []

    for seg in segments:
        assigned = False
        mid = seg_midpoint(seg)
        for cl in clusters:
            if (
                angle_diff(seg["azimuth"], cl["avgAzimuth"]) < 30.0
                and abs(seg["incl"] - cl["avgIncl"]) < 15.0
            ):
                n = plane_normal(cl["avgAzimuth"], cl["avgIncl"])
                n_len = sqrt(n["x"] ** 2 + n["y"] ** 2 + n["z"] ** 2)
                if n_len > 1e-6:
                    dot = (
                        (mid["x"] - cl["refPt"]["x"]) * n["x"]
                        + (mid["y"] - cl["refPt"]["y"]) * n["y"]
                        + (mid["z"] - cl["refPt"]["z"]) * n["z"]
                    )
                    if abs(dot / n_len) > COPLANAR_TOL:
                        continue

                cl["segs"].append(seg)
                cnt = len(cl["segs"])
                cl["avgIncl"] = sum(e["incl"] for e in cl["segs"]) / cnt
                sin_sum = sum(
                    sin(e["azimuth"] * 3.141592653589793 / 180.0) for e in cl["segs"]
                )
                cos_sum = sum(
                    cos(e["azimuth"] * 3.141592653589793 / 180.0) for e in cl["segs"]
                )
                cl["avgAzimuth"] = (
                    atan2(sin_sum, cos_sum) * 180.0 / 3.141592653589793
                ) % 360.0
                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "segs": [seg],
                    "avgIncl": seg["incl"],
                    "avgAzimuth": seg["azimuth"],
                    "refPt": mid,
                }
            )

    return [cl for cl in clusters if len(cl["segs"]) >= MIN_CLUSTER_SIZE]
