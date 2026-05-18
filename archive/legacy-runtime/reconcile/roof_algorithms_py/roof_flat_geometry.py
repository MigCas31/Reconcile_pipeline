from __future__ import annotations


def bbox(points: list) -> tuple:
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_z = min(p[2] for p in points)
    max_z = max(p[2] for p in points)
    return min_x, max_x, min_z, max_z


def bbox_surface(points: list, y: float, pad: float = 0.3) -> list:
    min_x, max_x, min_z, max_z = bbox(points)
    return [
        (min_x - pad, y, min_z - pad),
        (max_x + pad, y, min_z - pad),
        (max_x + pad, y, max_z + pad),
        (min_x - pad, y, max_z + pad),
    ]


def cluster_top_flat_segments(
    top_flat_segments: list, y_tol: float = 0.15, min_cluster_size: int = 2
) -> list:
    clusters = []
    for seg in top_flat_segments:
        assigned = False
        for cl in clusters:
            if abs(seg["y"] - cl["avgY"]) < y_tol:
                cl["segs"].append(seg)
                cl["avgY"] = sum(x["y"] for x in cl["segs"]) / len(cl["segs"])
                assigned = True
                break
        if not assigned:
            clusters.append({"segs": [seg], "avgY": seg["y"]})
    return [cl for cl in clusters if len(cl["segs"]) >= min_cluster_size]
