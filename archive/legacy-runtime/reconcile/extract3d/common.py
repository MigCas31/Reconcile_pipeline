"""Core geometry/transformation helpers for extract_3d."""

import numpy as np


def parse_transform(flat):
    return np.array(flat).reshape(4, 4, order="F")


def corners_to_world(polygon_corners, transform_flat):
    transform = parse_transform(transform_flat)
    return [
        (transform @ np.array([*corner, 1.0]))[:3].tolist()
        for corner in polygon_corners
    ]


def wall_world_corners(wall):
    """Get wall corners in world coordinates.

    Uses polygonCorners when present, otherwise constructs a rectangle from
    dimensions + transform.
    """
    transform = parse_transform(wall["transform"])
    polygon_corners = wall.get("polygonCorners") or []
    if len(polygon_corners) >= 3:
        return [
            (transform @ np.array([*corner, 1.0]))[:3].tolist()
            for corner in polygon_corners
        ]

    width = wall["dimensions"][0] / 2
    height = wall["dimensions"][1] / 2
    local = [
        [-width, -height, 0],
        [width, -height, 0],
        [width, height, 0],
        [-width, height, 0],
    ]
    return [(transform @ np.array([*corner, 1.0]))[:3].tolist() for corner in local]


def clamp_opening_to_parent(opening_corners, parent_corners):
    """Clamp opening corners to the parent wall extent in wall-local 2D."""
    if len(parent_corners) < 3 or len(opening_corners) < 3:
        return opening_corners

    parent = np.array(parent_corners)
    opening = np.array(opening_corners)

    edge1 = parent[1] - parent[0]
    edge1_len = np.linalg.norm(edge1)
    if edge1_len < 1e-9:
        return opening_corners
    edge1_norm = edge1 / edge1_len

    edge2_raw = parent[2] - parent[1]
    normal = np.cross(edge1, edge2_raw)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-9:
        return opening_corners
    normal = normal / normal_len

    edge2_norm = np.cross(normal, edge1_norm)
    origin = parent.mean(axis=0)

    def to_2d(points):
        rel = points - origin
        return np.column_stack([rel @ edge1_norm, rel @ edge2_norm])

    parent_2d = to_2d(parent)
    opening_2d = to_2d(opening)
    parent_min = parent_2d.min(axis=0)
    parent_max = parent_2d.max(axis=0)

    clamped_2d = np.clip(opening_2d, parent_min, parent_max)
    if np.allclose(clamped_2d, opening_2d, atol=1e-6):
        return opening_corners

    result = []
    for i in range(len(opening_corners)):
        rel = opening[i] - origin
        depth = rel @ normal
        point = (
            origin
            + clamped_2d[i, 0] * edge1_norm
            + clamped_2d[i, 1] * edge2_norm
            + depth * normal
        )
        result.append(point.tolist())
    return result


def hybrid_wall_corners(merged_wall, raw_wall, floor_y=None):
    """Use merged transform with the best available shape source (merged/raw)."""
    transform = parse_transform(merged_wall["transform"])
    merged_pc = merged_wall.get("polygonCorners", [])
    raw_pc = raw_wall.get("polygonCorners", [])

    if len(merged_pc) >= 3:
        local = merged_pc
    elif len(raw_pc) >= 3:
        local = raw_pc
    else:
        width = raw_wall["dimensions"][0] / 2
        height = raw_wall["dimensions"][1] / 2
        local = [
            [-width, -height, 0],
            [width, -height, 0],
            [width, height, 0],
            [-width, height, 0],
        ]

    corners = [(transform @ np.array([*corner, 1.0]))[:3].tolist() for corner in local]
    if floor_y is None:
        return corners

    current_bottom = min(corner[1] for corner in corners)
    dy = floor_y - current_bottom
    return [[corner[0], corner[1] + dy, corner[2]] for corner in corners]


def compute_svd(src, dst):
    """Compute rigid transform (R, t) from src to dst."""
    src_center = src.mean(0)
    dst_center = dst.mean(0)
    h_mat = (src - src_center).T @ (dst - dst_center)
    u_mat, _, v_t = np.linalg.svd(h_mat)
    rot = v_t.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        v_t[-1] *= -1
        rot = v_t.T @ u_mat.T
    trans = dst_center - rot @ src_center
    residual_cm = np.max(np.linalg.norm(dst - (rot @ src.T).T - trans, axis=1)) * 100
    return rot, trans, residual_cm
