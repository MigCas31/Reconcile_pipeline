#!/usr/bin/env python3
"""Prototype scorer for raw-ceiling support of oblique roof-plane targets.

Read-only diagnostic. Scores:

* candidate oblique ceiling planes from ``roof_algorithms_py_results.json``
* committed oblique roof surfaces from the same results

against raw RoomPlan ceiling planes in ``buildings_3d.json``.

The goal is to measure whether raw ceilings provide enough evidence to:

* strengthen target orientation confidence
* help decide which ridge/eave planes to keep
* flag likely local slope splits the current pipeline missed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parent.parent
BUILDINGS_PATH = REPO / "reconcile" / "buildings_3d.json"
ROOF_RESULTS_PATH = REPO / "reconcile" / "roof_algorithms_py_results.json"
V3_RESULTS_PATH = REPO / "reconcile" / "reconcile_v3_results.json"
OUT_DIR = REPO / "reports" / "raw_ceiling_plane_scorer"
RIDGE_EAVE_SCORES_PATH = REPO / "reports" / "ridge_eave_scores_20260420" / "scores.json"

TRUSTED_DIST_M = 0.30
WALL_TOP_TOL_M = 0.10

HORIZONTAL_DY_M = 0.05
MIN_EDGE_LEN_M = 0.2
RIDGE_MIN_ANGLE_DEG = 10.0
RIDGE_MATCH_XZ_TOL_M = 0.30
RIDGE_MATCH_Y_TOL_M = 0.20
EAVE_FP_TOL_M = 0.50

MIN_RAW_AREA_M2 = 0.5
MAX_RAW_INCLINATION_DEG = 80.0
MIN_ROOM_TRUST_SCORE = 0.75
LOW_TRUST_PROMOTION_MIN_OVERLAP_M2 = 0.5
LOW_TRUST_PROMOTION_MIN_RAW_OVERLAP_FRACTION = 0.6
LOW_TRUST_PROMOTION_MIN_NORMAL_DOT = 0.95
LOW_TRUST_PROMOTION_MAX_HEIGHT_RESIDUAL_M = 0.5
MIN_MATCH_OVERLAP_M2 = 0.1
MIN_CONFLICT_OVERLAP_M2 = 0.3
MIN_CONFLICT_ANGLE_DEG = 20.0
EDGE_ALIGNMENT_TOL_DEG = 15.0
EAVE_TARGET_BOUNDARY_TOL_M = 0.5
MIN_SPLIT_PIECE_AREA_M2 = 0.1
EAVE_CHAIN_ANGLE_TOL_DEG = 8.0
EAVE_CHAIN_Y_TOL_M = 0.35
EAVE_CHAIN_COLINEAR_OFFSET_M = 0.4
EAVE_CHAIN_GAP_M = 1.0
PLANE_EAVE_CHAIN_BOUNDARY_TOL_M = 0.75
PLANE_EAVE_CHAIN_BUFFER_M = 0.5
PLANE_EAVE_CHAIN_HEIGHT_TOL_M = 0.4
PLANE_EAVE_CHAIN_SUPPORT_THRESHOLD = 0.6
MIN_GAP_CONTINUATION_OVERLAP_M2 = 0.1
MIN_GAP_CONTINUATION_OVERLAP_FRACTION = 0.25
STORY_EXTENT_SMALL_HOLE_MAX_AREA_M2 = 1.0
CREATOR_RAIN_AREA_SUSPECT_MAX = 0.25
CREATOR_EXTENDED_AREA_SUSPECT_MIN = 0.75
CREATOR_NEG_TOP_SUSPECT_MIN = 0.25
RIDGE_EAVE_SEGMENT_ANCHOR_BUFFER_M = 1.0
RIDGE_EAVE_FINAL_MAX_LOCAL_COMPETITOR_LOSS = 0.5
RIDGE_EAVE_SUSPECT_ANCHOR_MIN_MIRROR_SCORE = 0.75
RIDGE_EAVE_SINGLE_COMPONENT_SCORE_GAP = 0.1
RIDGE_EAVE_ROOM_OWNERSHIP_BUFFER_M = 0.35
RIDGE_EAVE_OWNERSHIP_MAX_AZIMUTH_DELTA_DEG = 30.0
RIDGE_EAVE_OWNERSHIP_MAX_INCLINATION_DELTA_DEG = 10.0
RIDGE_EAVE_MIRROR_SLIVER_MAX_AREA_FRACTION = 0.25
RIDGE_EAVE_MIRROR_SLIVER_MIN_THROUGH_RATIO = 1.0
RIDGE_EAVE_COMMITTED_COVER_REDUNDANT_MIN_FRACTION = 0.85
RIDGE_EAVE_MIRROR_PRUNE_FINAL_MIN_MIRROR_SCORE = 0.9
RIDGE_EAVE_MIRROR_PRUNE_FINAL_MAX_THROUGH_RATIO = 0.8
RIDGE_EAVE_MIRROR_PRUNE_FINAL_MAX_PARTNER_OVERLAP_M2 = 0.005
RIDGE_EAVE_MIRROR_PRUNE_REDUNDANT_MIN_MIRROR_SCORE = 0.99
RIDGE_EAVE_MIRROR_PRUNE_REDUNDANT_MAX_PARTNER_OVERLAP_M2 = 0.05


@dataclass(frozen=True)
class RawPlaneRecord:
    uuid: str
    story: int
    room_index: int
    plane_index: int
    element_id: str
    corners: list[list[float]]
    poly_xz: Polygon
    centroid: tuple[float, float, float]
    normal: np.ndarray
    azimuth_deg: float
    inclination_deg: float
    area_xz_m2: float
    room_trust_score: float
    usable_for_support: bool


@dataclass(frozen=True)
class TargetPlaneRecord:
    uuid: str
    story: int
    target_kind: str
    target_index: int
    element_id: str
    poly_xz: Polygon
    normal: np.ndarray
    azimuth_deg: float
    inclination_deg: float
    ridge_dir_xz: tuple[float, float]
    area_xz_m2: float
    plane_point: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class RawEdgeRecord:
    story: int
    plane_element_id: str
    label: str
    length_m: float
    midpoint_xz: tuple[float, float]
    edge_azimuth_xz_deg: float
    start_xz: tuple[float, float] | None = None
    end_xz: tuple[float, float] | None = None
    y_mid: float | None = None


@dataclass(frozen=True)
class EaveChainRecord:
    uuid: str
    story: int
    chain_id: str
    edge_count: int
    total_length_m: float
    azimuth_deg: float
    y_mean: float
    start_xz: tuple[float, float]
    end_xz: tuple[float, float]
    line_xz: LineString
    member_plane_ids: tuple[str, ...]


@dataclass(frozen=True)
class PlaneEaveChainSupportRecord:
    uuid: str
    story: int
    target_element_id: str
    target_kind: str
    chain_id: str
    chain_azimuth_deg: float
    ridge_azimuth_deg: float
    angle_delta_deg: float
    boundary_distance_m: float
    overlap_fraction: float
    height_residual_m: float | None
    support_score: float
    supported: bool
    chain_length_m: float


@dataclass(frozen=True)
class TargetSplitPieceRecord:
    uuid: str
    story: int
    target_element_id: str
    target_kind: str
    piece_id: str
    piece_index: int
    piece_role: str
    area_xz_m2: float
    support_score: float | None
    chain_ids: tuple[str, ...]
    corners: list[list[float]]
    holes: list[list[list[float]]]


@dataclass(frozen=True)
class FaceRunRecord:
    uuid: str
    face_run_id: str
    story: int
    azimuth_deg: float | None
    inclination_deg: float | None
    member_piece_ids: tuple[str, ...]
    committed_piece_ids: tuple[str, ...]
    ridge_piece_ids: tuple[str, ...]
    core_committed_piece_ids: tuple[str, ...]
    core_committed_target_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    hypothesis_part_ids: tuple[str, ...]
    core_union: Polygon | None


@dataclass(frozen=True)
class FaceRunSeedRecord:
    uuid: str
    face_run_id: str
    story: int
    azimuth_deg: float | None
    inclination_deg: float | None
    member_target_ids: tuple[str, ...]
    committed_target_ids: tuple[str, ...]
    ridge_target_ids: tuple[str, ...]
    core_committed_target_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    hypothesis_part_ids: tuple[str, ...]
    core_target_union: Polygon | None


@dataclass(frozen=True)
class ConflictPairRecord:
    story: int
    plane_a_id: str
    plane_b_id: str
    overlap_area_m2: float
    angle_deg: float
    poly_xz: Polygon


def _xz_polygon(corners: list[list[float]]) -> Polygon | None:
    pts = [(float(c[0]), float(c[2])) for c in corners if len(c) >= 3]
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _fit_plane_svd(
    corners: list[list[float]],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    pts = np.asarray(corners, dtype=float)
    if pts.shape[0] < 3 or pts.shape[1] < 3:
        return None, None
    centroid = pts.mean(axis=0)
    diffs = pts - centroid
    try:
        _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    normal = vt[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None, None
    normal = normal / norm
    if normal[1] < 0:
        normal = -normal
    return centroid, normal


def _normalize_roof_up(normal: np.ndarray) -> np.ndarray:
    vec = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec
    vec = vec / norm
    if vec[1] < 0:
        vec = -vec
    return vec


def _normal_to_azimuth_inclination(normal: np.ndarray) -> tuple[float, float]:
    roof_up = _normalize_roof_up(normal)
    azimuth = float(
        (math.degrees(math.atan2(float(roof_up[0]), float(roof_up[2]))) + 360.0) % 360.0
    )
    inclination = float(
        math.degrees(math.acos(max(-1.0, min(1.0, abs(float(roof_up[1]))))))
    )
    return azimuth, inclination


def _wrapped_angle_delta_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs(float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


def _wrapped_angle_delta_mod180_deg(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def _ridge_dir_to_azimuth_mod180(ridge_dir_xz: tuple[float, float]) -> float:
    dx, dz = ridge_dir_xz
    return float(math.degrees(math.atan2(dx, dz)) % 180.0)


def _candidate_polygon(plane: dict[str, Any]) -> Polygon | None:
    rx, rz = float(plane["ridgeX"]), float(plane["ridgeZ"])
    sx, sz = float(plane["slopeX"]), float(plane["slopeZ"])
    ref = plane["ref"]
    ref_x, ref_z = float(ref["x"]), float(ref["z"])
    bounds = [
        (plane["minRidge"], plane["minSlope"]),
        (plane["maxRidge"], plane["minSlope"]),
        (plane["maxRidge"], plane["maxSlope"]),
        (plane["minRidge"], plane["maxSlope"]),
    ]
    corners_xz = []
    for ridge_coord, slope_coord in bounds:
        x = ref_x + float(ridge_coord) * rx + float(slope_coord) * sx
        z = ref_z + float(ridge_coord) * rz + float(slope_coord) * sz
        corners_xz.append((x, z))
    poly = Polygon(corners_xz)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _xz_pair(point: list[float]) -> tuple[float, float] | None:
    if len(point) >= 3:
        return (float(point[0]), float(point[2]))
    if len(point) >= 2:
        return (float(point[0]), float(point[1]))
    return None


def _poly_from_xz_ring(ring: list[list[float]]) -> Polygon | None:
    pts: list[tuple[float, float]] = []
    for point in ring:
        pair = _xz_pair(point)
        if pair is not None:
            pts.append(pair)
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _candidate_normal(plane: dict[str, Any]) -> np.ndarray:
    n = plane["n"]
    return _normalize_roof_up(
        np.array([float(n["x"]), float(n["y"]), float(n["z"])], dtype=float)
    )


def _wall_top_segments(
    room: dict[str, Any],
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for wall in room.get("walls_computed") or []:
        corners = wall.get("corners") or []
        if len(corners) < 3:
            continue
        ys = [float(c[1]) for c in corners]
        top_y = max(ys)
        tops = [
            tuple(float(v) for v in corner)
            for corner in corners
            if abs(float(corner[1]) - top_y) <= WALL_TOP_TOL_M
        ]
        if len(tops) < 2:
            continue
        tops.sort(key=lambda p: (p[0], p[2]))
        for idx in range(len(tops) - 1):
            segments.append((tops[idx], tops[idx + 1]))
        segments.append((tops[-1], tops[0]))
    return segments


def _dist_point_to_segment_3d(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    ap = np.array(
        [point[0] - start[0], point[1] - start[1], point[2] - start[2]], dtype=float
    )
    ab = np.array(
        [end[0] - start[0], end[1] - start[1], end[2] - start[2]], dtype=float
    )
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(ap))
    t = max(0.0, min(1.0, float(np.dot(ap, ab)) / denom))
    closest = np.array(start, dtype=float) + t * ab
    return float(np.linalg.norm(np.array(point, dtype=float) - closest))


def _min_dist_to_wall_tops(
    point: tuple[float, float, float],
    wall_segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> float:
    if not wall_segments:
        return float("inf")
    return min(
        _dist_point_to_segment_3d(point, start, end) for start, end in wall_segments
    )


def compute_room_trust_scores(building: dict[str, Any]) -> dict[tuple[int, int], float]:
    trust_by_room: dict[tuple[int, int], float] = {}
    for room_index, room in enumerate(building.get("rooms") or []):
        planes = room.get("raw_ceiling_planes") or []
        if not planes:
            continue
        story = int(room.get("story", 0))
        wall_segments = _wall_top_segments(room)
        if not wall_segments:
            continue
        distances: list[float] = []
        for plane in planes:
            corners = plane.get("corners") or []
            for corner in corners:
                if len(corner) < 3:
                    continue
                point = (float(corner[0]), float(corner[1]), float(corner[2]))
                distances.append(_min_dist_to_wall_tops(point, wall_segments))
        if not distances:
            continue
        trusted = sum(1 for dist in distances if dist <= TRUSTED_DIST_M)
        trust_by_room[(story, room_index)] = trusted / len(distances)
    return trust_by_room


def _exposed_room_keys(roof_result: dict[str, Any]) -> set[tuple[int, int]] | None:
    exposed = (roof_result.get("ceiling") or {}).get("exposed_rooms")
    if not exposed:
        return None
    keys: set[tuple[int, int]] = set()
    for entry in exposed:
        room_index = entry.get("room_index")
        story = entry.get("story")
        if room_index is None or story is None:
            continue
        keys.add((int(story), int(room_index)))
    return keys


def collect_raw_plane_records(
    building: dict[str, Any],
    roof_result: dict[str, Any],
    *,
    exposed_only: bool = True,
) -> list[RawPlaneRecord]:
    trust_scores = compute_room_trust_scores(building)
    allowed_keys = _exposed_room_keys(roof_result) if exposed_only else None
    uuid = str(building["uuid"])
    records: list[RawPlaneRecord] = []
    for room_index, room in enumerate(building.get("rooms") or []):
        story = int(room.get("story", 0))
        room_key = (story, room_index)
        if allowed_keys is not None and room_key not in allowed_keys:
            continue
        room_trust_score = float(trust_scores.get(room_key, 0.0))
        for plane_index, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            poly = _xz_polygon(corners)
            if poly is None:
                continue
            centroid, normal = _fit_plane_svd(corners)
            if centroid is None or normal is None:
                continue
            azimuth_deg, inclination_deg = _normal_to_azimuth_inclination(normal)
            area_xz_m2 = float(poly.area)
            usable = (
                area_xz_m2 >= MIN_RAW_AREA_M2
                and inclination_deg < MAX_RAW_INCLINATION_DEG
                and room_trust_score >= MIN_ROOM_TRUST_SCORE
            )
            records.append(
                RawPlaneRecord(
                    uuid=uuid,
                    story=story,
                    room_index=room_index,
                    plane_index=plane_index,
                    element_id=f"{uuid}::ceiling-raw::{story}:{room_index}:{plane_index}",
                    corners=corners,
                    poly_xz=poly,
                    centroid=(
                        float(centroid[0]),
                        float(centroid[1]),
                        float(centroid[2]),
                    ),
                    normal=normal,
                    azimuth_deg=azimuth_deg,
                    inclination_deg=inclination_deg,
                    area_xz_m2=area_xz_m2,
                    room_trust_score=room_trust_score,
                    usable_for_support=usable,
                )
            )
    return records


def _base_raw_plane_usable(record: RawPlaneRecord) -> bool:
    return (
        record.area_xz_m2 >= MIN_RAW_AREA_M2
        and record.inclination_deg < MAX_RAW_INCLINATION_DEG
    )


def _raw_plane_is_local_target_match(
    record: RawPlaneRecord,
    target: TargetPlaneRecord,
) -> bool:
    if record.story != target.story:
        return False
    if not _base_raw_plane_usable(record):
        return False
    try:
        overlap = record.poly_xz.intersection(target.poly_xz)
    except Exception:
        return False
    overlap_area = float(overlap.area) if not overlap.is_empty else 0.0
    if overlap_area < LOW_TRUST_PROMOTION_MIN_OVERLAP_M2:
        return False
    raw_overlap_fraction = overlap_area / max(record.area_xz_m2, 1e-9)
    if raw_overlap_fraction < LOW_TRUST_PROMOTION_MIN_RAW_OVERLAP_FRACTION:
        return False
    normal_dot = abs(
        float(
            np.dot(_normalize_roof_up(record.normal), _normalize_roof_up(target.normal))
        )
    )
    if normal_dot < LOW_TRUST_PROMOTION_MIN_NORMAL_DOT:
        return False
    y_on_target = _plane_y_at(
        target, float(record.centroid[0]), float(record.centroid[2])
    )
    if y_on_target is None:
        return False
    height_residual = abs(float(record.centroid[1]) - float(y_on_target))
    return height_residual <= LOW_TRUST_PROMOTION_MAX_HEIGHT_RESIDUAL_M


def promote_raw_plane_support_records(
    raw_records: list[RawPlaneRecord],
    targets: list[TargetPlaneRecord],
) -> list[RawPlaneRecord]:
    promoted: list[RawPlaneRecord] = []
    targets_by_story: dict[int, list[TargetPlaneRecord]] = defaultdict(list)
    for target in targets:
        targets_by_story[target.story].append(target)

    for record in raw_records:
        usable = record.usable_for_support
        if not usable and _base_raw_plane_usable(record):
            for target in targets_by_story.get(record.story, []):
                if _raw_plane_is_local_target_match(record, target):
                    usable = True
                    break
        promoted.append(
            record
            if usable == record.usable_for_support
            else replace(record, usable_for_support=usable)
        )
    return promoted


def _footprint_lines(
    roof_result: dict[str, Any],
) -> list[tuple[float, float, float, float]]:
    fp = (roof_result.get("ceiling") or {}).get("footprint") or []
    if len(fp) < 3:
        return []
    lines: list[tuple[float, float, float, float]] = []
    count = len(fp)
    for idx in range(count):
        curr = fp[idx]
        nxt = fp[(idx + 1) % count]
        ax = float(curr[0])
        az = float(curr[2] if len(curr) >= 3 else curr[1])
        bx = float(nxt[0])
        bz = float(nxt[2] if len(nxt) >= 3 else nxt[1])
        lines.append((ax, az, bx, bz))
    return lines


def _distance_point_to_segment_2d(
    px: float, pz: float, ax: float, az: float, bx: float, bz: float
) -> float:
    vx, vz = bx - ax, bz - az
    wx, wz = px - ax, pz - az
    denom = vx * vx + vz * vz
    if denom < 1e-12:
        return math.hypot(wx, wz)
    t = max(0.0, min(1.0, (vx * wx + vz * wz) / denom))
    cx, cz = ax + t * vx, az + t * vz
    return math.hypot(px - cx, pz - cz)


def _distance_point_to_infinite_line_2d(
    px: float,
    pz: float,
    ax: float,
    az: float,
    bx: float,
    bz: float,
) -> float:
    vx, vz = bx - ax, bz - az
    denom = math.hypot(vx, vz)
    if denom < 1e-12:
        return math.hypot(px - ax, pz - az)
    return abs(vz * px - vx * pz + bx * az - bz * ax) / denom


def _edge_near_footprint(
    start: list[float],
    end: list[float],
    footprint_lines: list[tuple[float, float, float, float]],
) -> bool:
    if not footprint_lines:
        return False
    sx, sz = float(start[0]), float(start[2])
    ex, ez = float(end[0]), float(end[2])
    mx, mz = (sx + ex) * 0.5, (sz + ez) * 0.5
    for ax, az, bx, bz in footprint_lines:
        for px, pz in ((sx, sz), (ex, ez), (mx, mz)):
            if _distance_point_to_segment_2d(px, pz, ax, az, bx, bz) <= EAVE_FP_TOL_M:
                return True
    return False


def _endpoints_match(
    a1: list[float], b1: list[float], a2: list[float], b2: list[float]
) -> bool:
    def close(p: list[float], q: list[float]) -> bool:
        return (
            math.hypot(float(p[0]) - float(q[0]), float(p[2]) - float(q[2]))
            <= RIDGE_MATCH_XZ_TOL_M
            and abs(float(p[1]) - float(q[1])) <= RIDGE_MATCH_Y_TOL_M
        )

    return (close(a1, a2) and close(b1, b2)) or (close(a1, b2) and close(b1, a2))


def _normals_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    dot = float(np.dot(_normalize_roof_up(n1), _normalize_roof_up(n2)))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(abs(dot))))


def _edge_azimuth_xz_deg(start: list[float], end: list[float]) -> float | None:
    dx = float(end[0] - start[0])
    dz = float(end[2] - start[2])
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return None
    return float(math.degrees(math.atan2(dx, dz)) % 180.0)


def collect_raw_edges(
    raw_records: list[RawPlaneRecord],
    roof_result: dict[str, Any],
) -> list[RawEdgeRecord]:
    usable_records = [record for record in raw_records if record.usable_for_support]
    records_by_story: dict[
        int, list[tuple[RawPlaneRecord, list[float], list[float], float, float | None]]
    ] = defaultdict(list)
    for record in usable_records:
        corners = record.corners
        count = len(corners)
        for idx in range(count):
            start = corners[idx]
            end = corners[(idx + 1) % count]
            dy = abs(float(end[1]) - float(start[1]))
            length = math.dist(start, end)
            if dy > HORIZONTAL_DY_M or length < MIN_EDGE_LEN_M:
                continue
            records_by_story[record.story].append(
                (
                    record,
                    start,
                    end,
                    length,
                    _edge_azimuth_xz_deg(start, end),
                )
            )

    footprint_lines = _footprint_lines(roof_result)
    edges: list[RawEdgeRecord] = []
    for story, story_edges in records_by_story.items():
        for idx, (record, start, end, length, edge_azimuth) in enumerate(story_edges):
            if edge_azimuth is None:
                continue
            label = "isolated"
            best_angle = 0.0
            for other_idx, (
                other_record,
                other_start,
                other_end,
                _other_len,
                _other_azimuth,
            ) in enumerate(story_edges):
                if idx == other_idx or other_record.element_id == record.element_id:
                    continue
                if not _endpoints_match(start, end, other_start, other_end):
                    continue
                angle = _normals_angle_deg(record.normal, other_record.normal)
                if angle >= RIDGE_MIN_ANGLE_DEG and angle > best_angle:
                    label = "ridge_or_hip"
                    best_angle = angle
            if label != "ridge_or_hip" and _edge_near_footprint(
                start, end, footprint_lines
            ):
                label = "eave"
            midpoint_xz = (
                (float(start[0]) + float(end[0])) * 0.5,
                (float(start[2]) + float(end[2])) * 0.5,
            )
            edges.append(
                RawEdgeRecord(
                    story=story,
                    plane_element_id=record.element_id,
                    label=label,
                    length_m=float(length),
                    midpoint_xz=midpoint_xz,
                    edge_azimuth_xz_deg=float(edge_azimuth),
                    start_xz=(float(start[0]), float(start[2])),
                    end_xz=(float(end[0]), float(end[2])),
                    y_mid=(float(start[1]) + float(end[1])) * 0.5,
                )
            )
    return edges


def collect_conflict_pairs(
    raw_records: list[RawPlaneRecord],
) -> list[ConflictPairRecord]:
    usable_records = [record for record in raw_records if record.usable_for_support]
    conflicts: list[ConflictPairRecord] = []
    for idx, left in enumerate(usable_records):
        for right in usable_records[idx + 1 :]:
            if left.story != right.story:
                continue
            try:
                overlap = left.poly_xz.intersection(right.poly_xz)
            except Exception:
                continue
            if overlap.is_empty or float(overlap.area) < MIN_CONFLICT_OVERLAP_M2:
                continue
            if overlap.geom_type == "MultiPolygon":
                overlap = max(overlap.geoms, key=lambda geom: geom.area)
            if overlap.geom_type != "Polygon" or overlap.is_empty:
                continue
            angle_deg = _normals_angle_deg(left.normal, right.normal)
            if angle_deg < MIN_CONFLICT_ANGLE_DEG:
                continue
            conflicts.append(
                ConflictPairRecord(
                    story=left.story,
                    plane_a_id=left.element_id,
                    plane_b_id=right.element_id,
                    overlap_area_m2=float(overlap.area),
                    angle_deg=angle_deg,
                    poly_xz=overlap,
                )
            )
    return conflicts


def _edge_chain_connects(left: RawEdgeRecord, right: RawEdgeRecord) -> bool:
    if left.label != "eave" or right.label != "eave":
        return False
    if left.story != right.story:
        return False
    if (
        left.start_xz is None
        or left.end_xz is None
        or right.start_xz is None
        or right.end_xz is None
    ):
        return False
    if left.y_mid is None or right.y_mid is None:
        return False
    if (
        _wrapped_angle_delta_mod180_deg(
            left.edge_azimuth_xz_deg, right.edge_azimuth_xz_deg
        )
        > EAVE_CHAIN_ANGLE_TOL_DEG
    ):
        return False
    if abs(float(left.y_mid) - float(right.y_mid)) > EAVE_CHAIN_Y_TOL_M:
        return False

    left_mid = left.midpoint_xz
    right_mid = right.midpoint_xz
    left_to_right = _distance_point_to_infinite_line_2d(
        left_mid[0],
        left_mid[1],
        right.start_xz[0],
        right.start_xz[1],
        right.end_xz[0],
        right.end_xz[1],
    )
    right_to_left = _distance_point_to_infinite_line_2d(
        right_mid[0],
        right_mid[1],
        left.start_xz[0],
        left.start_xz[1],
        left.end_xz[0],
        left.end_xz[1],
    )
    if max(left_to_right, right_to_left) > EAVE_CHAIN_COLINEAR_OFFSET_M:
        return False

    endpoint_gap = min(
        math.dist(left.start_xz, right.start_xz),
        math.dist(left.start_xz, right.end_xz),
        math.dist(left.end_xz, right.start_xz),
        math.dist(left.end_xz, right.end_xz),
    )
    line_gap = LineString([left.start_xz, left.end_xz]).distance(
        LineString([right.start_xz, right.end_xz])
    )
    return endpoint_gap <= EAVE_CHAIN_GAP_M or line_gap <= EAVE_CHAIN_COLINEAR_OFFSET_M


def build_eave_chains(
    uuid: str, raw_edges: list[RawEdgeRecord]
) -> list[EaveChainRecord]:
    eave_edges = [edge for edge in raw_edges if edge.label == "eave"]
    by_story: dict[int, list[RawEdgeRecord]] = defaultdict(list)
    for edge in eave_edges:
        by_story[edge.story].append(edge)

    chains: list[EaveChainRecord] = []
    for story, story_edges in sorted(by_story.items()):
        adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(story_edges))}
        for left_idx, left in enumerate(story_edges):
            for right_idx in range(left_idx + 1, len(story_edges)):
                right = story_edges[right_idx]
                if _edge_chain_connects(left, right):
                    adjacency[left_idx].add(right_idx)
                    adjacency[right_idx].add(left_idx)

        visited: set[int] = set()
        chain_index = 0
        for start_idx in range(len(story_edges)):
            if start_idx in visited:
                continue
            stack = [start_idx]
            component: list[RawEdgeRecord] = []
            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue
                visited.add(idx)
                component.append(story_edges[idx])
                stack.extend(adjacency[idx] - visited)

            points = np.array(
                [
                    point
                    for edge in component
                    for point in (edge.start_xz, edge.end_xz)
                    if point is not None
                ],
                dtype=float,
            )
            if len(points) < 2:
                continue
            center = points.mean(axis=0)
            if len(points) >= 3:
                try:
                    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
                    axis = vt[0]
                except np.linalg.LinAlgError:
                    axis = np.array([1.0, 0.0], dtype=float)
            else:
                axis = points[1] - points[0]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-9:
                axis = np.array([1.0, 0.0], dtype=float)
                axis_norm = 1.0
            axis = axis / axis_norm
            longest = max(component, key=lambda edge: edge.length_m)
            if longest.start_xz is not None and longest.end_xz is not None:
                longest_vec = np.array(
                    [
                        longest.end_xz[0] - longest.start_xz[0],
                        longest.end_xz[1] - longest.start_xz[1],
                    ],
                    dtype=float,
                )
                if float(np.dot(axis, longest_vec)) < 0:
                    axis = -axis
            projections = (points - center) @ axis
            start = center + axis * float(np.min(projections))
            end = center + axis * float(np.max(projections))
            total_length = float(sum(edge.length_m for edge in component))
            y_mean = float(
                np.average(
                    [float(edge.y_mid or 0.0) for edge in component],
                    weights=[edge.length_m for edge in component],
                )
            )
            chains.append(
                EaveChainRecord(
                    uuid=uuid,
                    story=story,
                    chain_id=f"{uuid}::eave-chain::{story}:{chain_index}",
                    edge_count=len(component),
                    total_length_m=total_length,
                    azimuth_deg=float(
                        math.degrees(math.atan2(float(axis[0]), float(axis[1]))) % 180.0
                    ),
                    y_mean=y_mean,
                    start_xz=(float(start[0]), float(start[1])),
                    end_xz=(float(end[0]), float(end[1])),
                    line_xz=LineString(
                        [
                            (float(start[0]), float(start[1])),
                            (float(end[0]), float(end[1])),
                        ]
                    ),
                    member_plane_ids=tuple(
                        sorted({edge.plane_element_id for edge in component})
                    ),
                )
            )
            chain_index += 1
    return chains


def build_story_extent_envelopes(building: dict[str, Any]) -> dict[int, Polygon]:
    by_story: dict[int, list[Polygon]] = defaultdict(list)
    for room in building.get("rooms") or []:
        story = int(room.get("story", 0))
        poly = _xz_polygon(room.get("floor_polygon") or [])
        if poly is not None:
            by_story[story].append(poly)
    for gap in building.get("cross_floor_gaps") or []:
        corners = gap.get("corners") or []
        story = int(gap.get("story", 0))
        poly = _xz_polygon(corners)
        if poly is not None:
            by_story[story].append(poly)
    # Ceiling gap quads can belong to the inter-story seam: include them in
    # both stories so upper-story roof targets are not clipped short above the
    # lower story's ceiling gap strip.
    for gap_wall in building.get("gap_walls") or []:
        if str(gap_wall.get("type") or "") != "gap_ceiling":
            continue
        poly = _xz_polygon(gap_wall.get("corners") or [])
        if poly is None:
            continue
        story = int(gap_wall.get("story", 0))
        by_story[story].append(poly)
        by_story[story + 1].append(poly)

    envelopes: dict[int, Polygon] = {}
    for story, polys in by_story.items():
        if not polys:
            continue
        try:
            merged = unary_union(polys)
        except Exception:
            continue
        candidates = _iter_polygons(merged)
        if not candidates:
            continue
        envelope = (
            max(candidates, key=lambda poly: float(poly.area))
            if len(candidates) == 1
            else unary_union(candidates)
        )
        envelope = _fill_small_polygon_holes(
            envelope,
            max(float(STORY_EXTENT_SMALL_HOLE_MAX_AREA_M2), 0.0),
        )
        if getattr(envelope, "is_empty", True):
            continue
        envelopes[story] = envelope
    return envelopes


def _fill_small_polygon_holes(geom: Any, max_hole_area_m2: float) -> Any:
    if geom is None or getattr(geom, "is_empty", True) or max_hole_area_m2 <= 0.0:
        return geom

    cleaned_polys: list[Polygon] = []
    for poly in _iter_polygons(geom):
        hole_rings: list[list[tuple[float, float]]] = []
        for ring in poly.interiors:
            ring_coords = list(ring.coords)
            if len(ring_coords) < 4:
                continue
            hole_poly = Polygon(ring_coords)
            if hole_poly.is_empty:
                continue
            if float(hole_poly.area) > max_hole_area_m2:
                hole_rings.append(ring_coords)
        try:
            cleaned = Polygon(list(poly.exterior.coords), holes=hole_rings or None)
        except Exception:
            cleaned = poly
        if not cleaned.is_valid:
            try:
                cleaned = cleaned.buffer(0)
            except Exception:
                cleaned = poly
        if cleaned.is_empty:
            cleaned = poly
        if isinstance(cleaned, Polygon):
            cleaned_polys.append(cleaned)

    if not cleaned_polys:
        return geom
    try:
        return unary_union(cleaned_polys)
    except Exception:
        return cleaned_polys[0] if len(cleaned_polys) == 1 else geom


def build_story_gap_polygons(building: dict[str, Any]) -> dict[int, list[Polygon]]:
    by_story: dict[int, list[Polygon]] = defaultdict(list)
    for gap in building.get("cross_floor_gaps") or []:
        poly = _xz_polygon(gap.get("corners") or [])
        if poly is None:
            continue
        by_story[int(gap.get("story", 0))].append(poly)
    # Gap ceilings are generated from cross-floor gaps but carry explicit
    # ceiling quads. Attach them to both their own story and story+1 so
    # upper-story roof targets can continue above inter-story seams.
    for gap_wall in building.get("gap_walls") or []:
        if str(gap_wall.get("type") or "") != "gap_ceiling":
            continue
        poly = _xz_polygon(gap_wall.get("corners") or [])
        if poly is None:
            continue
        story = int(gap_wall.get("story", 0))
        by_story[story].append(poly)
        by_story[story + 1].append(poly)
    return dict(by_story)


def _plane_coeff_point_and_normal(
    plane: list[float], poly: Polygon
) -> tuple[tuple[float, float, float], np.ndarray] | tuple[None, None]:
    if len(plane) != 4:
        return None, None
    a, b, c, d = (float(value) for value in plane)
    if abs(b) < 1e-9:
        return None, None
    centroid = poly.centroid
    x = float(centroid.x)
    z = float(centroid.y)
    y = float(-(a * x + c * z + d) / b)
    normal = _normalize_roof_up(np.asarray([a, b, c], dtype=float))
    return (x, y, z), normal


def _ridge_dir_from_plane_coeffs(plane: list[float]) -> tuple[float, float]:
    if len(plane) != 4:
        return (1.0, 0.0)
    a, _b, c, _d = (float(value) for value in plane)
    ridge = np.asarray([-c, a], dtype=float)
    norm = float(np.linalg.norm(ridge))
    if norm < 1e-9:
        return (1.0, 0.0)
    ridge = ridge / norm
    return (float(ridge[0]), float(ridge[1]))


def _infer_story_from_envelopes(
    poly: Polygon, story_extent_envelopes: dict[int, Polygon]
) -> int:
    best_story = 0
    best_overlap = -1.0
    for story, envelope in story_extent_envelopes.items():
        try:
            overlap = float(poly.intersection(envelope).area)
        except Exception:
            overlap = 0.0
        if overlap > best_overlap + 1e-9 or (
            abs(overlap - best_overlap) <= 1e-9 and int(story) > best_story
        ):
            best_story = int(story)
            best_overlap = overlap
    return best_story


def collect_selected_ridge_eave_plane_group_targets(
    uuid: str,
    ridge_eave_entry: dict[str, Any] | None,
    story_extent_envelopes: dict[int, Polygon],
) -> list[TargetPlaneRecord]:
    if not ridge_eave_entry:
        return []
    targets: list[TargetPlaneRecord] = []
    for idx, plane_group in enumerate(ridge_eave_entry.get("plane_groups") or []):
        if plane_group.get("selected") is False:
            continue
        poly = _poly_from_xz_ring(plane_group.get("union_xz") or [])
        plane = plane_group.get("plane") or []
        if poly is None or not isinstance(plane, list) or len(plane) != 4:
            continue
        plane_point, normal = _plane_coeff_point_and_normal(plane, poly)
        if plane_point is None or normal is None:
            continue
        story = _infer_story_from_envelopes(poly, story_extent_envelopes)
        azimuth_deg, inclination_deg = _normal_to_azimuth_inclination(normal)
        plane_group_suffix = str(plane_group.get("id", "")).split("::")[-1]
        element_id = f"{uuid}::ridge-eave-candidate::plane-group::{plane_group_suffix}"
        targets.append(
            TargetPlaneRecord(
                uuid=uuid,
                story=story,
                target_kind="ridge_eave_plane_group",
                target_index=idx,
                element_id=element_id,
                poly_xz=poly,
                normal=normal,
                azimuth_deg=azimuth_deg,
                inclination_deg=inclination_deg,
                ridge_dir_xz=_ridge_dir_from_plane_coeffs(plane),
                area_xz_m2=float(poly.area),
                plane_point=plane_point,
            )
        )
    return targets


def collect_ridge_eave_target_diagnostics(
    uuid: str,
    ridge_eave_entry: dict[str, Any] | None,
    v3_building: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not ridge_eave_entry or not v3_building:
        return {}

    candidate_by_id = {
        str(candidate.get("id")): candidate
        for candidate in (ridge_eave_entry.get("candidates") or [])
        if candidate.get("id")
    }
    segment_by_id = {
        str(segment.get("id")): segment
        for segment in (v3_building.get("merged_roof_segments") or [])
        if segment.get("id")
    }

    diagnostics: dict[str, dict[str, Any]] = {}
    for plane_group in ridge_eave_entry.get("plane_groups") or []:
        if plane_group.get("selected") is False:
            continue
        plane_group_suffix = str(plane_group.get("id", "")).split("::")[-1]
        if not plane_group_suffix:
            continue
        target_element_id = (
            f"{uuid}::ridge-eave-candidate::plane-group::{plane_group_suffix}"
        )
        member_candidates = [
            candidate_by_id[str(member_id)]
            for member_id in (plane_group.get("member_ids") or [])
            if str(member_id) in candidate_by_id
        ]
        if not member_candidates:
            continue

        total_candidate_area = float(
            sum(
                float(candidate.get("area_m2") or 0.0)
                for candidate in member_candidates
            )
        )
        extended_candidate_area = float(
            sum(
                float(candidate.get("area_m2") or 0.0)
                for candidate in member_candidates
                if candidate.get("extended")
            )
        )
        rain_candidate_area = 0.0
        rain_segment_count = 0
        covered_segment_count = 0
        source_rooms: set[str] = set()
        touched_rooms: set[str] = set()
        top_story_snapshot_count = 0
        negative_top_snapshot_count = 0
        seen_segment_ids: set[str] = set()

        for candidate in member_candidates:
            parent_segment_id = str(candidate.get("parent_segment_id") or "")
            if not parent_segment_id:
                continue
            segment = segment_by_id.get(parent_segment_id)
            if segment is None:
                continue
            features = segment.get("features") or {}
            rain_capable = (features.get("rain_hitting_side_count", 0) or 0) > 0
            covered_capable = (features.get("covered_side_count", 0) or 0) > 0
            if rain_capable:
                rain_candidate_area += float(candidate.get("area_m2") or 0.0)
            if parent_segment_id not in seen_segment_ids:
                seen_segment_ids.add(parent_segment_id)
                if rain_capable:
                    rain_segment_count += 1
                if covered_capable:
                    covered_segment_count += 1
                for snapshot in segment.get("member_snapshots") or []:
                    source_room_id = snapshot.get("source_room_id")
                    slab_room_id = snapshot.get("slab_room_id")
                    if source_room_id:
                        source_rooms.add(str(source_room_id))
                    if slab_room_id:
                        touched_rooms.add(str(slab_room_id))
                    snap_features = snapshot.get("features") or {}
                    if not (
                        snap_features.get("is_top_story_slab")
                        and snap_features.get("slab_kind") == "room"
                    ):
                        continue
                    plane_height = snap_features.get("plane_height_above_slab_m")
                    if plane_height is None:
                        continue
                    top_story_snapshot_count += 1
                    if float(plane_height) < -0.2:
                        negative_top_snapshot_count += 1

        total_segment_count = len(seen_segment_ids)
        creator_rain_segment_fraction = (
            float(rain_segment_count) / float(total_segment_count)
            if total_segment_count
            else 0.0
        )
        creator_rain_area_fraction = (
            float(rain_candidate_area) / float(total_candidate_area)
            if total_candidate_area > 1e-9
            else 0.0
        )
        creator_extended_area_fraction = (
            float(extended_candidate_area) / float(total_candidate_area)
            if total_candidate_area > 1e-9
            else 0.0
        )
        creator_negative_top_fraction = (
            float(negative_top_snapshot_count) / float(top_story_snapshot_count)
            if top_story_snapshot_count
            else 0.0
        )
        has_partner = plane_group.get("best_partner_plane_group_id") is not None
        suspect_reasons: list[str] = []
        if creator_rain_area_fraction < CREATOR_RAIN_AREA_SUSPECT_MAX:
            suspect_reasons.append("weak_creator_rain_area")
        if covered_segment_count > rain_segment_count:
            suspect_reasons.append("covered_creators_dominate")
        if creator_extended_area_fraction >= CREATOR_EXTENDED_AREA_SUSPECT_MIN:
            suspect_reasons.append("mostly_extended")
        if not has_partner:
            suspect_reasons.append("unpaired")
        if creator_negative_top_fraction >= CREATOR_NEG_TOP_SUSPECT_MIN:
            suspect_reasons.append("cuts_below_top_story")
        provenance_flag = (
            "suspect_interior_slice"
            if (
                creator_rain_area_fraction < CREATOR_RAIN_AREA_SUSPECT_MAX
                and covered_segment_count > rain_segment_count
                and creator_extended_area_fraction >= CREATOR_EXTENDED_AREA_SUSPECT_MIN
                and (
                    not has_partner
                    or creator_negative_top_fraction >= CREATOR_NEG_TOP_SUSPECT_MIN
                )
            )
            else "normal"
        )
        diagnostics[target_element_id] = {
            "creator_segment_count": total_segment_count,
            "creator_rain_segment_count": rain_segment_count,
            "creator_covered_segment_count": covered_segment_count,
            "creator_rain_segment_fraction": round(creator_rain_segment_fraction, 6),
            "creator_rain_area_fraction": round(creator_rain_area_fraction, 6),
            "creator_extended_area_fraction": round(creator_extended_area_fraction, 6),
            "creator_source_room_ids": sorted(source_rooms),
            "creator_touch_room_ids": sorted(touched_rooms),
            "creator_source_room_count": len(source_rooms),
            "creator_touch_room_count": len(touched_rooms),
            "creator_negative_top_fraction": round(creator_negative_top_fraction, 6),
            "creator_has_partner": bool(has_partner),
            "provenance_relevance_flag": provenance_flag,
            "provenance_relevance_reasons": suspect_reasons,
        }
    return diagnostics


def collect_ridge_eave_target_anchor_masks(
    uuid: str,
    ridge_eave_entry: dict[str, Any] | None,
    building: dict[str, Any] | None,
    v3_building: dict[str, Any] | None,
    target_diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build per-target XZ anchor masks from source oblique segments.

    For ridge/eave plane-group targets dominated by extended candidates, we
    constrain supported strips to where contributing oblique segments actually
    exist, with a small buffer applied at split time.
    """
    if not ridge_eave_entry or not v3_building:
        return {}

    candidate_by_id = {
        str(candidate.get("id")): candidate
        for candidate in (ridge_eave_entry.get("candidates") or [])
        if candidate.get("id")
    }
    segment_by_id = {
        str(segment.get("id")): segment
        for segment in (v3_building.get("merged_roof_segments") or [])
        if segment.get("id")
    }

    masks_by_target: dict[str, Any] = {}
    for plane_group in ridge_eave_entry.get("plane_groups") or []:
        if plane_group.get("selected") is False:
            continue
        plane_group_suffix = str(plane_group.get("id", "")).split("::")[-1]
        if not plane_group_suffix:
            continue
        target_element_id = (
            f"{uuid}::ridge-eave-candidate::plane-group::{plane_group_suffix}"
        )

        diagnostics = target_diagnostics.get(target_element_id) or {}
        if diagnostics.get("provenance_relevance_flag") != "suspect_interior_slice":
            continue
        creator_extended_area_fraction = float(
            diagnostics.get("creator_extended_area_fraction") or 0.0
        )
        if creator_extended_area_fraction < CREATOR_EXTENDED_AREA_SUSPECT_MIN:
            continue

        member_candidates = [
            candidate_by_id[str(member_id)]
            for member_id in (plane_group.get("member_ids") or [])
            if str(member_id) in candidate_by_id
        ]
        if not member_candidates:
            continue

        # Prefer non-extended segments as anchors. If none exist, fall back to
        # all members to avoid dropping targets due to sparse metadata.
        non_extended_candidates = [
            candidate
            for candidate in member_candidates
            if not bool(candidate.get("extended"))
        ]
        seed_candidates = non_extended_candidates or member_candidates

        seed_polys: list[Polygon] = []
        for candidate in seed_candidates:
            parent_segment_id = str(candidate.get("parent_segment_id") or "")
            if not parent_segment_id:
                continue
            segment = segment_by_id.get(parent_segment_id)
            if segment is None:
                continue
            footprint_poly = _poly_from_xz_ring(segment.get("footprint_xz") or [])
            if (
                footprint_poly is None
                or footprint_poly.is_empty
                or float(footprint_poly.area) < MIN_SPLIT_PIECE_AREA_M2
            ):
                continue
            seed_polys.append(footprint_poly)
        if not seed_polys:
            continue

        try:
            seed_union = unary_union(seed_polys)
        except Exception:
            continue
        if seed_union is None or getattr(seed_union, "is_empty", True):
            continue
        source_room_polys: list[Polygon] = []
        for room_id in diagnostics.get("creator_source_room_ids") or []:
            if (
                not isinstance(room_id, str)
                or not room_id.startswith("room:")
                or building is None
            ):
                continue
            try:
                room_index = int(room_id.split(":", 1)[1])
            except ValueError:
                continue
            rooms = building.get("rooms") or []
            if room_index < 0 or room_index >= len(rooms):
                continue
            room_poly = _xz_polygon(
                (rooms[room_index] or {}).get("floor_polygon") or []
            )
            if room_poly is not None and not room_poly.is_empty:
                source_room_polys.append(room_poly)
        if source_room_polys and hasattr(seed_union, "geoms"):
            try:
                source_union = unary_union(source_room_polys)
            except Exception:
                source_union = None
            if source_union is not None and not getattr(source_union, "is_empty", True):
                scored_components = [
                    (float(component.intersection(source_union).area), component)
                    for component in _iter_polygons(seed_union)
                ]
                scored_components = [
                    (overlap_area, component)
                    for overlap_area, component in scored_components
                    if overlap_area >= MIN_SPLIT_PIECE_AREA_M2
                ]
                if scored_components:
                    best_overlap = max(
                        overlap_area for overlap_area, _component in scored_components
                    )
                    kept_components = [
                        component
                        for overlap_area, component in scored_components
                        if overlap_area >= best_overlap - 1e-9
                    ]
                    try:
                        seed_union = unary_union(kept_components)
                    except Exception:
                        seed_union = kept_components[0]
        masks_by_target[target_element_id] = seed_union
    return masks_by_target


def constrain_ridge_eave_targets_with_anchor_masks(
    ridge_eave_targets: list[TargetPlaneRecord],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
    ridge_eave_target_anchor_masks: dict[str, Any],
) -> list[TargetPlaneRecord]:
    """For suspect interior-slice targets, use source-segment anchor geometry.

    ``union_xz`` from ridge/eave scoring can include long extrapolated slices
    through unrelated slab projections. The anchor masks are built from source
    oblique segments (non-extended preferred), so for suspect targets they are
    the more physical XZ support domain.
    """
    # Keep full target extents here. Anchor geometry is still useful, but only
    # as a late ownership/selection cue; clipping the target polygon itself
    # this early can truncate valid roof faces that extend beyond direct raw
    # oblique support.
    return list(ridge_eave_targets)


def _source_room_keys_from_ridge_diagnostics(
    building: dict[str, Any],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
) -> set[tuple[int, int]]:
    """Resolve ridge/eave creator source room IDs to ``(story, room_index)`` keys."""
    keys: set[tuple[int, int]] = set()
    rooms = building.get("rooms") or []
    for diag in ridge_eave_target_diagnostics.values():
        for room_id in diag.get("creator_source_room_ids") or []:
            if not isinstance(room_id, str):
                continue
            if not room_id.startswith("room:"):
                continue
            suffix = room_id.split(":", 1)[1]
            try:
                room_index = int(suffix)
            except ValueError:
                continue
            if room_index < 0 or room_index >= len(rooms):
                continue
            story = int((rooms[room_index] or {}).get("story", 0))
            keys.add((story, room_index))
    return keys


def _augment_raw_records_with_source_rooms(
    building: dict[str, Any],
    roof_result: dict[str, Any],
    raw_records: list[RawPlaneRecord],
    source_room_keys: set[tuple[int, int]],
) -> list[RawPlaneRecord]:
    """Add raw planes from ridge/eave source rooms even when not exposed."""
    if not source_room_keys:
        return raw_records
    raw_by_id: dict[str, RawPlaneRecord] = {
        record.element_id: record for record in raw_records
    }
    all_room_records = collect_raw_plane_records(
        building, roof_result, exposed_only=False
    )
    for record in all_room_records:
        if (record.story, record.room_index) in source_room_keys:
            raw_by_id[record.element_id] = record
    return list(raw_by_id.values())


def collect_target_plane_records(
    roof_result: dict[str, Any], uuid: str
) -> list[TargetPlaneRecord]:
    targets: list[TargetPlaneRecord] = []
    for idx, plane in enumerate((roof_result.get("ceiling") or {}).get("planes") or []):
        poly = _candidate_polygon(plane)
        if poly is None:
            continue
        normal = _candidate_normal(plane)
        azimuth_deg, inclination_deg = _normal_to_azimuth_inclination(normal)
        ridge_dir_xz = (float(plane["ridgeX"]), float(plane["ridgeZ"]))
        targets.append(
            TargetPlaneRecord(
                uuid=uuid,
                story=int(plane.get("dominantStory", 0)),
                target_kind="candidate_oblique",
                target_index=idx,
                element_id=f"{uuid}::ceiling-oblique::ceiling-oblique:{idx}",
                poly_xz=poly,
                normal=normal,
                azimuth_deg=azimuth_deg,
                inclination_deg=inclination_deg,
                ridge_dir_xz=ridge_dir_xz,
                area_xz_m2=float(poly.area),
                plane_point=(
                    float(plane["ref"]["x"]),
                    float(plane["ref"]["y"]),
                    float(plane["ref"]["z"]),
                ),
            )
        )
    for idx, surface in enumerate(
        (roof_result.get("roof_surfaces") or {}).get("oblique") or []
    ):
        corners = surface.get("corners") or []
        poly = _xz_polygon(corners)
        if poly is None:
            continue
        centroid, normal = _fit_plane_svd(corners)
        if centroid is None or normal is None:
            continue
        ridge = surface.get("ridge") or {}
        ridge_x = float(ridge.get("x", 0.0))
        ridge_z = float(ridge.get("z", 0.0))
        ridge_norm = math.hypot(ridge_x, ridge_z)
        if ridge_norm < 1e-9:
            roof_up = _normalize_roof_up(normal)
            ridge_x, ridge_z = float(roof_up[2]), float(-roof_up[0])
            ridge_norm = math.hypot(ridge_x, ridge_z)
        ridge_dir_xz = (
            (ridge_x / ridge_norm, ridge_z / ridge_norm)
            if ridge_norm >= 1e-9
            else (1.0, 0.0)
        )
        azimuth_deg, inclination_deg = _normal_to_azimuth_inclination(normal)
        targets.append(
            TargetPlaneRecord(
                uuid=uuid,
                story=int(surface.get("dominant_story", 0)),
                target_kind="committed_oblique",
                target_index=idx,
                element_id=f"{uuid}::roof-oblique::oblique:{idx}",
                poly_xz=poly,
                normal=normal,
                azimuth_deg=azimuth_deg,
                inclination_deg=inclination_deg,
                ridge_dir_xz=ridge_dir_xz,
                area_xz_m2=float(poly.area),
                plane_point=(
                    float(centroid[0]),
                    float(centroid[1]),
                    float(centroid[2]),
                ),
            )
        )
    return targets


def _ridge_edge_support_length(
    target: TargetPlaneRecord, raw_edges: list[RawEdgeRecord]
) -> float:
    ridge_azimuth = _ridge_dir_to_azimuth_mod180(target.ridge_dir_xz)
    total = 0.0
    for edge in raw_edges:
        if edge.story != target.story or edge.label != "ridge_or_hip":
            continue
        if (
            _wrapped_angle_delta_mod180_deg(edge.edge_azimuth_xz_deg, ridge_azimuth)
            > EDGE_ALIGNMENT_TOL_DEG
        ):
            continue
        midpoint = Point(edge.midpoint_xz)
        if not (target.poly_xz.contains(midpoint) or target.poly_xz.touches(midpoint)):
            continue
        total += edge.length_m
    return total


def _eave_edge_support_length(
    target: TargetPlaneRecord, raw_edges: list[RawEdgeRecord]
) -> float:
    ridge_azimuth = _ridge_dir_to_azimuth_mod180(target.ridge_dir_xz)
    total = 0.0
    for edge in raw_edges:
        if edge.story != target.story or edge.label != "eave":
            continue
        if (
            _wrapped_angle_delta_mod180_deg(edge.edge_azimuth_xz_deg, ridge_azimuth)
            > EDGE_ALIGNMENT_TOL_DEG
        ):
            continue
        midpoint = Point(edge.midpoint_xz)
        if target.poly_xz.boundary.distance(midpoint) > EAVE_TARGET_BOUNDARY_TOL_M:
            continue
        total += edge.length_m
    return total


def _conflicting_pair_count(
    target: TargetPlaneRecord, conflicts: list[ConflictPairRecord]
) -> int:
    count = 0
    for conflict in conflicts:
        if conflict.story != target.story:
            continue
        try:
            overlap = target.poly_xz.intersection(conflict.poly_xz)
        except Exception:
            continue
        if overlap.is_empty or float(overlap.area) < MIN_CONFLICT_OVERLAP_M2:
            continue
        count += 1
    return count


def _score_flags(
    orientation_support_score: float,
    retention_support_score: float,
    conflicting_raw_pair_count: int,
) -> tuple[str, str, bool]:
    if orientation_support_score >= 0.75:
        orientation_flag = "strong"
    elif orientation_support_score >= 0.55:
        orientation_flag = "medium"
    else:
        orientation_flag = "weak"

    if retention_support_score >= 0.65:
        retention_flag = "keep"
    elif retention_support_score >= 0.45:
        retention_flag = "maybe"
    else:
        retention_flag = "drop_candidate"

    return orientation_flag, retention_flag, conflicting_raw_pair_count >= 1


def _plane_y_at(target: TargetPlaneRecord, x: float, z: float) -> float | None:
    if target.plane_point is None:
        return None
    normal = _normalize_roof_up(target.normal)
    if abs(float(normal[1])) < 1e-6:
        return float(target.plane_point[1])
    px, py, pz = target.plane_point
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    return float(py - (nx * (x - px) + nz * (z - pz)) / ny)


def _ridge_axis_unit(target: TargetPlaneRecord) -> np.ndarray:
    axis = np.asarray(
        [float(target.ridge_dir_xz[0]), float(target.ridge_dir_xz[1])], dtype=float
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return axis / norm


def score_plane_eave_chain_supports(
    targets: list[TargetPlaneRecord],
    eave_chains: list[EaveChainRecord],
) -> list[PlaneEaveChainSupportRecord]:
    supports: list[PlaneEaveChainSupportRecord] = []
    chains_by_story: dict[int, list[EaveChainRecord]] = defaultdict(list)
    for chain in eave_chains:
        chains_by_story[chain.story].append(chain)

    for target in targets:
        ridge_azimuth = _ridge_dir_to_azimuth_mod180(target.ridge_dir_xz)
        for chain in chains_by_story.get(target.story, []):
            angle_delta = _wrapped_angle_delta_mod180_deg(
                chain.azimuth_deg, ridge_azimuth
            )
            angle_score = 1.0 - min(angle_delta / EDGE_ALIGNMENT_TOL_DEG, 1.0)
            boundary_distance = float(target.poly_xz.boundary.distance(chain.line_xz))
            boundary_score = 1.0 - min(
                boundary_distance / PLANE_EAVE_CHAIN_BOUNDARY_TOL_M, 1.0
            )
            overlap_length = float(
                target.poly_xz.buffer(PLANE_EAVE_CHAIN_BUFFER_M)
                .intersection(chain.line_xz)
                .length
            )
            chain_line_length = max(float(chain.line_xz.length), 1e-9)
            overlap_fraction = min(overlap_length / chain_line_length, 1.0)
            height_residual = None
            y_samples = []
            for point in (chain.start_xz, chain.end_xz):
                y_val = _plane_y_at(target, point[0], point[1])
                if y_val is not None:
                    y_samples.append(y_val)
            if y_samples:
                height_residual = abs(float(np.mean(y_samples)) - float(chain.y_mean))
                height_score = 1.0 - min(
                    height_residual / PLANE_EAVE_CHAIN_HEIGHT_TOL_M, 1.0
                )
            else:
                height_score = 0.0

            support_score = (
                0.4 * angle_score
                + 0.35 * overlap_fraction
                + 0.15 * boundary_score
                + 0.10 * height_score
            )
            supported = (
                support_score >= PLANE_EAVE_CHAIN_SUPPORT_THRESHOLD
                and overlap_fraction >= 0.25
                and boundary_distance <= PLANE_EAVE_CHAIN_BOUNDARY_TOL_M
            )
            supports.append(
                PlaneEaveChainSupportRecord(
                    uuid=target.uuid,
                    story=target.story,
                    target_element_id=target.element_id,
                    target_kind=target.target_kind,
                    chain_id=chain.chain_id,
                    chain_azimuth_deg=chain.azimuth_deg,
                    ridge_azimuth_deg=ridge_azimuth,
                    angle_delta_deg=angle_delta,
                    boundary_distance_m=boundary_distance,
                    overlap_fraction=overlap_fraction,
                    height_residual_m=height_residual,
                    support_score=support_score,
                    supported=supported,
                    chain_length_m=chain.total_length_m,
                )
            )
    return supports


def _chain_room_keys(chain: EaveChainRecord) -> set[str]:
    room_keys: set[str] = set()
    for plane_id in chain.member_plane_ids:
        suffix = str(plane_id).split("::")[-1]
        parts = suffix.split(":")
        if len(parts) != 3:
            continue
        room_keys.add(f"{parts[0]}:{parts[1]}")
    return room_keys


def _chain_endpoint_gap(chain_a: EaveChainRecord, chain_b: EaveChainRecord) -> float:
    endpoints_a = [chain_a.start_xz, chain_a.end_xz]
    endpoints_b = [chain_b.start_xz, chain_b.end_xz]
    min_gap = float("inf")
    for ax, az in endpoints_a:
        for bx, bz in endpoints_b:
            min_gap = min(
                min_gap, math.hypot(float(ax) - float(bx), float(az) - float(bz))
            )
    return min_gap


def _chains_share_facade_component(
    chain_a: EaveChainRecord, chain_b: EaveChainRecord
) -> bool:
    if chain_a.story != chain_b.story:
        return False
    if (
        _wrapped_angle_delta_mod180_deg(chain_a.azimuth_deg, chain_b.azimuth_deg)
        > EAVE_CHAIN_ANGLE_TOL_DEG
    ):
        return False
    if abs(float(chain_a.y_mean) - float(chain_b.y_mean)) > EAVE_CHAIN_Y_TOL_M:
        return False
    if set(chain_a.member_plane_ids).intersection(chain_b.member_plane_ids):
        return True
    if _chain_room_keys(chain_a).intersection(_chain_room_keys(chain_b)):
        return True
    if _chain_endpoint_gap(chain_a, chain_b) <= EAVE_CHAIN_GAP_M:
        return True
    return False


def _build_eave_chain_neighbors(
    eave_chains: list[EaveChainRecord],
) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    story_chains: dict[int, list[EaveChainRecord]] = defaultdict(list)
    for chain in eave_chains:
        story_chains[chain.story].append(chain)
    for chains in story_chains.values():
        for idx, chain_a in enumerate(chains):
            for chain_b in chains[idx + 1 :]:
                if not _chains_share_facade_component(chain_a, chain_b):
                    continue
                neighbors[chain_a.chain_id].add(chain_b.chain_id)
                neighbors[chain_b.chain_id].add(chain_a.chain_id)
    return neighbors


def expand_plane_eave_chain_supports_by_facade_continuity(
    eave_chains: list[EaveChainRecord],
    plane_chain_supports: list[PlaneEaveChainSupportRecord],
) -> list[PlaneEaveChainSupportRecord]:
    {chain.chain_id: chain for chain in eave_chains}
    neighbors = _build_eave_chain_neighbors(eave_chains)

    supports_by_target: dict[str, dict[str, PlaneEaveChainSupportRecord]] = defaultdict(
        dict
    )
    for support in plane_chain_supports:
        supports_by_target[support.target_element_id][support.chain_id] = support

    augmented: list[PlaneEaveChainSupportRecord] = []
    for target_supports in supports_by_target.values():
        seed_ids = {
            chain_id
            for chain_id, support in target_supports.items()
            if support.supported
        }
        visited = set(seed_ids)
        queue = list(seed_ids)
        promoted_ids: set[str] = set()
        while queue:
            chain_id = queue.pop(0)
            for neighbor_id in neighbors.get(chain_id, set()):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                support = target_supports.get(neighbor_id)
                if support is None:
                    continue
                if support.angle_delta_deg > EDGE_ALIGNMENT_TOL_DEG:
                    continue
                promoted_ids.add(neighbor_id)
                queue.append(neighbor_id)

        for chain_id, support in target_supports.items():
            if chain_id in promoted_ids and not support.supported:
                augmented.append(replace(support, supported=True))
            else:
                augmented.append(support)
    return augmented


def _iter_polygons(geom: Any) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom] if float(geom.area) >= MIN_SPLIT_PIECE_AREA_M2 else []
    polygons: list[Polygon] = []
    for part in getattr(geom, "geoms", []):
        polygons.extend(_iter_polygons(part))
    return polygons


def _projection_interval(
    axis: np.ndarray, start_xz: tuple[float, float], end_xz: tuple[float, float]
) -> tuple[float, float]:
    start_t = float(np.dot(axis, np.asarray(start_xz, dtype=float)))
    end_t = float(np.dot(axis, np.asarray(end_xz, dtype=float)))
    return (min(start_t, end_t), max(start_t, end_t))


def _polygon_projection_bounds(
    poly: Polygon, axis: np.ndarray
) -> tuple[float, float] | None:
    if poly.is_empty:
        return None
    values: list[float] = []
    polygons = [poly] if isinstance(poly, Polygon) else list(getattr(poly, "geoms", []))
    for part in polygons:
        if not isinstance(part, Polygon) or part.is_empty:
            continue
        for x, z in list(part.exterior.coords):
            values.append(
                float(np.dot(axis, np.asarray([float(x), float(z)], dtype=float)))
            )
        for ring in part.interiors:
            for x, z in list(ring.coords):
                values.append(
                    float(np.dot(axis, np.asarray([float(x), float(z)], dtype=float)))
                )
    if not values:
        return None
    return (min(values), max(values))


def _axis_strip_polygon(
    target_poly: Polygon, axis: np.ndarray, start_t: float, end_t: float
) -> Polygon:
    minx, minz, maxx, maxz = target_poly.bounds
    width = (
        math.hypot(maxx - minx, maxz - minz)
        + 2.0 * (PLANE_EAVE_CHAIN_BOUNDARY_TOL_M + EAVE_CHAIN_GAP_M)
        + 1.0
    )
    ortho = np.array([-float(axis[1]), float(axis[0])], dtype=float)
    corners = [
        axis * start_t - ortho * width,
        axis * end_t - ortho * width,
        axis * end_t + ortho * width,
        axis * start_t + ortho * width,
    ]
    return Polygon([(float(pt[0]), float(pt[1])) for pt in corners])


def _axis_window_polygon(
    target_poly: Polygon,
    axis: np.ndarray,
    ortho: np.ndarray,
    start_t: float,
    end_t: float,
    min_u: float,
    max_u: float,
    *,
    pad_m: float | None = None,
) -> Polygon:
    pad = (
        PLANE_EAVE_CHAIN_BOUNDARY_TOL_M + EAVE_CHAIN_GAP_M
        if pad_m is None
        else max(float(pad_m), 0.0)
    )
    corners = [
        axis * (start_t - pad) + ortho * (min_u - pad),
        axis * (end_t + pad) + ortho * (min_u - pad),
        axis * (end_t + pad) + ortho * (max_u + pad),
        axis * (start_t - pad) + ortho * (max_u + pad),
    ]
    poly = Polygon([(float(pt[0]), float(pt[1])) for pt in corners])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return target_poly.intersection(poly)


def _point_interior_distance(poly: Polygon, point_xz: tuple[float, float]) -> float:
    point = Point(float(point_xz[0]), float(point_xz[1]))
    if not (poly.contains(point) or poly.touches(point)):
        return -1.0
    try:
        boundary = poly.boundary
    except Exception:
        return -1.0
    if boundary is None or getattr(boundary, "is_empty", True):
        return -1.0
    try:
        return float(boundary.distance(point))
    except Exception:
        return -1.0


def _target_uphill_sweep_direction(
    target: TargetPlaneRecord, axis: np.ndarray
) -> np.ndarray | None:
    normal = _normalize_roof_up(np.asarray(target.normal, dtype=float))
    if normal.shape[0] < 3:
        return None
    normal_xz = np.asarray([float(normal[0]), float(normal[2])], dtype=float)
    normal_xz_norm = float(np.linalg.norm(normal_xz))
    if normal_xz_norm < 1e-9:
        return None
    downhill = normal_xz / normal_xz_norm
    uphill = -downhill
    ortho = np.asarray([-float(axis[1]), float(axis[0])], dtype=float)
    if float(np.dot(uphill, ortho)) >= 0.0:
        return ortho
    return -ortho


def _chain_inward_sweep_direction(
    target: TargetPlaneRecord,
    target_poly: Polygon,
    chain: EaveChainRecord,
    axis: np.ndarray,
) -> np.ndarray:
    plane_direction = _target_uphill_sweep_direction(target, axis)
    ortho = np.asarray([-float(axis[1]), float(axis[0])], dtype=float)
    midpoint = chain.line_xz.interpolate(0.5, normalized=True)
    midpoint_xz = (float(midpoint.x), float(midpoint.y))
    step = max(float(PLANE_EAVE_CHAIN_BUFFER_M), 1e-3)
    candidates = [ortho, -ortho]
    scored: list[tuple[float, np.ndarray]] = []
    for direction in candidates:
        probe = (
            midpoint_xz[0] + float(direction[0]) * step,
            midpoint_xz[1] + float(direction[1]) * step,
        )
        scored.append((_point_interior_distance(target_poly, probe), direction))
    if plane_direction is not None:
        for distance, direction in scored:
            if distance >= 0.0 and float(np.dot(direction, plane_direction)) > 0.0:
                return direction
        for distance, direction in scored:
            if distance >= 0.0:
                return direction
        return plane_direction
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored and scored[0][0] >= 0.0:
        return scored[0][1]

    rep = target_poly.representative_point()
    toward_rep = np.asarray(
        [float(rep.x) - midpoint_xz[0], float(rep.y) - midpoint_xz[1]],
        dtype=float,
    )
    if float(np.dot(toward_rep, ortho)) >= 0.0:
        return ortho
    return -ortho


def _chain_inward_sweep_polygon(
    target: TargetPlaneRecord,
    target_poly: Polygon,
    chain: EaveChainRecord,
    axis: np.ndarray,
) -> Polygon | None:
    if target_poly.is_empty:
        return None
    try:
        line_cover = target_poly.intersection(
            chain.line_xz.buffer(PLANE_EAVE_CHAIN_BUFFER_M, cap_style=2)
        )
    except Exception:
        line_cover = None

    minx, minz, maxx, maxz = target_poly.bounds
    sweep_len = (
        math.hypot(maxx - minx, maxz - minz)
        + 2.0 * (PLANE_EAVE_CHAIN_BOUNDARY_TOL_M + EAVE_CHAIN_GAP_M)
        + 1.0
    )
    start = np.asarray(chain.start_xz, dtype=float)
    end = np.asarray(chain.end_xz, dtype=float)
    ortho = np.asarray([-float(axis[1]), float(axis[0])], dtype=float)
    plane_direction = _target_uphill_sweep_direction(target, axis)

    candidate_directions: list[np.ndarray] = []
    if plane_direction is not None:
        if float(np.dot(ortho, plane_direction)) >= 0.0:
            candidate_directions.append(ortho)
            candidate_directions.append(-ortho)
        else:
            candidate_directions.append(-ortho)
            candidate_directions.append(ortho)
    else:
        direction = _chain_inward_sweep_direction(target, target_poly, chain, axis)
        candidate_directions.append(direction)
        if not np.allclose(direction, ortho):
            candidate_directions.append(ortho)
        if not np.allclose(direction, -ortho):
            candidate_directions.append(-ortho)

    best_poly: Polygon | None = None
    best_area = -1.0
    for direction in candidate_directions:
        sweep_poly = Polygon(
            [
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
                (
                    float(end[0] + direction[0] * sweep_len),
                    float(end[1] + direction[1] * sweep_len),
                ),
                (
                    float(start[0] + direction[0] * sweep_len),
                    float(start[1] + direction[1] * sweep_len),
                ),
            ]
        )
        if not sweep_poly.is_valid:
            sweep_poly = sweep_poly.buffer(0)
        if sweep_poly.is_empty:
            continue
        try:
            clipped = target_poly.intersection(sweep_poly)
        except Exception:
            continue
        if clipped.is_empty:
            continue
        pieces = _iter_polygons(clipped)
        if line_cover is not None and not getattr(line_cover, "is_empty", True):
            pieces = [
                poly for poly in pieces if not poly.intersection(line_cover).is_empty
            ]
        if not pieces:
            continue
        try:
            merged = unary_union(pieces)
        except Exception:
            merged = pieces[0]
        if getattr(merged, "is_empty", True):
            continue
        if isinstance(merged, Polygon):
            candidate_poly = merged
        else:
            polygons = _iter_polygons(merged)
            if not polygons:
                continue
            polygons.sort(key=lambda poly: float(poly.area), reverse=True)
            candidate_poly = polygons[0]
        candidate_area = float(candidate_poly.area)
        if (
            plane_direction is not None
            and float(np.dot(direction, plane_direction)) > 0.0
        ):
            return candidate_poly
        if candidate_area > best_area:
            best_poly = candidate_poly
            best_area = candidate_area
    return best_poly


def trim_ridge_eave_supported_pieces_to_chain_run_bands(
    split_targets: list[TargetPlaneRecord],
    eave_chains: list[EaveChainRecord],
    split_pieces: list[TargetSplitPieceRecord],
) -> list[TargetSplitPieceRecord]:
    """Partition same-facing ridge/eave runs by their owned chain intervals.

    Broad plane-group polygons can make two local runs of the same roof side
    overlap. The owned eave chains already tell us the architectural ordering
    of those runs along the ridge axis, so clip each supported piece to the
    midpoint band between adjacent disjoint chain signatures.
    """
    targets_by_id = {target.element_id: target for target in split_targets}
    chain_by_id = {chain.chain_id: chain for chain in eave_chains}
    piece_rows_by_story: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for piece in split_pieces:
        if (
            piece.target_kind != "ridge_eave_plane_group"
            or piece.piece_role != "supported"
        ):
            continue
        target = targets_by_id.get(piece.target_element_id)
        piece_poly = _xz_polygon(piece.corners)
        signature = _chain_signature(piece.chain_ids)
        if target is None or piece_poly is None or piece_poly.is_empty or not signature:
            continue
        axis = _ridge_axis_unit(target)
        interval_values: list[float] = []
        for chain_id in signature:
            chain = chain_by_id.get(chain_id)
            if chain is None:
                continue
            interval_values.extend(
                [
                    float(np.dot(axis, np.asarray(chain.start_xz, dtype=float))),
                    float(np.dot(axis, np.asarray(chain.end_xz, dtype=float))),
                ]
            )
        if not interval_values:
            continue
        piece_rows_by_story[piece.story].append(
            {
                "piece": piece,
                "target": target,
                "poly": piece_poly,
                "signature": signature,
                "axis": axis,
                "interval_min_t": min(interval_values),
                "interval_max_t": max(interval_values),
            }
        )

    trimmed_by_piece_id: dict[str, TargetSplitPieceRecord] = {}
    for story_rows in piece_rows_by_story.values():
        pending = list(story_rows)
        family_groups: list[list[dict[str, Any]]] = []
        while pending:
            seed = pending.pop(0)
            group = [seed]
            changed = True
            while changed:
                changed = False
                for row in list(pending):
                    if any(
                        _targets_are_local_ownership_competitors(
                            row["target"], member["target"]
                        )
                        for member in group
                    ):
                        group.append(row)
                        pending.remove(row)
                        changed = True
            family_groups.append(group)

        for family in family_groups:
            if len(family) <= 1:
                continue
            family_axis = np.asarray(family[0]["axis"], dtype=float)
            family_axis = family_axis / max(float(np.linalg.norm(family_axis)), 1e-9)
            family_ortho = np.asarray(
                [-float(family_axis[1]), float(family_axis[0])], dtype=float
            )

            normalized_rows: list[dict[str, Any]] = []
            for row in family:
                axis = np.asarray(row["axis"], dtype=float)
                if float(np.dot(axis, family_axis)) < 0.0:
                    axis = -axis
                interval_values: list[float] = []
                for chain_id in row["signature"]:
                    chain = chain_by_id.get(chain_id)
                    if chain is None:
                        continue
                    interval_values.extend(
                        [
                            float(
                                np.dot(
                                    family_axis, np.asarray(chain.start_xz, dtype=float)
                                )
                            ),
                            float(
                                np.dot(
                                    family_axis, np.asarray(chain.end_xz, dtype=float)
                                )
                            ),
                        ]
                    )
                if not interval_values:
                    continue
                normalized_rows.append(
                    {
                        **row,
                        "axis": axis,
                        "interval_min_t": min(interval_values),
                        "interval_max_t": max(interval_values),
                    }
                )
            if len(normalized_rows) <= 1:
                continue
            normalized_rows.sort(
                key=lambda row: (
                    0.5 * (float(row["interval_min_t"]) + float(row["interval_max_t"])),
                    float(row["interval_min_t"]),
                    str(row["piece"].piece_id),
                )
            )

            boundaries: dict[str, dict[str, float]] = {
                str(row["piece"].piece_id): {} for row in normalized_rows
            }
            for left_row, right_row in itertools.pairwise(normalized_rows):
                if (
                    _chain_signature_relation(
                        left_row["signature"], right_row["signature"]
                    )
                    != "disjoint"
                ):
                    continue
                boundary_t = 0.5 * (
                    float(left_row["interval_max_t"])
                    + float(right_row["interval_min_t"])
                )
                boundaries[str(left_row["piece"].piece_id)]["max_t"] = boundary_t
                boundaries[str(right_row["piece"].piece_id)]["min_t"] = boundary_t

            for row in normalized_rows:
                piece = row["piece"]
                piece_id = str(piece.piece_id)
                piece_poly = row["poly"]
                piece_bounds = _polygon_projection_bounds(piece_poly, family_axis)
                ortho_bounds = _polygon_projection_bounds(piece_poly, family_ortho)
                if piece_bounds is None or ortho_bounds is None:
                    continue
                min_t = float(boundaries[piece_id].get("min_t", piece_bounds[0]))
                max_t = float(boundaries[piece_id].get("max_t", piece_bounds[1]))
                if max_t <= min_t + 1e-9:
                    continue
                try:
                    clipped = _axis_window_polygon(
                        piece_poly,
                        family_axis,
                        family_ortho,
                        min_t,
                        max_t,
                        float(ortho_bounds[0]),
                        float(ortho_bounds[1]),
                        pad_m=0.0,
                    )
                except Exception:
                    continue
                clipped_area = float(getattr(clipped, "area", 0.0) or 0.0)
                if clipped.is_empty or clipped_area < MIN_SPLIT_PIECE_AREA_M2:
                    continue
                if clipped_area >= 0.995 * max(float(piece_poly.area), 1e-9):
                    continue
                target = row["target"]
                clipped_polys = _iter_polygons(clipped)
                if not clipped_polys:
                    continue
                clipped_polys.sort(key=lambda poly: float(poly.area), reverse=True)
                record = _piece_records_from_polygon(
                    target,
                    clipped_polys[0],
                    piece_id=piece.piece_id,
                    piece_index=piece.piece_index,
                    piece_role=piece.piece_role,
                    support_score=piece.support_score,
                    chain_ids=piece.chain_ids,
                )
                if record is not None:
                    trimmed_by_piece_id[piece_id] = record

    if not trimmed_by_piece_id:
        return split_pieces
    return [
        trimmed_by_piece_id.get(str(piece.piece_id), piece) for piece in split_pieces
    ]


def _ring_to_plane_corners(
    target: TargetPlaneRecord, coords: list[tuple[float, float]]
) -> list[list[float]]:
    corners: list[list[float]] = []
    for x, z in coords:
        y = _plane_y_at(target, float(x), float(z))
        if y is None:
            y = float(target.plane_point[1]) if target.plane_point is not None else 0.0
        corners.append([float(x), float(y), float(z)])
    return corners


def _target_plane_coeffs(
    target: TargetPlaneRecord,
) -> tuple[float, float, float, float] | None:
    if target.plane_point is None:
        return None
    normal = _normalize_roof_up(np.asarray(target.normal, dtype=float))
    if normal.shape[0] < 3:
        return None
    point = np.asarray(target.plane_point, dtype=float)
    if point.shape[0] < 3:
        return None
    a = float(normal[0])
    b = float(normal[1])
    c = float(normal[2])
    d = -float(np.dot(normal, point))
    return a, b, c, d


def _mirror_half_plane_coeffs(
    own_target: TargetPlaneRecord,
    other_target: TargetPlaneRecord,
) -> tuple[float, float, float] | None:
    own_plane = _target_plane_coeffs(own_target)
    other_plane = _target_plane_coeffs(other_target)
    if own_plane is None or other_plane is None:
        return None
    a_i, b_i, c_i, d_i = own_plane
    a_j, b_j, c_j, d_j = other_plane
    return (
        b_j * a_i - b_i * a_j,
        b_j * c_i - b_i * c_j,
        b_j * d_i - b_i * d_j,
    )


def _half_plane_polygon_for_bounds(
    coeffs: tuple[float, float, float],
    bounds: tuple[float, float, float, float],
    *,
    pad_m: float = 1.0,
    eps: float = 1e-9,
) -> Polygon | None:
    min_x, min_z, max_x, max_z = bounds
    min_x -= pad_m
    min_z -= pad_m
    max_x += pad_m
    max_z += pad_m
    a, b, c = coeffs
    bbox = [
        (min_x, min_z),
        (max_x, min_z),
        (max_x, max_z),
        (min_x, max_z),
    ]

    def _value(pt: tuple[float, float]) -> float:
        return a * float(pt[0]) + b * float(pt[1]) + c

    values = [_value(pt) for pt in bbox]
    if all(value >= -eps for value in values):
        return Polygon(bbox)
    if all(value <= eps for value in values):
        return None

    points: list[tuple[float, float]] = []
    for idx, point in enumerate(bbox):
        value = values[idx]
        if value >= -eps:
            points.append(point)
        nxt = bbox[(idx + 1) % len(bbox)]
        next_value = values[(idx + 1) % len(bbox)]
        crosses = (value > eps and next_value < -eps) or (
            value < -eps and next_value > eps
        )
        if not crosses:
            continue
        denom = value - next_value
        if abs(denom) <= 1e-12:
            continue
        t = value / denom
        points.append(
            (
                float(point[0]) + t * (float(nxt[0]) - float(point[0])),
                float(point[1]) + t * (float(nxt[1]) - float(point[1])),
            )
        )
    if len(points) < 3:
        return None
    centroid_x = float(np.mean([pt[0] for pt in points]))
    centroid_z = float(np.mean([pt[1] for pt in points]))
    ordered = sorted(
        points,
        key=lambda pt: math.atan2(float(pt[1]) - centroid_z, float(pt[0]) - centroid_x),
    )
    poly = Polygon(ordered)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _piece_records_from_polygon(
    target: TargetPlaneRecord,
    poly: Polygon,
    *,
    piece_id: str,
    piece_index: int,
    piece_role: str,
    support_score: float | None,
    chain_ids: tuple[str, ...],
) -> TargetSplitPieceRecord | None:
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if not isinstance(poly, Polygon):
        polys = _iter_polygons(poly)
        if not polys:
            return None
        poly = max(polys, key=lambda item: float(item.area))
    if poly.is_empty or float(poly.area) < MIN_SPLIT_PIECE_AREA_M2:
        return None
    outer = list(poly.exterior.coords)[:-1]
    if len(outer) < 3:
        return None
    holes = [
        _ring_to_plane_corners(target, list(ring.coords)[:-1])
        for ring in poly.interiors
        if len(list(ring.coords)) >= 4
    ]
    return TargetSplitPieceRecord(
        uuid=target.uuid,
        story=target.story,
        target_element_id=target.element_id,
        target_kind=target.target_kind,
        piece_id=piece_id,
        piece_index=piece_index,
        piece_role=piece_role,
        area_xz_m2=float(poly.area),
        support_score=support_score,
        chain_ids=chain_ids,
        corners=_ring_to_plane_corners(target, outer),
        holes=holes,
    )


def _merge_supported_gap_polygons(
    base_poly: Polygon,
    supported_candidates: list[dict[str, Any]],
    gap_polys: list[Polygon] | None = None,
) -> tuple[list[dict[str, Any]], list[Polygon]]:
    if not supported_candidates:
        return supported_candidates, []

    try:
        supported_union = unary_union(
            [candidate["poly"] for candidate in supported_candidates]
        )
    except Exception:
        return supported_candidates, []
    residual_polys = _iter_polygons(base_poly.difference(supported_union))
    if not residual_polys:
        return supported_candidates, []

    try:
        support_hull = supported_union.convex_hull
    except Exception:
        support_hull = None
    if gap_polys:
        try:
            gap_union = unary_union(gap_polys)
        except Exception:
            gap_union = None
    else:
        gap_union = None

    bridged_residuals: list[Polygon] = []
    for residual in residual_polys:
        touching = [
            candidate
            for candidate in supported_candidates
            if residual.distance(candidate["poly"]) <= 1e-6
        ]
        gap_overlap_area = 0.0
        if gap_polys:
            residual_area = max(float(residual.area), 1e-9)
            gap_bridge_parts: list[Polygon] = []
            for gap_overlap in _iter_polygons(
                residual.intersection(gap_union) if gap_union is not None else None
            ):
                gap_bridge_parts.append(gap_overlap)
                gap_overlap_area += float(gap_overlap.area)
            gap_overlap_fraction = gap_overlap_area / residual_area
        else:
            gap_bridge_parts = []
            gap_overlap_fraction = 0.0
        is_hull_bridge = len(touching) >= 2 and (
            support_hull is None
            or support_hull.buffer(1e-6).contains(residual.representative_point())
        )
        is_gap_bridge = (
            len(touching) >= 1
            and gap_overlap_area >= MIN_GAP_CONTINUATION_OVERLAP_M2
            and gap_overlap_fraction >= MIN_GAP_CONTINUATION_OVERLAP_FRACTION
        )
        if not (is_hull_bridge or is_gap_bridge):
            continue
        if is_hull_bridge:
            bridged_residuals.append(residual)
            continue
        bridged_residuals.extend(gap_bridge_parts)

    # Direct gap-slice continuation: when a gap polygon lies on the same
    # target and is inside/near the current support hull, bridge it even if the
    # residual island does not edge-touch an existing supported strip.
    if gap_union is not None:
        try:
            direct_gap_geom = base_poly.intersection(gap_union).difference(
                supported_union
            )
        except Exception:
            direct_gap_geom = None
        for gap_slice in _iter_polygons(direct_gap_geom):
            if float(gap_slice.area) < MIN_GAP_CONTINUATION_OVERLAP_M2:
                continue
            near_supported = (
                float(supported_union.distance(gap_slice)) <= PLANE_EAVE_CHAIN_BUFFER_M
            )
            in_support_hull = support_hull is not None and support_hull.buffer(
                1e-6
            ).contains(gap_slice.representative_point())
            if not (near_supported or in_support_hull):
                continue
            bridged_residuals.append(gap_slice)

    if not bridged_residuals:
        return supported_candidates, []

    try:
        merged_union = unary_union(
            [candidate["poly"] for candidate in supported_candidates]
            + bridged_residuals
        )
    except Exception:
        return supported_candidates, []

    merged_candidates: list[dict[str, Any]] = []
    for poly in _iter_polygons(merged_union):
        touching = [
            candidate
            for candidate in supported_candidates
            if not poly.intersection(candidate["poly"]).is_empty
        ]
        if not touching:
            continue
        merged_candidates.append(
            {
                "poly": poly,
                "support_score": max(
                    float(candidate["support_score"]) for candidate in touching
                ),
                "chain_ids": tuple(
                    sorted(
                        {
                            chain_id
                            for candidate in touching
                            for chain_id in candidate["chain_ids"]
                        }
                    )
                ),
            }
        )
    return merged_candidates, bridged_residuals


def _ridge_eave_chain_owner_rank(
    support: PlaneEaveChainSupportRecord,
) -> tuple[float, float, float, float]:
    height_residual = (
        round(float(support.height_residual_m), 6)
        if support.height_residual_m is not None
        else float("inf")
    )
    return (
        height_residual,
        -round(float(support.overlap_fraction), 6),
        round(float(support.boundary_distance_m), 6),
        -round(float(support.support_score), 6),
    )


def _ridge_eave_chain_owner_equivalent(
    left: PlaneEaveChainSupportRecord,
    right: PlaneEaveChainSupportRecord,
    *,
    eps: float = 1e-6,
) -> bool:
    left_height = (
        float(left.height_residual_m)
        if left.height_residual_m is not None
        else float("inf")
    )
    right_height = (
        float(right.height_residual_m)
        if right.height_residual_m is not None
        else float("inf")
    )
    return (
        abs(left_height - right_height) <= eps
        and abs(float(left.overlap_fraction) - float(right.overlap_fraction)) <= eps
        and abs(float(left.boundary_distance_m) - float(right.boundary_distance_m))
        <= eps
        and abs(float(left.support_score) - float(right.support_score)) <= eps
    )


def select_locally_owned_ridge_eave_chain_supports(
    targets: list[TargetPlaneRecord],
    plane_chain_supports: list[PlaneEaveChainSupportRecord],
) -> list[PlaneEaveChainSupportRecord]:
    targets_by_id = {target.element_id: target for target in targets}
    ridge_targets = {
        target.element_id: target
        for target in targets
        if target.target_kind == "ridge_eave_plane_group"
    }
    supports_by_story_chain: dict[
        tuple[int, str], list[PlaneEaveChainSupportRecord]
    ] = defaultdict(list)
    for support in plane_chain_supports:
        if not support.supported or support.target_element_id not in ridge_targets:
            continue
        supports_by_story_chain[(support.story, support.chain_id)].append(support)

    owned_pairs: set[tuple[str, str]] = set()
    for (_story, _chain_id), supports in supports_by_story_chain.items():
        for support in supports:
            owned_pairs.add((support.target_element_id, support.chain_id))
        family_groups: list[list[PlaneEaveChainSupportRecord]] = []
        for support in supports:
            target = ridge_targets.get(support.target_element_id)
            if target is None:
                continue
            placed = False
            for group in family_groups:
                group_target = ridge_targets.get(group[0].target_element_id)
                if group_target is None:
                    continue
                if _targets_are_local_ownership_competitors(target, group_target):
                    group.append(support)
                    placed = True
                    break
            if not placed:
                family_groups.append([support])
        for family in family_groups:
            if len(family) <= 1:
                continue
            ranked = sorted(family, key=_ridge_eave_chain_owner_rank)
            owned_pairs.discard((ranked[0].target_element_id, ranked[0].chain_id))
            owned_pairs.add((ranked[0].target_element_id, ranked[0].chain_id))
            for other in ranked[1:]:
                owned_pairs.discard((other.target_element_id, other.chain_id))
                if _ridge_eave_chain_owner_equivalent(ranked[0], other):
                    owned_pairs.add((other.target_element_id, other.chain_id))

    filtered: list[PlaneEaveChainSupportRecord] = []
    for support in plane_chain_supports:
        target = targets_by_id.get(support.target_element_id)
        if support.target_element_id not in ridge_targets or not support.supported:
            filtered.append(support)
            continue
        keep = (support.target_element_id, support.chain_id) in owned_pairs
        if keep:
            filtered.append(support)
            continue
        filtered.append(replace(support, supported=False))
    return filtered


def build_plane_extent_split_pieces(
    targets: list[TargetPlaneRecord],
    eave_chains: list[EaveChainRecord],
    plane_chain_supports: list[PlaneEaveChainSupportRecord],
    story_extent_envelopes: dict[int, Polygon] | None = None,
    story_gap_polygons: dict[int, list[Polygon]] | None = None,
    target_segment_anchor_masks: dict[str, Any] | None = None,
    segment_anchor_buffer_m: float = RIDGE_EAVE_SEGMENT_ANCHOR_BUFFER_M,
    part_eave_envelopes: dict[str, Any] | None = None,
    target_allowed_part_ids: dict[str, set[str]] | None = None,
    story_eave_envelopes: dict[int, Any] | None = None,
    extension_provenance_out: dict[str, dict[str, Any]] | None = None,
) -> list[TargetSplitPieceRecord]:
    chain_by_id = {chain.chain_id: chain for chain in eave_chains}
    # Pre-index target polygons per (uuid, story, target_kind) so eave
    # widening can subtract SAME-KIND sisters' fitted XZ extents — widening
    # must reach the eave but never absorb another roof face's area.
    #
    # Keyed by target_kind so a committed_oblique only subtracts sister
    # committed_obliques and a ridge_eave_plane_group only subtracts sister
    # ridge_eave_plane_groups. This preserves the historical property that a
    # committed_oblique can grow inside a matched ridge/eave group's
    # envelope (the group's poly_xz is a GROUP-level union of a whole
    # ridge-eave pair, so subtracting it would cancel widening on every
    # building whose committed faces sit inside a matched ridge-eave group).
    # Ridge/eave groups still subtract each other so widening one cannot
    # absorb a neighbour group's slab area.
    sibling_polys_by_key: dict[tuple[str, int, str], list[tuple[str, Polygon]]] = (
        defaultdict(list)
    )
    for other in targets:
        if other.target_kind not in ("committed_oblique", "ridge_eave_plane_group"):
            continue
        poly = other.poly_xz
        if poly is None or getattr(poly, "is_empty", True):
            continue
        sibling_polys_by_key[(other.uuid, other.story, other.target_kind)].append(
            (
                other.element_id,
                poly,
            )
        )
    chain_neighbors = _build_eave_chain_neighbors(eave_chains)
    plane_chain_supports = select_locally_owned_ridge_eave_chain_supports(
        targets,
        plane_chain_supports,
    )
    supports_by_target: dict[str, list[PlaneEaveChainSupportRecord]] = defaultdict(list)
    for support in plane_chain_supports:
        if support.supported:
            supports_by_target[support.target_element_id].append(support)

    pieces: list[TargetSplitPieceRecord] = []
    for target in targets:
        supported = supports_by_target.get(target.element_id, [])
        if not supported:
            continue
        base_poly = target.poly_xz
        extent_poly = (story_extent_envelopes or {}).get(target.story)
        if extent_poly is not None:
            try:
                base_poly = target.poly_xz.intersection(extent_poly)
            except Exception:
                base_poly = target.poly_xz
        if base_poly.is_empty or float(base_poly.area) < MIN_SPLIT_PIECE_AREA_M2:
            continue

        # Widen ``base_poly`` along the down-slope direction so supported +
        # residual pieces reach the physical eave (slab + neighbouring-gap
        # union) rather than stopping at the target's fitted XZ ring.
        # Envelope source preference:
        #   1. Per-part envelope (``part_eave_envelopes`` keyed by the
        #      target's ``allowed_part_ids``) — tight, respects the
        #      building-part graph.
        #   2. Per-story envelope fallback (``story_eave_envelopes``) when
        #      the part graph is absent or the target has no allowed parts.
        #      Strictly larger; sister-target subtraction is the safety net
        #      against absorbing another roof face's slab area.
        # Ridge/eave plane groups span both slopes around the ridge, so a
        # single down-slope half-plane would only widen one eave —
        # ``halfplane`` is set to ``None`` for them and ``_chain_inward_sweep``
        # below picks per-chain which slope each piece belongs to.
        pre_extension_base_area = float(getattr(base_poly, "area", 0.0) or 0.0)
        target_extension_info: dict[str, Any] | None = None
        allowed_part_ids = (target_allowed_part_ids or {}).get(
            target.element_id
        ) or set()
        envs: list[Any] = []
        envelope_source: str | None = None
        envelope_parts: list[str] = []
        # Only widen kinds that produce final-layer roof faces — committed
        # obliques (the user-visible roof) and ridge/eave plane groups (the
        # paired-plane representation). Candidate obliques never widen:
        # they exist as raw evidence and the partitioner uses them to
        # carve area away from the committed face, not to claim new area.
        widening_eligible = target.target_kind in (
            "committed_oblique",
            "ridge_eave_plane_group",
        )
        if widening_eligible and part_eave_envelopes and allowed_part_ids:
            envs = [
                part_eave_envelopes[pid]
                for pid in sorted(allowed_part_ids)
                if pid in part_eave_envelopes
            ]
            if envs:
                envelope_source = "part_slabs+neighbouring_gaps"
                envelope_parts = sorted(
                    pid for pid in allowed_part_ids if pid in part_eave_envelopes
                )
        if widening_eligible and not envs and story_eave_envelopes:
            story_env = story_eave_envelopes.get(target.story)
            if story_env is not None and not getattr(story_env, "is_empty", True):
                envs = [story_env]
                envelope_source = "story_slabs+neighbouring_gaps"
        if envs:
            try:
                eave_env = unary_union(envs) if len(envs) > 1 else envs[0]
            except Exception:
                eave_env = None
            if eave_env is not None and not getattr(eave_env, "is_empty", True):
                # Ridge/eave plane groups span both slopes around the
                # ridge, so a single down-slope half-plane would only
                # widen one eave. Skip the directional trim for them and
                # let the per-chain inward sweep below pick which slope
                # each piece belongs to.
                halfplane = (
                    None
                    if target.target_kind == "ridge_eave_plane_group"
                    else _downslope_halfplane_polygon(target, target.poly_xz)
                )
                try:
                    eave_add = (
                        eave_env.intersection(halfplane)
                        if halfplane is not None
                        else eave_env
                    )
                except Exception:
                    eave_add = None
                # Subtract the fitted XZ of OTHER roof-face targets on the
                # same story so widening cannot absorb a neighbour's slab
                # area. Subtraction set depends on the widening target's
                # kind:
                # - committed_oblique: subtract OTHER committed_obliques
                #   only. Do NOT subtract ridge/eave groups: a committed
                #   face often sits inside a matched ridge/eave group's
                #   GROUP-level envelope (which spans both slopes) and
                #   subtracting it would cancel widening for every
                #   committed face inside such a group.
                # - ridge_eave_plane_group: subtract BOTH committed_obliques
                #   AND OTHER ridge/eave groups. Group envelope spans both
                #   slopes, so widening it without subtracting same-story
                #   committed faces would let it swallow their XZ extent
                #   and trigger downstream overlay suppression.
                if target.target_kind == "ridge_eave_plane_group":
                    sister_kinds: tuple[str, ...] = (
                        "committed_oblique",
                        "ridge_eave_plane_group",
                    )
                else:
                    sister_kinds = (target.target_kind,)
                sisters = [
                    sister_poly
                    for kind in sister_kinds
                    for sister_id, sister_poly in sibling_polys_by_key.get(
                        (target.uuid, target.story, kind), []
                    )
                    if sister_id != target.element_id
                ]
                if eave_add is not None and sisters:
                    try:
                        sister_union = unary_union(sisters)
                        if not getattr(sister_union, "is_empty", True):
                            eave_add = eave_add.difference(sister_union)
                    except Exception:
                        pass
                if eave_add is not None and not getattr(eave_add, "is_empty", True):
                    try:
                        widened = unary_union([base_poly, eave_add])
                    except Exception:
                        widened = base_poly
                    if extent_poly is not None:
                        try:
                            widened = widened.intersection(extent_poly)
                        except Exception:
                            pass
                    widened_area = float(getattr(widened, "area", 0.0) or 0.0)
                    if (
                        widened is not None
                        and not getattr(widened, "is_empty", True)
                        and widened_area >= pre_extension_base_area
                    ):
                        base_poly = widened
                        extended_by_m2 = widened_area - pre_extension_base_area
                        if extended_by_m2 > MIN_SPLIT_PIECE_AREA_M2:
                            target_extension_info = {
                                "extended_to_eave": True,
                                "extended_by_m2": round(extended_by_m2, 6),
                                "eave_envelope_parts": envelope_parts,
                                "eave_envelope_source": envelope_source,
                                "directional_trim": halfplane is not None,
                            }
        support_base_poly = base_poly
        anchor_support_geom = (target_segment_anchor_masks or {}).get(target.element_id)
        if (
            target.target_kind == "ridge_eave_plane_group"
            and anchor_support_geom is not None
            and not getattr(anchor_support_geom, "is_empty", True)
        ):
            try:
                support_base_poly = unary_union(
                    [
                        support_base_poly,
                        anchor_support_geom,
                    ]
                )
            except Exception:
                pass
            if extent_poly is not None:
                try:
                    support_base_poly = support_base_poly.intersection(extent_poly)
                except Exception:
                    pass
        if (
            getattr(support_base_poly, "is_empty", True)
            or float(getattr(support_base_poly, "area", 0.0) or 0.0)
            < MIN_SPLIT_PIECE_AREA_M2
        ):
            continue

        axis = _ridge_axis_unit(target)
        ortho = np.asarray([-float(axis[1]), float(axis[0])], dtype=float)
        base_ortho_bounds = _polygon_projection_bounds(support_base_poly, ortho)
        if base_ortho_bounds is None:
            continue
        support_map = {
            support.chain_id: support
            for support in supported
            if support.chain_id in chain_by_id
            and chain_by_id[support.chain_id].story == target.story
        }
        if not support_map:
            continue

        component_rows: list[dict[str, Any]] = []
        remaining_chain_ids = set(support_map.keys())
        connected_components: list[list[PlaneEaveChainSupportRecord]] = []
        while remaining_chain_ids:
            seed_chain_id = sorted(remaining_chain_ids)[0]
            component_chain_ids: set[str] = set()
            queue = [seed_chain_id]
            while queue:
                chain_id = queue.pop(0)
                if chain_id in component_chain_ids:
                    continue
                component_chain_ids.add(chain_id)
                for neighbor_id in chain_neighbors.get(chain_id, set()):
                    if (
                        neighbor_id in remaining_chain_ids
                        and neighbor_id not in component_chain_ids
                    ):
                        queue.append(neighbor_id)
            remaining_chain_ids.difference_update(component_chain_ids)
            connected_components.append(
                [support_map[chain_id] for chain_id in sorted(component_chain_ids)]
            )
        if (
            target.target_kind == "ridge_eave_plane_group"
            and anchor_support_geom is not None
            and not getattr(anchor_support_geom, "is_empty", True)
            and connected_components
        ):
            component_anchor_overlaps: list[float] = []
            for component_supports in connected_components:
                overlap_area = 0.0
                for support in component_supports:
                    chain = chain_by_id.get(support.chain_id)
                    if chain is None:
                        continue
                    try:
                        overlap_area += float(
                            chain.line_xz.buffer(PLANE_EAVE_CHAIN_BUFFER_M, cap_style=2)
                            .intersection(anchor_support_geom)
                            .area
                        )
                    except Exception:
                        continue
                component_anchor_overlaps.append(overlap_area)
            max_anchor_overlap = max(component_anchor_overlaps, default=0.0)
            if max_anchor_overlap > 0.0:
                connected_components = [
                    component_supports
                    for component_supports, overlap_area in zip(
                        connected_components, component_anchor_overlaps, strict=False
                    )
                    if overlap_area >= max_anchor_overlap - 1e-9
                ]

        supported_candidates: list[dict[str, Any]] = []
        for component_supports in connected_components:
            if target.target_kind == "ridge_eave_plane_group":
                sweep_polys: list[Polygon] = []
                component_chain_ids: set[str] = set()
                component_support_score = 0.0
                for support in sorted(
                    component_supports,
                    key=lambda row: (-row.support_score, row.chain_id),
                ):
                    chain = chain_by_id.get(support.chain_id)
                    if chain is None:
                        continue
                    sweep_poly = _chain_inward_sweep_polygon(
                        target, support_base_poly, chain, axis
                    )
                    if sweep_poly is None or sweep_poly.is_empty:
                        continue
                    sweep_polys.append(sweep_poly)
                    component_chain_ids.add(chain.chain_id)
                    component_support_score = max(
                        component_support_score, float(support.support_score)
                    )
                if not sweep_polys:
                    continue
                try:
                    merged_sweep = unary_union(sweep_polys)
                except Exception:
                    merged_sweep = sweep_polys[0]
                for poly in _iter_polygons(merged_sweep):
                    supported_candidates.append(
                        {
                            "poly": poly,
                            "support_score": component_support_score,
                            "chain_ids": tuple(sorted(component_chain_ids)),
                        }
                    )
                continue
            interval_rows: list[dict[str, Any]] = []
            ortho_values: list[float] = []
            for support in sorted(
                component_supports, key=lambda row: (-row.support_score, row.chain_id)
            ):
                chain = chain_by_id.get(support.chain_id)
                if chain is None:
                    continue
                start_t, end_t = _projection_interval(
                    axis, chain.start_xz, chain.end_xz
                )
                start_u, end_u = _projection_interval(
                    ortho, chain.start_xz, chain.end_xz
                )
                interval_rows.append(
                    {
                        "start_t": start_t,
                        "end_t": end_t,
                        "chain_ids": {chain.chain_id},
                        "support_score": float(support.support_score),
                    }
                )
                ortho_values.extend([start_u, end_u])
            if not interval_rows:
                continue
            interval_rows.sort(
                key=lambda row: (float(row["start_t"]), float(row["end_t"]))
            )
            component_intervals: list[dict[str, Any]] = []
            for row in interval_rows:
                if not component_intervals:
                    component_intervals.append(row)
                    continue
                prev = component_intervals[-1]
                if float(row["start_t"]) <= float(prev["end_t"]) + EAVE_CHAIN_GAP_M:
                    prev["end_t"] = max(float(prev["end_t"]), float(row["end_t"]))
                    prev["chain_ids"].update(row["chain_ids"])
                    prev["support_score"] = max(
                        float(prev["support_score"]), float(row["support_score"])
                    )
                else:
                    component_intervals.append(row)
            if not component_intervals or not ortho_values:
                continue
            component_rows.append(
                {
                    "start_t": min(
                        float(row["start_t"]) for row in component_intervals
                    ),
                    "end_t": max(float(row["end_t"]) for row in component_intervals),
                    "min_u": min(ortho_values),
                    "max_u": max(ortho_values),
                    "chain_ids": {
                        chain_id
                        for row in component_intervals
                        for chain_id in row["chain_ids"]
                    },
                    "support_score": max(
                        float(row["support_score"]) for row in component_intervals
                    ),
                }
            )

        if target.target_kind != "ridge_eave_plane_group" and not component_rows:
            continue

        if component_rows:
            component_rows.sort(
                key=lambda row: (
                    (float(row["min_u"]) + float(row["max_u"])) * 0.5,
                    float(row["start_t"]),
                )
            )
            component_bands: list[dict[str, Any]] = []
            for component_index, component in enumerate(component_rows):
                lower_u = base_ortho_bounds[0]
                upper_u = base_ortho_bounds[1]
                if component_index > 0:
                    prev = component_rows[component_index - 1]
                    lower_u = 0.5 * (float(prev["max_u"]) + float(component["min_u"]))
                if component_index + 1 < len(component_rows):
                    nxt = component_rows[component_index + 1]
                    upper_u = 0.5 * (float(component["max_u"]) + float(nxt["min_u"]))
                component_bands.append(
                    {
                        **component,
                        "band_min_u": lower_u,
                        "band_max_u": upper_u,
                    }
                )

            for _component_index, component in enumerate(component_bands):
                window = _axis_window_polygon(
                    support_base_poly,
                    axis,
                    ortho,
                    float(component["start_t"]),
                    float(component["end_t"]),
                    float(component["band_min_u"]),
                    float(component["band_max_u"]),
                )
                geom = support_base_poly.intersection(window)
                for poly in _iter_polygons(geom):
                    supported_candidates.append(
                        {
                            "poly": poly,
                            "support_score": float(component["support_score"]),
                            "chain_ids": tuple(sorted(component["chain_ids"])),
                        }
                    )

        if not supported_candidates:
            continue

        if target.element_id not in (target_segment_anchor_masks or {}):
            supported_candidates, _bridged_residuals = _merge_supported_gap_polygons(
                support_base_poly,
                supported_candidates,
                gap_polys=(story_gap_polygons or {}).get(target.story),
            )
        supported_polys = [candidate["poly"] for candidate in supported_candidates]
        piece_index = 0
        for component_index, candidate in enumerate(supported_candidates):
            record = _piece_records_from_polygon(
                target,
                candidate["poly"],
                piece_id=f"{target.element_id}#supported:{component_index}:0",
                piece_index=piece_index,
                piece_role="supported",
                support_score=float(candidate["support_score"]),
                chain_ids=tuple(candidate["chain_ids"]),
            )
            if record is None:
                continue
            pieces.append(record)
            if (
                extension_provenance_out is not None
                and target_extension_info is not None
            ):
                extension_provenance_out[record.piece_id] = dict(target_extension_info)
            piece_index += 1

        try:
            covered = unary_union(supported_polys)
            residual_geom = support_base_poly.difference(covered)
        except Exception:
            residual_geom = support_base_poly
        for component_index, poly in enumerate(_iter_polygons(residual_geom)):
            record = _piece_records_from_polygon(
                target,
                poly,
                piece_id=f"{target.element_id}#residual:{component_index}",
                piece_index=piece_index,
                piece_role="residual",
                support_score=None,
                chain_ids=tuple(),
            )
            if record is None:
                continue
            pieces.append(record)
            if (
                extension_provenance_out is not None
                and target_extension_info is not None
            ):
                extension_provenance_out[record.piece_id] = dict(target_extension_info)
            piece_index += 1
    return pieces


def trim_ridge_eave_supported_pieces_to_room_ownership(
    building: dict[str, Any],
    split_targets: list[TargetPlaneRecord],
    split_target_score_rows: list[dict[str, Any]],
    split_pieces: list[TargetSplitPieceRecord],
    *,
    building_part_graph: dict[str, Any] | None = None,
    room_buffer_m: float = RIDGE_EAVE_ROOM_OWNERSHIP_BUFFER_M,
) -> list[TargetSplitPieceRecord]:
    """Late-trim ridge/eave supported pieces to the rooms they locally win.

    Raw eave chains are anchors, not a hard geometric cap. The actual trim
    happens after split pieces exist and only against room regions that a
    competing supported piece explains better on the same story.
    """
    targets_by_id = {target.element_id: target for target in split_targets}
    target_scores_by_id = {
        str(row.get("element_id") or ""): row
        for row in split_target_score_rows
        if row.get("element_id")
    }
    room_part_membership = _room_part_membership(building_part_graph)
    part_region_unions = _story_part_region_unions(building, building_part_graph)
    all_story_part_unions = _part_region_unions(building, building_part_graph)
    room_regions_by_story: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for piece in split_pieces:
        room_regions_by_story.setdefault(
            piece.story, _story_room_regions(building, piece.story)
        )

    piece_records_by_story: dict[int, list[dict[str, Any]]] = defaultdict(list)
    allowed_part_union_by_piece_id: dict[str, Any] = {}
    for piece in split_pieces:
        if piece.piece_role != "supported":
            continue
        poly = _xz_polygon(piece.corners)
        if poly is None or poly.is_empty:
            continue
        target_score_row = target_scores_by_id.get(piece.target_element_id) or {}
        allowed_part_ids = {
            str(part_id)
            for part_id in (target_score_row.get("face_run_hypothesis_part_ids") or [])
            if part_id
        }
        if allowed_part_ids:
            part_polys = [
                part_region_unions.get((piece.story, part_id))
                for part_id in sorted(allowed_part_ids)
            ]
            part_polys = [
                part_poly
                for part_poly in part_polys
                if part_poly is not None and not getattr(part_poly, "is_empty", True)
            ]
            if not part_polys:
                # Face-run ownership can come from a higher/lower story than the
                # split piece. Fall back to the part footprint across all stories
                # so stacked extensions still trim to the owning mass in plan.
                part_polys = [
                    all_story_part_unions.get(part_id)
                    for part_id in sorted(allowed_part_ids)
                ]
                part_polys = [
                    part_poly
                    for part_poly in part_polys
                    if part_poly is not None
                    and not getattr(part_poly, "is_empty", True)
                ]
            if part_polys:
                try:
                    allowed_part_union_by_piece_id[piece.piece_id] = unary_union(
                        part_polys
                    )
                except Exception:
                    pass
        piece_records_by_story[piece.story].append(
            {
                "piece": piece,
                "poly": poly,
                "kind_priority": float(_split_piece_kind_priority(piece.target_kind)),
                "target_strength": _piece_numeric_value(
                    target_score_row, "retention_support_score", 0.0
                ),
                "support_strength": 0.0
                if piece.support_score is None
                else float(piece.support_score),
                "allowed_part_ids": allowed_part_ids,
            }
        )

    owned_regions_by_piece_id: dict[str, list[Polygon]] = defaultdict(list)
    for story, piece_records in piece_records_by_story.items():
        for room_region in room_regions_by_story.get(story, []):
            room_poly = room_region["poly_xz"]
            room_area = max(float(room_region["area_xz_m2"]), 1e-9)
            room_part_ids = room_part_membership.get(str(room_region["room_id"]), set())
            contenders: list[tuple[tuple[float, ...], str, Any]] = []
            for record in piece_records:
                allowed_part_ids = record["allowed_part_ids"]
                if allowed_part_ids and (
                    not room_part_ids or allowed_part_ids.isdisjoint(room_part_ids)
                ):
                    continue
                try:
                    region_poly = record["poly"].intersection(room_poly)
                except Exception:
                    continue
                if region_poly.is_empty:
                    continue
                region_area = float(region_poly.area)
                if region_area < MIN_SPLIT_PIECE_AREA_M2:
                    continue
                overlap_fraction = region_area / room_area
                local_strength = overlap_fraction * max(
                    record["target_strength"], record["support_strength"], 0.0
                )
                rank = (
                    record["kind_priority"],
                    local_strength,
                    overlap_fraction,
                    record["target_strength"],
                    record["support_strength"],
                )
                contenders.append((rank, str(record["piece"].piece_id), region_poly))
            if not contenders:
                continue
            contenders.sort(key=lambda item: item[0], reverse=True)
            owned_regions_by_piece_id[contenders[0][1]].extend(
                _iter_polygons(contenders[0][2])
            )

    trimmed: list[TargetSplitPieceRecord] = []
    for piece in split_pieces:
        if (
            piece.piece_role != "supported"
            or piece.target_kind != "ridge_eave_plane_group"
        ):
            trimmed.append(piece)
            continue
        owned_regions = owned_regions_by_piece_id.get(piece.piece_id) or []
        allowed_part_union = allowed_part_union_by_piece_id.get(piece.piece_id)
        if not owned_regions and (
            allowed_part_union is None or getattr(allowed_part_union, "is_empty", True)
        ):
            trimmed.append(piece)
            continue
        piece_poly = _xz_polygon(piece.corners)
        target = targets_by_id.get(piece.target_element_id)
        if piece_poly is None or piece_poly.is_empty or target is None:
            trimmed.append(piece)
            continue
        trim_mask = None
        if owned_regions:
            try:
                owned_union = unary_union(owned_regions)
            except Exception:
                trimmed.append(piece)
                continue
            if not owned_union.is_empty:
                try:
                    trim_mask = owned_union.buffer(max(float(room_buffer_m), 0.0))
                except Exception:
                    trimmed.append(piece)
                    continue
        if allowed_part_union is not None and not getattr(
            allowed_part_union, "is_empty", True
        ):
            try:
                allowed_mask = allowed_part_union
            except Exception:
                trimmed.append(piece)
                continue
            trim_mask = (
                allowed_mask
                if trim_mask is None
                else trim_mask.intersection(allowed_mask)
            )
        if trim_mask is None or getattr(trim_mask, "is_empty", True):
            continue
        try:
            clipped = piece_poly.intersection(trim_mask)
        except Exception:
            trimmed.append(piece)
            continue
        clipped_area = float(getattr(clipped, "area", 0.0) or 0.0)
        original_area = max(float(piece_poly.area), 1e-9)
        if clipped.is_empty or clipped_area < MIN_SPLIT_PIECE_AREA_M2:
            continue
        if clipped_area >= 0.98 * original_area:
            trimmed.append(piece)
            continue
        clipped_polys = _iter_polygons(clipped)
        if not clipped_polys:
            trimmed.append(piece)
            continue
        clipped_polys = sorted(
            clipped_polys, key=lambda poly: float(poly.area), reverse=True
        )[:1]
        for poly_index, poly in enumerate(clipped_polys):
            record = _piece_records_from_polygon(
                target,
                poly,
                piece_id=piece.piece_id
                if poly_index == 0
                else f"{piece.piece_id}:{poly_index}",
                piece_index=piece.piece_index,
                piece_role=piece.piece_role,
                support_score=piece.support_score,
                chain_ids=piece.chain_ids,
            )
            if record is not None:
                trimmed.append(record)

    return trimmed


def trim_ridge_eave_supported_pieces_to_mirror_partner(
    uuid: str,
    ridge_eave_entry: dict[str, Any] | None,
    split_targets: list[TargetPlaneRecord],
    split_pieces: list[TargetSplitPieceRecord],
    *,
    pad_m: float = 1.0,
) -> list[TargetSplitPieceRecord]:
    """Late-trim ridge/eave supported pieces at the mirror-partner seam.

    The broad support window is still allowed to extend beyond direct eave
    support. Once the supported piece exists, clip it by the equal-height seam
    against the target's selected mirror partner when one is available.
    """
    targets_by_id = {target.element_id: target for target in split_targets}
    target_by_plane_group_id = {
        plane_group_id: target
        for target in split_targets
        for plane_group_id in [_ridge_eave_target_to_plane_group_id(target.element_id)]
        if plane_group_id is not None
    }
    meta_by_target = _ridge_eave_plane_group_meta_by_target(uuid, ridge_eave_entry)

    trimmed: list[TargetSplitPieceRecord] = []
    for piece in split_pieces:
        if (
            piece.piece_role != "supported"
            or piece.target_kind != "ridge_eave_plane_group"
        ):
            trimmed.append(piece)
            continue
        target = targets_by_id.get(piece.target_element_id)
        if target is None:
            trimmed.append(piece)
            continue
        plane_group_id = _ridge_eave_target_to_plane_group_id(piece.target_element_id)
        if not plane_group_id:
            trimmed.append(piece)
            continue
        partner_plane_group_id = str(
            meta_by_target.get(piece.target_element_id, {}).get(
                "best_partner_plane_group_id"
            )
            or ""
        )
        partner_target = target_by_plane_group_id.get(partner_plane_group_id)
        piece_poly = _xz_polygon(piece.corners)
        if piece_poly is None or piece_poly.is_empty:
            trimmed.append(piece)
            continue
        if partner_target is None:
            trimmed.append(piece)
            continue
        coeffs = _mirror_half_plane_coeffs(target, partner_target)
        if coeffs is None:
            trimmed.append(piece)
            continue
        try:
            bounds = unary_union(
                [
                    piece_poly,
                    target.poly_xz,
                    partner_target.poly_xz,
                ]
            ).bounds
        except Exception:
            bounds = piece_poly.bounds
        keep_poly = _half_plane_polygon_for_bounds(coeffs, bounds, pad_m=pad_m)
        if keep_poly is None:
            trimmed.append(piece)
            continue
        try:
            clipped = piece_poly.intersection(keep_poly)
        except Exception:
            trimmed.append(piece)
            continue
        clipped_area = float(getattr(clipped, "area", 0.0) or 0.0)
        original_area = max(float(piece_poly.area), 1e-9)
        if clipped.is_empty or clipped_area < MIN_SPLIT_PIECE_AREA_M2:
            trimmed.append(piece)
            continue
        if clipped_area >= 0.995 * original_area:
            trimmed.append(piece)
            continue
        clipped_polys = _iter_polygons(clipped)
        if not clipped_polys:
            trimmed.append(piece)
            continue
        for poly_index, poly in enumerate(clipped_polys):
            record = _piece_records_from_polygon(
                target,
                poly,
                piece_id=piece.piece_id
                if poly_index == 0
                else f"{piece.piece_id}:{poly_index}",
                piece_index=piece.piece_index,
                piece_role=piece.piece_role,
                support_score=piece.support_score,
                chain_ids=piece.chain_ids,
            )
            if record is not None:
                trimmed.append(record)

    return trimmed


def _polygon_projection_span(poly: Any, axis: np.ndarray) -> float:
    if poly is None or getattr(poly, "is_empty", True):
        return 0.0
    values: list[float] = []

    def _append_ring(coords: Any) -> None:
        for x, z in list(coords):
            values.append(
                float(np.dot(axis, np.asarray([float(x), float(z)], dtype=float)))
            )

    if isinstance(poly, Polygon):
        _append_ring(poly.exterior.coords)
        for ring in poly.interiors:
            _append_ring(ring.coords)
    else:
        for geom in getattr(poly, "geoms", []):
            span = _polygon_projection_span(geom, axis)
            if span <= 0.0:
                continue
            if isinstance(geom, Polygon):
                _append_ring(geom.exterior.coords)
                for ring in geom.interiors:
                    _append_ring(ring.coords)
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _story_room_regions(building: dict[str, Any], story: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for room_index, room in enumerate(building.get("rooms") or []):
        if int(room.get("story", 0)) != int(story):
            continue
        poly = _xz_polygon(room.get("floor_polygon") or [])
        if poly is None or poly.is_empty:
            continue
        room_id = room.get("id") or f"room:{room_index}"
        regions.append(
            {
                "room_id": str(room_id),
                "room_index": room_index,
                "poly_xz": poly,
                "area_xz_m2": float(poly.area),
            }
        )
    return regions


def _room_part_membership(
    building_part_graph: dict[str, Any] | None,
) -> dict[str, set[str]]:
    if not building_part_graph:
        return {}
    return {
        str(room_id): {str(part_id) for part_id in (part_ids or []) if part_id}
        for room_id, part_ids in (
            building_part_graph.get("room_membership") or {}
        ).items()
    }


def _story_part_region_unions(
    building: dict[str, Any],
    building_part_graph: dict[str, Any] | None,
) -> dict[tuple[int, str], Polygon]:
    room_membership = _room_part_membership(building_part_graph)
    polygons_by_story_part: dict[tuple[int, str], list[Polygon]] = defaultdict(list)
    for room_index, room in enumerate(building.get("rooms") or []):
        poly = _xz_polygon(room.get("floor_polygon") or [])
        if poly is None or poly.is_empty:
            continue
        room_id = str(room.get("id") or f"room:{room_index}")
        story = int(room.get("story", 0))
        for part_id in room_membership.get(room_id, set()):
            polygons_by_story_part[(story, part_id)].append(poly)

    unions: dict[tuple[int, str], Polygon] = {}
    for key, polys in polygons_by_story_part.items():
        try:
            union = unary_union(polys)
        except Exception:
            continue
        if union is None or getattr(union, "is_empty", True):
            continue
        unions[key] = union
    return unions


def _part_region_unions(
    building: dict[str, Any],
    building_part_graph: dict[str, Any] | None,
) -> dict[str, Polygon]:
    room_membership = _room_part_membership(building_part_graph)
    polygons_by_part: dict[str, list[Polygon]] = defaultdict(list)
    for room_index, room in enumerate(building.get("rooms") or []):
        poly = _xz_polygon(room.get("floor_polygon") or [])
        if poly is None or poly.is_empty:
            continue
        room_id = str(room.get("id") or f"room:{room_index}")
        for part_id in room_membership.get(room_id, set()):
            polygons_by_part[str(part_id)].append(poly)

    unions: dict[str, Polygon] = {}
    for part_id, polys in polygons_by_part.items():
        try:
            union = unary_union(polys)
        except Exception:
            continue
        if union is None or getattr(union, "is_empty", True):
            continue
        unions[part_id] = union
    return unions


# Gap polygons whose nearest slab is within this distance are considered
# "neighbouring" the part and absorbed into its eave envelope. Small enough
# to exclude gaps that sit between two disjoint parts; large enough to tolerate
# the sub-centimetre slivers that appear where a slab and a gap quad abut.
_PART_EAVE_GAP_NEIGHBOUR_TOL_M = 0.05


def _story_eave_envelopes(building: dict[str, Any]) -> dict[int, Any]:
    """``story → XZ polygon`` of the story's eave envelope.

    Story-level mirror of :func:`_part_eave_envelopes` for buildings that
    lack a building-part graph or whose targets have no ``allowed_part_ids``.
    Envelope = union of all room floor polygons on that story plus any
    cross-floor gap (or gap-ceiling quad) that abuts the story's slab union
    within ``_PART_EAVE_GAP_NEIGHBOUR_TOL_M``.

    Used as a fallback when the per-part lookup yields nothing — strictly
    larger than any single part's envelope, so the caller still relies on
    sister-target subtraction to keep widening from absorbing area belonging
    to another roof face.
    """

    slabs_by_story: dict[int, list[Polygon]] = defaultdict(list)
    for room in building.get("rooms") or []:
        poly = _xz_polygon(room.get("floor_polygon") or [])
        if poly is None or poly.is_empty:
            continue
        slabs_by_story[int(room.get("story", 0))].append(poly)

    gap_polys: list[Polygon] = []
    for gap in building.get("cross_floor_gaps") or []:
        poly = _xz_polygon(gap.get("corners") or [])
        if poly is not None and not poly.is_empty:
            gap_polys.append(poly)
    for gap_wall in building.get("gap_walls") or []:
        if str(gap_wall.get("type") or "") != "gap_ceiling":
            continue
        poly = _xz_polygon(gap_wall.get("corners") or [])
        if poly is not None and not poly.is_empty:
            gap_polys.append(poly)

    envelopes: dict[int, Any] = {}
    for story, polys in slabs_by_story.items():
        try:
            slab_union = unary_union(polys)
        except Exception:
            continue
        if slab_union is None or getattr(slab_union, "is_empty", True):
            continue
        neighbouring: list[Polygon] = []
        for gap_poly in gap_polys:
            try:
                if (
                    slab_union.intersects(gap_poly)
                    or slab_union.distance(gap_poly) <= _PART_EAVE_GAP_NEIGHBOUR_TOL_M
                ):
                    neighbouring.append(gap_poly)
            except Exception:
                continue
        if neighbouring:
            try:
                envelope = unary_union([slab_union, *neighbouring])
            except Exception:
                envelope = slab_union
        else:
            envelope = slab_union
        if envelope is None or getattr(envelope, "is_empty", True):
            continue
        envelopes[int(story)] = envelope
    return envelopes


def _part_eave_envelopes(
    building: dict[str, Any],
    building_part_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    """``part_id → XZ polygon`` of the part's eave envelope.

    Envelope = union of the part's room floor polygons plus any closed
    cross-floor gap (or gap-ceiling quad) that abuts one of those slabs.
    The proximity gate keeps a gap that straddles two parts from being
    absorbed into both — it only joins the part whose slab it actually
    touches within ``_PART_EAVE_GAP_NEIGHBOUR_TOL_M``.
    """

    room_membership = _room_part_membership(building_part_graph)
    slabs_by_part: dict[str, list[Polygon]] = defaultdict(list)
    for room_index, room in enumerate(building.get("rooms") or []):
        poly = _xz_polygon(room.get("floor_polygon") or [])
        if poly is None or poly.is_empty:
            continue
        room_id = str(room.get("id") or f"room:{room_index}")
        for part_id in room_membership.get(room_id, set()):
            slabs_by_part[str(part_id)].append(poly)

    gap_polys: list[Polygon] = []
    for gap in building.get("cross_floor_gaps") or []:
        poly = _xz_polygon(gap.get("corners") or [])
        if poly is not None and not poly.is_empty:
            gap_polys.append(poly)
    for gap_wall in building.get("gap_walls") or []:
        if str(gap_wall.get("type") or "") != "gap_ceiling":
            continue
        poly = _xz_polygon(gap_wall.get("corners") or [])
        if poly is not None and not poly.is_empty:
            gap_polys.append(poly)

    envelopes: dict[str, Any] = {}
    for part_id, polys in slabs_by_part.items():
        try:
            slab_union = unary_union(polys)
        except Exception:
            continue
        if slab_union is None or getattr(slab_union, "is_empty", True):
            continue
        neighbouring: list[Polygon] = []
        for gap_poly in gap_polys:
            try:
                if (
                    slab_union.intersects(gap_poly)
                    or slab_union.distance(gap_poly) <= _PART_EAVE_GAP_NEIGHBOUR_TOL_M
                ):
                    neighbouring.append(gap_poly)
            except Exception:
                continue
        if neighbouring:
            try:
                envelope = unary_union([slab_union, *neighbouring])
            except Exception:
                envelope = slab_union
        else:
            envelope = slab_union
        if envelope is None or getattr(envelope, "is_empty", True):
            continue
        envelopes[part_id] = envelope
    return envelopes


def _downslope_halfplane_polygon(
    target: TargetPlaneRecord,
    reference_poly: Polygon,
) -> Polygon | None:
    """Large XZ rectangle covering the down-slope side of ``reference_poly``.

    Anchored at the up-slope (ridge) edge of ``reference_poly`` projected
    onto the down-slope direction — everything ridgeward is excluded, so
    widening cannot cross the ridge into the opposing roof face. The
    caller further subtracts every sister committed_oblique's fitted XZ
    before unioning with ``reference_poly``, which keeps the extension
    from reaching into parallel roof faces on the same slab. Returns
    ``None`` for near-flat targets (caller then skips directional trim).
    """

    axis = _ridge_axis_unit(target)
    uphill = _target_uphill_sweep_direction(target, axis)
    if uphill is None:
        return None
    downhill = -uphill
    bounds = _polygon_projection_bounds(reference_poly, downhill)
    if bounds is None:
        return None
    u_ridge = float(bounds[0])  # min downhill projection = ridge-side edge
    minx, minz, maxx, maxz = reference_poly.bounds
    span = math.hypot(maxx - minx, maxz - minz) * 10.0 + 10.0
    centroid = reference_poly.centroid
    axis_center = float(
        np.dot(axis, np.asarray([float(centroid.x), float(centroid.y)], dtype=float))
    )
    corners = [
        axis * (axis_center - span) + downhill * (u_ridge - 0.01),
        axis * (axis_center + span) + downhill * (u_ridge - 0.01),
        axis * (axis_center + span) + downhill * (u_ridge + span),
        axis * (axis_center - span) + downhill * (u_ridge + span),
    ]
    poly = Polygon([(float(pt[0]), float(pt[1])) for pt in corners])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly


def _ridge_eave_plane_group_meta_by_target(
    uuid: str,
    ridge_eave_entry: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not ridge_eave_entry:
        return {}
    meta: dict[str, dict[str, Any]] = {}
    for plane_group in ridge_eave_entry.get("plane_groups") or []:
        plane_group_suffix = str(plane_group.get("id", "")).split("::")[-1]
        if not plane_group_suffix:
            continue
        target_element_id = (
            f"{uuid}::ridge-eave-candidate::plane-group::{plane_group_suffix}"
        )
        meta[target_element_id] = {
            "best_partner_plane_group_id": plane_group.get(
                "best_partner_plane_group_id"
            ),
            "best_score": plane_group.get("best_score"),
            "creator_eave_proximity": plane_group.get("creator_eave_proximity"),
        }
    return meta


def diagnose_ridge_eave_supported_piece_ownership(
    building: dict[str, Any],
    piece: TargetSplitPieceRecord,
    *,
    targets_by_id: dict[str, TargetPlaneRecord],
    target_scores_by_id: dict[str, dict[str, Any]],
    supported_chain_by_target: dict[str, PlaneEaveChainSupportRecord],
    supported_ridge_eave_pieces_by_signature: dict[
        tuple[int, tuple[str, ...]], list[TargetSplitPieceRecord]
    ],
    ridge_eave_meta_by_target: dict[str, dict[str, Any]],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if piece.target_kind != "ridge_eave_plane_group" or piece.piece_role != "supported":
        return None
    target = targets_by_id.get(piece.target_element_id)
    if target is None:
        return None
    subject_score_row = target_scores_by_id.get(piece.target_element_id)
    if subject_score_row is None:
        return None
    piece_poly = _xz_polygon(piece.corners)
    if piece_poly is None or piece_poly.is_empty:
        return None

    ridge_axis = _ridge_axis_unit(target)
    slope_axis = np.asarray([-float(ridge_axis[1]), float(ridge_axis[0])], dtype=float)
    room_regions = _story_room_regions(building, piece.story)
    occupied_intersections: list[Polygon] = []
    crossed_room_ids: list[str] = []
    crossed_room_area_m2 = 0.0
    competitor_loss_area_m2 = 0.0
    competitor_loss_room_count = 0
    competitor_counter: Counter[str] = Counter()
    competitor_piece_counter: Counter[str] = Counter()
    subject_quality = float(
        piece.support_score
        if piece.support_score is not None
        else subject_score_row.get("retention_support_score") or 0.0
    )
    piece_signature = _chain_signature(piece.chain_ids)
    signature_class = supported_ridge_eave_pieces_by_signature.get(
        (piece.story, piece_signature), []
    )
    local_signature_competitors = [
        other_piece
        for other_piece in signature_class
        if (
            other_piece.piece_id != piece.piece_id
            and other_piece.target_element_id != piece.target_element_id
            and (other_piece.target_element_id in targets_by_id)
            and _targets_are_local_ownership_competitors(
                target, targets_by_id[other_piece.target_element_id]
            )
        )
    ]

    story_targets = [
        other_target
        for other_target in targets_by_id.values()
        if (
            other_target.story == piece.story
            and other_target.element_id != piece.target_element_id
            and other_target.target_kind != "ridge_eave_plane_group"
            and _targets_are_local_ownership_competitors(target, other_target)
        )
    ]
    for room_region in room_regions:
        try:
            region_poly = piece_poly.intersection(room_region["poly_xz"])
        except Exception:
            continue
        if region_poly.is_empty:
            continue
        region_area = float(region_poly.area)
        if region_area < MIN_SPLIT_PIECE_AREA_M2:
            continue
        crossed_room_ids.append(str(room_region["room_id"]))
        crossed_room_area_m2 += region_area
        for poly in _iter_polygons(region_poly):
            occupied_intersections.append(poly)

        subject_local_score = subject_quality
        best_competitor_id = None
        best_competitor_piece_id = None
        best_competitor_score = subject_local_score
        for other_target in story_targets:
            other_score_row = target_scores_by_id.get(other_target.element_id)
            if other_score_row is None:
                continue
            try:
                overlap_area = float(
                    region_poly.intersection(other_target.poly_xz).area
                )
            except Exception:
                overlap_area = 0.0
            if overlap_area < MIN_SPLIT_PIECE_AREA_M2:
                continue
            other_quality = float(other_score_row.get("retention_support_score") or 0.0)
            competitor_score = (overlap_area / max(region_area, 1e-9)) * other_quality
            if competitor_score > best_competitor_score + 1e-6:
                best_competitor_score = competitor_score
                best_competitor_id = other_target.element_id
                best_competitor_piece_id = None
        for other_piece in local_signature_competitors:
            other_target = targets_by_id.get(other_piece.target_element_id)
            if other_target is None:
                continue
            other_poly = _xz_polygon(other_piece.corners)
            if other_poly is None or other_poly.is_empty:
                continue
            try:
                overlap_area = float(region_poly.intersection(other_poly).area)
            except Exception:
                overlap_area = 0.0
            if overlap_area < MIN_SPLIT_PIECE_AREA_M2:
                continue
            other_quality = float(
                other_piece.support_score
                if other_piece.support_score is not None
                else (
                    target_scores_by_id.get(other_target.element_id, {}).get(
                        "retention_support_score"
                    )
                    or 0.0
                )
            )
            competitor_score = (overlap_area / max(region_area, 1e-9)) * other_quality
            if competitor_score > best_competitor_score + 1e-6:
                best_competitor_score = competitor_score
                best_competitor_id = other_target.element_id
                best_competitor_piece_id = other_piece.piece_id
        if best_competitor_id is not None:
            competitor_loss_room_count += 1
            competitor_loss_area_m2 += region_area
            competitor_counter[best_competitor_id] += 1
            if best_competitor_piece_id is not None:
                competitor_piece_counter[best_competitor_piece_id] += 1

    occupied_piece = (
        unary_union(occupied_intersections) if occupied_intersections else piece_poly
    )
    if occupied_piece.is_empty:
        occupied_piece = piece_poly
    along_span_m = _polygon_projection_span(occupied_piece, ridge_axis)
    across_span_m = _polygon_projection_span(occupied_piece, slope_axis)
    through_ratio = across_span_m / max(along_span_m, 1e-9)

    chain_support = supported_chain_by_target.get(piece.target_element_id)
    ridge_meta = ridge_eave_meta_by_target.get(piece.target_element_id, {})
    target_diag = ridge_eave_target_diagnostics.get(piece.target_element_id, {})
    competition_class_target_ids = sorted(
        {other_piece.target_element_id for other_piece in signature_class}
    )
    return {
        "uuid": piece.uuid,
        "story": piece.story,
        "target_element_id": piece.target_element_id,
        "piece_id": piece.piece_id,
        "piece_area_xz_m2": round(piece.area_xz_m2, 6),
        "crossed_room_count": len(crossed_room_ids),
        "crossed_room_ids": crossed_room_ids,
        "crossed_room_area_m2": round(crossed_room_area_m2, 6),
        "along_span_m": round(along_span_m, 6),
        "across_span_m": round(across_span_m, 6),
        "through_ratio": round(through_ratio, 6),
        "local_competitor_loss_room_count": competitor_loss_room_count,
        "local_competitor_loss_area_m2": round(competitor_loss_area_m2, 6),
        "local_competitor_loss_fraction": round(
            competitor_loss_area_m2 / max(crossed_room_area_m2, 1e-9),
            6,
        )
        if crossed_room_area_m2 > 1e-9
        else 0.0,
        "local_top_competitor_ids": [
            competitor_id for competitor_id, _count in competitor_counter.most_common(3)
        ],
        "local_top_competitor_piece_ids": [
            competitor_piece_id
            for competitor_piece_id, _count in competitor_piece_counter.most_common(3)
        ],
        "chain_signature": list(piece_signature),
        "chain_signature_id": _chain_signature_id(
            piece.uuid, piece.story, piece.chain_ids
        ),
        "chain_signature_size": len(piece_signature),
        "local_signature_competitor_piece_count": len(local_signature_competitors),
        "local_signature_competitor_target_ids": competition_class_target_ids,
        "mirror_partner_plane_group_id": ridge_meta.get("best_partner_plane_group_id"),
        "mirror_support_score": round(float(ridge_meta["best_score"]), 6)
        if ridge_meta.get("best_score") is not None
        else None,
        "creator_eave_proximity": round(float(ridge_meta["creator_eave_proximity"]), 6)
        if ridge_meta.get("creator_eave_proximity") is not None
        else None,
        "best_supported_chain_id": chain_support.chain_id
        if chain_support is not None
        else None,
        "best_supported_chain_score": round(float(chain_support.support_score), 6)
        if chain_support is not None
        else None,
        "best_supported_chain_height_residual_m": round(
            float(chain_support.height_residual_m), 6
        )
        if chain_support is not None and chain_support.height_residual_m is not None
        else None,
        "creator_source_room_ids": list(
            target_diag.get("creator_source_room_ids") or []
        ),
        "creator_touch_room_ids": list(target_diag.get("creator_touch_room_ids") or []),
        "creator_source_room_count": target_diag.get("creator_source_room_count"),
        "creator_touch_room_count": target_diag.get("creator_touch_room_count"),
    }


def diagnose_ridge_eave_piece_ownership(
    building: dict[str, Any],
    ridge_eave_entry: dict[str, Any] | None,
    split_targets: list[TargetPlaneRecord],
    split_target_score_rows: list[dict[str, Any]],
    split_pieces: list[TargetSplitPieceRecord],
    plane_chain_supports: list[PlaneEaveChainSupportRecord],
    ridge_eave_target_diagnostics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    targets_by_id = {target.element_id: target for target in split_targets}
    target_scores_by_id = {
        str(row["element_id"]): row for row in split_target_score_rows
    }
    supported_chain_by_target: dict[str, PlaneEaveChainSupportRecord] = {}
    for support in plane_chain_supports:
        if not support.supported:
            continue
        prev = supported_chain_by_target.get(support.target_element_id)
        if prev is None or float(support.support_score) > float(prev.support_score):
            supported_chain_by_target[support.target_element_id] = support
    supported_ridge_eave_pieces_by_signature: dict[
        tuple[int, tuple[str, ...]], list[TargetSplitPieceRecord]
    ] = defaultdict(list)
    for piece in split_pieces:
        if (
            piece.target_kind != "ridge_eave_plane_group"
            or piece.piece_role != "supported"
        ):
            continue
        signature = _chain_signature(piece.chain_ids)
        if not signature:
            continue
        supported_ridge_eave_pieces_by_signature[(piece.story, signature)].append(piece)
    ridge_eave_meta_by_target = _ridge_eave_plane_group_meta_by_target(
        str(building["uuid"]), ridge_eave_entry
    )
    rows: list[dict[str, Any]] = []
    for piece in split_pieces:
        row = diagnose_ridge_eave_supported_piece_ownership(
            building,
            piece,
            targets_by_id=targets_by_id,
            target_scores_by_id=target_scores_by_id,
            supported_chain_by_target=supported_chain_by_target,
            supported_ridge_eave_pieces_by_signature=supported_ridge_eave_pieces_by_signature,
            ridge_eave_meta_by_target=ridge_eave_meta_by_target,
            ridge_eave_target_diagnostics=ridge_eave_target_diagnostics,
        )
        if row is not None:
            rows.append(row)
    return rows


def merge_split_piece_rows_with_ownership(
    split_piece_rows: list[dict[str, Any]],
    ownership_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ownership_by_piece_id = {
        str(row["piece_id"]): row for row in ownership_rows if row.get("piece_id")
    }
    merged_rows: list[dict[str, Any]] = []
    for row in split_piece_rows:
        ownership = ownership_by_piece_id.get(str(row.get("piece_id")))
        if ownership is None:
            merged_rows.append(dict(row))
            continue
        merged = dict(row)
        merged.update(ownership)
        merged_rows.append(merged)
    return merged_rows


def annotate_ridge_eave_rows_with_creator_source_overlap(
    split_piece_rows: list[dict[str, Any]],
    buildings_by_uuid: dict[str, dict[str, Any]],
    roof_results_by_uuid: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    room_polys_by_uuid: dict[str, dict[str, Polygon]] = {}
    room_membership_by_uuid: dict[str, dict[str, set[str]]] = {}
    part_unions_by_uuid: dict[str, dict[str, Polygon]] = {}

    for uuid, building in buildings_by_uuid.items():
        room_polys: dict[str, Polygon] = {}
        for room_index, room in enumerate(building.get("rooms") or []):
            poly = _xz_polygon(room.get("floor_polygon") or [])
            if poly is None or poly.is_empty:
                continue
            room_polys[str(room.get("id") or f"room:{room_index}")] = poly
        room_polys_by_uuid[uuid] = room_polys

        building_part_graph = (roof_results_by_uuid.get(uuid) or {}).get(
            "building_part_graph"
        ) or {}
        room_membership_by_uuid[uuid] = _room_part_membership(building_part_graph)
        part_unions_by_uuid[uuid] = _part_region_unions(building, building_part_graph)

    creator_union_cache: dict[
        tuple[str, tuple[str, ...]], tuple[Any | None, Any | None, tuple[str, ...]]
    ] = {}
    annotated_rows: list[dict[str, Any]] = []
    for row in split_piece_rows:
        if str(row.get("target_kind") or "") != "ridge_eave_plane_group":
            annotated_rows.append(dict(row))
            continue

        out = dict(row)
        uuid = str(row.get("uuid") or "")
        creator_source_room_ids = tuple(
            sorted(
                str(room_id)
                for room_id in (row.get("creator_source_room_ids") or [])
                if room_id
            )
        )
        piece_poly = _row_piece_polygon(row)
        if (
            not uuid
            or not creator_source_room_ids
            or piece_poly is None
            or piece_poly.is_empty
        ):
            out.setdefault("creator_source_part_ids", [])
            out.setdefault("creator_source_room_overlap_area_m2", 0.0)
            out.setdefault("creator_source_room_overlap_fraction", 0.0)
            out.setdefault("creator_source_part_overlap_area_m2", 0.0)
            out.setdefault("creator_source_part_overlap_fraction", 0.0)
            annotated_rows.append(out)
            continue

        cache_key = (uuid, creator_source_room_ids)
        cached = creator_union_cache.get(cache_key)
        if cached is None:
            room_polys = room_polys_by_uuid.get(uuid) or {}
            room_membership = room_membership_by_uuid.get(uuid) or {}
            part_unions = part_unions_by_uuid.get(uuid) or {}
            creator_room_polys = [
                room_polys.get(room_id) for room_id in creator_source_room_ids
            ]
            creator_room_polys = [
                poly
                for poly in creator_room_polys
                if poly is not None and not getattr(poly, "is_empty", True)
            ]
            creator_room_union = None
            if creator_room_polys:
                try:
                    creator_room_union = unary_union(creator_room_polys)
                except Exception:
                    creator_room_union = None
            creator_part_ids = tuple(
                sorted(
                    {
                        part_id
                        for room_id in creator_source_room_ids
                        for part_id in room_membership.get(room_id, set())
                    }
                )
            )
            creator_part_polys = [
                part_unions.get(part_id) for part_id in creator_part_ids
            ]
            creator_part_polys = [
                poly
                for poly in creator_part_polys
                if poly is not None and not getattr(poly, "is_empty", True)
            ]
            creator_part_union = None
            if creator_part_polys:
                try:
                    creator_part_union = unary_union(creator_part_polys)
                except Exception:
                    creator_part_union = None
            cached = (creator_room_union, creator_part_union, creator_part_ids)
            creator_union_cache[cache_key] = cached

        creator_room_union, creator_part_union, creator_part_ids = cached
        room_overlap_area = 0.0
        if creator_room_union is not None and not getattr(
            creator_room_union, "is_empty", True
        ):
            try:
                room_overlap_area = float(
                    piece_poly.intersection(creator_room_union).area
                )
            except Exception:
                room_overlap_area = 0.0
        part_overlap_area = 0.0
        if creator_part_union is not None and not getattr(
            creator_part_union, "is_empty", True
        ):
            try:
                part_overlap_area = float(
                    piece_poly.intersection(creator_part_union).area
                )
            except Exception:
                part_overlap_area = 0.0
        piece_area = max(float(piece_poly.area), 1e-9)
        out["creator_source_part_ids"] = list(creator_part_ids)
        out["creator_source_room_overlap_area_m2"] = round(room_overlap_area, 6)
        out["creator_source_room_overlap_fraction"] = round(
            room_overlap_area / piece_area, 6
        )
        out["creator_source_part_overlap_area_m2"] = round(part_overlap_area, 6)
        out["creator_source_part_overlap_fraction"] = round(
            part_overlap_area / piece_area, 6
        )
        annotated_rows.append(out)
    return annotated_rows


def _row_piece_polygon(row: dict[str, Any]) -> Polygon | None:
    corners = row.get("corners") or []
    outer = [(float(c[0]), float(c[2])) for c in corners if len(c) >= 3]
    if len(outer) < 3:
        return None
    holes: list[list[tuple[float, float]]] = []
    for hole in row.get("holes") or []:
        coords = [(float(c[0]), float(c[2])) for c in hole if len(c) >= 3]
        if len(coords) >= 3:
            holes.append(coords)
    poly = Polygon(outer, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _row_plane_model(
    row: dict[str, Any],
) -> tuple[tuple[float, float, float], np.ndarray] | tuple[None, None]:
    centroid, normal = _fit_plane_svd(list(row.get("corners") or []))
    if centroid is None or normal is None:
        return None, None
    return (float(centroid[0]), float(centroid[1]), float(centroid[2])), normal


def _row_plane_y_at(
    plane_model: tuple[tuple[float, float, float], np.ndarray] | tuple[None, None],
    x: float,
    z: float,
) -> float | None:
    plane_point, normal = plane_model
    if plane_point is None or normal is None:
        return None
    roof_up = _normalize_roof_up(normal)
    if abs(float(roof_up[1])) < 1e-6:
        return float(plane_point[1])
    px, py, pz = plane_point
    nx, ny, nz = float(roof_up[0]), float(roof_up[1]), float(roof_up[2])
    return float(py - (nx * (x - px) + nz * (z - pz)) / ny)


def _ring_to_row_plane_corners(
    row: dict[str, Any],
    plane_model: tuple[tuple[float, float, float], np.ndarray] | tuple[None, None],
    coords: list[tuple[float, float]],
) -> list[list[float]]:
    corners: list[list[float]] = []
    for x, z in coords:
        y = _row_plane_y_at(plane_model, float(x), float(z))
        if y is None:
            source_corner = list((row.get("corners") or [[0.0, 0.0, 0.0]])[0])
            y = float(source_corner[1]) if len(source_corner) >= 2 else 0.0
        corners.append([float(x), float(y), float(z)])
    return corners


def _row_piece_records_from_polygon(
    row: dict[str, Any],
    poly: Polygon,
    *,
    piece_id: str,
) -> dict[str, Any] | None:
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if not isinstance(poly, Polygon):
        polys = _iter_polygons(poly)
        if not polys:
            return None
        poly = max(polys, key=lambda item: float(item.area))
    if poly.is_empty or float(poly.area) < MIN_SPLIT_PIECE_AREA_M2:
        return None
    outer = list(poly.exterior.coords)[:-1]
    if len(outer) < 3:
        return None
    plane_model = _row_plane_model(row)
    record = dict(row)
    record["piece_id"] = piece_id
    record["area_xz_m2"] = round(float(poly.area), 6)
    record["corners"] = _ring_to_row_plane_corners(record, plane_model, outer)
    record["holes"] = [
        _ring_to_row_plane_corners(record, plane_model, list(ring.coords)[:-1])
        for ring in poly.interiors
        if len(list(ring.coords)) >= 4
    ]
    return record


def _row_plane_coeffs(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    plane_model = _row_plane_model(row)
    plane_point, normal = plane_model
    if plane_point is None or normal is None:
        return None
    roof_up = _normalize_roof_up(normal)
    if roof_up.shape[0] < 3:
        return None
    px, py, pz = plane_point
    a = float(roof_up[0])
    b = float(roof_up[1])
    c = float(roof_up[2])
    d = -(a * float(px) + b * float(py) + c * float(pz))
    return a, b, c, d


def _row_mirror_half_plane_coeffs(
    own_row: dict[str, Any],
    other_row: dict[str, Any],
) -> tuple[float, float, float] | None:
    own_plane = _row_plane_coeffs(own_row)
    other_plane = _row_plane_coeffs(other_row)
    if own_plane is None or other_plane is None:
        return None
    a_i, b_i, c_i, d_i = own_plane
    a_j, b_j, c_j, d_j = other_plane
    return (
        b_j * a_i - b_i * a_j,
        b_j * c_i - b_i * c_j,
        b_j * d_i - b_i * d_j,
    )


def trim_ridge_eave_rows_to_local_mirror_pieces(
    split_piece_rows: list[dict[str, Any]],
    *,
    pad_m: float = 1.0,
) -> list[dict[str, Any]]:
    supported_ridge_rows = [
        row
        for row in split_piece_rows
        if str(row.get("target_kind") or "") == "ridge_eave_plane_group"
        and str(row.get("piece_role") or "") == "supported"
    ]
    trimmed_rows: list[dict[str, Any]] = []
    for row in split_piece_rows:
        if row not in supported_ridge_rows:
            trimmed_rows.append(dict(row))
            continue
        piece_poly = _row_piece_polygon(row)
        if piece_poly is None or piece_poly.is_empty:
            trimmed_rows.append(dict(row))
            continue
        row_target_id = str(row.get("target_element_id") or "")
        row_plane_group_id = _ridge_eave_target_to_plane_group_id(row_target_id)
        row_partner_id = str(row.get("mirror_partner_plane_group_id") or "")
        row_signature_id = str(row.get("chain_signature_id") or "")
        if not row_partner_id or not row_signature_id:
            trimmed_rows.append(dict(row))
            continue

        partner_rows: list[dict[str, Any]] = []
        partner_polys: list[Polygon] = []
        seam_partner_row: dict[str, Any] | None = None
        best_overlap_area = 0.0
        for other in supported_ridge_rows:
            if (
                other is row
                or str(other.get("target_element_id") or "") == row_target_id
            ):
                continue
            if str(other.get("chain_signature_id") or "") != row_signature_id:
                continue
            if _rows_are_local_ownership_competitors(row, other):
                continue
            other_target_id = str(other.get("target_element_id") or "")
            other_plane_group_id = _ridge_eave_target_to_plane_group_id(other_target_id)
            other_partner_id = str(other.get("mirror_partner_plane_group_id") or "")
            has_reciprocal_partner = (
                other_plane_group_id is not None
                and row_partner_id == other_plane_group_id
            ) or (
                row_plane_group_id is not None
                and other_partner_id == row_plane_group_id
            )
            if not has_reciprocal_partner:
                continue
            other_poly = _row_piece_polygon(other)
            if other_poly is None or other_poly.is_empty:
                continue
            try:
                overlap = piece_poly.intersection(other_poly)
            except Exception:
                continue
            overlap_area = float(getattr(overlap, "area", 0.0) or 0.0)
            if overlap.is_empty or overlap_area < MIN_SPLIT_PIECE_AREA_M2:
                continue
            partner_rows.append(other)
            partner_polys.append(other_poly)
            if overlap_area > best_overlap_area:
                seam_partner_row = other
                best_overlap_area = overlap_area

        if seam_partner_row is None or not partner_polys:
            trimmed_rows.append(dict(row))
            continue

        coeffs = _row_mirror_half_plane_coeffs(row, seam_partner_row)
        if coeffs is None:
            trimmed_rows.append(dict(row))
            continue
        seam_partner_index = partner_rows.index(seam_partner_row)
        seam_partner_poly = partner_polys[seam_partner_index]
        extra_partner_polys = [
            poly for idx, poly in enumerate(partner_polys) if idx != seam_partner_index
        ]
        try:
            partner_union = unary_union(partner_polys)
            bounds = unary_union([piece_poly, partner_union]).bounds
        except Exception:
            partner_union = seam_partner_poly
            bounds = piece_poly.bounds
        keep_poly = _half_plane_polygon_for_bounds(coeffs, bounds, pad_m=pad_m)
        if keep_poly is None:
            trimmed_rows.append(dict(row))
            continue
        try:
            clipped = piece_poly.intersection(keep_poly)
            if extra_partner_polys:
                extra_partner_union = unary_union(extra_partner_polys)
                if not getattr(extra_partner_union, "is_empty", True):
                    clipped = clipped.difference(extra_partner_union)
        except Exception:
            trimmed_rows.append(dict(row))
            continue
        clipped_area = float(getattr(clipped, "area", 0.0) or 0.0)
        original_area = max(float(piece_poly.area), 1e-9)
        if (
            clipped.is_empty
            or clipped_area < MIN_SPLIT_PIECE_AREA_M2
            or clipped_area >= 0.995 * original_area
        ):
            trimmed_rows.append(dict(row))
            continue
        clipped_polys = _iter_polygons(clipped)
        if not clipped_polys:
            trimmed_rows.append(dict(row))
            continue
        piece_id = str(row.get("piece_id") or "")
        for poly_index, poly in enumerate(
            sorted(clipped_polys, key=lambda item: float(item.area), reverse=True)
        ):
            record = _row_piece_records_from_polygon(
                row,
                poly,
                piece_id=piece_id if poly_index == 0 else f"{piece_id}:{poly_index}",
            )
            if record is not None:
                trimmed_rows.append(record)
    return trimmed_rows


def _row_target_index(row: dict[str, Any]) -> int | None:
    target_element_id = str(row.get("target_element_id") or "")
    if "::roof-oblique::oblique:" not in target_element_id:
        return None
    try:
        return int(target_element_id.rsplit("oblique:", 1)[-1])
    except ValueError:
        return None


def _rounded_loop_3d(
    corners: list[list[float]], *, ndigits: int = 6
) -> list[list[float]]:
    rounded: list[list[float]] = []
    for corner in corners:
        if len(corner) < 3:
            continue
        point = [
            round(float(corner[0]), ndigits),
            round(float(corner[1]), ndigits),
            round(float(corner[2]), ndigits),
        ]
        if rounded and point == rounded[-1]:
            continue
        rounded.append(point)
    if len(rounded) >= 2 and rounded[0] == rounded[-1]:
        rounded.pop()
    return rounded


def _serialized_piece_holes(
    holes: list[list[list[float]]],
    *,
    ndigits: int = 6,
) -> list[list[list[float]]]:
    serialized: list[list[list[float]]] = []
    min_serialized_hole_area = 10.0 ** (-ndigits)
    for hole in holes:
        rounded = _rounded_loop_3d(hole, ndigits=ndigits)
        if len(rounded) < 3:
            continue
        coords_xz = [(float(corner[0]), float(corner[2])) for corner in rounded]
        if len(set(coords_xz)) < 3:
            continue
        hole_poly = Polygon(coords_xz)
        if (
            hole_poly.is_empty
            or not hole_poly.is_valid
            or float(hole_poly.area) <= min_serialized_hole_area
        ):
            continue
        serialized.append(rounded)
    return serialized


def _committed_supported_piece_hypothesis_metadata(
    split_piece_rows: list[dict[str, Any]],
    buildings_by_uuid: dict[str, dict[str, Any]],
    roof_results_by_uuid: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    room_polys_by_uuid: dict[str, dict[str, Polygon]] = {}
    hypothesis_part_union_by_uuid: dict[str, dict[str, Polygon]] = {}
    roof_surfaces_by_uuid: dict[str, list[dict[str, Any]]] = {}

    for uuid, roof_result in roof_results_by_uuid.items():
        building = buildings_by_uuid.get(uuid)
        if building is None:
            continue
        room_polys: dict[str, Polygon] = {}
        for room_index, room in enumerate(building.get("rooms") or []):
            poly = _xz_polygon(room.get("floor_polygon") or [])
            if poly is None or poly.is_empty:
                continue
            room_polys[f"room:{room_index}"] = poly
        room_polys_by_uuid[uuid] = room_polys

        building_part_graph = roof_result.get("building_part_graph") or {}
        room_membership = {
            str(room_id): {str(part_id) for part_id in (part_ids or []) if part_id}
            for room_id, part_ids in (
                building_part_graph.get("room_membership") or {}
            ).items()
        }
        unions_by_hypothesis: dict[str, Polygon] = {}
        for hypothesis_id, part_ids in (
            building_part_graph.get("hypothesis_membership") or {}
        ).items():
            wanted = {str(part_id) for part_id in (part_ids or []) if part_id}
            if not wanted:
                continue
            member_polys = [
                poly
                for room_id, poly in room_polys.items()
                if wanted.intersection(room_membership.get(room_id, set()))
            ]
            if not member_polys:
                continue
            try:
                unions_by_hypothesis[str(hypothesis_id)] = unary_union(member_polys)
            except Exception:
                continue
        hypothesis_part_union_by_uuid[uuid] = unions_by_hypothesis
        roof_surfaces_by_uuid[uuid] = list(
            (roof_result.get("roof_surfaces") or {}).get("oblique") or []
        )

    metadata_by_piece_id: dict[str, dict[str, Any]] = {}
    for row in split_piece_rows:
        if (
            str(row.get("target_kind") or "") != "committed_oblique"
            or str(row.get("piece_role") or "") != "supported"
        ):
            continue
        uuid = str(row.get("uuid") or "")
        target_index = _row_target_index(row)
        roof_surfaces = roof_surfaces_by_uuid.get(uuid) or []
        if (
            target_index is None
            or target_index < 0
            or target_index >= len(roof_surfaces)
        ):
            continue
        surface = roof_surfaces[target_index] or {}
        hypothesis_id = str(surface.get("roof_hypothesis_id") or "")
        if not hypothesis_id:
            continue
        part_ids = [
            str(part_id)
            for part_id in (
                (
                    (roof_results_by_uuid.get(uuid) or {}).get("building_part_graph")
                    or {}
                )
                .get("hypothesis_membership", {})
                .get(hypothesis_id, [])
            )
            if part_id
        ]
        poly = _row_piece_polygon(row)
        part_union = (hypothesis_part_union_by_uuid.get(uuid) or {}).get(hypothesis_id)
        overlap_fraction = None
        misaligned = False
        if (
            poly is not None
            and not poly.is_empty
            and part_union is not None
            and not part_union.is_empty
        ):
            try:
                overlap_area = float(poly.intersection(part_union).area)
            except Exception:
                overlap_area = 0.0
            overlap_fraction = round(overlap_area / max(float(poly.area), 1e-9), 6)
            misaligned = overlap_area <= 1e-9
        metadata_by_piece_id[str(row.get("piece_id") or "")] = {
            "roof_hypothesis_id": hypothesis_id,
            "hypothesis_part_ids": part_ids,
            "hypothesis_part_overlap_fraction": overlap_fraction,
            "hypothesis_part_misaligned": misaligned,
        }
    return metadata_by_piece_id


def _committed_target_hypothesis_metadata(
    committed_targets: list[TargetPlaneRecord],
    building: dict[str, Any],
    roof_result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    room_polys: dict[str, Polygon] = {}
    for room_index, room in enumerate(building.get("rooms") or []):
        poly = _xz_polygon(room.get("floor_polygon") or [])
        if poly is None or poly.is_empty:
            continue
        room_polys[f"room:{room_index}"] = poly

    building_part_graph = roof_result.get("building_part_graph") or {}
    room_membership = {
        str(room_id): {str(part_id) for part_id in (part_ids or []) if part_id}
        for room_id, part_ids in (
            building_part_graph.get("room_membership") or {}
        ).items()
    }
    unions_by_hypothesis: dict[str, Polygon] = {}
    for hypothesis_id, part_ids in (
        building_part_graph.get("hypothesis_membership") or {}
    ).items():
        wanted = {str(part_id) for part_id in (part_ids or []) if part_id}
        if not wanted:
            continue
        member_polys = [
            poly
            for room_id, poly in room_polys.items()
            if wanted.intersection(room_membership.get(room_id, set()))
        ]
        if not member_polys:
            continue
        try:
            unions_by_hypothesis[str(hypothesis_id)] = unary_union(member_polys)
        except Exception:
            continue

    roof_surfaces = list((roof_result.get("roof_surfaces") or {}).get("oblique") or [])
    metadata_by_target_id: dict[str, dict[str, Any]] = {}
    for target in committed_targets:
        target_index = int(target.target_index)
        if target_index < 0 or target_index >= len(roof_surfaces):
            continue
        hypothesis_id = str(
            (roof_surfaces[target_index] or {}).get("roof_hypothesis_id") or ""
        )
        if not hypothesis_id:
            continue
        part_ids = [
            str(part_id)
            for part_id in (building_part_graph.get("hypothesis_membership") or {}).get(
                hypothesis_id, []
            )
            if part_id
        ]
        part_union = unions_by_hypothesis.get(hypothesis_id)
        overlap_fraction = None
        misaligned = False
        if part_union is not None and not part_union.is_empty:
            try:
                overlap_area = float(target.poly_xz.intersection(part_union).area)
            except Exception:
                overlap_area = 0.0
            overlap_fraction = round(
                overlap_area / max(float(target.poly_xz.area), 1e-9), 6
            )
            misaligned = overlap_area <= 1e-9
        metadata_by_target_id[target.element_id] = {
            "roof_hypothesis_id": hypothesis_id,
            "hypothesis_part_ids": part_ids,
            "hypothesis_part_overlap_fraction": overlap_fraction,
            "hypothesis_part_misaligned": misaligned,
        }
    return metadata_by_target_id


def _target_face_run_annotations(
    face_run_seed_by_target_id: dict[str, FaceRunSeedRecord],
    target_metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for target_id, seed in face_run_seed_by_target_id.items():
        role = (
            "committed_core"
            if target_id in seed.core_committed_target_ids
            else "committed_misaligned"
            if target_id in seed.committed_target_ids
            else "ridge_member"
            if target_id in seed.ridge_target_ids
            else "face_member"
        )
        payload = {
            "face_run_id": seed.face_run_id,
            "face_run_role": role,
            "face_run_committed_target_ids": list(seed.core_committed_target_ids),
            "face_run_hypothesis_ids": list(seed.hypothesis_ids),
            "face_run_hypothesis_part_ids": list(seed.hypothesis_part_ids),
        }
        payload.update(target_metadata_by_id.get(target_id, {}))
        annotations[target_id] = payload
    return annotations


def annotate_rows_with_target_face_runs(
    rows: list[dict[str, Any]],
    target_face_annotations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    annotated_rows: list[dict[str, Any]] = []
    for row in rows:
        annotated = dict(row)
        target_id = str(row.get("target_element_id") or row.get("element_id") or "")
        payload = target_face_annotations.get(target_id)
        if payload is not None:
            annotated.update(payload)
        annotated_rows.append(annotated)
    return annotated_rows


def build_target_face_runs(
    targets: list[TargetPlaneRecord],
    ridge_eave_targets: list[TargetPlaneRecord],
    building: dict[str, Any],
    roof_result: dict[str, Any],
) -> tuple[list[FaceRunSeedRecord], dict[str, dict[str, Any]]]:
    committed_targets = [
        target for target in targets if target.target_kind == "committed_oblique"
    ]
    target_metadata_by_id = _committed_target_hypothesis_metadata(
        committed_targets, building, roof_result
    )

    run_entries: list[dict[str, Any]] = []
    run_index_by_committed_target_id: dict[str, int] = {}
    for target in committed_targets:
        metadata = target_metadata_by_id.get(target.element_id, {})
        is_core = not bool(metadata.get("hypothesis_part_misaligned"))
        run_entries.append(
            {
                "uuid": target.uuid,
                "story": target.story,
                "azimuth_deg": target.azimuth_deg,
                "inclination_deg": target.inclination_deg,
                "committed_targets": [target],
                "ridge_targets": [],
                "core_committed_targets": [target] if is_core else [],
                "core_polys": [target.poly_xz] if is_core else [],
                "hypothesis_ids": {str(metadata.get("roof_hypothesis_id") or "")}
                if metadata.get("roof_hypothesis_id")
                else set(),
                "hypothesis_part_ids": {
                    str(part_id)
                    for part_id in (metadata.get("hypothesis_part_ids") or [])
                    if part_id
                },
            }
        )
        run_index_by_committed_target_id[target.element_id] = len(run_entries) - 1

    for ridge_target in ridge_eave_targets:
        best_index = None
        best_key: tuple[float, float] | None = None
        for committed_target in committed_targets:
            metadata = target_metadata_by_id.get(committed_target.element_id, {})
            if bool(metadata.get("hypothesis_part_misaligned")):
                continue
            if not _targets_are_local_ownership_competitors(
                ridge_target, committed_target
            ):
                continue
            try:
                overlap = ridge_target.poly_xz.intersection(committed_target.poly_xz)
            except Exception:
                continue
            if overlap.is_empty or float(overlap.area) < MIN_MATCH_OVERLAP_M2:
                continue
            overlap_point = overlap.representative_point()
            ridge_y = _plane_y_at(
                ridge_target, float(overlap_point.x), float(overlap_point.y)
            )
            committed_y = _plane_y_at(
                committed_target, float(overlap_point.x), float(overlap_point.y)
            )
            if ridge_y is None or committed_y is None:
                continue
            if abs(ridge_y - committed_y) > PLANE_EAVE_CHAIN_HEIGHT_TOL_M:
                continue
            score_key = (float(overlap.area), -float(committed_target.poly_xz.area))
            if best_key is None or score_key > best_key:
                best_key = score_key
                best_index = run_index_by_committed_target_id.get(
                    committed_target.element_id
                )
        if best_index is None:
            run_entries.append(
                {
                    "uuid": ridge_target.uuid,
                    "story": ridge_target.story,
                    "azimuth_deg": ridge_target.azimuth_deg,
                    "inclination_deg": ridge_target.inclination_deg,
                    "committed_targets": [],
                    "ridge_targets": [ridge_target],
                    "core_committed_targets": [],
                    "core_polys": [],
                    "hypothesis_ids": set(),
                    "hypothesis_part_ids": set(),
                }
            )
            continue
        run_entries[best_index]["ridge_targets"].append(ridge_target)

    face_runs: list[FaceRunSeedRecord] = []
    for entry in run_entries:
        committed_target_ids = tuple(
            sorted({target.element_id for target in entry["committed_targets"]})
        )
        ridge_target_ids = tuple(
            sorted({target.element_id for target in entry["ridge_targets"]})
        )
        if not committed_target_ids and not ridge_target_ids:
            continue
        core_committed_target_ids = tuple(
            sorted({target.element_id for target in entry["core_committed_targets"]})
        )
        try:
            core_union = (
                unary_union(entry["core_polys"]) if entry["core_polys"] else None
            )
        except Exception:
            core_union = None
        face_runs.append(
            FaceRunSeedRecord(
                uuid=str(entry["uuid"]),
                face_run_id=_face_run_id(
                    str(entry["uuid"]), committed_target_ids, ridge_target_ids
                ),
                story=int(entry["story"]),
                azimuth_deg=float(entry["azimuth_deg"])
                if entry["azimuth_deg"] is not None
                else None,
                inclination_deg=float(entry["inclination_deg"])
                if entry["inclination_deg"] is not None
                else None,
                member_target_ids=tuple(
                    sorted(set(committed_target_ids).union(ridge_target_ids))
                ),
                committed_target_ids=committed_target_ids,
                ridge_target_ids=ridge_target_ids,
                core_committed_target_ids=core_committed_target_ids,
                hypothesis_ids=tuple(
                    sorted(value for value in entry["hypothesis_ids"] if value)
                ),
                hypothesis_part_ids=tuple(
                    sorted(value for value in entry["hypothesis_part_ids"] if value)
                ),
                core_target_union=core_union
                if core_union is not None and not getattr(core_union, "is_empty", True)
                else None,
            )
        )
    return face_runs, target_metadata_by_id


def annotate_committed_supported_pieces_with_hypothesis_part_overlap(
    split_piece_rows: list[dict[str, Any]],
    buildings_by_uuid: dict[str, dict[str, Any]],
    roof_results_by_uuid: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata_by_piece_id = _committed_supported_piece_hypothesis_metadata(
        split_piece_rows,
        buildings_by_uuid,
        roof_results_by_uuid,
    )

    annotated_rows: list[dict[str, Any]] = []
    for row in split_piece_rows:
        annotated = dict(row)
        annotated.setdefault("hypothesis_part_overlap_fraction", None)
        annotated.setdefault("hypothesis_part_misaligned", False)
        annotated.setdefault("hypothesis_part_ids", [])
        annotated.setdefault("roof_hypothesis_id", None)
        metadata = metadata_by_piece_id.get(str(row.get("piece_id") or ""))
        if metadata is None:
            annotated_rows.append(annotated)
            continue
        annotated.update(metadata)
        annotated_rows.append(annotated)
    return annotated_rows


def _face_run_id(
    uuid: str,
    committed_piece_ids: tuple[str, ...],
    ridge_piece_ids: tuple[str, ...],
) -> str:
    digest = hashlib.blake2b(
        repr((uuid, committed_piece_ids, ridge_piece_ids)).encode("utf-8"),
        digest_size=6,
    ).hexdigest()
    return f"{uuid}::face-run::{digest}"


def build_face_runs(
    split_piece_rows: list[dict[str, Any]],
) -> list[FaceRunRecord]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_piece_rows:
        grouped[str(row.get("uuid") or "")].append(dict(row))

    face_runs: list[FaceRunRecord] = []
    for uuid, building_rows in grouped.items():
        committed_rows: list[
            tuple[
                dict[str, Any],
                Polygon,
                tuple[tuple[float, float, float], np.ndarray] | tuple[None, None],
            ]
        ] = []
        for row in building_rows:
            if (
                str(row.get("target_kind") or "") != "committed_oblique"
                or str(row.get("piece_role") or "") != "supported"
            ):
                continue
            poly = _row_piece_polygon(row)
            if poly is None or poly.is_empty:
                continue
            committed_rows.append((row, poly, _row_plane_model(row)))

        run_entries: list[dict[str, Any]] = []
        run_index_by_anchor_piece_id: dict[str, int] = {}
        for row, poly, _plane_model in committed_rows:
            anchor_piece_id = str(row.get("piece_id") or "")
            hypothesis_ids = tuple(
                value for value in [str(row.get("roof_hypothesis_id") or "")] if value
            )
            hypothesis_part_ids = tuple(
                sorted(
                    {
                        str(part_id)
                        for part_id in (row.get("hypothesis_part_ids") or [])
                        if part_id
                    }
                )
            )
            run_entries.append(
                {
                    "uuid": uuid,
                    "story": int(row.get("story") or 0),
                    "azimuth_deg": row.get("target_azimuth_deg"),
                    "inclination_deg": row.get("target_inclination_deg"),
                    "committed_rows": [row],
                    "ridge_rows": [],
                    "core_committed_rows": []
                    if bool(row.get("hypothesis_part_misaligned"))
                    else [row],
                    "core_polys": []
                    if bool(row.get("hypothesis_part_misaligned"))
                    else [poly],
                    "hypothesis_ids": set(hypothesis_ids),
                    "hypothesis_part_ids": set(hypothesis_part_ids),
                }
            )
            run_index_by_anchor_piece_id[anchor_piece_id] = len(run_entries) - 1

        for row in building_rows:
            if (
                str(row.get("target_kind") or "") != "ridge_eave_plane_group"
                or str(row.get("piece_role") or "") != "supported"
            ):
                continue
            piece_poly = _row_piece_polygon(row)
            if piece_poly is None or piece_poly.is_empty:
                continue
            row_plane = _row_plane_model(row)
            best_index = None
            best_key: tuple[float, float] | None = None
            for index, (committed_row, committed_poly, committed_plane) in enumerate(
                committed_rows
            ):
                if bool(committed_row.get("hypothesis_part_misaligned")):
                    continue
                if not _rows_are_local_ownership_competitors(row, committed_row):
                    continue
                try:
                    overlap = piece_poly.intersection(committed_poly)
                except Exception:
                    continue
                if overlap.is_empty or float(overlap.area) < MIN_SPLIT_PIECE_AREA_M2:
                    continue
                overlap_point = overlap.representative_point()
                row_y = _row_plane_y_at(
                    row_plane, float(overlap_point.x), float(overlap_point.y)
                )
                committed_y = _row_plane_y_at(
                    committed_plane, float(overlap_point.x), float(overlap_point.y)
                )
                if row_y is None or committed_y is None:
                    continue
                if abs(row_y - committed_y) > PLANE_EAVE_CHAIN_HEIGHT_TOL_M:
                    continue
                score_key = (float(overlap.area), -float(committed_poly.area))
                if best_key is None or score_key > best_key:
                    best_key = score_key
                    best_index = index
            if best_index is None:
                run_entries.append(
                    {
                        "uuid": uuid,
                        "story": int(row.get("story") or 0),
                        "azimuth_deg": row.get("target_azimuth_deg"),
                        "inclination_deg": row.get("target_inclination_deg"),
                        "committed_rows": [],
                        "ridge_rows": [row],
                        "core_committed_rows": [],
                        "core_polys": [],
                        "hypothesis_ids": set(),
                        "hypothesis_part_ids": set(),
                    }
                )
                continue
            anchor_piece_id = str(committed_rows[best_index][0].get("piece_id") or "")
            run_index = run_index_by_anchor_piece_id.get(anchor_piece_id)
            if run_index is None:
                continue
            run_entries[run_index]["ridge_rows"].append(row)

        for entry in run_entries:
            committed_piece_ids = tuple(
                sorted(
                    {
                        str(row.get("piece_id") or "")
                        for row in entry["committed_rows"]
                        if row.get("piece_id")
                    }
                )
            )
            ridge_piece_ids = tuple(
                sorted(
                    {
                        str(row.get("piece_id") or "")
                        for row in entry["ridge_rows"]
                        if row.get("piece_id")
                    }
                )
            )
            if not committed_piece_ids and not ridge_piece_ids:
                continue
            core_committed_piece_ids = tuple(
                sorted(
                    {
                        str(row.get("piece_id") or "")
                        for row in entry["core_committed_rows"]
                        if row.get("piece_id")
                    }
                )
            )
            core_committed_target_ids = tuple(
                sorted(
                    {
                        str(row.get("target_element_id") or "")
                        for row in entry["core_committed_rows"]
                        if row.get("target_element_id")
                    }
                )
            )
            try:
                core_union = (
                    unary_union(entry["core_polys"]) if entry["core_polys"] else None
                )
            except Exception:
                core_union = None
            face_runs.append(
                FaceRunRecord(
                    uuid=uuid,
                    face_run_id=_face_run_id(
                        uuid, committed_piece_ids, ridge_piece_ids
                    ),
                    story=int(entry["story"]),
                    azimuth_deg=float(entry["azimuth_deg"])
                    if entry["azimuth_deg"] is not None
                    else None,
                    inclination_deg=float(entry["inclination_deg"])
                    if entry["inclination_deg"] is not None
                    else None,
                    member_piece_ids=tuple(
                        sorted(set(committed_piece_ids).union(ridge_piece_ids))
                    ),
                    committed_piece_ids=committed_piece_ids,
                    ridge_piece_ids=ridge_piece_ids,
                    core_committed_piece_ids=core_committed_piece_ids,
                    core_committed_target_ids=core_committed_target_ids,
                    hypothesis_ids=tuple(sorted(entry["hypothesis_ids"])),
                    hypothesis_part_ids=tuple(sorted(entry["hypothesis_part_ids"])),
                    core_union=core_union
                    if core_union is not None
                    and not getattr(core_union, "is_empty", True)
                    else None,
                )
            )
    return face_runs


def resolve_split_piece_rows_with_face_runs(
    split_piece_rows: list[dict[str, Any]],
    face_runs: list[FaceRunRecord],
) -> list[dict[str, Any]]:
    face_run_by_piece_id: dict[str, FaceRunRecord] = {}
    for face_run in face_runs:
        for piece_id in face_run.member_piece_ids:
            face_run_by_piece_id[piece_id] = face_run

    resolved_rows: list[dict[str, Any]] = []
    for row in split_piece_rows:
        piece_id = str(row.get("piece_id") or "")
        face_run = face_run_by_piece_id.get(piece_id)
        if face_run is None:
            resolved_rows.append(dict(row))
            continue

        def _annotate_face_run(
            record: dict[str, Any], *, role: str, face_run=face_run
        ) -> dict[str, Any]:
            annotated = dict(record)
            annotated["face_run_id"] = face_run.face_run_id
            annotated["face_run_role"] = role
            annotated["face_run_committed_target_ids"] = list(
                face_run.core_committed_target_ids
            )
            annotated["face_run_hypothesis_ids"] = list(face_run.hypothesis_ids)
            annotated["face_run_hypothesis_part_ids"] = list(
                face_run.hypothesis_part_ids
            )
            return annotated

        if (
            str(row.get("target_kind") or "") == "ridge_eave_plane_group"
            and str(row.get("piece_role") or "") == "supported"
            and face_run.core_union is not None
            and face_run.core_committed_target_ids
        ):
            piece_poly = _row_piece_polygon(row)
            if piece_poly is None or piece_poly.is_empty:
                resolved_rows.append(_annotate_face_run(row, role="ridge_continuation"))
                continue
            try:
                residual = piece_poly.difference(face_run.core_union)
            except Exception:
                resolved_rows.append(_annotate_face_run(row, role="ridge_continuation"))
                continue
            residual_polys = _iter_polygons(residual)
            if not residual_polys:
                continue
            residual_area = sum(float(poly.area) for poly in residual_polys)
            merge_fraction = max(
                0.0, min(1.0, 1.0 - residual_area / max(float(piece_poly.area), 1e-9))
            )
            for poly_index, poly in enumerate(
                sorted(residual_polys, key=lambda item: float(item.area), reverse=True)
            ):
                record = _row_piece_records_from_polygon(
                    row,
                    poly,
                    piece_id=piece_id
                    if poly_index == 0
                    else f"{piece_id}:{poly_index}",
                )
                if record is None:
                    continue
                annotated = _annotate_face_run(record, role="ridge_continuation")
                annotated["same_plane_committed_core_fraction"] = round(
                    merge_fraction, 6
                )
                annotated["same_plane_committed_core_target_ids"] = list(
                    face_run.core_committed_target_ids[:3]
                )
                resolved_rows.append(annotated)
            continue

        role = (
            "committed_core"
            if piece_id in face_run.core_committed_piece_ids
            else (
                "committed_misaligned"
                if piece_id in face_run.committed_piece_ids
                else "ridge_run"
            )
        )
        resolved_rows.append(_annotate_face_run(row, role=role))
    return resolved_rows


def resolve_split_piece_rows_with_target_face_runs(
    split_piece_rows: list[dict[str, Any]],
    face_runs: list[FaceRunSeedRecord],
) -> list[dict[str, Any]]:
    face_run_by_target_id: dict[str, FaceRunSeedRecord] = {}
    for face_run in face_runs:
        for target_id in face_run.member_target_ids:
            face_run_by_target_id[target_id] = face_run

    resolved_rows: list[dict[str, Any]] = []
    for row in split_piece_rows:
        target_id = str(row.get("target_element_id") or "")
        face_run = face_run_by_target_id.get(target_id)
        if face_run is None:
            resolved_rows.append(dict(row))
            continue

        def _annotate(
            record: dict[str, Any], *, role: str, face_run=face_run
        ) -> dict[str, Any]:
            annotated = dict(record)
            annotated["face_run_id"] = face_run.face_run_id
            annotated["face_run_role"] = role
            annotated["face_run_committed_target_ids"] = list(
                face_run.core_committed_target_ids
            )
            annotated["face_run_hypothesis_ids"] = list(face_run.hypothesis_ids)
            annotated["face_run_hypothesis_part_ids"] = list(
                face_run.hypothesis_part_ids
            )
            return annotated

        if (
            str(row.get("target_kind") or "") == "ridge_eave_plane_group"
            and str(row.get("piece_role") or "") == "supported"
            and target_id in face_run.ridge_target_ids
            and face_run.core_target_union is not None
            and face_run.core_committed_target_ids
        ):
            piece_poly = _row_piece_polygon(row)
            if piece_poly is None or piece_poly.is_empty:
                resolved_rows.append(_annotate(row, role="ridge_continuation"))
                continue
            try:
                residual = piece_poly.difference(face_run.core_target_union)
            except Exception:
                resolved_rows.append(_annotate(row, role="ridge_continuation"))
                continue
            residual_polys = _iter_polygons(residual)
            if not residual_polys:
                continue
            residual_area = sum(float(poly.area) for poly in residual_polys)
            merge_fraction = max(
                0.0, min(1.0, 1.0 - residual_area / max(float(piece_poly.area), 1e-9))
            )
            piece_id = str(row.get("piece_id") or "")
            for poly_index, poly in enumerate(
                sorted(residual_polys, key=lambda item: float(item.area), reverse=True)
            ):
                record = _row_piece_records_from_polygon(
                    row,
                    poly,
                    piece_id=piece_id
                    if poly_index == 0
                    else f"{piece_id}:{poly_index}",
                )
                if record is None:
                    continue
                annotated = _annotate(record, role="ridge_continuation")
                annotated["same_plane_committed_core_fraction"] = round(
                    merge_fraction, 6
                )
                annotated["same_plane_committed_core_target_ids"] = list(
                    face_run.core_committed_target_ids[:3]
                )
                resolved_rows.append(annotated)
            continue

        role = (
            "committed_misaligned"
            if str(row.get("target_kind") or "") == "committed_oblique"
            and bool(row.get("hypothesis_part_misaligned"))
            else "committed_core"
            if target_id in face_run.core_committed_target_ids
            else "committed_misaligned"
            if target_id in face_run.committed_target_ids
            else "ridge_run"
            if target_id in face_run.ridge_target_ids
            else "face_member"
        )
        resolved_rows.append(_annotate(row, role=role))
    return resolved_rows


def merge_same_plane_committed_oblique_cores(
    split_piece_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    face_runs = build_face_runs(split_piece_rows)
    return resolve_split_piece_rows_with_face_runs(split_piece_rows, face_runs)


def _split_piece_kind_priority(target_kind: str) -> int:
    if target_kind == "committed_oblique":
        return 3
    if target_kind == "candidate_oblique":
        return 2
    if target_kind == "ridge_eave_plane_group":
        return 1
    return 0


def _piece_numeric_value(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _chain_signature(chain_ids: Any) -> tuple[str, ...]:
    if not isinstance(chain_ids, (list, tuple)):
        return ()
    values = sorted({str(chain_id) for chain_id in chain_ids if str(chain_id)})
    return tuple(values)


def _chain_signature_id(uuid: str, story: int, chain_ids: Any) -> str | None:
    signature = _chain_signature(chain_ids)
    if not signature:
        return None
    digest = hashlib.blake2b(
        repr((uuid, story, signature)).encode("utf-8"), digest_size=6
    ).hexdigest()
    return f"{uuid}::face-run-signature::{story}:{digest}"


def _chain_signature_relation(
    chain_ids_a: Any,
    chain_ids_b: Any,
) -> str | None:
    signature_a = set(_chain_signature(chain_ids_a))
    signature_b = set(_chain_signature(chain_ids_b))
    if not signature_a or not signature_b:
        return None
    if signature_a == signature_b:
        return "equal"
    if signature_a.issubset(signature_b):
        return "subset"
    if signature_b.issubset(signature_a):
        return "superset"
    if signature_a.isdisjoint(signature_b):
        return "disjoint"
    return "partial_overlap"


def _piece_ownership_rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(_split_piece_kind_priority(str(row.get("target_kind") or ""))),
        _piece_numeric_value(row, "support_score", 0.0),
        -_piece_numeric_value(row, "local_competitor_loss_fraction", 1.0),
        -_piece_numeric_value(row, "best_supported_chain_height_residual_m", 1e6),
        _piece_numeric_value(row, "mirror_support_score", 0.0),
        -_piece_numeric_value(row, "through_ratio", 1e6),
    )


def _targets_are_local_ownership_competitors(
    subject_target: TargetPlaneRecord,
    other_target: TargetPlaneRecord,
) -> bool:
    azimuth_delta = _wrapped_angle_delta_deg(
        subject_target.azimuth_deg, other_target.azimuth_deg
    )
    if (
        azimuth_delta is None
        or azimuth_delta > RIDGE_EAVE_OWNERSHIP_MAX_AZIMUTH_DELTA_DEG
    ):
        return False
    inclination_delta = abs(
        float(subject_target.inclination_deg) - float(other_target.inclination_deg)
    )
    if inclination_delta > RIDGE_EAVE_OWNERSHIP_MAX_INCLINATION_DELTA_DEG:
        return False
    return True


def _rows_are_local_ownership_competitors(
    subject_row: dict[str, Any],
    other_row: dict[str, Any],
) -> bool:
    try:
        subject_azimuth = float(subject_row.get("target_azimuth_deg"))
        other_azimuth = float(other_row.get("target_azimuth_deg"))
        subject_inclination = float(subject_row.get("target_inclination_deg"))
        other_inclination = float(other_row.get("target_inclination_deg"))
    except (TypeError, ValueError):
        return True
    azimuth_delta = _wrapped_angle_delta_deg(subject_azimuth, other_azimuth)
    if (
        azimuth_delta is None
        or azimuth_delta > RIDGE_EAVE_OWNERSHIP_MAX_AZIMUTH_DELTA_DEG
    ):
        return False
    inclination_delta = abs(subject_inclination - other_inclination)
    if inclination_delta > RIDGE_EAVE_OWNERSHIP_MAX_INCLINATION_DELTA_DEG:
        return False
    return True


def _rows_share_ridge_eave_competition_class(
    subject_row: dict[str, Any],
    other_row: dict[str, Any],
) -> bool:
    if (
        str(subject_row.get("target_kind") or "") != "ridge_eave_plane_group"
        or str(other_row.get("target_kind") or "") != "ridge_eave_plane_group"
    ):
        return True
    relation = _chain_signature_relation(
        subject_row.get("chain_ids"), other_row.get("chain_ids")
    )
    if relation is None:
        # Synthetic tests and older rows may not carry chain_ids. Preserve the
        # previous fallback behavior in that case.
        return True
    return relation == "equal"


def _ridge_eave_target_to_plane_group_id(target_element_id: str) -> str | None:
    marker = "::ridge-eave-candidate::plane-group::"
    if marker not in target_element_id:
        return None
    prefix, suffix = target_element_id.split(marker, 1)
    if not prefix or not suffix:
        return None
    return f"{prefix}::plane-group::{suffix}"


def _plane_group_to_ridge_eave_target_element_id(plane_group_id: str) -> str | None:
    marker = "::plane-group::"
    if marker not in plane_group_id:
        return None
    prefix, suffix = plane_group_id.split(marker, 1)
    if not prefix or not suffix:
        return None
    return f"{prefix}::ridge-eave-candidate::plane-group::{suffix}"


def annotate_split_piece_rows_with_precedence(
    split_piece_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    rows_by_uuid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_piece_rows:
        row_copy = dict(row)
        grouped[(str(row["uuid"]), int(row["story"]))].append(row_copy)
        rows_by_uuid[str(row["uuid"])].append(row_copy)

    annotated_rows: list[dict[str, Any]] = []
    for (uuid, _story), story_rows in grouped.items():
        building_rows = rows_by_uuid[uuid]
        supported_polys: dict[str, Polygon] = {}
        for row in building_rows:
            if row.get("piece_role") != "supported":
                continue
            poly = _xz_polygon(row.get("corners") or [])
            if poly is None or poly.is_empty:
                continue
            supported_polys[str(row["piece_id"])] = poly

        for row in story_rows:
            annotated = dict(row)
            piece_id = str(row["piece_id"])
            poly = supported_polys.get(piece_id)
            if row.get("piece_role") != "supported" or poly is None:
                annotated["higher_priority_cover_fraction"] = 0.0
                annotated["higher_priority_covering_target_ids"] = []
                annotated["committed_cover_fraction"] = 0.0
                annotated["committed_covering_target_ids"] = []
                annotated["roof_surface_cover_fraction"] = 0.0
                annotated["roof_surface_covering_target_ids"] = []
                annotated["same_side_superset_cover_fraction"] = 0.0
                annotated["same_side_superset_covering_target_ids"] = []
                annotated["ownership_redundant"] = False
                annotated_rows.append(annotated)
                continue

            row_rank = _piece_ownership_rank(row)
            row_kind = str(row.get("target_kind") or "")
            row_target_id = str(row.get("target_element_id") or "")
            row_partner_id = str(row.get("mirror_partner_plane_group_id") or "")
            row_plane_group_id = _ridge_eave_target_to_plane_group_id(row_target_id)
            stronger_rows = [
                other
                for other in story_rows
                if other.get("piece_role") == "supported"
                and str(other.get("piece_id")) != piece_id
                and _piece_ownership_rank(other) > row_rank
                and _rows_share_ridge_eave_competition_class(row, other)
                and not (
                    row_kind == "ridge_eave_plane_group"
                    and not _rows_are_local_ownership_competitors(row, other)
                )
                and not (
                    row_kind == "ridge_eave_plane_group"
                    and str(other.get("target_kind") or "") == "ridge_eave_plane_group"
                    and (
                        (
                            bool(row_partner_id)
                            and bool(
                                _ridge_eave_target_to_plane_group_id(
                                    str(other.get("target_element_id") or "")
                                )
                            )
                            and row_partner_id
                            == _ridge_eave_target_to_plane_group_id(
                                str(other.get("target_element_id") or "")
                            )
                        )
                        or (
                            bool(str(other.get("mirror_partner_plane_group_id") or ""))
                            and bool(row_plane_group_id)
                            and str(other.get("mirror_partner_plane_group_id") or "")
                            == row_plane_group_id
                        )
                    )
                )
            ]
            covering_ids: list[tuple[str, float]] = []
            cover_parts: list[Polygon] = []
            for other in stronger_rows:
                other_poly = supported_polys.get(str(other["piece_id"]))
                if other_poly is None:
                    continue
                try:
                    overlap = poly.intersection(other_poly)
                except Exception:
                    continue
                if overlap.is_empty or float(overlap.area) < MIN_SPLIT_PIECE_AREA_M2:
                    continue
                cover_parts.append(other_poly)
                covering_ids.append(
                    (
                        str(other["target_element_id"]),
                        float(overlap.area),
                    )
                )

            if cover_parts:
                try:
                    higher_union = unary_union(cover_parts)
                    cover_fraction = float(poly.intersection(higher_union).area) / max(
                        float(poly.area), 1e-9
                    )
                except Exception:
                    cover_fraction = 0.0
            else:
                cover_fraction = 0.0
            covering_ids.sort(key=lambda item: (-item[1], item[0]))
            committed_cover_ids: list[tuple[str, float]] = []
            committed_cover_parts: list[Polygon] = []
            roof_surface_cover_ids: list[tuple[str, float]] = []
            roof_surface_cover_parts: list[Polygon] = []
            local_roof_cover_ids: list[tuple[str, float]] = []
            local_roof_cover_parts: list[Polygon] = []
            same_side_superset_cover_ids: list[tuple[str, float]] = []
            same_side_superset_cover_parts: list[Polygon] = []
            if row_kind == "ridge_eave_plane_group":
                row_signature_id = _chain_signature_id(
                    str(row.get("uuid") or ""),
                    int(row.get("story") or 0),
                    row.get("chain_ids"),
                )
                row_plane_group_id = _ridge_eave_target_to_plane_group_id(
                    str(row.get("target_element_id") or "")
                )
                row_partner_id = str(row.get("mirror_partner_plane_group_id") or "")
                for other in building_rows:
                    if other.get("piece_role") != "supported":
                        continue
                    if str(other.get("piece_id")) == piece_id:
                        continue
                    other_poly = supported_polys.get(str(other["piece_id"]))
                    if other_poly is None:
                        continue
                    try:
                        overlap = poly.intersection(other_poly)
                    except Exception:
                        continue
                    if (
                        overlap.is_empty
                        or float(overlap.area) < MIN_SPLIT_PIECE_AREA_M2
                    ):
                        continue
                    other_kind = str(other.get("target_kind") or "")
                    if other_kind in {"committed_oblique", "candidate_oblique"}:
                        if _piece_ownership_rank(other) <= row_rank:
                            continue
                        roof_surface_cover_parts.append(other_poly)
                        roof_surface_cover_ids.append(
                            (
                                str(other["target_element_id"]),
                                float(overlap.area),
                            )
                        )
                        if other_kind == "committed_oblique":
                            committed_cover_parts.append(other_poly)
                            committed_cover_ids.append(
                                (
                                    str(other["target_element_id"]),
                                    float(overlap.area),
                                )
                            )
                            local_roof_cover_parts.append(other_poly)
                            local_roof_cover_ids.append(
                                (
                                    str(other["target_element_id"]),
                                    float(overlap.area),
                                )
                            )
                        continue
                    if other_kind != "ridge_eave_plane_group":
                        continue
                    if _piece_ownership_rank(other) > row_rank:
                        azimuth_delta = _wrapped_angle_delta_deg(
                            _piece_numeric_value(row, "target_azimuth_deg", 0.0),
                            _piece_numeric_value(other, "target_azimuth_deg", 0.0),
                        )
                        relation = _chain_signature_relation(
                            row.get("chain_ids"), other.get("chain_ids")
                        )
                        if (
                            azimuth_delta is not None
                            and azimuth_delta
                            <= RIDGE_EAVE_OWNERSHIP_MAX_AZIMUTH_DELTA_DEG
                            and relation == "subset"
                            and not str(
                                other.get("mirror_partner_plane_group_id") or ""
                            )
                        ):
                            same_side_superset_cover_parts.append(other_poly)
                            same_side_superset_cover_ids.append(
                                (
                                    str(other["target_element_id"]),
                                    float(overlap.area),
                                )
                            )
                    if row_signature_id is None:
                        continue
                    other_signature_id = _chain_signature_id(
                        str(other.get("uuid") or ""),
                        int(other.get("story") or 0),
                        other.get("chain_ids"),
                    )
                    if other_signature_id != row_signature_id:
                        continue
                    if _rows_are_local_ownership_competitors(row, other):
                        continue
                    other_plane_group_id = _ridge_eave_target_to_plane_group_id(
                        str(other.get("target_element_id") or "")
                    )
                    other_partner_id = str(
                        other.get("mirror_partner_plane_group_id") or ""
                    )
                    if not (
                        (
                            other_plane_group_id is not None
                            and row_partner_id == other_plane_group_id
                        )
                        or (
                            row_plane_group_id is not None
                            and other_partner_id == row_plane_group_id
                        )
                    ):
                        continue
                    local_roof_cover_parts.append(other_poly)
                    local_roof_cover_ids.append(
                        (
                            str(other["target_element_id"]),
                            float(overlap.area),
                        )
                    )
            if committed_cover_parts:
                try:
                    committed_union = unary_union(committed_cover_parts)
                    committed_cover_fraction = float(
                        poly.intersection(committed_union).area
                    ) / max(float(poly.area), 1e-9)
                except Exception:
                    committed_cover_fraction = 0.0
            else:
                committed_cover_fraction = 0.0
            if roof_surface_cover_parts:
                try:
                    roof_surface_union = unary_union(roof_surface_cover_parts)
                    roof_surface_cover_fraction = float(
                        poly.intersection(roof_surface_union).area
                    ) / max(float(poly.area), 1e-9)
                except Exception:
                    roof_surface_cover_fraction = 0.0
            else:
                roof_surface_cover_fraction = committed_cover_fraction
            if local_roof_cover_parts:
                try:
                    local_roof_union = unary_union(local_roof_cover_parts)
                    local_roof_cover_fraction = float(
                        poly.intersection(local_roof_union).area
                    ) / max(float(poly.area), 1e-9)
                except Exception:
                    local_roof_cover_fraction = 0.0
            else:
                local_roof_cover_fraction = committed_cover_fraction
            if same_side_superset_cover_parts:
                try:
                    same_side_superset_union = unary_union(
                        same_side_superset_cover_parts
                    )
                    same_side_superset_cover_fraction = float(
                        poly.intersection(same_side_superset_union).area
                    ) / max(float(poly.area), 1e-9)
                except Exception:
                    same_side_superset_cover_fraction = 0.0
            else:
                same_side_superset_cover_fraction = 0.0
            committed_cover_ids.sort(key=lambda item: (-item[1], item[0]))
            roof_surface_cover_ids.sort(key=lambda item: (-item[1], item[0]))
            local_roof_cover_ids.sort(key=lambda item: (-item[1], item[0]))
            same_side_superset_cover_ids.sort(key=lambda item: (-item[1], item[0]))
            signature_id = _chain_signature_id(
                str(row.get("uuid") or ""),
                int(row.get("story") or 0),
                row.get("chain_ids"),
            )
            annotated["chain_signature_id"] = signature_id
            annotated["higher_priority_cover_fraction"] = round(cover_fraction, 6)
            annotated["higher_priority_covering_target_ids"] = [
                target_id for target_id, _area in covering_ids[:3]
            ]
            annotated["committed_cover_fraction"] = round(committed_cover_fraction, 6)
            annotated["committed_covering_target_ids"] = [
                target_id for target_id, _area in committed_cover_ids[:3]
            ]
            annotated["roof_surface_cover_fraction"] = round(
                roof_surface_cover_fraction, 6
            )
            annotated["roof_surface_covering_target_ids"] = [
                target_id for target_id, _area in roof_surface_cover_ids[:3]
            ]
            annotated["local_roof_cover_fraction"] = round(local_roof_cover_fraction, 6)
            annotated["local_roof_covering_target_ids"] = [
                target_id for target_id, _area in local_roof_cover_ids[:3]
            ]
            annotated["same_side_superset_cover_fraction"] = round(
                same_side_superset_cover_fraction, 6
            )
            annotated["same_side_superset_covering_target_ids"] = [
                target_id for target_id, _area in same_side_superset_cover_ids[:3]
            ]
            annotated["ownership_redundant"] = (
                cover_fraction >= 0.98
                or committed_cover_fraction
                >= RIDGE_EAVE_COMMITTED_COVER_REDUNDANT_MIN_FRACTION
                or roof_surface_cover_fraction
                >= RIDGE_EAVE_COMMITTED_COVER_REDUNDANT_MIN_FRACTION
                or local_roof_cover_fraction >= 0.98
                or same_side_superset_cover_fraction >= 0.98
            )
            annotated_rows.append(annotated)
    return annotated_rows


def clip_unpaired_ridge_run_supported_pieces_to_support_union(
    split_piece_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    rows_by_uuid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_piece_rows:
        row_copy = dict(row)
        grouped[(str(row["uuid"]), int(row["story"]))].append(row_copy)
        rows_by_uuid[str(row["uuid"])].append(row_copy)

    clipped_rows: list[dict[str, Any]] = []
    for (uuid, _story), story_rows in grouped.items():
        building_rows = rows_by_uuid[uuid]
        supported_polys: dict[str, Polygon] = {}
        for row in building_rows:
            if row.get("piece_role") != "supported":
                continue
            poly = _row_piece_polygon(row)
            if poly is None or poly.is_empty:
                continue
            supported_polys[str(row["piece_id"])] = poly

        for row in story_rows:
            row_kind = str(row.get("target_kind") or "")
            piece_role = str(row.get("piece_role") or "")
            face_run_role = str(row.get("face_run_role") or "")
            mirror_partner_plane_group_id = str(
                row.get("mirror_partner_plane_group_id") or ""
            )
            if (
                row_kind != "ridge_eave_plane_group"
                or piece_role != "supported"
                or face_run_role != "ridge_run"
                or mirror_partner_plane_group_id
            ):
                clipped_rows.append(dict(row))
                continue

            piece_id = str(row.get("piece_id") or "")
            piece_poly = supported_polys.get(piece_id)
            if piece_poly is None or piece_poly.is_empty:
                clipped_rows.append(dict(row))
                continue

            support_overlaps: list[Polygon] = []
            support_covering_target_ids: list[tuple[str, float]] = []
            for other in building_rows:
                if other.get("piece_role") != "supported":
                    continue
                if str(other.get("piece_id") or "") == piece_id:
                    continue
                other_kind = str(other.get("target_kind") or "")
                if other_kind not in {"committed_oblique", "candidate_oblique"}:
                    continue
                other_poly = supported_polys.get(str(other.get("piece_id") or ""))
                if other_poly is None or other_poly.is_empty:
                    continue
                try:
                    overlap = piece_poly.intersection(other_poly)
                except Exception:
                    continue
                if overlap.is_empty:
                    continue
                overlap_polys = [
                    poly
                    for poly in _iter_polygons(overlap)
                    if not poly.is_empty and float(poly.area) >= MIN_SPLIT_PIECE_AREA_M2
                ]
                if not overlap_polys:
                    continue
                support_overlaps.extend(overlap_polys)
                overlap_area = sum(float(poly.area) for poly in overlap_polys)
                support_covering_target_ids.append(
                    (
                        str(other.get("target_element_id") or ""),
                        overlap_area,
                    )
                )

            if not support_overlaps:
                clipped_rows.append(dict(row))
                continue

            try:
                support_union = unary_union(support_overlaps)
            except Exception:
                clipped_rows.append(dict(row))
                continue

            clipped_polys = _iter_polygons(support_union)
            if not clipped_polys:
                clipped_rows.append(dict(row))
                continue

            original_area = max(float(piece_poly.area), 1e-9)
            clipped_area = sum(float(poly.area) for poly in clipped_polys)
            if clipped_area >= original_area - 1e-6:
                clipped_rows.append(dict(row))
                continue

            support_covering_target_ids.sort(key=lambda item: (-item[1], item[0]))
            for poly_index, poly in enumerate(
                sorted(clipped_polys, key=lambda item: float(item.area), reverse=True)
            ):
                record = _row_piece_records_from_polygon(
                    row,
                    poly,
                    piece_id=piece_id
                    if poly_index == 0
                    else f"{piece_id}:{poly_index}",
                )
                if record is None:
                    continue
                record["support_clip_fraction"] = round(
                    clipped_area / original_area,
                    6,
                )
                record["support_clip_target_ids"] = [
                    target_id for target_id, _area in support_covering_target_ids[:3]
                ]
                clipped_rows.append(record)
    return clipped_rows


def classify_split_piece_final_layer(
    split_piece_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classify split pieces into final-vs-candidate diagnostic layers.

    Rules:
    * committed_oblique targets default to final, then one logical owner row is
      kept per target in a post-pass (no polygon union rewrite).
    * candidate_oblique targets are always candidate.
    * ridge_eave_plane_group rows are classified per supported piece first, then
      corrected at the target level when source-part overlap shows the kept side
      is inconsistent with the creator's owning mass.
    """
    classified_rows: list[dict[str, Any]] = []
    max_supported_area_by_target: dict[str, float] = {}
    for row in split_piece_rows:
        if str(row.get("piece_role") or "") != "supported":
            continue
        target_element_id = str(row.get("target_element_id") or "")
        if not target_element_id:
            continue
        area = _piece_numeric_value(row, "area_xz_m2", 0.0)
        prev = max_supported_area_by_target.get(target_element_id, 0.0)
        if area > prev:
            max_supported_area_by_target[target_element_id] = area

    for row in split_piece_rows:
        target_kind = str(row.get("target_kind") or "")
        piece_role = str(row.get("piece_role") or "")
        out = dict(row)
        if target_kind == "committed_oblique":
            if bool(row.get("hypothesis_part_misaligned")):
                out["final_layer"] = False
                out["final_layer_reason"] = "committed_wrong_building_part"
                classified_rows.append(out)
                continue
            out["final_layer"] = True
            out["final_layer_reason"] = "committed_oblique"
            classified_rows.append(out)
            continue
        if target_kind == "candidate_oblique":
            out["final_layer"] = False
            out["final_layer_reason"] = "candidate_oblique"
            classified_rows.append(out)
            continue
        if target_kind == "ridge_eave_plane_group":
            creator_source_rooms = {
                str(room_id)
                for room_id in (row.get("creator_source_room_ids") or [])
                if str(room_id)
            }
            crossed_rooms = {
                str(room_id)
                for room_id in (row.get("crossed_room_ids") or [])
                if str(room_id)
            }
            source_crossed_disjoint = (
                bool(creator_source_rooms)
                and bool(crossed_rooms)
                and creator_source_rooms.isdisjoint(crossed_rooms)
            )
            provenance_reasons = {
                str(reason)
                for reason in (row.get("provenance_relevance_reasons") or [])
                if reason
            }
            strict_disjoint_extension_bypass = (
                source_crossed_disjoint
                and str(row.get("provenance_relevance_flag") or "")
                == "suspect_interior_slice"
                and int(row.get("creator_source_room_count") or 0) == 1
                and _piece_numeric_value(row, "through_ratio", 0.0) > 1.0
                and {
                    "weak_creator_rain_area",
                    "covered_creators_dominate",
                    "mostly_extended",
                    "cuts_below_top_story",
                }.issubset(provenance_reasons)
            )
            if piece_role != "supported" or (
                bool(row.get("ownership_redundant"))
                and not strict_disjoint_extension_bypass
            ):
                out["final_layer"] = False
                out["final_layer_reason"] = "ridge_eave_competitor_loss"
                classified_rows.append(out)
                continue
            competitor_loss = row.get("local_competitor_loss_fraction")
            owns_local_rooms = False
            if competitor_loss is not None:
                try:
                    owns_local_rooms = (
                        float(competitor_loss)
                        <= RIDGE_EAVE_FINAL_MAX_LOCAL_COMPETITOR_LOSS
                    )
                except (TypeError, ValueError):
                    owns_local_rooms = False
            mirror_score = row.get("mirror_support_score")
            anchored_suspect = (
                str(row.get("provenance_relevance_flag") or "")
                == "suspect_interior_slice"
                and mirror_score is not None
            )
            if anchored_suspect:
                try:
                    anchored_suspect = (
                        float(mirror_score)
                        >= RIDGE_EAVE_SUSPECT_ANCHOR_MIN_MIRROR_SCORE
                    )
                except (TypeError, ValueError):
                    anchored_suspect = False
            creator_touch_rooms = {
                str(room_id)
                for room_id in (row.get("creator_touch_room_ids") or [])
                if str(room_id)
            }
            crossed_rooms = {
                str(room_id)
                for room_id in (row.get("crossed_room_ids") or [])
                if str(room_id)
            }
            disqualifying_interior_slice = (
                str(row.get("provenance_relevance_flag") or "")
                == "suspect_interior_slice"
                and int(row.get("creator_source_room_count") or 0) == 1
                and mirror_score is None
                and {
                    "weak_creator_rain_area",
                    "covered_creators_dominate",
                    "cuts_below_top_story",
                    "unpaired",
                }.issubset(provenance_reasons)
            )
            disqualifying_creator_disconnected = (
                bool(creator_touch_rooms)
                and bool(crossed_rooms)
                and creator_touch_rooms.isdisjoint(crossed_rooms)
                and "cuts_below_top_story" in provenance_reasons
                and _piece_numeric_value(row, "through_ratio", 0.0) > 1.0
            )
            max_supported_area = max_supported_area_by_target.get(
                str(row.get("target_element_id") or ""),
                0.0,
            )
            area_fraction_of_target_max = (
                _piece_numeric_value(row, "area_xz_m2", 0.0)
                / max(max_supported_area, 1e-9)
                if max_supported_area > 1e-9
                else 0.0
            )
            is_mirror_sliver = (
                max_supported_area > 1e-9
                and area_fraction_of_target_max
                <= RIDGE_EAVE_MIRROR_SLIVER_MAX_AREA_FRACTION
                and _piece_numeric_value(row, "through_ratio", 0.0)
                >= RIDGE_EAVE_MIRROR_SLIVER_MIN_THROUGH_RATIO
                and mirror_score is not None
            )
            has_roof_cover_support = any(
                _piece_numeric_value(row, key, 0.0) > 0.0
                for key in (
                    "committed_cover_fraction",
                    "roof_surface_cover_fraction",
                    "local_roof_cover_fraction",
                )
            )
            out["final_layer"] = owns_local_rooms or anchored_suspect
            if is_mirror_sliver:
                out["final_layer"] = False
            if not has_roof_cover_support:
                out["final_layer"] = False
            if disqualifying_interior_slice:
                out["final_layer"] = False
            if disqualifying_creator_disconnected:
                out["final_layer"] = False
            out["final_layer_reason"] = (
                "ridge_eave_mirror_sliver"
                if is_mirror_sliver
                else "ridge_eave_no_roof_cover_support"
                if not has_roof_cover_support
                else "ridge_eave_suspect_interior_slice"
                if disqualifying_interior_slice
                else "ridge_eave_creator_disconnected"
                if disqualifying_creator_disconnected
                else "ridge_eave_local_ownership"
                if owns_local_rooms
                else "ridge_eave_suspect_anchor"
                if anchored_suspect
                else "ridge_eave_competitor_loss"
            )
            classified_rows.append(out)
            continue
        out["final_layer"] = False
        out["final_layer_reason"] = "unknown_target_kind"
        classified_rows.append(out)

    ridge_rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in classified_rows:
        if str(row.get("target_kind") or "") != "ridge_eave_plane_group":
            continue
        target_element_id = str(row.get("target_element_id") or "")
        if target_element_id:
            ridge_rows_by_target[target_element_id].append(row)

    for rows in ridge_rows_by_target.values():
        positive_overlap_rows = [
            row
            for row in rows
            if _piece_numeric_value(row, "creator_source_part_overlap_area_m2", 0.0)
            >= MIN_SPLIT_PIECE_AREA_M2
        ]
        if not positive_overlap_rows:
            continue
        final_rows = [row for row in rows if bool(row.get("final_layer"))]
        if not final_rows:
            continue
        bad_final_rows = [
            row
            for row in final_rows
            if _piece_numeric_value(row, "creator_source_part_overlap_area_m2", 0.0)
            < MIN_SPLIT_PIECE_AREA_M2
        ]
        if not bad_final_rows:
            continue
        for row in bad_final_rows:
            row["final_layer"] = False
            row["final_layer_reason"] = "ridge_eave_source_part_mismatch"

        has_positive_final = any(
            bool(row.get("final_layer"))
            and _piece_numeric_value(row, "creator_source_part_overlap_area_m2", 0.0)
            >= MIN_SPLIT_PIECE_AREA_M2
            for row in rows
        )
        if has_positive_final:
            continue

        promoted = max(
            positive_overlap_rows,
            key=lambda row: (
                _piece_numeric_value(row, "creator_source_part_overlap_area_m2", 0.0),
                _piece_numeric_value(row, "creator_source_room_overlap_area_m2", 0.0),
                1 if str(row.get("piece_role") or "") == "supported" else 0,
                _piece_numeric_value(row, "support_score", 0.0),
                _piece_numeric_value(row, "area_xz_m2", 0.0),
            ),
        )
        promoted["final_layer"] = True
        promoted["final_layer_reason"] = "ridge_eave_source_part_owner"

    committed_rows_by_target: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in classified_rows:
        if str(row.get("target_kind") or "") != "committed_oblique":
            continue
        if str(row.get("piece_role") or "") != "supported":
            continue
        target_element_id = str(row.get("target_element_id") or "")
        if not target_element_id:
            continue
        committed_rows_by_target[
            (str(row.get("uuid") or ""), target_element_id)
        ].append(row)

    for rows in committed_rows_by_target.values():
        final_rows = [row for row in rows if bool(row.get("final_layer"))]
        if len(final_rows) <= 1:
            continue
        owner_row = max(
            final_rows,
            key=lambda row: (
                _piece_numeric_value(row, "area_xz_m2", 0.0),
                _piece_numeric_value(row, "support_score", 0.0),
                -_piece_numeric_value(row, "higher_priority_cover_fraction", 0.0),
                str(row.get("piece_id") or ""),
            ),
        )
        for row in final_rows:
            if row is owner_row:
                continue
            row["final_layer"] = False
            row["final_layer_reason"] = "committed_union_demoted"
    return classified_rows


def prune_ridge_eave_rows_with_unreliable_mirror_pairs(
    split_piece_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop mirror-paired ridge/eave rows that fail high-confidence seam sanity.

    These rows are diagnostics only; when a row is declared mirror-supported but
    cannot resolve a plausible partner seam (or only overlaps a same-signature
    partner by a near-zero sliver), keeping the row produces obvious mirror
    overrun artifacts in the viewer.
    """
    supported_ridge_rows = [
        row
        for row in split_piece_rows
        if str(row.get("target_kind") or "") == "ridge_eave_plane_group"
        and str(row.get("piece_role") or "") == "supported"
    ]
    rows_by_plane_group_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in supported_ridge_rows:
        plane_group_id = _ridge_eave_target_to_plane_group_id(
            str(row.get("target_element_id") or "")
        )
        if plane_group_id:
            rows_by_plane_group_id[plane_group_id].append(row)

    pruned_rows: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    dropped_piece_ids: list[str] = []

    for row in split_piece_rows:
        if row not in supported_ridge_rows:
            pruned_rows.append(dict(row))
            continue

        row_piece_id = str(row.get("piece_id") or "")
        mirror_partner_plane_group_id = str(
            row.get("mirror_partner_plane_group_id") or ""
        )
        if not mirror_partner_plane_group_id:
            pruned_rows.append(dict(row))
            continue

        piece_poly = _row_piece_polygon(row)
        if piece_poly is None or piece_poly.is_empty:
            pruned_rows.append(dict(row))
            continue

        row_target_id = str(row.get("target_element_id") or "")
        partner_rows = [
            candidate
            for candidate in rows_by_plane_group_id.get(
                mirror_partner_plane_group_id, []
            )
            if str(candidate.get("target_element_id") or "") != row_target_id
        ]

        mirror_support_score = _piece_numeric_value(row, "mirror_support_score", 0.0)
        is_final_layer = bool(row.get("final_layer"))
        is_redundant_candidate = not is_final_layer and bool(
            row.get("ownership_redundant")
        )

        if not partner_rows:
            if (
                is_final_layer
                and mirror_support_score
                >= RIDGE_EAVE_MIRROR_PRUNE_FINAL_MIN_MIRROR_SCORE
            ):
                drop_reasons["final_missing_mirror_partner"] += 1
                dropped_piece_ids.append(row_piece_id)
                continue
            pruned_rows.append(dict(row))
            continue

        best_partner_row: dict[str, Any] | None = None
        best_partner_overlap_m2 = -1.0
        for partner_row in partner_rows:
            partner_poly = _row_piece_polygon(partner_row)
            if partner_poly is None or partner_poly.is_empty:
                continue
            try:
                overlap_m2 = float(piece_poly.intersection(partner_poly).area)
            except Exception:
                overlap_m2 = 0.0
            if overlap_m2 > best_partner_overlap_m2:
                best_partner_overlap_m2 = overlap_m2
                best_partner_row = partner_row

        if best_partner_row is None:
            if (
                is_final_layer
                and mirror_support_score
                >= RIDGE_EAVE_MIRROR_PRUNE_FINAL_MIN_MIRROR_SCORE
            ):
                drop_reasons["final_missing_mirror_partner_geometry"] += 1
                dropped_piece_ids.append(row_piece_id)
                continue
            pruned_rows.append(dict(row))
            continue

        best_partner_sig = str(best_partner_row.get("chain_signature_id") or "")
        row_sig = str(row.get("chain_signature_id") or "")
        partner_sig_matches = bool(row_sig) and row_sig == best_partner_sig
        through_ratio = _piece_numeric_value(row, "through_ratio", 0.0)

        if (
            is_final_layer
            and partner_sig_matches
            and best_partner_overlap_m2
            < RIDGE_EAVE_MIRROR_PRUNE_FINAL_MAX_PARTNER_OVERLAP_M2
            and mirror_support_score >= RIDGE_EAVE_MIRROR_PRUNE_FINAL_MIN_MIRROR_SCORE
            and through_ratio < RIDGE_EAVE_MIRROR_PRUNE_FINAL_MAX_THROUGH_RATIO
        ):
            drop_reasons["final_same_signature_tiny_partner_overlap"] += 1
            dropped_piece_ids.append(row_piece_id)
            continue

        if (
            is_redundant_candidate
            and not partner_sig_matches
            and best_partner_overlap_m2
            < RIDGE_EAVE_MIRROR_PRUNE_REDUNDANT_MAX_PARTNER_OVERLAP_M2
            and mirror_support_score
            >= RIDGE_EAVE_MIRROR_PRUNE_REDUNDANT_MIN_MIRROR_SCORE
        ):
            drop_reasons["redundant_mismatched_signature_tiny_partner_overlap"] += 1
            dropped_piece_ids.append(row_piece_id)
            continue

        pruned_rows.append(dict(row))

    return (
        pruned_rows,
        {
            "n_dropped_rows": int(sum(drop_reasons.values())),
            "drop_reason_counts": dict(drop_reasons),
            "dropped_piece_ids": dropped_piece_ids,
        },
    )


def score_target(
    target: TargetPlaneRecord,
    raw_records: list[RawPlaneRecord],
    raw_edges: list[RawEdgeRecord],
    conflicts: list[ConflictPairRecord],
    plane_chain_supports: list[PlaneEaveChainSupportRecord] | None = None,
) -> dict[str, Any]:
    story_raw = [
        record
        for record in raw_records
        if record.story == target.story and record.usable_for_support
    ]
    matches: list[RawPlaneRecord] = []
    for record in story_raw:
        try:
            overlap = target.poly_xz.intersection(record.poly_xz)
        except Exception:
            continue
        if overlap.is_empty or float(overlap.area) < MIN_MATCH_OVERLAP_M2:
            continue
        matches.append(record)

    raw_xz_coverage = 0.0
    raw_room_trust_mean = 0.0
    raw_normal_dot_p50: float | None = None
    raw_azimuth_delta_p50_deg: float | None = None
    raw_inclination_delta_p50_deg: float | None = None
    if matches and target.area_xz_m2 > 1e-9:
        try:
            raw_union = unary_union([record.poly_xz for record in matches])
            covered = target.poly_xz.intersection(raw_union)
            raw_xz_coverage = (
                float(covered.area) / target.area_xz_m2 if not covered.is_empty else 0.0
            )
        except Exception:
            raw_xz_coverage = 0.0
        raw_room_trust_mean = float(
            np.mean([record.room_trust_score for record in matches])
        )
        normal_dots = [
            abs(
                float(
                    np.dot(
                        _normalize_roof_up(target.normal),
                        _normalize_roof_up(record.normal),
                    )
                )
            )
            for record in matches
        ]
        raw_normal_dot_p50 = float(np.median(normal_dots)) if normal_dots else None
        azimuth_deltas = [
            delta
            for record in matches
            for delta in [
                _wrapped_angle_delta_deg(target.azimuth_deg, record.azimuth_deg)
            ]
            if delta is not None
        ]
        inclination_deltas = [
            abs(float(target.inclination_deg) - float(record.inclination_deg))
            for record in matches
        ]
        raw_azimuth_delta_p50_deg = (
            float(np.median(azimuth_deltas)) if azimuth_deltas else None
        )
        raw_inclination_delta_p50_deg = (
            float(np.median(inclination_deltas)) if inclination_deltas else None
        )

    ridge_edge_support_len_m = _ridge_edge_support_length(target, raw_edges)
    eave_edge_support_len_m = _eave_edge_support_length(target, raw_edges)
    conflicting_raw_pair_count = _conflicting_pair_count(target, conflicts)
    chain_supports = [
        support
        for support in (plane_chain_supports or [])
        if support.target_element_id == target.element_id
    ]
    supported_chain_supports = [
        support for support in chain_supports if support.supported
    ]
    best_chain = max(
        chain_supports, key=lambda support: support.support_score, default=None
    )

    orientation_support_score = (
        0.5 * float(raw_normal_dot_p50 or 0.0)
        + 0.3 * raw_room_trust_mean
        + 0.2 * raw_xz_coverage
    )
    retention_support_score = (
        0.4 * raw_xz_coverage
        + 0.2 * raw_room_trust_mean
        + 0.2 * min((ridge_edge_support_len_m + eave_edge_support_len_m) / 4.0, 1.0)
        + 0.2 * float(raw_normal_dot_p50 or 0.0)
        - 0.15 * min(conflicting_raw_pair_count / 3.0, 1.0)
    )
    orientation_flag, retention_flag, split_flag = _score_flags(
        orientation_support_score=orientation_support_score,
        retention_support_score=retention_support_score,
        conflicting_raw_pair_count=conflicting_raw_pair_count,
    )

    return {
        "uuid": target.uuid,
        "story": target.story,
        "target_kind": target.target_kind,
        "target_index": target.target_index,
        "element_id": target.element_id,
        "target_area_xz_m2": round(target.area_xz_m2, 6),
        "target_azimuth_deg": round(target.azimuth_deg, 6),
        "target_inclination_deg": round(target.inclination_deg, 6),
        "raw_match_count": len(matches),
        "raw_xz_coverage": round(raw_xz_coverage, 6),
        "raw_room_trust_mean": round(raw_room_trust_mean, 6),
        "raw_normal_dot_p50": round(raw_normal_dot_p50, 6)
        if raw_normal_dot_p50 is not None
        else None,
        "raw_azimuth_delta_p50_deg": round(raw_azimuth_delta_p50_deg, 6)
        if raw_azimuth_delta_p50_deg is not None
        else None,
        "raw_inclination_delta_p50_deg": round(raw_inclination_delta_p50_deg, 6)
        if raw_inclination_delta_p50_deg is not None
        else None,
        "ridge_edge_support_len_m": round(ridge_edge_support_len_m, 6),
        "eave_edge_support_len_m": round(eave_edge_support_len_m, 6),
        "supported_eave_chain_count": len(supported_chain_supports),
        "supported_eave_chain_total_length_m": round(
            float(sum(support.chain_length_m for support in supported_chain_supports)),
            6,
        ),
        "best_eave_chain_id": best_chain.chain_id if best_chain is not None else None,
        "best_eave_chain_support_score": round(best_chain.support_score, 6)
        if best_chain is not None
        else None,
        "conflicting_raw_pair_count": conflicting_raw_pair_count,
        "orientation_support_score": round(orientation_support_score, 6),
        "retention_support_score": round(retention_support_score, 6),
        "orientation_flag": orientation_flag,
        "retention_flag": retention_flag,
        "split_flag": split_flag,
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: _json_dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _summarize_story_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["uuid"]), int(row["story"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (uuid, story), story_rows in sorted(grouped.items()):
        candidate_rows = [
            row for row in story_rows if row["target_kind"] == "candidate_oblique"
        ]
        committed_rows = [
            row for row in story_rows if row["target_kind"] == "committed_oblique"
        ]
        top_retention = sorted(
            story_rows,
            key=lambda row: (
                float(row["retention_support_score"]),
                str(row["element_id"]),
            ),
            reverse=True,
        )[:3]
        top_orientation = sorted(
            story_rows,
            key=lambda row: (
                float(row["orientation_support_score"]),
                str(row["element_id"]),
            ),
            reverse=True,
        )[:3]
        summary_rows.append(
            {
                "uuid": uuid,
                "story": story,
                "candidate_count": len(candidate_rows),
                "committed_count": len(committed_rows),
                "split_target_count": sum(
                    1 for row in story_rows if bool(row["split_flag"])
                ),
                "top_retention_targets": [
                    {
                        "element_id": row["element_id"],
                        "target_kind": row["target_kind"],
                        "retention_support_score": row["retention_support_score"],
                    }
                    for row in top_retention
                ],
                "top_orientation_targets": [
                    {
                        "element_id": row["element_id"],
                        "target_kind": row["target_kind"],
                        "orientation_support_score": row["orientation_support_score"],
                    }
                    for row in top_orientation
                ],
            }
        )
    return summary_rows


def _summary_payload(
    rows: list[dict[str, Any]], per_story_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    orientation_counts = Counter(row["orientation_flag"] for row in rows)
    retention_counts = Counter(row["retention_flag"] for row in rows)
    split_counts = Counter("true" if row["split_flag"] else "false" for row in rows)

    split_by_building = Counter(row["uuid"] for row in rows if row["split_flag"])

    candidate_vs_committed_by_building: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    rows_by_story: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_story[(str(row["uuid"]), int(row["story"]))].append(row)

    for (uuid, story), story_rows in rows_by_story.items():
        candidate_rows = [
            row for row in story_rows if row["target_kind"] == "candidate_oblique"
        ]
        committed_rows = [
            row for row in story_rows if row["target_kind"] == "committed_oblique"
        ]
        if not candidate_rows or not committed_rows:
            continue
        candidate_mean = float(
            np.mean([float(row["retention_support_score"]) for row in candidate_rows])
        )
        committed_mean = float(
            np.mean([float(row["retention_support_score"]) for row in committed_rows])
        )
        if committed_mean < candidate_mean:
            candidate_vs_committed_by_building[uuid].append(
                {
                    "story": story,
                    "candidate_mean_retention_support_score": round(candidate_mean, 6),
                    "committed_mean_retention_support_score": round(committed_mean, 6),
                    "gap": round(candidate_mean - committed_mean, 6),
                }
            )

    return {
        "thresholds": {
            "trusted_dist_m": TRUSTED_DIST_M,
            "wall_top_tol_m": WALL_TOP_TOL_M,
            "horizontal_dy_m": HORIZONTAL_DY_M,
            "ridge_min_angle_deg": RIDGE_MIN_ANGLE_DEG,
            "ridge_match_xz_tol_m": RIDGE_MATCH_XZ_TOL_M,
            "ridge_match_y_tol_m": RIDGE_MATCH_Y_TOL_M,
            "eave_fp_tol_m": EAVE_FP_TOL_M,
            "min_raw_area_m2": MIN_RAW_AREA_M2,
            "max_raw_inclination_deg": MAX_RAW_INCLINATION_DEG,
            "min_room_trust_score": MIN_ROOM_TRUST_SCORE,
            "low_trust_promotion_min_overlap_m2": LOW_TRUST_PROMOTION_MIN_OVERLAP_M2,
            "low_trust_promotion_min_raw_overlap_fraction": (
                LOW_TRUST_PROMOTION_MIN_RAW_OVERLAP_FRACTION
            ),
            "low_trust_promotion_min_normal_dot": LOW_TRUST_PROMOTION_MIN_NORMAL_DOT,
            "low_trust_promotion_max_height_residual_m": (
                LOW_TRUST_PROMOTION_MAX_HEIGHT_RESIDUAL_M
            ),
            "min_gap_continuation_overlap_m2": MIN_GAP_CONTINUATION_OVERLAP_M2,
            "min_gap_continuation_overlap_fraction": (
                MIN_GAP_CONTINUATION_OVERLAP_FRACTION
            ),
            "ridge_eave_segment_anchor_buffer_m": RIDGE_EAVE_SEGMENT_ANCHOR_BUFFER_M,
            "ridge_eave_room_ownership_buffer_m": RIDGE_EAVE_ROOM_OWNERSHIP_BUFFER_M,
            "min_match_overlap_m2": MIN_MATCH_OVERLAP_M2,
            "min_conflict_overlap_m2": MIN_CONFLICT_OVERLAP_M2,
            "min_conflict_angle_deg": MIN_CONFLICT_ANGLE_DEG,
            "edge_alignment_tol_deg": EDGE_ALIGNMENT_TOL_DEG,
            "eave_target_boundary_tol_m": EAVE_TARGET_BOUNDARY_TOL_M,
        },
        "counts": {
            "n_targets": len(rows),
            "n_candidate_targets": sum(
                1 for row in rows if row["target_kind"] == "candidate_oblique"
            ),
            "n_committed_targets": sum(
                1 for row in rows if row["target_kind"] == "committed_oblique"
            ),
            "n_stories": len(per_story_rows),
        },
        "orientation_flag_counts": dict(orientation_counts),
        "retention_flag_counts": dict(retention_counts),
        "split_flag_counts": dict(split_counts),
        "top_buildings_by_split_target_count": [
            {"uuid": uuid, "split_target_count": count}
            for uuid, count in split_by_building.most_common(20)
        ],
        "top_buildings_where_committed_scores_worse_than_candidate": [
            {
                "uuid": uuid,
                "story_count": len(stories),
                "mean_gap": round(
                    float(np.mean([float(entry["gap"]) for entry in stories])), 6
                ),
                "stories": sorted(
                    stories,
                    key=lambda entry: (-float(entry["gap"]), int(entry["story"])),
                ),
            }
            for uuid, stories in sorted(
                candidate_vs_committed_by_building.items(),
                key=lambda item: (
                    -len(item[1]),
                    -float(np.mean([float(entry["gap"]) for entry in item[1]])),
                    item[0],
                ),
            )[:20]
        ],
    }


def score_buildings(
    buildings: list[dict[str, Any]],
    roof_results: dict[str, Any],
    ridge_eave_scores_by_uuid: dict[str, dict[str, Any]] | None = None,
    v3_results_by_uuid: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    plane_chain_rows: list[dict[str, Any]] = []
    split_piece_rows: list[dict[str, Any]] = []
    ownership_rows: list[dict[str, Any]] = []
    face_run_seeds: list[FaceRunSeedRecord] = []
    for building in buildings:
        uuid = str(building["uuid"])
        roof_result = roof_results.get(uuid)
        if roof_result is None:
            continue
        story_extent_envelopes = build_story_extent_envelopes(building)
        story_gap_polygons = build_story_gap_polygons(building)
        targets = collect_target_plane_records(roof_result, uuid)
        ridge_eave_targets = collect_selected_ridge_eave_plane_group_targets(
            uuid,
            (ridge_eave_scores_by_uuid or {}).get(uuid),
            story_extent_envelopes,
        )
        ridge_eave_target_diagnostics = collect_ridge_eave_target_diagnostics(
            uuid,
            (ridge_eave_scores_by_uuid or {}).get(uuid),
            (v3_results_by_uuid or {}).get(uuid),
        )
        ridge_eave_target_anchor_masks = collect_ridge_eave_target_anchor_masks(
            uuid,
            (ridge_eave_scores_by_uuid or {}).get(uuid),
            building,
            (v3_results_by_uuid or {}).get(uuid),
            ridge_eave_target_diagnostics,
        )
        ridge_eave_targets = constrain_ridge_eave_targets_with_anchor_masks(
            ridge_eave_targets,
            ridge_eave_target_diagnostics,
            ridge_eave_target_anchor_masks,
        )
        target_face_runs, committed_target_metadata = build_target_face_runs(
            targets,
            ridge_eave_targets,
            building,
            roof_result,
        )
        face_run_seed_by_target_id = {
            target_id: face_run
            for face_run in target_face_runs
            for target_id in face_run.member_target_ids
        }
        target_face_annotations = _target_face_run_annotations(
            face_run_seed_by_target_id,
            committed_target_metadata,
        )
        face_run_seeds.extend(target_face_runs)
        split_targets = targets + ridge_eave_targets
        raw_records = collect_raw_plane_records(
            building, roof_result, exposed_only=True
        )
        source_room_keys = _source_room_keys_from_ridge_diagnostics(
            building,
            ridge_eave_target_diagnostics,
        )
        raw_records = _augment_raw_records_with_source_rooms(
            building,
            roof_result,
            raw_records,
            source_room_keys,
        )
        # Promotion must see split targets too; otherwise low-trust extension
        # planes never promote against ridge/eave-only targets.
        raw_records = promote_raw_plane_support_records(raw_records, split_targets)
        raw_edges = collect_raw_edges(raw_records, roof_result)
        conflicts = collect_conflict_pairs(raw_records)
        eave_chains = build_eave_chains(uuid, raw_edges)
        plane_chain_supports = expand_plane_eave_chain_supports_by_facade_continuity(
            eave_chains,
            score_plane_eave_chain_supports(split_targets, eave_chains),
        )
        split_pieces = build_plane_extent_split_pieces(
            split_targets,
            eave_chains,
            plane_chain_supports,
            story_extent_envelopes=story_extent_envelopes,
            story_gap_polygons=story_gap_polygons,
            target_segment_anchor_masks=ridge_eave_target_anchor_masks,
        )
        split_pieces = trim_ridge_eave_supported_pieces_to_chain_run_bands(
            split_targets,
            eave_chains,
            split_pieces,
        )
        split_target_score_rows = [
            score_target(
                target, raw_records, raw_edges, conflicts, plane_chain_supports
            )
            for target in split_targets
        ]
        split_target_score_rows = annotate_rows_with_target_face_runs(
            split_target_score_rows,
            target_face_annotations,
        )
        split_pieces = trim_ridge_eave_supported_pieces_to_room_ownership(
            building,
            split_targets,
            split_target_score_rows,
            split_pieces,
            building_part_graph=roof_result.get("building_part_graph") or {},
        )
        split_target_score_by_id = {
            str(row["element_id"]): row for row in split_target_score_rows
        }
        ownership_rows.extend(
            diagnose_ridge_eave_piece_ownership(
                building,
                (ridge_eave_scores_by_uuid or {}).get(uuid),
                split_targets,
                split_target_score_rows,
                split_pieces,
                plane_chain_supports,
                ridge_eave_target_diagnostics,
            )
        )
        chain_rows.extend(
            {
                "uuid": chain.uuid,
                "story": chain.story,
                "chain_id": chain.chain_id,
                "edge_count": chain.edge_count,
                "total_length_m": round(chain.total_length_m, 6),
                "azimuth_deg": round(chain.azimuth_deg, 6),
                "y_mean": round(chain.y_mean, 6),
                "start_xz": [round(chain.start_xz[0], 6), round(chain.start_xz[1], 6)],
                "end_xz": [round(chain.end_xz[0], 6), round(chain.end_xz[1], 6)],
                "member_plane_ids": list(chain.member_plane_ids),
            }
            for chain in eave_chains
        )
        plane_chain_rows.extend(
            {
                "uuid": support.uuid,
                "story": support.story,
                "target_element_id": support.target_element_id,
                "target_kind": support.target_kind,
                "chain_id": support.chain_id,
                "chain_azimuth_deg": round(support.chain_azimuth_deg, 6),
                "ridge_azimuth_deg": round(support.ridge_azimuth_deg, 6),
                "angle_delta_deg": round(support.angle_delta_deg, 6),
                "boundary_distance_m": round(support.boundary_distance_m, 6),
                "overlap_fraction": round(support.overlap_fraction, 6),
                "height_residual_m": round(support.height_residual_m, 6)
                if support.height_residual_m is not None
                else None,
                "support_score": round(support.support_score, 6),
                "supported": support.supported,
                "chain_length_m": round(support.chain_length_m, 6),
            }
            for support in plane_chain_supports
        )
        split_piece_rows.extend(
            {
                "uuid": piece.uuid,
                "story": piece.story,
                "target_element_id": piece.target_element_id,
                "target_kind": piece.target_kind,
                "piece_id": piece.piece_id,
                "piece_index": piece.piece_index,
                "piece_role": piece.piece_role,
                "area_xz_m2": round(piece.area_xz_m2, 6),
                "target_azimuth_deg": round(target.azimuth_deg, 6),
                "target_inclination_deg": round(target.inclination_deg, 6),
                "support_score": round(piece.support_score, 6)
                if piece.support_score is not None
                else None,
                "chain_ids": list(piece.chain_ids),
                "corners": _rounded_loop_3d(piece.corners),
                "holes": _serialized_piece_holes(piece.holes),
                **ridge_eave_target_diagnostics.get(piece.target_element_id, {}),
            }
            for piece in split_pieces
            for target in [
                next(
                    t for t in split_targets if t.element_id == piece.target_element_id
                )
            ]
        )
        if split_pieces:
            split_piece_rows[-len(split_pieces) :] = (
                annotate_rows_with_target_face_runs(
                    split_piece_rows[-len(split_pieces) :],
                    target_face_annotations,
                )
            )
        rows.extend(
            split_target_score_by_id[str(target.element_id)] for target in targets
        )
    per_story_rows = _summarize_story_rows(rows)
    summary = _summary_payload(rows, per_story_rows)
    summary["counts"]["n_eave_chains"] = len(chain_rows)
    summary["counts"]["n_plane_eave_chain_pairs"] = len(plane_chain_rows)
    summary["counts"]["n_supported_plane_eave_chain_pairs"] = sum(
        1 for row in plane_chain_rows if row["supported"]
    )
    summary["counts"]["n_plane_extent_split_pieces"] = len(split_piece_rows)
    summary["counts"]["n_targets_with_plane_extent_splits"] = len(
        {str(row["target_element_id"]) for row in split_piece_rows}
    )
    summary["counts"]["n_suspect_interior_slice_split_pieces"] = sum(
        1
        for row in split_piece_rows
        if row.get("provenance_relevance_flag") == "suspect_interior_slice"
    )
    summary["counts"]["n_suspect_interior_slice_targets"] = len(
        {
            str(row["target_element_id"])
            for row in split_piece_rows
            if row.get("provenance_relevance_flag") == "suspect_interior_slice"
        }
    )
    summary["counts"]["n_ridge_eave_supported_piece_ownership_rows"] = len(
        ownership_rows
    )
    summary["counts"]["n_ridge_eave_piece_competitor_losses"] = sum(
        1
        for row in ownership_rows
        if float(row.get("local_competitor_loss_area_m2") or 0.0) > 0.0
    )
    buildings_by_uuid = {
        str(building.get("uuid")): building
        for building in buildings
        if building.get("uuid")
    }
    split_piece_rows = merge_split_piece_rows_with_ownership(
        split_piece_rows, ownership_rows
    )
    split_piece_rows = annotate_committed_supported_pieces_with_hypothesis_part_overlap(
        split_piece_rows,
        buildings_by_uuid,
        roof_results,
    )
    split_piece_rows = resolve_split_piece_rows_with_target_face_runs(
        split_piece_rows, face_run_seeds
    )
    split_piece_rows = trim_ridge_eave_rows_to_local_mirror_pieces(split_piece_rows)
    split_piece_rows = annotate_ridge_eave_rows_with_creator_source_overlap(
        split_piece_rows,
        buildings_by_uuid,
        roof_results,
    )
    split_piece_rows = clip_unpaired_ridge_run_supported_pieces_to_support_union(
        split_piece_rows
    )
    split_piece_rows = annotate_split_piece_rows_with_precedence(split_piece_rows)
    split_piece_rows = classify_split_piece_final_layer(split_piece_rows)
    split_piece_rows, mirror_prune_diag = (
        prune_ridge_eave_rows_with_unreliable_mirror_pairs(split_piece_rows)
    )
    summary["counts"]["n_face_runs"] = len(face_run_seeds)
    summary["counts"]["n_face_runs_with_committed_core"] = sum(
        1 for face_run in face_run_seeds if face_run.core_committed_target_ids
    )
    summary["counts"]["n_face_runs_with_ridge_continuation"] = sum(
        1 for face_run in face_run_seeds if face_run.ridge_target_ids
    )
    summary["counts"]["n_plane_extent_split_pieces_final"] = sum(
        1 for row in split_piece_rows if bool(row.get("final_layer"))
    )
    summary["counts"]["n_plane_extent_split_pieces_candidate"] = sum(
        1 for row in split_piece_rows if not bool(row.get("final_layer"))
    )
    summary["counts"]["n_plane_extent_split_pieces_mirror_pruned"] = int(
        mirror_prune_diag.get("n_dropped_rows") or 0
    )
    summary["mirror_prune"] = {
        "drop_reason_counts": mirror_prune_diag.get("drop_reason_counts") or {},
    }
    return (
        rows,
        per_story_rows,
        chain_rows,
        plane_chain_rows,
        split_piece_rows,
        ownership_rows,
        summary,
    )


def _write_v1_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    per_story_rows: list[dict[str, Any]],
    chain_rows: list[dict[str, Any]],
    plane_chain_rows: list[dict[str, Any]],
    split_piece_rows: list[dict[str, Any]],
    ownership_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_target.csv", rows)
    _write_csv(out_dir / "per_story.csv", per_story_rows)
    _write_csv(out_dir / "eave_chains.csv", chain_rows)
    _write_csv(out_dir / "plane_eave_support.csv", plane_chain_rows)
    _write_csv(out_dir / "plane_extent_splits.csv", split_piece_rows)
    _write_csv(out_dir / "ridge_eave_piece_ownership.csv", ownership_rows)
    (out_dir / "per_target.json").write_text(
        json.dumps({"rows": rows}, indent=2),
        encoding="utf-8",
    )
    split_payload = {
        "available": bool(split_piece_rows),
        "buildings": defaultdict(list),
    }
    for row in split_piece_rows:
        split_payload["buildings"][str(row["uuid"])].append(row)
    split_payload["buildings"] = dict(split_payload["buildings"])
    (out_dir / "plane_extent_splits.json").write_text(
        json.dumps(split_payload, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "ridge_eave_piece_ownership.json").write_text(
        json.dumps({"rows": ownership_rows}, indent=2),
        encoding="utf-8",
    )


def _shadow_diff_payload(
    v1_rows: list[dict[str, Any]],
    v1_split_piece_rows: list[dict[str, Any]],
    v1_summary: dict[str, Any],
    v2_output: Any,
) -> dict[str, Any]:
    v2_rows = list(v2_output.per_target_rows)
    v2_split_piece_rows = list(v2_output.split_piece_rows)
    v2_summary = dict(v2_output.summary)

    v1_target_fields = sorted({key for row in v1_rows for key in row.keys()})
    v2_target_fields = sorted({key for row in v2_rows for key in row.keys()})
    v1_split_fields = sorted({key for row in v1_split_piece_rows for key in row.keys()})
    v2_split_fields = sorted({key for row in v2_split_piece_rows for key in row.keys()})

    return {
        "counts": {
            "v1_targets": len(v1_rows),
            "v2_targets": len(v2_rows),
            "v1_split_pieces": len(v1_split_piece_rows),
            "v2_split_pieces": len(v2_split_piece_rows),
            "v1_final_split_pieces": sum(
                1 for row in v1_split_piece_rows if bool(row.get("final_layer"))
            ),
            "v2_final_split_pieces": sum(
                1 for row in v2_split_piece_rows if bool(row.get("final_layer"))
            ),
        },
        "schema": {
            "v1_only_target_fields": sorted(
                set(v1_target_fields) - set(v2_target_fields)
            ),
            "v2_only_target_fields": sorted(
                set(v2_target_fields) - set(v1_target_fields)
            ),
            "v1_only_split_fields": sorted(set(v1_split_fields) - set(v2_split_fields)),
            "v2_only_split_fields": sorted(set(v2_split_fields) - set(v1_split_fields)),
        },
        "summary_counts": {
            "v1": dict(v1_summary.get("counts") or {}),
            "v2": dict(v2_summary.get("counts") or {}),
        },
        "summary_relation_kind_counts_v2": dict(
            v2_summary.get("relation_kind_counts") or {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", help="Score a single building UUID")
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--roof-results", type=Path, default=ROOF_RESULTS_PATH)
    parser.add_argument("--v3-results", type=Path, default=V3_RESULTS_PATH)
    parser.add_argument(
        "--ridge-eave-scores", type=Path, default=RIDGE_EAVE_SCORES_PATH
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--engine",
        choices=["v1", "v2", "shadow"],
        default="shadow",
        help="Scoring engine to run. shadow runs both and writes a diff payload.",
    )
    args = parser.parse_args()

    with args.buildings.open() as handle:
        buildings = json.load(handle)
    with args.roof_results.open() as handle:
        roof_results = json.load(handle)
    ridge_eave_scores_by_uuid: dict[str, dict[str, Any]] = {}
    if args.ridge_eave_scores.exists():
        with args.ridge_eave_scores.open() as handle:
            ridge_eave_payload = json.load(handle)
        ridge_eave_scores_by_uuid = {
            str(entry.get("building_uuid")): entry
            for entry in (ridge_eave_payload.get("buildings") or [])
            if entry.get("building_uuid")
        }
    v3_results_by_uuid: dict[str, dict[str, Any]] = {}
    if args.v3_results.exists():
        with args.v3_results.open() as handle:
            v3_results_payload = json.load(handle)
        v3_results_by_uuid = {
            str(entry.get("building_uuid")): entry
            for entry in v3_results_payload
            if entry.get("building_uuid")
        }

    if args.uuid:
        buildings = [
            building for building in buildings if str(building.get("uuid")) == args.uuid
        ]
        if not buildings:
            raise SystemExit(f"No building found for UUID {args.uuid}")

    if args.engine == "v1":
        (
            rows,
            per_story_rows,
            chain_rows,
            plane_chain_rows,
            split_piece_rows,
            ownership_rows,
            summary,
        ) = score_buildings(
            buildings,
            roof_results,
            ridge_eave_scores_by_uuid=ridge_eave_scores_by_uuid,
            v3_results_by_uuid=v3_results_by_uuid,
        )
        _write_v1_outputs(
            args.out_dir,
            rows,
            per_story_rows,
            chain_rows,
            plane_chain_rows,
            split_piece_rows,
            ownership_rows,
            summary,
        )
        print(f"Targets scored: {len(rows)}")
        print(f"Stories summarized: {len(per_story_rows)}")
        print(f"Output: {args.out_dir}")
        return

    if args.engine == "v2":
        try:
            from scripts.raw_ceiling_plane_scorer_v2.reporting import write_outputs
            from scripts.raw_ceiling_plane_scorer_v2.runner import score_corpus_v2
        except ModuleNotFoundError:
            import sys

            sys.path.insert(0, str(REPO))
            from raw_ceiling_plane_scorer_v2.reporting import write_outputs
            from raw_ceiling_plane_scorer_v2.runner import score_corpus_v2

        output = score_corpus_v2(
            buildings,
            roof_results,
            ridge_eave_scores_by_uuid=ridge_eave_scores_by_uuid,
            v3_results_by_uuid=v3_results_by_uuid,
        )
        write_outputs(args.out_dir, output)
        print(f"Targets scored: {len(output.per_target_rows)}")
        print(f"Stories summarized: {len(output.per_story_rows)}")
        print(f"Output: {args.out_dir}")
        return

    # shadow mode
    try:
        from scripts.raw_ceiling_plane_scorer_v2.reporting import write_outputs
        from scripts.raw_ceiling_plane_scorer_v2.runner import score_corpus_v2
    except ModuleNotFoundError:
        import sys

        sys.path.insert(0, str(REPO))
        from raw_ceiling_plane_scorer_v2.reporting import write_outputs
        from raw_ceiling_plane_scorer_v2.runner import score_corpus_v2

    (
        v1_rows,
        v1_per_story_rows,
        v1_chain_rows,
        v1_plane_chain_rows,
        v1_split_piece_rows,
        v1_ownership_rows,
        v1_summary,
    ) = score_buildings(
        buildings,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_eave_scores_by_uuid,
        v3_results_by_uuid=v3_results_by_uuid,
    )
    v2_output = score_corpus_v2(
        buildings,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_eave_scores_by_uuid,
        v3_results_by_uuid=v3_results_by_uuid,
    )

    v1_out_dir = args.out_dir / "v1"
    v2_out_dir = args.out_dir / "v2"
    _write_v1_outputs(
        v1_out_dir,
        v1_rows,
        v1_per_story_rows,
        v1_chain_rows,
        v1_plane_chain_rows,
        v1_split_piece_rows,
        v1_ownership_rows,
        v1_summary,
    )
    write_outputs(v2_out_dir, v2_output)
    shadow_diff = _shadow_diff_payload(
        v1_rows=v1_rows,
        v1_split_piece_rows=v1_split_piece_rows,
        v1_summary=v1_summary,
        v2_output=v2_output,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "shadow_diff.json").write_text(
        json.dumps(shadow_diff, indent=2),
        encoding="utf-8",
    )
    print(f"V1 targets scored: {len(v1_rows)}")
    print(f"V2 targets scored: {len(v2_output.per_target_rows)}")
    print(f"Output: {args.out_dir} (v1/, v2/, shadow_diff.json)")


if __name__ == "__main__":
    main()
