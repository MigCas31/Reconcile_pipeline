"""Realism check on a LoD2 reconstruction OBJ via trimesh.

Generic mesh stats are noise. The questions we actually want to answer are:
"is this LoD2 a plausible building?" and, if it is not watertight,
"where are the holes?"

Output is a small set of pass/warn/fail flags, each with the value that drove
it and the threshold it was checked against. A 'flags' list at the top
collects the names of failures so summarize.py can render a tight section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import MultiPoint

# --- Plausibility thresholds (tuned for Danish single-family / townhouse) ---

VOLUME_MIN_M3 = 30.0  # smaller is implausible for a habitable building
VOLUME_MAX_M3 = 5000.0  # larger than a small apartment block is suspicious
ASPECT_MAX_HEIGHT_RATIO = 1.5  # height / max(width, depth) > 1.5 = flagpole
CONVEXITY_MIN = 0.35  # below this, shape is implausibly carved
CONVEXITY_MAX = 0.99  # above this, it is indistinguishable from a box
WALL_PCT_MIN = 0.20  # walls should dominate face area; below this = no walls
TYPICAL_STORY_M = 3.0  # for height -> story count comparison
STORY_COUNT_TOL = 1  # off by this many is acceptable
FOOTPRINT_DELTA_MAX = 0.30  # |lod2 - tier ground sum| / lod2 > 30% = mismatch

# Face-normal binning. Y is up.
NORMAL_UP = 0.85  # ny >= this -> "ground" or roof flat (depending on direction)
NORMAL_VERTICAL_MAX = 0.20  # |ny| <= this -> wall


def _flag(out: dict, name: str, level: str, value, threshold, reason: str) -> None:
    """Record a realism flag.

    level is 'pass' | 'warn' | 'fail'. Failures land in out['flags'] for
    summarize.py to display at the top of the report.
    """
    out["flags_detail"].append(
        {
            "name": name,
            "level": level,
            "value": value,
            "threshold": threshold,
            "reason": reason,
        }
    )
    if level == "fail":
        out["flags"].append(name)


def _hole_loops(mesh: trimesh.Trimesh) -> list[dict]:
    """Locate the boundary loops on a non-watertight mesh.

    Each loop = one missing patch. Reported as: vertex count, perimeter,
    centroid, bbox, and a heuristic location label (roof / ground / wall-N/S/E/W
    / interior) based on where the loop sits relative to the bbox.
    """
    try:
        outline = mesh.outline()
    except Exception:
        return []
    if not hasattr(outline, "entities"):
        return []
    loops: list[dict] = []
    bounds = mesh.bounds
    bbox_min, bbox_max = bounds[0], bounds[1]
    extents = bbox_max - bbox_min
    for entity in outline.entities:
        try:
            verts = outline.vertices[entity.points]
        except Exception:
            continue
        if len(verts) < 2:
            continue
        diffs = np.diff(verts, axis=0)
        perim = float(np.linalg.norm(diffs, axis=1).sum())
        centroid = verts.mean(axis=0)
        loop_min = verts.min(axis=0)
        loop_max = verts.max(axis=0)
        # Where on the building? Use centroid position relative to bbox.
        cy_rel = (centroid[1] - bbox_min[1]) / max(extents[1], 1e-9)
        cx_rel = (centroid[0] - bbox_min[0]) / max(extents[0], 1e-9)
        cz_rel = (centroid[2] - bbox_min[2]) / max(extents[2], 1e-9)
        if cy_rel > 0.85:
            location = "roof"
        elif cy_rel < 0.15:
            location = "ground"
        else:
            # Pick the dominant horizontal side.
            sides = {
                "wall-east": cx_rel,
                "wall-west": 1.0 - cx_rel,
                "wall-south": cz_rel,
                "wall-north": 1.0 - cz_rel,
            }
            label, score = max(sides.items(), key=lambda kv: kv[1])
            location = label if score > 0.65 else "wall-interior"
        loops.append(
            {
                "vertex_count": len(verts),
                "perimeter_m": perim,
                "centroid": [float(c) for c in centroid],
                "bbox_min": [float(c) for c in loop_min],
                "bbox_max": [float(c) for c in loop_max],
                "location": location,
            }
        )
    loops.sort(key=lambda L: -L["perimeter_m"])
    return loops


def _face_normal_distribution(mesh: trimesh.Trimesh) -> dict:
    normals = mesh.face_normals
    areas = mesh.area_faces
    total = float(areas.sum()) or 1.0
    ny = normals[:, 1]
    # ground: faces pointing strongly down (these face the floor)
    ground_mask = ny <= -NORMAL_UP
    # roof flat: faces pointing strongly up
    roof_flat_mask = ny >= NORMAL_UP
    # walls: nearly horizontal normals
    wall_mask = np.abs(ny) <= NORMAL_VERTICAL_MAX
    # everything else is roof oblique (sloped surface)
    oblique_mask = ~(ground_mask | roof_flat_mask | wall_mask)
    return {
        "wall_pct": float(areas[wall_mask].sum() / total),
        "roof_flat_pct": float(areas[roof_flat_mask].sum() / total),
        "roof_oblique_pct": float(areas[oblique_mask].sum() / total),
        "ground_pct": float(areas[ground_mask].sum() / total),
    }


def _ground_face_area_xz(mesh: trimesh.Trimesh) -> float:
    """Project downward-facing faces onto XZ and sum their projected area.

    Approximates the building's ground footprint as Lod2 sees it.
    """
    normals = mesh.face_normals
    areas = mesh.area_faces
    ground_mask = normals[:, 1] <= -NORMAL_UP
    if not ground_mask.any():
        return 0.0
    # Projected area on XZ ≈ area * |ny|. For ny ≈ -1, projection equals area.
    return float((areas[ground_mask] * np.abs(normals[ground_mask, 1])).sum())


def _ground_room_floor_sum(tier_payload: dict | None) -> float | None:
    """Sum of ground-story room floor areas, projected to XZ."""
    if not tier_payload:
        return None
    rooms = tier_payload.get("rooms") or []
    if not rooms:
        return None
    stories = [r.get("story") for r in rooms if r.get("story") is not None]
    if not stories:
        return None
    ground_story = min(stories)
    total = 0.0
    for room in rooms:
        if room.get("story") != ground_story:
            continue
        floor = room.get("floor") or []
        floor_pieces = floor if isinstance(floor, list) else [floor]
        for piece in floor_pieces:
            corners = (piece or {}).get("corners") or []
            if len(corners) < 3:
                continue
            pts = [(c.get("x", 0.0), c.get("z", 0.0)) for c in corners]
            area = 0.0
            for i in range(len(pts)):
                x1, z1 = pts[i]
                x2, z2 = pts[(i + 1) % len(pts)]
                area += x1 * z2 - x2 * z1
            total += abs(area) * 0.5
    return total


def _detect_up_axis(mesh: trimesh.Trimesh) -> int:
    """Index of the axis (0=X, 1=Y, 2=Z) along which the mesh is oriented up.

    For Danish single-family / townhouse buildings (the scope of these
    realism thresholds) the up axis is reliably the smallest extent -- typical
    homes are 4-10 m tall and 8-25 m wide. CityGML / LoD2 reconstructions are
    Z-up by convention; the rest of this realism check assumes Y-up.
    """
    extents = mesh.bounds[1] - mesh.bounds[0]
    return int(np.argmin(extents))


def _rotate_to_y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    up = _detect_up_axis(mesh)
    if up == 1:
        return mesh
    if up == 2:
        # Z-up -> Y-up: 90 deg rotation about +X axis. (x, y, z) -> (x, z, -y).
        transform = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=float,
        )
    else:
        # X-up -> Y-up: 90 deg rotation about +Z axis. (x, y, z) -> (-y, x, z).
        transform = np.array(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
    rotated = mesh.copy()
    rotated.apply_transform(transform)
    return rotated


def compute_realism(obj_path: Path, tier_payload: dict | None) -> dict:
    mesh = trimesh.load(obj_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a single mesh: {type(mesh)}")
    mesh = _rotate_to_y_up(mesh)

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    width, height_m, depth = float(extents[0]), float(extents[1]), float(extents[2])

    out: dict = {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "extents": {"x": width, "y": height_m, "z": depth},
        "flags": [],
        "flags_detail": [],
    }

    # 1. Watertight + connected components
    is_watertight = bool(mesh.is_watertight)
    out["is_watertight"] = is_watertight
    out["is_winding_consistent"] = bool(mesh.is_winding_consistent)
    if not is_watertight:
        out["holes"] = _hole_loops(mesh)
        _flag(
            out,
            "not_watertight",
            "fail",
            value={"hole_count": len(out["holes"])},
            threshold="watertight",
            reason=(
                f"Mesh has {len(out['holes'])} boundary loop(s); "
                "real building shells are sealed."
            ),
        )
    else:
        out["holes"] = []
        _flag(out, "not_watertight", "pass", True, "watertight", "Sealed shell.")

    components = list(mesh.split(only_watertight=False))
    out["component_count"] = len(components)
    if len(components) > 1:
        _flag(
            out,
            "multiple_components",
            "fail",
            value=len(components),
            threshold=1,
            reason=(f"Mesh has {len(components)} disconnected pieces; expected 1."),
        )
    else:
        _flag(out, "multiple_components", "pass", 1, 1, "Single connected piece.")

    # 2. Volume + convex hull (only meaningful if watertight)
    out["volume_m3"] = float(mesh.volume) if is_watertight else None
    try:
        hull = mesh.convex_hull
        out["convex_hull_volume_m3"] = float(hull.volume)
    except Exception as exc:
        out["convex_hull_volume_m3"] = None
        out["convex_hull_error"] = str(exc)

    if out["volume_m3"] is not None:
        v = out["volume_m3"]
        if v < VOLUME_MIN_M3:
            _flag(
                out,
                "implausible_volume",
                "fail",
                v,
                f"[{VOLUME_MIN_M3}, {VOLUME_MAX_M3}]",
                f"Volume {v:.0f} m^3 is below {VOLUME_MIN_M3} m^3 -- too small to be a "
                f"building.",
            )
        elif v > VOLUME_MAX_M3:
            _flag(
                out,
                "implausible_volume",
                "warn",
                v,
                f"[{VOLUME_MIN_M3}, {VOLUME_MAX_M3}]",
                f"Volume {v:.0f} m^3 exceeds {VOLUME_MAX_M3} m^3 -- large for a single "
                f"home.",
            )
        else:
            _flag(
                out,
                "implausible_volume",
                "pass",
                v,
                f"[{VOLUME_MIN_M3}, {VOLUME_MAX_M3}]",
                "Plausible volume.",
            )

    if out.get("convex_hull_volume_m3") and out.get("volume_m3"):
        ratio = out["volume_m3"] / out["convex_hull_volume_m3"]
        out["convexity_ratio"] = ratio
        if ratio < CONVEXITY_MIN:
            _flag(
                out,
                "extreme_convexity",
                "fail",
                ratio,
                f"[{CONVEXITY_MIN}, {CONVEXITY_MAX}]",
                f"Convexity {ratio:.2f} below {CONVEXITY_MIN} -- bizarre carved shape.",
            )
        elif ratio > CONVEXITY_MAX:
            _flag(
                out,
                "extreme_convexity",
                "warn",
                ratio,
                f"[{CONVEXITY_MIN}, {CONVEXITY_MAX}]",
                f"Convexity {ratio:.2f} above {CONVEXITY_MAX} -- "
                f"indistinguishable from a box.",
            )
        else:
            _flag(
                out,
                "extreme_convexity",
                "pass",
                ratio,
                f"[{CONVEXITY_MIN}, {CONVEXITY_MAX}]",
                "Plausible convexity.",
            )

    # 3. Aspect ratio (Y is up)
    horiz_max = max(width, depth, 1e-6)
    aspect = height_m / horiz_max
    out["aspect_ratio"] = aspect
    if aspect > ASPECT_MAX_HEIGHT_RATIO:
        _flag(
            out,
            "implausible_aspect",
            "fail",
            aspect,
            ASPECT_MAX_HEIGHT_RATIO,
            f"Height {height_m:.1f} m vs footprint {horiz_max:.1f} m -> flagpole "
            f"shape.",
        )
    else:
        _flag(
            out,
            "implausible_aspect",
            "pass",
            aspect,
            ASPECT_MAX_HEIGHT_RATIO,
            "Plausible aspect.",
        )

    # 4. Face-normal distribution
    dist = _face_normal_distribution(mesh)
    out["normal_distribution"] = dist
    if dist["wall_pct"] < WALL_PCT_MIN:
        _flag(
            out,
            "no_walls",
            "fail",
            dist["wall_pct"],
            WALL_PCT_MIN,
            f"Only {dist['wall_pct'] * 100:.0f}% of face area is wall-like -- "
            f"reconstruction has no walls.",
        )
    else:
        _flag(out, "no_walls", "pass", dist["wall_pct"], WALL_PCT_MIN, "Walls present.")

    # 5. Implied story count vs classification
    implied_stories = round(height_m / TYPICAL_STORY_M)
    out["implied_story_count"] = implied_stories
    classified_stories = None
    if tier_payload:
        classified_stories = (tier_payload.get("classification") or {}).get("n_stories")
    out["classified_story_count"] = classified_stories
    if classified_stories is not None:
        delta = abs(implied_stories - classified_stories)
        if delta > STORY_COUNT_TOL:
            _flag(
                out,
                "story_count_mismatch",
                "fail",
                {"implied": implied_stories, "classified": classified_stories},
                f"|delta| <= {STORY_COUNT_TOL}",
                (
                    f"Height {height_m:.1f} m -> ~{implied_stories} stories, "
                    f"but classification says {classified_stories}."
                ),
            )
        else:
            _flag(
                out,
                "story_count_mismatch",
                "pass",
                {"implied": implied_stories, "classified": classified_stories},
                f"|delta| <= {STORY_COUNT_TOL}",
                "Story count agrees.",
            )

    # 6. Footprint cross-check
    lod2_footprint = _ground_face_area_xz(mesh)
    # Convex hull of all vertices on XZ as a sanity reference.
    xz = np.asarray(mesh.vertices)[:, [0, 2]]
    if len(xz) >= 3:
        hull_xz = MultiPoint(xz.tolist()).convex_hull
        out["convex_footprint_m2"] = float(hull_xz.area)
    else:
        out["convex_footprint_m2"] = None
    out["lod2_ground_face_area_m2"] = lod2_footprint
    tier_ground = _ground_room_floor_sum(tier_payload)
    out["tier_ground_floor_sum_m2"] = tier_ground
    if lod2_footprint and tier_ground:
        delta = abs(lod2_footprint - tier_ground) / max(lod2_footprint, 1e-6)
        out["footprint_delta_pct"] = delta
        if delta > FOOTPRINT_DELTA_MAX:
            _flag(
                out,
                "footprint_mismatch",
                "fail",
                delta,
                FOOTPRINT_DELTA_MAX,
                (
                    f"LoD2 ground area {lod2_footprint:.1f} m^2 vs sum of ground-story "
                    f"room floors {tier_ground:.1f} m^2 differ by {delta * 100:.0f}%."
                ),
            )
        else:
            _flag(
                out,
                "footprint_mismatch",
                "pass",
                delta,
                FOOTPRINT_DELTA_MAX,
                "Footprint agrees with tier rooms.",
            )

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tier-payload", type=Path, default=None)
    args = parser.parse_args()

    if not args.obj.exists():
        args.out.write_text(json.dumps({"error": f"OBJ not found: {args.obj}"}))
        print(f"OBJ not found: {args.obj}")
        return 1

    tier_payload = None
    if args.tier_payload and args.tier_payload.exists():
        try:
            tier_payload = json.loads(args.tier_payload.read_text())
        except json.JSONDecodeError:
            tier_payload = None

    realism = compute_realism(args.obj, tier_payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(realism, indent=2))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
