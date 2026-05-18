#!/usr/bin/env python3
"""Compare roof oblique surfaces before/after lower-envelope cut change.

Runs the extraction + roof pipeline on every building in pipeline-outputs/,
once with the OLD azimuth threshold (140°, opposing-only) and once with the
NEW threshold (60°, covering L-junctions too).  Reports per-building diffs.
"""

import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconcile.extract3d import extract_building
from reconcile.roof_algorithms_py.ceiling_clipping_caps import compute_plane_height_caps
from reconcile.roof_algorithms_py.ceiling_clipping_initial import (
    build_initial_plane_clips,
)
from reconcile.roof_algorithms_py.ceiling_clipping_l_junction import (
    detect_and_expand_l_junctions,
)
from reconcile.roof_algorithms_py.ceiling_clipping_opposing import (
    apply_lower_envelope_cuts,
)
from reconcile.roof_algorithms_py.ceiling_plane_generation import (
    build_ceiling_planes,
    collect_exposed_rooms,
)
from reconcile.roof_algorithms_py.footprint_derivation import build_building_footprint
from reconcile.roof_algorithms_py.math_utils import (
    angle_diff,
    clip_poly_by_half_plane_2d,
    point_in_poly_2d,
    point_in_poly_xz,
)
from reconcile.roof_algorithms_py.oblique_clustering import cluster_oblique_segments
from reconcile.roof_algorithms_py.oblique_surface_generation import (
    build_oblique_roof_surfaces,
)
from reconcile.roof_algorithms_py.segment_collection import collect_oblique_segments
from reconcile.roof_algorithms_py.simple_slant import identify_simple_slant_rooms
from reconcile.roof_algorithms_py.story_index import (
    build_story_index,
    build_story_stats,
    compute_building_y_bounds,
)


def polygon_area_2d(pts):
    """Shoelace formula for 2D polygon area."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def polygon_area_3d(corners):
    """Approximate area of 3D polygon by projecting to XZ."""
    if len(corners) < 3:
        return 0.0
    pts = [(c[0], c[2]) for c in corners]
    return polygon_area_2d(pts)


def polygons_overlap_2d(poly_a, poly_b):
    """Check if two 2D polygons overlap using Shapely."""
    try:
        from shapely.geometry import Polygon

        a = Polygon(poly_a)
        b = Polygon(poly_b)
        if not a.is_valid or not b.is_valid:
            return 0.0
        inter = a.intersection(b)
        return inter.area
    except Exception:
        return 0.0


def apply_old_opposing_cuts(*, ceiling_planes, plane_clipped):
    """Original logic: only opposing planes (140°+), full half-plane cut."""
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
            if ad < 140.0 or ad > 220.0:
                continue

            aj, bj = pj["n"]["x"] / pj["n"]["y"], pj["n"]["z"] / pj["n"]["y"]
            cj = pj["ref"]["y"] + aj * pj["ref"]["x"] + bj * pj["ref"]["z"]
            dx, dz, offset = aj - ai, bj - bi, ci - cj
            if abs(dx) < 1e-9 and abs(dz) < 1e-9:
                continue

            plane_clipped[i]["clipped"] = clip_poly_by_half_plane_2d(
                plane_clipped[i]["clipped"], dx, dz, offset
            )
            plane_clipped[j]["clipped"] = clip_poly_by_half_plane_2d(
                plane_clipped[j]["clipped"], -dx, -dz, -offset
            )


# Also keep the old opposing-only behavior but using Shapely overlap method
# for comparison if needed


def run_clipping_with_threshold(
    bldg,
    ceiling_planes,
    building_footprint,
    exposed_rooms,
    top_story,
    all_stories,
    floors_by_story,
    story_floor_y,
    use_new,
):
    """Run the clipping pipeline with either old or new threshold."""
    plane_clipped = build_initial_plane_clips(
        ceiling_planes=ceiling_planes,
        building_footprint=building_footprint,
        exposed_rooms=exposed_rooms,
        all_rooms=bldg.get("rooms"),
    )

    detect_and_expand_l_junctions(
        ceiling_planes=ceiling_planes,
        plane_clipped=plane_clipped,
        building_footprint=building_footprint,
        point_in_poly_2d=point_in_poly_2d,
    )

    plane_max_y = compute_plane_height_caps(
        bldg=bldg,
        ceiling_planes=ceiling_planes,
        plane_clipped=plane_clipped,
        top_story=top_story,
        all_stories=all_stories,
        floors_by_story=floors_by_story,
        point_in_poly_xz=point_in_poly_xz,
        point_in_poly_2d=point_in_poly_2d,
    )

    if use_new:
        apply_lower_envelope_cuts(
            ceiling_planes=ceiling_planes,
            plane_clipped=plane_clipped,
        )
    else:
        apply_old_opposing_cuts(
            ceiling_planes=ceiling_planes,
            plane_clipped=plane_clipped,
        )

    surfaces = build_oblique_roof_surfaces(
        ceiling_planes=ceiling_planes,
        plane_clipped=plane_clipped,
        plane_max_y=plane_max_y,
        story_floor_y=story_floor_y,
    )
    return surfaces


def process_building(uuid, pipeline_dir, scan_cache_root):
    """Process a single building and compare old vs new."""
    bldg = extract_building(uuid, pipeline_dir, scan_cache_root)
    if not bldg:
        return None

    story_index = build_story_index(bldg)
    story_stats = build_story_stats(bldg)
    compute_building_y_bounds(bldg)

    simple_slant = identify_simple_slant_rooms(
        bldg, story_index["has_floor_above"], story_index["all_stories"]
    )
    exclude_rooms = simple_slant["simple_slant_rooms"]

    segments = collect_oblique_segments(
        bldg, story_index["has_floor_above"], exclude_room_indices=exclude_rooms
    )
    valid_clusters = cluster_oblique_segments(segments)

    exposed_rooms = collect_exposed_rooms(
        bldg, story_index["has_floor_above"], exclude_room_indices=exclude_rooms
    )
    ceiling_planes = build_ceiling_planes(valid_clusters)
    footprint_info = build_building_footprint(
        exposed_rooms, story_stats["story_floor_polys"]
    )

    building_footprint = footprint_info["building_footprint"]
    if not building_footprint or len(building_footprint) < 3:
        return None
    if len(ceiling_planes) < 2:
        return None  # No pairs to compare

    # Run both versions
    old_surfaces = run_clipping_with_threshold(
        bldg,
        ceiling_planes,
        building_footprint,
        exposed_rooms,
        footprint_info["top_story"],
        story_index["all_stories"],
        story_index["floors_by_story"],
        story_stats["story_floor_y"],
        use_new=False,
    )

    # Ceiling planes are modified in place by clipping, so rebuild
    ceiling_planes_new = build_ceiling_planes(valid_clusters)
    new_surfaces = run_clipping_with_threshold(
        bldg,
        ceiling_planes_new,
        building_footprint,
        exposed_rooms,
        footprint_info["top_story"],
        story_index["all_stories"],
        story_index["floors_by_story"],
        story_stats["story_floor_y"],
        use_new=True,
    )

    return {
        "uuid": uuid,
        "n_planes": len(ceiling_planes),
        "old_n_surfaces": len(old_surfaces),
        "new_n_surfaces": len(new_surfaces),
        "old_surfaces": old_surfaces,
        "new_surfaces": new_surfaces,
    }


def main():
    pipeline_dir = Path("pipeline-outputs")
    scan_cache_root = Path(".scan-cache")

    uuids = sorted(
        entry.name
        for entry in pipeline_dir.iterdir()
        if entry.is_dir() and (entry / "merged.json").exists()
    )

    print(f"Processing {len(uuids)} buildings...\n")

    changed = []
    errors = []
    total = 0
    skipped = 0

    for i, uuid in enumerate(uuids):
        try:
            result = process_building(uuid, pipeline_dir, scan_cache_root)
        except Exception as exc:
            errors.append((uuid, str(exc)))
            continue

        if result is None:
            skipped += 1
            continue
        total += 1

        old_areas = [polygon_area_3d(s["corners"]) for s in result["old_surfaces"]]
        new_areas = [polygon_area_3d(s["corners"]) for s in result["new_surfaces"]]
        old_total = sum(old_areas)
        new_total = sum(new_areas)

        # Check for overlap between surfaces (old)
        old_overlap = 0.0
        for si in range(len(result["old_surfaces"])):
            for sj in range(si + 1, len(result["old_surfaces"])):
                poly_a = [(c[0], c[2]) for c in result["old_surfaces"][si]["corners"]]
                poly_b = [(c[0], c[2]) for c in result["old_surfaces"][sj]["corners"]]
                old_overlap += polygons_overlap_2d(poly_a, poly_b)

        new_overlap = 0.0
        for si in range(len(result["new_surfaces"])):
            for sj in range(si + 1, len(result["new_surfaces"])):
                poly_a = [(c[0], c[2]) for c in result["new_surfaces"][si]["corners"]]
                poly_b = [(c[0], c[2]) for c in result["new_surfaces"][sj]["corners"]]
                new_overlap += polygons_overlap_2d(poly_a, poly_b)

        area_diff = new_total - old_total
        overlap_diff = new_overlap - old_overlap
        surface_diff = result["new_n_surfaces"] - result["old_n_surfaces"]

        if abs(area_diff) > 0.01 or abs(overlap_diff) > 0.01 or surface_diff != 0:
            changed.append(
                {
                    "uuid": uuid,
                    "n_planes": result["n_planes"],
                    "old_n_surfaces": result["old_n_surfaces"],
                    "new_n_surfaces": result["new_n_surfaces"],
                    "old_total_area": round(old_total, 2),
                    "new_total_area": round(new_total, 2),
                    "area_diff": round(area_diff, 2),
                    "old_overlap": round(old_overlap, 2),
                    "new_overlap": round(new_overlap, 2),
                    "overlap_diff": round(overlap_diff, 2),
                    "old_areas": [round(a, 2) for a in old_areas],
                    "new_areas": [round(a, 2) for a in new_areas],
                }
            )

        if (i + 1) % 25 == 0:
            print(
                f"  [{i + 1}/{len(uuids)}] processed, {len(changed)} changed so far..."
            )

    # Summary
    print(f"\n{'=' * 80}")
    print(
        f"RESULTS: {total} buildings processed, {skipped} skipped, {len(errors)} errors"
    )
    print(f"{'=' * 80}\n")

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for uuid, err in errors[:10]:
            print(f"  {uuid}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()

    if not changed:
        print("NO CHANGES — old and new produce identical results for all buildings.")
    else:
        print(f"CHANGED BUILDINGS ({len(changed)}):\n")

        # Categorize
        surfaces_lost = [
            c for c in changed if c["new_n_surfaces"] < c["old_n_surfaces"]
        ]
        surfaces_same = [
            c for c in changed if c["new_n_surfaces"] == c["old_n_surfaces"]
        ]
        overlap_reduced = [c for c in changed if c["overlap_diff"] < -0.01]
        overlap_increased = [c for c in changed if c["overlap_diff"] > 0.01]

        print(f"  Surfaces eliminated: {len(surfaces_lost)} buildings")
        print(
            f"  Surfaces same count but different area: {len(surfaces_same)} buildings"
        )
        print(f"  Overlap REDUCED: {len(overlap_reduced)} buildings")
        print(f"  Overlap INCREASED: {len(overlap_increased)} buildings")
        print()

        for c in changed:
            flag = ""
            if c["new_n_surfaces"] < c["old_n_surfaces"]:
                flag = " ⚠ SURFACE LOST"
            elif c["overlap_diff"] > 0.01:
                flag = " ⚠ OVERLAP INCREASED"
            print(f"  {c['uuid']}:{flag}")
            print(
                f"    planes={c['n_planes']}  surfaces: "
                f"{c['old_n_surfaces']}→{c['new_n_surfaces']}"
            )
            print(
                f"    area: {c['old_total_area']}→{c['new_total_area']} "
                f"(diff={c['area_diff']})"
            )
            print(
                f"    overlap: {c['old_overlap']}→{c['new_overlap']} "
                f"(diff={c['overlap_diff']})"
            )
            if c["old_areas"] != c["new_areas"]:
                print(f"    old per-surface: {c['old_areas']}")
                print(f"    new per-surface: {c['new_areas']}")
            print()

    # Write JSON report
    report_path = Path("scripts/lower_envelope_report.json")
    report = {
        "total_processed": total,
        "skipped": skipped,
        "errors": len(errors),
        "changed_count": len(changed),
        "changed": changed,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report written to {report_path}")


if __name__ == "__main__":
    main()
