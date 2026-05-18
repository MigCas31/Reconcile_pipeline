from __future__ import annotations

from .boundary_model import (
    build_boundary_face_model,
    derive_roof_surfaces_from_boundary_model,
)
from .ceiling_plane_clipping import clip_ceiling_planes
from .ceiling_plane_generation import (
    build_ceiling_planes,
    collect_exposed_rooms,
    collect_room_top_boundary_records,
)
from .flat_surface_generation import build_flat_roof_surfaces
from .footprint_derivation import build_building_footprint
from .math_utils import point_in_poly_2d, point_in_poly_xz
from .oblique_clustering import cluster_oblique_segments
from .oblique_surface_generation import build_oblique_roof_surfaces
from .occupied_room_cell_complex import build_occupied_room_cell_complex
from .raw_ceiling_sources import collect_raw_oblique_rectangle_clusters
from .roof_arrangement import build_roof_arrangement
from .roof_building_parts import (
    build_building_part_graph,
    classify_selected_flat_hypothesis_roles,
    refine_building_part_graph,
)
from .roof_cell_complex import (
    build_roof_cell_complex,
    filter_knee_walls_by_occupied_shell,
)
from .roof_coverage_graph import build_roof_coverage_graph
from .roof_envelope_continuation import continue_roof_envelopes
from .roof_evidence_graph import (
    annotate_roof_coverage_graph,
    build_roof_evidence_graph,
)
from .roof_flat_oblique_clipping import clip_flat_surfaces_against_obliques
from .roof_graph import build_roof_boundary_graph
from .roof_hypothesis_graph import (
    build_roof_hypothesis_graph,
    select_roof_surfaces_from_hypotheses,
)
from .roof_partitioning import (
    derive_room_ceiling_partitions,
    inject_simple_slant_partitions,
)
from .segment_collection import collect_oblique_segments
from .simple_slant import identify_simple_slant_rooms
from .steps import ROOF_PIPELINE_STEPS
from .story_index import build_story_index, build_story_stats, compute_building_y_bounds
from .top_boundary_graph import build_top_boundary_graph


def run_roof_algorithms(bldg: dict, graph=None) -> dict:
    story_index = build_story_index(bldg, graph=graph)
    story_stats = build_story_stats(bldg)
    y_bounds = compute_building_y_bounds(bldg)

    simple_slant = identify_simple_slant_rooms(
        bldg, story_index["has_floor_above"], story_index["all_stories"], graph=graph
    )
    exclude_rooms = simple_slant["simple_slant_rooms"]

    segments = collect_oblique_segments(
        bldg,
        story_index["has_floor_above"],
        exclude_room_indices=exclude_rooms,
        graph=graph,
    )
    valid_clusters = cluster_oblique_segments(segments)
    raw_rectangle_clusters = collect_raw_oblique_rectangle_clusters(
        bldg,
        existing_clusters=valid_clusters,
        has_floor_above=story_index["has_floor_above"],
        exclude_room_indices=exclude_rooms,
        graph=graph,
    )
    valid_clusters.extend(raw_rectangle_clusters)

    roof_cluster_data = [
        {
            "count": len(cl["segs"]),
            "avgIncl": cl["avgIncl"],
            "avgAzimuth": cl["avgAzimuth"],
            "flat": False,
            "source": cl.get("source", "wall_segments"),
        }
        for cl in valid_clusters
    ]

    flat_roof_surfaces, flat_legend = build_flat_roof_surfaces(
        bldg,
        story_index["all_stories"],
        story_index["has_floor_above"],
        y_bounds["bldg_min_y"],
        graph=graph,
    )
    roof_cluster_data.extend(flat_legend)

    exposed_rooms = collect_exposed_rooms(
        bldg,
        story_index["has_floor_above"],
        exclude_room_indices=exclude_rooms,
        graph=graph,
    )
    top_boundary_rooms = collect_room_top_boundary_records(
        bldg,
        story_index["has_floor_above"],
        graph=graph,
    )
    ceiling_planes = build_ceiling_planes(valid_clusters, bldg=bldg, graph=graph)
    footprint_info = build_building_footprint(
        exposed_rooms, story_stats["story_floor_polys"]
    )

    # Build oblique roof surfaces using the ceiling pipeline's smart clipping
    oblique_roof_surfaces: list[dict] = []
    building_footprint = footprint_info["building_footprint"]
    wing_polygons: list = []
    if building_footprint and len(building_footprint) >= 3:
        from shapely.geometry import Polygon as _ShPolygon

        from reconcile.wing_decomposition import decompose_to_wings

        try:
            ring = [
                (float(p[0]), float(p[2] if len(p) >= 3 else p[1]))
                for p in building_footprint
            ]
            footprint_poly = _ShPolygon(ring)
            if footprint_poly.is_valid and not footprint_poly.is_empty:
                wing_polygons = [w.polygon for w in decompose_to_wings(footprint_poly)]
        except Exception:
            wing_polygons = []
        # Only feed wings into clipping when there are 2+ wings — single-wing
        # buildings don't need the constraint and we avoid cost-for-nothing.
        clipping_wings = wing_polygons if len(wing_polygons) >= 2 else None
        clipped = clip_ceiling_planes(
            bldg=bldg,
            ceiling_planes=ceiling_planes,
            building_footprint=building_footprint,
            exposed_rooms=exposed_rooms,
            all_rooms=bldg.get("rooms"),
            top_story=footprint_info["top_story"],
            all_stories=story_index["all_stories"],
            floors_by_story=story_index["floors_by_story"],
            point_in_poly_xz=point_in_poly_xz,
            point_in_poly_2d=point_in_poly_2d,
            graph=graph,
            wing_polygons=clipping_wings,
        )
        oblique_roof_surfaces = build_oblique_roof_surfaces(
            ceiling_planes=ceiling_planes,
            plane_clipped=clipped["plane_clipped"],
            plane_max_y=clipped["plane_max_y"],
            story_floor_y=story_stats["story_floor_y"],
            all_rooms=bldg.get("rooms"),
        )

    candidate_boundary_model = build_boundary_face_model(
        bldg=bldg,
        roof_result={
            "roof_surfaces": {
                "oblique": oblique_roof_surfaces,
                "flat": flat_roof_surfaces,
            }
        },
        graph=graph,
    )
    candidate_roof_surfaces = derive_roof_surfaces_from_boundary_model(
        candidate_boundary_model
    )
    candidate_roof_graph = build_roof_boundary_graph(
        candidate_boundary_model, graph=graph
    )
    roof_hypothesis_graph = build_roof_hypothesis_graph(
        bldg=bldg,
        exposed_rooms=exposed_rooms,
        oblique_surfaces=candidate_roof_surfaces["oblique"],
        flat_surfaces=candidate_roof_surfaces["flat"],
        roof_graph=candidate_roof_graph,
        graph=graph,
    )
    selected_oblique_surfaces, selected_flat_surfaces = (
        select_roof_surfaces_from_hypotheses(
            oblique_surfaces=candidate_roof_surfaces["oblique"],
            flat_surfaces=candidate_roof_surfaces["flat"],
            hypothesis_graph=roof_hypothesis_graph,
        )
    )
    building_part_graph = build_building_part_graph(
        exposed_rooms=exposed_rooms,
        hypothesis_graph=roof_hypothesis_graph,
        bldg=bldg,
        graph=graph,
    )
    flat_hypothesis_roles = classify_selected_flat_hypothesis_roles(
        exposed_rooms=exposed_rooms,
        hypothesis_graph=roof_hypothesis_graph,
        building_part_graph=building_part_graph,
    )
    selected_flat_with_roles = []
    for surface in selected_flat_surfaces:
        hypothesis_id = surface.get("roof_hypothesis_id")
        role_info = flat_hypothesis_roles.get(
            str(hypothesis_id), {"role": "roof_flat", "reason": "default"}
        )
        enriched = dict(surface)
        enriched["flat_role"] = role_info["role"]
        enriched["flat_role_reason"] = role_info["reason"]
        selected_flat_with_roles.append(enriched)
    roof_flat_support_surfaces = clip_flat_surfaces_against_obliques(
        flat_surfaces=selected_flat_with_roles,
        oblique_surfaces=selected_oblique_surfaces,
        hypothesis_graph=roof_hypothesis_graph,
        keep_side="above",
        suppress_consumed_top_flats=True,
    )
    ceiling_flat_support_surfaces = clip_flat_surfaces_against_obliques(
        flat_surfaces=selected_flat_with_roles,
        oblique_surfaces=selected_oblique_surfaces,
        keep_side="below",
        suppress_consumed_top_flats=False,
    )
    roof_flat_surfaces = [
        surface
        for surface in roof_flat_support_surfaces
        if surface.get("flat_role") != "ceiling_cap"
    ]

    boundary_model = build_boundary_face_model(
        bldg=bldg,
        roof_result={
            "roof_surfaces": {
                "oblique": selected_oblique_surfaces,
                "flat": roof_flat_surfaces,
            }
        },
        graph=graph,
    )
    derived_roof_surfaces = derive_roof_surfaces_from_boundary_model(boundary_model)
    roof_graph = build_roof_boundary_graph(boundary_model, graph=graph)
    working_hypothesis_graph = roof_hypothesis_graph
    ceiling_partitions = derive_room_ceiling_partitions(
        room_records=top_boundary_rooms,
        oblique_roof_surfaces=derived_roof_surfaces["oblique"],
        flat_roof_surfaces=ceiling_flat_support_surfaces,
        hypothesis_graph=working_hypothesis_graph,
    )
    ceiling_partitions = inject_simple_slant_partitions(
        partitions=ceiling_partitions,
        simple_slant_ceilings=simple_slant["simple_slant_ceilings"],
        room_records=top_boundary_rooms,
    )
    roof_coverage_graph = build_roof_coverage_graph(
        room_partitions=ceiling_partitions["room_partitions"],
        selected_oblique_surfaces=derived_roof_surfaces["oblique"],
        selected_flat_surfaces=derived_roof_surfaces["flat"],
        building_part_graph=building_part_graph,
    )
    continuation = continue_roof_envelopes(
        exposed_rooms=exposed_rooms,
        room_partitions=ceiling_partitions["room_partitions"],
        selected_oblique_surfaces=derived_roof_surfaces["oblique"],
        hypothesis_graph=working_hypothesis_graph,
        building_part_graph=building_part_graph,
        roof_coverage_graph=roof_coverage_graph,
    )
    if continuation["continued_surfaces"]:
        # Continuation remains a graph/diagnostic inference until it is backed by
        # explicit arrangement faces. Do not feed polygon-lifted continuation
        # patches back into the committed roof surface set.
        working_hypothesis_graph = continuation["hypothesis_graph"]
    roof_arrangement = build_roof_arrangement(
        bldg=bldg,
        oblique_surfaces=derived_roof_surfaces["oblique"],
        building_footprint=building_footprint,
    )
    occupied_room_cell_complex = build_occupied_room_cell_complex(
        bldg=bldg,
        room_partitions=ceiling_partitions["room_partitions"],
        building_part_graph=building_part_graph,
    )
    roof_cell_complex = build_roof_cell_complex(
        room_partitions=ceiling_partitions["room_partitions"],
        selected_oblique_surfaces=derived_roof_surfaces["oblique"],
        selected_flat_surfaces=derived_roof_surfaces["flat"],
        building_part_graph=building_part_graph,
        roof_coverage_graph=roof_coverage_graph,
    )
    roof_cell_complex = filter_knee_walls_by_occupied_shell(
        roof_cell_complex=roof_cell_complex,
        occupied_room_cell_complex=occupied_room_cell_complex,
    )
    roof_evidence_graph = build_roof_evidence_graph(
        exposed_rooms=exposed_rooms,
        room_partitions=ceiling_partitions["room_partitions"],
        selected_oblique_surfaces=derived_roof_surfaces["oblique"],
        building_part_graph=building_part_graph,
        roof_coverage_graph=roof_coverage_graph,
        roof_cell_complex=roof_cell_complex,
    )
    roof_coverage_graph = annotate_roof_coverage_graph(
        roof_coverage_graph=roof_coverage_graph,
        roof_evidence_graph=roof_evidence_graph,
    )
    building_part_graph = refine_building_part_graph(
        building_part_graph=building_part_graph,
        roof_evidence_graph=roof_evidence_graph,
    )
    top_boundary_graph = build_top_boundary_graph(
        room_records=top_boundary_rooms,
        room_partitions=ceiling_partitions["room_partitions"],
        building_part_graph=building_part_graph,
        roof_coverage_graph=roof_coverage_graph,
        roof_cell_complex=roof_cell_complex,
        roof_evidence_graph=roof_evidence_graph,
    )
    flat_ceilings = ceiling_partitions["flat"]

    # --- Dormer detection and geometry ---
    from .dormer_detection import detect_dormers, group_dormer_walls
    from .dormer_geometry import build_dormer_geometry

    dormer_candidates = detect_dormers(
        bldg,
        derived_roof_surfaces["oblique"],
        story_stats["story_floor_y"],
        roof_coverage_graph=roof_coverage_graph,
        top_boundary_graph=top_boundary_graph,
        knee_walls=roof_cell_complex.get("knee_walls") or [],
        roof_evidence_graph=roof_evidence_graph,
    )
    dormers_out: list[dict] = []
    for d in dormer_candidates:
        surf_idx = d["roof_surface_index"]
        surf = derived_roof_surfaces["oblique"][surf_idx]
        room = bldg["rooms"][d["front_wall"]["room_index"]]
        group = group_dormer_walls(d["front_wall"], room.get("walls_merged", []))

        geom = build_dormer_geometry(
            group, surf, room_walls=room.get("walls_merged", [])
        )

        if geom["cutout_corners"]:
            surf.setdefault("cutout_holes", []).append(geom["cutout_corners"])
            dormers_out.append(
                {
                    "front_wall_id": d["front_wall"]["id"],
                    "roof_surface_index": surf_idx,
                    "room_index": d["front_wall"]["room_index"],
                    "cheeks": geom["cheeks"],
                    "header": geom["header"],
                    "cutout_corners": geom["cutout_corners"],
                }
            )

    # --- Thermal-envelope ceilings (after dormers so cutout_holes are present) ---
    from .thermal_ceiling import build_thermal_ceilings

    thermal_ceilings = build_thermal_ceilings(
        exposed_rooms=exposed_rooms,
        oblique_roof_surfaces=derived_roof_surfaces["oblique"],
        ceiling_planes=ceiling_planes,
        flat_ceilings=flat_ceilings,
        simple_slant_ceilings=simple_slant["simple_slant_ceilings"],
        floors_by_story=story_index["floors_by_story"],
        story_floor_y=story_stats["story_floor_y"],
        story_floor_regions=story_stats["story_floor_regions"],
        dormers=dormers_out,
        gap_walls=bldg.get("gap_walls", []),
        rooms=bldg.get("rooms", []),
    )

    return {
        "pipeline_steps": [step["id"] for step in ROOF_PIPELINE_STEPS],
        "segments": segments,
        "valid_clusters": valid_clusters,
        "roof_cluster_data": roof_cluster_data,
        "roof_surfaces": {
            "oblique": derived_roof_surfaces["oblique"],
            "oblique_split": roof_arrangement["oblique_split"],
            "flat": derived_roof_surfaces["flat"],
        },
        "roof_arrangement": {
            "cells": roof_arrangement["cells"],
            "edges": roof_arrangement["edges"],
            "metadata": roof_arrangement["metadata"],
        },
        "boundary_faces": boundary_model,
        "roof_graph": roof_graph,
        "roof_hypothesis_graph": working_hypothesis_graph,
        "building_part_graph": building_part_graph,
        "roof_coverage_graph": roof_coverage_graph,
        "roof_cell_complex": roof_cell_complex,
        "occupied_room_cell_complex": occupied_room_cell_complex,
        "roof_evidence_graph": roof_evidence_graph,
        "top_boundary_graph": top_boundary_graph,
        "roof_continuation_diagnostics": continuation,
        "ceiling_partitions": ceiling_partitions,
        "ceiling": {
            "exposed_rooms": exposed_rooms,
            "planes": ceiling_planes,
            "footprint": building_footprint,
            "flat": flat_ceilings,
            "oblique": ceiling_partitions["oblique"],
            "simple_slant": simple_slant["simple_slant_ceilings"],
            "room_partitions": ceiling_partitions["room_partitions"],
            "partition_metadata": ceiling_partitions["metadata"],
            "coverage_graph": roof_coverage_graph,
            "cell_complex": roof_cell_complex,
            "occupied_room_cell_complex": occupied_room_cell_complex,
            "top_boundary_graph": top_boundary_graph,
            "continuation_diagnostics": continuation,
            "thermal": thermal_ceilings,
        },
        "knee_walls": roof_cell_complex.get("knee_walls") or [],
        "dormers": dormers_out,
    }
