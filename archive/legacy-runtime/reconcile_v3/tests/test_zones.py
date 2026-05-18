from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import Polygon

from reconcile.extract3d import extract_building
from reconcile_v3.reconstruction.zones import (
    _suppress_duplicate_zones,
    derive_authoritative_zones,
)

_BLDG = "test-building"


def _ring_xyz(
    x0: float, z0: float, x1: float, z1: float, y: float
) -> list[list[float]]:
    return [
        [x0, y, z0],
        [x1, y, z0],
        [x1, y, z1],
        [x0, y, z1],
    ]


def _cell(
    cell_id: str,
    room_index: int,
    room_id: str,
    footprint: tuple[float, float, float, float],
    *,
    base_y: float = 0.0,
    top_y: float = 2.5,
    volume_m3: float = 10.0,
    top_atom_ids: list[str] | None = None,
    roof_hypothesis_id: str | None = None,
    part_id: str | None = None,
) -> dict:
    x0, z0, x1, z1 = footprint
    return {
        "id": cell_id,
        "room_index": room_index,
        "room_id": room_id,
        "part_id": part_id,
        "roof_hypothesis_id": roof_hypothesis_id,
        "volume_m3": volume_m3,
        "bbox_xyz": [x0, base_y, z0, x1, top_y, z1],
        "top_boundary_atom_id": (top_atom_ids or [None])[0],
        "top_boundary_atom_ids": top_atom_ids or [],
        "faces": [
            {"kind": "top", "corners": _ring_xyz(x0, z0, x1, z1, top_y)},
            {"kind": "bottom", "corners": _ring_xyz(x0, z0, x1, z1, base_y)},
        ],
    }


def _room_partition(
    room_index: int, footprint: tuple[float, float, float, float], atom_id: str
) -> dict:
    x0, z0, x1, z1 = footprint
    poly = _ring_xyz(x0, z0, x1, z1, 0.0)
    return {
        "room_index": room_index,
        "room_outline": poly,
        "partitions": [
            {
                "id": atom_id,
                "room_index": room_index,
                "story": 0,
                "kind": "flat",
                "poly": poly,
                "roof_hypothesis_id": None,
            }
        ],
    }


def _roof_building(
    *,
    occupied_cells: list[dict],
    room_partitions: list[dict],
    subparts: list[dict] | None = None,
    room_subpart_membership: dict[str, list[str]] | None = None,
    atom_subpart_membership: dict[str, list[str]] | None = None,
    part_nodes: list[dict] | None = None,
) -> dict:
    return {
        "occupied_room_cell_complex": {"cells": occupied_cells},
        "roof_cell_complex": {"cells": []},
        "ceiling_partitions": {"room_partitions": room_partitions},
        "roof_coverage_graph": {
            "subparts": subparts or [],
            "room_subpart_membership": room_subpart_membership or {},
            "atom_subpart_membership": atom_subpart_membership or {},
        },
        "building_part_graph": {
            "nodes": part_nodes or [],
            "room_membership": {
                f"room:{room_partition['room_index']}": [
                    str(node["id"])
                    for node in (part_nodes or [])
                    if f"room:{room_partition['room_index']}"
                    in (node.get("room_ids") or [])
                ]
                for room_partition in room_partitions
            },
        },
        "roof_evidence_graph": {"part_evidence": {}, "subpart_evidence": {}},
    }


def test_support_component_merges_touching_occupied_cells() -> None:
    occupied = [
        _cell("cell-a", 0, "room:0", (0.0, 0.0, 2.0, 2.0), top_atom_ids=["atom-a"]),
        _cell("cell-b", 1, "room:1", (2.0, 0.0, 4.0, 2.0), top_atom_ids=["atom-b"]),
    ]
    roof_building = _roof_building(
        occupied_cells=occupied,
        room_partitions=[
            _room_partition(0, (0.0, 0.0, 2.0, 2.0), "atom-a"),
            _room_partition(1, (2.0, 0.0, 4.0, 2.0), "atom-b"),
        ],
        part_nodes=[
            {"id": "building-part:main", "room_ids": ["room:0", "room:1"]},
        ],
    )

    zones = derive_authoritative_zones(
        building_uuid=_BLDG,
        roof_building=roof_building,
        bldg=None,
    )

    assert len(zones) == 1
    assert zones[0]["room_ids"] == ["room:0", "room:1"]
    assert zones[0]["fallback_kind"] == "support_component_without_subpart"


def test_subpart_attaches_by_room_overlap_even_when_part_hint_is_wrong() -> None:
    occupied = [
        _cell(
            "cell-annex",
            2,
            "room:2",
            (0.0, 0.0, 3.0, 2.0),
            top_atom_ids=["atom-annex"],
            roof_hypothesis_id="roof-hypothesis:oblique:annex",
            part_id="building-part:true",
        ),
    ]
    subparts = [
        {
            "id": "coverage-subpart:wrong-hint",
            "roof_hypothesis_id": "roof-hypothesis:oblique:annex",
            "room_indices": [2],
            "part_ids": ["building-part:wrong"],
            "semantic_kind": "gable_run",
            "support_evidence_score": 5,
            "adjacent_subpart_ids": [],
            "polygon_xz": [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]],
        },
    ]
    roof_building = _roof_building(
        occupied_cells=occupied,
        room_partitions=[_room_partition(2, (0.0, 0.0, 3.0, 2.0), "atom-annex")],
        subparts=subparts,
        room_subpart_membership={"room:2": ["coverage-subpart:wrong-hint"]},
        atom_subpart_membership={"atom-annex": ["coverage-subpart:wrong-hint"]},
        part_nodes=[
            {"id": "building-part:true", "room_ids": ["room:2"]},
            {"id": "building-part:wrong", "room_ids": []},
        ],
    )

    zones = derive_authoritative_zones(
        building_uuid=_BLDG,
        roof_building=roof_building,
        bldg=None,
    )

    assert len(zones) == 1
    assert zones[0]["part_id"] == "building-part:true"
    assert zones[0]["seed_subpart_ids"] == ["coverage-subpart:wrong-hint"]


def test_fallback_annex_without_seed_becomes_zone() -> None:
    occupied = [
        _cell("cell-annex", 4, "room:4", (0.0, 0.0, 2.0, 2.0), volume_m3=8.0),
    ]
    roof_building = _roof_building(
        occupied_cells=occupied,
        room_partitions=[_room_partition(4, (0.0, 0.0, 2.0, 2.0), "atom-annex")],
        part_nodes=[{"id": "building-part:annex", "room_ids": ["room:4"]}],
    )

    zones = derive_authoritative_zones(
        building_uuid=_BLDG,
        roof_building=roof_building,
        bldg=None,
    )

    assert len(zones) == 1
    assert zones[0]["fallback_kind"] == "support_component_without_subpart"
    assert zones[0]["seed_subpart_ids"] == []


def test_disconnected_support_components_split_zone_geometry() -> None:
    occupied = [
        _cell("cell-left", 0, "room:0", (0.0, 0.0, 2.0, 2.0), volume_m3=8.0),
        _cell("cell-right", 1, "room:1", (10.0, 0.0, 12.0, 2.0), volume_m3=8.0),
    ]
    roof_building = _roof_building(
        occupied_cells=occupied,
        room_partitions=[
            _room_partition(0, (0.0, 0.0, 2.0, 2.0), "atom-left"),
            _room_partition(1, (10.0, 0.0, 12.0, 2.0), "atom-right"),
        ],
        part_nodes=[{"id": "building-part:main", "room_ids": ["room:0", "room:1"]}],
    )

    zones = derive_authoritative_zones(
        building_uuid=_BLDG,
        roof_building=roof_building,
        bldg=None,
    )

    assert len(zones) == 2
    assert {tuple(zone["room_ids"]) for zone in zones} == {("room:0",), ("room:1",)}


def test_seed_spill_does_not_expand_owned_room_set() -> None:
    occupied = [
        _cell(
            "cell-a", 0, "room:0", (0.0, 0.0, 2.0, 2.0), part_id="building-part:left"
        ),
        _cell(
            "cell-b", 1, "room:1", (2.0, 0.0, 4.0, 2.0), part_id="building-part:right"
        ),
    ]
    subparts = [
        {
            "id": "coverage-subpart:broad",
            "roof_hypothesis_id": "roof-hypothesis:gable",
            "room_indices": [0, 1, 2, 3],
            "part_ids": ["building-part:wrong"],
            "semantic_kind": "gable_run",
            "support_evidence_score": 6,
            "adjacent_subpart_ids": [],
            "polygon_xz": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        },
    ]
    roof_building = _roof_building(
        occupied_cells=occupied,
        room_partitions=[
            _room_partition(0, (0.0, 0.0, 2.0, 2.0), "atom-a"),
            _room_partition(1, (2.0, 0.0, 4.0, 2.0), "atom-b"),
            _room_partition(2, (4.0, 0.0, 6.0, 2.0), "atom-c"),
            _room_partition(3, (6.0, 0.0, 8.0, 2.0), "atom-d"),
        ],
        subparts=subparts,
        room_subpart_membership={
            "room:0": ["coverage-subpart:broad"],
            "room:1": ["coverage-subpart:broad"],
            "room:2": ["coverage-subpart:broad"],
            "room:3": ["coverage-subpart:broad"],
        },
        atom_subpart_membership={"atom-a": ["coverage-subpart:broad"]},
        part_nodes=[
            {"id": "building-part:left", "room_ids": ["room:0"]},
            {"id": "building-part:right", "room_ids": ["room:1"]},
        ],
    )

    zones = derive_authoritative_zones(
        building_uuid=_BLDG,
        roof_building=roof_building,
        bldg=None,
    )

    seeded_zone = next(
        zone for zone in zones if zone["seed_subpart_ids"] == ["coverage-subpart:broad"]
    )
    assert seeded_zone["room_ids"] == ["room:0"]
    assert seeded_zone["seed_spill_room_indices"] == [1, 2, 3]


def test_overlapping_support_projections_are_relabeled_disjointly() -> None:
    occupied = [
        _cell(
            "cell-left", 0, "room:0", (0.0, 0.0, 4.0, 2.0), part_id="building-part:left"
        ),
        _cell(
            "cell-right",
            1,
            "room:1",
            (2.0, 0.0, 6.0, 2.0),
            part_id="building-part:right",
        ),
    ]
    roof_building = _roof_building(
        occupied_cells=occupied,
        room_partitions=[
            _room_partition(0, (0.0, 0.0, 3.0, 2.0), "atom-left"),
            _room_partition(1, (3.0, 0.0, 6.0, 2.0), "atom-right"),
        ],
        part_nodes=[
            {"id": "building-part:left", "room_ids": ["room:0"]},
            {"id": "building-part:right", "room_ids": ["room:1"]},
        ],
    )

    zones = derive_authoritative_zones(
        building_uuid=_BLDG,
        roof_building=roof_building,
        bldg=None,
    )

    assert len(zones) == 2
    zone_polys = [Polygon(zone["footprint_xz"]) for zone in zones]
    assert zone_polys[0].intersection(zone_polys[1]).area < 1e-6


def test_no_part_shadow_zone_is_suppressed() -> None:
    part_poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    shadow_poly = Polygon([(0.1, 0.1), (3.9, 0.1), (3.9, 3.9), (0.1, 3.9)])
    zones = _suppress_duplicate_zones(
        [
            {
                "id": "shadow",
                "part_id": None,
                "support_component_id": "support-a",
                "seed_hypothesis_ids": ["roof-hypothesis:1"],
                "confidence": 0.4,
                "owned_support_area_m2": shadow_poly.area,
                "_poly": shadow_poly,
            },
            {
                "id": "real",
                "part_id": "building-part:main",
                "support_component_id": "support-b",
                "seed_hypothesis_ids": ["roof-hypothesis:1"],
                "confidence": 0.8,
                "owned_support_area_m2": part_poly.area,
                "_poly": part_poly,
            },
        ]
    )

    assert [zone["id"] for zone in zones] == ["real"]


def test_same_support_duplicate_zone_prefers_seed_not_reused_elsewhere() -> None:
    shared_poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    zones = _suppress_duplicate_zones(
        [
            {
                "id": "keep-me",
                "part_id": None,
                "support_component_id": "support-main",
                "room_ids": ["room:0", "room:1"],
                "seed_subpart_ids": ["coverage-subpart:local"],
                "seed_hypothesis_ids": ["roof-hypothesis:0"],
                "seed_spill_room_indices": [2],
                "confidence": 0.6,
                "owned_support_area_m2": shared_poly.area,
                "_poly": shared_poly,
            },
            {
                "id": "drop-me",
                "part_id": None,
                "support_component_id": "support-main",
                "room_ids": ["room:0", "room:1"],
                "seed_subpart_ids": ["coverage-subpart:reused"],
                "seed_hypothesis_ids": ["roof-hypothesis:1"],
                "seed_spill_room_indices": [2],
                "confidence": 0.6,
                "owned_support_area_m2": shared_poly.area,
                "_poly": shared_poly,
            },
            {
                "id": "other-support",
                "part_id": "building-part:annex",
                "support_component_id": "support-annex",
                "room_ids": ["room:4"],
                "seed_subpart_ids": ["coverage-subpart:reused"],
                "seed_hypothesis_ids": ["roof-hypothesis:1"],
                "seed_spill_room_indices": [],
                "confidence": 0.8,
                "owned_support_area_m2": 1.0,
                "_poly": Polygon([(10.0, 0.0), (11.0, 0.0), (11.0, 1.0), (10.0, 1.0)]),
            },
        ]
    )

    assert {zone["id"] for zone in zones} == {"keep-me", "other-support"}


def test_unassigned_seeded_piece_splits_by_part_overlap() -> None:
    occupied = [
        _cell(
            "cell-main",
            0,
            "room:0",
            (0.0, 0.0, 4.0, 2.0),
            part_id="building-part:main",
            top_atom_ids=["atom-main"],
            roof_hypothesis_id="roof-hypothesis:gable",
        ),
    ]
    subparts = [
        {
            "id": "coverage-subpart:broad",
            "roof_hypothesis_id": "roof-hypothesis:gable",
            "room_indices": [0, 1],
            "part_ids": ["building-part:wrong"],
            "semantic_kind": "gable_run",
            "support_evidence_score": 6,
            "adjacent_subpart_ids": [],
            "polygon_xz": [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
        },
    ]
    roof_building = _roof_building(
        occupied_cells=occupied,
        room_partitions=[
            _room_partition(0, (0.0, 0.0, 2.0, 2.0), "atom-main"),
            _room_partition(1, (2.0, 0.0, 4.0, 2.0), "atom-spill"),
        ],
        subparts=subparts,
        room_subpart_membership={
            "room:0": ["coverage-subpart:broad"],
            "room:1": ["coverage-subpart:broad"],
        },
        atom_subpart_membership={"atom-main": ["coverage-subpart:broad"]},
        part_nodes=[
            {"id": "building-part:main", "room_ids": ["room:0"]},
            {"id": "building-part:spill", "room_ids": ["room:1"]},
        ],
    )

    zones = derive_authoritative_zones(
        building_uuid=_BLDG,
        roof_building=roof_building,
        bldg=None,
    )

    assert len(zones) == 2
    assert {(zone["part_id"], tuple(zone["room_ids"])) for zone in zones} == {
        ("building-part:main", ("room:0",)),
        ("building-part:spill", ("room:1",)),
    }


def test_broad_seed_can_survive_on_multiple_owned_components_without_overlap() -> None:
    occupied = [
        _cell(
            "cell-left", 0, "room:0", (0.0, 0.0, 2.0, 2.0), part_id="building-part:left"
        ),
        _cell(
            "cell-right",
            1,
            "room:1",
            (3.0, 0.0, 5.0, 2.0),
            part_id="building-part:right",
        ),
    ]
    subparts = [
        {
            "id": "coverage-subpart:bridge",
            "roof_hypothesis_id": "roof-hypothesis:gable",
            "room_indices": [0, 1],
            "part_ids": ["building-part:left"],
            "semantic_kind": "gable_run",
            "support_evidence_score": 8,
            "adjacent_subpart_ids": [],
            "polygon_xz": [[0.0, 0.0], [5.0, 0.0], [5.0, 2.0], [0.0, 2.0]],
        },
    ]
    roof_building = _roof_building(
        occupied_cells=occupied,
        room_partitions=[
            _room_partition(0, (0.0, 0.0, 2.0, 2.0), "atom-left"),
            _room_partition(1, (3.0, 0.0, 5.0, 2.0), "atom-right"),
        ],
        subparts=subparts,
        room_subpart_membership={
            "room:0": ["coverage-subpart:bridge"],
            "room:1": ["coverage-subpart:bridge"],
        },
        atom_subpart_membership={
            "atom-left": ["coverage-subpart:bridge"],
            "atom-right": ["coverage-subpart:bridge"],
        },
        part_nodes=[
            {"id": "building-part:left", "room_ids": ["room:0"]},
            {"id": "building-part:right", "room_ids": ["room:1"]},
        ],
    )

    zones = derive_authoritative_zones(
        building_uuid=_BLDG,
        roof_building=roof_building,
        bldg=None,
    )

    assert len(zones) == 2
    assert {tuple(zone["room_ids"]) for zone in zones} == {("room:0",), ("room:1",)}
    left_poly = Polygon(zones[0]["footprint_xz"])
    right_poly = Polygon(zones[1]["footprint_xz"])
    assert left_poly.intersection(right_poly).area < 1e-6


def test_real_building_extension_gets_compact_hybrid_zone() -> None:
    roof_results_path = Path("reconcile/roof_algorithms_py_results.json")
    if not roof_results_path.exists():
        return
    uuid = "c87c1e25-ff00-44ec-b823-b0966c81af70"
    with roof_results_path.open() as handle:
        roof_building = json.load(handle)[uuid]
    if "occupied_room_cell_complex" not in roof_building:
        return
    bldg = extract_building(uuid, Path("pipeline-outputs"), None)

    zones = derive_authoritative_zones(
        building_uuid=uuid,
        roof_building=roof_building,
        bldg=bldg,
    )

    extension_zones = [
        zone
        for zone in zones
        if {"room:4", "room:5", "room:6"}.issubset(set(zone["room_ids"]))
    ]
    assert extension_zones
    assert all(zone["footprint_area_m2"] > 2.0 for zone in extension_zones)


def test_target_building_artifact_has_shadow_zone_baseline() -> None:
    artifact_path = Path("reports/candidate_faces/candidates.json")
    if not artifact_path.exists():
        return
    with artifact_path.open() as handle:
        rows = json.load(handle)
    row = next(
        (
            item
            for item in rows
            if item.get("building_uuid") == "c87c1e25-ff00-44ec-b823-b0966c81af70"
        ),
        None,
    )
    if row is None:
        return
    shadow_zones = [
        zone for zone in (row.get("zones") or []) if zone.get("part_id") is None
    ]
    assert any(zone["id"].endswith("906e1ac846c14908506f") for zone in shadow_zones)
