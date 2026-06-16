from pathlib import Path

import pytest

from reconcile_tiers.ingest.scan_cache_view import (
    element_world_corners,
    export_merged_wall_overlay,
    export_reconciled_wall_overlay,
    export_scan_building,
    export_tier_payload_wall_overlay,
    list_scan_buildings,
    parse_uuid_from_scan_dir,
)


def test_parse_uuid_from_scan_dir():
    name = (
        "scans_botjek_A.M.G._Friis_Vej_8B_6000_Kolding_"
        "0101eb0f-9eef-43b7-9773-8226d99985f9_1776022042.0933518"
    )
    assert parse_uuid_from_scan_dir(name) == "0101eb0f-9eef-43b7-9773-8226d99985f9"


def test_element_world_corners_from_dimensions():
    corners = element_world_corners(
        {
            "transform": [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                2, 3, 4, 1,
            ],
            "dimensions": [2.0, 4.0, 0.0],
        }
    )
    assert len(corners) == 4
    assert corners[0] == pytest.approx([1.0, 1.0, 4.0])


def test_export_merged_wall_overlay_missing():
    assert export_merged_wall_overlay("00000000-0000-0000-0000-000000000000", Path("pipeline-outputs")) is None


def test_export_reconciled_wall_overlay_missing():
    assert export_reconciled_wall_overlay("00000000-0000-0000-0000-000000000000", Path("pipeline-outputs")) is None


@pytest.mark.skipif(
    not Path("pipeline-outputs").is_dir(),
    reason="pipeline-outputs not available",
)
def test_export_tier_payload_wall_overlay_when_present():
    root = Path("pipeline-outputs")
    for entry in sorted(root.iterdir()):
        tier_path = entry / "tier_payload.json"
        if entry.is_dir() and tier_path.is_file():
            overlay = export_tier_payload_wall_overlay(entry.name, root)
            assert overlay is not None
            assert overlay["wall_count"] > 0
            assert overlay["rooms"][0]["walls"]
            return
    pytest.skip("no tier_payload.json in pipeline-outputs")


@pytest.mark.skipif(
    not Path(".scan-cache").is_dir(),
    reason="scan-cache not available in this checkout",
)
def test_export_scan_building_real_corpus():
    buildings = list_scan_buildings(Path(".scan-cache"))
    assert buildings
    payload = export_scan_building(
        buildings[0]["uuid"],
        Path(".scan-cache"),
        pipeline_dir=Path("pipeline-outputs"),
    )
    assert payload is not None
    assert payload["rooms"]
    assert payload["rooms"][0]["surfaces"]
    overlays = payload.get("pipeline_overlays") or {}
    if overlays.get("merged"):
        assert overlays["merged"]["wall_count"] > 0
    if overlays.get("tier_payload"):
        assert overlays["tier_payload"]["wall_count"] > 0
    assert payload.get("merged_overlay") == overlays.get("merged")
