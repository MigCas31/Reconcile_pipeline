from __future__ import annotations

from pathlib import Path

import pytest

from reconcile.extract3d.builder import extract_building
from reconcile_v2.graph_builder import build_topology_graph

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = ROOT / "pipeline-outputs"
SCAN_CACHE = ROOT / ".scan-cache"
REAL_UUIDS = [
    "0b75d30e-c50c-4fc6-88ff-fce983078aa4",
    "a6cb04fa-e84a-4641-a667-b4dd05dd7d41",
]


@pytest.mark.skipif(
    not PIPELINE_ROOT.exists() or not SCAN_CACHE.exists(),
    reason="real pipeline data unavailable",
)
@pytest.mark.parametrize("uuid", REAL_UUIDS)
def test_topology_graph_builds_on_real_buildings(uuid: str) -> None:
    graph = build_topology_graph(
        merged_path=PIPELINE_ROOT / uuid / "merged.json",
        scan_dir=SCAN_CACHE,
        uuid=uuid,
    )
    assert graph.quality["counts"]["cells"] >= 1
    assert "cell_complex" in graph.geometry_index
    assert graph.quality["graph_validation"]["issue_count"] >= 0


@pytest.mark.skipif(
    not PIPELINE_ROOT.exists() or not SCAN_CACHE.exists(),
    reason="real pipeline data unavailable",
)
@pytest.mark.parametrize("uuid", REAL_UUIDS[:1])
def test_extract3d_builder_loads_graph_on_real_building(uuid: str) -> None:
    result = extract_building(uuid, PIPELINE_ROOT, SCAN_CACHE)
    assert result is not None
    assert result["topology_graph_loaded"] is True


@pytest.mark.skipif(
    not PIPELINE_ROOT.exists() or not SCAN_CACHE.exists(),
    reason="real pipeline data unavailable",
)
@pytest.mark.parametrize("uuid", REAL_UUIDS)
def test_roof_pipeline_produces_reasonable_outputs(uuid: str) -> None:
    """Verify roof pipeline outputs are within physically reasonable bounds."""
    from reconcile.roof_algorithms_py.pipeline import run_roof_algorithms

    building = extract_building(
        uuid, PIPELINE_ROOT, SCAN_CACHE, load_topology_graph=False
    )
    assert building is not None

    roof = run_roof_algorithms(building)
    assert roof is not None

    # Roof cell complex should exist and have reasonable counts
    cell_complex = roof.get("ceiling", {}).get("cell_complex")
    if cell_complex:
        cells = cell_complex.get("cells", [])
        knee_walls = cell_complex.get("knee_walls", [])
        # Physically reasonable: fewer knee walls than 3x cells
        if cells:
            assert len(knee_walls) <= len(cells) * 3, (
                f"Knee-wall count ({len(knee_walls)}) exceeds 3x cell count "
                f"({len(cells)})"
            )
        # Each knee wall should have reasonable area
        for kw in knee_walls:
            if "face_area_m2" in kw:
                assert kw["face_area_m2"] >= 0.1, (
                    f"Knee wall too small: {kw['face_area_m2']}"
                )

    # Should produce some roof surfaces
    roof_surfaces = roof.get("roof_surfaces") or {}
    oblique = roof_surfaces.get("oblique") or []
    flat = roof_surfaces.get("flat") or []
    assert len(oblique) + len(flat) > 0, "No roof surfaces produced"
