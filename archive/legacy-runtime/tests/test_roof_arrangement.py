from __future__ import annotations

from shapely.geometry import Polygon

from reconcile.roof_algorithms_py.math_utils import plane_normal, plane_y_at
from reconcile.roof_algorithms_py.roof_arrangement import build_roof_arrangement


def _surface(
    *,
    index: int,
    azimuth: float,
    incl: float,
    ref: tuple[float, float, float],
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    source: str = "wall_segments",
    raw_plane_ids: list[str] | None = None,
    support_score: float = 0.7,
) -> dict:
    plane = {
        "n": plane_normal(azimuth, incl),
        "ref": {"x": ref[0], "y": ref[1], "z": ref[2]},
    }

    def p(x: float, z: float) -> list[float]:
        return [x, plane_y_at(plane, x, z), z]

    cluster = {
        "avgAzimuth": azimuth,
        "avgIncl": incl,
        "refPt": {"x": ref[0], "y": ref[1], "z": ref[2]},
        "room_indices": [0],
        "source": source,
        "segs": [
            {"a": p(x0, z0), "b": p(x1, z0)},
            {"a": p(x0, z1), "b": p(x1, z1)},
        ],
    }
    if raw_plane_ids:
        cluster["raw_plane_ids"] = raw_plane_ids
    return {
        "kind": "oblique",
        "surface_kind": "oblique",
        "dominant_story": 0,
        "corners": [p(x0, z0), p(x1, z0), p(x1, z1), p(x0, z1)],
        "cluster": cluster,
        "roof_hypothesis_id": f"roof-hypothesis:oblique:{index}",
        "roof_hypothesis_support_score": support_score,
    }


def _overlap_area(pieces: list[dict]) -> float:
    total = 0.0
    polys = [Polygon([(c[0], c[2]) for c in p["corners"]]) for p in pieces]
    for idx, left in enumerate(polys):
        for right in polys[idx + 1 :]:
            total += left.intersection(right).area
    return total


def test_simple_gable_arrangement_splits_at_ridge_without_overlap() -> None:
    left = _surface(
        index=0,
        azimuth=180.0,
        incl=45.0,
        ref=(0.0, 3.0, 0.0),
        x0=-3.0,
        z0=-2.0,
        x1=3.0,
        z1=0.0,
    )
    right = _surface(
        index=1,
        azimuth=0.0,
        incl=45.0,
        ref=(0.0, 3.0, 0.0),
        x0=-3.0,
        z0=0.0,
        x1=3.0,
        z1=2.0,
    )

    result = build_roof_arrangement(
        bldg={"uuid": "gable", "rooms": []},
        oblique_surfaces=[left, right],
        building_footprint=[
            [-3.0, 0.0, -2.0],
            [3.0, 0.0, -2.0],
            [3.0, 0.0, 2.0],
            [-3.0, 0.0, 2.0],
        ],
    )

    assert result["metadata"]["cell_count"] >= 2
    assert any(edge["kind"] == "ridge" for edge in result["edges"])
    assert _overlap_area(result["oblique_split"]) <= 1e-6


def test_near_ridge_gap_extends_to_equal_height_seam() -> None:
    left = _surface(
        index=0,
        azimuth=180.0,
        incl=45.0,
        ref=(0.0, 3.0, 0.0),
        x0=-3.0,
        z0=-2.0,
        x1=3.0,
        z1=-0.2,
    )
    right = _surface(
        index=1,
        azimuth=0.0,
        incl=45.0,
        ref=(0.0, 3.0, 0.0),
        x0=-3.0,
        z0=0.2,
        x1=3.0,
        z1=2.0,
    )

    result = build_roof_arrangement(
        bldg={"uuid": "gable-gap", "rooms": []},
        oblique_surfaces=[left, right],
        building_footprint=[
            [-3.0, 0.0, -2.0],
            [3.0, 0.0, -2.0],
            [3.0, 0.0, 2.0],
            [-3.0, 0.0, 2.0],
        ],
    )

    assert any(edge["kind"] == "ridge" for edge in result["edges"])
    assert _overlap_area(result["oblique_split"]) <= 1e-6
    polygons = [
        Polygon([(corner[0], corner[2]) for corner in piece["corners"]])
        for piece in result["oblique_split"]
    ]
    covered = sum(
        poly.intersection(
            Polygon([(-3, -0.05), (3, -0.05), (3, 0.05), (-3, 0.05)])
        ).area
        for poly in polygons
    )
    assert covered > 0.5


def test_perpendicular_faces_are_split_by_hip_or_valley_cells() -> None:
    north = _surface(
        index=0,
        azimuth=0.0,
        incl=35.0,
        ref=(0.0, 3.0, 0.0),
        x0=-2.0,
        z0=-2.0,
        x1=2.0,
        z1=2.0,
    )
    east = _surface(
        index=1,
        azimuth=90.0,
        incl=35.0,
        ref=(0.0, 3.0, 0.0),
        x0=-2.0,
        z0=-2.0,
        x1=2.0,
        z1=2.0,
    )

    result = build_roof_arrangement(
        bldg={"uuid": "junction", "rooms": []},
        oblique_surfaces=[north, east],
        building_footprint=[
            [-2.0, 0.0, -2.0],
            [2.0, 0.0, -2.0],
            [2.0, 0.0, 2.0],
            [-2.0, 0.0, 2.0],
        ],
    )

    assert result["metadata"]["cell_count"] >= 2
    assert any(edge["kind"] in {"hip", "valley"} for edge in result["edges"])
    assert any(
        cell["intersection_kind"] in {"hip", "valley"} for cell in result["cells"]
    )
    assert _overlap_area(result["oblique_split"]) <= 1e-6


def test_clean_raw_rectangle_can_own_extension_cell_beyond_sparse_surface() -> None:
    uuid = "raw-owner"
    raw_id = f"{uuid}::ceiling-raw::0:0:0"
    raw = _surface(
        index=0,
        azimuth=90.0,
        incl=30.0,
        ref=(0.0, 3.0, 0.0),
        x0=0.0,
        z0=0.0,
        x1=2.0,
        z1=2.0,
        source="raw_ceiling_rectangle",
        raw_plane_ids=[raw_id],
        support_score=0.55,
    )
    competitor = _surface(
        index=1,
        azimuth=270.0,
        incl=30.0,
        ref=(3.0, 3.0, 0.0),
        x0=2.0,
        z0=0.0,
        x1=4.0,
        z1=2.0,
    )
    bldg = {
        "uuid": uuid,
        "rooms": [
            {
                "story": 0,
                "raw_ceiling_planes": [
                    {
                        "corners": [
                            [0.0, 3.0, 0.0],
                            [4.0, 0.691, 0.0],
                            [4.0, 0.691, 2.0],
                            [0.0, 3.0, 2.0],
                        ]
                    }
                ],
            }
        ],
    }

    result = build_roof_arrangement(
        bldg=bldg,
        oblique_surfaces=[raw, competitor],
        building_footprint=[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
    )

    raw_cells = [
        cell
        for cell in result["cells"]
        if cell["owner_roof_hypothesis_id"] == "roof-hypothesis:oblique:0"
        and cell["evidence"]["raw_rectangle_overlap_ratio"] >= 0.3
    ]
    assert raw_cells
    assert max(corner[0] for cell in raw_cells for corner in cell["corners"]) > 2.5


def test_extended_scan_segments_fill_narrow_eave_gap() -> None:
    surface = _surface(
        index=0,
        azimuth=180.0,
        incl=35.0,
        ref=(0.0, 3.0, 0.0),
        x0=0.0,
        z0=0.4,
        x1=4.0,
        z1=2.0,
    )
    plane = {
        "n": plane_normal(180.0, 35.0),
        "ref": {"x": 0.0, "y": 3.0, "z": 0.0},
    }

    def p(x: float, z: float) -> list[float]:
        return [x, plane_y_at(plane, x, z), z]

    surface["cluster"]["segs"] = [
        {"a": p(1.0, 0.4), "b": p(1.0, 1.8)},
        {"a": p(3.0, 0.4), "b": p(3.0, 1.8)},
    ]
    bldg = {
        "uuid": "segment-eave-gap",
        "rooms": [
            {
                "story": 0,
                "floor_polygon": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            }
        ],
    }

    result = build_roof_arrangement(
        bldg=bldg,
        oblique_surfaces=[surface],
        building_footprint=[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
    )

    polygons = [
        Polygon([(corner[0], corner[2]) for corner in piece["corners"]])
        for piece in result["oblique_split"]
    ]
    eave_strip = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 0.4), (0.0, 0.4)])
    covered = sum(poly.intersection(eave_strip).area for poly in polygons)
    assert covered > 1.5


def test_triangle_raw_plane_does_not_create_raw_rectangle_ownership() -> None:
    uuid = "raw-triangle"
    raw_id = f"{uuid}::ceiling-raw::0:0:0"
    surface = _surface(
        index=0,
        azimuth=90.0,
        incl=30.0,
        ref=(0.0, 3.0, 0.0),
        x0=0.0,
        z0=0.0,
        x1=2.0,
        z1=2.0,
        source="raw_ceiling_rectangle",
        raw_plane_ids=[raw_id],
    )
    bldg = {
        "uuid": uuid,
        "rooms": [
            {
                "story": 0,
                "raw_ceiling_planes": [
                    {"corners": [[0.0, 3.0, 0.0], [2.0, 1.8, 0.0], [0.0, 3.0, 2.0]]}
                ],
            }
        ],
    }

    result = build_roof_arrangement(
        bldg=bldg,
        oblique_surfaces=[surface],
        building_footprint=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
    )

    assert result["cells"]
    assert all(
        cell["evidence"]["raw_rectangle_overlap_ratio"] == 0.0
        for cell in result["cells"]
    )
