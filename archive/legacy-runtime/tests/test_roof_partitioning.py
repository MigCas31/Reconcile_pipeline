from __future__ import annotations

from shapely.geometry import Polygon
from shapely.validation import explain_validity

from reconcile.roof_algorithms_py.roof_partitioning import (
    derive_room_ceiling_partitions,
)


def _project_xz(
    poly3d: list[list[float]] | list[tuple[float, float, float]],
) -> Polygon:
    return Polygon([(float(x), float(z)) for x, _, z in poly3d])


def test_derive_room_ceiling_partitions_repairs_snapped_oblique_atom_regression() -> (
    None
):
    exposed_room = {
        "room_index": 6,
        "story": 1,
        "graph_room_id": "room:merged_room_6",
        "fp": [
            [3.667, -1.6833567615813962, -3.0285],
            [3.4695, -1.6833567615813962, -0.501],
            [-0.9279, -1.6833567615813962, -0.8445],
            [-0.9341, -1.6833567615813962, -0.7653],
            [-1.238, -1.6833567615813962, -0.789],
            [0.4874, -1.6833567615813962, -0.6543],
            [0.4994, -1.6833567615813962, -0.6533],
            [0.4234, -1.6833567615813962, 0.3195],
            [4.3883, -1.6833567615813962, 0.6292],
            [4.425, -1.6833567615813962, 0.1585],
            [5.5533, -1.6833567615813962, 0.2466],
            [5.5501, -1.6833567615813962, 0.2866],
            [5.5143, -1.6833567615813962, 0.7451],
            [8.8912, -1.6833567615813962, 1.0089],
            [8.9118, -1.6833567615813962, 0.7448],
            [6.0334, -1.6833567615813962, 0.52],
            [6.0517, -1.6833567615813962, 0.2855],
            [5.5706, -1.6833567615813962, 0.248],
            [8.9301, -1.6833567615813962, 0.5104],
            [9.1846, -1.6833567615813962, -2.7481],
            [7.4563, -1.6833567615813962, -2.8831],
            [7.2802, -1.6833567615813962, -0.6281],
            [7.4461, -1.6833567615813962, -2.7517],
            [3.6607, -1.6833567615813962, -3.0532],
            [7.398, -1.6833567615813962, -2.7555],
            [7.2344, -1.6833567615813962, -0.7018],
            [6.2928, -1.6833567615813962, -0.7768],
            [3.5069, -1.6833567615813962, -0.9794],
        ],
        "floorY": -1.6833567615813962,
        "wallTopY": 3.2809763484186033,
        "wallTopMin": -0.5635356815813965,
        "wallTopYs": [
            0.6648259184186033,
            0.6648259184186033,
            0.6648259184186033,
            0.6648259184186033,
            0.6648257184186034,
            0.6648257184186034,
            0.6648257184186034,
            0.6648257184186034,
            0.6648257184186034,
            0.6648257184186034,
            0.6648257184186034,
        ],
        "top_boundary_mode": "roof_candidate",
        "top_boundary_reason": "top_story_without_above_relation",
    }
    oblique_roof_surfaces = [
        {
            "roof_hypothesis_id": "roof-hypothesis:oblique:0",
            "corners": [
                [8.909, 3.253, 0.778],
                [8.942, 2.791, 0.354],
                [9.185, -0.591, -2.748],
                [7.456, -0.591, -2.883],
                [7.446, -0.448, -2.752],
                [6.511, -0.45, -2.826],
                [6.542, -0.865, -3.207],
                [6.334, -0.865, -3.224],
                [6.352, -1.113, -3.451],
                [5.159, -1.117, -3.547],
                [5.11, -0.452, -2.938],
                [3.669, -0.454, -3.053],
                [2.464, -0.456, -3.148],
                [2.462, -0.428, -3.123],
                [-0.669, -0.428, -3.367],
                [-0.668, -0.441, -3.379],
                [-4.503, -0.441, -3.679],
                [-4.768, 3.253, -0.29],
            ],
            "cluster": {
                "avgAzimuth": 175.53314138105878,
                "avgIncl": 47.38319423762699,
                "refPt": {
                    "x": 9.141543439256612,
                    "y": 0.03750290641860338,
                    "z": -2.196298150688173,
                },
            },
        },
        {
            "roof_hypothesis_id": "roof-hypothesis:oblique:1",
            "corners": [
                [-4.951, -0.431, 2.836],
                [-5.016, -1.372, 3.634],
                [-0.027, -1.389, 4.038],
                [-0.029, -1.427, 4.07],
                [8.433, -1.457, 4.757],
                [8.506, -0.391, 3.854],
                [8.684, -0.391, 3.867],
                [8.889, 2.705, 1.245],
                [8.873, 2.705, 1.243],
                [8.909, 3.253, 0.778],
                [-4.71, 3.253, -0.285],
                [-4.953, -0.431, 2.836],
            ],
            "cluster": {
                "avgAzimuth": 355.53312410748003,
                "avgIncl": 49.638589414011214,
                "refPt": {
                    "x": 4.2756725217177465,
                    "y": 0.12401340759176435,
                    "z": 3.019676678689593,
                },
            },
        },
    ]
    hypothesis_graph = {
        "nodes": [
            {
                "id": "roof-hypothesis:oblique:0",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "selected": True,
            },
            {
                "id": "roof-hypothesis:oblique:1",
                "type": "RoofHypothesis",
                "surface_kind": "oblique",
                "selected": True,
            },
        ],
        "edges": [
            {
                "id": "edge:covers:0",
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:oblique:0",
                "to": "room:6",
                "selected": True,
                "evidence": {"edge_score": 0.814363},
            },
            {
                "id": "edge:covers:1",
                "type": "COVERS_ROOM",
                "from": "roof-hypothesis:oblique:1",
                "to": "room:6",
                "selected": False,
                "evidence": {"edge_score": 0.463132},
            },
        ],
        "selected_room_assignments": {"room:6": ["roof-hypothesis:oblique:0"]},
        "selected_hypothesis_ids": [
            "roof-hypothesis:oblique:0",
            "roof-hypothesis:oblique:1",
        ],
    }

    partitions = derive_room_ceiling_partitions(
        room_records=[exposed_room],
        oblique_roof_surfaces=oblique_roof_surfaces,
        flat_roof_surfaces=[],
        hypothesis_graph=hypothesis_graph,
    )

    room_partition = partitions["room_partitions"][0]
    large_oblique = max(
        (
            partition
            for partition in room_partition["partitions"]
            if partition["kind"] == "oblique"
        ),
        key=lambda partition: partition["area_m2"],
    )

    projected = _project_xz(large_oblique["poly"])
    assert projected.is_valid, explain_validity(projected)
    assert round(projected.area, 3) == 12.976
    assert large_oblique["roof_hypothesis_id"] == "roof-hypothesis:oblique:0"
