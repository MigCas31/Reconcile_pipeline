"""Pin the pipeline stage order — slanted_roofs must run before wall_extensions."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from reconcile_v3 import pipeline as v3_pipeline


def _ordered_call_names() -> list[str]:
    source = inspect.getsource(v3_pipeline.run_pipeline)
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return names


def test_slanted_roofs_before_wall_extensions():
    names = _ordered_call_names()
    slanted = names.index("cluster_oblique_to_roof_faces")
    walls = names.index("extend_walls_to_slab_above")
    assert slanted < walls, (
        "slanted_roofs must run before wall_extensions so knee-wall decisions "
        "can consume SlopeHypothesis objects."
    )


def test_parts_before_gaps_and_slabs():
    names = _ordered_call_names()
    parts = names.index("identify_building_parts")
    gaps = names.index("close_obvious_gaps")
    slabs = names.index("build_room_slabs")
    assert parts < gaps
    assert parts < slabs


def test_fill_remaining_before_dormers():
    names = _ordered_call_names()
    fill = names.index("cover_remaining_slab_above")
    dormers = names.index("detect_dormers_v3")
    assert fill < dormers


def test_pipeline_file_exists():
    path = Path(v3_pipeline.__file__)
    assert path.exists()
