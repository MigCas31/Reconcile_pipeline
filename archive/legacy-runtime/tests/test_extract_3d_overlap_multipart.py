from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import MultiLineString

from reconcile.extract_3d import _project_line_interval, extract_building


def test_project_line_interval_accepts_multiline_segments() -> None:
    line = MultiLineString(
        [
            [(1.0, 0.0), (2.0, 0.0)],
            [(4.0, 0.0), (5.0, 0.0)],
        ]
    )
    origin = np.array([0.0, 0.0], dtype=float)
    direction = np.array([1.0, 0.0], dtype=float)

    start, end = _project_line_interval(line, origin, direction)

    assert start == 1.0
    assert end == 5.0


def test_extract_building_handles_known_multipart_overlap_uuid() -> None:
    uuid = "a6cb04fa-e84a-4641-a667-b4dd05dd7d41"

    result = extract_building(uuid, Path("pipeline-outputs"), Path(".scan-cache"))

    assert result is not None
    assert result["uuid"] == uuid
    assert result["rooms"]
