from __future__ import annotations

from math import atan2, sqrt

from .graph_support import (
    classify_graph_roof_room,
    graph_allows_roof_candidate,
    graph_requires_geometry_fallback,
)

MIN_SEG_LEN = 0.3


def wall_quads(wall: dict) -> list[list]:
    quads: list[list] = []
    corners = wall.get("corners")
    if corners and len(corners) >= 3:
        quads.append(corners)

    extension_strip = wall.get("extension_strip")
    if extension_strip:
        is_nested = (
            isinstance(extension_strip[0][0], (list, tuple))
            if extension_strip and extension_strip[0]
            else False
        )
        strips = extension_strip if is_nested else [extension_strip]
        for q in strips:
            if q and len(q) >= 3:
                quads.append(q)

    return quads


def collect_oblique_segments(
    bldg: dict,
    has_floor_above,
    exclude_room_indices: set[int] | None = None,
    graph=None,
) -> list[dict]:
    segments: list[dict] = []

    for room_idx, room in enumerate(bldg.get("rooms", [])):
        if exclude_room_indices and room_idx in exclude_room_indices:
            continue
        fp = room.get("floor_polygon") or []
        story = int(room.get("story", 0))
        graph_room, roof_decision = classify_graph_roof_room(graph, story, fp)
        if roof_decision is not None and not graph_allows_roof_candidate(roof_decision):
            continue
        for w in room.get("walls_computed", []) or []:
            for corners in wall_quads(w):
                n = len(corners)
                for i in range(n):
                    pa = corners[i]
                    pb = corners[(i + 1) % n]
                    if pa[1] < pb[1]:
                        pa, pb = pb, pa

                    dx, dy, dz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
                    length = sqrt(dx * dx + dy * dy + dz * dz)
                    if length < 1e-4:
                        continue

                    horiz = sqrt(dx * dx + dz * dz)
                    incl = atan2(abs(dy), horiz) * 180.0 / 3.141592653589793
                    if incl <= 5.0 or incl >= 80.0:
                        continue

                    azimuth = (atan2(dx, dz) * 180.0 / 3.141592653589793) % 360.0
                    mx = (pa[0] + pb[0]) / 2.0
                    mz = (pa[2] + pb[2]) / 2.0

                    if length < MIN_SEG_LEN:
                        continue
                    if graph_requires_geometry_fallback(roof_decision):
                        if has_floor_above(mx, mz, story):
                            continue
                    segments.append(
                        {
                            "a": pa,
                            "b": pb,
                            "incl": incl,
                            "azimuth": azimuth,
                            "len": length,
                            "story": story,
                            "room_idx": room_idx,
                            "graph_room_id": graph_room.id
                            if graph_room is not None
                            else None,
                        }
                    )

    return segments


def collect_top_flat_segments(
    bldg: dict,
    max_story: int,
    has_floor_above=None,
    graph=None,
) -> list[dict]:
    top_flat: list[dict] = []

    for room in bldg.get("rooms", []):
        story = int(room.get("story", 0))
        if graph is None and story != max_story:
            continue
        fp = room.get("floor_polygon") or []
        if graph is not None:
            _graph_room, roof_decision = classify_graph_roof_room(graph, story, fp)
            if roof_decision is not None and not graph_allows_roof_candidate(
                roof_decision
            ):
                continue
            if graph_requires_geometry_fallback(roof_decision):
                if has_floor_above is None:
                    if story != max_story:
                        continue
                elif fp:
                    fcx = sum(p[0] for p in fp) / len(fp)
                    fcz = sum(p[2] for p in fp) / len(fp)
                    if has_floor_above(fcx, fcz, story):
                        continue
        for w in room.get("walls_computed", []) or []:
            for corners in wall_quads(w):
                wall_max_y = max(p[1] for p in corners)
                n = len(corners)
                for i in range(n):
                    a = corners[i]
                    b = corners[(i + 1) % n]
                    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
                    length = sqrt(dx * dx + dy * dy + dz * dz)
                    if length < MIN_SEG_LEN:
                        continue

                    horiz = sqrt(dx * dx + dz * dz)
                    incl = atan2(abs(dy), horiz) * 180.0 / 3.141592653589793
                    if incl >= 5.0:
                        continue

                    edge_y = (a[1] + b[1]) / 2.0
                    if abs(edge_y - wall_max_y) > 0.05:
                        continue
                    top_flat.append(
                        {
                            "a": a,
                            "b": b,
                            "len": length,
                            "y": edge_y,
                            "story": story,
                        }
                    )

    return top_flat
