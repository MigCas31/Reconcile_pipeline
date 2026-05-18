from __future__ import annotations


def build_flat_ceilings(exposed_rooms: list) -> list:
    out = []
    for er in exposed_rooms:
        if (er["wallTopY"] - er["wallTopMin"]) >= 0.3:
            continue
        avg_top = (er["wallTopY"] + er["wallTopMin"]) / 2.0
        if avg_top <= er["floorY"] + 0.1:
            continue
        poly = [(p[0], avg_top, p[2]) for p in er["fp"]]
        out.append({"kind": "flat", "story": er["story"], "poly": poly})
    return out
