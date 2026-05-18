"""Element lineage/provenance tracking helpers."""

from __future__ import annotations

# Step name constants (matching pipeline order)
STEP_EXTRACT_WALLS = "extract_room_walls"
STEP_DEDUP_WALLS = "dedup_walls"
STEP_EXTRACT_OPENINGS = "extract_openings"
STEP_EXTRACT_STORAGES = "extract_storages"
STEP_CLIP_FLOOR_OVERLAPS = "clip_floor_overlaps"
STEP_ALIGN_ROOM_HEIGHTS = "align_room_heights"
STEP_CLIP_WALLS_STORY = "clip_walls_to_story_bounds"
STEP_EXTEND_WALL_SLAB = "extend_wall_to_slab"
STEP_EXTEND_WALL_DOMINANT = "extend_wall_to_dominant"
STEP_INFER_CEILINGS = "infer_ceilings"
STEP_CROSS_FLOOR_GAPS = "cross_floor_gaps"
STEP_GAP_WALLS = "gap_walls"
STEP_STITCH_WALLS = "stitch_wall_gaps"
STEP_OPENING_CLAMPS = "opening_clamps"


def record(
    element: dict,
    step: str,
    action: str,
    detail: str = "",
) -> None:
    """Append a lineage event to an element dict."""
    entry: dict[str, str] = {"step": step, "action": action}
    if detail:
        entry["detail"] = detail
    element.setdefault("lineage", []).append(entry)
