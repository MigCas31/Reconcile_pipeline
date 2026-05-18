from __future__ import annotations

from .roof_flat_intermediate import collect_intermediate_flat_surfaces
from .roof_flat_top import collect_top_story_flat_surfaces


def build_flat_roof_surfaces(
    bldg: dict, all_stories: list, has_floor_above, bldg_min_y: float, graph=None
) -> tuple:
    intermediate_surfaces, intermediate_legend = collect_intermediate_flat_surfaces(
        bldg=bldg,
        has_floor_above=has_floor_above,
        bldg_min_y=bldg_min_y,
        graph=graph,
    )
    top_surfaces, top_legend = collect_top_story_flat_surfaces(
        bldg=bldg,
        all_stories=all_stories,
        has_floor_above=has_floor_above,
        bldg_min_y=bldg_min_y,
        graph=graph,
    )
    return intermediate_surfaces + top_surfaces, intermediate_legend + top_legend
