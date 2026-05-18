from __future__ import annotations

from .roof_flat_geometry import bbox_surface, cluster_top_flat_segments
from .segment_collection import collect_top_flat_segments


def collect_top_story_flat_surfaces(
    *,
    bldg: dict,
    all_stories: list,
    has_floor_above=None,
    bldg_min_y: float,
    graph=None,
) -> tuple[list, list]:
    flat_surfaces = []
    roof_legend_flat = []

    max_story = max(all_stories or [0])
    top_flat_segments = collect_top_flat_segments(
        bldg,
        max_story,
        has_floor_above=has_floor_above,
        graph=graph,
    )
    valid_top = cluster_top_flat_segments(top_flat_segments)

    for cl in valid_top:
        all_pts = [p for s in cl["segs"] for p in (s["a"], s["b"])]
        corners = bbox_surface(all_pts, cl["avgY"])
        story_counts = {}
        for seg in cl["segs"]:
            story = int(seg.get("story", max_story))
            story_counts[story] = story_counts.get(story, 0) + 1
        dominant_story = sorted(
            story_counts.items(), key=lambda kv: kv[1], reverse=True
        )[0][0]
        flat_surfaces.append(
            {
                "kind": "top",
                "story": dominant_story,
                "y": cl["avgY"],
                "segments": cl["segs"],
                "corners": corners,
            }
        )
        roof_legend_flat.append(
            {
                "count": len(cl["segs"]),
                "avgIncl": 0.0,
                "avgAzimuth": None,
                "flat": True,
                "height": (cl["avgY"] - bldg_min_y)
                if bldg_min_y != float("inf")
                else cl["avgY"],
            }
        )

    return flat_surfaces, roof_legend_flat
