"""Canonical roof pipeline step definitions.

Step IDs and labels are intentionally non-numeric so they remain stable if
steps are inserted, removed, or reordered.
"""

from reconcile_v2.decision_logic import classify_roof_room

ROOF_PIPELINE_STEPS = [
    {
        "id": "story_index_and_bounds",
        "label": "Build story index and vertical bounds",
    },
    {
        "id": "simple_slant_ceilings",
        "label": "Detect simple slanted ceilings (room-level)",
    },
    {
        "id": "collect_oblique_segments",
        "label": "Collect oblique roof edge segments",
    },
    {
        "id": "cluster_oblique_segments",
        "label": "Cluster oblique segments into roof planes",
    },
    {
        "id": "build_oblique_roof_surfaces",
        "label": "Build oblique roof surface polygons",
    },
    {
        "id": "build_flat_roof_surfaces",
        "label": "Build flat roof surface polygons",
    },
    {
        "id": "collect_exposed_rooms_and_planes",
        "label": "Collect exposed rooms and derive ceiling planes",
    },
    {
        "id": "derive_building_footprint",
        "label": "Derive ceiling clipping footprint",
    },
    {
        "id": "clip_ceiling_planes",
        "label": "Clip ceiling planes to footprint and constraints",
    },
    {
        "id": "build_ceiling_surfaces",
        "label": "Build flat and oblique ceiling polygons",
    },
    {
        "id": "detect_dormers",
        "label": "Detect dormers and cut roof openings",
    },
]


def build_roof_graph_context(graph):
    """Extract graph-derived roof exposure context for hybrid migration."""
    context = {"top_story_rooms": [], "roof_candidate_rooms": []}
    for room in graph.nodes_by_type("Room"):
        decision = classify_roof_room(graph, room)
        if decision["is_top_story"]:
            context["top_story_rooms"].append(room.id)
        if decision["mode"] == "roof_candidate":
            context["roof_candidate_rooms"].append(room.id)
    return context
