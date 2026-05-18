from reconcile_tiers.roof.arrangement import split_oblique_surfaces
from reconcile_tiers.roof.clipping import clip_planes_to_footprint
from reconcile_tiers.roof.clustering import cluster_oblique_segments
from reconcile_tiers.roof.footprint import build_building_footprint
from reconcile_tiers.roof.obliques import build_oblique_surfaces, story_floor_y
from reconcile_tiers.roof.planes import build_roof_planes
from reconcile_tiers.roof.segments import collect_oblique_segments
from tests.reconcile_tiers.roof.helpers import make_gable_model


def test_arrangement_emits_stable_split_cell_ids_for_each_oblique():
    model = make_gable_model()
    footprint = build_building_footprint(model)
    planes = build_roof_planes(
        cluster_oblique_segments(collect_oblique_segments(model)), footprint
    )
    clipped = clip_planes_to_footprint(planes, footprint)
    obliques = build_oblique_surfaces(clipped, story_floor_y(model))

    split = split_oblique_surfaces(obliques)

    assert len(split) == len(obliques) == 1
    assert split[0].arrangement_cell_id == "cell:0"
    assert split[0].source_oblique_index == 0
