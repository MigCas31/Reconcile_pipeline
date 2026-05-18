from reconcile_tiers.roof.flats import build_flat_surfaces
from tests.reconcile_tiers.roof.helpers import make_two_story_flat_model


def test_flat_surfaces_emit_intermediate_and_top_flats_with_sources():
    flats = build_flat_surfaces(make_two_story_flat_model())

    assert {flat.source for flat in flats} == {"intermediate", "top"}
    assert sum(1 for flat in flats if flat.source == "intermediate") == 1
    assert sum(1 for flat in flats if flat.source == "top") == 1
    assert all(len(flat.corners) == 4 for flat in flats)
