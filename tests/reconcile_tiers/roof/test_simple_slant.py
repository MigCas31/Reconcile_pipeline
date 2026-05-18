from reconcile_tiers.roof.roof import build_roof_model
from reconcile_tiers.roof.simple_slant import (
    build_simple_slant_surfaces,
    detect_simple_slant_rooms,
)
from tests.reconcile_tiers.roof.helpers import make_simple_slant_model


def test_simple_slant_prepass_marks_mono_pitch_room():
    model = make_simple_slant_model()

    assert detect_simple_slant_rooms(model) == {0}


def test_simple_slant_rooms_emit_oblique_surface():
    model = make_simple_slant_model()

    surfaces = build_simple_slant_surfaces(model, detect_simple_slant_rooms(model))

    assert len(surfaces) == 1
    assert surfaces[0].dominant_story == 0
    assert surfaces[0].cluster.avg_incl > 5.0
    assert surfaces[0].cluster.avg_azimuth == 90.0


def test_roof_model_counts_simple_slant_as_oblique_surface():
    roof = build_roof_model(make_simple_slant_model())

    assert roof.simple_slant_room_indices == {0}
    assert roof.segments == []
    assert len(roof.oblique) == 1
