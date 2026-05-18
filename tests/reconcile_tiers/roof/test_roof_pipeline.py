from reconcile_tiers.payload.schema import KneeWallKind
from reconcile_tiers.roof.roof import build_roof_model
from tests.reconcile_tiers.roof.helpers import make_gable_model


def test_build_roof_model_returns_stage_outputs_not_just_final_surfaces():
    # room_depth=6 keeps the parent roof's ridge_span >= MIN_PARENT_RIDGE_SPAN_M (5 m),
    # so the production dormer gate (which rejects short room-local slants) accepts it.
    roof = build_roof_model(make_gable_model(include_dormer=True, room_depth=6.0))

    assert roof.simple_slant_room_indices == set()
    assert len(roof.segments) >= 2
    assert len(roof.clusters) == 1
    assert roof.footprint is not None
    assert len(roof.planes) == 1
    assert len(roof.clipped_planes) == 1
    assert len(roof.oblique) == 1
    assert len(roof.oblique_split) == 1
    assert len(roof.dormer_candidates) == 1
    assert KneeWallKind.KNEE in {surface.kind for surface in roof.thermal}
    # Dormer cheek/header thermals are produced by reconstruct_dormers, not
    # build_roof_model.
    assert all(surface.kind == KneeWallKind.KNEE for surface in roof.thermal)
