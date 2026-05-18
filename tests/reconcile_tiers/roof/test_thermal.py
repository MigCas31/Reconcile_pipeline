import pytest

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.extract.building import (
    BuildingModel,
    ExtractedRoom,
    RawCeilingPlane,
)
from reconcile_tiers.payload.schema import KneeWallKind
from reconcile_tiers.roof.clipping import clip_planes_to_footprint
from reconcile_tiers.roof.clustering import cluster_oblique_segments
from reconcile_tiers.roof.footprint import build_building_footprint
from reconcile_tiers.roof.obliques import build_oblique_surfaces, story_floor_y
from reconcile_tiers.roof.planes import build_roof_planes
from reconcile_tiers.roof.roof import ObliqueSurface, RoofCluster
from reconcile_tiers.roof.segments import collect_oblique_segments
from reconcile_tiers.roof.thermal import (
    THERMAL_KINDS,
    attic_lid_y_by_room,
    build_thermal_surfaces,
)
from tests.reconcile_tiers.roof.helpers import _wall, make_gable_model


def _flat_plane(y: float) -> RawCeilingPlane:
    return RawCeilingPlane(
        corners=[
            [0.0, y, 0.0],
            [1.0, y, 0.0],
            [1.0, y, 1.0],
            [0.0, y, 1.0],
        ]
    )


def _room_with_planes(
    *,
    index: int,
    apex: float | None,
    raw_planes: list[RawCeilingPlane],
    story: int = 0,
) -> ExtractedRoom:
    return ExtractedRoom(
        index=index,
        story=story,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        walls_merged=[],
        walls_computed=[],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=raw_planes,
        raw_ceiling_source="test",
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=apex,
    )


def test_attic_lid_picks_mid_pitch_flat_below_apex():
    """A flat raw plane sitting clearly below the apex marks a real attic
    conversion — lid = p90 of those mid-pitch flats."""
    room = _room_with_planes(
        index=0,
        apex=4.0,
        raw_planes=[_flat_plane(2.4), _flat_plane(2.4), _flat_plane(2.5)],
    )
    model = BuildingModel("synthetic", None, 1, False, [room], 1, 1)

    lids = attic_lid_y_by_room(model, [])

    # p90 of [2.4, 2.4, 2.5] = idx 0.9*2 = 1.8, interp 2.4 + 0.8*0.1 = 2.48.
    assert lids[0] == pytest.approx(2.48, abs=1e-9)


def test_attic_lid_pins_to_apex_when_only_apex_flat_exists():
    """A flat raw plane near the apex doesn't count — it's the cathedral
    apex itself, not a mid-pitch lid. Lid = apex."""
    room = _room_with_planes(
        index=0,
        apex=4.0,
        raw_planes=[_flat_plane(3.95)],
    )
    model = BuildingModel("synthetic", None, 1, False, [room], 1, 1)

    lids = attic_lid_y_by_room(model, [])

    assert lids[0] == pytest.approx(4.0, abs=1e-9)


def test_attic_lid_pins_to_apex_when_no_flat_planes():
    """Cathedral case: no flat raw plane (only sloped scan). Lid = apex."""
    sloped = RawCeilingPlane(
        corners=[
            [0.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    room = _room_with_planes(index=0, apex=4.0, raw_planes=[sloped])
    model = BuildingModel("synthetic", None, 1, False, [room], 1, 1)

    lids = attic_lid_y_by_room(model, [])

    assert lids[0] == pytest.approx(4.0, abs=1e-9)


def test_attic_lid_unset_when_no_apex_and_no_flat_below():
    """No apex (`ceiling_ridge_height is None`) and no qualifying flat
    plane → lid stays unset (the caller treats this as cathedral / no
    signal)."""
    room = _room_with_planes(index=0, apex=None, raw_planes=[])
    model = BuildingModel("synthetic", None, 1, False, [room], 1, 1)

    lids = attic_lid_y_by_room(model, [])

    assert 0 not in lids


def test_attic_lid_tolerates_noisy_flat_plane():
    """A raw plane with small Y noise (< LID_RAW_PLANE_DELTA_M = 10cm) still
    counts as flat."""
    noisy = RawCeilingPlane(
        corners=[
            [0.0, 2.40, 0.0],
            [1.0, 2.46, 0.0],
            [1.0, 2.43, 1.0],
            [0.0, 2.41, 1.0],
        ]
    )
    room = _room_with_planes(index=0, apex=4.0, raw_planes=[noisy])
    model = BuildingModel("synthetic", None, 1, False, [room], 1, 1)

    lids = attic_lid_y_by_room(model, [])

    expected_mean = (2.40 + 2.46 + 2.43 + 2.41) / 4.0
    assert lids[0] == pytest.approx(expected_mean, abs=1e-9)


def test_thermal_surfaces_emit_only_knee_walls_not_caps():
    model = make_gable_model(include_dormer=True)
    footprint = build_building_footprint(model)
    planes = build_roof_planes(
        cluster_oblique_segments(collect_oblique_segments(model)), footprint
    )
    clipped = clip_planes_to_footprint(planes, footprint)
    obliques = build_oblique_surfaces(clipped, story_floor_y(model))

    thermal = build_thermal_surfaces(model, obliques)
    kinds = {surface.kind for surface in thermal}

    assert kinds <= THERMAL_KINDS
    assert KneeWallKind.KNEE in kinds
    assert kinds == {KneeWallKind.KNEE}
    assert all("cap" not in surface.kind.value for surface in thermal)


def test_thermal_knee_wall_requires_wall_top_under_oblique_surface_support():
    wall = _wall(
        "wall-outside-roof-face",
        [
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 2.0],
            [10.0, 1.0, 2.0],
            [10.0, 1.0, 0.0],
        ],
    )
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [9.0, 0.0, -1.0],
            [11.0, 0.0, -1.0],
            [11.0, 0.0, 3.0],
            [9.0, 0.0, 3.0],
        ],
        walls_merged=[wall],
        walls_computed=[wall],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )
    model = BuildingModel("synthetic-offset", None, 1, False, [room], 1, 1)
    cluster = RoofCluster(
        segments=[], avg_incl=15.0, avg_azimuth=270.0, ref_pt=[0.0, 2.0, 0.0]
    )
    nearby_oblique = ObliqueSurface(
        corners=[
            [0.0, 2.0, 0.0],
            [2.0, 2.4, 0.0],
            [2.0, 2.4, 2.0],
            [0.0, 2.0, 2.0],
        ],
        plane=Plane(a=-0.2, b=1.0, c=0.0, d=2.0),
        cluster=cluster,
        dominant_story=0,
        ridge={},
    )

    assert build_thermal_surfaces(model, [nearby_oblique]) == []


def test_thermal_knee_wall_does_not_extend_above_existing_room_max_y():
    wall = _wall(
        "wall-under-unscanned-roof-face",
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [0.0, 2.0, 0.0],
        ],
    )
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        walls_merged=[wall],
        walls_computed=[wall],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type=None,
        ceiling_eave_height=None,
        ceiling_ridge_height=None,
    )
    model = BuildingModel("synthetic-above-room-max", None, 1, False, [room], 1, 1)
    surface = ObliqueSurface(
        corners=[
            [-0.5, 3.0, -0.5],
            [-0.5, 3.0, 2.5],
            [0.5, 3.5, 2.5],
            [0.5, 3.5, -0.5],
        ],
        plane=Plane(a=-0.5, b=1.0, c=0.0, d=3.0),
        cluster=RoofCluster(
            segments=[], avg_incl=20.0, avg_azimuth=270.0, ref_pt=[0.0, 3.0, 0.0]
        ),
        dominant_story=0,
        ridge={},
    )

    assert build_thermal_surfaces(model, [surface]) == []
