from __future__ import annotations

from math import radians, tan

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.extract.building import (
    BuildingModel,
    ExtractedRoom,
    ExtractedWall,
)
from reconcile_tiers.payload.schema import GableClosureKind
from reconcile_tiers.roof.gable_closures import build_gable_closures
from reconcile_tiers.roof.roof import (
    Footprint,
    ObliqueSurface,
    RoofCluster,
    RoofKinks,
    RoofModel,
)


def _gable_roof_model(
    half_span_z: float = 2.0,
    half_span_x: float = 3.0,
    eave_y: float = 2.0,
    ridge_y: float = 3.5,
    pitch_deg: float = 30.0,
) -> tuple[BuildingModel, RoofModel]:
    """Two opposing oblique surfaces on a rectangular footprint forming a clean gable.

    The roof spans X in [-half_span_x, +half_span_x] and Z in [-half_span_z,
    +half_span_z]. The ridge runs along the X axis at Z=0; both planes start at
    the eaves on each Z side and meet at the ridge.
    """
    incl = pitch_deg
    slope = tan(radians(incl))
    # plane_a (faces +Z): y = ridge_y - slope * z, valid for z >= 0
    # equation: slope * z + y = ridge_y → a*x + b*y + c*z = d with (a, b, c, d) = (0, 1,
    # slope, ridge_y)
    plane_a = Plane(a=0.0, b=1.0, c=slope, d=ridge_y)
    plane_b = Plane(a=0.0, b=1.0, c=-slope, d=ridge_y)

    cluster_a = RoofCluster(
        segments=[], avg_incl=incl, avg_azimuth=0.0, ref_pt=[0.0, ridge_y, 0.0]
    )
    cluster_b = RoofCluster(
        segments=[], avg_incl=incl, avg_azimuth=180.0, ref_pt=[0.0, ridge_y, 0.0]
    )

    surface_a = ObliqueSurface(
        corners=[
            [-half_span_x, eave_y, half_span_z],
            [half_span_x, eave_y, half_span_z],
            [half_span_x, ridge_y, 0.0],
            [-half_span_x, ridge_y, 0.0],
        ],
        plane=plane_a,
        cluster=cluster_a,
        dominant_story=0,
        ridge={"x": 1.0, "z": 0.0, "min": -half_span_x, "max": half_span_x},
    )
    surface_b = ObliqueSurface(
        corners=[
            [-half_span_x, ridge_y, 0.0],
            [half_span_x, ridge_y, 0.0],
            [half_span_x, eave_y, -half_span_z],
            [-half_span_x, eave_y, -half_span_z],
        ],
        plane=plane_b,
        cluster=cluster_b,
        dominant_story=0,
        ridge={"x": 1.0, "z": 0.0, "min": -half_span_x, "max": half_span_x},
    )

    attic_lid_y = (eave_y + ridge_y) / 2.0
    floor = [
        [-half_span_x, 0.0, -half_span_z],
        [half_span_x, 0.0, -half_span_z],
        [half_span_x, 0.0, half_span_z],
        [-half_span_x, 0.0, half_span_z],
    ]
    east_gable_wall = ExtractedWall(
        id="east-gable",
        corners=[
            [half_span_x, 0.0, -half_span_z],
            [half_span_x, 0.0, half_span_z],
            [half_span_x, attic_lid_y, half_span_z],
            [half_span_x, attic_lid_y, -half_span_z],
        ],
        source="test",
    )
    west_gable_wall = ExtractedWall(
        id="west-gable",
        corners=[
            [-half_span_x, 0.0, half_span_z],
            [-half_span_x, 0.0, -half_span_z],
            [-half_span_x, attic_lid_y, -half_span_z],
            [-half_span_x, attic_lid_y, half_span_z],
        ],
        source="test",
    )
    long_wall = ExtractedWall(
        id="long-wall",
        corners=[
            [-half_span_x, 0.0, half_span_z],
            [half_span_x, 0.0, half_span_z],
            [half_span_x, eave_y, half_span_z],
            [-half_span_x, eave_y, half_span_z],
        ],
        source="test",
    )
    room = ExtractedRoom(
        index=0,
        story=0,
        floor_polygon=floor,
        walls_merged=[east_gable_wall, west_gable_wall, long_wall],
        walls_computed=[east_gable_wall, west_gable_wall, long_wall],
        doors=[],
        windows=[],
        openings=[],
        storages=[],
        raw_ceiling_planes=[],
        raw_ceiling_source=None,
        ceiling_polygon=[],
        ceiling_type="sloped",
        ceiling_eave_height=eave_y,
        ceiling_ridge_height=attic_lid_y,
    )
    model = BuildingModel(
        uuid="synthetic-gable",
        address=None,
        stories_found=1,
        split_level=False,
        rooms=[room],
        scan_rooms_found=1,
        scan_rooms_transformed=1,
    )
    kinks = RoofKinks(
        eave_y_by_room={0: eave_y},
        attic_lid_y_by_room={0: attic_lid_y},
        ridge_y=ridge_y,
    )
    roof = RoofModel(
        simple_slant_room_indices=set(),
        segments=[],
        clusters=[cluster_a, cluster_b],
        footprint=Footprint(
            polygon_xz=[
                (-half_span_x, -half_span_z),
                (half_span_x, -half_span_z),
                (half_span_x, half_span_z),
                (-half_span_x, half_span_z),
            ],
            top_story=0,
            area=4.0 * half_span_x * half_span_z,
        ),
        planes=[],
        clipped_planes=[],
        oblique=[surface_a, surface_b],
        flat=[],
        oblique_split=[],
        dormer_candidates=[],
        thermal=[],
        kinks=kinks,
    )
    return model, roof


def test_synthetic_gable_emits_lower_and_upper_per_end():
    model, roof = _gable_roof_model()
    surfaces = build_gable_closures(model, roof)
    by_kind = {GableClosureKind.LOWER: 0, GableClosureKind.UPPER: 0}
    for s in surfaces:
        by_kind[s.kind] += 1
    assert by_kind[GableClosureKind.LOWER] == 2
    assert by_kind[GableClosureKind.UPPER] == 2


def test_synthetic_gable_upper_panel_peaks_at_ridge_y():
    model, roof = _gable_roof_model(eave_y=2.0, ridge_y=3.5)
    surfaces = build_gable_closures(model, roof)
    upper = [s for s in surfaces if s.kind == GableClosureKind.UPPER]
    assert upper, "expected upper panels"
    peak_ys = []
    for s in upper:
        peak_ys.append(max(c[1] for c in s.corners))
    for y in peak_ys:
        assert abs(y - roof.kinks.ridge_y) < 1e-6


def test_synthetic_gable_lower_panel_anchored_at_eave_and_attic_lid():
    eave_y = 2.0
    ridge_y = 3.5
    model, roof = _gable_roof_model(eave_y=eave_y, ridge_y=ridge_y)
    surfaces = build_gable_closures(model, roof)
    lower = [s for s in surfaces if s.kind == GableClosureKind.LOWER]
    assert lower, "expected lower panels"
    attic_lid_y = roof.kinks.attic_lid_y(0)
    for s in lower:
        ys = [c[1] for c in s.corners]
        assert min(ys) >= eave_y - 1e-6
        assert max(ys) <= attic_lid_y + 1e-6


def test_no_gable_means_no_closures():
    model, roof = _gable_roof_model()
    no_ridge_kinks = RoofKinks(
        eave_y_by_room=roof.kinks.eave_y_by_room,
        attic_lid_y_by_room=roof.kinks.attic_lid_y_by_room,
        ridge_y=None,
    )
    flat_roof = RoofModel(
        simple_slant_room_indices=roof.simple_slant_room_indices,
        segments=roof.segments,
        clusters=roof.clusters,
        footprint=roof.footprint,
        planes=roof.planes,
        clipped_planes=roof.clipped_planes,
        oblique=roof.oblique,
        flat=roof.flat,
        oblique_split=roof.oblique_split,
        dormer_candidates=roof.dormer_candidates,
        thermal=roof.thermal,
        kinks=no_ridge_kinks,
    )
    assert build_gable_closures(model, flat_roof) == []
