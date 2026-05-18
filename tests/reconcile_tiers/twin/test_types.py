"""Phase A invariant tests: every primitive's `__post_init__` must reject
the proposed geometry it claims to reject. No assembly algorithm is tested
here — those tests arrive in Phase B."""

from __future__ import annotations

import pytest

from reconcile_tiers.payload.schema import Plane, Vec3
from reconcile_tiers.twin import (
    FLOAT_EPS,
    Building,
    Ceiling,
    CeilingKind,
    CeilingSeam,
    Dormer,
    Eave,
    Evidence,
    Floor,
    Gap,
    GapKind,
    InvariantViolation,
    KneeWall,
    Opening,
    OpeningKind,
    Provenance,
    Residual,
    Ridge,
    Roof,
    RoofKind,
    RoofSurface,
    Room,
    Stair,
    Story,
    Twin,
    Wall,
    Wing,
)


def _scan_evidence(geometry: tuple[Vec3, ...] = ()) -> Evidence:
    return Evidence(
        provenance=Provenance(kind="scan", source="test"),
        geometry=geometry,
    )


def _computed_evidence(
    parents: tuple[str, ...] = ("p1",), geometry: tuple[Vec3, ...] = ()
) -> Evidence:
    return Evidence(
        provenance=Provenance(kind="computed", source="test_synth"),
        geometry=geometry,
        parents=parents,
    )


# ----- Provenance / Evidence ---------------------------------------------


def test_evidence_scan_with_parents_rejected():
    with pytest.raises(InvariantViolation, match="must not have parents"):
        Evidence(
            provenance=Provenance(kind="scan", source="x"),
            geometry=(),
            parents=("p1",),
        )


def test_evidence_computed_without_parents_rejected():
    with pytest.raises(InvariantViolation, match="has no parents"):
        Evidence(
            provenance=Provenance(kind="computed", source="x"),
            geometry=(),
        )


# ----- Floor --------------------------------------------------------------


def _square(y: float = 0.0) -> tuple[Vec3, ...]:
    return (
        Vec3(0.0, y, 0.0),
        Vec3(1.0, y, 0.0),
        Vec3(1.0, y, 1.0),
        Vec3(0.0, y, 1.0),
    )


def test_floor_constructs_when_horizontal_and_evidenced():
    Floor(id="f1", polygon=_square(0.0), evidence=(_scan_evidence(),))


def test_floor_rejects_non_horizontal():
    skewed = (
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.5, 0.0),  # tilted corner
        Vec3(1.0, 0.0, 1.0),
        Vec3(0.0, 0.0, 1.0),
    )
    with pytest.raises(InvariantViolation, match="not horizontal"):
        Floor(id="f1", polygon=skewed, evidence=(_scan_evidence(),))


def test_floor_rejects_few_corners():
    with pytest.raises(InvariantViolation, match="<3 corners"):
        Floor(
            id="f1",
            polygon=(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)),
            evidence=(_scan_evidence(),),
        )


def test_floor_rejects_no_evidence():
    with pytest.raises(InvariantViolation, match="has no evidence"):
        Floor(id="f1", polygon=_square(), evidence=())


# ----- Wall --------------------------------------------------------------


def _vertical_plane_x_eq_zero() -> Plane:
    # x = 0 has normal (1, 0, 0); plane.b is the Y component.
    return Plane(a=1.0, b=0.0, c=0.0, d=0.0)


def _horizontal_plane_y_eq_zero() -> Plane:
    return Plane(a=0.0, b=1.0, c=0.0, d=0.0)


def _rect_wall_corners() -> tuple:
    return (
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 2.5, 1.0),
        Vec3(0.0, 2.5, 0.0),
    )


def test_wall_constructs_when_vertical_and_evidenced():
    Wall(
        id="w1",
        polygon=_rect_wall_corners(),
        plane=_vertical_plane_x_eq_zero(),
        evidence=(_scan_evidence(),),
    )


def test_wall_rejects_horizontal_plane():
    with pytest.raises(InvariantViolation, match="plane is horizontal"):
        Wall(
            id="w1",
            polygon=_rect_wall_corners(),
            plane=_horizontal_plane_y_eq_zero(),
            evidence=(_scan_evidence(),),
        )


def test_wall_constructs_pentagonal_gable_end():
    """A real 5-corner gable end wall: 2 bottom + 2 mid-top + 1 peak.
    The corpus has 16 of these in bf264fa6 alone; the type must
    accept polygons of any corner count, not just rectangles."""
    Wall(
        id="w_gable_end",
        polygon=(
            Vec3(0.0, 0.0, 0.0),  # bottom
            Vec3(0.0, 0.0, 4.0),  # bottom
            Vec3(0.0, 2.5, 4.0),  # eave-level top
            Vec3(0.0, 4.0, 2.0),  # peak
            Vec3(0.0, 2.5, 0.0),  # eave-level top
        ),
        plane=_vertical_plane_x_eq_zero(),
        evidence=(_scan_evidence(),),
    )


def test_wall_rejects_zero_height():
    with pytest.raises(InvariantViolation, match="zero height"):
        Wall(
            id="w1",
            polygon=(
                Vec3(0.0, 2.5, 0.0),
                Vec3(0.0, 2.5, 1.0),
                Vec3(0.0, 2.5, 0.5),
            ),
            plane=_vertical_plane_x_eq_zero(),
            evidence=(_scan_evidence(),),
        )


def test_wall_rejects_corner_off_plane():
    with pytest.raises(InvariantViolation, match="exceeds float precision"):
        Wall(
            id="w1",
            polygon=(
                Vec3(0.0, 0.0, 0.0),
                Vec3(0.0, 0.0, 1.0),
                Vec3(0.5, 2.5, 1.0),  # x=0.5 — off the x=0 plane
                Vec3(0.0, 2.5, 0.0),
            ),
            plane=_vertical_plane_x_eq_zero(),
            evidence=(_scan_evidence(),),
        )


def test_wall_opening_must_lie_in_plane():
    bad_opening = Opening(
        id="o1",
        kind=OpeningKind.WINDOW,
        polygon=(
            Vec3(0.5, 1.0, 0.2),
            Vec3(0.5, 1.0, 0.6),
            Vec3(0.5, 1.5, 0.6),
            Vec3(0.5, 1.5, 0.2),
        ),
        evidence=(_scan_evidence(),),
    )
    with pytest.raises(InvariantViolation, match="escapes wall plane"):
        Wall(
            id="w1",
            polygon=_rect_wall_corners(),
            plane=_vertical_plane_x_eq_zero(),
            openings=(bad_opening,),
            evidence=(_scan_evidence(),),
        )


# ----- Opening ------------------------------------------------------------


def test_opening_rejects_non_planar_corners():
    with pytest.raises(InvariantViolation, match="not coplanar"):
        Opening(
            id="o1",
            kind=OpeningKind.WINDOW,
            polygon=(
                Vec3(0.0, 1.0, 0.0),
                Vec3(0.0, 1.0, 0.5),
                Vec3(0.0, 1.5, 0.5),
                Vec3(0.5, 1.5, 0.0),  # escapes the x=0 plane
            ),
            evidence=(_scan_evidence(),),
        )


# ----- Ceiling ------------------------------------------------------------


def _flat_ceiling(y: float = 2.5) -> Ceiling:
    return Ceiling(
        id="c1",
        kind=CeilingKind.FLAT,
        polygon=_square(y),
        plane=Plane(a=0.0, b=1.0, c=0.0, d=-y),
        evidence=(_scan_evidence(),),
    )


def test_ceiling_flat_constructs():
    _flat_ceiling()


def test_ceiling_flat_rejects_non_horizontal_plane():
    with pytest.raises(InvariantViolation, match="flat ceiling not horizontal"):
        Ceiling(
            id="c1",
            kind=CeilingKind.FLAT,
            polygon=_square(2.5),
            plane=Plane(a=0.5, b=0.5, c=0.0, d=-1.25),
            evidence=(_scan_evidence(),),
        )


def test_ceiling_oblique_rejects_horizontal_plane():
    with pytest.raises(InvariantViolation, match="not oblique"):
        Ceiling(
            id="c1",
            kind=CeilingKind.OBLIQUE,
            polygon=_square(2.5),
            plane=Plane(a=0.0, b=1.0, c=0.0, d=-2.5),
            evidence=(_scan_evidence(),),
        )


def test_ceiling_rejects_inverted_normal():
    # normal (0, -1, 0): faces down
    with pytest.raises(InvariantViolation, match="normal points down"):
        Ceiling(
            id="c1",
            kind=CeilingKind.FLAT,
            polygon=_square(2.5),
            plane=Plane(a=0.0, b=-1.0, c=0.0, d=2.5),
            evidence=(_scan_evidence(),),
        )


def test_ceiling_composite_requires_parts_and_seams():
    with pytest.raises(InvariantViolation, match="composite needs ≥2 parts"):
        Ceiling(
            id="c1",
            kind=CeilingKind.COMPOSITE,
            polygon=_square(2.5),
            evidence=(_scan_evidence(),),
        )


def test_ceiling_single_plane_rejects_parts():
    inner = _flat_ceiling()
    with pytest.raises(InvariantViolation, match="carries parts/seams"):
        Ceiling(
            id="c2",
            kind=CeilingKind.FLAT,
            polygon=_square(2.5),
            plane=Plane(a=0.0, b=1.0, c=0.0, d=-2.5),
            parts=(inner, inner),
            evidence=(_scan_evidence(),),
        )


# ----- CeilingSeam --------------------------------------------------------


def test_ceiling_seam_constructs():
    CeilingSeam(
        id="s1",
        endpoint_a=Vec3(0.0, 2.0, 0.5),
        endpoint_b=Vec3(1.0, 2.0, 0.5),
        member_part_ids=("ceiling::flat", "ceiling::oblique"),
    )


def test_ceiling_seam_with_optional_knee_wall():
    CeilingSeam(
        id="s1",
        endpoint_a=Vec3(0.0, 2.0, 0.5),
        endpoint_b=Vec3(1.0, 2.0, 0.5),
        member_part_ids=("ceiling::flat", "ceiling::oblique"),
        knee_wall_id="kw1",
    )


def test_ceiling_seam_rejects_wrong_member_count():
    with pytest.raises(InvariantViolation, match="exactly 2 member parts"):
        CeilingSeam(
            id="s1",
            endpoint_a=Vec3(0.0, 2.0, 0.5),
            endpoint_b=Vec3(1.0, 2.0, 0.5),
            member_part_ids=("only_one",),
        )


def test_ceiling_seam_rejects_degenerate():
    with pytest.raises(InvariantViolation, match="degenerate length"):
        CeilingSeam(
            id="s1",
            endpoint_a=Vec3(0.0, 2.0, 0.5),
            endpoint_b=Vec3(0.0, 2.0, 0.5),
            member_part_ids=("flat", "oblique"),
        )


# ----- KneeWall -----------------------------------------------------------


def test_knee_wall_constructs_with_horizontal_top():
    KneeWall(
        id="kw1",
        bottom_a=Vec3(0.0, 1.5, 0.0),
        bottom_b=Vec3(0.0, 1.5, 1.0),
        top_a=Vec3(0.0, 2.2, 0.0),
        top_b=Vec3(0.0, 2.2, 1.0),
        plane=_vertical_plane_x_eq_zero(),
        evidence=(_scan_evidence(),),
    )


def test_knee_wall_constructs_with_tilted_top():
    # 71% of corpus knee walls have tilted tops; the invariant must accept them.
    KneeWall(
        id="kw1",
        bottom_a=Vec3(0.0, 1.5, 0.0),
        bottom_b=Vec3(0.0, 1.5, 1.0),
        top_a=Vec3(0.0, 2.2, 0.0),
        top_b=Vec3(0.0, 2.6, 1.0),
        plane=_vertical_plane_x_eq_zero(),
        evidence=(_scan_evidence(),),
    )


def test_knee_wall_rejects_non_horizontal_bottom():
    with pytest.raises(InvariantViolation, match="bottom edge not horizontal"):
        KneeWall(
            id="kw1",
            bottom_a=Vec3(0.0, 1.4, 0.0),
            bottom_b=Vec3(0.0, 1.5, 1.0),
            top_a=Vec3(0.0, 2.6, 0.0),
            top_b=Vec3(0.0, 2.6, 1.0),
            plane=_vertical_plane_x_eq_zero(),
            evidence=(_scan_evidence(),),
        )


def test_knee_wall_rejects_top_not_above_bottom():
    with pytest.raises(InvariantViolation, match="top not above bottom"):
        KneeWall(
            id="kw1",
            bottom_a=Vec3(0.0, 1.5, 0.0),
            bottom_b=Vec3(0.0, 1.5, 1.0),
            top_a=Vec3(0.0, 1.4, 0.0),
            top_b=Vec3(0.0, 2.0, 1.0),  # one corner below bottom
            plane=_vertical_plane_x_eq_zero(),
            evidence=(_scan_evidence(),),
        )


# ----- Room ---------------------------------------------------------------


def _three_walls() -> tuple[Wall, ...]:
    # Three walls each on their own plane x = i.
    return tuple(
        Wall(
            id=f"w{i}",
            polygon=(
                Vec3(float(i), 0.0, 0.0),
                Vec3(float(i), 0.0, 1.0),
                Vec3(float(i), 2.5, 1.0),
                Vec3(float(i), 2.5, 0.0),
            ),
            plane=Plane(a=1.0, b=0.0, c=0.0, d=-float(i)),
            evidence=(_scan_evidence(),),
        )
        for i in range(3)
    )


def test_room_rejects_too_few_walls():
    floor = Floor(id="f1", polygon=_square(), evidence=(_scan_evidence(),))
    ceiling = _flat_ceiling()
    walls = _three_walls()[:2]
    with pytest.raises(InvariantViolation, match="≥3 walls"):
        Room(
            id="r1",
            story_index=0,
            floor=floor,
            walls=walls,
            ceiling=ceiling,
        )


# ----- Story / Wing / Building -------------------------------------------


def _minimal_room(idx: int = 0) -> Room:
    floor = Floor(id="f1", polygon=_square(), evidence=(_scan_evidence(),))
    ceiling = _flat_ceiling()
    walls = _three_walls()
    return Room(
        id=f"r{idx}",
        story_index=idx,
        floor=floor,
        walls=walls,
        ceiling=ceiling,
    )


def test_story_rejects_empty():
    with pytest.raises(InvariantViolation, match="empty"):
        Story(id="s0", rooms=())


def test_wing_rejects_non_horizontal_footprint():
    skewed = (
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.5, 0.0),
        Vec3(1.0, 0.0, 1.0),
        Vec3(0.0, 0.0, 1.0),
    )
    with pytest.raises(InvariantViolation, match="footprint not horizontal"):
        Wing(
            id="wg0",
            stories=(Story(id="s0", rooms=(_minimal_room(),)),),
            footprint=skewed,
        )


def test_building_rejects_no_uuid():
    wing = Wing(
        id="wg0",
        stories=(Story(id="s0", rooms=(_minimal_room(),)),),
        footprint=_square(),
    )
    with pytest.raises(InvariantViolation, match="missing uuid"):
        Building(id="b0", uuid="", wings=(wing,))


# ----- Roof primitives ----------------------------------------------------


def _oblique_roof_surface() -> RoofSurface:
    # plane y = 2 + 0.5 x, in form a*x + b*y + c*z + d = 0:
    # 0.5 x - y + 2 = 0  -> a=0.5, b=-1, c=0, d=2
    # That has normal (0.5, -1, 0) — y component is negative (faces down).
    # Need normal facing up: -0.5 x + y - 2 = 0 -> a=-0.5, b=1, c=0, d=-2.
    # Normalise: magnitude = sqrt(0.25+1) ≈ 1.118
    import math

    mag = math.sqrt(0.25 + 1.0)
    a, b, c, d = -0.5 / mag, 1.0 / mag, 0.0, -2.0 / mag
    plane = Plane(a=a, b=b, c=c, d=d)
    # planar quad on this plane: y = 2 + 0.5x
    corners = (
        Vec3(0.0, 2.0, 0.0),
        Vec3(1.0, 2.5, 0.0),
        Vec3(1.0, 2.5, 1.0),
        Vec3(0.0, 2.0, 1.0),
    )
    return RoofSurface(
        id="rs0",
        polygon=corners,
        plane=plane,
        evidence=(_scan_evidence(),),
    )


def test_roof_surface_constructs():
    _oblique_roof_surface()


def test_roof_surface_rejects_inverted_normal():
    # normal facing down: 0.5 x - y + 2 = 0 -> b = -1
    import math

    mag = math.sqrt(0.25 + 1.0)
    plane = Plane(a=0.5 / mag, b=-1.0 / mag, c=0.0, d=2.0 / mag)
    corners = (
        Vec3(0.0, 2.0, 0.0),
        Vec3(1.0, 2.5, 0.0),
        Vec3(1.0, 2.5, 1.0),
        Vec3(0.0, 2.0, 1.0),
    )
    with pytest.raises(InvariantViolation, match="does not face upward"):
        RoofSurface(
            id="rs0",
            polygon=corners,
            plane=plane,
            evidence=(_scan_evidence(),),
        )


def test_roof_rejects_no_surfaces():
    with pytest.raises(InvariantViolation, match="no surfaces"):
        Roof(id="r0", kind=RoofKind.GABLE, surfaces=())


def test_ridge_rejects_too_few_members():
    with pytest.raises(InvariantViolation, match="≥2 members"):
        Ridge(
            id="r0",
            endpoint_a=Vec3(0.0, 3.0, 0.0),
            endpoint_b=Vec3(1.0, 3.0, 0.0),
            member_ids=("rs0",),
        )


def test_eave_rejects_missing_member():
    with pytest.raises(InvariantViolation, match="missing member id"):
        Eave(
            id="e0",
            endpoint_a=Vec3(0.0, 2.0, 0.0),
            endpoint_b=Vec3(1.0, 2.0, 0.0),
            surface_id="rs0",
            wall_id="",
        )


def test_dormer_rejects_orphan():
    with pytest.raises(InvariantViolation, match="must reference a host"):
        Dormer(
            id="d0",
            polygon=_square(2.5),
            host_surface_id="",
            evidence=(_scan_evidence(),),
        )


def test_stair_rejects_inverted_stories():
    with pytest.raises(InvariantViolation, match="upper story must be above"):
        Stair(
            id="st0",
            story_lower=2,
            story_upper=1,
            polygon=_square(),
            evidence=(_scan_evidence(),),
        )


# ----- Gap ---------------------------------------------------------------


def test_gap_rejects_no_incident_primitives():
    with pytest.raises(InvariantViolation, match="no incident primitives"):
        Gap(
            id="g0",
            kind=GapKind.CEILING,
            polygon=_square(2.5),
            incident_primitive_ids=(),
        )


# ----- Twin / Residual ---------------------------------------------------


def test_twin_carries_building_and_residual():
    wing = Wing(
        id="wg0",
        stories=(Story(id="s0", rooms=(_minimal_room(),)),),
        footprint=_square(),
    )
    building = Building(id="b0", uuid="abc-123", wings=(wing,))
    twin = Twin(building=building)
    assert twin.building is building
    assert isinstance(twin.residual, Residual)


def test_float_eps_is_language_semantics():
    # Documents the *only* tolerance in the module: it must be float-precision,
    # not building-parameter scale.
    assert FLOAT_EPS == 1e-6
    assert FLOAT_EPS < 1e-3  # well below 1mm
