"""Twin primitives and the structural invariants that define them.

Every primitive is a frozen dataclass whose `__post_init__` enforces
type-system invariants. Construction fails (raises `InvariantViolation`)
when the proposed geometry violates a structural rule. Filtering happens
by failing construction, never by post-hoc thresholding.

The only tolerance is `FLOAT_EPS` (`_geometry`). Scan-precision and
building-parameter tolerances do not appear in this module.

Higher-level primitives (`Story`, `Wing`, `Building`, `Roof*`, `Gap`) are
defined here as types but their invariants are deliberately lighter in
Phase A; the per-room invariants (`Floor`, `Wall`, `Opening`, `Ceiling`,
`KneeWall`, `CeilingSeam`, `Room`) are the ones the assembly algorithm
in Phase B will rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from reconcile_tiers.payload.schema import Plane, Vec3
from reconcile_tiers.twin._geometry import (
    FLOAT_EPS,
    plane_is_horizontal,
    plane_is_oblique,
    plane_is_vertical,
    plane_normal_y,
    polygon_area_xz,
    polygon_is_planar,
    segment_endpoints_horizontal,
    segment_length,
    y_span,
)

__all__ = [
    "FLOAT_EPS",
    "Building",
    "Ceiling",
    "CeilingKind",
    "CeilingSeam",
    "Dormer",
    "Eave",
    "Evidence",
    "Floor",
    "Gable",
    "Gap",
    "GapKind",
    "InvariantViolation",
    "KneeWall",
    "Opening",
    "OpeningKind",
    "Primitive",
    "Provenance",
    "ProvenanceKind",
    "Residual",
    "Ridge",
    "Roof",
    "RoofKind",
    "RoofSurface",
    "Room",
    "Stair",
    "Story",
    "Twin",
    "Wall",
    "Wing",
]


class InvariantViolation(ValueError):
    """A primitive's structural invariants do not hold for the proposed
    geometry. Raised from `__post_init__`. The assembler is responsible
    for not attempting to construct violating primitives — these
    exceptions surface assembler bugs, not user-facing errors."""


# ----- Provenance & Evidence ----------------------------------------------

ProvenanceKind = Literal["scan", "computed"]


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where one piece of evidence came from."""

    kind: ProvenanceKind
    source: str  # e.g. "roomplan_wall", "raw_ceiling_plane",
    #             "classification.gable_pair", "ridge_intersection"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Geometric witness for one or more primitives.

    `parents` is non-empty for `ComputedEvidence`: it lists the primitive
    ids that synthesised this evidence. Confidence propagation depends on
    these (see Phase A confidence_semantics.md).
    """

    provenance: Provenance
    geometry: tuple[Vec3, ...]
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.provenance.kind == "computed" and not self.parents:
            raise InvariantViolation(
                f"ComputedEvidence from '{self.provenance.source}' has no parents"
            )
        if self.provenance.kind == "scan" and self.parents:
            raise InvariantViolation(
                f"ScanEvidence from '{self.provenance.source}' must not have parents"
            )


# ----- Primitive interface ------------------------------------------------


@runtime_checkable
class Primitive(Protocol):
    """Structural marker. Every concrete primitive carries an id and
    evidence. Container primitives (Room, Story, Wing, Building) carry
    an empty evidence tuple — their existence is implied by their parts."""

    id: str
    evidence: tuple[Evidence, ...]


# ----- Per-room primitives ------------------------------------------------


@dataclass(frozen=True, slots=True)
class Floor:
    """A horizontal polygon at the bottom of a Room. May have interior
    holes (for stair shafts). Holes do not change the structural
    horizontality."""

    id: str
    polygon: tuple[Vec3, ...]
    holes: tuple[tuple[Vec3, ...], ...] = ()
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise InvariantViolation(f"Floor {self.id}: <3 corners")
        if y_span(self.polygon) > FLOAT_EPS:
            raise InvariantViolation(
                f"Floor {self.id}: not horizontal (y-span={y_span(self.polygon):.6f})"
            )
        if polygon_area_xz(self.polygon) <= FLOAT_EPS:
            raise InvariantViolation(f"Floor {self.id}: degenerate area")
        for i, hole in enumerate(self.holes):
            if len(hole) < 3:
                raise InvariantViolation(f"Floor {self.id}: hole {i} <3 corners")
            if y_span(hole) > FLOAT_EPS:
                raise InvariantViolation(f"Floor {self.id}: hole {i} not horizontal")
        if not self.evidence:
            raise InvariantViolation(f"Floor {self.id}: has no evidence")


class OpeningKind(StrEnum):
    DOOR = "door"
    WINDOW = "window"
    PASSAGE = "passage"


@dataclass(frozen=True, slots=True)
class Opening:
    """A bounded region within a Wall's plane. Doors/windows/passages."""

    id: str
    kind: OpeningKind
    polygon: tuple[Vec3, ...]
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise InvariantViolation(f"Opening {self.id}: <3 corners")
        if not polygon_is_planar(self.polygon):
            raise InvariantViolation(f"Opening {self.id}: corners not coplanar")
        if not self.evidence:
            raise InvariantViolation(f"Opening {self.id}: has no evidence")


@dataclass(frozen=True, slots=True)
class Wall:
    """A non-horizontal planar polygon. Walls in the corpus aren't
    always 4-corner quads — gable end walls have 5 corners (a peak),
    walls with stepped tops have 6 or more. Force-fitting them to
    rectangles loses architectural features. The Wall holds the
    actual polygon (≥3 corners, coplanar with the declared plane).
    Carries any Openings that pierce its plane.
    """

    id: str
    polygon: tuple[Vec3, ...]
    plane: Plane
    openings: tuple[Opening, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise InvariantViolation(f"Wall {self.id}: <3 corners")
        if plane_is_horizontal(self.plane):
            raise InvariantViolation(f"Wall {self.id}: plane is horizontal, not a wall")
        for c in self.polygon:
            residual = abs(
                self.plane.a * c.x
                + self.plane.b * c.y
                + self.plane.c * c.z
                + self.plane.d
            )
            if residual > FLOAT_EPS:
                raise InvariantViolation(
                    f"Wall {self.id}: corner {c} residual {residual:.3e} "
                    f"exceeds float precision against declared plane"
                )
        ys = [c.y for c in self.polygon]
        if max(ys) - min(ys) < FLOAT_EPS:
            raise InvariantViolation(f"Wall {self.id}: zero height (top equals bottom)")
        for op in self.openings:
            if not all(_point_on_plane(self.plane, c) for c in op.polygon):
                raise InvariantViolation(
                    f"Wall {self.id}: opening {op.id} escapes wall plane"
                )
        if not self.evidence:
            raise InvariantViolation(f"Wall {self.id}: has no evidence")

    @property
    def bottom_y(self) -> float:
        return min(c.y for c in self.polygon)

    @property
    def top_y(self) -> float:
        return max(c.y for c in self.polygon)


def _point_on_plane(plane: Plane, p: Vec3) -> bool:
    return abs(plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d) < FLOAT_EPS


class CeilingKind(StrEnum):
    FLAT = "flat"
    OBLIQUE = "oblique"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class Ceiling:
    """A planar (flat or oblique) or composite (kinked) top to a Room.

    For composite ceilings, `parts` lists the sub-Ceilings and `seams`
    lists the CeilingSeam primitives that anchor the joins. Single-plane
    ceilings have empty `parts` and `seams`.
    """

    id: str
    kind: CeilingKind
    polygon: tuple[Vec3, ...]
    plane: Plane | None = None  # None for composite (each part has its own)
    parts: tuple[Ceiling, ...] = ()
    seams: tuple[CeilingSeam, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise InvariantViolation(f"Ceiling {self.id}: <3 corners")
        if self.kind in (CeilingKind.FLAT, CeilingKind.OBLIQUE):
            if self.plane is None:
                raise InvariantViolation(
                    f"Ceiling {self.id}: {self.kind.value} requires a plane"
                )
            if self.kind is CeilingKind.FLAT and not plane_is_horizontal(self.plane):
                raise InvariantViolation(
                    f"Ceiling {self.id}: flat ceiling not horizontal"
                )
            if self.kind is CeilingKind.OBLIQUE and not plane_is_oblique(self.plane):
                raise InvariantViolation(
                    f"Ceiling {self.id}: oblique ceiling not oblique"
                )
            ny = plane_normal_y(self.plane)
            if ny <= 0.0:
                raise InvariantViolation(
                    f"Ceiling {self.id}: normal points down (n.y={ny:.6f})"
                )
            if self.parts or self.seams:
                raise InvariantViolation(
                    f"Ceiling {self.id}: single-plane ceiling carries parts/seams"
                )
        else:  # COMPOSITE
            if self.plane is not None:
                raise InvariantViolation(
                    f"Ceiling {self.id}: composite must not carry a single plane"
                )
            if len(self.parts) < 2:
                raise InvariantViolation(f"Ceiling {self.id}: composite needs ≥2 parts")
            if not self.seams:
                raise InvariantViolation(f"Ceiling {self.id}: composite needs ≥1 seam")
        if not self.evidence and not self.parts:
            raise InvariantViolation(f"Ceiling {self.id}: has no evidence and no parts")
        if self.evidence and any(
            ev.provenance.kind == "computed" and ev.parents == ()
            for ev in self.evidence
        ):
            # Defensive: a computed evidence entry without parents would
            # have failed Evidence.__post_init__; this is a redundant guard.
            raise InvariantViolation(
                f"Ceiling {self.id}: computed evidence missing parents"
            )


@dataclass(frozen=True, slots=True)
class CeilingSeam:
    """The line where two sub-Ceilings of a composite Ceiling meet.

    The seam is anchored by the intersection of the two adjacent ceiling
    planes (a horizontal line, when one part is flat). It optionally
    references a `KneeWall` when extraction finds a vertical step at this
    line; many composite ceilings have no knee wall (the slope smoothly
    meets the flat lid).
    """

    id: str
    endpoint_a: Vec3
    endpoint_b: Vec3
    member_part_ids: tuple[str, ...]  # the two Ceiling part ids meeting here
    knee_wall_id: str | None = None
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if segment_length(self.endpoint_a, self.endpoint_b) < FLOAT_EPS:
            raise InvariantViolation(f"CeilingSeam {self.id}: degenerate length")
        if len(self.member_part_ids) != 2:
            raise InvariantViolation(
                f"CeilingSeam {self.id}: needs exactly 2 member parts"
            )


@dataclass(frozen=True, slots=True)
class KneeWall:
    """A short vertical wall that supports the kink between a flat lid
    and an oblique sub-ceiling within a single Room. Distinct from
    `Wall` because its bottom edge is not on the Room's Floor — it sits
    on the Floor of a higher region or on a ceiling sub-piece below."""

    id: str
    bottom_a: Vec3
    bottom_b: Vec3
    top_a: Vec3
    top_b: Vec3
    plane: Plane
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        # Survey of 154 corpus knee walls: vertical plane is universal,
        # horizontal bottom is universal in 97% (the 4 outliers turn out
        # to be sloped panels mislabelled as knee walls — correctly
        # rejected here so they become orphans). The top edge tilts in
        # 71% of the corpus to follow the slope above; not invariant.
        if not plane_is_vertical(self.plane):
            raise InvariantViolation(f"KneeWall {self.id}: not vertical")
        if not segment_endpoints_horizontal(self.bottom_a, self.bottom_b):
            raise InvariantViolation(f"KneeWall {self.id}: bottom edge not horizontal")
        bottom_y = self.bottom_a.y
        if min(self.top_a.y, self.top_b.y) - bottom_y < FLOAT_EPS:
            raise InvariantViolation(f"KneeWall {self.id}: top not above bottom")
        if not self.evidence:
            raise InvariantViolation(f"KneeWall {self.id}: has no evidence")


@dataclass(frozen=True, slots=True)
class Room:
    """A bounded volume with one Floor, walls forming a closed plan-view
    loop, and one Ceiling. Container primitive: holds no evidence of its
    own, but its existence is implied by its parts.

    Phase A invariants are minimal; Phase B's assembler enforces stronger
    structural ties (each Floor edge incident to a Wall bottom, Ceiling
    covers Floor) once it can compute incidence.
    """

    id: str
    story_index: int
    floor: Floor
    walls: tuple[Wall, ...]
    ceiling: Ceiling
    knee_walls: tuple[KneeWall, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.walls) < 3:
            raise InvariantViolation(
                f"Room {self.id}: needs ≥3 walls to enclose a polygon"
            )


# ----- Story / Wing / Building -------------------------------------------


@dataclass(frozen=True, slots=True)
class Story:
    """A maximal connected set of Rooms. Co-storied via wall adjacency,
    not by metric Y. Phase A keeps this as a container; Phase C will
    enforce the adjacency invariant."""

    id: str
    rooms: tuple[Room, ...]
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.rooms:
            raise InvariantViolation(f"Story {self.id}: empty")


@dataclass(frozen=True, slots=True)
class Wing:
    """A connected component of the building plan. Each Wing has its own
    Roof. Phase A keeps this light; Phase C will enforce the
    plan-view connectivity invariant."""

    id: str
    stories: tuple[Story, ...]
    footprint: tuple[Vec3, ...]
    roof: Roof | None = None
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.stories:
            raise InvariantViolation(f"Wing {self.id}: empty")
        if len(self.footprint) < 3:
            raise InvariantViolation(f"Wing {self.id}: degenerate footprint")
        if y_span(self.footprint) > FLOAT_EPS:
            raise InvariantViolation(f"Wing {self.id}: footprint not horizontal")


@dataclass(frozen=True, slots=True)
class Building:
    """The root primitive. Every other primitive is reachable from here
    via the Wing/Story/Room/* tree."""

    id: str
    uuid: str
    wings: tuple[Wing, ...]
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.wings:
            raise InvariantViolation(f"Building {self.id}: no wings")
        if not self.uuid:
            raise InvariantViolation(f"Building {self.id}: missing uuid")


# ----- Roof primitives ----------------------------------------------------


class RoofKind(StrEnum):
    FLAT = "flat"
    GABLE = "gable"
    HIP = "hip"
    MANSARD = "mansard"
    SHED = "shed"
    COMPLEX = "complex"


@dataclass(frozen=True, slots=True)
class Roof:
    """The roof of one Wing. Composed of RoofSurfaces tied together by
    Ridges/Eaves/Gables. Phase A: container; Phase C: structural ties."""

    id: str
    kind: RoofKind
    surfaces: tuple[RoofSurface, ...]
    ridges: tuple[Ridge, ...] = ()
    eaves: tuple[Eave, ...] = ()
    gables: tuple[Gable, ...] = ()
    dormers: tuple[Dormer, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.surfaces:
            raise InvariantViolation(f"Roof {self.id}: no surfaces")


@dataclass(frozen=True, slots=True)
class RoofSurface:
    id: str
    polygon: tuple[Vec3, ...]
    plane: Plane
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise InvariantViolation(f"RoofSurface {self.id}: <3 corners")
        # Check coplanarity against the surface's own plane equation,
        # not against a plane re-derived from the first three corners
        # (which loses precision when those three are nearly collinear).
        for c in self.polygon:
            residual = abs(
                self.plane.a * c.x
                + self.plane.b * c.y
                + self.plane.c * c.z
                + self.plane.d
            )
            if residual > FLOAT_EPS:
                raise InvariantViolation(
                    f"RoofSurface {self.id}: corner residual {residual:.3e} "
                    f"exceeds float precision against declared plane"
                )
        if plane_is_vertical(self.plane):
            raise InvariantViolation(f"RoofSurface {self.id}: vertical plane")
        if plane_normal_y(self.plane) <= 0.0:
            raise InvariantViolation(
                f"RoofSurface {self.id}: normal does not face upward"
            )
        if not self.evidence:
            raise InvariantViolation(f"RoofSurface {self.id}: no evidence")


@dataclass(frozen=True, slots=True)
class Ridge:
    """A line shared by ≥2 roof primitives whose planes intersect at it."""

    id: str
    endpoint_a: Vec3
    endpoint_b: Vec3
    member_ids: tuple[str, ...]  # ids of RoofSurfaces / Gables sharing this line
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if segment_length(self.endpoint_a, self.endpoint_b) < FLOAT_EPS:
            raise InvariantViolation(f"Ridge {self.id}: degenerate length")
        if len(self.member_ids) < 2:
            raise InvariantViolation(f"Ridge {self.id}: needs ≥2 members")


@dataclass(frozen=True, slots=True)
class Eave:
    """A line shared by exactly one RoofSurface and one Wall.top."""

    id: str
    endpoint_a: Vec3
    endpoint_b: Vec3
    surface_id: str
    wall_id: str
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if segment_length(self.endpoint_a, self.endpoint_b) < FLOAT_EPS:
            raise InvariantViolation(f"Eave {self.id}: degenerate length")
        if not self.surface_id or not self.wall_id:
            raise InvariantViolation(f"Eave {self.id}: missing member id")


@dataclass(frozen=True, slots=True)
class Gable:
    """A vertical Wall whose top edge IS a Ridge endpoint or line."""

    id: str
    wall_id: str
    ridge_id: str
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.wall_id or not self.ridge_id:
            raise InvariantViolation(f"Gable {self.id}: missing member id")


@dataclass(frozen=True, slots=True)
class Dormer:
    """A protrusion from a RoofSurface. Incident to one host RoofSurface."""

    id: str
    polygon: tuple[Vec3, ...]
    host_surface_id: str
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise InvariantViolation(f"Dormer {self.id}: <3 corners")
        if not self.host_surface_id:
            raise InvariantViolation(
                f"Dormer {self.id}: must reference a host RoofSurface"
            )


@dataclass(frozen=True, slots=True)
class Stair:
    """A vertical connection between Stories, sitting in a Floor.holes."""

    id: str
    story_lower: int
    story_upper: int
    polygon: tuple[Vec3, ...]
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if self.story_upper <= self.story_lower:
            raise InvariantViolation(
                f"Stair {self.id}: upper story must be above lower"
            )
        if len(self.polygon) < 3:
            raise InvariantViolation(f"Stair {self.id}: <3 corners")


# ----- Gap (residual) -----------------------------------------------------


class GapKind(StrEnum):
    CEILING = "ceiling"
    FLOOR = "floor"
    SIDE = "side"
    ROOF = "roof"
    CROSS_STORY = "cross_story"


@dataclass(frozen=True, slots=True)
class Gap:
    """An unclaimed region bounded by primitive edges. First-class output:
    not silently filled by synthesis. Carries the ids of the primitives
    bordering it for diagnostic surfacing."""

    id: str
    kind: GapKind
    polygon: tuple[Vec3, ...]
    incident_primitive_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise InvariantViolation(f"Gap {self.id}: <3 corners")
        if not self.incident_primitive_ids:
            raise InvariantViolation(f"Gap {self.id}: no incident primitives")


# ----- Twin / Residual containers ----------------------------------------


@dataclass(frozen=True, slots=True)
class Residual:
    """The non-twin half of the pipeline output. Surfaces what the
    assembler could not claim: scan fragments without a host primitive,
    bounded regions without a Ceiling/Floor/Roof, and primitives that
    exist only via ComputedEvidence."""

    orphan_evidence: tuple[Evidence, ...] = ()
    unclaimed_gaps: tuple[Gap, ...] = ()
    fully_inferred_primitive_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Twin:
    """The full digital-twin output: a Building + a Residual stream.

    Consumers (viewer, energy estimator, audit) read this. The legacy
    `tier_payload.json` is one serialisation of `Twin.building`; the
    Residual is its sibling stream.
    """

    building: Building
    residual: Residual = field(default_factory=Residual)


# ----- Re-exports the package wants visible -----------------------------

__all__ = [name for name in __all__]  # the literal above is canonical
