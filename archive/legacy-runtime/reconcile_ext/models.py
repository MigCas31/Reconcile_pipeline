"""Domain types for the extension-detection pipeline.

All coordinates follow the repo-wide convention: (x, y, z) tuples with y-up.
Footprints are XZ-plane polygons stored as (x, 0, z) rings for viewer
parity, with `floor_y` / `y` fields when elevation matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Vec3 = tuple[float, float, float]
Polygon = list[Vec3]

ReflexKind = Literal["concave-corner", "concave-edge-step"]
DiscontinuityKind = Literal[
    "ridge-delta", "eave-delta", "azimuth-mismatch", "inclination-mismatch"
]
WallStepKind = Literal["parallel-step", "collinear-step"]
DeltaKind = Literal["ceiling-delta", "floor-delta"]


@dataclass(frozen=True)
class ExtReflexVertex:
    id: str
    x: float
    z: float
    interior_angle_deg: float
    """Interior angle measured through the building footprint's interior —
    values > 180° are reflex. Larger = sharper notch."""
    notch_depth_m: float
    """Shortest-path distance from the vertex to the footprint's convex hull
    boundary. Proxy for how deep the notch is."""
    left_edge_len_m: float
    right_edge_len_m: float
    left_edge_azimuth_deg: float
    right_edge_azimuth_deg: float
    part_id: str | None
    """Part whose footprint contributed this vertex (if single-part footprint
    contribution can be determined)."""
    rationale: str


@dataclass(frozen=True)
class ExtDiscontinuity:
    id: str
    kind: DiscontinuityKind
    roof_a_id: str
    roof_b_id: str
    delta: float
    """Signed delta in meters or degrees, depending on `kind`."""
    mid_xz: tuple[float, float]
    """Approximate XZ midpoint of the shared edge / proximity region."""
    strength: Literal["weak", "strong"]
    rationale: str


@dataclass(frozen=True)
class ExtWallStep:
    id: str
    kind: WallStepKind
    wall_a_id: str
    wall_b_id: str
    room_a_id: str
    room_b_id: str
    azimuth_deg: float
    offset_m: float
    """Perpendicular distance between the two wall planes, in meters."""
    overlap_m: float
    """Overlap of the two walls' projections along their shared direction."""
    anchor_xz: tuple[float, float]
    """Approximate midpoint of the overlap, for viewer annotation."""
    rationale: str


@dataclass(frozen=True)
class ExtHeightDelta:
    id: str
    kind: DeltaKind
    room_a_id: str
    room_b_id: str
    delta_m: float
    y_a: float
    y_b: float
    anchor_xz: tuple[float, float]
    rationale: str


@dataclass(frozen=True)
class ExtSeamHypothesis:
    """A speculative part-seam produced by combining multiple tier-1/tier-2
    signals near the same location. Diagnostic — does NOT replace V3Parts."""

    id: str
    anchor_xz: tuple[float, float]
    supporting_signal_ids: tuple[str, ...]
    composite_score: float
    confidence: Literal["weak", "medium", "strong"]
    rationale: str


@dataclass(frozen=True)
class ExtUnit:
    """A sub-part of a V3 building after cutting the merged footprint at
    each confident seam. Unit 0 = ``main`` (largest sub-polygon); remaining
    sub-polygons become ``extension`` units.

    Diagnostic only — does not alter V3 geometry. Room membership is decided
    by floor-centroid containment, so rooms straddling a seam fall to the
    side containing their centroid.
    """

    id: str
    role: Literal["main", "extension"]
    index: int
    """0 for main, 1..N for extensions in descending area order."""
    room_ids: tuple[str, ...]
    footprint_xz: Polygon
    area_m2: float
    seam_ids: tuple[str, ...]
    """Which seam hypotheses bounded this unit."""
    rationale: str


@dataclass
class ExtReport:
    building_uuid: str
    address: str | None = None
    reflex_vertices: list[ExtReflexVertex] = field(default_factory=list)
    discontinuities: list[ExtDiscontinuity] = field(default_factory=list)
    wall_steps: list[ExtWallStep] = field(default_factory=list)
    height_deltas: list[ExtHeightDelta] = field(default_factory=list)
    seam_hypotheses: list[ExtSeamHypothesis] = field(default_factory=list)
    units: list[ExtUnit] = field(default_factory=list)
    seam_cut_lines: list[tuple[Polygon, str]] = field(default_factory=list)
    """Geometric cut segments used to partition the footprint into units.
    Each entry is ``(line_coords, seam_id)``; line_coords is a 2-vertex
    polygon-like list ``[(x, y, z), (x, y, z)]``. Viewer draws these so the
    user can verify where the partition happened."""
    # Rendering helpers — snapshots of the inputs so the viewer doesn't need
    # to re-cross-reference buildings_3d.json.
    building_footprint_xz: Polygon = field(default_factory=list)
    parts_footprints_xz: list[Polygon] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtSnapshot:
    """Everything the extension pipeline needs as input.

    Carries only the shapes we look at — footprints, wall corners, slanted
    roof corners + planes, room floor/ceiling heights. No V3 private types
    leak in here so the pipeline stays decoupled.
    """

    building_uuid: str
    address: str | None
    parts: list[SnapshotPart]
    rooms: list[SnapshotRoom]
    slanted_roofs: list[SnapshotRoof]
    gap_fill_footprints: list[Polygon] = field(default_factory=list)
    """XZ polygons from V3's gap-closure stage (slabs with ``room_id=None``).
    Unioned with part footprints before reflex detection so scan-registration
    gaps that V3 already healed don't register as extension seams."""


@dataclass
class SnapshotPart:
    id: str
    footprint_xz: Polygon
    room_ids: tuple[str, ...]
    stories: tuple[int, ...]


@dataclass
class SnapshotRoom:
    id: str
    story: int
    floor_polygon: Polygon
    floor_y: float
    ceiling_y: float | None
    walls: list[SnapshotWall]


@dataclass
class SnapshotWall:
    id: str
    corners: Polygon
    azimuth_deg: float
    plane_offset_m: float
    """Signed distance from the origin to the wall plane, along its outward
    normal. Used to compare coplanarity between walls."""
    top_y: float
    bottom_y: float
    length_m: float


@dataclass
class SnapshotRoof:
    id: str
    corners: Polygon
    plane: tuple[float, float, float, float]
    azimuth_deg: float
    inclination_deg: float
    ridge_y: float
    """Highest Y value on the roof polygon — proxy for ridge height."""
    eave_y: float
    """Lowest Y value — proxy for eave height."""
