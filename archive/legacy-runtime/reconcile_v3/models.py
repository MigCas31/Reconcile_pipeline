"""V3 domain types.

Polygons are lists of (x, y, z) tuples; XZ-plane footprints keep a nominal Y
(story floor Y or ceiling Y) so the viewer can render without further lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .audit import HypothesisTrace

Vec3 = tuple[float, float, float]
Polygon = list[Vec3]


@dataclass
class V3Part:
    id: str
    room_ids: tuple[str, ...]
    footprint_xz: Polygon
    stories: tuple[int, ...]
    trace: HypothesisTrace
    gable_extension: GableExtension | None = None


@dataclass(frozen=True)
class GableExtension:
    """Per-part classification of whether existing slanted roofs can safely be
    extrapolated across the full part footprint. See plan
    `classify_gable_extension` stage for the two-tier rule."""

    status: Literal[
        "not_gable",
        "gable_complete",
        "gable_along_extend",
        "gable_cross_review",
        "gable_ambiguous",
    ]
    metrics: dict[str, Any]
    tier1_reasons: tuple[str, ...]
    tier2_reasons: tuple[str, ...]
    ridge_line: tuple[Vec3, Vec3] | None
    uncovered_region_xz: Polygon | None
    decision_reason: str


@dataclass(frozen=True)
class V3Gap:
    id: str
    footprint_xz: Polygon
    floor_y: float
    ceiling_y: float
    between_rooms: tuple[str | None, str | None] | None
    status: Literal["closed", "ambiguous"]
    trace: HypothesisTrace


@dataclass(frozen=True)
class V3Slab:
    id: str
    room_id: str | None
    polygon: Polygon
    trace: HypothesisTrace


@dataclass(frozen=True)
class V3WallExtension:
    id: str
    wall_id: str
    room_id: str
    strip_corners: Polygon
    target_slab_id: str | None
    behind_knee_wall: bool
    trace: HypothesisTrace


@dataclass(frozen=True)
class V3FlatCeiling:
    id: str
    room_id: str | None
    footprint_xz: Polygon
    y: float
    over: Literal["room", "extension", "fill", "gap"]
    trace: HypothesisTrace


@dataclass(frozen=True)
class V3SlantedRoof:
    id: str
    plane: tuple[float, float, float, float]  # ax + by + cz + d = 0
    corners: Polygon
    source_segment_ids: tuple[str, ...]
    trace: HypothesisTrace


@dataclass(frozen=True)
class V3RoofProposal:
    """Permissive (slanted-segment x slab) candidate for human labeling.

    One emission per (segment, slab) pair -- every slanted wall edge is paired
    with every room slab in the building. ``features`` is a flat dict
    snapshotted into the label store, so adding fields later does not break
    historical labels.
    """

    id: str
    segment_index: int
    source_room_id: str | None
    source_wall_id: str | None
    slab_room_id: str | None
    slab_id: str | None
    plane: tuple[float, float, float, float]
    corners: Polygon
    features: dict[str, Any]
    heuristic_label: Literal["accepted", "rejected", "not_evaluated"]
    trace: HypothesisTrace


@dataclass(frozen=True)
class V3MergedRoofSegment:
    """One labelable piece of a merged slanted-roof plane.

    A "merged plane" is the cluster of near-coplanar ``V3RoofProposal``s
    whose planes have the same normal direction and offset within tolerance.
    A "segment" is one connected piece of that plane's footprint after
    clipping to the building boundary and splitting along:

      - seams where this merged plane intersects another merged plane in 3D,
      - XZ boundaries of rain-exposed room/part/gap pieces sitting below it.

    Each segment is the unit the labeler accepts or rejects. The record
    carries the full provenance needed to reverse-engineer "why was this
    segment a roof / why was it not" without re-running the proposer:
    contributing raw proposals, merge thresholds, opposing merged planes
    and their seam lines, the room/gap boundaries that split it, and the
    building boundary polygon used for the outer clip.
    """

    id: str
    cluster_canonical_id: str
    merged_plane: tuple[float, float, float, float]
    corners: Polygon  # final post-clip+split 3D ring, lifted to merged_plane
    footprint_xz: Polygon  # same ring projected to y=0 for hit-testing
    member_proposal_ids: tuple[str, ...]
    member_snapshots: tuple[dict[str, Any], ...]
    cluster_params: dict[str, float]
    building_boundary_xz: Polygon | None
    opposing_cluster_canonicals: tuple[str, ...]
    opposing_planes: tuple[tuple[float, float, float, float], ...]
    room_boundary_refs: tuple[dict[str, Any], ...]
    features: dict[str, Any]
    heuristic_label: Literal["accepted", "rejected", "not_evaluated"]
    trace: HypothesisTrace


@dataclass(frozen=True)
class V3Dormer:
    id: str
    front_wall_id: str
    roof_surface_id: str
    corners: Polygon
    trace: HypothesisTrace


@dataclass(frozen=True)
class V3UnresolvedRegion:
    id: str
    room_id: str | None
    footprint_xz: Polygon
    y_range: tuple[float, float] | None
    reason: str
    context: dict[str, Any]
    trace: HypothesisTrace


@dataclass
class V3Building:
    uuid: str
    address: str | None
    parts: list[V3Part] = field(default_factory=list)
    gaps: list[V3Gap] = field(default_factory=list)
    slabs: list[V3Slab] = field(default_factory=list)
    wall_extensions: list[V3WallExtension] = field(default_factory=list)
    flat_ceilings: list[V3FlatCeiling] = field(default_factory=list)
    slanted_roofs: list[V3SlantedRoof] = field(default_factory=list)
    roof_proposals: list[V3RoofProposal] = field(default_factory=list)
    merged_roof_segments: list[V3MergedRoofSegment] = field(default_factory=list)
    dormers: list[V3Dormer] = field(default_factory=list)
    unresolved: list[V3UnresolvedRegion] = field(default_factory=list)
    audit_log: list[dict[str, Any]] = field(default_factory=list)
