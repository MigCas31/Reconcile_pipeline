from __future__ import annotations

import dataclasses
from dataclasses import MISSING, dataclass, field
from enum import StrEnum
from types import UnionType
from typing import Any, Literal, get_args, get_origin, get_type_hints


class GapKind(StrEnum):
    GAP_FLOOR = "gap_floor"
    GAP_CEILING = "gap_ceiling"
    SIDE = "side"
    STITCH = "stitch"
    STITCH_FLOOR = "stitch_floor"
    STITCH_CEIL = "stitch_ceiling"
    EXTERIOR_SIDE = "exterior_side"
    EXTERIOR_FLOOR = "exterior_floor"
    EXTERIOR_CEIL = "exterior_ceiling"


class GapScope(StrEnum):
    INTRA_STORY = "intra_story"
    INTER_STORY = "inter_story"
    EXTERIOR = "exterior"
    JUNCTION = "junction"


class CeilingSource(StrEnum):
    # v1 sources (legacy ceiling_painter)
    FLAT_EMIT = "flat_emit"
    ATTIC_FLAT_LID = "attic_flat_lid"
    ROOF_ARRANGEMENT = "roof_arrangement"
    ROOF_ARRANGEMENT_ATTIC = "roof_arrangement_attic"
    THERMAL_CAP = "thermal_cap"
    RAW_FALLBACK = "raw_fallback"
    # v2 sources (thermal_envelope)
    FLAT_CEILING = "flat_ceiling"
    COMPUTED_OBLIQUE = "computed_oblique"
    FLAT_GAP_CLOSURE = "flat_gap_closure"
    KNEEWALL = "kneewall"
    RAW_SCAN = "raw_scan"
    # Synthetic source for the coplanar merge pass (Plan A). Two or more
    # COMPUTED_OBLIQUE / RAW_SCAN pieces sharing a plane within tolerance get
    # unioned into one piece tagged MERGED_COPLANAR. Provenance retained via
    # CeilingPiece.merged_from.
    MERGED_COPLANAR = "merged_coplanar"


class ShellTag(StrEnum):
    ABOVE_CEILING_BELOW_RIDGE = "above_ceiling_below_ridge"
    OUTSIDE_KNEEWALL_BELOW_EAVE = "outside_kneewall_below_eave"
    GABLE_END = "gable_end"


class CeilingRole(StrEnum):
    CEILING = "ceiling"
    KNEEWALL = "kneewall"


class RoofType(StrEnum):
    NONE = "none"
    FLAT = "flat"
    SHED = "shed"
    GABLE = "gable"
    HIP = "hip"
    MANSARD = "mansard"
    PYRAMID = "pyramid"
    CROSS_GABLE = "cross_gable"
    COMPLEX = "complex"


class KneeWallKind(StrEnum):
    KNEE = "knee"


class DormerFaceKind(StrEnum):
    DORMER_CHEEK = "dormer_cheek"
    DORMER_HEADER = "dormer_header"
    DORMER_FRONT = "dormer_front"


class GableClosureKind(StrEnum):
    LOWER = "lower"
    UPPER = "upper"


class AdjacencyKind(StrEnum):
    EXTERNAL_AIR = "externalAir"
    GROUND_SLAB = "groundSlab"
    GROUND_SLAB_UFH = "groundSlabUFH"
    UNHEATED_ATTIC = "unheatedAttic"
    UNHEATED_BASEMENT_FLOOR = "unheatedBasementFloor"
    BASEMENT_WALL_GROUND_SHALLOW = "basementWallGroundShallow"
    BASEMENT_WALL_GROUND_DEEP = "basementWallGroundDeep"
    INTERNAL_TO_HEATED = "internalToHeated"
    INTERNAL_TO_UNHEATED_HOST = "internalToUnheatedHost"
    PARTY_WALL = "partyWall"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class Plane:
    a: float
    b: float
    c: float
    d: float


@dataclass(frozen=True, slots=True)
class Quad:
    corners: list[Vec3]


@dataclass(frozen=True, slots=True)
class HorizontalLid:
    corners: list[Vec3]
    adjacency: AdjacencyKind = field(
        default=AdjacencyKind.UNKNOWN, metadata={"schema_required": False}
    )
    holes: list[list[Vec3]] = field(
        default_factory=list, metadata={"schema_required": False}
    )


@dataclass(frozen=True, slots=True)
class Wall:
    corners: list[Vec3]
    descent_strip: list[Vec3] | None
    uplift_strip: list[Vec3] | None
    cutouts: list[Quad]
    locator_id: str
    synthetic: bool = False
    adjacency: AdjacencyKind = field(
        default=AdjacencyKind.UNKNOWN, metadata={"schema_required": False}
    )


@dataclass(frozen=True, slots=True)
class Room:
    story: int
    floor: list[HorizontalLid]
    walls: list[Wall]
    doors: list[Quad]
    windows: list[Quad]
    locator_id: str
    heating: str | None = field(default=None, metadata={"schema_required": False})


@dataclass(frozen=True, slots=True)
class GapPiece:
    corners: list[Vec3]
    kind: GapKind
    scope: GapScope
    locator_id: str
    adjacency: AdjacencyKind = field(
        default=AdjacencyKind.UNKNOWN, metadata={"schema_required": False}
    )


@dataclass(frozen=True, slots=True)
class CeilingPiece:
    corners: list[Vec3]
    holes: list[list[Vec3]]
    plane: Plane
    source: CeilingSource
    arrangement_cell_id: str | None
    locator_id: str
    # v2-only fields. Default values keep v1 payloads byte-compatible.
    support_quality: float = field(default=0.0, metadata={"schema_required": False})
    role: CeilingRole = field(
        default=CeilingRole.CEILING, metadata={"schema_required": False}
    )
    adjacency: AdjacencyKind = field(
        default=AdjacencyKind.UNKNOWN, metadata={"schema_required": False}
    )
    # When source == MERGED_COPLANAR, lists the locator_ids of the source
    # pieces that were unioned. Empty for non-merged pieces.
    merged_from: list[str] = field(
        default_factory=list, metadata={"schema_required": False}
    )


@dataclass(frozen=True, slots=True)
class VisualShellPiece:
    corners: list[Vec3]
    plane: Plane
    tag: ShellTag
    source_oblique_index: int
    locator_id: str


@dataclass(frozen=True, slots=True)
class KneeWall:
    corners: list[Vec3]
    kind: KneeWallKind
    locator_id: str
    adjacency: AdjacencyKind = field(
        default=AdjacencyKind.UNKNOWN, metadata={"schema_required": False}
    )


@dataclass(frozen=True, slots=True)
class DormerFace:
    corners: list[Vec3]
    kind: DormerFaceKind
    locator_id: str
    cutouts: list[Quad] = field(
        default_factory=list, metadata={"schema_required": False}
    )
    adjacency: AdjacencyKind = field(
        default=AdjacencyKind.UNKNOWN, metadata={"schema_required": False}
    )


@dataclass(frozen=True, slots=True)
class GableClosure:
    corners: list[Vec3]
    kind: GableClosureKind
    locator_id: str
    adjacency: AdjacencyKind = field(
        default=AdjacencyKind.UNKNOWN, metadata={"schema_required": False}
    )


@dataclass(frozen=True, slots=True)
class TierClassification:
    tier: int
    tier_label: str
    roof_type: RoofType
    n_stories: int
    n_rooms: int
    n_oblique: int
    n_flat: int
    has_half_height: bool
    has_gable: bool


@dataclass(frozen=True, slots=True)
class RoofQuality:
    # Predicted manual roof rating, fit on existing rater labels. Single-rater
    # personal-preference predictor — not an objective quality metric. Internal
    # tooling only; never expose via public API.
    score: float
    predicted_rating: float
    components: dict[str, float]
    quality_version: str


@dataclass(frozen=True, slots=True)
class BuildMeta:
    # Epoch seconds at the moment the payload was written. Compared against
    # input mtimes (scan-cache, merged.json, reconcile_tiers/ source) to
    # detect stale outputs without re-running the pipeline.
    built_at: float


class StairType(StrEnum):
    STRAIGHT = "straight"
    CURVED = "curved"
    SPIRAL = "spiral"


class StairSegmentType(StrEnum):
    STAIR = "stair"
    LANDING = "landing"


class StairAttachmentSide(StrEnum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class StairSegment:
    segment_type: StairSegmentType
    width: float
    length: float
    height: float
    step_count: int
    attachment_side: StairAttachmentSide


@dataclass(frozen=True, slots=True)
class Stair:
    # Pascal-editor-compatible StairNode + StairSegmentNode payload, reconstructed
    # from RoomPlan `stairs` objects + room geometry + BR18 priors. Position is the
    # base anchor in building-local frame (XZ at the foot of the lower flight, Y at
    # the base story floor); rotation is the run heading around +Y in radians.
    locator_id: str
    position: Vec3
    rotation: float
    stair_type: StairType
    width: float
    total_rise: float
    step_count: int
    from_story: int
    to_story: int
    segments: list[StairSegment]
    riser_height: float
    # Source RoomPlan object identifiers (UUIDs) merged into this stair. Lets
    # downstream attribute back to scan fragments for debugging.
    source_fragment_ids: list[str]
    # captured_rise / nominal_rise. 1.0 = full flight scanned; <1.0 = partial.
    capture_coverage: float
    # Literal slab opening polygon (XZ in building frame, Y = top story floor)
    # taken from the raw-ceiling/upper-floor overlap when present. Empty list
    # means renderer falls back to its default rectangular cut.
    slab_opening_polygon: list[Vec3] = field(
        default_factory=list, metadata={"schema_required": False}
    )


@dataclass(frozen=True, slots=True)
class TierPayload:
    schema_version: Literal["1"]
    uuid: str
    address: str | None
    building_center: Vec3
    classification: TierClassification
    rooms: list[Room]
    gaps: list[GapPiece]
    ceiling: list[CeilingPiece]
    knee_walls: list[KneeWall]
    dormer_faces: list[DormerFace]
    gable_closures: list[GableClosure]
    story_labels: list[str] = field(
        default_factory=list, metadata={"schema_required": False}
    )
    visual_shells: list[VisualShellPiece] = field(
        default_factory=list, metadata={"schema_required": False}
    )
    build_meta: BuildMeta | None = field(
        default=None, metadata={"schema_required": False}
    )
    roof_quality: RoofQuality | None = field(
        default=None, metadata={"schema_required": False}
    )
    stairs: list[Stair] = field(
        default_factory=list, metadata={"schema_required": False}
    )


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def payload_to_dict(payload: TierPayload) -> dict[str, Any]:
    return _to_jsonable(payload)


def _is_optional(field_type: Any) -> bool:
    return get_origin(field_type) in (UnionType, __import__("typing").Union) and type(
        None
    ) in get_args(field_type)


def _from_jsonable(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin is Literal:
        if value not in args:
            raise ValueError(f"expected one of {args}, got {value!r}")
        return value
    if _is_optional(field_type):
        if value is None:
            return None
        non_none = next(arg for arg in args if arg is not type(None))
        return _from_jsonable(non_none, value)
    if origin is list:
        (item_type,) = args
        return [_from_jsonable(item_type, item) for item in value]
    if origin is dict:
        key_type, val_type = args
        return {
            _from_jsonable(key_type, k): _from_jsonable(val_type, v)
            for k, v in value.items()
        }
    if isinstance(field_type, type) and issubclass(field_type, StrEnum):
        return field_type(value)
    if dataclasses.is_dataclass(field_type):
        hints = get_type_hints(field_type)
        values = {}
        for field_ in dataclasses.fields(field_type):
            if field_.name in value:
                values[field_.name] = _from_jsonable(
                    hints[field_.name], value[field_.name]
                )
            elif field_.default is not MISSING:
                values[field_.name] = field_.default
            elif field_.default_factory is not MISSING:
                values[field_.name] = field_.default_factory()
            else:
                raise KeyError(field_.name)
        return field_type(**values)
    return value


def payload_from_dict(data: dict[str, Any]) -> TierPayload:
    return _from_jsonable(TierPayload, data)
