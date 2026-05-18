from __future__ import annotations

from dataclasses import dataclass, field

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.config import architectural_priors_enabled
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import DormerFaceKind, KneeWallKind


@dataclass(frozen=True, slots=True)
class RoofSegment:
    a: list[float]
    b: list[float]
    incl: float
    azimuth: float
    length: float
    story: int
    room_index: int
    wall_id: str | None = None

    @property
    def midpoint(self) -> list[float]:
        return [
            (self.a[0] + self.b[0]) / 2.0,
            (self.a[1] + self.b[1]) / 2.0,
            (self.a[2] + self.b[2]) / 2.0,
        ]


@dataclass(frozen=True, slots=True)
class RoofCluster:
    segments: list[RoofSegment]
    avg_incl: float
    avg_azimuth: float
    ref_pt: list[float]

    @property
    def dominant_story(self) -> int:
        counts: dict[int, int] = {}
        for segment in self.segments:
            counts[segment.story] = counts.get(segment.story, 0) + 1
        return max(counts, key=lambda story: (counts[story], story))


@dataclass(frozen=True, slots=True)
class Footprint:
    polygon_xz: list[tuple[float, float]]
    top_story: int
    area: float


@dataclass(frozen=True, slots=True)
class RoofPlaneCandidate:
    cluster: RoofCluster
    plane: Plane
    polygon_xz: list[tuple[float, float]]
    ridge_span: float
    dominant_story: int


@dataclass(frozen=True, slots=True)
class ClippedRoofPlane:
    candidate: RoofPlaneCandidate
    polygon_xz: list[tuple[float, float]]
    max_y: float | None = None


@dataclass(slots=True)
class ObliqueSurface:
    corners: list[list[float]]
    plane: Plane
    cluster: RoofCluster
    dominant_story: int
    ridge: dict[str, float]
    # Stable index assigned in `build_roof_model` before any filtering. Used
    # by `arrangement.py`, `_ceiling_candidates`, and `visual_shell.py` so
    # that cell IDs (`cell:<source_index>:room:<r>:<p>`) and locator IDs
    # remain consistent across runs even when prior filters drop surfaces.
    source_index: int = -1


@dataclass(frozen=True, slots=True)
class FlatSurface:
    corners: list[list[float]]
    story: int
    dominant_story: int
    source: str


@dataclass(frozen=True, slots=True)
class SplitObliqueSurface:
    corners: list[list[float]]
    plane: Plane
    arrangement_cell_id: str
    source_oblique_index: int
    dominant_story: int


@dataclass(frozen=True, slots=True)
class DormerCandidate:
    roof_surface_index: int
    room_index: int
    front_wall_id: str | None
    front_opening_id: str | None = None


@dataclass(frozen=True, slots=True)
class Dormer:
    roof_surface_index: int
    room_index: int
    front_wall_id: str | None
    cutout_quad: list[list[float]]
    cheek_quads: list[list[list[float]]]
    header_quad: list[list[float]]


@dataclass(frozen=True, slots=True)
class ThermalSurface:
    corners: list[list[float]]
    kind: KneeWallKind | DormerFaceKind
    room_index: int | None
    source: str
    cutouts: list[list[list[float]]] = field(default_factory=list)
    source_wall_id: str | None = None


@dataclass(frozen=True, slots=True)
class RoofKinks:
    """Y heights where the roof geometry changes character.

    All fields are scan-derived (no synthesis beyond the geometric plane intersection
    used for ridge_y). Per-room mappings are keyed by ExtractedRoom.index.

    eave_y_by_room: where the room's vertical wall ends and the slope begins
        (room.ceiling_eave_height for sloped rooms; same as the lid for flat rooms).
    attic_lid_y_by_room: where the room's vertical envelope ends and the attic
        begins. p10 of non-kneewall wall-top heights per room (with story-level
        fallback). Unset for cathedral rooms — the oblique runs to the ridge.
    ridge_y: building-level y where two opposing oblique planes intersect
        (gables only — None otherwise).
    """

    eave_y_by_room: dict[int, float] = field(default_factory=dict)
    attic_lid_y_by_room: dict[int, float] = field(default_factory=dict)
    ridge_y: float | None = None

    def eave_y(self, room_index: int) -> float | None:
        return self.eave_y_by_room.get(room_index)

    def attic_lid_y(self, room_index: int) -> float | None:
        return self.attic_lid_y_by_room.get(room_index)

    def min_eave_y(self) -> float | None:
        return min(self.eave_y_by_room.values()) if self.eave_y_by_room else None

    def min_attic_lid_y(self) -> float | None:
        return (
            min(self.attic_lid_y_by_room.values()) if self.attic_lid_y_by_room else None
        )


@dataclass(frozen=True, slots=True)
class RoofModel:
    simple_slant_room_indices: set[int]
    segments: list[RoofSegment]
    clusters: list[RoofCluster]
    footprint: Footprint | None
    planes: list[RoofPlaneCandidate]
    clipped_planes: list[ClippedRoofPlane]
    oblique: list[ObliqueSurface]
    flat: list[FlatSurface]
    oblique_split: list[SplitObliqueSurface]
    dormer_candidates: list[DormerCandidate]
    thermal: list[ThermalSurface]
    kinks: RoofKinks = field(default_factory=RoofKinks)
    # Phase 4: per-primitive params recorded at synthesis time. Each entry
    # captures the expected eave_y / ridge_y / wing_index for a primitive-
    # synthesised roof part. Used by `validate_primitive_emissions` after
    # the tier_payload is built — verifies the painted pieces still span
    # the full eave→ridge range no downstream clip truncated them.
    primitive_records: list = field(default_factory=list)


def build_roof_model(model: BuildingModel) -> RoofModel:

    from reconcile_tiers.roof.arrangement import split_oblique_surfaces
    from reconcile_tiers.roof.clipping import (
        clip_planes_to_footprint,
    )
    from reconcile_tiers.roof.clustering import cluster_oblique_segments
    from reconcile_tiers.roof.dormers import detect_dormers
    from reconcile_tiers.roof.flats import build_flat_surfaces
    from reconcile_tiers.roof.footprint import build_building_footprint
    from reconcile_tiers.roof.obliques import (
        build_oblique_surfaces,
        build_raw_oblique_surfaces,
        story_floor_y,
    )
    from reconcile_tiers.roof.planes import build_roof_planes
    from reconcile_tiers.roof.priors import filter_obliques_by_roof_type
    from reconcile_tiers.roof.segments import collect_oblique_segments
    from reconcile_tiers.roof.simple_slant import (
        build_simple_slant_surfaces,
        detect_simple_slant_rooms,
    )
    from reconcile_tiers.roof.thermal import (
        attic_lid_y_by_room,
        build_thermal_surfaces,
    )

    architectural_priors_on = architectural_priors_enabled()
    wall_axis_math: float | None = None
    if architectural_priors_on:
        from reconcile_tiers._core.wall_axis import axis_from_building_model

        axis_info = axis_from_building_model(model)
        if axis_info is not None and axis_info[1] >= 0.70:
            wall_axis_math = axis_info[0]

    simple_slants = detect_simple_slant_rooms(model)
    footprint = build_building_footprint(model)
    floors = story_floor_y(model)

    # Phase 5: per-building-part fan-out. Each wing gets its own segment
    # pool, cluster pass, plane fit, and clip. This keeps a wing's roof
    # geometry from being polluted by a neighbouring wing's segments and
    # eliminates the building-wide `ridge_y` cap that truncated taller
    # wings against shorter wings' ridges. The global `segments` /
    # `clusters` / `planes` / `clipped` lists returned in RoofModel are
    # now the union across all wings (wing-pooled). When wings can't be
    # computed (no footprint), we fall back to the legacy single-pass
    # behaviour over the whole model.
    from reconcile_tiers._core.wing_decomposition import compute_wings_for_model
    from reconcile_tiers.roof.part_record import PartRecord

    wings = compute_wings_for_model(model) if footprint is not None else []
    segments: list = []
    clusters: list = []
    planes: list = []
    clipped: list = []
    part_records: list[PartRecord] = []
    legacy_oblique: list = []
    # Phase 5 Step 5: per-wing primitive classifier. A clear gable primitive
    # is stronger evidence than noisy scan-derived oblique clusters because
    # it encodes the physical roof pattern: two mirrored slopes with a shared
    # ridge. Non-gable primitives remain fallbacks when the cluster path emits
    # no oblique surfaces.
    primitive_records: list = []
    classifier_active = architectural_priors_on and wall_axis_math is not None
    if classifier_active:
        from reconcile_tiers.roof_primitive import (
            classify_part,
            synthesise_gable,
            synthesise_oblique,
        )
        from reconcile_tiers.roof_primitive.types import (
            FlatParams,
            GableParams,
            ObliqueParams,
        )
    if wings and footprint is not None:
        for wing_index, wing in enumerate(wings):
            wing_segs = collect_oblique_segments(
                model,
                exclude_room_indices=simple_slants,
                wall_axis_math=wall_axis_math,
                wing_polygon=wing.polygon,
            )
            wing_clusters = cluster_oblique_segments(
                wing_segs, wall_axis_math=wall_axis_math
            )
            wing_planes = build_roof_planes(wing_clusters, footprint)
            _, wing_ridge_y = _eaves_and_ridge(model, wing_planes)
            wing_clipped = clip_planes_to_footprint(
                wing_planes, footprint, model, ridge_y=wing_ridge_y
            )
            wing_surfaces = build_oblique_surfaces(wing_clipped, floors)
            segments.extend(wing_segs)
            clusters.extend(wing_clusters)
            planes.extend(wing_planes)
            clipped.extend(wing_clipped)
            params = (
                classify_part(
                    wing,
                    wing_index,
                    model,
                    wall_axis_math=wall_axis_math,
                    exclude_room_indices=simple_slants,
                )
                if classifier_active
                else None
            )
            if classifier_active and isinstance(params, GableParams):
                primitive_surfaces = list(synthesise_gable(params))
                primitive_records.append(params)
                part_records.append(
                    PartRecord(
                        wing=wing,
                        kind="gable",
                        surfaces=primitive_surfaces,
                        params=params,
                    )
                )
                legacy_oblique.extend(primitive_surfaces)
                continue
            if wing_surfaces:
                legacy_oblique.extend(wing_surfaces)
                part_records.append(
                    PartRecord(
                        wing=wing,
                        kind="legacy",
                        surfaces=list(wing_surfaces),
                        params=None,
                    )
                )
                continue
            # Cluster path produced nothing for this wing; try the primitive
            # classifier as a fallback.
            if classifier_active and isinstance(params, ObliqueParams):
                primitive_surfaces = list(synthesise_oblique(params))
                primitive_records.append(params)
                part_records.append(
                    PartRecord(
                        wing=wing,
                        kind="oblique",
                        surfaces=primitive_surfaces,
                        params=params,
                    )
                )
                legacy_oblique.extend(primitive_surfaces)
            elif classifier_active and isinstance(params, FlatParams):
                primitive_records.append(params)
                part_records.append(
                    PartRecord(wing=wing, kind="flat", surfaces=[], params=params)
                )
            else:
                part_records.append(
                    PartRecord(wing=wing, kind="legacy", surfaces=[], params=None)
                )
    else:
        # No usable footprint → fall back to building-wide collection.
        segments = collect_oblique_segments(
            model,
            exclude_room_indices=simple_slants,
            wall_axis_math=wall_axis_math,
        )
        clusters = cluster_oblique_segments(segments, wall_axis_math=wall_axis_math)
        planes = build_roof_planes(clusters, footprint) if footprint is not None else []
        _, _global_ridge = _eaves_and_ridge(model, planes)
        clipped = (
            clip_planes_to_footprint(planes, footprint, model, ridge_y=_global_ridge)
            if footprint is not None
            else []
        )
        legacy_oblique = build_oblique_surfaces(clipped, floors)
        part_records.append(
            PartRecord(
                wing=None, kind="legacy", surfaces=list(legacy_oblique), params=None
            )
        )

    # Phase 5 Step 6: pairwise valley resolution between adjacent wings.
    # Where two wings share a boundary, their slope planes meet at a valley
    # line; each slope keeps the side where its own plane is higher. This
    # turns L/T/U-junctions into proper valleys instead of independent
    # per-wing fragments. Operates on the wing-pooled `part_records`;
    # rebuilds `legacy_oblique` from the clipped surfaces afterward.
    if architectural_priors_on and wings:
        from reconcile_tiers.roof.valleys import resolve_valleys

        part_records = resolve_valleys(part_records)
        legacy_oblique = [s for pr in part_records for s in pr.surfaces]

    # Recompute `eave_y_by_room` and a global `ridge_y` (max across all
    # wings) — used by the `is_gable` compat gate downstream.
    eave_y_by_room, ridge_y = _eaves_and_ridge(model, planes)
    oblique = list(legacy_oblique)
    oblique.extend(
        build_simple_slant_surfaces(model, simple_slants, wall_axis_math=wall_axis_math)
    )
    oblique.extend(
        build_raw_oblique_surfaces(model, oblique, wall_axis_math=wall_axis_math)
    )
    # Phase 5 Step 5: the primitive classifier now runs *first* in the
    # per-wing fan-out above (replacing the legacy cluster path for wings
    # that classify). The old post-cluster classifier with a 30% existing-
    # oblique coverage gate is retired.
    for original_idx, surface in enumerate(oblique):
        surface.source_index = original_idx
    oblique, _dropped_priors = filter_obliques_by_roof_type(oblique)
    # Phase 1 yaw-snap (`clean_oblique_surfaces`) was retired — its work is
    # done at LAYER 2 in `cluster_oblique_segments` above (priors-at-source).
    flat = build_flat_surfaces(model)
    split = split_oblique_surfaces(oblique, model)
    if architectural_priors_on:
        from reconcile_tiers.roof.architectural_priors import (
            clean_split_oblique_surfaces,
        )

        split = clean_split_oblique_surfaces(split)
    dormer_candidates = detect_dormers(model, oblique)
    thermal = build_thermal_surfaces(model, oblique)
    # Phase 5 Step 7: per-part kinks. `ridge_y` is the *max* across the
    # cluster-derived value and any classified parts' ridge heights — the
    # `is_gable` compat gate downstream just needs a non-None ridge_y when
    # the building has any gable. `eave_y_by_room` stays per-room (already
    # populated from `room.ceiling_eave_height` via `_eaves_and_ridge`,
    # which is the most accurate per-room value we have).
    primitive_ridge_ys = [
        float(p.ridge_y)
        for p in primitive_records
        if getattr(p, "ridge_y", None) is not None
    ]
    if primitive_ridge_ys:
        cluster_ridge = ridge_y if ridge_y is not None else float("-inf")
        ridge_y = max(cluster_ridge, max(primitive_ridge_ys))
    kinks = RoofKinks(
        eave_y_by_room=eave_y_by_room,
        attic_lid_y_by_room=attic_lid_y_by_room(model, thermal),
        ridge_y=ridge_y,
    )
    return RoofModel(
        simple_slant_room_indices=simple_slants,
        segments=segments,
        clusters=clusters,
        footprint=footprint,
        planes=planes,
        clipped_planes=clipped,
        oblique=oblique,
        flat=flat,
        oblique_split=split,
        dormer_candidates=dormer_candidates,
        thermal=thermal,
        kinks=kinks,
        primitive_records=primitive_records,
    )


def _eaves_and_ridge(
    model: BuildingModel, planes: list[RoofPlaneCandidate]
) -> tuple[dict[int, float], float | None]:
    from reconcile_tiers.roof.clipping import compute_gable_ridge_y

    eave_y_by_room: dict[int, float] = {}
    for room in model.rooms:
        if room.ceiling_eave_height is not None:
            eave_y_by_room[room.index] = float(room.ceiling_eave_height)
    return eave_y_by_room, compute_gable_ridge_y(planes)
