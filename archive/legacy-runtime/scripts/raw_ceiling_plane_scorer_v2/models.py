from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlaneContext:
    target_id: str
    story: int
    source_edge_ids: tuple[str, ...]
    source_edge_overlap_m: float
    source_edge_alignment_deg: float | None
    source_edge_height_residual_m: float | None
    building_part_ids: tuple[str, ...]
    source_part_ids: tuple[str, ...]
    crossed_part_ids: tuple[str, ...]
    in_main_body: bool
    in_extension: bool
    crosses_main_extension_boundary: bool


@dataclass(frozen=True)
class PlaneRelation:
    a_target_id: str
    b_target_id: str
    relation_kind: str
    overlap_m2: float
    azimuth_delta_deg: float | None
    inclination_delta_deg: float | None
    height_residual_m: float | None
    same_story: bool
    same_part: bool
    shared_boundary_len_m: float = 0.0
    covering_target_id: str | None = None
    covered_target_id: str | None = None


@dataclass
class RelationGraph:
    contexts_by_target: dict[str, PlaneContext]
    relations: list[PlaneRelation]
    relations_by_target: dict[str, list[PlaneRelation]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.relations_by_target:
            return
        out: dict[str, list[PlaneRelation]] = {
            target_id: [] for target_id in self.contexts_by_target
        }
        for relation in self.relations:
            out.setdefault(relation.a_target_id, []).append(relation)
            out.setdefault(relation.b_target_id, []).append(relation)
        self.relations_by_target = out


@dataclass(frozen=True)
class RunOutput:
    per_target_rows: list[dict[str, Any]]
    per_story_rows: list[dict[str, Any]]
    split_piece_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    relation_sidecar: dict[str, Any]


@dataclass(frozen=True)
class BuildingResult:
    building_uuid: str
    per_target_rows: list[dict[str, Any]]
    per_story_rows: list[dict[str, Any]]
    split_piece_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    relation_sidecar: dict[str, Any]


@dataclass(frozen=True)
class CorpusResult:
    per_target_rows: list[dict[str, Any]]
    per_story_rows: list[dict[str, Any]]
    split_piece_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    relation_sidecar: dict[str, Any]
