"""Per-room Kinetic Shape Reconstruction ceiling pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reconcile_tiers.ceiling_reconstruction.graph_cut import graph_cut_label_cells
from reconcile_tiers.ceiling_reconstruction.input_tiles import (
    KsrRoomTiles,
    collect_ksr_room_tiles,
    has_minimum_structure,
    room_bounding_prism,
)
from reconcile_tiers.ceiling_reconstruction.oriented_samples import sample_oriented_points
from reconcile_tiers.ceiling_reconstruction.plane_regularization import (
    PlaneGroup,
    detect_plane_groups,
    regularize_planes,
)
from reconcile_tiers.polyhedron.face_selection import CandidateFace
from reconcile_tiers.polyhedron.kinetic_partition import (
    BoundingPrism,
    ConvexCell,
    cells_to_candidate_faces,
    kinetic_partition,
)
from reconcile_tiers.polyhedron.manifold_repair import TileFace

_MIN_WALLS = 3
_CEILING_MIN_NY = 0.173  # sin(10 deg) upward-facing


@dataclass(slots=True)
class KsrPipelineResult:
    room_tiles: KsrRoomTiles
    structure_tiles: list[TileFace]
    evidence_tiles: list[TileFace]
    oriented_samples: np.ndarray
    oriented_samples_structure: np.ndarray
    oriented_samples_evidence: np.ndarray
    plane_groups_detected: list[PlaneGroup]
    plane_groups_regularized: list[PlaneGroup]
    partition_planes: list[Any]
    cells: list[ConvexCell]
    labels: dict[int, str]
    boundary_faces: list[CandidateFace]
    ceiling_faces: list[CandidateFace]
    bounding_prism: BoundingPrism | None
    graph_cut_status: str
    stop_reason: str
    stop_message: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tiles(self) -> list[TileFace]:
        """Backward-compatible alias: structure tiles for trace step 0."""
        return self.structure_tiles


def _empty_result(
    room_tiles: KsrRoomTiles,
    *,
    stop_reason: str,
    stop_message: str,
    graph_cut_status: str = "skipped",
    **meta: Any,
) -> KsrPipelineResult:
    empty6 = np.empty((0, 6), dtype=float)
    return KsrPipelineResult(
        room_tiles=room_tiles,
        structure_tiles=list(room_tiles.structure),
        evidence_tiles=list(room_tiles.evidence),
        oriented_samples=empty6,
        oriented_samples_structure=empty6,
        oriented_samples_evidence=empty6,
        plane_groups_detected=[],
        plane_groups_regularized=[],
        partition_planes=[],
        cells=[],
        labels={},
        boundary_faces=[],
        ceiling_faces=[],
        bounding_prism=None,
        graph_cut_status=graph_cut_status,
        stop_reason=stop_reason,
        stop_message=stop_message,
        meta=dict(meta),
    )


def _floor_y(tiles: Sequence[TileFace]) -> float | None:
    ys = [c[1] for tile in tiles if tile.source == "floor" for c in tile.corners]
    if not ys:
        ys = [c[1] for tile in tiles for c in tile.corners]
    return float(min(ys)) if ys else None


def _extract_ceiling_faces(
    candidates: Sequence[CandidateFace],
    *,
    floor_y: float | None,
) -> list[CandidateFace]:
    out: list[CandidateFace] = []
    for face in candidates:
        ny = float(face.plane.b)
        if ny < _CEILING_MIN_NY:
            continue
        avg_y = float(np.mean([c[1] for c in face.corners]))
        if floor_y is not None and avg_y <= floor_y + 0.05:
            continue
        out.append(face)
    return out


def _combine_samples(
    structure: np.ndarray,
    evidence: np.ndarray,
) -> np.ndarray:
    if structure.size == 0:
        return evidence
    if evidence.size == 0:
        return structure
    return np.vstack([structure, evidence])


def run_ksr_pipeline(
    payload: Mapping[str, Any],
    room: Mapping[str, Any],
    *,
    corner_tol: float = 0.02,
    graph_cut_lambda: float = 0.75,
    bbox_margin: float = 1.0,
    k_intersections: int = 2,
) -> KsrPipelineResult:
    """Run KSR steps for one segment-tier room."""
    _ = k_intersections  # reserved for future kinetic depth tuning
    room_tiles = collect_ksr_room_tiles(payload, room, corner_tol=corner_tol)

    if not has_minimum_structure(room_tiles, min_walls=_MIN_WALLS):
        return _empty_result(
            room_tiles,
            stop_reason="insufficient_tiles",
            stop_message="insufficient_tiles",
        )

    structure = room_tiles.structure
    evidence = room_tiles.evidence

    samples_structure = sample_oriented_points(structure)
    samples_evidence = (
        sample_oriented_points(evidence) if evidence else np.empty((0, 6), dtype=float)
    )
    samples_all = _combine_samples(samples_structure, samples_evidence)

    detected = detect_plane_groups(structure)
    regularized_groups, _ = regularize_planes(structure)
    partition_planes = [
        g.plane for g in regularized_groups if g.kind in ("wall", "floor")
    ]

    if not partition_planes:
        return KsrPipelineResult(
            room_tiles=room_tiles,
            structure_tiles=list(structure),
            evidence_tiles=list(evidence),
            oriented_samples=samples_all,
            oriented_samples_structure=samples_structure,
            oriented_samples_evidence=samples_evidence,
            plane_groups_detected=detected,
            plane_groups_regularized=regularized_groups,
            partition_planes=[],
            cells=[],
            labels={},
            boundary_faces=[],
            ceiling_faces=[],
            bounding_prism=None,
            graph_cut_status="skipped",
            stop_reason="partition_failed",
            stop_message="no_structure_planes",
        )

    try:
        prism = room_bounding_prism(structure, margin=bbox_margin)
    except ValueError:
        return KsrPipelineResult(
            room_tiles=room_tiles,
            structure_tiles=list(structure),
            evidence_tiles=list(evidence),
            oriented_samples=samples_all,
            oriented_samples_structure=samples_structure,
            oriented_samples_evidence=samples_evidence,
            plane_groups_detected=detected,
            plane_groups_regularized=regularized_groups,
            partition_planes=partition_planes,
            cells=[],
            labels={},
            boundary_faces=[],
            ceiling_faces=[],
            bounding_prism=None,
            graph_cut_status="skipped",
            stop_reason="partition_failed",
            stop_message="invalid_bbox",
        )

    xyz = (
        samples_structure[:, :3]
        if samples_structure.size
        else np.asarray([c for t in structure for c in t.corners], dtype=float)
    )
    cells = kinetic_partition(partition_planes, xyz, prism)
    if not cells:
        return KsrPipelineResult(
            room_tiles=room_tiles,
            structure_tiles=list(structure),
            evidence_tiles=list(evidence),
            oriented_samples=samples_all,
            oriented_samples_structure=samples_structure,
            oriented_samples_evidence=samples_evidence,
            plane_groups_detected=detected,
            plane_groups_regularized=regularized_groups,
            partition_planes=partition_planes,
            cells=[],
            labels={},
            boundary_faces=[],
            ceiling_faces=[],
            bounding_prism=prism,
            graph_cut_status="skipped",
            stop_reason="partition_failed",
            stop_message="empty_partition",
        )

    gc = graph_cut_label_cells(
        cells,
        samples_all,
        prism,
        lambda_param=graph_cut_lambda,
    )
    if gc.solver_status in ("no_cells",):
        return KsrPipelineResult(
            room_tiles=room_tiles,
            structure_tiles=list(structure),
            evidence_tiles=list(evidence),
            oriented_samples=samples_all,
            oriented_samples_structure=samples_structure,
            oriented_samples_evidence=samples_evidence,
            plane_groups_detected=detected,
            plane_groups_regularized=regularized_groups,
            partition_planes=partition_planes,
            cells=cells,
            labels={},
            boundary_faces=[],
            ceiling_faces=[],
            bounding_prism=prism,
            graph_cut_status=gc.solver_status,
            stop_reason="label_failed",
            stop_message=gc.solver_status,
        )

    boundary = cells_to_candidate_faces(cells, gc.labels)
    floor_y = _floor_y(structure)
    ceiling = _extract_ceiling_faces(boundary, floor_y=floor_y)
    base_meta = {
        "graph_cut_lambda": graph_cut_lambda,
        "data_energy": gc.data_energy,
        "smoothness_energy": gc.smoothness_energy,
        "cell_count": len(cells),
        "boundary_face_count": len(boundary),
        "partition_plane_count": len(partition_planes),
        "evidence_tile_count": len(evidence),
        "structure_sample_count": int(samples_structure.shape[0]),
        "evidence_sample_count": int(samples_evidence.shape[0]),
    }

    if not ceiling:
        return KsrPipelineResult(
            room_tiles=room_tiles,
            structure_tiles=list(structure),
            evidence_tiles=list(evidence),
            oriented_samples=samples_all,
            oriented_samples_structure=samples_structure,
            oriented_samples_evidence=samples_evidence,
            plane_groups_detected=detected,
            plane_groups_regularized=regularized_groups,
            partition_planes=partition_planes,
            cells=cells,
            labels=dict(gc.labels),
            boundary_faces=boundary,
            ceiling_faces=[],
            bounding_prism=prism,
            graph_cut_status=gc.solver_status,
            stop_reason="no_ceiling_faces",
            stop_message="no_ceiling_faces",
            meta=base_meta,
        )

    return KsrPipelineResult(
        room_tiles=room_tiles,
        structure_tiles=list(structure),
        evidence_tiles=list(evidence),
        oriented_samples=samples_all,
        oriented_samples_structure=samples_structure,
        oriented_samples_evidence=samples_evidence,
        plane_groups_detected=detected,
        plane_groups_regularized=regularized_groups,
        partition_planes=partition_planes,
        cells=cells,
        labels=dict(gc.labels),
        boundary_faces=boundary,
        ceiling_faces=ceiling,
        bounding_prism=prism,
        graph_cut_status=gc.solver_status,
        stop_reason="ceiling_extracted",
        stop_message="ceiling_extracted",
        meta={**base_meta, "ceiling_face_count": len(ceiling)},
    )
