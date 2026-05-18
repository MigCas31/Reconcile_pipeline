"""Helpers for relabeling painted CeilingPieces to the v2 source taxonomy
(FLAT_CEILING / COMPUTED_OBLIQUE / RAW_SCAN / ...) and computing the
per-piece raw-plane support quality. Consumed by
`synthesis.synthesise_thermal_envelope`.
"""

from __future__ import annotations

from reconcile_tiers.assemble.raw_quality import score_raw_plane
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import (
    CeilingPiece,
    CeilingRole,
    CeilingSource,
)
from reconcile_tiers.roof.roof import RoofModel

RAW_QUALITY_THRESHOLD = 0.55

V2_SOURCE_MAP: dict[CeilingSource, CeilingSource] = {
    CeilingSource.FLAT_EMIT: CeilingSource.FLAT_CEILING,
    CeilingSource.ATTIC_FLAT_LID: CeilingSource.FLAT_CEILING,
    CeilingSource.ROOF_ARRANGEMENT: CeilingSource.COMPUTED_OBLIQUE,
    CeilingSource.ROOF_ARRANGEMENT_ATTIC: CeilingSource.COMPUTED_OBLIQUE,
    CeilingSource.THERMAL_CAP: CeilingSource.FLAT_GAP_CLOSURE,
    CeilingSource.RAW_FALLBACK: CeilingSource.RAW_SCAN,
}


def _parse_room_idx(arrangement_cell_id: str | None) -> int | None:
    if not arrangement_cell_id:
        return None
    parts = arrangement_cell_id.split(":")
    if len(parts) >= 5 and parts[2] == "room":
        try:
            return int(parts[3])
        except ValueError:
            return None
    return None


def _parse_raw_locator(locator_id: str) -> tuple[int | None, int | None]:
    marker = "::tier-ceiling-raw::"
    if marker not in locator_id:
        return None, None
    suffix = locator_id.rsplit(marker, 1)[1]
    parts = suffix.split(":")
    if len(parts) < 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def _relabel_piece(
    piece: CeilingPiece, model: BuildingModel, roof: RoofModel
) -> CeilingPiece:
    v2_source = V2_SOURCE_MAP.get(piece.source, piece.source)
    new_locator = _v2_locator(piece.locator_id)
    quality = (
        _piece_raw_quality(piece, model, roof)
        if v2_source == CeilingSource.RAW_SCAN
        else 1.0
    )
    return CeilingPiece(
        corners=piece.corners,
        holes=piece.holes,
        plane=piece.plane,
        source=v2_source,
        arrangement_cell_id=piece.arrangement_cell_id,
        locator_id=new_locator,
        support_quality=quality,
        role=CeilingRole.CEILING,
    )


def _v2_locator(locator_id: str) -> str:
    replacements = (
        (
            "::tier-ceiling-roof-arrangement-room::",
            "::tier-ceiling-computed-oblique-room::",
        ),
        (
            "::tier-ceiling-roof-arrangement-attic::",
            "::tier-ceiling-computed-oblique-attic::",
        ),
        ("::tier-ceiling-roof-arrangement::", "::tier-ceiling-computed-oblique::"),
    )
    for old, new in replacements:
        if old in locator_id:
            return locator_id.replace(old, new)
    return locator_id


def _piece_raw_quality(
    piece: CeilingPiece, model: BuildingModel, roof: RoofModel
) -> float:
    room_idx, plane_idx = _parse_raw_locator(piece.locator_id)
    if room_idx is None or plane_idx is None:
        return 0.0
    room = next((r for r in model.rooms if r.index == room_idx), None)
    if room is None or plane_idx >= len(room.raw_ceiling_planes):
        return 0.0
    return score_raw_plane(room.raw_ceiling_planes[plane_idx], room, roof.oblique)
