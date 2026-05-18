"""Serialize pre-painter roof/ceiling plane evidence for reconstruction."""

from __future__ import annotations

from math import atan2, hypot
from typing import Any

from shapely.geometry import Polygon

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers.assemble.raw_quality import score_raw_plane
from reconcile_tiers.extract.building import BuildingModel, RawCeilingPlane
from reconcile_tiers.roof.roof import RoofModel


def build_plane_evidence(
    model: BuildingModel,
    roof: RoofModel,
    *,
    pre_filter_candidates: list[Any],
    post_filter_candidates: list[Any],
    raw_gate_drops: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return raw/candidate plane evidence before final payload painting.

    ``tier_payload.ceiling`` contains only the pieces that survived painting and
    post-processing. This sidecar preserves the candidate universe that a
    PolyFit/Kinetic-style selector needs: raw observed planes, generated roof
    candidates, and filtered-out raw candidates with their reasons.
    """

    kept_locators = {str(candidate.locator_id) for candidate in post_filter_candidates}
    drop_by_locator = {
        str(drop.get("locator_id")): drop
        for drop in raw_gate_drops or []
        if drop.get("locator_id") is not None
    }
    return {
        "schema_version": 1,
        "uuid": model.uuid,
        "rooms": [_room_record(room) for room in model.rooms],
        "raw_ceiling_planes": [
            _raw_plane_record(model, roof, room, plane_index, raw)
            for room in model.rooms
            for plane_index, raw in enumerate(room.raw_ceiling_planes)
        ],
        "ceiling_candidates": [
            _candidate_record(
                candidate,
                kept_after_raw_gate=str(candidate.locator_id) in kept_locators,
                raw_gate_drop=drop_by_locator.get(str(candidate.locator_id)),
            )
            for candidate in pre_filter_candidates
        ],
        "roof_oblique_surfaces": [
            _roof_surface_record(index, surface)
            for index, surface in enumerate(getattr(roof, "oblique", []) or [])
        ],
    }


def _room_record(room: Any) -> dict[str, Any]:
    return {
        "room_index": room.index,
        "story": room.story,
        "ceiling_type": room.ceiling_type,
        "ceiling_is_kinked": bool(getattr(room, "ceiling_is_kinked", False)),
        "raw_ceiling_source": room.raw_ceiling_source,
        "floor_polygon": _round_corners(room.floor_polygon),
        "ceiling_polygon": _round_corners(room.ceiling_polygon),
        "ceiling_flat_polygon": _round_corners(
            getattr(room, "ceiling_flat_polygon", [])
        ),
        "ceiling_slope_polygons": [
            _round_corners(poly)
            for poly in (getattr(room, "ceiling_slope_polygons", []) or [])
        ],
    }


def _raw_plane_record(
    model: BuildingModel,
    roof: RoofModel,
    room: Any,
    plane_index: int,
    raw: RawCeilingPlane,
) -> dict[str, Any]:
    plane = Plane.fit(raw.corners)
    plane_dict = _plane_dict(plane)
    quality = score_raw_plane(raw, room, getattr(roof, "oblique", []) or [])
    locator_id = f"{model.uuid}::raw-ceiling-plane::{room.index}:{plane_index}"
    return {
        "locator_id": locator_id,
        "room_index": room.index,
        "story": room.story,
        "raw_room_key": raw.raw_room_key,
        "raw_plane_index": raw.raw_plane_index
        if raw.raw_plane_index is not None
        else plane_index,
        "raw_source": raw.raw_source or room.raw_ceiling_source,
        "source": "raw_observed_ceiling_plane",
        "corners": _round_corners(raw.corners),
        "plane": plane_dict,
        "fit_status": plane if isinstance(plane, FitFailure) else "ok",
        "support_quality": float(quality),
        "area_xz_m2": _area_xz(raw.corners),
        "inclination_deg": _inclination_deg(plane),
    }


def _candidate_record(
    candidate: Any,
    *,
    kept_after_raw_gate: bool,
    raw_gate_drop: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "locator_id": str(candidate.locator_id),
        "story": candidate.story,
        "source": str(candidate.source),
        "arrangement_cell_id": candidate.arrangement_cell_id,
        "corners": _round_corners(candidate.corners),
        "plane": _plane_dict(candidate.plane),
        "area_xz_m2": _area_xz(candidate.corners),
        "inclination_deg": _inclination_deg(candidate.plane),
        "kept_after_raw_gate": kept_after_raw_gate,
        "raw_gate_drop": raw_gate_drop,
    }


def _roof_surface_record(index: int, surface: Any) -> dict[str, Any]:
    return {
        "surface_index": index,
        "locator_id": f"roof-oblique:{index}",
        "corners": _round_corners(surface.corners),
        "plane": _plane_dict(surface.plane),
        "area_xz_m2": _area_xz(surface.corners),
        "inclination_deg": _inclination_deg(surface.plane),
    }


def _plane_dict(plane: Plane | FitFailure | None) -> dict[str, float] | None:
    if plane is None or isinstance(plane, FitFailure):
        return None
    return {
        "a": round(float(plane.a), 9),
        "b": round(float(plane.b), 9),
        "c": round(float(plane.c), 9),
        "d": round(float(plane.d), 9),
    }


def _round_corners(corners: list[list[float]]) -> list[list[float]]:
    return [
        [
            round(float(corner[0]), 4),
            round(float(corner[1]), 4),
            round(float(corner[2]), 4),
        ]
        for corner in corners
    ]


def _area_xz(corners: list[list[float]]) -> float:
    if len(corners) < 3:
        return 0.0
    try:
        footprint = [(float(corner[0]), float(corner[2])) for corner in corners]
        return round(
            float(Polygon(footprint).buffer(0).area),
            4,
        )
    except Exception:
        return 0.0


def _inclination_deg(plane: Plane | FitFailure | None) -> float | None:
    if plane is None or isinstance(plane, FitFailure):
        return None
    radians = atan2(
        hypot(float(plane.a), float(plane.c)),
        abs(float(plane.b)),
    )
    return round(
        float(radians * 180.0 / 3.141592653589793),
        3,
    )
