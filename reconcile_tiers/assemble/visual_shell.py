"""v2 visual shell assembly.

A visual shell is a translucent oblique surface drawn ON TOP OF the thermal
envelope, purely for rendering — it never participates in (x,z) coverage
selection. v0 emits one tag:

- GABLE_END: the full unclipped oblique surface for each member of a gable
  pair on a gable-classified building. v1 used to inject these as
  ROOF_ARRANGEMENT_ATTIC pieces at the synthetic ATTIC_SHELL_STORY=-1, where
  they competed for visualization in the same payload field as real thermal
  pieces. In v2 they live in their own field with an explicit tag.

ABOVE_CEILING_BELOW_RIDGE and OUTSIDE_KNEEWALL_BELOW_EAVE are reserved for
later iterations — they require differencing each oblique against the thermal
envelope and intersecting with explicit ridge / kneewall transition lines.
"""

from __future__ import annotations

from reconcile_tiers._core.plane import FitFailure, Plane
from reconcile_tiers._core.wing_decomposition import (
    clip_3d_polygon_to_wing,
    compute_wings_for_model,
    wing_polygon_for_xz_polygon,
)
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.payload.schema import (
    Plane as PayloadPlane,
)
from reconcile_tiers.payload.schema import (
    ShellTag,
    Vec3,
    VisualShellPiece,
)
from reconcile_tiers.roof.roof import RoofModel


def assemble_visual_shells(
    uuid: str, roof: RoofModel, model: BuildingModel | None = None
) -> list[VisualShellPiece]:
    if roof.kinks.ridge_y is None:
        return []
    wings = compute_wings_for_model(model) if model is not None else []
    shells: list[VisualShellPiece] = []
    for fallback_idx, surface in enumerate(roof.oblique):
        if len(surface.corners) < 3:
            continue
        # Drop near-flat surfaces that sit in a separate (lower) pitch
        # band from steeper surfaces in the same building — those are
        # scan artefacts (attic-floor scraps, dormer scraps), not real
        # gable ends. A uniformly low-pitch gable keeps all its shells.
        if float(
            surface.cluster.avg_incl
        ) < GABLE_END_MIN_PITCH_DEG and _has_steeper_band_member(surface, roof.oblique):
            continue
        if not _has_gable_partner(surface, roof.oblique):
            continue
        plane = Plane.fit(surface.corners)
        if isinstance(plane, FitFailure):
            continue
        idx = surface.source_index if surface.source_index >= 0 else fallback_idx

        wing_poly = _wing_for_corners(surface.corners, wings)
        clipped_parts = clip_3d_polygon_to_wing(surface.corners, plane, wing_poly)
        if not clipped_parts:
            continue
        for part_idx, part in enumerate(clipped_parts):
            corners = [Vec3(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in part]
            locator_id = (
                f"{uuid}::tier-shell-gable-end::{idx}"
                if len(clipped_parts) == 1
                else f"{uuid}::tier-shell-gable-end::{idx}:{part_idx}"
            )
            shells.append(
                VisualShellPiece(
                    corners=corners,
                    plane=PayloadPlane(a=plane.a, b=plane.b, c=plane.c, d=plane.d),
                    tag=ShellTag.GABLE_END,
                    source_oblique_index=idx,
                    locator_id=locator_id,
                )
            )
    return shells


def _wing_for_corners(corners, wings):
    if not wings or len(wings) == 1 or len(corners) < 3:
        return wings[0].polygon if wings else None
    from shapely.geometry import Polygon

    from reconcile_tiers._core.shapely2 import make_valid_polygon

    xz = make_valid_polygon(Polygon([(float(p[0]), float(p[2])) for p in corners]))
    if xz is None or xz.is_empty:
        return None
    return wing_polygon_for_xz_polygon(xz, wings)


GABLE_PARTNER_AZIMUTH_TOLERANCE_DEG = 40.0
GABLE_PARTNER_INCL_TOLERANCE_DEG = 10.0
# Below this pitch a surface no longer sheds water as a roof. When such a
# surface coexists with a steeper band (gap >= GABLE_END_PITCH_BAND_GAP_DEG),
# it is treated as scan artefact and skipped from gable-end tagging so the
# viewer matches the classifier's artefact-strip behaviour.
GABLE_END_MIN_PITCH_DEG = 15.0
GABLE_END_PITCH_BAND_GAP_DEG = 20.0


def _has_steeper_band_member(surface, all_obliques) -> bool:
    base = float(surface.cluster.avg_incl)
    return any(
        float(other.cluster.avg_incl) - base >= GABLE_END_PITCH_BAND_GAP_DEG
        for other in all_obliques
        if other is not surface
    )


def _has_gable_partner(surface, all_obliques) -> bool:
    from reconcile_tiers.roof.geometry import angle_diff_deg

    for other in all_obliques:
        if other is surface:
            continue
        diff = angle_diff_deg(surface.cluster.avg_azimuth, other.cluster.avg_azimuth)
        if (
            not (180.0 - GABLE_PARTNER_AZIMUTH_TOLERANCE_DEG)
            <= diff
            <= (180.0 + GABLE_PARTNER_AZIMUTH_TOLERANCE_DEG)
        ):
            continue
        if (
            abs(surface.cluster.avg_incl - other.cluster.avg_incl)
            > GABLE_PARTNER_INCL_TOLERANCE_DEG
        ):
            continue
        return True
    return False
