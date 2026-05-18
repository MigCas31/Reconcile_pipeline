"""Wing decomposition wrappers used by `reconcile_tiers.build`.

Thin facade over `_core.wing_decomposition` (and the v2 variant gated by
the WING_DETECTION_V2 env var). Re-exported from `reconcile_tiers.build`.
"""

from __future__ import annotations

import os

from reconcile_tiers.extract.building import BuildingModel


def _compute_wings(model: BuildingModel) -> list:
    if os.environ.get("WING_DETECTION_V2") == "1":
        from reconcile_tiers._core.wing_decomposition_v2 import (
            compute_wings_for_model_v2,
        )

        return compute_wings_for_model_v2(model)
    from reconcile_tiers._core.wing_decomposition import compute_wings_for_model

    return compute_wings_for_model(model)


def _wing_polygon_for_room(room, wings):
    from reconcile_tiers._core.wing_decomposition import wing_polygon_for_room

    return wing_polygon_for_room(room, wings)


def _filter_obliques_by_wing(obliques, wing_poly):
    from reconcile_tiers._core.wing_decomposition import filter_by_wing_xz
    from reconcile_tiers.build_internals.gable_selection import _oblique_xz_polygon

    return filter_by_wing_xz(obliques, wing_poly, _oblique_xz_polygon)


def _filter_candidates_by_wing(candidates, wing_poly):
    from shapely.geometry import Polygon as _ShPolygon

    from reconcile_tiers._core.shapely2 import make_valid_polygon
    from reconcile_tiers._core.wing_decomposition import filter_by_wing_xz

    def _candidate_xz(cand):
        ring = getattr(cand, "polygon_xz", None)
        if not ring or len(ring) < 3:
            return None
        return make_valid_polygon(_ShPolygon([(float(x), float(z)) for x, z in ring]))

    return filter_by_wing_xz(candidates, wing_poly, _candidate_xz)
