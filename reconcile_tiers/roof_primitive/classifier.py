"""Top-level roof-part classifier.

Tries each registered primitive in order and returns the first match.
The primitive vocabulary stays intentionally small: flat parts, single
axis-snapped oblique parts, and composites such as gables.
"""

from __future__ import annotations

from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof_primitive.flat import classify_flat
from reconcile_tiers.roof_primitive.gable import classify_gable
from reconcile_tiers.roof_primitive.shed import classify_oblique
from reconcile_tiers.roof_primitive.types import RoofPartParams


def classify_part(
    wing,
    wing_index: int,
    model: BuildingModel,
    *,
    wall_axis_math: float,
    exclude_room_indices: set[int] | None = None,
) -> RoofPartParams | None:
    """Classify the roof shape for a single wing. Returns None if no
    primitive matches (caller should fall back to scan-derived geometry)."""
    params = classify_gable(wing, wing_index, model, wall_axis_math=wall_axis_math)
    if params is not None:
        return params
    params = classify_oblique(
        wing,
        wing_index,
        model,
        exclude_room_indices=exclude_room_indices,
    )
    if params is not None:
        return params
    params = classify_flat(
        wing,
        wing_index,
        model,
        exclude_room_indices=exclude_room_indices,
    )
    if params is not None:
        return params
    return None
