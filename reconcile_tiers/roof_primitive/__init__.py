"""Roof-part primitive classifier + synthesiser.

Given a wing of the building and the underlying scan-derived geometry, pick
a roof primitive (flat, oblique, or a composite such as gable) and synthesise
the surfaces parametrically — corners on the plane by construction,
axis-aligned by construction, eaves on the wing's footprint by construction.

This is the priors-driven roof reconstruction. Surfaces emitted from this
module need no post-snap, no post-filter, no post-edit — they are the right
shape because they were generated from the priors directly.

Entry points:
    classify_part(wing, model) -> RoofPartParams | None
    synthesise_part(params) -> list[ObliqueSurface | FlatSurface]
"""

from reconcile_tiers.roof_primitive.classifier import classify_part
from reconcile_tiers.roof_primitive.flat import classify_flat
from reconcile_tiers.roof_primitive.gable import classify_gable, synthesise_gable
from reconcile_tiers.roof_primitive.shed import (
    classify_oblique,
    classify_shed,
    synthesise_oblique,
    synthesise_shed,
)
from reconcile_tiers.roof_primitive.types import (
    FlatParams,
    GableParams,
    ObliqueParams,
    PrimitiveKind,
    RoofPartParams,
    ShedParams,
)

__all__ = [
    "FlatParams",
    "GableParams",
    "ObliqueParams",
    "PrimitiveKind",
    "RoofPartParams",
    "ShedParams",
    "classify_flat",
    "classify_gable",
    "classify_oblique",
    "classify_part",
    "classify_shed",
    "synthesise_gable",
    "synthesise_oblique",
    "synthesise_shed",
]
