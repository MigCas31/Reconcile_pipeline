"""Digital-twin model: typed primitives for the reconciled building.

Phase A scope (this module): the type hierarchy, structural invariants
enforced at construction, and the provenance/evidence framework. No
assembly algorithm yet; that arrives in Phase B.

See `.context/digital_twin_walkthrough/implementation_sketch.md` for the
overall design and phasing.
"""

from __future__ import annotations

from reconcile_tiers.twin.queries import (
    Contender,
    ResolvedY,
    ceiling_y_at,
    contenders_at,
    roof_y_at,
)
from reconcile_tiers.twin.serialise import twin_to_payload
from reconcile_tiers.twin.types import (
    FLOAT_EPS,
    Building,
    Ceiling,
    CeilingKind,
    CeilingSeam,
    Dormer,
    Eave,
    Evidence,
    Floor,
    Gable,
    Gap,
    GapKind,
    InvariantViolation,
    KneeWall,
    Opening,
    OpeningKind,
    Primitive,
    Provenance,
    ProvenanceKind,
    Residual,
    Ridge,
    Roof,
    RoofKind,
    RoofSurface,
    Room,
    Stair,
    Story,
    Twin,
    Wall,
    Wing,
)

__all__ = [
    "FLOAT_EPS",
    "Building",
    "Ceiling",
    "CeilingKind",
    "CeilingSeam",
    "Contender",
    "Dormer",
    "Eave",
    "Evidence",
    "Floor",
    "Gable",
    "Gap",
    "GapKind",
    "InvariantViolation",
    "KneeWall",
    "Opening",
    "OpeningKind",
    "Primitive",
    "Provenance",
    "ProvenanceKind",
    "Residual",
    "ResolvedY",
    "Ridge",
    "Roof",
    "RoofKind",
    "RoofSurface",
    "Room",
    "Stair",
    "Story",
    "Twin",
    "Wall",
    "Wing",
    "ceiling_y_at",
    "contenders_at",
    "roof_y_at",
    "twin_to_payload",
]
