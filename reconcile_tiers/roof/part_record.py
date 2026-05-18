"""Per-building-part roof reconstruction unit (Phase 5).

A `PartRecord` is the per-wing slot that owns:
- the wing this part covers (geometry + axis)
- the primitive params we identified for this wing's roof, if any
- the surfaces emitted for this wing (parametric synthesis when classified,
  cluster-derived fallback otherwise)
- the kind label (`"gable"`, `"flat"`, `"legacy"`, ...) -- drives downstream
  behaviour like which downstream gates apply.

The Phase 5 pipeline replaces the building-wide cluster spine with a loop
that produces one `PartRecord` per wing. Surfaces from all parts are
flattened into `roof.oblique` after valley-resolution, so downstream
consumers (`_ceiling_candidates`, painter, dormer_reconstruction,
thermal_envelope) see the same shape they always have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reconcile_tiers._core.wing_decomposition import Wing
    from reconcile_tiers.roof.roof import ObliqueSurface


@dataclass(frozen=True, slots=True)
class PartRecord:
    """One building part's roof contribution.

    `wing` is None for the legacy whole-building shim used during the
    Phase 5 transition (Steps 3-4). Once the per-wing fan-out lands
    (Step 4 onward), every record carries a wing.
    """

    wing: Wing | None
    kind: str  # "gable" | "flat" | "legacy" | ...
    surfaces: list[ObliqueSurface] = field(default_factory=list)
    params: Any = None  # `RoofPartParams | None`; typed loosely to avoid import cycle
