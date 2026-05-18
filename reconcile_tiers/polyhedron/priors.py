"""Default priors for the polyhedron reconstruction pipeline.

Stage D will replace these hand-seeded defaults with corpus-fitted values.
Until then they are intentionally conservative and only drive the standalone
Stage A selector tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Priors:
    epsilon_meters: float
    sharp_edge_radians: float
    alpha_shape_radius: float
    weight_data_fit: float
    weight_complexity: float
    weight_coverage: float
    min_support_points: int


DEFAULT = Priors(
    epsilon_meters=0.05,
    sharp_edge_radians=0.35,
    alpha_shape_radius=0.50,
    weight_data_fit=0.43,
    weight_complexity=0.27,
    weight_coverage=0.30,
    min_support_points=0,
)


@dataclass(frozen=True, slots=True)
class SelectionWeights:
    data_fit: float = DEFAULT.weight_data_fit
    complexity: float = DEFAULT.weight_complexity
    coverage: float = DEFAULT.weight_coverage

