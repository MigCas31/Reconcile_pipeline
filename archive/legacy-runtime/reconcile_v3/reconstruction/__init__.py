"""Reconstruction track (Phases A-D of the revised V3 plan).

Per-building roof-envelope reconstruction via binary integer programming
over clipped candidate faces. The GBM classifier score is demoted to a
per-face data-term prior; the decision is made at building level by the
solver, not per-segment.

Planned modules (not yet written):

- ``candidate_faces``  Phase A — expand merged-plane hypotheses into
  pairwise-clipped candidate faces, each carrying footprint, adjacency,
  support, and prior.
- ``solver``  Phase B — python-mip / pulp BIP that selects a subset of
  candidate faces maximising fit to scan evidence under coverage,
  non-overlap, azimuth-coherence, and topology constraints.
- ``triage``  Phase C — map solver status (solved / ambiguous /
  infeasible) plus LP gap and runner-up margin to
  auto_accept / review decisions.

Companion plan:
``~/.claude/plans/system-instruction-you-are-working-compiled-swing.md``.
Critical review: ``reports/slanted_roof_v3_critical_review_20260419.md``.
"""
