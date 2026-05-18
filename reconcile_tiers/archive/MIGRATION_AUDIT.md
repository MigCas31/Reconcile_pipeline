# reconcile_tiers Phase J Migration Audit

Date: 2026-04-26

Phase J status: blocked.

This audit intentionally does not delete the legacy tier viewer path. Phase I
passed for the new static artefacts, but it also exposed generated-payload tier
distribution drift that is too large for a same-turn migration.

The new static viewer path is separate from this legacy surface. The page at
`reconcile_tiers/web/viewer-tiers.html` must not import or fetch through
`reconcile/`, `reconcile_v2/`, `reconcile_v3/`, `/tier-index`, or
`/building-merged`; it should load only static `tier_index.json` and
`tier_payload.json` artefacts.

## Deletion Decision

No legacy files were deleted.

The old consumer path remains the regression baseline until the drift below is
explained or accepted:

| Tier | Legacy `/tier-index` count | New generated-payload count |
|---:|---:|---:|
| 1 | 85 | 85 |
| 2 | 10 | 8 |
| 3 | 0 | 0 |
| 4 | 5 | 3 |
| 5 | 20 | 24 |
| 6 | 25 | 25 |
| 7 | 57 | 38 |
| 8 | 21 | 40 |

The Phase I classifier-count test still passes against the legacy classifier
baseline. The drift appears when classifying payloads produced by the new
extract/roof path. That makes it a producer parity question, not a reason to
remove the old viewer/server code.

## Tier 1 Candidate Audit

| Candidate | Current consumer / blocker | Action |
|---|---|---|
| `reconcile/complexity_tiers.py` | Imported by `reconcile/viewer_server.py`, archived `tests/test_complexity_tiers.py`, `tests/reconcile_tiers/_core/test_legacy_parity.py`, `tests/reconcile_tiers/classify/test_tiers.py`, `tests/reconcile_tiers/test_phase_i_validation.py`, and archived `scripts/tier6_audit.py`. | Keep until the legacy `/tier-index` endpoint and legacy parity tests are retired or rewritten. |
| `reconcile/viewer-tiers.html` | Still fetches `/tier-index` and `/building-merged`; it is the old browser entry point. | Keep until users switch to `reconcile_tiers/web/viewer-tiers.html` or the old route is intentionally removed. |
| `reconcile/viewer-modules/tier-preview.js` | Imported by `reconcile/viewer-tiers.html`; documents the old `/building-merged` payload shape. | Keep with the old HTML until both are removed together. |
| `reconcile/viewer_server.py` tier blocks | `/tier-index` and `/building-merged` are still routed from the server; `_ensure_roof_caches` is shared with roof viewer handlers. | Do not remove line ranges piecemeal; split only after route migration and shared-cache audit. |
| `.context/audit_residual_tier_pieces.py` | Gitignored local audit script, not a tracked deletion candidate in this checkout. | No repository action. |
| `archive/legacy-runtime/tests/test_complexity_tiers.py` | Still covers the legacy classifier baseline used by Phase I, but is no longer part of the active tier test tree. | Keep archived until baseline tests move fully to `reconcile_tiers`. |

## Out Of Scope

Tier 2 deletion remains out of scope:

- `reconcile/extract_3d.py`
- `reconcile/extract3d/*`
- `reconcile/roof_algorithms_py/*`
- `reconcile/buildings_3d.json`
- `reconcile/roof_algorithms_py_results.json`
- `scripts/raw_ceiling_plane_scorer_v2/*`

These still feed the main viewer and the legacy validation baseline.

## Required Next Step

Before any deletion PR, resolve the generated-payload tier distribution drift.
The immediate useful checks are:

1. Port or approximate the legacy ceiling clipping/support-domain logic so
   each oblique surface covers its supported roof part rather than the broad
   building footprint.
2. Re-check legacy and new `n_oblique` / `n_flat` per UUID after the clipping
   fix.
3. Promote any accepted drift to an explicit decision log entry and update the
   Phase I tier-count gate accordingly.

## Drift Localization

Investigation on the 82 legacy tier 6/7 buildings localized the collapse to
the new roof model, not building/story extraction and not the classifier port.

Stage-swap result:

| Input combination | Result |
|---|---|
| Legacy building + legacy roof + legacy classifier | 82/82 remain tier 6/7 |
| Legacy building + legacy roof + new classifier | 82/82 remain tier 6/7 |
| New building + legacy roof + new classifier | 82/82 remain tier 6/7 |
| Legacy building + new roof + new classifier | 80/82 fall out of tier 6/7 |
| New building + new roof + new classifier | 80/82 fall out of tier 6/7 |

The roof-stage failure modes among the 82 legacy tier 6/7 homes were:

| Failure mode in new roof signals | Count |
|---|---:|
| No opposing similar-pitch oblique pair | 46 |
| Best opposing pair covers less than 70% of oblique area | 21 |
| Fewer than two valid oblique surfaces | 13 |
| Still classifies as tier 6/7 | 2 |

The first concrete break is the handoff from segment clustering to plane/surface
generation:

- Segment collection still sees the physical opposite roof directions. Example
  `019e1376-9762-42d6-8520-b664b8c752df` has segment azimuths near `13°` and
  `193°`.
- `reconcile_tiers.roof.clustering` intentionally uses a bidirectional
  180-degree axis and stores `avg_azimuth % 180`. That keeps roof-axis grouping,
  but loses which face points to which side.
- `build_roof_planes()` then fits one plane per bidirectional cluster. When a
  cluster mixes both roof faces, the fitted plane can become nearly horizontal
  or fail entirely.
- `build_oblique_surfaces()` currently trusts `cluster.avg_incl` and
  `cluster.avg_azimuth` instead of the fitted plane's actual inclination and
  directional slope. This can emit a nearly horizontal fitted plane as an
  oblique surface and can make two opposite roof faces look same-facing to
  `detect_gable`.

Representative examples:

| UUID | Legacy signal | New roof signal | Localized issue |
|---|---|---|---|
| `019e1376-9762-42d6-8520-b664b8c752df` | 2 oblique, 6 flat, gable | 2 oblique, 4 flat, no gable | Cluster mixes `13°/193°`; one fitted plane is only `0.3°` inclined but is emitted as oblique. |
| `0d3f2993-8386-4130-8f1c-b2938c410828` | 4 oblique, 3 flat, gable | 4 oblique, 6 flat, no gable | Fitted planes have opposite slope directions, but surface metadata reports same-axis cluster azimuths (`92°/93°` and `3°/3°`). |
| `6203a969-742b-4935-bc4d-8eae644b8f73` | 2 oblique, 12 flat, gable | 0 oblique, 16 flat, no gable | Opposite directions are clustered bidirectionally, then no roof plane survives fitting. |

A local simulation that derived oblique azimuth/inclination from fitted planes
and chose the 180-degree sign closest to the directional segment mean recovered
33/82 legacy tier 6/7 buildings, compared with 2/82 in the current new roof
signals. That is a strong first fix target, but it does not resolve all drift:
some buildings still have missing planes or extra raw/low-area oblique pieces
that dilute the 70% gable-area predicate.

Recommended fix order:

1. Preserve directional face metadata alongside the bidirectional roof-axis
   cluster, or split bidirectional clusters into per-face plane candidates
   before fitting.
2. Reject emitted oblique surfaces whose fitted plane inclination is outside
   the oblique range, regardless of cluster inclination.
3. Re-run the 82-building legacy tier 6/7 stage-swap diagnostic and only then
   inspect the remaining area-dilution and missing-plane cases.

## Directional Roof Refactor Result

The first fix target was implemented by matching the legacy directional
clustering/plane-generation semantics:

- Clustering now uses ordinary shortest-arc `angle_diff` over full `0..360`
  azimuths, so opposite gable faces remain separate clusters.
- Cluster averages use a normal circular mean, not a doubled-angle
  bidirectional mean.
- Roof plane candidates are generated analytically from directional
  `avg_azimuth` / `avg_incl` and the cluster midpoint reference, matching the
  legacy `plane_normal(avgAzimuth, avgIncl)` path.

Result on the same 82 legacy tier 6/7 buildings:

| New full-payload tier after refactor | Count |
|---:|---:|
| 1 | 2 |
| 5 | 4 |
| 6 | 23 |
| 7 | 35 |
| 8 | 18 |

So tier 6/7 recovery improved from `2/82` to `58/82`.

Full corpus tier counts after the refactor:

| Tier | Legacy `/tier-index` count | New generated-payload count |
|---:|---:|---:|
| 1 | 85 | 92 |
| 2 | 10 | 5 |
| 3 | 0 | 0 |
| 4 | 5 | 2 |
| 5 | 20 | 19 |
| 6 | 25 | 24 |
| 7 | 57 | 38 |
| 8 | 21 | 43 |

The major directional regression is fixed, but deletion remains blocked: tier 7
is still under-recovered and tier 8 is still over-produced. Remaining failures
are now mostly residual gable-area dilution or missing/extra roof planes, not
the original opposite-face clustering collapse.

## Storey And Raw-Oblique Cleanup Result

A follow-up check showed storey extraction is not the current drift source:
`n_stories` and `split_level` match legacy for all 223 buildings in this
checkout.

The new pipeline was updated to preserve sloped wall-top ceilings, emit
simple-slant oblique surfaces for rooms excluded from segment clustering, and
tighten raw-ceiling fallback to the legacy clean-rectangle guards:

- 4 raw ceiling corners with 4 unique XZ points
- XZ area >= 5 m²
- 10-75° inclination
- <= 0.08 m plane residual
- at least two ridge-like edges >= 2 m
- duplicate-plane suppression against existing oblique surfaces

Full corpus tier counts after this cleanup:

| Tier | Legacy `/tier-index` count | New generated-payload count |
|---:|---:|---:|
| 1 | 85 | 85 |
| 2 | 10 | 8 |
| 3 | 0 | 0 |
| 4 | 5 | 3 |
| 5 | 20 | 24 |
| 6 | 25 | 25 |
| 7 | 57 | 38 |
| 8 | 21 | 40 |

The remaining tier 7 under-recovery is no longer a storey/split-level problem
and no longer the original directional-clustering problem. It is now mainly
oblique surface support-domain drift. Example:
`0d3f2993-8386-4130-8f1c-b2938c410828` has legacy oblique areas around
`7, 7, 18, 23 m²`, while the new simplified clipping emits five surfaces around
`222-247 m²`, diluting the 70% dominant-gable-area predicate.

## Oblique Support-Domain Clipping Result

The simplified full-footprint oblique clipping was tightened to follow the
original ceiling clipping intent:

- per-plane support domains come from the rooms that contributed sloped wall
  evidence, buffered by 1 m and intersected with the scanned building envelope;
- ridge/slope extents are clipped from the contributing segments and room
  footprints;
- projected 3D obliques are clipped above the story floor and below observed
  contributing wall/ceiling height plus a 0.5 m eave allowance.

For `0d3f2993-8386-4130-8f1c-b2938c410828`, roof-arrangement payload pieces now
have Y range `0.942..3.681 m` instead of reaching ~21 m, which removes the
static viewer's wall-like projected slabs.

Full corpus tier counts after support-domain clipping:

| Tier | Legacy `/tier-index` count | New generated-payload count |
|---:|---:|---:|
| 1 | 85 | 85 |
| 2 | 10 | 8 |
| 3 | 0 | 0 |
| 4 | 5 | 3 |
| 5 | 20 | 23 |
| 6 | 25 | 23 |
| 7 | 57 | 48 |
| 8 | 21 | 33 |

Tier 7 recovery improved from 38 to 48 buildings. Migration deletion still
remains blocked because tier 8 is still over-produced relative to legacy, but
the broad-oblique support-domain failure is no longer the main visual blocker.
