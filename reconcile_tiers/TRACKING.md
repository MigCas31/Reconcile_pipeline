# reconcile_tiers — Team Tracking Document

**Status**: Phase 0 complete (2026-04-26). Phases A–I in review. Phase J blocked pending migration alignment.
**Mission**: Replace the existing `merged.json → viewer-tiers.html` data path
with a clean, self-contained `reconcile_tiers/` package. Read `merged.json` and
the `.scan-cache/` directly; produce typed, validated `tier_payload.json`
artefacts; render via static-file serving.

The existing producers (`reconcile/extract_3d.py`, `reconcile/extract3d/`,
`reconcile/roof_algorithms_py/`) keep running unchanged — they still feed the
main viewer's ontology overlays. Deletion scope is decided post-validation
(Phase J).

> **Read this document end-to-end before claiming a phase.** The "Pitfalls"
> section is the most concentrated source of "don't make this mistake" in the
> repo. Half the smells in the current code are mistakes the original authors
> would not make today; the other half are intentional contracts that look like
> mistakes. This document tells you which is which.
>
> **Way of working: Test-Driven Development.** Every line of code in this
> package is written test-first. See §2 for the loop, the per-phase first
> tests, and how golden fixtures are bootstrapped. PRs that don't show
> red→green→refactor in commit history will be sent back.

---

## Table of contents

1. [Required reading](#required-reading)
2. [Way of working — Test-Driven Development](#way-of-working)
3. [Work breakdown](#work-breakdown)
4. [Critical references (where to look for everything)](#critical-references)
5. [Pitfalls — do not repeat these](#pitfalls)
6. [Load-bearing decisions — do not change these](#load-bearing-decisions)
7. [Decision log](#decision-log)
8. [Coordination protocol](#coordination-protocol)
9. [Verification checklist per phase](#verification-checklist-per-phase)
10. [Out of scope](#out-of-scope)

---

<a id="required-reading"></a>
## 1. Required reading (in this order)

| # | What | Where | Why |
|---|---|---|---|
| 1 | Repo orientation | `CLAUDE.md` (repo root) | Tech stack, "think in buildings", coding principles, gotchas |
| 2 | This document | `reconcile_tiers/TRACKING.md` | The team's working document |
| 3 | Implementation plan | `~/.claude/plans/system-instruction-you-are-working-soft-firefly.md` | Phase-by-phase blueprint with thresholds and file structure |
| 4 | Architecture review | `.context/plans/viewer-tiers-architecture-review.md` | Why the design is shaped this way (painter's algorithm, wire format, etc.) |
| 5 | Source-code walkthrough | `.context/attachments/pasted_text_2026-04-25_14-44-10.txt` (1303 lines) | Tagged claims about the current path; cite `file:line` when porting |
| 6 | Phase 0 audit results | `.context/phase0_audit_results.json` + summary in `tracking_progress.md` (entry 2026-04-26) | Cohort metrics that justify our defaults (V2 retire, thermal-cap include, V1 stories_found) |

> **`.context/` is gitignored.** If you need to share something with someone
> outside the local checkout, copy it into the repo (e.g. into
> `reconcile_tiers/`) before referencing it.

Skills (read the relevant ones before touching their domain):

- `.agents/skills/topology-v2/SKILL.md` — only relevant if you change `reconcile_v2/` (we don't, but the V2 graph is mentioned)
- `.agents/skills/extraction-pipeline/SKILL.md` — read before Phase D
- `.agents/skills/roof-pipeline/SKILL.md` — read before Phase E (note: walkthrough §4.5 says this skill is stale at "9 stages"; the actual roof pipeline runs 18+; the skill needs a refresh)
- `.agents/skills/viewer/SKILL.md` — read before Phase H
- `.agents/skills/run-and-verify/SKILL.md` — read before Phase I
- `.agents/skills/debug-element/SKILL.md` — read before wiring up locator IDs in Phase H

---

<a id="way-of-working"></a>
## 2. Way of working — Test-Driven Development

**Every line of code in `reconcile_tiers/` is written test-first.** This is
not aspirational; it is how PRs get reviewed and merged. The plan and the
architecture review both lean on contracts, invariants, and cohort metrics —
all of which are easier to write as tests *before* the code than to retrofit
afterwards.

### 2.1 The TDD loop we use

For each unit of work — a function, a dataclass, a phase deliverable — repeat:

1. **Red**: write a failing test that captures the behaviour you're about to
   add. Run it, watch it fail for the right reason. (A test that fails for
   `ImportError` doesn't count — make it fail on assertion.)
2. **Green**: write the *minimum* implementation that makes the test pass.
   Resist the urge to add functionality the test doesn't cover.
3. **Refactor**: clean up the implementation while staying green. Run the
   full module's test file; if any other test goes red, you've introduced
   a regression — fix it before continuing.

A commit boundary that is **green tests + minimum code** is the unit we
review and bisect against. PRs whose first commit adds an implementation
without a paired failing-test commit will be sent back.

### 2.2 What "test-first" means concretely per phase

Each phase has a natural "first test" that is cheap to write and gates the
implementation. Use these as your starting points:

| Phase | First failing test (write before any module code) |
|---|---|
| A — `_core/` | `test_plane.py::test_fit_horizontal_plane` (synthetic four corners at y=2.5; expect `Plane(0, 1, 0, 2.5)` after normalisation) |
| B — `payload/` | `test_schema.py::test_schema_emits_expected_top_level_keys` (asserts `TierPayload` schema has `schema_version`, `uuid`, `rooms`, `gaps`, `ceiling`, `knee_walls`, `classification`) |
| C — `ingest/` | `test_merged.py::test_load_merged_returns_known_room_count` (3 fixture UUIDs with hand-counted rooms; expected counts hard-coded) |
| D — `extract/` | One snapshot test per module whose input is a fixture UUID's `merged.json` slice and whose output is an in-repo expected dataclass. Write the snapshot test FIRST against the expected output you derive from a one-time read of the legacy code on the same building; then port until green. |
| E — `roof/` | `test_clustering.py::test_two_opposing_segments_form_one_gable_cluster` (synthetic two segments at az=90°/270°, incl=30°; expect 1 cluster with `MIN_CLUSTER_SIZE=2`) — followed by per-stage tests pinned against the legacy `roof_algorithms_py_results.json` cohort outputs. |
| F — `assemble/` + `classify/` | `test_ceiling_painter.py::test_higher_priority_dominates_lower` (two overlapping candidates, FLAT_EMIT vs RAW_FALLBACK; assert only FLAT_EMIT visible after fold) — plus a property test asserting Σ visible XZ area ≤ Σ candidate XZ area. |
| G — orchestrator | `test_build.py::test_validate_only_passes_on_cohort_uuid` (3 cohort UUIDs; assert `build_tier_payload(uuid).validate()` returns no errors). |
| H — renderer | `test_locator.js::roundtrip` (Node-based JS test harness or `playwright`) — `parseElementUid(makeElementUid(uuid, "wall", 0, 3))` returns the same parts. Visual tests come later in Phase I; locator/parse logic is unit-testable now. |
| I — validation | `test_cohort_metrics.py::test_n_oblique_within_tolerance` (per-uuid integer counts vs `tests/golden/cohort_metrics.json`); plus pixel-diff harness on screenshots. |

### 2.3 Test categories — pick the right tool

| Category | When | Where | Example |
|---|---|---|---|
| **Unit / synthetic** | Pure functions, math, data-structure logic. **Always start here.** | `tests/reconcile_tiers/<module>/test_*.py` | `Plane.fit` on a hand-built 4-corner ring |
| **Property test** | Invariants that must hold across many inputs (winding, area conservation, idempotence). Use `hypothesis`. | Same file, `@given(...)` | "for any valid candidate set, painter output area ≤ candidate union area" |
| **Snapshot / golden** | Behaviour pinned against a known-good output (cohort buildings). | `tests/golden/{tier_payload,cohort_metrics,screenshots}/` | `test_tier_payload_matches_golden(uuid)` reads `tests/golden/tier_payload/<uuid>.json` |
| **Integration** | End-to-end on a small set of cohort UUIDs. | `tests/reconcile_tiers/integration/` | `test_build_one_uuid_produces_valid_payload` |
| **Regression / bug fix** | Someone reports a misrendered building. **Add the failing test first**, *then* fix. | Same module's test file with a clear bug-ID comment | `# regression: <uuid> rendered with no thermal-cap before <commit>` |

### 2.4 Producing golden fixtures the first time

Phase D and beyond pin behaviour against snapshots. To bootstrap a golden:

1. Pick a cohort UUID (from §7 Decision log).
2. Run the **legacy** code path against it (e.g. read `buildings_3d.json[uuid]` for Phase D snapshots; read `roof_algorithms_py_results.json[uuid].roof_surfaces` for Phase E).
3. Extract the *fields the new code is responsible for* into a JSON fixture under `tests/golden/`.
4. Hand-edit the fixture only if the legacy output has a known bug captured in the Pitfalls section (e.g. dropped `thermal-cap` at priority 70 — the new pipeline emits it; the golden should reflect the new behaviour, with a comment pointing at the Decision log entry).
5. Commit the fixture in the same PR as the test that consumes it.

When the new pipeline output diverges from the golden during normal work,
**do not regenerate the golden silently**. The diff is the test result; if
the new behaviour is correct, update the golden in a separate commit with
a one-line rationale and a Decision log entry.

### 2.5 What tests must catch (the pitfalls in test form)

Many entries in §5 (Pitfalls) are exactly the kind of thing tests should
prevent regressing. Add one of these to your phase's test file as
appropriate:

- `test_no_90deg_relaxation`: assert the azimuth tolerance constant is `>= 180` (or that the API uses shortest-arc semantics) — guards saved-memory `feedback_no_90deg_clip`.
- `test_plane_fit_rejects_near_vertical_with_reason`: `Plane.fit(near_vertical_corners)` returns `FitFailure.NEAR_VERTICAL`, not `None` and not a brittle `Plane`.
- `test_no_substring_dispatch_in_payload`: scan the schema's `GapKind` to assert no `Literal` is a free-form string.
- `test_payload_invariants_enforced`: feed a deliberately wrong-winding `HorizontalLid` to the validator and assert `PayloadInvariantError`.
- `test_no_v2_graph_import`: `import reconcile_tiers; assert not any("reconcile_v2" in m for m in reconcile_tiers.__dict__.values()...)` — guards Phase 0 decision.
- `test_renderer_does_not_compute_building_center`: grep test that `tier-preview.js` contains no occurrence of "computeBuildingCenter".

These "negative" tests are cheap and prevent the team accidentally
recreating the smell during a future refactor.

### 2.6 Running tests

- Whole package: `python -m pytest tests/reconcile_tiers/`
- Single module fast loop: `python -m pytest tests/reconcile_tiers/_core/test_plane.py -x -q --tb=short`
- With coverage gate (locally): `python -m pytest tests/reconcile_tiers/ --cov=reconcile_tiers --cov-report=term-missing` — aim for ≥ 90% on `_core/`, `payload/`, `assemble/`, `classify/`. Lower (≥ 70%) is acceptable on `extract/` and `roof/` because some branches require real cohort buildings to exercise; those are covered by integration + golden tests instead.
- JS side (Phase H): use `vitest` or `node --test`; integrate into the same `pytest` run via a Makefile target so CI runs both.
- Pre-commit hook runs `pytest tests/reconcile_tiers/ -q` and the schema-stable test. Don't bypass with `--no-verify`.

### 2.7 When TDD is hard — and what to do

Two situations where strict red-green-refactor breaks down:

1. **Spike / exploration**: you don't know yet what the right API is. Write
   a throwaway script in `reconcile_tiers/scripts/` (or `.context/scripts/`
   if private), explore, *delete or move it*, then go back to TDD. The
   spike never gets merged into the package without tests.
2. **Visual / 3D**: there is no test that says "this looks right". Use a
   pixel-diff golden for the rendered screenshot; for geometry, write
   property tests on numerical invariants (no NaN; planes coplanar within
   tolerance; corners on the expected side of a half-plane). Every
   visually-judged decision becomes a numerical assertion that *would
   fail* if the visual broke.

If you are about to commit code without a paired test and you are not in
either of the two situations above, stop. Write the test.

---

<a id="work-breakdown"></a>
## 3. Work breakdown

Owner = the engineer currently responsible. Update via PR or direct edit.
Status = `pending | in_progress | review | merged | blocked`.
Phases marked **parallelisable** can be claimed by different engineers
simultaneously (no shared files).

| Phase | Scope | Owner | Status | Blocked by | Effort | Files |
|---|---|---|---|---|---|---|
| 0 | Pre-work cohort audits | claude (2026-04-26) | merged | — | ½ day | `.context/scripts/phase0_audits.py` (committed-on-demand) |
| A | Core primitives — `_core/` | codex (2026-04-26) | review | — | 1 day | `reconcile_tiers/_core/{plane,newell,transforms,svd,shapely2,ids,lineage}.py` |
| B | Wire format — `payload/` | codex (2026-04-26) | review | A | 1 day | `reconcile_tiers/payload/{schema,validate,emit_jsonschema}.py` + `tier_payload_schema.json` |
| C **(parallelisable with B after A)** | Ingest — `ingest/` | codex (2026-04-26) | review | A | 1 day | `reconcile_tiers/ingest/{merged,scan_cache,room_transforms}.py` |
| D | Extract layer — `extract/` (V1 equivalent) | codex (2026-04-26) | review | B + C | **5 days, biggest** | `reconcile_tiers/extract/{stories,walls,floors,ceilings,openings,storages,overlaps,height_align,extension,gaps,stitches,exterior,building}.py` |
| E **(parallelisable with D)** | Roof layer — `roof/` (7 stages) | codex (2026-04-26) | review | B + C | 4 days | `reconcile_tiers/roof/{simple_slant,segments,clustering,footprint,planes,clipping,obliques,flats,arrangement,dormers,thermal,roof}.py` |
| F | Assemble + classify — `assemble/` + `classify/` | codex (2026-04-26) | review | D + E | 3 days | `reconcile_tiers/assemble/{ceiling_painter,gaps_to_pieces,walls_to_rooms,building_center}.py` + `reconcile_tiers/classify/{tiers,roof_type}.py` |
| G | Build orchestrator + CLI | codex (2026-04-26) | review | F | 1 day | `reconcile_tiers/{build,cli}.py` |
| H **(parallelisable with G after B)** | Renderer — `web/` | codex (2026-04-26) | review | B | 3 days | `reconcile_tiers/web/{viewer-tiers.html,tier-preview.js,render-tuning.js,locator.js,material-palette.js,geometry.js}` |
| I | Validation cohort + golden snapshots | codex (2026-04-26) | review | G + H | 2 days | `tests/golden/{tier_payload,screenshots}/` |
| J | Migration & deletion (deferred) | codex (2026-04-26) | blocked | I merged + drift alignment | follow-up | `reconcile_tiers/archive/MIGRATION_AUDIT.md` |

Estimated total: ~21 working days with one engineer; ~12 calendar days with
two parallelising on D vs E + G vs H.

The plan file (link above) has the full per-phase deliverable list, threshold
table, and test structure. **Read the plan before claiming a phase.**

---

<a id="critical-references"></a>
## 4. Critical references — where to look for everything

### 4.1 Existing code (to port from)

| Concern | File | Lines | Notes |
|---|---|---|---|
| V1 extraction entry point | `reconcile/extract_3d.py` | `2585–3054` (`extract_building`) | 3300+ LoC; map in walkthrough §4.4 |
| V1 modular submodules | `reconcile/extract3d/` | 11 files | Walkthrough §4.4 + plan Phase D table |
| Story Y-cluster threshold | `reconcile/extract_3d.py` | `2604–2633` | Y gap > 1.0 m → new story |
| Split-level rule | `reconcile/extract3d/builder.py` | `93–119` | Mezzanine OR Δy < 2.0 m |
| Wall reconciliation cascade | `reconcile/extract_3d.py` | `2751–2821` | scan-cache > hybrid > merged > scan-cache-dedup |
| Outward orientation | `reconcile/extract_3d.py` | `_orient_walls_outward` near `2819` | Newell normal vs centroid |
| Flat consensus + slant fallback | `reconcile/extract3d/ceilings.py` | `295–474` | All thresholds in plan §Phase D |
| 3-phase XZ gap detection | `reconcile/extract3d/gaps.py` | `283–547` | walkthrough §4.4.6 |
| Gap walls (snap + lift) | `reconcile/extract3d/gaps.py` | `633–1219` | Heaviest single helper in the codebase |
| Stitch endpoint pairing | `reconcile/extract3d/stitch.py` | `452–729` | walkthrough §4.4.6 |
| Wall extension to slab | `reconcile/extract3d/ceilings.py` | `194–246`, `699–806` | `compute_story_wall_top_cohort` |
| Roof pipeline entry | `reconcile/roof_algorithms_py/pipeline.py` | `1–80` (imports + `run_roof_algorithms` body) | All 34 stages listed in roof-agent map; we keep 7 |
| Oblique clustering | `reconcile/roof_algorithms_py/oblique_clustering.py` | `:7-20` | Constants: 30°/15°/0.5°/min2 |
| Segment collection filter | `reconcile/roof_algorithms_py/segment_collection.py` | `:35,67,95,135` | 5° < incl < 80°, ≥ 0.3 m |
| Dormer detection | `reconcile/roof_algorithms_py/dormer_detection.py` | `:13–18` | All thresholds |
| Dormer geometry | `reconcile/roof_algorithms_py/dormer_geometry.py` | `:124` | `DEPTH_FRACTION = 0.70` |
| Thermal envelope (6 kinds) | `reconcile/roof_algorithms_py/thermal_ceiling.py` | `:111` (`BARRIER_REACH = 0.30`) | We keep 3 (knee + dormer-cheek + dormer-header) + add cap (Phase 0 finding) |
| Roof arrangement (oblique_split) | `reconcile/roof_algorithms_py/roof_arrangement.py` | `:975` (`build_roof_arrangement`) | Cell decomposition; 100% coverage per Phase 0 |
| Tier classifier (current) | `reconcile/complexity_tiers.py` | `25–226` | 8-tier predicate ladder; port to `classify/tiers.py` |
| Server tier slice | `reconcile/viewer_server.py` | `36–665, 5046–5389` | Replace, don't extend |
| Painter's-algorithm replacement target | same | `547, 597, 617, 535, 482, 470, 445` | The 7 helpers that collapse into one fold |
| Renderer (current) | `reconcile/viewer-modules/tier-preview.js` | 749 LoC | Walkthrough §7 |
| Shared geometry JS | `reconcile/viewer-modules/geometry.js` | 423 LoC | Only 4 of 12 exports used by tier preview |
| Tier viewer HTML | `reconcile/viewer-tiers.html` | 475 LoC | Walkthrough §8 |

### 4.2 Repo conventions (do these)

- **Progress tracking** (CLAUDE.md): every change to `reconcile/`, `reconcile_v2/`, or `reconcile_v3/` requires an entry in `tracking_progress.md` with date / what / why / result. **`reconcile_tiers/` is new and follows the same rule.**
- **Element IDs**: `<building_uuid>::<kind>::<id>`. We use `<uuid>::tier-<scope>::<id>` for tier-specific locators. Right-click in the viewer copies the ID; pasting in the search bar jumps back to the element.
- **Coordinate systems**: UTM32N (EPSG:25832) for metric, WGS84 for GPS. Always apply grid convergence. (Tier viewer is ENU-local-relative; coordinate-system pitfalls live in the producers we're porting from.)
- **Geometry primitive**: numpy for vectors, Shapely 2.0 for polygons. **Don't reinvent SVD plane fits or Newell normals — use `_core/plane.py` and `_core/newell.py`.**
- **Tests**: pytest under `tests/reconcile_tiers/`. Snapshot fixtures under `tests/golden/`. Run with `python -m pytest tests/reconcile_tiers/`.

### 4.3 External docs

- [Three.js docs](https://threejs.org/docs/) — before writing custom geometry/material code
- [Shapely docs](https://shapely.readthedocs.io/) — many polygon ops already exist; check before rolling your own
- [numpy docs](https://numpy.org/doc/) — vector/matrix
- [MapLibre docs](https://maplibre.org/maplibre-gl-js/docs/) — only if you touch orthophoto overlays (Phase H optional)
- [Datafordeler API docs](https://datafordeler.dk/) — only if you touch Danish geodata (out of scope for this package)

---

<a id="pitfalls"></a>
## 5. Pitfalls — do not repeat these

These are mistakes already documented in user memories, the architecture review, or as smells in the current code. They will bite again unless you internalise them now.

### 5.1 Geometry / numerics

| ⚠️ Don't | ✅ Do | Why |
|---|---|---|
| Don't relax the **180° azimuth threshold** to 90° in roof clustering or filtering | Keep the 180° (or 360°-aware shortest-arc) check | The 90° threshold caused **false roof clips in production**; a saved-memory rule (`feedback_no_90deg_clip`) |
| Don't reject vertical planes via `abs(b) < 1e-6` | Use `Plane.MIN_NY = sin(5°) ≈ 0.087` and return a typed `FitFailure.NEAR_VERTICAL` | The 1e-6 bound is arithmetically right but classification-dangerous; a tier-renderer-grade ceiling cannot be "5.7e-5° from vertical" (review §4.1) |
| Don't use 3-point cross product for polygon plane normals | Always Newell over all corners | The current `polygonPlaneBasis` (`geometry.js:209`) returns `null` on near-collinear first 3 corners. Newell is robust; the rest of `makePolyGeometry` already uses it |
| Don't write a new `np.linalg.svd` plane fit | Reuse `_core/plane.py::Plane.fit` | Smell #2: same SVD lives in `viewer_server.py:445` and `audit_residual_tier_pieces.py:55`. We're consolidating; don't fork again |
| Don't use `buffer(0)` for Shapely polygon repair | Use `make_valid()` (already in `_core/shapely2.py`) | Shapely 2.0 idiom; `buffer(0)` can collapse to empty MultiPolygon |
| Don't compute the building centroid in JS | Ship it from the producer (`payload.building_center`) | Smell #13. The current renderer averages every wall corner across every room per request |

### 5.2 Wire format / contracts

| ⚠️ Don't | ✅ Do | Why |
|---|---|---|
| Don't dispatch on **substring of `gap.type`** in the renderer | Use the typed `GapKind` enum | Smell #4. `gap.type` is a free string today; the renderer matches `/floor\|ceiling/` and silently routes unknown types to `structureFill` |
| Don't `flattenToMeanY` or `orientHorizontalLidUp` in the renderer | Enforce planar Y and Newell-+Y winding **at the producer** in `assemble/gaps_to_pieces.py` | Review §5 + smell #5. Renderer fixups are contract leaks |
| Don't ship `walls_merged` to the renderer | Remove from payload | Smell #3: shipped today but not rendered (the renderer's only reference is a comment explaining why it's NOT drawn). Dead bandwidth |
| Don't ship `cross_floor_gaps[].type` if the renderer doesn't use it | Ship it OR don't, but pick one | Smell #16: shipped but ignored |
| Don't keep `MATERIALS.ceiling` in the JS | Delete it | Smell #8. Defined but never used (`tier-preview.js:64–68`) |
| Don't accept `Plane | None` returns | Return `Plane | FitFailure` | Review §4.1: callers want a *reason*, not just an absent answer |
| Don't make `cutouts` a free list of N-corner polygons | Enforce 4-corner via the dataclass | The 4-corner gate in `collectWallCutoutHoles` is a tractable contract; relaxing it requires fixing the wall mesh emitter (review §5.4) |
| Don't silently drop openings whose corners aren't 4 | `console.warn` + producer-side assertion | Smell #11 / review §5.4 |

### 5.3 Pipeline architecture

| ⚠️ Don't | ✅ Do | Why |
|---|---|---|
| Don't depend on `reconcile_v2.graph_builder` | Internal heuristics in `extract/gaps.py` and `extract/stitches.py` | Phase 0: V2 sidecar covered 0% of cases that needed it (`oblique_split` is 100%); we removed the V2 graph dependency from the new pipeline |
| Don't replicate the 17 ontology stages of `roof_algorithms_py` | Stop at the moderate 7 stages (see Phase E table) | They feed the **main viewer's** overlays only. The walkthrough §4.5 + roof-agent map confirms the tier viewer reads none of them |
| Don't import from `reconcile/viewer-main.js` or `reconcile/viewer-modules/full-model-ontology.js` | The tier renderer must remain decoupled | Walkthrough §1 #5 — the **decoupling is the reason the tier viewer keeps working while the main viewer churns** |
| Don't run SVD + Shapely on every selection click | Precompute `tier_payload.json` per building (Phase G) | Review §5.2: server-side per-request work was the bulk of the 50–200 ms hitch |
| Don't fold `tier-preview.js` into the main viewer | Keep `reconcile_tiers/web/` self-contained | Review §9. Tempting; risky |

### 5.4 Process / methodology

| ⚠️ Don't | ✅ Do | Why |
|---|---|---|
| Don't tweak a threshold to fix one building | Generalise: measure cohort size first | Saved-memory rule (`feedback_generalize_before_specialize`). Phase 0 audits exist for this reason |
| Don't trust BBR / Datafordeler / GeoDanmark as ground truth | Trust the scan-derived geometry | Saved-memory rule (`feedback_no_public_dk_data_as_truth`). Public DK data can be wrong; useful as discrepancy signal, not prior |
| Don't anchor metrics to extrapolated footprints | Anchor to scan-derived `buildings_3d.json` walls / floors | Saved-memory rule (`feedback_physical_ground_truth_over_extrapolation`). The new pipeline runs the scan all the way through; do not introduce extrapolation as truth |
| Don't deselect or `pytest.mark.skip` a pre-existing failing test | Fix it, or ask before deferring | Saved-memory rule (`feedback_fix_preexisting_test_failures`) |
| Don't suppress a scan signal because synthesised geometry disagrees | Fix the synthesiser to follow the scan | Saved-memory rule (`feedback_synthesis_should_follow_scan`) |
| Don't loosen thresholds when a feature obvious to a human is missing | Enumerate **unused signals** first; prefer adding signals over relaxing gates | Saved-memory rule (`feedback_human_first_multi_signal_roofs`). Applies acutely to the RoofType classifier in `classify/roof_type.py` |
| Don't write code without running tests | Each phase's PR runs `pytest tests/reconcile_tiers/` green before merge | Pre-commit hook will block; don't `--no-verify` |
| Don't `git rm` the existing tier code yet | Phase J only, after Phase I gates pass | Validation before deletion; the existing path is the regression baseline |

### 5.5 The "looks like a smell, but is intentional" list

The architecture review §2 calls these out. Don't fix them:

- **`slanted_pieces[].source` / `arrangement_cell_id` / `roof_hypothesis_id` / `intersection_kind`** are kept on payload pieces even though the current renderer ignores them. They cost two strings per piece and they are the only thing a future locator UI needs. *Keep them in the new payload too* (we ship them as `CeilingPiece.arrangement_cell_id` etc.).
- **The three-pass shading recipe** `mergeGeometries → mergeVertices(weldTol) → toCreasedNormals(20°)`. Don't simplify; don't replace `toCreasedNormals` with `computeVertexNormals`. (Saved-memory rule: `feedback_tier_preview_normals_averaging`.) See §5.3 below.
- **Per-material weld tolerances** `{roof: 0.03, structureFill: 0, default: 0.01}`. Don't unify. The `0` for `structureFill` is what prevents black seams on opposite-winding gap geometry.
- **The 4-corner gate in `collectWallCutoutHoles`**. RoomPlan only emits quads; relaxing requires fixing the wall mesh emitter. Document, don't relax.
- **The `simple_slant` pre-pass** in the roof pipeline. Mono-pitch attic rooms are excluded from clustering on purpose — including them creates spurious mini-clusters.

---

<a id="load-bearing-decisions"></a>
## 6. Load-bearing decisions — do not change without alignment

These are baked into the design. If you find yourself wanting to change one, post in the team channel before editing.

### 6.1 The wire format is the contract

`reconcile_tiers/payload/schema.py` is the producer/consumer boundary. **All
fixups happen on the producer side.** The renderer trusts:
- Polygons are simple (no self-intersection) — Shapely `make_valid` upstream.
- Horizontal lids have Y-spread ≤ 1e-3 and Newell normal +Y.
- Walls have outward-pointing Newell normals.
- Cutouts have exactly 4 corners.
- Plane coefficients have `|b| ≥ 0.087`.

If you find a building that violates an invariant, **fix the producer**, not
the validator. The validator's job is to catch regressions, not to relax.

### 6.2 The painter's algorithm priorities

In `assemble/ceiling_painter.py`:

```
PRIORITY = {
    FLAT_EMIT:        100,
    DORMER_CUTOUT:     90,   # negative — punches hole
    ROOF_ARRANGEMENT:  80,   # oblique_split
    THERMAL_CAP:       70,   # loftrum lid (Phase 0: 47.1% of buildings)
    RAW_FALLBACK:      40,
}
MIN_PIECE_AREA_M2 = 0.05
```

These numbers encode the dedup ordering of the four current ceiling helpers
(review §5.1). Don't rename or renumber without re-running the cohort
regression — the per-source XZ-area conservation test guards them.

### 6.3 The three-pass shading recipe

In `reconcile_tiers/web/tier-preview.js`'s `flushBatches`:

```
mergeGeometries → mergeVertices(weldTol) → toCreasedNormals(20°)
```

Don't skip `toCreasedNormals`. Don't replace with `computeVertexNormals`.
Don't relax the 20° crease angle without paired adjustment of the 18°
`EDGE_THRESHOLD_DEG` for outlines. Saved memory `feedback_tier_preview_normals_averaging`.

### 6.4 Decoupling from the main viewer

`reconcile_tiers/web/` does **not** import from `reconcile/viewer-modules/full-model-ontology.js`, `reconcile/viewer-main.js`, or any `ontology-*` module. The decoupling is the reason this page works while the main viewer keeps churning. Resist any "reuse" that re-couples them.

### 6.5 Static-file serving, not a service

The new pipeline writes `pipeline-outputs/{uuid}/tier_payload.json` and
`pipeline-outputs/tier_index.json`. The viewer fetches them with plain GETs.
**No HTTP endpoints, no FastAPI, no `BaseHTTPRequestHandler`, no caching
layer.** If you find yourself adding any of those, you've gone the wrong
direction.

### 6.6 The 8 tiers stay user-facing; RoofType rides alongside

We're not changing tier labels. `RoofType` is *additive metadata* on the
sidebar pill and in the payload. A hip-roof building that the current code
classifies as tier 8 still classifies as tier 8 — but with `roof_type=HIP`.
Whether tier 8 buckets break out into sub-categories is a future product
decision, not part of this refactor.

---

<a id="decision-log"></a>
## 7. Decision log

### 2026-04-25 — Scope confirmed (user-approved plan)

- Build new pipeline parallel and self-contained from `merged.json`. Defer deletion until post-validation. Main viewer untouched.
- Moderate 7-stage roof pipeline (skip 17 ontology-only stages).
- Plain dataclasses + jsonschema emitter (no Pydantic).

### 2026-04-26 — Phase 0 audit findings

- **Story-index disagreement**: 3.3% (4/123 comparable). Below 5% threshold. **Decision**: use V1 `stories_found` in the new pipeline; document outliers (`287808db`, `7dbc53a6`, `938d6ed6`, `9bc73438`).
- **`oblique_split` coverage**: **100%** (123/123). **Decision**: V2 raw-split sidecar fully retired. New pipeline has no V2 dependency.
- **`thermal-cap` incidence**: **47.1%** (105/223). **Decision**: include `CeilingSource.THERMAL_CAP` at priority 70. ~Half the corpus is visually affected (loftrum fix).
- **Tier 8 distribution**: 9.4% (21/223). Cohort screenshot picks: HIP=`16784bad`, MANSARD=`7153d532`, plus tier-1, tier-5, tier-6/7, loftrum candidates from the 105 thermal-cap buildings.

### 2026-04-26 — Phase E raw ceiling oblique fallback

- **Finding**: the segment/cluster path matched the main attic roof planes on
  the c72 cohort building but missed low-story, low-pitch sloped ceiling
  planes that legacy emits as obliques. Relaxing `MIN_CLUSTER_SIZE=2` would
  violate the clustering contract and admit unsupported single wall edges.
- **Decision**: keep segment clustering strict and add a raw-ceiling fallback
  for lower-story sloped raw ceiling planes (`5° < inclination < 80°`,
  XZ area >= 0.5 m²). These are scan-derived surfaces, not extrapolated
  public-data geometry.
- **Impact**: c72 oblique count now matches legacy (4 new vs 4 legacy) while
  flat-only cohort buildings remain at 0 obliques.

### 2026-04-26 — Raw ceiling fallback tightened to legacy rectangle guards

- **Finding**: full-corpus comparison showed storey count was not the drift
  source: `n_stories` and `split_level` matched legacy on all 223 buildings.
  Remaining over-production came partly from permissive raw oblique promotion
  and dropped sloped-ceiling metadata.
- **Decision**: preserve sloped wall-top ceilings, emit simple-slant oblique
  surfaces for the rooms intentionally excluded from segment clustering, and
  tighten raw-ceiling fallback to the legacy clean-rectangle gates: 4 unique
  XZ corners, area >= 5 m², 10-75° inclination, <=8 cm plane residual, two
  ridge-like edges >=2 m, and duplicate-plane suppression.
- **Impact**: corpus mismatches versus legacy dropped from 47 to 37. Counts
  moved to `{1: 85, 2: 8, 4: 3, 5: 24, 6: 25, 7: 38, 8: 40}` versus legacy
  `{1: 85, 2: 10, 4: 5, 5: 20, 6: 25, 7: 57, 8: 21}`. Tier 7 remains
  under-recovered because the simplified oblique clipping still emits broad
  footprint-sized surfaces instead of legacy's supported roof-part polygons.

### 2026-04-26 — Phase I validation baseline

- **Finding**: the static tier viewer now has committed payload, metric, and
  screenshot goldens for a six-building cohort covering tier 1 flat, tier 5
  shed/slanted, tier 6 gable, HIP, MANSARD, and dormer/loftrum-heavy cases.
- **Decision**: the `±5%` tier-count gate is asserted against the current
  `/tier-index` classifier baseline (legacy `buildings_3d.json` +
  `roof_algorithms_py_results.json`) because the new static `tier_index.json`
  is an artefact manifest, not the legacy bucketed endpoint shape.
- **Impact**: deletion remains deferred to Phase J. During Phase I discovery,
  the generated payload corpus showed larger tier-distribution drift from the
  legacy endpoint than the classifier-only gate. Treat that as a migration
  review risk, not a reason to delete the legacy path.

### 2026-04-26 — Phase J no-delete audit

- **Finding**: Tier 1 deletion candidates still have active consumers. The old
  `reconcile/viewer-tiers.html` path still fetches `/tier-index` and
  `/building-merged`; `reconcile/complexity_tiers.py` remains the legacy
  classifier baseline for tests and scripts; `viewer_server.py` shares roof
  cache loading with non-tier handlers.
- **Decision**: mark Phase J blocked and commit a migration audit instead of
  deleting files. No legacy files are removed in this pass.
- **Impact**: `reconcile_tiers/archive/MIGRATION_AUDIT.md` records the exact blockers,
  generated-payload tier drift, and the next checks needed before a deletion
  PR. Tier 2 deletion remains out of scope.

### 2026-04-26 — Tier 6/7 drift localized to new roof model

- **Finding**: On the 82 legacy tier 6/7 buildings, swapping only the new
  extracted building model into the legacy roof data keeps all 82 in tier 6/7.
  Swapping only the new roof model into the legacy building data drops 80/82
  out of tier 6/7.
- **Decision**: treat the drift as a roof-stage parity issue. The first
  concrete break is between bidirectional segment clustering and plane/surface
  emission: clusters preserve a 180-degree roof axis but lose face direction,
  and emitted oblique surfaces trust cluster inclination/azimuth instead of the
  fitted plane's actual direction.
- **Impact**: deletion remains blocked. `reconcile_tiers/archive/MIGRATION_AUDIT.md`
  now records the stage-swap counts, failure-mode counts, examples, and a
  recommended fix order.

### 2026-04-26 — Directional roof clustering restored

- **Finding**: The legacy path clusters roof segments by full directional
  azimuth and only compares 180-degree opposition during gable pairing and
  clipping. The new path had collapsed opposite faces into bidirectional
  clusters too early.
- **Decision**: refactor `reconcile_tiers.roof.clustering` and
  `reconcile_tiers.roof.planes` to match the legacy semantics: full 0-360
  directional clusters, normal circular mean, and analytic roof planes from
  directional azimuth/inclination rather than SVD fitting over a bidirectional
  cluster.
- **Impact**: legacy tier 6/7 recovery improved from 2/82 to 58/82. Corpus
  tier counts moved from `{1: 93, 2: 8, 4: 2, 5: 27, 6: 2, 7: 4, 8: 87}` to
  `{1: 92, 2: 5, 4: 2, 5: 19, 6: 24, 7: 38, 8: 43}`. Remaining drift is
  concentrated in tier 7 under-recovery and tier 8 over-production.

### 2026-04-26 — Oblique support-domain clipping restored

- **Finding**: The static viewer's apparent "slanted walls pointing the wrong
  way" on `0d3f2993-8386-4130-8f1c-b2938c410828` was caused by roof-arrangement
  ceiling pieces, not room wall winding. The simplified tier roof path clipped
  every analytic oblique to the full building footprint and then split it by
  every room cell; on 45-50° planes this projected distant cells up to ~21 m.
- **Decision**: match the original ceiling clipping intent by constraining
  each plane to the rooms that contributed sloped evidence, clipping ridge and
  slope bounds from those segments/rooms, and capping 3D projection to observed
  room wall/ceiling height plus a 0.5 m eave overhang allowance.
- **Impact**: the screenshot building now emits 5 roof-arrangement pieces with
  Y range `0.942..3.681 m` instead of broad slabs reaching ~21 m. Full corpus
  tier counts moved to `{1: 85, 2: 8, 4: 3, 5: 23, 6: 23, 7: 48, 8: 33}`;
  tier 7 recovery improved materially, though deletion remains blocked until
  the remaining tier 8 over-production is reviewed.

### 2026-04-26 — Deferred migration material archived

- **Finding**: The remaining Phase J work is not active runtime code; it is a
  set of blocked deletion decisions and deferred parity/process items.
- **Decision**: keep runtime code in place, but move Phase J audit material to
  `reconcile_tiers/archive/` and add `DEFERRED_ITEMS.md` as the index for work
  deliberately left outside this migration.
- **Impact**: active package code stays unchanged. Follow-up migration work has
  one archive entry point without making the legacy viewer path look deleted.

### Future decisions

Add new entries here when:
- A threshold is changed from the plan default.
- A scope item is added or removed.
- A phase deliverable changes shape (e.g. a module is split or merged).

Format: `### YYYY-MM-DD — <one-line summary>` followed by what / why / impact.

---

<a id="coordination-protocol"></a>
## 8. Coordination protocol

### 8.1 Claiming a phase

1. Check the [Work breakdown](#work-breakdown) table — phase status must be `pending` and (if `parallelisable`) no overlapping phase has the same files.
2. Verify all `Blocked by` phases are `merged`.
3. Update the table with your name as Owner and status as `in_progress`.
4. Make a feature branch named `tiers/<phase-letter>-<short-slug>` (e.g. `tiers/A-core-primitives`).
5. Read the linked plan section + the relevant source-code references in §4.
6. Read the [Pitfalls](#pitfalls) section.
7. Read §2 (Way of working — TDD) before writing any code. **The first commit on the branch must be a failing test.**

### 8.2 Working on the phase

- **TDD is mandatory.** Loop: red test → green minimum impl → refactor.
  Each unit lands as a failing-test commit followed by a make-it-pass
  commit. PRs whose first commit is implementation without a paired
  failing test will be sent back. See §2 for the detailed workflow.
- Make incremental commits; small commits are reviewable.
- For each non-trivial port, cite `file:line` in the commit message so reviewers can verify against the original (e.g. "Port flat consensus classifier from `reconcile/extract3d/ceilings.py:295–423`").
- Run `python -m pytest tests/reconcile_tiers/` green before pushing. If your phase touches JS, run the JS test suite too.
- New behaviour must arrive with at least one of: unit test, property test, snapshot test, or integration test. Visual-only validation is allowed only on the renderer (Phase H/I) and only for things that genuinely cannot be asserted numerically — and even then, pair it with a pixel-diff golden.
- Don't bypass pre-commit hooks. Fix the underlying issue.
- Don't `git rm` anything outside `reconcile_tiers/`. Phase J only.

### 8.3 Submitting for review

1. Update phase status to `review` in this document.
2. Open a PR with `--base main`.
3. PR title: `[tiers] Phase X: <one-line summary>`.
4. PR body: include the verification checklist for the phase (see §9) with each item ticked. The "tests-first" item is non-negotiable.
5. Add a `tracking_progress.md` entry per CLAUDE.md.

### 8.4 Merging

1. CI green (`pytest tests/reconcile_tiers/` + `python -m reconcile_tiers.build --validate-only` once Phase G lands).
2. Reviewer cohort-checks against the relevant `tests/golden/` snapshots.
3. Squash-merge.
4. Update phase status to `merged` in this document.
5. Notify the next-phase owner if applicable.

### 8.5 Handling drift / mistakes

If you discover the plan is wrong (e.g. a threshold was misread, a current
behaviour doesn't match the walkthrough), **don't silently fix it in code**:

1. Add a **Decision log** entry here in this document explaining what you found and what you'd change.
2. If the plan file (`~/.claude/plans/...`) needs updating, post in the team channel for sign-off — the plan is user-approved.
3. Then proceed with the corrected approach.

This is the same protocol the architecture review used: it verified ~30 specific claims against the code before agreeing with the walkthrough, and called out the discrepancies it found (e.g. the 9-vs-18-stage roof pipeline drift in walkthrough §4.5 vs the skill file).

---

<a id="verification-checklist-per-phase"></a>
## 9. Verification checklist per phase

Each phase's PR must tick the relevant items. Failed checks block the merge.

### Phase A — Core primitives
- [ ] **TDD evidence**: branch's first commit is a failing test in `tests/reconcile_tiers/_core/`; subsequent commits show red→green transitions per module
- [ ] `pytest tests/reconcile_tiers/_core/` green
- [ ] `Plane.fit + Plane.y_at` numerically equivalent to `viewer_server.py:_fit_plane_coeffs + _ring_to_3d_on_plane` on cohort building corners (record max diff in PR body)
- [ ] `_core/newell.py::polygon_area_3d` numerically matches `complexity_tiers.py:_polygon_area_3d` on synthetic + cohort polygons
- [ ] `_core/svd.py::compute_svd` reproduces `extract_3d.py:147–158` residuals on a noise-free synthetic (residual = 0) and a noisy cohort case (residual ≤ original)
- [ ] Property tests via `hypothesis` cover plane-fit invariants (round-trip Y; near-vertical detection)
- [ ] Coverage on `_core/` ≥ 90% (`pytest --cov=reconcile_tiers._core`)
- [ ] No imports from `reconcile/`, `reconcile_v2/`, `reconcile_v3/`, `viewer-main.js`, `ontology-*`

### Phase B — Wire format
- [ ] **TDD evidence**: tests in `tests/reconcile_tiers/payload/` precede each dataclass and validator in commit history
- [ ] `pytest tests/reconcile_tiers/payload/` green
- [ ] `tier_payload_schema.json` regenerates byte-equal (the `test_schema_is_stable` test)
- [ ] Each invariant violation has a dedicated failing fixture (winding, planar Y, plane.b, quad corners, hole containment)
- [ ] Coverage on `payload/` ≥ 90%
- [ ] No imports from `reconcile/`

### Phase C — Ingest
- [x] **TDD evidence**: failing room-count test on 3 cohort UUIDs lands before any loader code
- [x] `pytest tests/reconcile_tiers/ingest/` green
- [x] Runs on 3 cohort UUIDs producing the expected room counts
- [x] Method-rank cascade exercised on a building with both floor-svd and wall-center-svd cases

### Phase D — Extract layer
- [x] **TDD evidence**: per-module snapshot test (deriving expected output from a one-time legacy read) committed BEFORE the corresponding implementation; commit log shows red→green per submodule (stories, walls, floors, ceilings, openings, storages, overlaps, height_align, extension, gaps, stitches, exterior)
- [x] `pytest tests/reconcile_tiers/extract/` green
- [x] Integration test on 3 cohort UUIDs passes
- [x] Per-cohort `BuildingModel` room/wall/gap counts within ±2 of current `buildings_3d.json` per-building entry
- [x] Negative test: `test_no_v2_graph_import` — guards Phase 0 decision
- [x] No `reconcile_v2.graph_builder` import
- [x] Three near-identical opening dedup blocks collapsed to one helper
- [x] Newell-normal computation lives only in `_core/newell.py`

### Phase E — Roof layer
- [ ] **TDD evidence**: per-stage failing tests precede each stage (segments, clustering, footprint, planes, clipping, obliques, flats, arrangement, dormers, thermal); commits show red→green per stage
- [ ] `pytest tests/reconcile_tiers/roof/` green
- [ ] Cohort integration: oblique surface count and dormer cutout count match current `roof_algorithms_py_results.json` within ±1 per cohort building
- [ ] Negative test: `test_no_90deg_relaxation` — asserts azimuth tolerance ≥ 180° / shortest-arc semantics in clustering
- [ ] No graph / hypothesis / evidence / coverage stages added back
- [ ] Thermal emitter produces only `knee + dormer_cheek + dormer_header + thermal_cap` (Phase 0 decision)

### Phase F — Assemble + classify
- [x] **TDD evidence**: painter's-algorithm property tests (area conservation, no overlap, plane–corner consistency) and the 25 ported tier-classifier tests land BEFORE their implementations; RoofType has a hand-built fixture per type with the test before the classifier
- [x] `pytest tests/reconcile_tiers/assemble/` and `tests/reconcile_tiers/classify/` green
- [x] Painter's-algorithm visible-area conservation holds across cohort: per-source XZ area within ±0.5 m² of current `_combined_ceiling_subtraction` output
- [x] Deterministic property-style tests: painter output XZ area ≤ Σ candidate XZ area; no two visible pieces overlap; every visible corner remains on its plane
- [x] 8-tier classifier matches current per cohort building (RoofType is purely additive)
- [x] `total_area` fix in `detect_gable` validated: cohort tier counts unchanged or documented drift
- [x] Coverage on `assemble/` and `classify/` ≥ 90%

### Phase G — Build orchestrator + CLI
- [x] **TDD evidence**: `test_build.py::test_validate_only_passes_on_cohort_uuid` (3 UUIDs) committed before `build.py` body
- [x] `python -m reconcile_tiers.build --all --validate-only` exits 0 across all 223 `pipeline-outputs/` buildings present in this checkout
- [x] `pipeline-outputs/{uuid}/tier_payload.json` validates against `tier_payload_schema.json`
- [x] Mtime gating works (rerun is no-op when inputs unchanged) — covered by a unit test, not just manual
- [x] Multiprocess run is deterministic (rerun produces byte-identical artefacts) — assert via test, not manual

### Phase H — Renderer
- [x] **TDD evidence**: JS-side unit tests (`vitest` or `node --test`) for `locator.js` round-trip, `gapMaterial(kind)` lookup, `material-palette.js` colour table, and `geometry.js` Newell normal land before their implementations. Visual checks come *in addition*, not instead of, unit tests
- [x] `viewer-tiers.html` opens without console errors
- [x] Cohort 6 buildings render
- [x] Right-click on any element copies a `<uuid>::tier-<scope>::<id>` UID
- [x] Pasting that UID into the search bar reselects the same element
- [x] Negative grep test: `MATERIALS.ceiling`, `walls_merged` rendering, `flattenToMeanY`, `orientHorizontalLidUp`, 3-point `polygonPlaneBasis`, `computeBuildingCenter` all absent from `reconcile_tiers/web/` (asserted via a Python test that grep-greps the directory)

### Phase I — Validation cohort
- [x] **TDD evidence**: cohort-metrics tolerance tests (`test_n_oblique_within_tolerance`, `test_n_dormers_within_tolerance`, etc.) and pixel-diff harness land before any tuning. Tolerances are committed numbers, not "feels right"
- [x] All Phase A–H checks above are still green together
- [x] Tier counts vs current `/tier-index` within ±5% — asserted by a test against the current classifier baseline
- [x] 6 golden screenshots committed at 1280×720
- [x] Pixel-diff against goldens passes (5% tolerance) — automated by the Phase I pytest harness
- [x] `tests/golden/cohort_metrics.json` committed; new run within tolerance

### Phase J — Migration & deletion (deferred)
- [x] Tier 1 deletions executed only after Phase I `merged` — blocked; Phase I is in review, not merged, and generated-payload tier drift is unresolved
- [x] `git grep` for each deleted symbol returns zero hits before `git rm` — audited; active consumers remain, so no `git rm` was run
- [x] `tracking_progress.md` updated with the migration entry
- [x] Tier 2 deletions explicitly out of plan scope; require separate alignment
- [x] Blocked/deferred migration material moved under `reconcile_tiers/archive/`

---

<a id="out-of-scope"></a>
## 10. Out of scope

Out of scope means: do not work on these in this package. If a phase tempts you toward one, stop and post in the team channel.

- Rewriting the **main viewer** (`reconcile/viewer.html`, `reconcile/viewer-main.js`).
- Touching **`reconcile_v2/`** internals (we drop the dependency from the tier pipeline; we don't change V2 itself).
- Touching **`reconcile_v3/`** internals.
- **IFC export**, BBR comparison, calor parity checks.
- **Adopting a UI framework** (vanilla JS + importmap stays).
- **Adopting Pydantic / FastAPI / a service layer** (Phase 0 decision: dataclasses + jsonschema, static-file serving).
- **Replicating the 17 ontology stages** of `roof_algorithms_py` (they feed the main viewer's overlays only).
- **Changing the 8 tier labels** (`RoofType` rides alongside).
- **Deleting `reconcile/extract_3d.py`, `reconcile/extract3d/`, or `reconcile/roof_algorithms_py/`** (still feed the main viewer; deletion would be a separate, follow-up effort once the main viewer either migrates or is retired).

---

## Footer

Document owner: tier viewer refactor team.
First written: 2026-04-26.
Last edit: see `git log -- reconcile_tiers/TRACKING.md`.

If you're starting fresh on this codebase, read in this order:
1. `CLAUDE.md` (root)
2. This document — pay special attention to §2 (TDD) and §5 (Pitfalls)
3. The plan: `~/.claude/plans/system-instruction-you-are-working-soft-firefly.md`
4. The architecture review: `.context/plans/viewer-tiers-architecture-review.md`
5. The walkthrough: `.context/attachments/pasted_text_2026-04-25_14-44-10.txt`

Then claim a phase, **write a failing test**, and ship.
