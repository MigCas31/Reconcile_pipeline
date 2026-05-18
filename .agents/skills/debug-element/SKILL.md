---
name: debug-element
description: >
  Use when the user shares a viewer element ID (building_uuid::kind::id) and asks why it
  looks wrong or what caused a geometry issue. Covers legacy kinds (floor, roof-oblique,
  wall-merged, ceiling-flat, …) and ontology-* kinds (ontology-renderable-ceiling,
  ontology-renderable-roof, ontology-knee-wall, ontology-unresolved-coverage,
  ontology-base-*, …), plus static tier viewer tier-* locators from
  pipeline-outputs/<uuid>/tier_payload.json. Resolves the element, traces it to
  the pipeline step and thresholds that produced it, runs existing audits as
  evidence, proposes ranked root causes and candidate fixes — without committing
  changes.
---

# Debug an Element

A building surveyor or engineer saw something off in the viewer, right-clicked to copy the element's shareable locator, and now wants a root-cause analysis. This skill turns that ID into a reproducible triage: **parse → resolve → probe with evals → propose fixes**.

The skill is for **diagnosis**. It may write scratch scripts, reproduction tests, and temporary logging. It must not leave shared pipeline code or heuristics modified at the end of a session, and must not commit.

## Core Rule

When the question is "why is this element classified as X instead of Y?" or "why was it included here but excluded there?", do not anchor on the current heuristic first. Anchor on the human reading first:

- Explain what a human observer is using to make the judgment: support, slope, footprint containment, continuity with neighboring surfaces, story/room context, wall-vs-roof behavior, or other physical cues.
- Treat the current thresholds and branches as suspects, not as ground truth.
- Frame the debugging task as: "what signal is the human using that the pipeline failed to encode, propagated incorrectly, or overrode with a weaker heuristic?"
- Prefer root causes that close that perception gap over local threshold nudges that only make the output look right on one building.

## How to Work on an Element ID

1. **Always run the locator first — don't parse the ID by hand.**

   ```bash
   python -m reconcile.element_locator --element-id "<token>" --trace
   ```

   The output gives: parsed kind, resolved atom / surface record, `provenance_paths` into `roof_algorithms_py_results.json`, `evidence`, the candidate `thresholds`, and the `pipeline_step` that likely minted the atom. Legacy kinds resolve against `reconcile/buildings_3d.json`; ontology kinds resolve against `reconcile/roof_algorithms_py_results.json`; tier kinds resolve against `pipeline-outputs/<uuid>/tier_payload.json`.

2. **Characterize the element geometrically with the probe.** Before asking any questions, dump the stick-out metrics and the neighborhood around the atom — it usually rules out two or three symptoms on its own.

   ```bash
   python -m scripts.probe_element --element-id "<token>" --human
   ```

   Output fields map directly to the Symptom → Suspect table below:

   | Probe field | What it answers |
   |---|---|
   | `stickout.verdict` ∈ {`inside`, `edge_overhang`, `partial_stickout`, `mostly_outside`} | Is this surface extending past the footprint? |
   | `stickout.outside_ratio`, `max_overhang_m` | How bad the stick-out is — quantifies "past the footprint" |
   | `flatness.verdict` ∈ {`consistent_flat`, `consistent_sloped`, `stored_flat_but_tilted`, `stored_sloped_but_horizontal`} | Is the stored role (flat / sloped) consistent with the measured inclination? |
   | `flatness.measured_inclination_deg` vs `flatness.stored_role` | Primary signal for role misclassification — a `stored_*` verdict means the atom's own geometry contradicts its kind label |
   | `flatness.overhead_roof_surfaces[0]` | The roof surface directly covering this element in 2D — cross-check for region-wide role (useful when `flatness.verdict` is a mismatch) |
   | `level.verdict` ∈ {`consistent_level`, `borderline_level`, `above_room_ceiling_cohort`, `below_room_ceiling_cohort`, `no_siblings`} | Is the element at the right y vs its sibling ceilings/floors in the same (room, story)? |
   | `level.delta_from_sibling_mean_m` | Signed y-offset from the sibling cohort's mean; large values (`> 0.3` m) suggest a wrong-story or wrong-source-geometry assignment |
   | `neighbors[].poly_min_gap_m == 0` with non-zero `overlap_area_m2` | Overlaps / duplicates another atom |
   | `neighbors[].vertical_delta_m` close to 0, `same_room=true` | Sibling atoms on the same ceiling plane — candidate sliver / duplicate partners |
   | `neighbors[].vertical_delta_m` large, `same_hypothesis=true` | Atoms from the same roof hypothesis — check if hypothesis split was correct |
   | `neighbors` count == 0 within 3 m | Atom is isolated — investigate why the coverage graph put it alone |

   Default radius is 3 m; raise it with `--radius` for attic-wide probes.

3. **Confirm the building payload exists on disk**: `ls pipeline-outputs/<uuid>/`. If not, stop and ask the user to rerun extraction (`python reconcile/extract_3d.py <uuid>`) — you can't debug a rendering issue without the underlying data.

4. **Ask the user what looks wrong before forming a hypothesis.** Describe the symptom in human/building terms first, not pipeline terms. Never guess from the ID alone. Good defaults:
   - Wrong shape / missing corners
   - Surface extends past the building footprint
   - Missing (hole where this element should be)
   - Overlapping or duplicated with another element
   - Wrong orientation (azimuth / inclination)
   - Positioned wrongly (too high / too low / wrong story)

   Before you inspect thresholds, write down one sentence answering: "Why would a competent surveyor instantly call this X and not Y?" If you cannot answer that clearly, you are not ready to debug the classification.

   If the user reports a **missing** element (there's no ID to paste because the element isn't there), ask them for an approximate 3D point from the viewer and probe *that*:

   ```bash
   python -m scripts.probe_element --uuid <uuid> --point "x,y,z" --radius 2.0 --human
   ```

   `containing` lists atoms whose 2D footprint covers the point (stacked vertically); `nearby` lists atoms within the radius. `likely_hole=true` means the pipeline produced nothing within the radius — a genuine coverage gap, not an element with wrong geometry.

5. **Cross-reference `roof_algorithms_py_results.json`**: that file is the last known good baseline per UUID. `git log -p -- reconcile/roof_algorithms_py_results.json` scoped to the UUID tells you whether this element's hash has changed recently. A hash change with no user-visible input change implies an upstream geometry shift — investigate inputs, not the hasher.

6. **Generalize before specializing — the single most important step.** Before drilling into this building, prove whether the symptom is *general* (affects many buildings, pointing at a heuristic or algorithm bug) or *specific* (unique to this building's input geometry). A fix that only repairs one building is almost always wrong — either it overfits to a single edge case, or the "real" bug is masked elsewhere.

   For the symptom the user reported, scan across **all** UUIDs in `roof_algorithms_py_results.json`:

   ```bash
   # Example: count buildings with oblique ceiling partitions whose area is
   # disproportionate vs room footprint (tune predicate to the symptom).
   python3 -c "
   import json
   results = json.load(open('reconcile/roof_algorithms_py_results.json'))
   hits = []
   for uuid, b in results.items():
       for p in b.get('ceiling_partitions', {}).get('oblique', []):
           if <predicate that matches the symptom>:
               hits.append((uuid, p['id'], p['area_m2']))
   print(f'{len(hits)}/{len(results)} buildings affected')
   for h in hits[:10]: print(h)
   "
   ```

   Then decide:

   | Finding | What it means | Where to look next |
   |---|---|---|
   | Symptom hits 1 building only | Input-specific — scan data anomaly, unusual geometry, edge case in this file | `pipeline-outputs/<uuid>/merged.json`, `reconciled.json`, the scans themselves |
   | Symptom hits a small cluster (2–10) | Probably a narrow input pattern the heuristic mis-handles | Compare the affected buildings' inputs — what do they share? |
   | Symptom hits many buildings (>10%) | Heuristic or algorithm is wrong — fix belongs in `reconcile/roof_algorithms_py/`, not in this building's data | Walk the pipeline step from `pipeline_step.file` |

   **If the count is 1, pause and ask the user**: is this a known edge case, or should the heuristic handle it? Do not propose a threshold tweak until you know the blast radius.

   **After counting, inspect 3–5 representative cases from the cohort** — don't just count and declare a root cause. A cohort of 60 buildings might contain two distinct sub-patterns that share the same surface-level symptom but need different fixes. For each sampled case, check:
   - Does the same upstream field / condition trigger the symptom? (e.g., same `flat_role`, same condition branch)
   - Are the input geometries structurally similar (room shape, hypothesis type, coverage state)?
   - Would the proposed root cause explain *all* sampled cases, or only a subset?

   If sampled cases diverge — same predicate match, different underlying cause — split them into sub-cohorts and treat each separately. A fix aimed at the blended cohort will be whack-a-mole.

   Useful cross-building probes to keep in mind:

   - **Atom structure match**: grep the results JSON for atoms with the same `role` + `semantic_kinds` + area/perimeter bracket.
   - **Pipeline-step match**: scan `pipeline_steps` arrays and `run_meta.json` statuses across `pipeline-outputs/`.
   - **Audit re-run with a filter**: the existing `audit_*.py` scripts iterate all buildings — read their output and filter to rows matching the symptom.
   - **Hash-change cohort**: `git log -p -- reconcile/roof_algorithms_py_results.json` at the last commit that touched it, then search for other buildings whose hashes changed in the same commit.

7. **Pick the audit that matches the symptom**:

   | Symptom | Audit |
   |---|---|
   | Surface outside footprint | `python scripts/audit_surfaces_outside_footprint.py` (writes `/tmp/surfaces_outside_footprint.json`) |
   | Missing / wrong ceiling coverage | `python scripts/audit_ceiling_parity_deficits.py` → `.context/ceiling_parity_deficit_audit.{json,md}` |
   | Category / role counts off | `python scripts/audit_full_model_payloads.py` → `.context/full_model_full_building_audit.{json,md}` |
   | Pipeline crashed / partial | `python scripts/audit_building_runtime.py` → `.context/building-runtime-audit.json` |
   | Candidate-atom blast radius | `python scripts/scan_candidate_atom_blast_radius.py` (needs viewer on `:8090`) |

   Run only the audit(s) that match the symptom. None of them need to be fast in full — most accept a UUID filter or can be run over a single building directory.

8. **Re-run the producing pipeline step with verbose logging if the audit isn't enough.** Add temporary `print` / `logging.debug` inside the step indicated by `pipeline_step.file` from the locator. Re-extract the one building (`python reconcile/extract_3d.py <uuid>`) and read the output. **Revert the temporary logging before handing back.**

9. **Simulate every candidate fix against the full dataset before proposing it.** For each candidate fix, write a scratch script that reads `roof_algorithms_py_results.json`, applies the proposed logic in-memory, and counts four numbers:

   ```
   repaired:   cohort cases where the symptom is gone under the fix
   unchanged:  cohort cases where the symptom persists (fix didn't help)
   regressed:  non-cohort buildings where the fix introduces the symptom
   neutral:    non-cohort buildings unaffected
   ```

   Minimal template:

   ```python
   import json

   results = json.load(open("reconcile/roof_algorithms_py_results.json"))


   def has_symptom(b):  # current behaviour — predicate from step 6
       ...


   def has_symptom_fixed(b):  # same predicate, but with the proposed logic applied
       ...


   repaired = regressed = 0
   for uuid, b in results.items():
       before = has_symptom(b)
       after = has_symptom_fixed(b)
       if before and not after:
           repaired += 1
       if not before and after:
           regressed += 1

   print(f"repaired={repaired}, regressed={regressed}")
   ```

   A fix is credible only when `repaired` is large relative to the cohort and `regressed` is zero or negligible. If `regressed > 0`, inspect those buildings manually — understand *why* the fix breaks them before deciding whether the tradeoff is acceptable. Report all four numbers in the final answer. A fix that repairs the reported building but regresses others is whack-a-mole: down-rank it and explain what the regression cases have in common.

10. **Research before inventing.** When the algorithm has a name (Sutherland-Hodgman clip, plane-plane intersection, convex-hull merge, Shapely union), check online / library docs before proposing custom geometry. Check `../calor` (Go backend) and `.context/web-main-latest/` (TS frontend) for parallel implementations that must stay in lock-step.

11. **Write up the final answer as**: `symptom → cohort size → root cause → evidence → ranked candidate fixes (with cohort blast radius)`. End with a workspace-state footer. Do **not** implement the fix.

## Anatomy of an Element ID

```
<building_uuid>::<kind>::<id>
```

### Legacy kinds (14 supported)

Live in `buildings_3d.json`. Handled by `find_element` in `reconcile/element_locator.py`. Kinds: `wall-merged`, `wall-computed`, `wall-extension`, `wall-clipped-original`, `door`, `window`, `floor`, `floor-overlap`, `gap-cross-story`, `gap-within-story`, `wall-stitch`, `gap-wall`, `exterior-gap-element`, `exterior-gap-wall`, `gap-closure`, `roof-oblique`, `roof-flat`, `ceiling-flat`, `ceiling-simple-slant`, `dormer-cheek`, `dormer-header`. See `CLAUDE.md` for the per-kind ID format.

### Ontology kinds

Emitted by the full-model viewer in `reconcile/viewer-modules/full-model-ontology.js:117-134`. The `id` portion has the shape `renderable:<surface_category>:<source_id>` where `source_id` is one of:

- A semantic atom id (`ceiling-partition:<hash>`, `knee-wall:<hash>`, `implicit-flat-atom:<hash>`).
- A roof cell/face composite: `roof-cell:<kind>:<hash>` (bare cell) or `roof-cell:<kind>:<hash>:arr-face:<face_hash>` (face within cell) — resolved against `roof_cell_complex.cells[*]` and its `.faces[*]`. Locator exposes the `parent_cell` dict so probes inherit `room_index` / `story` / `roof_hypothesis_id`.
- An occupied-cell/face composite (`occupied-cell:<hash>:face:<id>`) assembled in `reconcile/viewer_server.py` — may only exist in the live `/ontology-artifacts?view=full-model` payload.

| Kind | Surface category (frontend) | Resolves against |
|---|---|---|
| `ontology-renderable-ceiling` | `occupied_room_ceiling` / `room_ceiling_sloped` / `room_ceiling_flat` | `ceiling_partitions.{oblique,flat,room_partitions}` |
| `ontology-renderable-roof` | `exterior_roof` | `roof_surfaces.{oblique,flat}` + `roof_hypothesis_graph`; roof cell/face composites via `roof_cell_complex.cells[*].faces[*]` |
| `ontology-renderable-wall` | `exterior_wall` | occupied/top-boundary cell complex |
| `ontology-renderable-room-wall` | `occupied_room_wall` | occupied cell complex |
| `ontology-renderable-floor` | `occupied_room_floor` | occupied cell complex |
| `ontology-base-exterior-wall` | `base_exterior_wall` | viewer-side projection of V1 room data |
| `ontology-base-interior-wall` | `base_interior_wall` | viewer-side projection of V1 room data |
| `ontology-base-floor` / `-ceiling` / `-window` / `-door` / `-opening` | `base_*` | V1 room data |
| `ontology-knee-wall` | `knee_wall` | `knee_walls[*]` in roof results |
| `ontology-unresolved-coverage` | `unresolved_region` | `roof_coverage_graph` unresolved regions |
| `ontology-fallback-ceiling` | `fallback_room_ceiling` | `ceiling.simple_slant` / `ceiling.room_partitions` |

The 20-char hash (e.g. `b1cdb83686f103bb1e26`) is produced by `_stable_hash(...)` in `reconcile/roof_algorithms_py/graph_utils.py` — deterministic from the atom inputs (room id, corners, owner). **A hash change means inputs changed**, not randomness.

### Static tier viewer kinds

Emitted by `reconcile_tiers/web/viewer-tiers.html` and resolved against `pipeline-outputs/<uuid>/tier_payload.json`. Kinds include `tier-room`, `tier-wall`, `tier-gap`, `tier-ceiling-*`, and `tier-knee-wall`; wall extensions, doors, and windows are suffixes on the room/wall locator IDs. The probe command loads this payload through `--pipeline-dir` (default `pipeline-outputs`), so copied IDs from the static tier viewer can be used directly:

```bash
python -m reconcile.element_locator --element-id "<uuid>::tier-knee-wall::<id>" --trace
python -m scripts.probe_element --element-id "<uuid>::tier-knee-wall::<id>" --human
```

## Symptom → Suspect Map

| Symptom | First suspects | Where to look |
|---|---|---|
| Surface extends past building footprint | Clipping in `roof_partitioning.py`; footprint in `reconcile/roof_algorithms_py/footprint_derivation.py` | `audit_surfaces_outside_footprint.py` + `ROOM_TOP_MIN_CLEARANCE_M`, `ROOM_TOP_SHELL_TOL_M` |
| Missing ceiling under a room | `thermal_ceiling.py` (`THRESHOLD_M=0.30`), `derive_room_ceiling_partitions` | `audit_ceiling_parity_deficits.py` |
| Ceiling sliver / duplicate partition | Stable-hash inputs in `roof_partitioning.py:190,532,690`; clustering in `oblique_clustering.py` | Diff `roof_algorithms_py_results.json` vs git history for the UUID |
| Wrong azimuth / flipped slope | Segment orientation in `top_boundary_graph.py`; UTM→local grid convergence | See gotcha below — azimuth filter is **180°** on purpose |
| Unresolved region blob | `roof_coverage_graph.py` (`SEED_ROOM_BUFFER_M=0.75`) | `audit_full_model_payloads.py` unresolved counts |
| Knee wall in the wrong place | `thermal_ceiling.py` knee-wall detection | `audit_ceiling_parity_deficits.py`, viewer inspection |
| Roof hypothesis selected wrong | `roof_hypothesis_graph.selected_hypothesis_ids` | `roof_evidence_graph.atom_evidence[<atom>]` (confidence + support) |
| Sloped ceiling where room is flat (or vice versa) | Role assignment in `roof_partitioning.py` (sloped vs flat split); clustering inclination filter in `oblique_clustering.py` | `probe_element.py` `flatness.verdict` (`stored_flat_but_tilted` / `stored_sloped_but_horizontal`); cohort-scan `roof_algorithms_py_results.json` for partitions whose measured inclination disagrees with their `kind` |
| Floor/ceiling at the wrong y vs rest of the room | Story assignment in `story_index.py`; corner inheritance in `roof_partitioning.py`; room index lookup in `ceiling_partitions.*` | `probe_element.py` `level.verdict` (`above_room_ceiling_cohort` / `below_room_ceiling_cohort`); compare `delta_from_sibling_mean_m` to `ROOM_LEVEL_TOLERANCE_M=0.3` |

## Heuristic Lookup (authoritative)

| File:Line | Constant | Value | Role |
|---|---|---|---|
| `reconcile/roof_algorithms_py/oblique_clustering.py:19-20` | `COPLANAR_TOL` | 0.5 m | Plane-normal offset tolerance |
| `reconcile/roof_algorithms_py/oblique_clustering.py:19-20` | Δazimuth | 30° | Max azimuth diff in cluster |
| `reconcile/roof_algorithms_py/oblique_clustering.py:19-20` | Δinclination | 15° | Max inclination diff in cluster |
| `reconcile/roof_algorithms_py/oblique_clustering.py:8` | `MIN_CLUSTER_SIZE` | 2 | Min segments per roof cluster |
| `reconcile/roof_algorithms_py/roof_partitioning.py:22-23` | `ROOM_TOP_MIN_CLEARANCE_M` | 0.15 m | Room/shell clearance |
| `reconcile/roof_algorithms_py/roof_partitioning.py:22-23` | `ROOM_TOP_SHELL_TOL_M` | 0.08 m | Shell alignment tolerance |
| `reconcile/roof_algorithms_py/roof_partitioning.py:20` | `AREA_EPS` | 0.01 m² | Min polygon area |
| `reconcile/roof_algorithms_py/roof_coverage_graph.py:15` | `SEED_ROOM_BUFFER_M` | 0.75 m | Hypothesis-seed room buffer |
| `reconcile/roof_algorithms_py/thermal_ceiling.py:551` | `THRESHOLD_M` | 0.30 m | Knee-wall gap |
| `reconcile/roof_algorithms_py/simple_slant.py:21` | `_AZIMUTH_MULTI_THRESHOLD` | 90° | Multi-slant fallback |
| `reconcile/roof_algorithms_py/roof_envelope_continuation.py:17` | `MAX_CONTINUATION_DISTANCE_M` | 4.0 m | Max continuation across gaps |

Verify with `grep -n` before quoting — constants occasionally move.

## Evals (Existing)

| Script | Output | What it checks |
|---|---|---|
| `scripts/audit_building_runtime.py` | `.context/building-runtime-audit.json` | Pipeline status, timings, failure modes |
| `scripts/audit_ceiling_parity_deficits.py` | `.context/ceiling_parity_deficit_audit.{json,md}` | Ceiling semantic vs heuristic coverage |
| `scripts/audit_full_model_payloads.py` | `.context/full_model_full_building_audit.{json,md}` | Surface category + area counts |
| `scripts/audit_surfaces_outside_footprint.py` | `/tmp/surfaces_outside_footprint.json` | Surfaces >1 m² or >10% outside footprint |
| `scripts/scan_candidate_atom_blast_radius.py` | stdout + JSON | Impact of demoting candidate atoms (viewer on `:8090`) |
| `scripts/build_roof_algorithms_py_results.py` | `reconcile/roof_algorithms_py_results.json` (451 MB) | Regenerate the baseline — **slow, only on explicit user request** |

## Autonomy — What This Skill May And May Not Do

| | Allowed | Forbidden |
|---|---|---|
| Read | any file, any pipeline output, any log | — |
| Write new files | scratch scripts under `scripts/`, scratch JSON under `.context/`, new audit variants, new unit tests that reproduce the bug | overwriting files in `pipeline-outputs/` |
| Edit existing code | temporary `print` / `logging.debug` inside pipeline steps, temporary threshold tweaks for bisection | leaving any edit in `reconcile/`, `reconcile_v2/`, `tests/` at session end — revert before handing back |
| Run | any audit, any test, any extraction/reconciliation over the affected UUID, any `grep`/`git log` | `git commit`, `git push`, opening PRs, regenerating `roof_algorithms_py_results.json` without explicit approval |
| Research | web searches on algorithm names, reading `../calor` and `.context/web-main-latest/` | — |
| Propose fixes | yes — include diffs, explain blast radius, rank by confidence | implementing the fix |

**Workspace-state footer** — every final message MUST end with:

```
Workspace state:
  files added:      <list, or "(none)">
  files reverted:   <list of files touched during diagnosis and restored>
  files still dirty: <list — should be empty; if not, explain why>
```

`git status` must agree with the footer. If a file is dirty you didn't intend, revert it with `git checkout -- <path>` before reporting.

## Gotchas

- **Azimuth filter is 180° by design.** Do not propose relaxing it to 90° without an incident-grade reason; a prior 90° relaxation caused false clips in production (see `CLAUDE.md` gotcha).
- **Ontology kinds need `roof_algorithms_py_results.json`**; legacy kinds read `buildings_3d.json`. The locator routes automatically — don't manually load the wrong file.
- **Stable hashes are not random.** A hash change between runs always means an input changed. Bisect inputs (room corners, atom owner, story index) — don't touch `_stable_hash`.
- **Roof pipeline steps must run in order.** If you re-run a single step with injected logging, re-run the full pipeline afterwards to confirm you haven't cached half-stale state in `roof_algorithms_py_results.json`.
- **Roof cell/face composites (`roof-cell:<kind>:<hash>[:arr-face:<face_hash>]`) resolve directly** via the locator against `roof_cell_complex.cells[*]` — no viewer endpoint needed. `probe_element.py` reads `parent_cell` so `room_index` / `story` / `roof_hypothesis_id` are populated even when the atom is a face (which lacks them). **Occupied-cell composites (`occupied-cell:<hash>:face:<id>`)** are still viewer-assembled in `reconcile/viewer_server.py` and may only exist in the live payload — for those, hit `curl 'http://127.0.0.1:8080/ontology-artifacts?uuid=<uuid>&view=full-model' > /tmp/full_model.json` and grep that.
- **451 MB results file.** Never `cat` or print it whole. Use `python3 -c "import json; b=json.load(open(...))['<uuid>']; ..."` to scope to one UUID.

## Output Contract

The skill's final message to the user MUST contain, in order:

1. **Parsed ID** — uuid, kind, source_id.
2. **Resolved atom** — atom kind, role, room, story, hypothesis id, evidence summary.
3. **User symptom** — as captured via `AskUserQuestion`.
4. **Cohort size** — how many other buildings show the same symptom, out of how many total. Include the probe predicate used and a few example UUIDs. If 1-of-N, call that out explicitly and note the overfit risk.
5. **Root-cause hypotheses (ranked)** — for each: claim, supporting evidence (file:line, audit line, numeric comparison), disconfirming evidence if any, and whether the hypothesis explains the whole cohort or only this building.
6. **Candidate fixes** — proposals only, each with: what file/constant to change, one-line rationale, and **simulation results** (repaired / unchanged / regressed counts from the step-9 script). If simulation was not run, state why and flag the fix as unverified. A fix without numbers is a guess. Include a note if the fix is building-specific rather than heuristic-level.
7. **Workspace-state footer** (see Autonomy section above).

## Module Reference

| File | Role |
|---|---|
| `reconcile/element_locator.py` | Parses IDs, resolves legacy + ontology kinds, emits trace metadata |
| `scripts/probe_element.py` | Stick-out / neighborhood / missing-at-point probes — run immediately after the locator |
| `reconcile/viewer-modules/full-model-ontology.js` | Canonical frontend kind↔category map |
| `reconcile/viewer_server.py` | Builds `renderable_surfaces` from atoms at HTTP time |
| `reconcile/roof_algorithms_py/roof_partitioning.py` | Produces `ceiling-partition:*` atoms |
| `reconcile/roof_algorithms_py/thermal_ceiling.py` | Produces `knee-wall:*` atoms; THRESHOLD_M |
| `reconcile/roof_algorithms_py/oblique_clustering.py` | Oblique segment clustering thresholds |
| `reconcile/roof_algorithms_py/roof_coverage_graph.py` | Unresolved-region seeding |
| `reconcile/roof_algorithms_py/graph_utils.py` | `_stable_hash` |
| `reconcile/roof_algorithms_py_results.json` | Baseline per UUID — always loaded for ontology kinds |
| `scripts/audit_*.py` | Existing evals |
