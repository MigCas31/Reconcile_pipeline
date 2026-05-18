---
name: inspect-building
description: >
  Run this FIRST whenever the user reports an issue with a specific building
  or element — anything looks wrong, missing, broken, weird, off, suspicious,
  not what they expect, "why does this..." — so you have screenshots, realism
  flags, and a defect audit in front of you before forming hypotheses or
  asking clarifying questions. Also use when the user explicitly asks for a
  "full picture" of a UUID, or when they share an element locator and you
  want the surrounding building context. Inputs: a building UUID
  (016980bc-…) or an element locator (<uuid>::<kind>::<id>). If the user
  describes an issue without a UUID, ask for one before guessing. Output is
  a single-folder report (.context/building-reports/<uuid>/<timestamp>/).
  Use `debug-element` afterwards for single-element root cause analysis.
---

# Inspect a Building

The user reports a problem ("this building looks weird", "the roof is
missing", "why is there a gap here", "this came out broken"), shares a
UUID or element locator, or asks for the full picture. This skill turns
that into one command and one folder so you can respond with grounded
visual + structural context instead of guessing.

**Use this skill BEFORE forming any hypothesis or asking clarifying
questions about a specific building or element.** The cost is ~30 seconds
of tool runtime; the benefit is screenshots showing what the user is
looking at, plus realism flags showing whether the LoD2 reconstruction is
broken (often the real story behind a "weird" report) and a defect audit
listing every issue with a paste-into-viewer locator. Skipping this step
and asking "what looks wrong?" puts the user in the position of describing
geometry to a blind agent.

If the user reports an issue without a UUID, ask for one — don't try to
debug from a screenshot or description alone. The skill needs a UUID
(or element locator, from which it parses the UUID) to do anything useful.

If they have a specific element they suspect is wrong and want root cause,
run this skill first for context, **then** hand off to `debug-element`
with the locator id.

## The one command

```bash
reconcile/inspect_building.sh <uuid>
reconcile/inspect_building.sh '<uuid>::<kind>::<id>'
```

Output goes to `.context/building-reports/<uuid>/<timestamp>/`. The script
prints the absolute path to `report.md` at the end — open that to see
everything.

If `lod2/reconstruction.obj` or `tier_payload.json` is missing, the script
skips the dependent step with a one-line note instead of failing. To enable
the third-party validator: `brew tap tudelft3d/software && brew install
val3dity` (optional — adds an ISO 19107 pass on the LoD2 mesh).

## What the report contains

`report.md` sections, in order:

1. **Tier classification** — tier, label, roof type, story/room counts, knee
   walls, gap count. Cross-reference these against the realism flags.
2. **Element trace** — only when an element id was supplied. Same JSON shape
   as `python -m reconcile.element_locator --element-id "…"`.
3. **Realism (LoD2 sanity)** — pass/warn/fail for 8 questions about the
   `lod2/reconstruction.obj` mesh. See "Reading the realism section" below.
4. **Defects (tier_payload audit)** — what's wrong in the data the tiers
   viewer renders. Each defect carries the locator id — paste it into the
   viewer's search box.
5. **val3dity validation** — present if val3dity is installed, otherwise a
   one-line install hint.
6. **Screenshots** — four angles of the tiers viewer (`iso`, `overhead`,
   `south`, `east`) plus an `element` shot when a locator was given.
7. **Pipeline outputs** — file listing with sizes, so you know what's there
   and what's missing.

Sidecars in the same folder: `report.json` (machine-readable equivalent),
`metrics.json` (full realism flags + per-hole detail), `audit.json` (full
defect listing), `element.json` (when applicable), `val3dity.json` (when
the binary is installed), `viewer.log`, `artifacts/` (symlinks to the
canonical pipeline outputs).

## Reading the realism section

The realism section asks "is this LoD2 a plausible building?" Each flag
fires `pass` / `warn` / `fail`. The flag list at the top of the section is
the failures only — that's where to start.

| Flag | What a failure means | First thing to check |
|---|---|---|
| `not_watertight` | The exterior shell is open. The "Watertight holes" subtable lists every boundary loop with its location label (`roof`, `ground`, `wall-east/west/north/south`, `wall-interior`) and centroid. | The hole's centroid coordinates point you at the missing patch — open the LoD2 OBJ in MeshLab/Blender at those coords. Often a missing roof face. |
| `multiple_components` | The mesh has disconnected pieces; expected one. | The LoD2 service split the building (e.g. an outbuilding got included). Compare to the merged scan's coverage. |
| `implausible_volume` | Volume outside `[30, 5000]` m³. | If absurdly small, the LoD2 service produced almost nothing. If huge, look for an exploded coordinate transform. |
| `extreme_convexity` | Volume / convex-hull-volume outside `[0.35, 0.99]`. >0.99 = just a box. <0.35 = bizarre carved shape. | Compare against the iso screenshot — does the shape look like a real house? |
| `implausible_aspect` | `height / max(width, depth) > 1.5`. | A flagpole shape — usually the height bound is wrong (mesh extends way above the building). |
| `no_walls` | <20% of face area is wall-like. | The LoD2 service didn't produce vertical walls — it's just a roof slab + ground. |
| `story_count_mismatch` | `round(height / 3 m)` differs from `classification.n_stories` by more than 1. | Either height is wrong (LoD2 too tall/short) or the classifier mislabeled the building. Look at the `extents.y` value vs the merged scan height. |
| `footprint_mismatch` | LoD2 ground-face area differs from sum of ground-story room floors by >30%. | These should describe the same patch of ground. A 5× mismatch usually means the LoD2 reconstruction is for a different building or geometry got translated. |

When `not_watertight` fires, the per-hole table has the centroid and a
location label — that's what "where" means for `is_watertight: false`. Use
the centroid to localize in the source mesh.

## Reading the defects section

The defect audit walks `tier_payload.json` — the same data the tiers
viewer reads in `populateBuildingScene` (`reconcile_tiers/web/tier-preview.js`).
The flags map to user-visible failure modes:

| Flag | What a failure means |
|---|---|
| `ceiling_coverage_gaps` | Per-room: union of overhead ceiling polygons covers <50% of the room's floor footprint (XZ). The user sees through the roof above this room. |
| `ceiling_orientation_wrong` | A `ceiling[]` entry's Newell normal has Y < 0.10 — it's near-vertical or pointing down. Should be near-up for ceilings/roofs. |
| `out_of_envelope` | Walls / ceilings / gaps / knee walls extending past the convex hull of ground-story floors (buffered 0.5 m). Listed only if outside-area > 0.25 m² AND outside-ratio > 5%. |
| `silent_drops` | Polygons present in JSON that the renderer would skip — `corners < 3` or area `< RENDER_TUNING.minPolygonAreaM2 = 1e-4`, or doors/windows below `opening.minDim = 0.05`. They exist in the data but the user never sees them in the viewer. |
| `knee_walls_misplaced` | Knee walls more than 1 m from the top story's `Y_max` — they should sit at the perimeter of the top story near the roof, not lower down. |

The story coverage census table (rooms vs ceilings per story) is included
even when nothing flagged — useful sanity check. Same for the gap census
(grouped by `kind × scope`).

Every defect entry includes the original `locator_id`. Paste it into the
tiers viewer search bar to jump to the geometry. From there you can decide
whether to spawn `debug-element` for a root-cause analysis.

## What this skill does NOT do

- It does not diagnose a single element to root cause. For that, hand the
  locator id to `debug-element`.
- It does not modify the pipeline or commit any changes — it's a
  read-only triage.
- It does not run the upstream `audit_*.py` scripts (`audit_ceiling_parity_deficits.py`,
  `audit_surfaces_outside_footprint.py`, etc.). Those operate on different
  inputs (`buildings_3d.json`, `roof_algorithms_py_results.json`) and
  produce per-cohort summaries; this skill is per-building. If the
  inspection report flags many buildings with the same defect, run those
  audits next.
- It does not compute CityJSON / 3d-building-metrics. We evaluated and
  decided against — `tudelft3d/3d-building-metrics` is unmaintained
  (last substantive commit 2024-03), no off-the-shelf OBJ→CityJSON
  converter exists, and the building-relevant metrics it produces are
  already covered by the trimesh realism flags here.

## Implementation map

| File | Role |
|---|---|
| `reconcile/inspect_building.sh` | Bash orchestrator — argument parsing, server lifecycle, calls into the helpers |
| `reconcile/inspect_building/metrics.py` | LoD2 realism check via trimesh; per-hole boundary-loop reporting |
| `reconcile/inspect_building/audit.py` | tier_payload defect audit (7 checks); mirrors the renderer's silent-drop thresholds |
| `reconcile/inspect_building/screenshot.py` | Headless tiers-viewer captures via Playwright; uses `window.__tierViewer` exposed by `reconcile_tiers/web/viewer-tiers-main.js` |
| `reconcile/inspect_building/summarize.py` | Combines everything into `report.md` + `report.json` |
| `reconcile_tiers/web/viewer-tiers-main.js` | Exposes `window.__tierViewer = { scene, camera, controls, renderer, requestRender }` so the screenshot harness can drive camera presets without simulating mouse events |

## Constants tied to the renderer

These constants in `audit.py` mirror values in
`reconcile_tiers/web/render-tuning.js`. Keep them in sync when the
renderer thresholds change, otherwise the "silent drops" check will
disagree with what the user actually sees:

| `audit.py` constant | Source in `render-tuning.js` |
|---|---|
| `MIN_POLYGON_AREA_M2 = 1e-4` | `RENDER_TUNING.minPolygonAreaM2` |
| `OPENING_MIN_DIM = 0.05` | `RENDER_TUNING.opening.minDim` |

## Output Contract

When you finish working in this skill, the user message MUST include:

1. **What ran** — UUID (and element id if any), report folder path.
2. **Realism summary** — failed-flag list (one line). If no flags fired,
   say "LoD2 sanity passed."
3. **Defect summary** — failed-flag list with counts (e.g.
   `ceiling_coverage_gaps: 7 rooms`). If empty, say "No tier_payload
   defects flagged."
4. **The most striking screenshot** — embed `iso.png` at minimum. Embed
   `element.png` if an element id was given.
5. **Recommended next step** — usually one of:
   - "Open report.md" if everything looks routine.
   - "Use `debug-element` on `<locator>`" if a specific defect needs
     root-cause analysis.
   - "Re-extract the building" if `realism.flags` includes
     `footprint_mismatch` or `multiple_components` (the LoD2 itself is
     broken; downstream debugging is moot until reconstruction is
     repaired).

No workspace-state footer required — this skill writes only into
`.context/building-reports/`, which is gitignored.
