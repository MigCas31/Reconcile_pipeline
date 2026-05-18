---
name: extraction-pipeline
description: >
  Use when working on 3D extraction from scan data, the extract_3d pipeline,
  builder pattern in extract3d/, cross-floor gap detection, or the reconcile/
  CLI entry points.
---

# Extraction Pipeline

## How to Work on Extraction

1. **Read the full pipeline first** — understand the data flow before modifying a single step. Changes to early steps cascade through everything downstream.
2. **Use the builder pattern** — don't bypass `extract3d/builder.py` to create Building objects directly. The builder ensures consistent state.
3. **Test on real buildings** — run extraction on buildings from `pipeline-outputs/` and visually verify in the viewer. Synthetic tests alone miss spatial bugs.
4. **Check how calor does it** — the Go backend (`../calor` or github.com/lun-energy/calor) processes the same scan data. Consult it for reference on how elements should be matched, merged, or trusted.
5. **Search for prior art** — Apple RoomPlan processing, multi-session scan merging, and building element matching are documented in Apple developer forums and WWDC sessions. Search before inventing new approaches.

## What We're Extracting (and Why)

A building surveyor walks through a home with an iPad, scanning room by room using Apple RoomPlan. Each scan session captures one room's geometry: walls, floor, ceiling, doors, windows. But a building is more than a collection of rooms — it has:

- **Shared walls** between adjacent rooms (the wall between your kitchen and living room is one wall, scanned from both sides)
- **Stories** that stack vertically (the ceiling of the ground floor is roughly the floor of the first floor)
- **Exterior vs interior walls** (exterior walls face outside, interior walls separate rooms)
- **Gaps and overlaps** from scan imprecision (two rooms might report slightly different positions for their shared wall)

The extraction pipeline reconciles these individual room scans into a single coherent building model, resolving conflicts, merging shared elements, and establishing the spatial relationships that make it a *building* rather than a bag of rooms.

## Data Flow

```
scan.json
  -> reconcile/extract_3d.py          (main pipeline)
  -> reconcile/extract3d/builder.py   (accumulates rooms/walls/elements)
  -> reconcile/extract3d/ceilings.py  (ceiling plane detection)
  -> reconcile/extract3d/exterior.py  (exterior wall identification)
  -> reconcile/extract3d/gaps.py      (gap detection between rooms/stories)
  -> reconcile/extract3d/overlaps.py  (overlap resolution)
  -> reconcile/extract3d/stitch.py    (cross-session stitching)
  -> buildings_3d.json                (output)
```

## Entry Points

| File | Function | Purpose |
|------|----------|---------|
| `reconcile/extract_3d.py` | — | Main V1 extraction pipeline |
| `reconcile/cli.py` | `reconcile_building()` | Chains: load, match elements, compute walls, trust merge, quality report, output |
| `reconcile/cli_v2.py` | — | Shim — delegates to `reconcile_v2.cli` |

## Modular Extraction (`reconcile/extract3d/`)

| Module | Responsibility |
|--------|---------------|
| `builder.py` | Builder pattern orchestrator — accumulates rooms, walls, elements into Building |
| `ceilings.py` | Ceiling plane detection from room geometry |
| `exterior.py` | Exterior wall identification (inside vs outside) |
| `gaps.py` | Gap detection between adjacent rooms and stories |
| `overlaps.py` | Overlap resolution when rooms share geometry |
| `stitch.py` | Geometry stitching across multiple scan sessions |

## Builder Pattern

The builder in `extract3d/builder.py` accumulates state progressively:

1. Initialize with raw scan data
2. Process rooms (geometry extraction)
3. Compute walls (corner polygons, extension strips)
4. Detect ceilings, exterior walls
5. Resolve gaps and overlaps
6. Stitch across sessions
7. Output final Building

**Order matters** — each module depends on state from previous steps.

## Cross-Floor Gap Detection

`reconcile/cross_floor_gaps.py` detects vertical gaps between stories by comparing floor/ceiling heights. Critical for multi-story buildings where scan sessions don't perfectly align vertically.

## Gotchas

- Builder accumulates state — calling modules out of order produces incorrect results
- Cross-floor gaps depend on consistent story numbering across scan sessions
- The `reconcile_building()` CLI function is the canonical pipeline — use it as reference for step ordering
- Output goes to `buildings_3d.json` — this is the input for both the viewer and the roof pipeline
