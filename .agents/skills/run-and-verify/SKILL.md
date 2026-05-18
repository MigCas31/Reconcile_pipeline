---
name: run-and-verify
description: >
  Use when running pipelines end-to-end, restarting the viewer server,
  opening the browser, or checking for JS/HTML errors. Covers the full
  extract -> view -> verify loop.
---

# Run & Verify

End-to-end instructions for running pipelines, serving the viewer, and verifying everything works in the browser.

## Quick Reference

| Task | Command |
|------|---------|
| Extract 3D (all buildings) | `python reconcile/extract_3d.py` |
| Extract 3D (one building) | `python reconcile/extract_3d.py <uuid>` |
| Reconcile V1 (one building) | `python -m reconcile.cli --building pipeline-outputs/<uuid>/merged.json --scan-cache .scan-cache --output pipeline-outputs/<uuid>/reconciled.json --uuid <uuid>` |
| Topology V2 (one building) | `python -m reconcile_v2.cli --building pipeline-outputs/<uuid>/merged.json --scan-cache .scan-cache --output-v2 pipeline-outputs/<uuid>/topology-v2.json --uuid <uuid>` |
| Start viewer server | `python -m reconcile_tiers.server` |
| Run tests | `python -m pytest tests/ -v` |
| Full lint/format/test | `make verify` |

## End-to-End Pipeline

### Step 1: Pick a Building

Buildings live in `pipeline-outputs/<uuid>/`. Each has at minimum a `merged.json`.

```bash
# List available buildings
ls pipeline-outputs/ | head -10

# Pick one UUID for testing, e.g.:
UUID="016980bc-6762-4022-bfbf-17df4112e10c"
```

### Step 2: Run Extraction (produces buildings_3d.json)

The extraction pipeline reads `merged.json` + `.scan-cache/` and writes `reconcile/buildings_3d.json`, which is what the viewer loads.

```bash
# Single building (fast, use for iteration):
python reconcile/extract_3d.py $UUID

# All buildings (slow, ~200 buildings):
python reconcile/extract_3d.py
```

**Output**: `reconcile/buildings_3d.json` — an array of building objects with rooms, walls, floors, ceilings, doors, windows, roof data.

**Depends on**: `pipeline-outputs/<uuid>/merged.json` and optionally `.scan-cache/` directories.

### Step 3: Start the Viewer Server

```bash
python -m reconcile_tiers.server
```

- Serves on **http://127.0.0.1:8080/reconcile_tiers/web/viewer-tiers.html** (port configurable via `VIEWER_PORT` env var)
- Serves static files from workspace root (so `/reconcile_tiers/web/*`, `/pipeline-outputs/<uuid>/*` all resolve)
- **Do NOT use** `python reconcile/viewer_server.py` — `reconcile/` is a symlink to `archive/legacy-runtime/reconcile/`

### Step 4: Kill & Restart the Server

If the server is already running and you need to restart (e.g., after changing `reconcile_tiers/server.py`):

```bash
# Find and kill existing server
lsof -ti :8080 | xargs kill -9 2>/dev/null

# Restart
python -m reconcile_tiers.server &
```

For JS/HTML changes, you do NOT need to restart the server — just reload the browser.

### Step 5: Open in Browser and Verify

Use browser automation tools to open the viewer and check for errors:

1. **Open the viewer**:
   - Navigate to `http://127.0.0.1:8080/reconcile_tiers/web/viewer-tiers.html`
   - Or if a specific building: `http://127.0.0.1:8080/reconcile_tiers/web/viewer-tiers.html#uuid=<uuid>`

2. **Check the JS console for errors**:
   - Use `mcp__claude-in-chrome__read_console_messages` with `pattern: "error|Error|ERR|Uncaught|TypeError|ReferenceError|SyntaxError"` to catch JS errors
   - Also check without filter to see warnings

3. **Check the page rendered correctly**:
   - Use `mcp__claude-in-chrome__read_page` to verify the HTML loaded
   - Use `mcp__claude-in-chrome__javascript_tool` to check Three.js scene state, e.g.:
     ```js
     // Check if scene has children (buildings loaded)
     document.querySelector('canvas') !== null && typeof scene !== 'undefined' ? scene.children.length : 'no scene'
     ```

4. **Visual verification** (when needed):
   - Use `mcp__claude-in-chrome__computer` to take a screenshot and visually inspect the 3D model
   - Check that buildings render, floors/walls/roofs appear, no geometry artifacts

## What to Verify After Changes

### After JS/HTML changes (viewer-main.js, viewer-modules/, viewer.html)
- No need to restart server or re-run extraction
- Just reload the browser page
- Check console for JS errors (SyntaxError, ReferenceError, TypeError)
- Verify the 3D model still renders (canvas exists, scene has children)
- Check that UI controls still work (dropdowns, checkboxes, search bar)

### After Python extraction changes (extract_3d.py, extract3d/)
- Re-run extraction: `python reconcile/extract_3d.py <uuid>`
- Reload the browser (no server restart needed)
- Verify building geometry looks correct in viewer
- Check console for data format errors (the viewer logs warnings for unexpected data shapes)

### After server changes (reconcile_tiers/server.py)
- Kill and restart the server: `lsof -ti :8080 | xargs kill -9 && python -m reconcile_tiers.server &`
- Reload the browser
- Check that static files still serve correctly

### After roof pipeline changes (roof_algorithms_py/)
- Re-run extraction (roof pipeline runs as part of extraction): `python reconcile/extract_3d.py <uuid>`
- Reload the browser
- Check roof surfaces render (enable roof layers in viewer UI)
- Verify no console errors from roof-python.js

### After reconcile V1 changes (cli.py, matcher, trust_merge, etc.)
- Re-run reconciliation: `python -m reconcile.cli --building pipeline-outputs/<uuid>/merged.json --scan-cache .scan-cache --output pipeline-outputs/<uuid>/reconciled.json --uuid <uuid>`
- Then re-run extraction (it reads reconciled.json for classification): `python reconcile/extract_3d.py <uuid>`
- Reload browser

### After topology V2 changes (reconcile_v2/)
- Re-run V2 pipeline: `python -m reconcile_v2.cli --building pipeline-outputs/<uuid>/merged.json --scan-cache .scan-cache --output-v2 pipeline-outputs/<uuid>/topology-v2.json --uuid <uuid>`
- V2 output is independent of the viewer (topology-v2.json is consumed by calor, not the viewer)

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Address already in use` | Server already running on port | `lsof -ti :8080 \| xargs kill -9` |
| Blank viewer, no buildings | `buildings_3d.json` empty or missing | Re-run extraction |
| `ModuleNotFoundError` | Missing dependency | `pip install numpy shapely` |
| Orthophotos not loading | Missing API key | Set `DATAFORDELEREN_API_KEY` env var |
| JS `SyntaxError` | Broken JS edit | Check the file for syntax issues |
| `TypeError: Cannot read properties of undefined` | Data shape changed | Check extraction output format matches what viewer expects |
| Console: `Failed to load buildings_3d.json` | File doesn't exist | Run `python reconcile/extract_3d.py` first |

## File Dependencies

```
Pipeline data flow:

  pipeline-outputs/<uuid>/merged.json   (input: scan data)
  .scan-cache/scans_*                   (input: raw room scans)
       |
       v
  reconcile/cli.py                      (V1 reconciliation)
       |-> pipeline-outputs/<uuid>/reconciled.json
       |
  reconcile/extract_3d.py               (3D extraction + roof pipeline)
       |-> reconcile/buildings_3d.json   (viewer input)
       |
  reconcile_tiers/server.py             (HTTP server — python -m reconcile_tiers.server)
       |-> serves reconcile_tiers/web/viewer-tiers.html
       |-> serves reconcile_tiers/web/*.js
       |-> serves pipeline-outputs/<uuid>/tier_payload.json

NOTE: reconcile/ is a symlink → archive/legacy-runtime/reconcile/. Never use reconcile/viewer_server.py.
```

## Testing Checklist

Before considering a change complete:

- [ ] `python -m pytest tests/ -v` passes
- [ ] Extraction runs without errors on at least one building
- [ ] Viewer loads without JS console errors
- [ ] 3D model renders visually (buildings visible, no obvious geometry bugs)
- [ ] `make verify` passes (lint + format + typecheck + tests)
