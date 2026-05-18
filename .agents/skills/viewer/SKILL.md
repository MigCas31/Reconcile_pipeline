---
name: building-viewer
description: >
  Use when working on the Three.js building viewer, viewer-modules/,
  viewer_server.py, orthophoto overlay, or MapLibre integration.
---

# Building Viewer

## How to Work on the Viewer

1. **Check Three.js docs first** — [threejs.org/docs](https://threejs.org/docs/) has examples for nearly every geometry, material, and control pattern. Don't write custom code when Three.js has a built-in solution.
2. **Check MapLibre docs** — [maplibre.org/maplibre-gl-js/docs](https://maplibre.org/maplibre-gl-js/docs/) for anything map/tile related. The existing integration is in `map-ortho.js`.
3. **Use the existing `geometry.js` utilities** — `createPolygonMesh()`, `createEdgeLoop()`, etc. already handle polygon holes, vertex dedup, and edge cases. Don't bypass them.
4. **Follow the module pattern** — each visualization concern lives in its own file in `viewer-modules/`. Don't add large new features to `viewer-main.js` directly.
5. **Search for Three.js examples** — the Three.js examples page and discourse forum have solutions for most 3D visualization problems. Search before building custom.
6. **Test with real data** — load buildings from `pipeline-outputs/` in the viewer. Rendering bugs often only appear with real-world geometry (non-convex polygons, degenerate faces, etc.).
7. **Dispose properly** — Three.js leaks GPU memory if geometries/materials aren't disposed. Call `.dispose()` when removing objects from the scene.

## Architecture

```
viewer_server.py (Python HTTP :8765)
  |-> serves static assets -> viewer.html -> viewer-main.js -> viewer-modules/*
  |-> proxies Datafordeler WMTS orthophoto tiles (avoids CORS)
```

## Module Reference

| Module | Key Exports | Purpose |
|--------|------------|---------|
| `viewer-main.js` | — | Scene bootstrap: WebGLRenderer, OrbitControls, lighting, camera. Wires all modules. |
| `constants.js` | `STORY_COLORS`, `ROOM_COLORS`, `ROOF_CLUSTER_COLORS`, `SOURCE_COLORS` | Color palettes for visualization |
| `geometry.js` | `createPolygonMesh()`, `createEdgeLoop()`, `createLine()`, `projectToPlane2()` | 3D mesh primitives. Supports polygon holes, vertex deduplication. |
| `roof-python.js` | `renderRoofFromPythonResult()` | Maps Python roof pipeline output to Three.js meshes with cluster color lookup |
| `ui-bindings.js` | `bindUIEventHandlers()` | Event handlers via dependency injection (large param object) |
| `map-ortho.js` | `createOrthoMapController()` | MapLibre satellite overlay, marker placement, distance calc, anchor mode |

## Three.js Patterns

- **Coordinate system**: Y-up (matches Python codebase)
- **Renderer**: WebGLRenderer with OrbitControls
- **Materials**: MeshBasicMaterial and MeshPhongMaterial — no custom shaders
- **Geometry**: BufferGeometry for building meshes, created via `geometry.js` utilities
- **Lighting**: Three-point lighting setup in `viewer-main.js`

## Server (`viewer_server.py`)

- Serves viewer assets on port **8765**
- Proxies Datafordeler orthophoto WMTS tiles to avoid CORS
- API key resolution: `DATAFORDELEREN_API_KEY` env var, falls back to GCP Secret Manager
- Run: `DATAFORDELEREN_API_KEY=... python reconcile/viewer_server.py`

**CRITICAL**: Without a valid API key, orthophoto tiles will fail silently. The 3D model still renders but without satellite background.

## MapLibre Integration

`map-ortho.js` manages the satellite imagery overlay:
- Creates MapLibre map instance alongside Three.js canvas
- Supports marker placement for building location
- Anchor mode for aligning 3D model position to map coordinates
- Distance calculations between map points

## Adding New Visualizations

1. Create a new module in `reconcile/viewer-modules/`
2. Export a render function that takes building data and Three.js scene
3. Import in `viewer-main.js`
4. Wire event handlers via `bindUIEventHandlers()` dependency injection pattern
5. Add color constants to `constants.js` if needed

## Element Locator / Shareable IDs

Every mesh in the viewer can be stamped with a shareable element ID via `attachLocator(mesh, { buildingUuid, kind, id, corners, ... })`. This sets `mesh.userData.elementUid` and registers the mesh in `elementMeshByUid`.

- **Right-click** (contextmenu handler, viewer-main.js:1325): raycasts against all groups, copies `elementUid` to clipboard
- **Search bar** (Enter in `#search`): calls `jumpToElementUid()` to select + focus the element
- **URL hash**: `#eid=<encoded_uid>` persists selection across reloads

**Rule**: Any new renderable mesh type MUST call `attachLocator()` to be right-clickable and selectable. Without it, the raycaster hits the mesh but finds no `elementUid`.

The `roof-python.js` module receives `attachLocator` and `buildingUuid` as dependencies and attaches locators to all roof/ceiling surface meshes (but not cluster segment lines or edge loops).

## Gotchas

- The `bindUIEventHandlers()` function takes a large parameter object (dependency injection) — pass all dependencies explicitly, don't use globals
- `createPolygonMesh()` handles polygon holes — don't manually triangulate
- Building data is loaded from `buildings_3d.json` (output of extraction pipeline)
- Vertex deduplication in `geometry.js` — duplicate vertices are merged automatically
