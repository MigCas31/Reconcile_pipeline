---
name: danish-geodata
description: >
  Use when working with Danish government data APIs (Datafordeler,
  Dataforsyningen), building footprints, orthophoto tiles, or the
  WMTS proxy in viewer_server.py.
---

# Danish Geodata

## How to Work with Danish Geodata

1. **Read the official API docs** — [datafordeler.dk](https://datafordeler.dk/) has the canonical reference for all available services, query parameters, and data formats. Always check the docs before adding or modifying API calls.
2. **Check Dataforsyningen too** — [dataforsyningen.dk](https://dataforsyningen.dk/) is a related Danish geodata portal that may have newer or additional services. Some endpoints have migrated between the two.
3. **Test with real coordinates** — use actual Danish building coordinates from `pipeline-outputs/` when testing API calls. The APIs behave differently for edge cases (building straddling tile boundaries, demolished buildings, etc.).
4. **Handle API failures gracefully** — Datafordeler has occasional downtime and rate limits. Code should degrade gracefully (viewer works without orthophotos, footprint queries retry or show clear errors).
5. **Keep coordinate conversions correct** — always convert WGS84 to UTM before querying. Search for "pyproj" or "EPSG:25832" if you need to add new coordinate operations — don't write manual projection formulas.
6. **Check grid convergence parity** — grid convergence is implemented in both Python (here) and TypeScript (web-main). Changes must be mirrored and verified with `tests/test_grid_convergence.py`.

## APIs Used

| API | Purpose | File |
|-----|---------|------|
| Datafordeler (GeoDanmark) | Building footprints via GraphQL | `reconcile/datafordeler.py` |
| Datafordeler WMTS | Orthophoto satellite tiles | proxied via `reconcile/viewer_server.py` |

## Authentication

- **Primary**: `DATAFORDELEREN_API_KEY` environment variable
- **Fallback**: GCP Secret Manager (`resolve_datafordeleren_api_key()` in `viewer_server.py`)
- **CRITICAL**: Never hardcode API keys in source code

## Building Footprint Queries (`datafordeler.py`)

- GraphQL query `BYGNING_QUERY` to GeoDanmark API
- Input: WGS84 lat/lon coordinates
- Internally converts to UTM via `_wgs84_to_utm()` before querying
- Filters out unbuilt projects (status-based filtering)
- Returns building geometry polygons

## Coordinate Conversion

| Function | File | Purpose |
|----------|------|---------|
| `_wgs84_to_utm()` | `datafordeler.py` | WGS84 -> UTM32N for API queries |
| `compute_grid_convergence_rad()` | `grid_convergence.py` | True north to grid north angle |
| `GridNorthReference` | `grid_convergence.py` | Dataclass storing convergence result |

Supported projections:
- **UTM** (Denmark, EPSG:25832) — primary
- **Lambert** (France, EPSG:2154) — secondary

Grid convergence was ported from web-main's `north-reference.ts` and verified in `tests/test_grid_convergence.py`.

## WMTS Tile Proxy

`viewer_server.py` proxies orthophoto tile requests to Datafordeler:
- Avoids CORS restrictions in the browser
- Tiles are Danish orthophotos (aerial/satellite imagery)
- Used by `map-ortho.js` MapLibre controller in the viewer

## Anti-Patterns

| Wrong | Correct | Why |
|-------|---------|-----|
| Query Datafordeler with WGS84 coords directly | Convert to UTM first via `_wgs84_to_utm()` | API expects UTM32N coordinates |
| Hardcode API key in source | Use env var `DATAFORDELEREN_API_KEY` | Security — key rotates |
| Skip grid convergence for small areas | Always apply correction | Error accumulates at building scale |
| Fetch tiles directly from browser | Use viewer_server.py proxy | CORS blocks direct Datafordeler requests |
