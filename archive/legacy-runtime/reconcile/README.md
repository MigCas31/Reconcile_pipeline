# Reconcile Viewer

## Regenerate Viewer Data (includes GPS + parcel boundary)

```bash
python reconcile/extract_3d.py
```

This writes `reconcile/buildings_3d.json` and now includes:

- `gps`: `{ "lat": number, "lng": number }` (from `datapack.json` `resolved_location.coordinates_wgs84`)
- `parcel_boundary_geojson`: parcel boundary in WGS84 (from `datapack.json` `parcel_boundary.geojson_wgs84`)

## Run Viewer with Orthophoto Proxy

```bash
DATAFORDELEREN_API_KEY=your_key python reconcile/viewer_server.py
```

Open:

- `http://127.0.0.1:8765/viewer.html`

In the viewer, enable `Orthophoto` in the controls bar to show the WMTS orthophoto panel with GPS center + parcel outline.
