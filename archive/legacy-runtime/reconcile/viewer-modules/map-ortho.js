export function createOrthoMapController({
  maplibregl,
  emptyMapStyle,
  getCurrentBuilding,
  getAllData,
  getCurrentBuildingIndex,
  getAnchorModeEnabled,
  setAnchorModeEnabled,
  getModelAlignment,
  setModelOffsets,
  applyAlignmentStateToControls,
  queueSaveAlignmentForCurrentBuilding,
  computeModelFootprintGeoJSON,
  onSetStatus,
}) {
  let orthoMap = null;
  let orthoMarker = null;
  let parcelSourceInitialized = false;

  function setMapStatus(text) {
    if (typeof onSetStatus === 'function') onSetStatus(text || '');
  }

  function distanceMeters(aLng, aLat, bLng, bLat) {
    const R = 6371000;
    const toRad = d => d * Math.PI / 180;
    const dLat = toRad(bLat - aLat);
    const dLng = toRad(bLng - aLng);
    const s1 = Math.sin(dLat / 2);
    const s2 = Math.sin(dLng / 2);
    const aa = s1 * s1 + Math.cos(toRad(aLat)) * Math.cos(toRad(bLat)) * s2 * s2;
    return 2 * R * Math.asin(Math.sqrt(aa));
  }

  function ensureOrthoMap() {
    if (orthoMap) return orthoMap;
    if (!maplibregl) {
      setMapStatus('Map library failed to load');
      return null;
    }

    orthoMap = new maplibregl.Map({
      container: 'map-canvas',
      style: emptyMapStyle,
      center: [10.0, 56.0],
      zoom: 17,
      minZoom: 5,
      maxZoom: 21,
      dragRotate: false,
      pitchWithRotate: false,
      attributionControl: false,
    });
    orthoMap.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    orthoMap.on('click', (e) => {
      if (!getAnchorModeEnabled()) return;
      const bldg = getCurrentBuilding();
      if (!bldg?.gps) return;
      const metersPerDegLat = 111320;
      const metersPerDegLng = 111320 * Math.cos(bldg.gps.lat * Math.PI / 180);
      const eastM = (e.lngLat.lng - bldg.gps.lng) * metersPerDegLng;
      const northM = (e.lngLat.lat - bldg.gps.lat) * metersPerDegLat;
      setModelOffsets(eastM, northM);
      const align = getModelAlignment();
      applyAlignmentStateToControls({
        rotation_deg: align.rotationDeg,
        offset_east_m: eastM,
        offset_north_m: northM,
      });
      updateModelMapOverlay(bldg);
      queueSaveAlignmentForCurrentBuilding();
      setAnchorModeEnabled(false);
      const btn = document.getElementById('map-anchor-mode');
      if (btn) {
        btn.style.borderColor = '#3a3a3a';
        btn.style.color = '#e5e5e5';
      }
      setMapStatus('Anchor applied from map click');
    });

    function ensureOrthoSources() {
      if (!orthoMap.getSource('ortho')) {
        orthoMap.addSource('ortho', {
          type: 'raster',
          tiles: ['/ortofoto?z={z}&x={x}&y={y}'],
          tileSize: 256,
          maxzoom: 19,
        });
        orthoMap.addLayer({ id: 'ortho-tiles', type: 'raster', source: 'ortho' });
      }
      if (!orthoMap.getSource('parcel')) {
        orthoMap.addSource('parcel', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
        });
        orthoMap.addLayer({
          id: 'parcel-fill',
          type: 'fill',
          source: 'parcel',
          paint: { 'fill-color': '#ffffff', 'fill-opacity': 0.08 },
        });
        orthoMap.addLayer({
          id: 'parcel-outline',
          type: 'line',
          source: 'parcel',
          paint: { 'line-color': '#4ecdc4', 'line-width': 2 },
        });
        parcelSourceInitialized = true;
      }
      if (!orthoMap.getSource('osm-candidates')) {
        orthoMap.addSource('osm-candidates', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
        });
        orthoMap.addLayer({
          id: 'osm-candidates-fill',
          type: 'fill',
          source: 'osm-candidates',
          paint: { 'fill-color': '#f59e0b', 'fill-opacity': 0.08 },
        });
        orthoMap.addLayer({
          id: 'osm-candidates-outline',
          type: 'line',
          source: 'osm-candidates',
          paint: { 'line-color': '#f59e0b', 'line-width': 1.5 },
        });
      }
      if (!orthoMap.getSource('osm-selected')) {
        orthoMap.addSource('osm-selected', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
        });
        orthoMap.addLayer({
          id: 'osm-selected-outline',
          type: 'line',
          source: 'osm-selected',
          paint: { 'line-color': '#ef4444', 'line-width': 3 },
        });
      }
      if (!orthoMap.getSource('model-footprint')) {
        orthoMap.addSource('model-footprint', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
        });
        orthoMap.addLayer({
          id: 'model-footprint-fill',
          type: 'fill',
          source: 'model-footprint',
          paint: { 'fill-color': '#06b6d4', 'fill-opacity': 0.10 },
        });
        orthoMap.addLayer({
          id: 'model-footprint-outline',
          type: 'line',
          source: 'model-footprint',
          paint: { 'line-color': '#06b6d4', 'line-width': 2.5 },
        });
      }
    }

    orthoMap.on('load', ensureOrthoSources);
    if (orthoMap.isStyleLoaded()) ensureOrthoSources();
    else orthoMap.once('styledata', () => {
      if (orthoMap && orthoMap.isStyleLoaded()) ensureOrthoSources();
    });

    return orthoMap;
  }

  function updateModelMapOverlay(bldg) {
    const map = orthoMap;
    if (!map) return;
    const src = map.getSource('model-footprint');
    if (!src) return;
    const { enabled } = getModelAlignment();
    if (!enabled) {
      src.setData({ type: 'FeatureCollection', features: [] });
      return;
    }
    const gj = computeModelFootprintGeoJSON(bldg);
    src.setData(gj || { type: 'FeatureCollection', features: [] });
  }

  function updateOrthoPanelForBuilding(bldg, orthoEnabled) {
    const panel = document.getElementById('map-panel');
    panel.classList.toggle('visible', orthoEnabled);
    if (!orthoEnabled) return;

    const map = ensureOrthoMap();
    if (!map) return;
    const gps = bldg.gps;
    if (!gps || typeof gps.lng !== 'number' || typeof gps.lat !== 'number') {
      setMapStatus('No GPS location in datapack for this address');
      return;
    }

    const recenter = () => {
      if (!parcelSourceInitialized && !map.getSource('parcel')) return;

      if (!orthoMarker) {
        orthoMarker = new maplibregl.Marker({ color: '#ff6b6b' });
      }
      orthoMarker.setLngLat([gps.lng, gps.lat]).addTo(map);

      let hasParcel = false;
      const parcel = bldg.parcel_boundary_geojson;
      const parcelSource = map.getSource('parcel');
      if (parcelSource && parcel && typeof parcel === 'object') {
        parcelSource.setData(parcel);
        hasParcel = true;
      } else if (parcelSource) {
        parcelSource.setData({ type: 'FeatureCollection', features: [] });
      }

      let usedParcelBounds = false;
      if (hasParcel) {
        try {
          const coords = parcel.geometry?.coordinates?.[0] || [];
          if (coords.length >= 3) {
            const lons = coords.map(c => c[0]);
            const lats = coords.map(c => c[1]);
            map.fitBounds(
              [[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]],
              { padding: 40, duration: 300 }
            );
            usedParcelBounds = true;
          }
        } catch (_err) {}
      }

      const candidates = Array.isArray(bldg.building_features) ? bldg.building_features
        : Array.isArray(bldg.osm_on_address_features) ? bldg.osm_on_address_features : [];
      const buildingSource = bldg.building_data_source || 'osm';
      const cSrc = map.getSource('osm-candidates');
      const sSrc = map.getSource('osm-selected');
      if (cSrc) cSrc.setData({ type: 'FeatureCollection', features: candidates });

      let selected = null;
      let nearestM = Infinity;
      for (const f of candidates) {
        const c = f?.properties?.centroid_wgs84;
        if (!c || typeof c.lng !== 'number' || typeof c.lat !== 'number') continue;
        const m = distanceMeters(gps.lng, gps.lat, c.lng, c.lat);
        if (m < nearestM) {
          nearestM = m;
          selected = f;
        }
      }
      if (sSrc) {
        sSrc.setData({ type: 'FeatureCollection', features: selected ? [selected] : [] });
      }
      updateModelMapOverlay(bldg);

      if (selected) {
        const facades = selected?.properties?.selected_facades || [];
        const az = Array.isArray(facades) && facades.length > 0 ? facades[0].azimuth_deg : null;
        const srcLabel = buildingSource === 'datafordeler' ? 'GeoDanmark' : 'OSM';
        if (typeof az === 'number') {
          setMapStatus(
            `${usedParcelBounds ? 'Orthophoto + parcel' : 'Orthophoto'} + ${srcLabel} building `
            + `(nearest ${nearestM.toFixed(1)}m, facade azimuth ${az.toFixed(1)}deg)`
          );
        } else {
          setMapStatus(
            `${usedParcelBounds ? 'Orthophoto + parcel' : 'Orthophoto'} + ${srcLabel} building `
            + `(nearest ${nearestM.toFixed(1)}m)`
          );
        }
      }

      if (!usedParcelBounds) {
        map.flyTo({ center: [gps.lng, gps.lat], zoom: 19, duration: 300 });
      }
      if (!selected && hasParcel) setMapStatus('Orthophoto + parcel boundary');
      if (!selected && !hasParcel) setMapStatus('Orthophoto centered on GPS');
    };

    if (map.loaded()) recenter();
    else map.once('load', recenter);
  }

  function getMap() {
    return orthoMap;
  }

  return {
    setMapStatus,
    ensureOrthoMap,
    updateModelMapOverlay,
    updateOrthoPanelForBuilding,
    getMap,
  };
}
