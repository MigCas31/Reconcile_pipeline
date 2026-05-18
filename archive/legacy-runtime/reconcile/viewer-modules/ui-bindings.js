export function bindUIEventHandlers({
  documentRef = document,
  LAYER_CONTROL_IDS,
  LAYER_KEYS,
  groups,
  getData,
  getCurrentBuilding,
  getPipelineStepIndex,
  applyPipelineStep,
  setLayerVisibility,
  onLayerVisibilityChanged,
  renderLegend,
  loadBuilding,
  setColorByStory,
  setColorBySource,
  setOrthoEnabled,
  updateOrthoPanelForBuilding,
  getOrthoMap,
  setModelMapEnabled,
  setModelMapRotationDeg,
  setModelMapOffsetEastM,
  setModelMapOffsetNorthM,
  getModelMapRotationDeg,
  getModelMapOffsetEastM,
  getModelMapOffsetNorthM,
  applyAlignmentStateToControls,
  updateModelMapOverlay,
  queueSaveAlignmentForCurrentBuilding,
  normalizeDeg,
  setMapStatus,
  setAnchorModeEnabled,
  getAnchorModeEnabled,
}) {
  const searchInput = documentRef.getElementById('search');
  searchInput.addEventListener('input', () => {
    const q = searchInput.value.toLowerCase();
    documentRef.querySelectorAll('.bldg-item').forEach(el => {
      const text = el.dataset.search;
      el.style.display = text.includes(q) ? '' : 'none';
    });
    const visible = documentRef.querySelectorAll('.bldg-item:not([style*="display: none"])').length;
    const total = (getData() || []).length;
    documentRef.getElementById('sidebar-stats').textContent = q ? `${visible} / ${total} buildings` : `${total} buildings`;
  });

  documentRef.addEventListener('keydown', e => {
    const t = e.target;
    if (t === searchInput || (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.tagName === 'BUTTON'))) return;
    const labelingMode = documentRef.body?.dataset?.mode === 'labeling';
    if (e.key === 'ArrowDown' || e.key === 'j') {
      if (labelingMode) return;
      e.preventDefault();
      const visible = [...documentRef.querySelectorAll('.bldg-item:not([style*="display: none"])')];
      const curIdx = visible.findIndex(el => el.classList.contains('active'));
      if (curIdx < visible.length - 1) { visible[curIdx + 1]?.click(); visible[curIdx + 1]?.scrollIntoView({ block: 'nearest' }); }
    }
    if (e.key === 'ArrowUp' || e.key === 'k') {
      if (labelingMode) return;
      e.preventDefault();
      const visible = [...documentRef.querySelectorAll('.bldg-item:not([style*="display: none"])')];
      const curIdx = visible.findIndex(el => el.classList.contains('active'));
      if (curIdx > 0) { visible[curIdx - 1]?.click(); visible[curIdx - 1]?.scrollIntoView({ block: 'nearest' }); }
    }
    if (e.key === 'ArrowRight' || e.key === 'l') {
      if (labelingMode) return;
      e.preventDefault();
      applyPipelineStep(getPipelineStepIndex() + 1);
    }
    if (e.key === 'ArrowLeft' || e.key === 'h') {
      if (labelingMode) return;
      e.preventDefault();
      applyPipelineStep(getPipelineStepIndex() - 1);
    }
  });

  documentRef.getElementById('pipeline-prev').addEventListener('click', () => applyPipelineStep(getPipelineStepIndex() - 1));
  documentRef.getElementById('pipeline-next').addEventListener('click', () => applyPipelineStep(getPipelineStepIndex() + 1));
  documentRef.getElementById('pipeline-steps').addEventListener('click', (e) => {
    const target = e.target.closest('[data-step]');
    if (!target) return;
    const i = Number(target.dataset.step);
    if (Number.isInteger(i)) applyPipelineStep(i);
  });

  for (const [layer, id] of Object.entries(LAYER_CONTROL_IDS)) {
    documentRef.getElementById(id).addEventListener('change', e => {
      setLayerVisibility(layer, e.target.checked, false);
      if (onLayerVisibilityChanged) onLayerVisibilityChanged(layer, e.target.checked);
      renderLegend();
    });
  }

  documentRef.getElementById('show-wireframe').addEventListener('change', e => {
    for (const key of LAYER_KEYS) {
      const g = groups[key];
      g.traverse(c => { if (c.isMesh) c.material.wireframe = e.target.checked; });
    }
  });

  documentRef.getElementById('color-by-story').addEventListener('change', e => {
    setColorByStory(e.target.checked);
    loadBuilding(getCurrentBuilding(), { resetPipeline: false });
  });

  documentRef.getElementById('color-by-source').addEventListener('change', e => {
    setColorBySource(e.target.checked);
    loadBuilding(getCurrentBuilding(), { resetPipeline: false });
  });

  documentRef.getElementById('show-ortho').addEventListener('change', e => {
    setOrthoEnabled(e.target.checked);
    documentRef.getElementById('map-panel').classList.toggle('visible', e.target.checked);
    const data = getData();
    if (data[getCurrentBuilding()]) updateOrthoPanelForBuilding(data[getCurrentBuilding()]);
    const map = getOrthoMap();
    if (map) setTimeout(() => map.resize(), 0);
  });

  function wireMapAlignmentControl(id, setter, labelId) {
    const el = documentRef.getElementById(id);
    const label = documentRef.getElementById(labelId);
    const apply = () => {
      setter(Number(el.value));
      label.textContent = String(Number(el.value));
      const data = getData();
      const bldg = data[getCurrentBuilding()];
      if (bldg && getOrthoMap()) updateModelMapOverlay(bldg);
      queueSaveAlignmentForCurrentBuilding();
    };
    el.addEventListener('input', apply);
    apply();
  }

  documentRef.getElementById('show-model-map').addEventListener('change', e => {
    setModelMapEnabled(e.target.checked);
    const data = getData();
    const bldg = data[getCurrentBuilding()];
    if (bldg && getOrthoMap()) updateModelMapOverlay(bldg);
  });

  wireMapAlignmentControl('map-rot', setModelMapRotationDeg, 'map-rot-val');
  wireMapAlignmentControl('map-east', setModelMapOffsetEastM, 'map-east-val');
  wireMapAlignmentControl('map-north', setModelMapOffsetNorthM, 'map-north-val');

  documentRef.getElementById('map-align-reset').addEventListener('click', () => {
    applyAlignmentStateToControls({ rotation_deg: 0, offset_east_m: 0, offset_north_m: 0 });
    const data = getData();
    const bldg = data[getCurrentBuilding()];
    if (bldg && getOrthoMap()) updateModelMapOverlay(bldg);
    queueSaveAlignmentForCurrentBuilding();
  });

  documentRef.getElementById('map-auto-align').addEventListener('click', () => {
    const data = getData();
    const bldg = data[getCurrentBuilding()];
    if (!bldg) return;
    const scanAlign = bldg.scan_alignment || {};
    const suggested = scanAlign.suggested_rotation_deg;
    if (typeof suggested !== 'number') {
      setMapStatus('Auto align needs scan JSON heading metadata');
      return;
    }

    const rot = normalizeDeg(suggested);
    setModelMapRotationDeg(rot);
    applyAlignmentStateToControls({
      rotation_deg: rot,
      offset_east_m: getModelMapOffsetEastM(),
      offset_north_m: getModelMapOffsetNorthM(),
    });
    if (getOrthoMap()) updateModelMapOverlay(bldg);
    queueSaveAlignmentForCurrentBuilding();

    const gcDeg = scanAlign.grid_convergence_deg;
    const gcInfo = typeof gcDeg === 'number' && Math.abs(gcDeg) > 0.001
      ? `, grid convergence ${gcDeg.toFixed(2)}deg`
      : '';
    setMapStatus(
      `Auto rotation set to ${rot.toFixed(1)}deg from scan JSON `
      + `(${scanAlign.source || 'unknown source'}, confidence `
      + `${Number(scanAlign.mode_confidence ?? 0).toFixed(2)}${gcInfo})`
    );
  });

  documentRef.getElementById('map-anchor-mode').addEventListener('click', () => {
    setAnchorModeEnabled(!getAnchorModeEnabled());
    const btn = documentRef.getElementById('map-anchor-mode');
    btn.style.borderColor = getAnchorModeEnabled() ? '#06b6d4' : '#3a3a3a';
    btn.style.color = getAnchorModeEnabled() ? '#06b6d4' : '#e5e5e5';
    if (getAnchorModeEnabled()) setMapStatus('Anchor mode: click map to set model center');
  });
}
