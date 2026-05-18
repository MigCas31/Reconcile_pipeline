import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { createPolygonMesh, createEdgeLoop } from './viewer-modules/geometry.js';
import { RAW_CEILING_COLOR, RAW_CEILING_EDGE } from './viewer-modules/constants.js';

// ---- state ----
const state = {
  buildings: [],          // from /roof-index
  filter: 'all',
  search: '',
  currentUuid: null,
  detail: null,           // from /roof-detail?uuid=...
  toggles: {
    segments: true,
    segUnused: false,
    candidates: true,
    committed: true,
    flat: false,
    rawCeilings: false,
    roomCeilings: false,
    floors: true,
    colorByClass: true,
  },
  buildingObjects: [],    // three objects we add per building; cleared on switch
  clickables: [],         // {object, kind, payload} for picking
  hudEl: document.getElementById('hud'),
};

// Cluster palette — stable across selections so the same cluster keeps its hue.
const CLUSTER_COLORS = [
  0xff6464, 0x64ff64, 0x6496ff, 0xffa864, 0xff64ff,
  0x64ffff, 0xffff64, 0xff8c42, 0x42ff8c, 0x8c42ff,
  0xffbbbb, 0xbbffbb, 0xbbbbff, 0xffddaa, 0xddaaff,
];
function clusterColor(i) {
  if (i === null || i === undefined) return 0x666666;
  return CLUSTER_COLORS[i % CLUSTER_COLORS.length];
}

function classColor(cls) {
  if (cls === 'strong') return 0x64ff9a;
  if (cls === 'weak') return 0xff6a6a;
  if (cls === 'none') return 0xffd864;
  return 0x888888;
}

// ---- three.js scene ----
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a12);
window.__scene = scene;
window.__state = state;

const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 2000);
camera.position.set(12, 14, 18);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.panSpeed = 0.8;

// Match viewer.html lighting so raw scan ceilings shade the same way.
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const dl1 = new THREE.DirectionalLight(0xffffff, 1.0);
dl1.position.set(10, 20, 10);
dl1.castShadow = true;
dl1.shadow.mapSize.set(2048, 2048);
dl1.shadow.bias = 0.0002;
dl1.shadow.normalBias = 0.25;
dl1.shadow.camera.near = 1;
dl1.shadow.camera.far = 120;
dl1.shadow.camera.left = -40;
dl1.shadow.camera.right = 40;
dl1.shadow.camera.top = 40;
dl1.shadow.camera.bottom = -40;
scene.add(dl1);
const dl2 = new THREE.DirectionalLight(0xffffff, 0.3);
dl2.position.set(-10, 10, -10);
scene.add(dl2);

// grid + axes for orientation
const grid = new THREE.GridHelper(40, 40, 0x444466, 0x222233);
scene.add(grid);
const axes = new THREE.AxesHelper(2);
scene.add(axes);

// LineMaterial instances that need their `resolution` kept in sync with the
// canvas size; registered as each fat line is created.
const lineMaterials = new Set();

function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  for (const m of lineMaterials) m.resolution.set(w, h);
}
window.addEventListener('resize', resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ---- picking ----
const raycaster = new THREE.Raycaster();
raycaster.params.Line = { threshold: 0.05 };
const pointer = new THREE.Vector2();

canvas.addEventListener('click', (ev) => {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const objs = state.clickables.map((c) => c.object);
  const hits = raycaster.intersectObjects(objs, false);
  if (!hits.length) {
    hideInspector();
    return;
  }
  const hit = hits[0];
  const entry = state.clickables.find((c) => c.object === hit.object);
  if (entry) showInspector(entry);
});

const inspectorEl = document.getElementById('inspector');
const inspectorBody = document.getElementById('inspector-body');
document.querySelector('#inspector .close').addEventListener('click', hideInspector);

function hideInspector() {
  inspectorEl.classList.remove('show');
}

function showInspector(entry) {
  inspectorEl.classList.add('show');
  const { kind, payload } = entry;
  let html = '';
  if (kind === 'committed' || kind === 'candidate' || kind === 'flat') {
    const a = payload.audit || {};
    const rows = [
      ['kind', kind],
      ['story', payload.story],
      ['cluster', payload.cluster_index],
      ['avg_incl', fmt(payload.avg_incl, 2)],
      ['avg_azimuth', fmt(payload.avg_azimuth, 2)],
      ['class', a.classification || '(not scored)'],
      ['match_count', a.match_count],
      ['xz_coverage', fmt(a.xz_coverage, 3)],
      ['median_dy', fmt(a.median_dy, 3)],
      ['p95_abs_dy', fmt(a.p95_abs_dy, 3)],
      ['normal_dot', fmt(a.normal_dot_median, 3)],
    ];
    html += `<h3>${kind}[${payload.index}]</h3>`;
    html += `<div class="kv">`;
    for (const [k, v] of rows) {
      if (v === null || v === undefined) continue;
      html += `<div class="k">${k}</div><div class="v">${v}</div>`;
    }
    html += `</div>`;
    if (payload.audit_id) html += `<div class="eid" title="click to copy">${payload.audit_id}</div>`;
    if (a.ceiling_raw_ids && a.ceiling_raw_ids.length) {
      html += `<div style="margin-top:6px;color:#888">matched raw ceilings:</div>`;
      for (const rid of a.ceiling_raw_ids) {
        html += `<div class="eid" title="click to copy">${rid}</div>`;
      }
    }
  } else if (kind === 'raw_ceiling') {
    const r = payload;
    const ys = r.corners.map((c) => c[1]);
    html += `<h3>raw scan ceiling</h3>`;
    html += `<div class="kv">`;
    html += `<div class="k">story</div><div class="v">${r.story}</div>`;
    html += `<div class="k">room</div><div class="v">${r.room_index}</div>`;
    html += `<div class="k">plane</div><div class="v">${r.plane_index}</div>`;
    html += `<div class="k">exposed</div><div class="v">${r.exposed}</div>`;
    html += `<div class="k">y range</div><div class="v">${fmt(Math.min(...ys), 2)} – ${fmt(Math.max(...ys), 2)}</div>`;
    html += `<div class="k">corners</div><div class="v">${r.corners.length}</div>`;
    html += `</div>`;
    if (r.element_id) html += `<div class="eid" title="click to copy">${r.element_id}</div>`;
  } else if (kind === 'segment') {
    const s = payload;
    html += `<h3>segment ${s.id}</h3>`;
    html += `<div class="kv">`;
    html += `<div class="k">cluster</div><div class="v">${s.cluster_index ?? '(unused)'}</div>`;
    html += `<div class="k">incl</div><div class="v">${fmt(s.incl, 2)}</div>`;
    html += `<div class="k">azimuth</div><div class="v">${fmt(s.azimuth, 2)}</div>`;
    html += `<div class="k">len</div><div class="v">${fmt(s.len, 3)}</div>`;
    html += `<div class="k">story</div><div class="v">${s.story}</div>`;
    html += `<div class="k">room</div><div class="v">${s.room_idx}</div>`;
    html += `</div>`;
  }
  inspectorBody.innerHTML = html;
  inspectorBody.querySelectorAll('.eid').forEach((el) =>
    el.addEventListener('click', () => navigator.clipboard.writeText(el.textContent))
  );
}

function fmt(v, n) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return typeof v === 'number' ? v.toFixed(n) : String(v);
}

// ---- building list / sidebar ----
const listEl = document.getElementById('building-list');
const statsEl = document.getElementById('sidebar-stats');
const searchEl = document.getElementById('search');
const filterBtns = [...document.querySelectorAll('#filter-bar button')];

searchEl.addEventListener('input', () => {
  state.search = searchEl.value.toLowerCase();
  renderBuildingList();
});
filterBtns.forEach((b) =>
  b.addEventListener('click', () => {
    filterBtns.forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    state.filter = b.dataset.filter;
    renderBuildingList();
  })
);

async function loadIndex() {
  statsEl.textContent = 'Loading index...';
  try {
    const res = await fetch('/roof-index');
    const data = await res.json();
    state.buildings = data.buildings || [];
  } catch (err) {
    statsEl.textContent = `Error: ${err}`;
    return;
  }
  renderBuildingList();
  const hashUuid = getHashParam('b');
  if (hashUuid && state.buildings.find((b) => b.uuid === hashUuid)) {
    selectBuilding(hashUuid);
  } else if (state.buildings.length) {
    selectBuilding(state.buildings[0].uuid);
  }
}

function renderBuildingList() {
  const rows = state.buildings.filter((b) => matchesFilter(b));
  listEl.innerHTML = '';
  for (const b of rows) {
    const el = document.createElement('div');
    el.className = 'bldg-item';
    if (b.uuid === state.currentUuid) el.classList.add('active');
    const pills = [];
    if (b.n_oblique) pills.push(`<span class="pill ob">${b.n_oblique} obl</span>`);
    if (b.n_weak_committed) pills.push(`<span class="pill weak">${b.n_weak_committed} weak</span>`);
    if (b.n_oblique && !b.n_weak_committed) pills.push(`<span class="pill strong">ok</span>`);
    el.innerHTML = `
      <div class="bldg-addr">${escapeHtml(b.address || '')}</div>
      <div class="bldg-meta">${pills.join('')} <span>${b.n_stories}s·${b.n_rooms}r</span></div>
    `;
    el.addEventListener('click', () => selectBuilding(b.uuid));
    listEl.appendChild(el);
  }
  statsEl.textContent = `${rows.length} / ${state.buildings.length} buildings`;
}

function matchesFilter(b) {
  if (state.filter === 'weak' && !b.n_weak_committed) return false;
  if (state.filter === 'oblique' && !b.n_oblique) return false;
  if (state.search) {
    const hay = `${b.address} ${b.uuid}`.toLowerCase();
    if (!hay.includes(state.search)) return false;
  }
  return true;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---- selection + detail load ----
async function selectBuilding(uuid) {
  state.currentUuid = uuid;
  setHashParam('b', uuid);
  renderBuildingList();
  hideInspector();
  state.hudEl.innerHTML = '<div class="addr">Loading…</div>';
  try {
    const res = await fetch(`/roof-detail?uuid=${encodeURIComponent(uuid)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.detail = await res.json();
  } catch (err) {
    state.hudEl.innerHTML = `<div class="addr">Error</div><div class="stats">${err}</div>`;
    return;
  }
  renderDetail();
}

function getHashParam(key) {
  const h = window.location.hash.replace(/^#/, '');
  if (!h) return null;
  const sp = new URLSearchParams(h);
  return sp.get(key);
}
function setHashParam(key, value) {
  const h = window.location.hash.replace(/^#/, '');
  const sp = new URLSearchParams(h);
  sp.set(key, value);
  window.location.hash = sp.toString();
}

// ---- scene drawing ----
function clearBuildingObjects() {
  for (const obj of state.buildingObjects) {
    scene.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) {
        obj.material.forEach((m) => { lineMaterials.delete(m); m.dispose(); });
      } else {
        lineMaterials.delete(obj.material);
        obj.material.dispose();
      }
    }
  }
  state.buildingObjects = [];
  state.clickables = [];
}

function renderDetail() {
  clearBuildingObjects();
  if (!state.detail) return;
  const d = state.detail;

  // Compute center for camera framing in XZ (Y-up world).
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity, minY = Infinity, maxY = -Infinity;
  const track = (c) => {
    if (c[0] < minX) minX = c[0]; if (c[0] > maxX) maxX = c[0];
    if (c[2] < minZ) minZ = c[2]; if (c[2] > maxZ) maxZ = c[2];
    if (c[1] < minY) minY = c[1]; if (c[1] > maxY) maxY = c[1];
  };
  for (const f of d.floors) for (const c of f.corners) track(c);
  for (const s of d.segments) { track(s.a); track(s.b); }
  for (const p of d.candidates) for (const c of p.corners) track(c);
  for (const p of d.committed) for (const c of p.corners) track(c);

  if (!isFinite(minX)) { minX = maxX = minZ = maxZ = minY = maxY = 0; }
  const cx = (minX + maxX) / 2;
  const cz = (minZ + maxZ) / 2;
  const cy = (minY + maxY) / 2;
  const span = Math.max(maxX - minX, maxZ - minZ, 6);

  // Reframe camera once per selection.
  controls.target.set(cx, cy, cz);
  camera.position.set(cx + span * 0.9, cy + span * 0.8, cz + span * 0.9);

  if (state.toggles.floors) drawFloors(d.floors);
  if (state.toggles.roomCeilings) drawRoomCeilings(d.room_ceilings);
  if (state.toggles.rawCeilings) drawRawCeilings(d.raw_ceilings);
  if (state.toggles.candidates) drawCandidates(d.candidates);
  if (state.toggles.committed) drawCommitted(d.committed);
  if (state.toggles.flat) drawFlat(d.flat);
  if (state.toggles.segments) drawSegments(d.segments);

  // HUD summary.
  const nWeak = d.committed.filter((c) => c.audit && c.audit.classification === 'weak').length;
  const nStrong = d.committed.filter((c) => c.audit && c.audit.classification === 'strong').length;
  state.hudEl.innerHTML = `
    <div class="addr">${escapeHtml(d.address || d.uuid)}</div>
    <div class="stats">
      ${d.clusters.length} clusters · ${d.segments.length} segs ·
      ${d.candidates.length} candidates · ${d.committed.length} committed (${nStrong} strong / ${nWeak} weak) ·
      ${d.flat.length} flat · ${d.raw_ceilings.length} raw ceilings
    </div>
  `;
}

function polygonToGeometry(corners) {
  if (corners.length < 3) return null;
  // Triangulate as a fan around corner[0] — corners are planar (or near-planar).
  const positions = [];
  for (let i = 1; i < corners.length - 1; i++) {
    positions.push(...corners[0], ...corners[i], ...corners[i + 1]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return geo;
}

function polygonEdgesGeometry(corners) {
  const positions = [];
  for (let i = 0; i < corners.length; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    positions.push(...a, ...b);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  return geo;
}

function drawFloors(floors) {
  const color = 0x4488aa;
  for (const f of floors) {
    if (!f.corners || f.corners.length < 3) continue;
    const mesh = createPolygonMesh(f.corners, color, 0.12);
    if (mesh) {
      scene.add(mesh);
      state.buildingObjects.push(mesh);
    }
    const edges = createEdgeLoop(f.corners, color, 0.5);
    scene.add(edges);
    state.buildingObjects.push(edges);
  }
}

function drawRoomCeilings(ceilings) {
  const color = 0x88aabb;
  for (const c of ceilings) {
    if (!c.corners || c.corners.length < 3) continue;
    const mesh = createPolygonMesh(c.corners, color, 0.45);
    if (mesh) {
      scene.add(mesh);
      state.buildingObjects.push(mesh);
    }
    const edges = createEdgeLoop(c.corners, color, 0.8);
    scene.add(edges);
    state.buildingObjects.push(edges);
  }
}

function drawRawCeilings(raws) {
  // Match viewer.html: tan fill at 0.4 opacity, light-tan edges — same call
  // shape used in viewer-main.js:2624.
  for (const r of raws) {
    if (!r.corners || r.corners.length < 3) continue;
    const mesh = createPolygonMesh(r.corners, RAW_CEILING_COLOR, 0.4);
    if (mesh) {
      scene.add(mesh);
      state.buildingObjects.push(mesh);
      state.clickables.push({ object: mesh, kind: 'raw_ceiling', payload: r });
    }
    const edges = createEdgeLoop(r.corners, RAW_CEILING_EDGE);
    scene.add(edges);
    state.buildingObjects.push(edges);
  }
}

function drawCandidates(candidates) {
  for (const p of candidates) {
    if (!p.corners || p.corners.length < 3) continue;
    const fill = polygonToGeometry(p.corners);
    const mat = new THREE.MeshBasicMaterial({
      color: clusterColor(p.cluster_index),
      transparent: true,
      opacity: 0.18,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(fill, mat);
    scene.add(mesh);
    state.buildingObjects.push(mesh);
    state.clickables.push({ object: mesh, kind: 'candidate', payload: p });

    const edges = new THREE.LineSegments(
      polygonEdgesGeometry(p.corners),
      new THREE.LineBasicMaterial({
        color: clusterColor(p.cluster_index),
        transparent: true,
        opacity: 0.7,
      })
    );
    scene.add(edges);
    state.buildingObjects.push(edges);
  }
}

function drawCommitted(committed) {
  for (const p of committed) {
    if (!p.corners || p.corners.length < 3) continue;
    const fill = polygonToGeometry(p.corners);
    const cls = p.audit && p.audit.classification;
    const color = state.toggles.colorByClass ? classColor(cls) : 0x88ccff;
    const mat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(fill, mat);
    scene.add(mesh);
    state.buildingObjects.push(mesh);
    state.clickables.push({ object: mesh, kind: 'committed', payload: p });

    const edges = new THREE.LineSegments(
      polygonEdgesGeometry(p.corners),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 })
    );
    scene.add(edges);
    state.buildingObjects.push(edges);
  }
}

function drawFlat(flats) {
  for (const p of flats) {
    if (!p.corners || p.corners.length < 3) continue;
    const fill = polygonToGeometry(p.corners);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xcc66ff,
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(fill, mat);
    scene.add(mesh);
    state.buildingObjects.push(mesh);
    state.clickables.push({ object: mesh, kind: 'flat', payload: p });
  }
}

function drawSegments(segments) {
  const showUnused = state.toggles.segUnused;
  // Group by color so each cluster gets one fat-line mesh.
  const groups = new Map();  // color -> {positions: number[], refs: segment[]}
  for (const s of segments) {
    if (!s.used && !showUnused) continue;
    const color = s.used ? clusterColor(s.cluster_index) : 0x555555;
    if (!groups.has(color)) groups.set(color, { positions: [], refs: [] });
    const g = groups.get(color);
    g.positions.push(...s.a, ...s.b);
    g.refs.push(s);
  }
  const w = canvas.clientWidth || window.innerWidth;
  const h = canvas.clientHeight || window.innerHeight;
  for (const [color, g] of groups) {
    const geo = new LineSegmentsGeometry();
    geo.setPositions(g.positions);
    const mat = new LineMaterial({
      color,
      linewidth: 4,          // pixels in screen space
      transparent: true,
      opacity: 0.95,
      depthTest: true,
    });
    mat.resolution.set(w, h);
    lineMaterials.add(mat);
    const line = new LineSegments2(geo, mat);
    line.computeLineDistances();
    scene.add(line);
    state.buildingObjects.push(line);

    for (const s of g.refs) {
      // Endpoint markers make the 3D drop visible even at steep camera angles.
      for (const pt of [s.a, s.b]) {
        const sph = new THREE.Mesh(
          new THREE.SphereGeometry(0.06, 6, 6),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 })
        );
        sph.position.set(pt[0], pt[1], pt[2]);
        scene.add(sph);
        state.buildingObjects.push(sph);
      }
      // Midpoint sphere acts as the segment's click target for the inspector.
      const mid = [
        (s.a[0] + s.b[0]) / 2,
        (s.a[1] + s.b[1]) / 2,
        (s.a[2] + s.b[2]) / 2,
      ];
      const pick = new THREE.Mesh(
        new THREE.SphereGeometry(0.1, 8, 8),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.0, depthTest: false })
      );
      pick.position.set(mid[0], mid[1], mid[2]);
      scene.add(pick);
      state.buildingObjects.push(pick);
      state.clickables.push({ object: pick, kind: 'segment', payload: s });
    }
  }
}

// ---- toggle wiring ----
const toggleMap = {
  't-segments': 'segments',
  't-seg-unused': 'segUnused',
  't-candidates': 'candidates',
  't-committed': 'committed',
  't-flat': 'flat',
  't-raw-ceilings': 'rawCeilings',
  't-room-ceilings': 'roomCeilings',
  't-floors': 'floors',
  't-color-by-class': 'colorByClass',
};
for (const [elId, key] of Object.entries(toggleMap)) {
  const el = document.getElementById(elId);
  if (!el) continue;
  el.checked = !!state.toggles[key];
  el.addEventListener('change', () => {
    state.toggles[key] = el.checked;
    renderDetail();
  });
}

// ---- boot ----
loadIndex();
