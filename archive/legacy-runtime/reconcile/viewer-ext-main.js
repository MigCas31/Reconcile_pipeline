// Standalone viewer for reconcile_ext diagnostic overlays.
// Loads buildings_3d.json (geometry) + reconcile_ext_results.json (overlays).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA = { buildings: null, ext: null };
let current = null;

const canvas = document.getElementById('c');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a12);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.1;

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(30, 60, 20);
scene.add(dir);

scene.add(new THREE.GridHelper(60, 60, 0x223, 0x112));

const LAYERS = {
  rooms: new THREE.Group(),
  walls: new THREE.Group(),
  footprint: new THREE.Group(),
  parts: new THREE.Group(),
  reflex: new THREE.Group(),
  disc: new THREE.Group(),
  steps: new THREE.Group(),
  delta: new THREE.Group(),
  'seam-strong': new THREE.Group(),
  'seam-med': new THREE.Group(),
  'seam-weak': new THREE.Group(),
  units: new THREE.Group(),
  cuts: new THREE.Group(),
};

// Unit palette (role+index → hex). Main green; extensions cycle through warm/cool accents.
const UNIT_COLORS = [0x6cd0a8, 0xe0b356, 0x8bb8ff, 0xe06eb6, 0xf08a6f, 0x9be066];
function unitColor(index) {
  return UNIT_COLORS[Math.min(index, UNIT_COLORS.length - 1)];
}
for (const g of Object.values(LAYERS)) scene.add(g);

function onResize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);

function clearGroup(g) {
  while (g.children.length) {
    const c = g.children.pop();
    if (c.geometry) c.geometry.dispose();
    if (c.material) {
      const mats = Array.isArray(c.material) ? c.material : [c.material];
      for (const m of mats) m.dispose();
    }
  }
}

function clearAll() {
  for (const g of Object.values(LAYERS)) clearGroup(g);
}

// --- data loading --------------------------------------------------

async function loadData() {
  const [b, e] = await Promise.all([
    fetch('buildings_3d.json').then(r => r.json()),
    fetch('reconcile_ext_results.json').then(r => r.json()).catch(() => []),
  ]);
  DATA.buildings = b;
  DATA.ext = e;
  renderList();
}

function extByUuid(uuid) {
  if (!DATA.ext) return null;
  return DATA.ext.find(x => x.building_uuid === uuid) || null;
}

// --- sidebar -------------------------------------------------------

function renderList() {
  const list = document.getElementById('building-list');
  const stats = document.getElementById('sidebar-stats');
  const q = (document.getElementById('search').value || '').toLowerCase();

  const rows = [];
  for (const b of DATA.buildings) {
    const addr = (b.address || '').toLowerCase();
    const uuid = b.uuid || '';
    if (q && !addr.includes(q) && !uuid.toLowerCase().includes(q)) continue;
    const ext = extByUuid(uuid);
    rows.push({ b, ext });
  }

  // Sort: buildings with ext signals first, then those without.
  rows.sort((a, b) => {
    const sa = scoreRow(a), sb = scoreRow(b);
    return sb - sa;
  });

  const withExt = rows.filter(r => r.ext).length;
  stats.textContent = `${rows.length} buildings · ${withExt} with ext results`;

  list.innerHTML = '';
  for (const { b, ext } of rows) {
    const item = document.createElement('div');
    item.className = 'bldg-item' + (current && current.uuid === b.uuid ? ' active' : '');
    item.dataset.uuid = b.uuid;

    const addr = document.createElement('div');
    addr.className = 'bldg-addr';
    addr.textContent = b.address || b.uuid;
    item.appendChild(addr);

    const meta = document.createElement('div');
    meta.className = 'bldg-meta';
    const s = ext?.stats;
    if (s) {
      if (s.reflex_count) meta.appendChild(pill('reflex', `R${s.reflex_count}`));
      if (s.discontinuity_count) meta.appendChild(pill('disc', `D${s.discontinuity_count}`));
      if (s.wall_step_count) meta.appendChild(pill('step', `W${s.wall_step_count}`));
      if (s.height_delta_count) meta.appendChild(pill('delta', `H${s.height_delta_count}`));
      if (s.seam_strong) meta.appendChild(pill('seam-strong', `S${s.seam_strong}`));
      if (s.seam_medium) meta.appendChild(pill('seam-med', `M${s.seam_medium}`));
      if (s.seam_weak) meta.appendChild(pill('seam-weak', `w${s.seam_weak}`));
      if (!meta.children.length) {
        const span = document.createElement('span');
        span.style.color = '#444';
        span.textContent = 'no signals';
        meta.appendChild(span);
      }
    } else {
      const span = document.createElement('span');
      span.style.color = '#333';
      span.textContent = '(no ext run)';
      meta.appendChild(span);
    }
    item.appendChild(meta);

    item.addEventListener('click', () => selectBuilding(b.uuid));
    list.appendChild(item);
  }
}

function scoreRow({ ext }) {
  if (!ext) return -1;
  const s = ext.stats || {};
  return (s.seam_strong || 0) * 4 + (s.seam_medium || 0) * 2 + (s.reflex_count || 0);
}

function pill(kind, text) {
  const s = document.createElement('span');
  s.className = 'pill ' + kind;
  s.textContent = text;
  return s;
}

document.getElementById('search').addEventListener('input', renderList);

// --- rendering primitives ------------------------------------------

function makePolyline(points, color) {
  const geom = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geom, new THREE.LineBasicMaterial({ color }));
}

function makeDot(x, y, z, color, radius = 0.12) {
  const g = new THREE.SphereGeometry(radius, 12, 8);
  const m = new THREE.MeshBasicMaterial({ color });
  const mesh = new THREE.Mesh(g, m);
  mesh.position.set(x, y, z);
  return mesh;
}

// --- main render ---------------------------------------------------

function selectBuilding(uuid) {
  const b = DATA.buildings.find(x => x.uuid === uuid);
  if (!b) return;
  current = b;
  renderList();
  render(b, extByUuid(uuid));
}

function render(b, ext) {
  clearAll();

  // Centre at XZ centroid of all corners for a stable frame.
  const pts = [];
  for (const r of b.rooms || []) {
    for (const w of (r.walls_merged || [])) for (const c of (w.corners || [])) pts.push(c);
    for (const w of (r.walls_computed || [])) for (const c of (w.corners || [])) pts.push(c);
    for (const f of (r.floor_polygon || [])) pts.push(f);
  }
  if (!pts.length) return;
  let cx = 0, cz = 0;
  for (const p of pts) { cx += p[0]; cz += p[2]; }
  cx /= pts.length; cz /= pts.length;

  const OFFSET = { x: -cx, z: -cz };

  // Rooms: semi-transparent floor polygons + outlines.
  for (const r of b.rooms || []) {
    const fp = r.floor_polygon || [];
    if (fp.length >= 3) {
      const shape = new THREE.Shape(fp.map(p => new THREE.Vector2(p[0] + OFFSET.x, p[2] + OFFSET.z)));
      const geom = new THREE.ShapeGeometry(shape);
      const mat = new THREE.MeshBasicMaterial({ color: 0x334, transparent: true, opacity: 0.35, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geom, mat);
      // ShapeGeometry lives in XY; rotate +90° around X so shape-space Y maps to world +Z (not -Z).
      mesh.rotation.x = Math.PI / 2;
      mesh.position.y = fp[0][1];
      LAYERS.rooms.add(mesh);

      const outlinePts = fp.concat([fp[0]]).map(p => new THREE.Vector3(p[0] + OFFSET.x, p[1], p[2] + OFFSET.z));
      LAYERS.rooms.add(makePolyline(outlinePts, 0x667788));
    }
  }

  // Room → unit colour lookup so wall outlines can be tinted by membership.
  const roomColor = new Map();
  for (const u of (ext?.units || [])) {
    const c = unitColor(u.index ?? 0);
    for (const rid of (u.room_ids || [])) roomColor.set(rid, c);
  }

  // Walls — default layer (neutral blue) + parallel "units" layer (tinted by unit).
  for (const r of b.rooms || []) {
    const tint = roomColor.get(r.id);
    for (const w of (r.walls_merged || [])) {
      drawWall(w, 0x6677aa, OFFSET, LAYERS.walls);
      if (tint !== undefined) drawWall(w, tint, OFFSET, LAYERS.units);
    }
    for (const w of (r.walls_computed || [])) {
      drawWall(w, 0x445577, OFFSET, LAYERS.walls);
      if (tint !== undefined) drawWall(w, tint, OFFSET, LAYERS.units);
    }
  }

  // Footprint + parts (from ext).
  if (ext) {
    if (ext.building_footprint_xz?.length >= 3) {
      const pts = ext.building_footprint_xz.concat([ext.building_footprint_xz[0]])
        .map(p => new THREE.Vector3(p[0] + OFFSET.x, p[1] ?? 0, p[2] + OFFSET.z));
      LAYERS.footprint.add(makePolyline(pts, 0x88bbff));
    }
    for (const fp of (ext.parts_footprints_xz || [])) {
      if (fp.length < 3) continue;
      const pts = fp.concat([fp[0]]).map(p => new THREE.Vector3(p[0] + OFFSET.x, p[1] ?? 0, p[2] + OFFSET.z));
      LAYERS.parts.add(makePolyline(pts, 0xffaa44));
    }

    drawReflexVertices(ext.reflex_vertices || [], OFFSET);
    drawDiscontinuities(ext.discontinuities || [], OFFSET);
    drawWallSteps(ext.wall_steps || [], OFFSET);
    drawHeightDeltas(ext.height_deltas || [], OFFSET);
    drawSeams(ext.seam_hypotheses || [], OFFSET);
    drawUnits(ext.units || [], OFFSET);
    drawCutLines(ext.seam_cut_lines || [], OFFSET);
  }

  // Fit camera.
  const bbox = new THREE.Box3();
  for (const g of Object.values(LAYERS)) bbox.expandByObject(g);
  if (!bbox.isEmpty()) {
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const centre = new THREE.Vector3();
    bbox.getCenter(centre);
    const diag = Math.max(size.x, size.z, size.y) * 1.8 + 5;
    camera.position.set(centre.x, centre.y + diag * 0.7, centre.z + diag * 0.7);
    camera.lookAt(centre);
    controls.target.copy(centre);
    controls.update();
  }

  onResize();
  updateHud(b, ext);
}

function drawWall(w, color, OFFSET, target) {
  const c = w.corners || [];
  if (c.length < 3) return;
  const pts = c.concat([c[0]]).map(p => new THREE.Vector3(p[0] + OFFSET.x, p[1], p[2] + OFFSET.z));
  target.add(makePolyline(pts, color));
}

// --- overlays ------------------------------------------------------

function drawReflexVertices(list, OFFSET) {
  for (const v of list) {
    const x = v.x + OFFSET.x;
    const z = v.z + OFFSET.z;
    const y = v.y ?? 0;
    const dot = makeDot(x, y + 0.05, z, 0xc8a8ff, 0.22);
    dot.userData = { kind: 'reflex', id: v.id, info: `reflex ${v.id} — notch ${v.notch_depth_m?.toFixed(2)}m` };
    LAYERS.reflex.add(dot);
    LAYERS.reflex.add(makePolyline([
      new THREE.Vector3(x, y, z),
      new THREE.Vector3(x, y + 4, z),
    ], 0xc8a8ff));
  }
}

function drawDiscontinuities(list, OFFSET) {
  for (const d of list) {
    const [x, z] = d.mid_xz || [0, 0];
    const X = x + OFFSET.x, Z = z + OFFSET.z;
    const dot = makeDot(X, 3, Z, 0xffc888, 0.18);
    dot.userData = { kind: 'disc', id: d.id, info: `${d.kind} Δ=${d.delta?.toFixed(2)} strength=${d.strength}` };
    LAYERS.disc.add(dot);
    LAYERS.disc.add(makePolyline([
      new THREE.Vector3(X, 0, Z),
      new THREE.Vector3(X, 6, Z),
    ], 0xffc888));
  }
}

function drawWallSteps(list, OFFSET) {
  for (const w of list) {
    const [x, z] = w.anchor_xz || [0, 0];
    const X = x + OFFSET.x, Z = z + OFFSET.z;
    const dot = makeDot(X, 1.5, Z, 0x88ffc8, 0.14);
    dot.userData = { kind: 'wall-step', id: w.id, info: `wall-step offset=${w.offset_m?.toFixed(2)} overlap=${w.overlap_m?.toFixed(2)}` };
    LAYERS.steps.add(dot);
  }
}

function drawHeightDeltas(list, OFFSET) {
  for (const h of list) {
    const [x, z] = h.anchor_xz || [0, 0];
    const X = x + OFFSET.x, Z = z + OFFSET.z;
    const y = ((h.y_a ?? 0) + (h.y_b ?? 0)) / 2;
    const dot = makeDot(X, y, Z, 0xff88a8, 0.17);
    dot.userData = { kind: h.kind, id: h.id, info: `${h.kind} Δ=${h.delta_m?.toFixed(2)}m ${h.room_a_id}↔${h.room_b_id}` };
    LAYERS.delta.add(dot);
  }
}

function drawUnits(list, OFFSET) {
  for (const u of list) {
    const fp = u.footprint_xz || [];
    if (fp.length < 3) continue;
    const colour = unitColor(u.index ?? 0);
    const y0 = fp[0][1] ?? 0;

    // Outline at floor level + a second copy slightly lifted — two overlaid
    // polylines read as a thicker ribbon in WebGL (LineBasicMaterial.linewidth
    // is hardcoded to 1 on most platforms).
    for (const yOffset of [0.04, 0.10]) {
      const ring = fp.concat([fp[0]])
        .map(p => new THREE.Vector3(p[0] + OFFSET.x, (p[1] ?? 0) + yOffset, p[2] + OFFSET.z));
      const line = makePolyline(ring, colour);
      line.userData = { kind: 'unit', id: u.id, info: u.rationale || u.id };
      LAYERS.units.add(line);
    }

    const shape = new THREE.Shape(fp.map(p => new THREE.Vector2(p[0] + OFFSET.x, p[2] + OFFSET.z)));
    const geom = new THREE.ShapeGeometry(shape);
    const mat = new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.28, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geom, mat);
    // +π/2 (not −π/2) so shape-space Y maps to world +Z; otherwise the fill
    // is mirrored across the X axis relative to the outline polylines.
    mesh.rotation.x = Math.PI / 2;
    mesh.position.y = y0 + 0.02;
    mesh.userData = { kind: 'unit', id: u.id, info: u.rationale || u.id };
    LAYERS.units.add(mesh);
  }
}

function drawCutLines(list, OFFSET) {
  for (const entry of list) {
    const [seg, seamId] = entry;
    if (!seg || seg.length < 2) continue;
    for (const yOffset of [0.12, 0.20]) {
      const pts = seg.map(p => new THREE.Vector3(p[0] + OFFSET.x, (p[1] ?? 0) + yOffset, p[2] + OFFSET.z));
      const line = makePolyline(pts, 0xffffff);
      line.userData = { kind: 'cut', id: seamId, info: `cut for ${seamId}` };
      LAYERS.cuts.add(line);
    }
  }
}

function drawSeams(list, OFFSET) {
  const colors = { strong: 0xff4444, medium: 0xff9944, weak: 0xdddd44 };
  const targets = { strong: LAYERS['seam-strong'], medium: LAYERS['seam-med'], weak: LAYERS['seam-weak'] };
  for (const s of list) {
    const [x, z] = s.anchor_xz || [0, 0];
    const X = x + OFFSET.x, Z = z + OFFSET.z;
    const colour = colors[s.confidence] || 0xdddd44;
    const tgt = targets[s.confidence] || LAYERS['seam-weak'];
    const dot = makeDot(X, 0.1, Z, colour, 0.32);
    dot.userData = {
      kind: 'seam',
      id: s.id,
      info: `seam ${s.confidence} score=${s.composite_score?.toFixed(2)} — ${s.rationale}`,
    };
    tgt.add(dot);
    tgt.add(makePolyline([
      new THREE.Vector3(X, 0, Z),
      new THREE.Vector3(X, 8, Z),
    ], colour));
  }
}

// --- hud + interaction ---------------------------------------------

function updateHud(b, ext) {
  const hud = document.getElementById('hud');
  const s = ext?.stats;
  const addr = b.address || b.uuid;
  const statsLine = s
    ? `reflex=${s.reflex_count} roof-disc=${s.discontinuity_count} wall-steps=${s.wall_step_count} height-deltas=${s.height_delta_count} · seams ${s.seam_strong}/${s.seam_medium}/${s.seam_weak} (s/m/w) · units=${s.unit_count ?? 0} (ext=${s.extension_count ?? 0})`
    : `(no ext_results for this building — run python -m reconcile_ext --uuid ${b.uuid})`;
  hud.innerHTML = `<div class="addr">${escapeHtml(addr)}</div>
    <div style="color:#888">${escapeHtml(b.uuid)}</div>
    <div class="stats">${escapeHtml(statsLine)}</div>`;
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(Object.values(LAYERS), true);
  const hit = hits.find(h => h.object.userData && h.object.userData.info);
  if (hit) {
    const { id, info } = hit.object.userData;
    try { navigator.clipboard.writeText(id); } catch {}
    const hud = document.getElementById('hud');
    const extra = document.createElement('div');
    extra.style.color = '#8cf';
    extra.style.marginTop = '6px';
    extra.innerHTML = `📋 ${escapeHtml(id)}<br/><span style="color:#bbb">${escapeHtml(info)}</span>`;
    hud.appendChild(extra);
  }
});

// --- layer toggles -------------------------------------------------

const TOGGLES = {
  't-rooms': 'rooms',
  't-walls': 'walls',
  't-footprint': 'footprint',
  't-parts': 'parts',
  't-reflex': 'reflex',
  't-disc': 'disc',
  't-steps': 'steps',
  't-delta': 'delta',
  't-seam-strong': 'seam-strong',
  't-seam-med': 'seam-med',
  't-seam-weak': 'seam-weak',
  't-units': 'units',
  't-cuts': 'cuts',
};
for (const [id, layer] of Object.entries(TOGGLES)) {
  const el = document.getElementById(id);
  if (!el) continue;
  LAYERS[layer].visible = el.checked;
  el.addEventListener('change', () => { LAYERS[layer].visible = el.checked; });
}

// --- main loop + bootstrap -----------------------------------------

function tick() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

onResize();
tick();

loadData().then(() => {
  let pick = null;
  if (DATA.ext && DATA.ext.length) {
    pick = DATA.ext.find(e => {
      const s = e.stats || {};
      return s.reflex_count || s.seam_strong || s.seam_medium;
    });
  }
  const uuid = pick?.building_uuid || DATA.buildings[0]?.uuid;
  if (uuid) selectBuilding(uuid);
});
