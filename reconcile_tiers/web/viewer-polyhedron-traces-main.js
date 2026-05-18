import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DEFAULT_INDEX_URL =
  "../../.context/polyhedron-envelope-roof-selection-safe-check/index.json";
const params = new URLSearchParams(window.location.search);
const INDEX_URL = params.get("index") || DEFAULT_INDEX_URL;
const TRACE_ROOT = params.get("root") || traceRootForIndex(INDEX_URL);

function traceRootForIndex(indexUrl) {
  const slashIndex = indexUrl.lastIndexOf("/");
  return slashIndex >= 0 ? indexUrl.slice(0, slashIndex) : ".";
}

const viewport = document.querySelector("#viewport");
const canvas = document.querySelector("#view");
const traceList = document.querySelector("#trace-list");
const search = document.querySelector("#search");
const sidebarStats = document.querySelector("#sidebar-stats");
const currentTitle = document.querySelector("#current-title");
const currentMeta = document.querySelector("#current-meta");
const pill = document.querySelector("#pill");
const status = document.querySelector("#status");
const frameSlider = document.querySelector("#frame-slider");
const frameReadout = document.querySelector("#frame-readout");
const prevTraceButton = document.querySelector("#prev-trace");
const nextTraceButton = document.querySelector("#next-trace");
const prevFrameButton = document.querySelector("#prev-frame");
const nextFrameButton = document.querySelector("#next-frame");
const edgesToggle = document.querySelector("#edges-toggle");
const ghostToggle = document.querySelector("#ghost-toggle");
const stepTitle = document.querySelector("#step-title");
const stepCounts = document.querySelector("#step-counts");
const stepBody = document.querySelector("#step-body");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf6f7f8);

const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 1000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.addEventListener("change", requestRender);

scene.add(new THREE.HemisphereLight(0xffffff, 0xb8c0cc, 1.8));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(3, 7, 5);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xdde9ff, 0.9);
fillLight.position.set(-6, 3, -4);
scene.add(fillLight);

const modelGroup = new THREE.Group();
scene.add(modelGroup);

const facePalette = [
  0x3b82f6, 0x16a34a, 0xf97316, 0xa855f7, 0x0f766e, 0xe11d48,
  0x4f46e5, 0xca8a04, 0x0891b2, 0x65a30d, 0xc026d3, 0xea580c,
];

const materials = new Map();
const edgeMaterial = new THREE.LineBasicMaterial({
  color: 0x17202a,
  transparent: true,
  opacity: 0.72,
});
const ghostMaterial = new THREE.MeshBasicMaterial({
  color: 0x64748b,
  transparent: true,
  opacity: 0.16,
  side: THREE.DoubleSide,
  depthWrite: false,
});
const highlightMaterial = new THREE.MeshPhongMaterial({
  color: 0xfacc15,
  emissive: 0x4d3d00,
  shininess: 20,
  transparent: true,
  opacity: 0.92,
  side: THREE.DoubleSide,
});

let indexData = null;
let rows = [];
let visibleRows = [];
let activeRow = null;
let activeTrace = null;
let frameIndex = 0;
let renderQueued = false;

function materialForFace(faceId) {
  if (!materials.has(faceId)) {
    const color = facePalette[Math.abs(Number(faceId)) % facePalette.length];
    materials.set(
      faceId,
      new THREE.MeshPhongMaterial({
        color,
        shininess: 18,
        transparent: true,
        opacity: 0.72,
        side: THREE.DoubleSide,
      }),
    );
  }
  return materials.get(faceId);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatCounts(counts) {
  return `${counts.faces}F ${counts.vertices}V ${counts.half_edges}HE`;
}

function matches(row, query) {
  if (!query) return true;
  return `${row.uuid} ${recordLabel(row)} ${row.stop_reason}`
    .toLowerCase()
    .includes(query);
}

function recordLabel(row) {
  if (row.room_index !== undefined && row.room_index !== null) {
    return `room ${row.room_index}`;
  }
  return row.locator_id || `part ${row.part_index}`;
}

function renderTraceList() {
  const query = search.value.trim().toLowerCase();
  visibleRows = rows.filter((row) => matches(row, query));
  traceList.innerHTML = visibleRows
    .map((row) => {
      const active = row === activeRow ? " active" : "";
      return `
        <button class="trace-row${active}" data-trace="${escapeHtml(row.trace)}">
          <span class="label">${escapeHtml(row.uuid)}</span>
          <span class="meta">
            <span class="badge">${escapeHtml(recordLabel(row))}</span>
            <span class="badge">${row.step_count} steps</span>
            <span class="badge">${formatCounts(row.initial_counts)}</span>
          </span>
        </button>
      `;
    })
    .join("");

  for (const button of traceList.querySelectorAll(".trace-row")) {
    button.addEventListener("click", () => {
      const row = rows.find((candidate) => candidate.trace === button.dataset.trace);
      if (row) void loadTrace(row);
    });
  }
}

async function loadTrace(row) {
  activeRow = row;
  frameIndex = 0;
  renderTraceList();
  status.textContent = "Loading trace...";
  const response = await fetch(`${TRACE_ROOT}/${row.trace}`);
  if (!response.ok) throw new Error(`trace fetch failed: ${response.status}`);
  activeTrace = await response.json();
  frameSlider.min = "0";
  frameSlider.max = String(Math.max(0, activeTrace.frames.length - 1));
  frameSlider.value = "0";
  renderFrame();
  focusModel();
}

function currentStepForFrame() {
  if (!activeTrace) return null;
  if (frameIndex <= 0) return activeTrace.steps[0] ?? null;
  return activeTrace.steps.find((step) => step.after_frame === frameIndex) ?? null;
}

function triggerFaceIds(step) {
  if (!step || step.action !== "adjacent_coplanar_face_merge") return new Set();
  return new Set(step.trigger_ids.map((id) => Number(id)));
}

function renderFrame() {
  clearGroup(modelGroup);
  if (!activeTrace || !activeRow) return;

  const frame = activeTrace.frames[frameIndex];
  const previousFrame =
    ghostToggle.checked && frameIndex > 0 ? activeTrace.frames[frameIndex - 1] : null;
  const step = currentStepForFrame();
  const highlightFaces = triggerFaceIds(step);

  if (previousFrame) {
    const ghostGroup = new THREE.Group();
    ghostGroup.name = "previousFrame";
    addFrameMeshes(ghostGroup, previousFrame, {
      ghost: true,
      highlightFaces: new Set(),
    });
    modelGroup.add(ghostGroup);
  }

  addFrameMeshes(modelGroup, frame, {
    ghost: false,
    highlightFaces,
  });

  updateHeader(frame, step);
  requestRender();
}

function addFrameMeshes(group, frame, { ghost, highlightFaces }) {
  for (const face of frame.faces || []) {
    if (!Array.isArray(face.corners) || face.corners.length < 3) continue;
    const geometry = polygonGeometry(face.corners);
    if (!geometry) continue;
    const isHighlighted = highlightFaces.has(Number(face.id));
    const material = ghost
      ? ghostMaterial
      : isHighlighted
        ? highlightMaterial
        : materialForFace(face.id);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.faceId = face.id;
    group.add(mesh);

    if (edgesToggle.checked && !ghost) {
      const edge = edgeLoop(face.corners);
      if (edge) group.add(edge);
    }
  }
}

function updateHeader(frame, step) {
  const stepCount = activeTrace.steps.length;
  currentTitle.textContent = `${activeRow.uuid} · ${recordLabel(activeRow)}`;
  currentMeta.textContent = `${stepCount} topology steps · ${activeTrace.stop.reason}`;
  pill.textContent = `Frame ${frameIndex + 1}/${activeTrace.frames.length}`;
  frameReadout.textContent = `${frameIndex} / ${activeTrace.frames.length - 1}`;
  stepTitle.textContent = step ? step.action.replaceAll("_", " ") : "Initial frame";
  stepCounts.textContent = formatCounts(frame.counts);
  status.textContent = `${frame.faces.length} faces rendered`;

  if (!step) {
    stepBody.innerHTML = `<div class="delta-line"><span>state</span><span>${escapeHtml(frame.label)}</span></div>`;
    return;
  }

  const before = step.before_counts;
  const after = step.after_counts;
  stepBody.innerHTML = `
    <div class="delta-line"><span>trigger</span><span>${escapeHtml(step.trigger_ids.join(", "))}</span></div>
    <div class="delta-line"><span>faces</span><span>${before.faces} → ${after.faces}</span></div>
    <div class="delta-line"><span>vertices</span><span>${before.vertices} → ${after.vertices}</span></div>
    <div class="delta-line"><span>half-edges</span><span>${before.half_edges} → ${after.half_edges}</span></div>
  `;
}

function polygonGeometry(corners) {
  const frame = projectionFrame(corners);
  if (!frame) return null;
  const points2 = corners.map((corner) => {
    const rel = vec(corner).sub(frame.origin);
    return new THREE.Vector2(rel.dot(frame.u), rel.dot(frame.v));
  });
  let triangles = [];
  try {
    triangles = THREE.ShapeUtils.triangulateShape(points2, []);
  } catch (_err) {
    return null;
  }
  if (!triangles.length) return null;

  const positions = new Float32Array(corners.length * 3);
  corners.forEach((corner, index) => {
    positions[index * 3] = corner[0];
    positions[index * 3 + 1] = corner[1];
    positions[index * 3 + 2] = corner[2];
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(triangles.flat());
  geometry.computeVertexNormals();
  return geometry;
}

function edgeLoop(corners) {
  const points = [...corners, corners[0]].map((corner) => vec(corner));
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geometry, edgeMaterial);
}

function projectionFrame(corners) {
  if (!Array.isArray(corners) || corners.length < 3) return null;
  const origin = vec(corners[0]);
  const normal = newellNormal(corners);
  let u = null;
  for (let i = 1; i < corners.length; i += 1) {
    const edge = vec(corners[i]).sub(origin);
    if (edge.lengthSq() > 1e-10) {
      u = edge.normalize();
      break;
    }
  }
  if (!u) return null;
  const projected = u.clone().sub(normal.clone().multiplyScalar(u.dot(normal)));
  if (projected.lengthSq() > 1e-10) u = projected.normalize();
  const v = new THREE.Vector3().crossVectors(normal, u).normalize();
  return { origin, u, v };
}

function newellNormal(corners) {
  const normal = new THREE.Vector3();
  for (let i = 0; i < corners.length; i += 1) {
    const a = vec(corners[i]);
    const b = vec(corners[(i + 1) % corners.length]);
    normal.x += (a.y - b.y) * (a.z + b.z);
    normal.y += (a.z - b.z) * (a.x + b.x);
    normal.z += (a.x - b.x) * (a.y + b.y);
  }
  if (normal.lengthSq() <= 1e-10) return new THREE.Vector3(0, 1, 0);
  return normal.normalize();
}

function vec(corner) {
  return new THREE.Vector3(Number(corner[0]), Number(corner[1]), Number(corner[2]));
}

function clearGroup(group) {
  for (const child of group.children) {
    child.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
    });
  }
  group.clear();
}

function focusModel() {
  const box = new THREE.Box3().setFromObject(modelGroup);
  if (box.isEmpty()) {
    camera.position.set(4, 3, 5);
    controls.target.set(0, 0, 0);
    return;
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 1);
  camera.position.copy(center).add(new THREE.Vector3(radius * 1.3, radius * 1.0, radius * 1.45));
  camera.near = Math.max(0.01, radius / 200);
  camera.far = Math.max(100, radius * 20);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
  requestRender();
}

function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    controls.update();
    renderer.render(scene, camera);
  });
}

function resize() {
  const rect = viewport.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  requestRender();
}

function moveTrace(delta) {
  if (!visibleRows.length || !activeRow) return;
  const index = visibleRows.indexOf(activeRow);
  const next = visibleRows[(index + delta + visibleRows.length) % visibleRows.length];
  void loadTrace(next);
}

function moveFrame(delta) {
  if (!activeTrace) return;
  const max = activeTrace.frames.length - 1;
  frameIndex = Math.max(0, Math.min(max, frameIndex + delta));
  frameSlider.value = String(frameIndex);
  renderFrame();
}

async function init() {
  const response = await fetch(INDEX_URL);
  if (!response.ok) throw new Error(`index fetch failed: ${response.status}`);
  indexData = await response.json();
  rows = (indexData.records || [])
    .filter((row) => row.trace)
    .sort((a, b) => b.step_count - a.step_count || a.uuid.localeCompare(b.uuid));
  const builtCount =
    indexData.summary.built_parts ?? indexData.summary.built_rooms ?? rows.length;
  const domain = indexData.domain || "room-shell";
  sidebarStats.textContent = `${rows.length} traces · ${builtCount} built ${domain}s`;
  renderTraceList();
  if (rows.length) await loadTrace(rows[0]);
  else status.textContent = "No traces found.";
}

search.addEventListener("input", () => {
  renderTraceList();
});
prevTraceButton.addEventListener("click", () => moveTrace(-1));
nextTraceButton.addEventListener("click", () => moveTrace(1));
prevFrameButton.addEventListener("click", () => moveFrame(-1));
nextFrameButton.addEventListener("click", () => moveFrame(1));
frameSlider.addEventListener("input", () => {
  frameIndex = Number(frameSlider.value);
  renderFrame();
});
edgesToggle.addEventListener("change", renderFrame);
ghostToggle.addEventListener("change", renderFrame);
window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(viewport);

resize();
init().catch((err) => {
  console.error(err);
  status.textContent = String(err.message || err);
});
