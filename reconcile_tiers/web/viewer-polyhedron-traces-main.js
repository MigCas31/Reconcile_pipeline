import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { populateBuildingScene } from "./tier-preview.js";

const DEFAULT_INDEX_URL =
  "../../.context/polyhedron-envelope-roof-selection-safe-check/index.json";
const PIPELINE_ROOT = "../../pipeline-outputs";
const CONTEXT_OPACITY = 0.22;

const params = new URLSearchParams(window.location.search);
const INDEX_URL = params.get("index") || DEFAULT_INDEX_URL;
const TRACE_ROOT = params.get("root") || traceRootForIndex(INDEX_URL);

function traceRootForIndex(indexUrl) {
  const slashIndex = indexUrl.lastIndexOf("/");
  return slashIndex >= 0 ? indexUrl.slice(0, slashIndex) : ".";
}

const viewport = document.querySelector("#viewport");
const canvas = document.querySelector("#view");
const buildingList = document.querySelector("#building-list");
const roomPanel = document.querySelector("#room-panel");
const roomList = document.querySelector("#room-list");
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
const fullBuildingToggle = document.querySelector("#full-building-toggle");
const openOriginal = document.querySelector("#open-original");
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
const contextBuildingGroup = new THREE.Group();
contextBuildingGroup.name = "contextBuilding";
const traceGroup = new THREE.Group();
traceGroup.name = "trace";
modelGroup.add(contextBuildingGroup);
modelGroup.add(traceGroup);
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
const footprintEdgeMaterial = new THREE.LineBasicMaterial({
  color: 0xc026d3,
  transparent: true,
  opacity: 0.95,
});
const coherenceEdgeMaterial = new THREE.LineBasicMaterial({
  color: 0xdb2777,
  transparent: true,
  opacity: 0.9,
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
let groups = [];
let visibleGroups = [];
let activeGroup = null;
let activeRow = null;
let activeTrace = null;
let activePayload = null;
let contextBuildingUuid = null;
const payloadCache = new Map();
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
  if (!counts) return "-";
  return `${counts.faces}F ${counts.vertices}V ${counts.half_edges}HE`;
}

function recordLabel(row) {
  if (row.is_building) return "Building";
  if (row.room_index !== undefined && row.room_index !== null) {
    return `room ${row.room_index}`;
  }
  return row.locator_id || `part ${row.part_index}`;
}

function isBuildingRow(row) {
  return Boolean(row.is_building) || row.room_index == null;
}

function isRoomRow(row) {
  return Boolean(row.trace) && !isBuildingRow(row);
}

function buildHouseGroups(records) {
  const byUuid = new Map();
  for (const row of records) {
    if (!row.trace) continue;
    if (!byUuid.has(row.uuid)) {
      byUuid.set(row.uuid, { uuid: row.uuid, rooms: [], building: null });
    }
    const group = byUuid.get(row.uuid);
    if (isBuildingRow(row)) {
      group.building = row;
    } else {
      group.rooms.push(row);
    }
  }
  return [...byUuid.values()]
    .map((group) => {
      group.rooms.sort(
        (a, b) => Number(a.room_index ?? a.part_index) - Number(b.room_index ?? b.part_index),
      );
      const stepCount = group.rooms.reduce((sum, row) => sum + Number(row.step_count || 0), 0);
      return { ...group, stepCount };
    })
    .sort(
      (a, b) =>
        b.stepCount - a.stepCount ||
        b.rooms.length - a.rooms.length ||
        a.uuid.localeCompare(b.uuid),
    );
}

function groupMatches(group, query) {
  if (!query) return true;
  const haystack = [
    group.uuid,
    ...group.rooms.map((row) => `${recordLabel(row)} ${row.stop_reason}`),
    group.building ? "building" : "",
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

function renderBuildingList() {
  const query = search.value.trim().toLowerCase();
  visibleGroups = groups.filter((group) => groupMatches(group, query));
  buildingList.innerHTML = visibleGroups
    .map((group) => {
      const active = group === activeGroup ? " active" : "";
      const roomCount = group.rooms.length;
      const buildingBadge = group.building ? `<span class="badge">building trace</span>` : "";
      return `
        <button class="building-row${active}" data-uuid="${escapeHtml(group.uuid)}">
          <span class="label">${escapeHtml(group.uuid)}</span>
          <span class="meta">
            <span class="badge">${roomCount} room${roomCount === 1 ? "" : "s"}</span>
            ${buildingBadge}
            <span class="badge">${group.stepCount} steps</span>
          </span>
        </button>
      `;
    })
    .join("");

  for (const button of buildingList.querySelectorAll(".building-row")) {
    button.addEventListener("click", () => {
      const group = groups.find((candidate) => candidate.uuid === button.dataset.uuid);
      if (group) void selectHouse(group);
    });
  }
}

function renderRoomList() {
  if (!activeGroup) {
    roomPanel.classList.add("hidden");
    roomList.innerHTML = "";
    return;
  }
  roomPanel.classList.remove("hidden");
  const rows = [];
  if (activeGroup.building) rows.push(activeGroup.building);
  rows.push(...activeGroup.rooms);

  roomList.innerHTML = rows
    .map((row) => {
      const active = row === activeRow ? " active" : "";
      return `
        <button class="trace-row${active}" data-trace="${escapeHtml(row.trace)}">
          <span class="label">${escapeHtml(recordLabel(row))}</span>
          <span class="meta">
            <span class="badge">${row.frame_count ?? row.step_count ?? 0} frames</span>
            <span class="badge">${escapeHtml(row.stop_reason || "")}</span>
            <span class="badge">${formatCounts(row.initial_counts)}</span>
          </span>
        </button>
      `;
    })
    .join("");

  for (const button of roomList.querySelectorAll(".trace-row")) {
    button.addEventListener("click", () => {
      const row = rows.find((candidate) => candidate.trace === button.dataset.trace);
      if (row) void selectRoom(row);
    });
  }
}

function renderSidebar() {
  renderBuildingList();
  renderRoomList();
}

async function fetchPayload(uuid) {
  if (payloadCache.has(uuid)) return payloadCache.get(uuid);
  const response = await fetch(`${PIPELINE_ROOT}/${uuid}/tier_payload.json`);
  const payload = response.ok ? await response.json() : null;
  payloadCache.set(uuid, payload);
  return payload;
}

async function selectHouse(group, { preferredRoomIndex = null } = {}) {
  activeGroup = group;
  activePayload = await fetchPayload(group.uuid);
  contextBuildingUuid = null;
  renderSidebar();
  openOriginal.href = `./viewer-tiers.html#b=${encodeURIComponent(group.uuid)}`;

  let target = null;
  if (preferredRoomIndex != null) {
    target =
      group.rooms.find((row) => Number(row.room_index) === Number(preferredRoomIndex)) ||
      group.building;
  }
  if (!target) target = group.rooms[0] || group.building;
  if (target) await selectRoom(target, { refocusCamera: true });
  else status.textContent = "No traces for this building.";
}

async function selectRoom(row, { refocusCamera = false } = {}) {
  await loadTrace(row, { resetFrame: false, refocusCamera });
}

async function loadTrace(row, { resetFrame = false, refocusCamera = false } = {}) {
  const preservedFrame = resetFrame ? 0 : frameIndex;
  activeRow = row;
  renderSidebar();
  status.textContent = "Loading trace...";

  const response = await fetch(`${TRACE_ROOT}/${row.trace}`);
  if (!response.ok) throw new Error(`trace fetch failed: ${response.status}`);
  activeTrace = await response.json();

  const maxFrame = Math.max(0, activeTrace.frames.length - 1);
  frameIndex = Math.min(preservedFrame, maxFrame);
  frameSlider.min = "0";
  frameSlider.max = String(maxFrame);
  frameSlider.value = String(frameIndex);

  rebuildContextBuilding();
  renderFrame();
  if (refocusCamera) focusModel();
}

function rebuildContextBuilding() {
  clearGroup(contextBuildingGroup, { disposeMaterials: true });
  contextBuildingUuid = null;
  if (!fullBuildingToggle.checked || !activePayload || !activeGroup) return;
  addContextBuilding(contextBuildingGroup, activePayload);
  contextBuildingUuid = activeGroup.uuid;
}

function addContextBuilding(group, payload) {
  populateBuildingScene(group, payload, { style: "calm" });
  group.traverse((obj) => {
    if (!obj.isMesh) return;
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    const cloned = materials.map((material) => {
      const copy = material.clone();
      const baseOpacity = copy.opacity ?? 1;
      copy.transparent = true;
      copy.opacity = Math.min(1, baseOpacity * CONTEXT_OPACITY);
      copy.depthWrite = false;
      return copy;
    });
    obj.material = Array.isArray(obj.material) ? cloned : cloned[0];
    obj.userData.contextBuilding = true;
  });
}

function currentStepForFrame() {
  if (!activeTrace) return null;
  if (frameIndex <= 0) return activeTrace.steps?.[0] ?? null;
  return activeTrace.steps?.find((step) => step.after_frame === frameIndex) ?? null;
}

function triggerFaceIds(step) {
  if (!step || step.action !== "adjacent_coplanar_face_merge") return new Set();
  return new Set(step.trigger_ids.map((id) => Number(id)));
}

function displayFrame(index) {
  const frame = activeTrace.frames[index];
  if (!frame) return frame;
  if (frame.faces?.length) return frame;
  if (frame.pipeline_step !== "tier_payload_input") return frame;
  const inputTiles = activeTrace.frames.find(
    (candidate) => candidate.pipeline_step === "input_tiles" && candidate.faces?.length,
  );
  if (!inputTiles) return frame;
  return { ...frame, faces: inputTiles.faces };
}

function renderFrame() {
  clearGroup(traceGroup);
  if (!activeTrace || !activeRow) return;

  if (
    fullBuildingToggle.checked &&
    activePayload &&
    activeGroup &&
    contextBuildingUuid !== activeGroup.uuid
  ) {
    rebuildContextBuilding();
  }

  const frame = displayFrame(frameIndex);
  const previousFrame =
    ghostToggle.checked && frameIndex > 0 ? displayFrame(frameIndex - 1) : null;
  const step = currentStepForFrame();
  const highlightFaces = triggerFaceIds(step);

  if (previousFrame) {
    const ghostFrameGroup = new THREE.Group();
    ghostFrameGroup.name = "previousFrame";
    addFrameMeshes(ghostFrameGroup, previousFrame, {
      ghost: true,
      highlightFaces: new Set(),
    });
    traceGroup.add(ghostFrameGroup);
  }

  addFrameMeshes(traceGroup, frame, {
    ghost: false,
    highlightFaces,
  });
  addOverlayEdges(traceGroup, frame.coherence_edges, coherenceEdgeMaterial);
  addOverlayEdges(traceGroup, frame.footprint_edges, footprintEdgeMaterial);

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
  currentTitle.textContent = `${activeRow.uuid} · ${recordLabel(activeRow)}`;
  const stopReason = activeTrace.stop?.reason ?? "unknown";
  currentMeta.textContent = `${activeTrace.frames.length} pipeline frames · ${stopReason}`;
  pill.textContent = `Frame ${frameIndex + 1}/${activeTrace.frames.length}`;
  frameReadout.textContent = `${frameIndex} / ${activeTrace.frames.length - 1}`;
  stepTitle.textContent = step
    ? step.action.replaceAll("_", " ")
    : frame.label || frame.pipeline_step || "Frame";
  stepCounts.textContent = formatCounts(frame.counts);
  status.textContent = `${frame.faces.length} faces rendered`;

  if (!step) {
    stepBody.innerHTML = formatFrameMeta(frame);
    return;
  }

  const before = step.before_counts;
  const after = step.after_counts;
  stepBody.innerHTML = `
    <div class="delta-line"><span>trigger</span><span>${escapeHtml(step.trigger_ids.join(", "))}</span></div>
    <div class="delta-line"><span>faces</span><span>${before.faces} → ${after.faces}</span></div>
    <div class="delta-line"><span>vertices</span><span>${before.vertices} → ${after.vertices}</span></div>
    <div class="delta-line"><span>half-edges</span><span>${before.half_edges} → ${after.half_edges}</span></div>
    ${formatFrameMeta(frame)}
  `;
}

function formatFrameMeta(frame) {
  const meta = frame.meta || {};
  const lines = Object.entries(meta).map(([key, value]) => {
    const text = Array.isArray(value) ? value.join(", ") : String(value);
    return `<div class="delta-line"><span>${escapeHtml(key)}</span><span>${escapeHtml(text)}</span></div>`;
  });
  if (!lines.length) {
    return `<div class="delta-line"><span>state</span><span>${escapeHtml(frame.label || frame.pipeline_step || "")}</span></div>`;
  }
  return lines.join("");
}

function addOverlayEdges(group, edges, material) {
  if (!Array.isArray(edges) || !edges.length) return;
  for (const edge of edges) {
    if (!edge?.a || !edge?.b) continue;
    const geometry = new THREE.BufferGeometry().setFromPoints([vec(edge.a), vec(edge.b)]);
    group.add(new THREE.Line(geometry, material));
  }
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

function clearGroup(group, { disposeMaterials = false } = {}) {
  for (const child of group.children) {
    child.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (disposeMaterials && obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const material of mats) material.dispose?.();
      }
    });
  }
  group.clear();
}

function focusSubject() {
  if (traceGroup.children.length) return traceGroup;
  if (contextBuildingGroup.children.length) return contextBuildingGroup;
  return modelGroup;
}

function focusModel() {
  const subject = focusSubject();
  const box = new THREE.Box3().setFromObject(subject);
  if (box.isEmpty()) {
    camera.position.set(4, 3, 5);
    controls.target.set(0, 0, 0);
    controls.update();
    requestRender();
    return;
  }
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const center = sphere.center;
  const radius = Math.max(sphere.radius, 0.5);
  // Frame the active room trace, not the full tier_payload shell (avoids ultra-wide pulls).
  camera.position.copy(center).add(new THREE.Vector3(radius * 0.92, radius * 0.68, radius * 1.02));
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

function moveBuilding(delta) {
  if (!visibleGroups.length || !activeGroup) return;
  const preferredRoomIndex = activeRow?.room_index;
  const index = visibleGroups.indexOf(activeGroup);
  const next = visibleGroups[(index + delta + visibleGroups.length) % visibleGroups.length];
  void selectHouse(next, { preferredRoomIndex });
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
  groups = buildHouseGroups(indexData.records || []);
  const roomCount = groups.reduce((sum, group) => sum + group.rooms.length, 0);
  const domain = indexData.domain || "room-shell";
  sidebarStats.textContent = `${groups.length} buildings · ${roomCount} rooms · ${domain}`;

  renderBuildingList();
  const first = visibleGroups[0] || groups[0];
  if (first) await selectHouse(first, { preferredRoomIndex: null });
  else status.textContent = "No traces found.";
}

search.addEventListener("input", () => {
  const query = search.value.trim().toLowerCase();
  const previousUuid = activeGroup?.uuid;
  renderBuildingList();
  if (!visibleGroups.length) {
    roomPanel.classList.add("hidden");
    return;
  }
  const stillVisible = visibleGroups.find((group) => group.uuid === previousUuid);
  if (!stillVisible) void selectHouse(visibleGroups[0]);
  else if (stillVisible !== activeGroup) void selectHouse(stillVisible);
  else renderRoomList();
});

prevTraceButton.addEventListener("click", () => moveBuilding(-1));
nextTraceButton.addEventListener("click", () => moveBuilding(1));
prevFrameButton.addEventListener("click", () => moveFrame(-1));
nextFrameButton.addEventListener("click", () => moveFrame(1));
frameSlider.addEventListener("input", () => {
  frameIndex = Number(frameSlider.value);
  renderFrame();
});
edgesToggle.addEventListener("change", renderFrame);
ghostToggle.addEventListener("change", renderFrame);
fullBuildingToggle.addEventListener("change", () => {
  rebuildContextBuilding();
  renderFrame();
  focusModel();
});
window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(viewport);

resize();
init().catch((err) => {
  console.error(err);
  status.textContent = String(err.message || err);
});
