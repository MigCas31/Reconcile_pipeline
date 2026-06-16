import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BUILDINGS_API = "/scan-cache/buildings";
const BUILDING_API = "/scan-cache/building";
const UUID_RE = /^[0-9a-fA-F-]{8,64}$/;

const params = new URLSearchParams(window.location.search);
let currentUuid = params.get("uuid") || "";

const buildingList = document.querySelector("#building-list");
const search = document.querySelector("#search");
const sidebarStats = document.querySelector("#sidebar-stats");
const currentTitle = document.querySelector("#current-title");
const currentMeta = document.querySelector("#current-meta");
const openTiers = document.querySelector("#open-tiers");
const statusEl = document.querySelector("#status");
const roomList = document.querySelector("#room-list");
const roomCountEl = document.querySelector("#room-count");
const canvas = document.querySelector("#view");
const prevBuilding = document.querySelector("#prev-building");
const nextBuilding = document.querySelector("#next-building");

const toggles = {
  wall: document.querySelector("#toggle-walls"),
  floor: document.querySelector("#toggle-floors"),
  ceiling: document.querySelector("#toggle-ceilings"),
  door: document.querySelector("#toggle-doors"),
  window: document.querySelector("#toggle-windows"),
  edges: document.querySelector("#toggle-edges"),
  colorByRoom: document.querySelector("#toggle-color-by-room"),
};

const OVERLAY_CONFIG = {
  merged: {
    toggle: document.querySelector("#toggle-merged-apple"),
    color: 0xd946ef,
    label: "merged.json",
  },
  reconciled: {
    toggle: document.querySelector("#toggle-reconciled"),
    color: 0x22d3ee,
    label: "reconciled.json",
  },
  tier_payload: {
    toggle: document.querySelector("#toggle-tier-payload"),
    color: 0xfbbf24,
    label: "tier_payload.json",
  },
};

const KIND_COLORS = {
  wall: 0xd8dde8,
  floor: 0x6b9080,
  ceiling: 0xc7a36f,
  door: 0xb88e65,
  window: 0x87ceeb,
};

const ROOM_PALETTE = [
  0x5b8cff, 0xf97316, 0x22c55e, 0xe879f9, 0x14b8a6,
  0xf43f5e, 0xa855f7, 0xfacc15, 0x06b6d4, 0xfb7185,
];

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1117);

const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.addEventListener("change", requestRender);

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
keyLight.position.set(6, 12, 8);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.35);
fillLight.position.set(-6, 4, -6);
scene.add(fillLight);

const modelGroup = new THREE.Group();
const overlayGroups = {
  merged: new THREE.Group(),
  reconciled: new THREE.Group(),
  tier_payload: new THREE.Group(),
};
scene.add(modelGroup);
for (const group of Object.values(overlayGroups)) {
  scene.add(group);
}

let rows = [];
let payload = null;
let hiddenRooms = new Set();
let renderQueued = false;

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
  const rect = canvas.parentElement.getBoundingClientRect();
  const w = Math.max(1, rect.width);
  const h = Math.max(1, rect.height);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  requestRender();
}

function pipelineOverlays() {
  return payload?.pipeline_overlays || {};
}

function roomColorIndex(filename) {
  if (!payload?.rooms) return 0;
  const idx = payload.rooms.findIndex((room) => room.filename === filename);
  return idx >= 0 ? idx : 0;
}

function colorForSurface(surface) {
  if (toggles.colorByRoom.checked) {
    return ROOM_PALETTE[roomColorIndex(surface.room_file) % ROOM_PALETTE.length];
  }
  return KIND_COLORS[surface.kind] ?? 0xcccccc;
}

function trianglesFromCorners(corners) {
  if (corners.length < 3) return null;
  const positions = [];
  for (let i = 1; i < corners.length - 1; i += 1) {
    const a = corners[0];
    const b = corners[i];
    const c = corners[i + 1];
    positions.push(
      a[0], a[1], a[2],
      b[0], b[1], b[2],
      c[0], c[1], c[2],
    );
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function edgeGeometryFromCorners(corners) {
  if (corners.length < 2) return null;
  const positions = [];
  for (let i = 0; i < corners.length; i += 1) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return geometry;
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.geometry?.dispose?.();
    child.material?.dispose?.();
  }
}

function kindVisible(kind) {
  if (kind === "wall") return toggles.wall.checked;
  if (kind === "floor") return toggles.floor.checked;
  if (kind === "ceiling") return toggles.ceiling.checked;
  if (kind === "door") return toggles.door.checked;
  if (kind === "window") return toggles.window.checked;
  return true;
}

function renderWallOverlay(group, overlay, color, enabled) {
  clearGroup(group);
  if (!overlay || !enabled) return 0;

  let count = 0;
  for (const room of overlay.rooms || []) {
    for (const wall of room.walls || []) {
      const geometry = trianglesFromCorners(wall.corners);
      if (!geometry) continue;
      const material = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity: 0.38,
        side: THREE.DoubleSide,
        depthWrite: false,
        roughness: 0.9,
        metalness: 0.0,
      });
      group.add(new THREE.Mesh(geometry, material));
      count += 1;
      if (toggles.edges.checked) {
        const edgeGeo = edgeGeometryFromCorners(wall.corners);
        if (edgeGeo) {
          group.add(new THREE.LineSegments(
            edgeGeo,
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 }),
          ));
        }
      }
    }
  }
  return count;
}

function updateOverlayToggles() {
  const overlays = pipelineOverlays();
  for (const [key, config] of Object.entries(OVERLAY_CONFIG)) {
    const toggle = config.toggle;
    if (!toggle) continue;
    const available = Boolean(overlays[key]?.wall_count);
    toggle.disabled = !available;
    toggle.closest(".overlay-toggle")?.classList.toggle("disabled", !available);
    if (!available) toggle.checked = false;
  }
}

function renderPipelineOverlays() {
  const overlays = pipelineOverlays();
  const active = [];
  for (const [key, config] of Object.entries(OVERLAY_CONFIG)) {
    const count = renderWallOverlay(
      overlayGroups[key],
      overlays[key],
      config.color,
      config.toggle?.checked,
    );
    if (config.toggle?.checked && count) {
      active.push(`${count} ${config.label}`);
    }
  }
  return active;
}

function renderModel() {
  clearGroup(modelGroup);
  if (!payload) {
    for (const group of Object.values(overlayGroups)) clearGroup(group);
    return;
  }

  let surfaceCount = 0;
  for (const room of payload.rooms || []) {
    const roomHidden = hiddenRooms.has(room.filename);
    for (const surface of room.surfaces || []) {
      if (roomHidden || !kindVisible(surface.kind)) continue;
      const corners = surface.corners;
      if (!corners || corners.length < 3) continue;

      const color = colorForSurface(surface);
      const opacity = surface.kind === "window" ? 0.45 : surface.kind === "door" ? 0.85 : 0.72;
      const geometry = trianglesFromCorners(corners);
      if (!geometry) continue;

      const material = new THREE.MeshStandardMaterial({
        color,
        transparent: opacity < 1,
        opacity,
        side: THREE.DoubleSide,
        roughness: 0.85,
        metalness: 0.05,
      });
      modelGroup.add(new THREE.Mesh(geometry, material));
      surfaceCount += 1;

      if (toggles.edges.checked) {
        const edgeGeo = edgeGeometryFromCorners(corners);
        if (edgeGeo) {
          modelGroup.add(new THREE.LineSegments(
            edgeGeo,
            new THREE.LineBasicMaterial({ color: 0x1a1a1a, transparent: true, opacity: 0.35 }),
          ));
        }
      }
    }
  }

  const activeOverlays = renderPipelineOverlays();
  const overlayNote = activeOverlays.length ? ` · ${activeOverlays.join(" · ")}` : "";
  statusEl.textContent = `${payload.room_count} scan rooms · ${surfaceCount} raw surfaces${overlayNote}`;
  requestRender();
}

function expandOverlayCorners(box, overlay) {
  if (!overlay) return false;
  let hasGeometry = false;
  for (const room of overlay.rooms || []) {
    for (const wall of room.walls || []) {
      for (const corner of wall.corners || []) {
        box.expandByPoint(new THREE.Vector3(corner[0], corner[1], corner[2]));
        hasGeometry = true;
      }
    }
  }
  return hasGeometry;
}

function framePayload(data) {
  const box = new THREE.Box3();
  let hasGeometry = false;
  for (const room of data.rooms || []) {
    for (const surface of room.surfaces || []) {
      for (const corner of surface.corners || []) {
        box.expandByPoint(new THREE.Vector3(corner[0], corner[1], corner[2]));
        hasGeometry = true;
      }
    }
  }
  for (const overlay of Object.values(data.pipeline_overlays || {})) {
    if (expandOverlayCorners(box, overlay)) hasGeometry = true;
  }
  if (!hasGeometry) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 1) * 0.75;
  camera.position.set(center.x + radius, center.y + radius * 0.6, center.z + radius);
  controls.target.copy(center);
  controls.update();
}

function renderRoomList() {
  roomList.innerHTML = "";
  if (!payload?.rooms?.length) {
    roomCountEl.textContent = "0";
    return;
  }
  roomCountEl.textContent = String(payload.rooms.length);
  for (const room of payload.rooms) {
    const row = document.createElement("div");
    row.className = `room-row${hiddenRooms.has(room.filename) ? " hidden-room" : ""}`;
    const swatch = document.createElement("span");
    swatch.className = "room-swatch";
    swatch.style.background = `#${ROOM_PALETTE[roomColorIndex(room.filename) % ROOM_PALETTE.length].toString(16).padStart(6, "0")}`;
    const label = document.createElement("span");
    label.className = "room-label";
    label.textContent = room.filename.replace(/\.json$/, "");
    label.title = `${room.filename} · story ${room.story ?? "?"} · ${room.surfaces?.length ?? 0} surfaces`;
    row.appendChild(swatch);
    row.appendChild(label);
    row.addEventListener("click", () => {
      if (hiddenRooms.has(room.filename)) hiddenRooms.delete(room.filename);
      else hiddenRooms.add(room.filename);
      renderRoomList();
      renderModel();
    });
    roomList.appendChild(row);
  }
}

function renderBuildingList() {
  buildingList.innerHTML = "";
  const query = (search.value || "").trim().toLowerCase();
  const filtered = rows.filter((row) => {
    if (!query) return true;
    return row.uuid.toLowerCase().includes(query)
      || (row.address || "").toLowerCase().includes(query);
  });
  for (const row of filtered) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `building-row${row.uuid === currentUuid ? " active" : ""}`;
    btn.innerHTML = `<span class="addr">${escapeHtml(row.address || row.uuid.slice(0, 8))}</span>`
      + `<span class="meta">${row.room_count} rooms · ${row.uuid.slice(0, 8)}…</span>`;
    btn.addEventListener("click", () => loadBuilding(row.uuid));
    buildingList.appendChild(btn);
  }
  sidebarStats.textContent = `${filtered.length} / ${rows.length} scan-cache buildings`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function updateUrl(uuid) {
  const url = new URL(window.location.href);
  url.searchParams.set("uuid", uuid);
  window.history.replaceState({}, "", url);
}

function overlaySummary() {
  const overlays = pipelineOverlays();
  const parts = [];
  if (overlays.merged) parts.push(`merged ${overlays.merged.wall_count}w`);
  if (overlays.reconciled) {
    const cls = overlays.reconciled.classification ? ` ${overlays.reconciled.classification}` : "";
    parts.push(`reconciled ${overlays.reconciled.wall_count}w${cls}`);
  }
  if (overlays.tier_payload) {
    const tier = overlays.tier_payload.tier != null ? ` tier ${overlays.tier_payload.tier}` : "";
    parts.push(`tier ${overlays.tier_payload.wall_count}w${tier}`);
  }
  return parts.join(" · ");
}

function updateHeader() {
  if (!payload) {
    currentTitle.textContent = "Select a building";
    currentMeta.textContent = "";
    openTiers.href = "#";
    return;
  }
  currentTitle.textContent = payload.address || payload.uuid;
  const pipeline = overlaySummary();
  currentMeta.textContent = pipeline
    ? `${payload.room_count} scan rooms · ${pipeline} · ${payload.scan_dir}`
    : `${payload.room_count} scan rooms · ${payload.scan_dir} · no pipeline outputs`;
  openTiers.href = `viewer-tiers.html#uuid=${encodeURIComponent(payload.uuid)}`;
}

async function fetchBuildings() {
  const response = await fetch(BUILDINGS_API, { cache: "no-store" });
  if (!response.ok) throw new Error(`Buildings API ${response.status}`);
  const data = await response.json();
  return data.buildings || [];
}

async function fetchBuilding(uuid) {
  const response = await fetch(`${BUILDING_API}?uuid=${encodeURIComponent(uuid)}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Building API ${response.status} for ${uuid}`);
  return response.json();
}

async function loadBuilding(uuid) {
  if (!UUID_RE.test(uuid)) return;
  currentUuid = uuid;
  hiddenRooms = new Set();
  updateUrl(uuid);
  renderBuildingList();
  statusEl.textContent = "Loading scan-cache…";
  try {
    payload = await fetchBuilding(uuid);
    updateHeader();
    updateOverlayToggles();
    renderRoomList();
    renderModel();
    framePayload(payload);
  } catch (err) {
    payload = null;
    clearGroup(modelGroup);
    for (const group of Object.values(overlayGroups)) clearGroup(group);
    updateOverlayToggles();
    statusEl.textContent = `Failed: ${err.message}`;
    updateHeader();
    renderRoomList();
  }
  requestRender();
}

function navigate(delta) {
  if (!rows.length || !currentUuid) return;
  const idx = rows.findIndex((row) => row.uuid === currentUuid);
  const next = rows[(idx + delta + rows.length) % rows.length];
  if (next) loadBuilding(next.uuid);
}

search.addEventListener("input", renderBuildingList);
prevBuilding.addEventListener("click", () => navigate(-1));
nextBuilding.addEventListener("click", () => navigate(1));
for (const toggle of Object.values(toggles)) {
  toggle?.addEventListener("change", renderModel);
}
for (const config of Object.values(OVERLAY_CONFIG)) {
  config.toggle?.addEventListener("change", renderModel);
}
window.addEventListener("resize", resize);

async function init() {
  resize();
  statusEl.textContent = "Loading building index…";
  try {
    rows = await fetchBuildings();
  } catch (err) {
    statusEl.textContent = `Index failed: ${err.message}`;
    return;
  }
  renderBuildingList();
  if (!rows.length) {
    statusEl.textContent = "No scan-cache buildings found";
    return;
  }
  const initial = rows.some((row) => row.uuid === currentUuid)
    ? currentUuid
    : rows[0].uuid;
  await loadBuilding(initial);
}

init();
