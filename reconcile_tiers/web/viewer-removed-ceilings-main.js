import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  addPascalLighting,
  clearBuildingMeshes,
  populateBuildingScene,
  setStoryExplode,
  setVisibleStory,
} from "./tier-preview.js";

const DATA_ROOT = "../../pipeline-outputs";
const REMOVAL_DATA_URL = "../../.context/removed-ceiling-fragments.json";

const canvas = document.querySelector("#view");
const list = document.querySelector("#list");
const search = document.querySelector("#search");
const sidebarStats = document.querySelector("#sidebar-stats");
const status = document.querySelector("#status");
const pill = document.querySelector("#pill");
const loading = document.querySelector("#loading");
const currentAddress = document.querySelector("#current-address");
const currentMeta = document.querySelector("#current-meta");
const navPrev = document.querySelector("#nav-prev");
const navNext = document.querySelector("#nav-next");
const sortModeSelect = document.querySelector("#sort-mode");
const explodeSlider = document.querySelector("#explode-slider");
const explodeReadout = document.querySelector("#explode-readout");
const storySelect = document.querySelector("#story-select");
const baseToggle = document.querySelector("#base-toggle");
const removedToggle = document.querySelector("#removed-toggle");
const replacementToggle = document.querySelector("#replacement-toggle");
const fragmentList = document.querySelector("#fragment-list");
const fragmentPanelCount = document.querySelector("#fragment-panel-count");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.addEventListener("change", requestRender);
addPascalLighting(scene);

const removedMaterial = new THREE.MeshBasicMaterial({
  color: 0xff4b26,
  transparent: true,
  opacity: 0.82,
  side: THREE.DoubleSide,
  depthWrite: false,
  polygonOffset: true,
  polygonOffsetFactor: -3,
  polygonOffsetUnits: -3,
});
removedMaterial.name = "removedCeiling";
const selectedMaterial = removedMaterial.clone();
selectedMaterial.color = new THREE.Color(0xffd23f);
selectedMaterial.opacity = 0.96;
selectedMaterial.name = "removedCeilingSelected";
const outlineMaterial = new THREE.LineBasicMaterial({
  color: 0x8f1608,
  transparent: true,
  opacity: 0.9,
});
const replacementMaterial = new THREE.MeshBasicMaterial({
  color: 0x18a8c8,
  transparent: true,
  opacity: 0.58,
  side: THREE.DoubleSide,
  depthWrite: false,
  polygonOffset: true,
  polygonOffsetFactor: -6,
  polygonOffsetUnits: -6,
});
replacementMaterial.name = "replacementCeiling";
const selectedReplacementMaterial = replacementMaterial.clone();
selectedReplacementMaterial.color = new THREE.Color(0x30d6f0);
selectedReplacementMaterial.opacity = 0.82;
selectedReplacementMaterial.name = "replacementCeilingSelected";
const nonCoplanarMaterial = replacementMaterial.clone();
nonCoplanarMaterial.color = new THREE.Color(0xc7367f);
nonCoplanarMaterial.opacity = 0.48;
nonCoplanarMaterial.name = "coverageOverlapNotCoplanar";
const selectedNonCoplanarMaterial = nonCoplanarMaterial.clone();
selectedNonCoplanarMaterial.color = new THREE.Color(0xe85aa0);
selectedNonCoplanarMaterial.opacity = 0.74;
selectedNonCoplanarMaterial.name = "coverageOverlapNotCoplanarSelected";
const extensionMaterial = replacementMaterial.clone();
extensionMaterial.color = new THREE.Color(0x2f9b63);
extensionMaterial.opacity = 0.62;
extensionMaterial.name = "neighborPlaneExtension";
const selectedExtensionMaterial = extensionMaterial.clone();
selectedExtensionMaterial.color = new THREE.Color(0x45c982);
selectedExtensionMaterial.opacity = 0.84;
selectedExtensionMaterial.name = "neighborPlaneExtensionSelected";
const replacementOutlineMaterial = new THREE.LineBasicMaterial({
  color: 0x0e7186,
  transparent: true,
  opacity: 0.92,
});
const extensionOutlineMaterial = new THREE.LineBasicMaterial({
  color: 0x1d6840,
  transparent: true,
  opacity: 0.9,
});
const nonCoplanarOutlineMaterial = new THREE.LineBasicMaterial({
  color: 0x8f1e56,
  transparent: true,
  opacity: 0.86,
});

let corpus = null;
let rows = [];
let visibleRows = [];
let activeUuid = null;
let activePayload = null;
let activeRow = null;
let activeFragmentLocator = null;
let viewExplodeM = 0;
let viewSelection = "all";
let sortMode = "count-desc";
let renderQueued = false;
const removedMeshesByLocator = new Map();
const replacementMeshesByRemovedLocator = new Map();

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function vec(corner) {
  if (Array.isArray(corner)) return new THREE.Vector3(corner[0], corner[1], corner[2]);
  return new THREE.Vector3(corner.x, corner.y, corner.z);
}

function basis(corners) {
  const origin = vec(corners[0]);
  let u = new THREE.Vector3(1, 0, 0);
  for (let i = 1; i < corners.length; i += 1) {
    const edge = vec(corners[i]).sub(origin);
    if (edge.lengthSq() > 1e-10) {
      u = edge.normalize();
      break;
    }
  }
  let normal = new THREE.Vector3();
  for (let i = 0; i < corners.length; i += 1) {
    const a = vec(corners[i]);
    const b = vec(corners[(i + 1) % corners.length]);
    normal.x += (a.y - b.y) * (a.z + b.z);
    normal.y += (a.z - b.z) * (a.x + b.x);
    normal.z += (a.x - b.x) * (a.y + b.y);
  }
  if (normal.lengthSq() <= 1e-10) normal = new THREE.Vector3(0, 1, 0);
  else normal.normalize();
  const projected = u.clone().sub(normal.clone().multiplyScalar(u.dot(normal)));
  if (projected.lengthSq() > 1e-10) u = projected.normalize();
  const v = new THREE.Vector3().crossVectors(normal, u).normalize();
  return { origin, u, v };
}

function polygonGeometry(corners) {
  if (!Array.isArray(corners) || corners.length < 3) return null;
  const frame = basis(corners);
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
  corners.forEach((corner, idx) => {
    const p = vec(corner);
    positions[idx * 3] = p.x;
    positions[idx * 3 + 1] = p.y;
    positions[idx * 3 + 2] = p.z;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(triangles.flat());
  geometry.computeVertexNormals();
  return geometry;
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
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  requestRender();
}

window.addEventListener("resize", resize);

function matchesQuery(row, query) {
  if (!query) return true;
  const haystack = [
    row.uuid,
    row.address,
    Object.keys(row.reason_counts || {}).join(" "),
    Object.keys(row.source_counts || {}).join(" "),
    ...(row.items || []).map((item) => item.locator_id),
  ].join(" ").toLowerCase();
  return haystack.includes(query);
}

function sortRows() {
  rows.sort((a, b) => {
    if (sortMode === "area-desc") {
      return (b.removed_area_xz_m2 || 0) - (a.removed_area_xz_m2 || 0) || a.uuid.localeCompare(b.uuid);
    }
    if (sortMode === "uuid-asc") return a.uuid.localeCompare(b.uuid);
    return (b.removed_count || 0) - (a.removed_count || 0) || a.uuid.localeCompare(b.uuid);
  });
}

function isSamePlaneCoverage(replacement) {
  return replacement?.coverage_status === "same_plane_cover";
}

function isNeighborExtension(replacement) {
  return replacement?.coverage_status === "neighbor_plane_extension";
}

function isPlaneCoverage(replacement) {
  return isSamePlaneCoverage(replacement) || isNeighborExtension(replacement);
}

function coverageStatusLabel(replacement) {
  const status = replacement?.coverage_status || "unknown";
  if (status === "same_plane_cover") return "same-plane";
  if (status === "neighbor_plane_extension") return "neighbour extension";
  if (status === "partial_same_plane_overlap") return "partial same-plane";
  if (status === "overlap_not_coplanar") return "not coplanar";
  if (status === "overlap_unknown_plane") return "unknown plane";
  return status.replaceAll("_", " ");
}

function renderList() {
  const query = search.value.trim().toLowerCase();
  sortRows();
  visibleRows = rows.filter((row) => matchesQuery(row, query));
  list.innerHTML = visibleRows.map((row) => {
    const cls = row.classification || {};
    const reasons = Object.entries(row.reason_counts || {})
      .map(([reason, count]) => `${reason.replace(/^interior_/, "")}:${count}`)
      .join(" ");
    return `
      <button class="row ${row.uuid === activeUuid ? "active" : ""}" data-uuid="${escapeHtml(row.uuid)}">
        <div class="label">${escapeHtml(row.address || row.uuid)}</div>
        <div class="uuid">${escapeHtml(row.uuid)}</div>
        <div class="row-meta">
          <span class="badge badge-removed">${row.removed_count} removed</span>
          <span class="badge">${Number(row.removed_area_xz_m2 || 0).toFixed(2)} m²</span>
          <span class="badge badge-tier">T${escapeHtml(cls.tier ?? "?")}</span>
          <span class="badge badge-reason">${escapeHtml(reasons || "interior")}</span>
        </div>
      </button>
    `;
  }).join("");
  const summary = corpus?.summary || {};
  sidebarStats.textContent = `${visibleRows.length} shown · ${rows.length} buildings with removals · ${summary.total_removed_count ?? 0} pieces total`;
}

function populateStorySelect(payload, row) {
  const stories = new Set();
  for (const room of payload.rooms || []) {
    if (Number.isFinite(room.story)) stories.add(room.story);
  }
  for (const item of row?.items || []) {
    if (Number.isFinite(item.story)) stories.add(item.story);
  }
  const options = [...stories].sort((a, b) => a - b);
  storySelect.innerHTML = '<option value="all">All</option>' + options
    .map((story) => `<option value="${story}">Storey ${story}</option>`)
    .join("");
  if (viewSelection !== "all" && !stories.has(Number(viewSelection))) viewSelection = "all";
  storySelect.value = viewSelection;
}

function setBaseVisible(visible) {
  scene.traverse((obj) => {
    if (!obj.userData?.tierPreview || obj.userData?.removedCeiling || obj.userData?.replacementCeiling || obj.userData?.pickOnly) return;
    if (obj.isMesh || obj.isLineSegments) obj.visible = visible;
  });
  requestRender();
}

function setRemovedVisible(visible) {
  scene.traverse((obj) => {
    if (obj.userData?.removedCeiling) obj.visible = visible;
  });
  requestRender();
}

function setReplacementVisible(visible) {
  scene.traverse((obj) => {
    if (obj.userData?.replacementCeiling) obj.visible = visible;
  });
  requestRender();
}

function renderFragmentsPanel(row) {
  const items = row?.items || [];
  fragmentPanelCount.textContent = String(items.length);
  fragmentList.innerHTML = items.map((item) => {
    const replacements = item.replacements || [];
    const planeCoverageCount = replacements.filter((replacement) => isPlaneCoverage(replacement)).length;
    const replacementHtml = replacements.length
      ? `<div class="replacement-list">${replacements.slice(0, 3).map((replacement) => `
          <div class="replacement-item ${isNeighborExtension(replacement) ? "extension" : isSamePlaneCoverage(replacement) ? "" : "bad"}">→ ${escapeHtml(coverageStatusLabel(replacement))} · ov ${Number(replacement.overlap_ratio || 0).toFixed(2)} · dz ${replacement.plane_delta_m == null ? "?" : Number(replacement.plane_delta_m).toFixed(2)}m · ${escapeHtml(replacement.locator_id)}</div>
        `).join("")}</div>`
      : `<div class="replacement-item bad">→ no overlapping final ceiling piece</div>`;
    return `
      <li>
        <button class="fragment-row ${item.locator_id === activeFragmentLocator ? "active" : ""}" data-locator="${escapeHtml(item.locator_id)}">
        <div class="fragment-main">
          <span>${escapeHtml(item.reason)}</span>
          <span>${Number(item.area_xz_m2 || 0).toFixed(3)} m²</span>
        </div>
        <div class="row-meta">
          <span class="badge badge-source">${escapeHtml(item.source || "unknown")}</span>
          <span class="badge">storey ${escapeHtml(item.story ?? "?")}</span>
          <span class="badge">${escapeHtml(item.drop_stage || "drop")}</span>
          <span class="badge">${planeCoverageCount}/${replacements.length} plane cover</span>
        </div>
        ${replacementHtml}
        <div class="fragment-locator">${escapeHtml(item.locator_id)}</div>
      </button>
    </li>
    `;
  }).join("");
}

function addRemovedFragments(row) {
  removedMeshesByLocator.clear();
  replacementMeshesByRemovedLocator.clear();
  const groups = scene.userData.tierStoryGroups || new Map();
  for (const item of row.items || []) {
    const geometry = polygonGeometry(item.corners);
    if (!geometry) continue;
    const mesh = new THREE.Mesh(geometry, removedMaterial);
    mesh.userData.tierPreview = true;
    mesh.userData.removedCeiling = true;
    mesh.userData.locator = item.locator_id;
    mesh.userData.story = item.story;
    mesh.renderOrder = 30;
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), outlineMaterial);
    edge.userData.tierPreview = true;
    edge.userData.removedCeiling = true;
    edge.userData.locator = item.locator_id;
    edge.userData.story = item.story;
    edge.renderOrder = 31;
    const key = item.story == null ? "null" : String(item.story);
    const parent = groups.get(key) || scene;
    parent.add(mesh);
    parent.add(edge);
    removedMeshesByLocator.set(item.locator_id, mesh);
    for (const replacement of item.replacements || []) {
      const replacementCorners = replacement.overlap_corners?.length >= 3
        ? replacement.overlap_corners
        : replacement.corners;
      const replacementGeometry = polygonGeometry(replacementCorners);
      if (!replacementGeometry) continue;
      const samePlane = isSamePlaneCoverage(replacement);
      const extension = isNeighborExtension(replacement);
      const replacementMesh = new THREE.Mesh(
        replacementGeometry,
        extension ? extensionMaterial : samePlane ? replacementMaterial : nonCoplanarMaterial,
      );
      replacementMesh.userData.tierPreview = true;
      replacementMesh.userData.replacementCeiling = true;
      replacementMesh.userData.samePlaneCoverage = samePlane;
      replacementMesh.userData.neighborExtension = extension;
      replacementMesh.userData.removedLocator = item.locator_id;
      replacementMesh.userData.locator = replacement.locator_id;
      replacementMesh.userData.story = item.story;
      replacementMesh.renderOrder = 20;
      const replacementEdge = new THREE.LineSegments(
        new THREE.EdgesGeometry(replacementGeometry),
        extension ? extensionOutlineMaterial : samePlane ? replacementOutlineMaterial : nonCoplanarOutlineMaterial,
      );
      replacementEdge.userData.tierPreview = true;
      replacementEdge.userData.replacementCeiling = true;
      replacementEdge.userData.samePlaneCoverage = samePlane;
      replacementEdge.userData.neighborExtension = extension;
      replacementEdge.userData.removedLocator = item.locator_id;
      replacementEdge.userData.locator = replacement.locator_id;
      replacementEdge.userData.story = item.story;
      replacementEdge.renderOrder = 21;
      parent.add(replacementMesh);
      parent.add(replacementEdge);
      const bucket = replacementMeshesByRemovedLocator.get(item.locator_id) || [];
      bucket.push(replacementMesh, replacementEdge);
      replacementMeshesByRemovedLocator.set(item.locator_id, bucket);
    }
  }
  setRemovedVisible(removedToggle?.checked !== false);
  setReplacementVisible(replacementToggle?.checked !== false);
}

function selectFragment(locator, { frame = true } = {}) {
  activeFragmentLocator = locator;
  for (const [key, mesh] of removedMeshesByLocator.entries()) {
    mesh.material = key === locator ? selectedMaterial : removedMaterial;
  }
  for (const [removedLocator, meshes] of replacementMeshesByRemovedLocator.entries()) {
    for (const mesh of meshes) {
      if (mesh.isMesh) {
        const samePlane = mesh.userData?.samePlaneCoverage;
        const extension = mesh.userData?.neighborExtension;
        mesh.material = removedLocator === locator
          ? (extension ? selectedExtensionMaterial : samePlane ? selectedReplacementMaterial : selectedNonCoplanarMaterial)
          : (extension ? extensionMaterial : samePlane ? replacementMaterial : nonCoplanarMaterial);
      }
      mesh.visible = replacementToggle?.checked !== false && (locator ? removedLocator === locator : true);
    }
  }
  renderFragmentsPanel(activeRow);
  const mesh = removedMeshesByLocator.get(locator);
  if (frame && mesh) {
    const box = new THREE.Box3().setFromObject(mesh);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 1.5);
    controls.target.copy(center);
    camera.position.set(center.x + radius * 2.2, center.y + radius * 1.4, center.z + radius * 2.2);
    camera.near = 0.02;
    camera.far = Math.max(camera.far, radius * 50);
    camera.updateProjectionMatrix();
    controls.update();
  }
  const item = activeRow?.items?.find((entry) => entry.locator_id === locator);
  const replacementCount = item?.replacements?.length || 0;
  const planeCoverageCount = item?.replacements?.filter((replacement) => isPlaneCoverage(replacement)).length || 0;
  status.textContent = locator
    ? `${locator} · ${planeCoverageCount}/${replacementCount} plane coverage candidate${replacementCount === 1 ? "" : "s"} highlighted`
    : `${activeRow?.removed_count ?? 0} removed pieces`;
  requestRender();
}

function frameScene(payload) {
  const box = new THREE.Box3();
  scene.traverse((obj) => {
    if (obj.userData?.tierPreview && obj.isMesh && !obj.userData.pickOnly && !obj.userData.framingIgnore) {
      box.expandByObject(obj);
    }
  });
  const center = box.isEmpty()
    ? new THREE.Vector3(payload.building_center?.x ?? 0, payload.building_center?.y ?? 0, payload.building_center?.z ?? 0)
    : box.getCenter(new THREE.Vector3());
  const size = box.isEmpty() ? new THREE.Vector3(8, 4, 8) : box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 4);
  controls.target.copy(center);
  camera.position.set(center.x + radius * 1.25, center.y + radius * 0.95, center.z + radius * 1.25);
  camera.near = Math.max(0.05, radius / 250);
  camera.far = radius * 25;
  camera.updateProjectionMatrix();
  controls.update();
}

async function loadPayload(uuid) {
  activeUuid = uuid;
  activeRow = rows.find((row) => row.uuid === uuid);
  if (!activeRow) return;
  loading.classList.remove("hidden");
  currentAddress.textContent = "Loading...";
  currentMeta.textContent = uuid;
  try {
    const response = await fetch(`${DATA_ROOT}/${uuid}/tier_payload.json`, { cache: "no-store" });
    if (!response.ok) throw new Error(`tier_payload.json missing for ${uuid}`);
    activePayload = await response.json();
    activeFragmentLocator = null;
    clearBuildingMeshes(scene);
    populateBuildingScene(scene, activePayload, { heatingMode: false });
    addRemovedFragments(activeRow);
    populateStorySelect(activePayload, activeRow);
    setStoryExplode(scene, viewExplodeM);
    setVisibleStory(scene, viewSelection);
    setBaseVisible(baseToggle.checked);
    frameScene(activePayload);

    const cls = activePayload.classification || activeRow.classification || {};
    currentAddress.textContent = activePayload.address || activeRow.address || uuid;
    currentMeta.textContent = `${uuid} · Tier ${cls.tier ?? "-"} · ${activeRow.removed_count} removed · ${Number(activeRow.removed_area_xz_m2 || 0).toFixed(2)} m²`;
    pill.textContent = Object.entries(activeRow.reason_counts || {})
      .map(([reason, count]) => `${reason}:${count}`)
      .join(" · ");
    renderFragmentsPanel(activeRow);
    renderList();
    status.textContent = "Removed fragments are orange. Cyan is same-plane, green is a neighbouring-plane extension, and pink is overlapping non-coplanar geometry. Click a fragment row to isolate its candidates.";
    history.replaceState(null, "", `#uuid=${encodeURIComponent(uuid)}`);
  } catch (error) {
    status.textContent = String(error);
    console.error(error);
  } finally {
    loading.classList.add("hidden");
    requestRender();
  }
}

function navigate(delta) {
  if (!visibleRows.length) return;
  const idx = Math.max(0, visibleRows.findIndex((row) => row.uuid === activeUuid));
  const next = visibleRows[(idx + delta + visibleRows.length) % visibleRows.length];
  if (next) loadPayload(next.uuid);
}

list.addEventListener("click", (event) => {
  const row = event.target.closest(".row");
  if (!row?.dataset.uuid) return;
  loadPayload(row.dataset.uuid);
});

fragmentList.addEventListener("click", (event) => {
  const row = event.target.closest(".fragment-row");
  if (!row?.dataset.locator) return;
  selectFragment(row.dataset.locator);
});

search.addEventListener("input", () => {
  renderList();
  requestRender();
});

sortModeSelect.addEventListener("change", () => {
  sortMode = sortModeSelect.value;
  renderList();
});

navPrev.addEventListener("click", () => navigate(-1));
navNext.addEventListener("click", () => navigate(1));

explodeSlider.addEventListener("input", () => {
  viewExplodeM = Number(explodeSlider.value) || 0;
  explodeReadout.textContent = `${viewExplodeM.toFixed(1)} m`;
  setStoryExplode(scene, viewExplodeM);
  requestRender();
});

storySelect.addEventListener("change", () => {
  viewSelection = storySelect.value;
  setVisibleStory(scene, viewSelection);
  requestRender();
});

baseToggle.addEventListener("change", () => {
  setBaseVisible(baseToggle.checked);
});

removedToggle.addEventListener("change", () => {
  setRemovedVisible(removedToggle.checked);
});

replacementToggle.addEventListener("change", () => {
  if (activeFragmentLocator) {
    selectFragment(activeFragmentLocator, { frame: false });
  } else {
    setReplacementVisible(replacementToggle.checked);
  }
});

async function init() {
  resize();
  const response = await fetch(REMOVAL_DATA_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Missing ${REMOVAL_DATA_URL}. Run python -m reconcile_tiers.scripts.build_removed_ceiling_viewer_data first.`);
  }
  corpus = await response.json();
  rows = corpus.buildings || [];
  renderList();
  const hashUuid = new URLSearchParams(location.hash.slice(1)).get("uuid");
  const first = rows.find((row) => row.uuid === hashUuid) || rows[0];
  if (first) await loadPayload(first.uuid);
  else {
    currentAddress.textContent = "No removed fragments";
    currentMeta.textContent = "";
    status.textContent = "The removal corpus contains no buildings with removed fragments.";
  }
}

init().catch((error) => {
  console.error(error);
  currentAddress.textContent = "Could not load removal corpus";
  currentMeta.textContent = "";
  status.textContent = String(error);
  loading.classList.add("hidden");
  requestRender();
});
