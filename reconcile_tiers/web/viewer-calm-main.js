import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { populateBuildingScene } from "./tier-preview.js";
import { renderImpactSidebar } from "./calm/confidence-impact-sidebar.js";
import { shouldShowDefect } from "./calm/impact-threshold.js";
import { frameScene, flyToLocator } from "./calm/camera-rig.js";
import { applyLod, lodFromCamera } from "./calm/lod-ladder.js";
import { renderSilentFixToast } from "./calm/silent-fix-toast.js";
import { bindDetailSlider } from "./calm/detail-slider.js";
import { applyStyle } from "./calm/render-style.js";
import { buildBuildingConfidence, buildPerElementConfidence, indexFlagSignals } from "./calm/confidence.js";
import { setSketchyEdgeResolution } from "./calm/sketchy-edges.js";

const DATA_ROOT = "../../pipeline-outputs";
const FALLBACK_UUID = "016980bc-6762-4022-bfbf-17df4112e10c";

const canvas = document.querySelector("#view");
const sidebar = document.querySelector("#impact-sidebar");
const slider = document.querySelector("#lod-slider");
const status = document.querySelector("#status");
const toast = document.querySelector("#toast");
const addressEl = document.querySelector("#current-address");
const uuidEl = document.querySelector("#current-uuid");
const auditDeepLink = document.querySelector("#audit-deeplink");
const screenshotButton = document.querySelector("#screenshot-button");
const approximateChip = document.querySelector("#approximate-chip");
const projectionToggle = document.querySelector("#projection-toggle");

const params = new URLSearchParams(window.location.search);
const uuid = params.get("uuid") || FALLBACK_UUID;
const initialMode = params.get("mode") || "browse";

const scene = new THREE.Scene();
let camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
let useOrthographic = false;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
renderer.shadowMap.enabled = false;
renderer.toneMapping = THREE.LinearToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;

let controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

let frame = { center: new THREE.Vector3(), diagonal: 8 };
let manualLod = Number(slider.value) || 2;
let mode = "browse";
let currentPayload = null;
let signals = { flagQueue: null, audit: null, flagIndex: new Map() };
let twinGroup = null;
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
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  const aspect = w / Math.max(h, 1);
  if (camera.isOrthographicCamera) {
    const halfHeight = (camera.top - camera.bottom) / 2;
    const halfWidth = halfHeight * aspect;
    camera.left = -halfWidth;
    camera.right = halfWidth;
  } else {
    camera.aspect = aspect;
  }
  camera.updateProjectionMatrix();
  setSketchyEdgeResolution(w, h);
  requestRender();
}

function swapProjection() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  const aspect = w / Math.max(h, 1);
  const target = controls.target.clone();
  const position = camera.position.clone();
  const distance = position.distanceTo(target);

  let next;
  if (useOrthographic) {
    // The orthographic frustum height is sized so the existing perspective
    // FOV (50°) covers approximately the same on-screen area at this
    // distance.
    const fovRadians = THREE.MathUtils.degToRad(50);
    const halfHeight = Math.tan(fovRadians / 2) * distance;
    const halfWidth = halfHeight * aspect;
    next = new THREE.OrthographicCamera(-halfWidth, halfWidth, halfHeight, -halfHeight, 0.1, 1000);
  } else {
    next = new THREE.PerspectiveCamera(50, aspect, 0.1, 1000);
  }
  next.position.copy(position);
  next.lookAt(target);
  next.updateProjectionMatrix();

  controls.dispose();
  camera = next;
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.copy(target);
  controls.update();
  bindControlsListeners();
  requestRender();
}

async function fetchJson(url, fallback = null) {
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return fallback;
    return await res.json();
  } catch (err) {
    console.warn(`[calm] fetch ${url} failed:`, err);
    return fallback;
  }
}

async function loadPayload() {
  const url = `${DATA_ROOT}/${uuid}/tier_payload.json`;
  const payload = await fetchJson(url);
  if (!payload) throw new Error(`No tier_payload.json for ${uuid}`);
  return payload;
}

async function loadSignals(payload) {
  const flagQueue = await fetchJson(`/flag-queues?uuid=${encodeURIComponent(uuid)}`, { items: [] });
  const audit = null;
  const flagIndex = indexFlagSignals({ flagQueue, audit });
  const confidenceLookup = buildPerElementConfidence(payload, { flagIndex });
  return { flagQueue, audit, flagIndex, confidenceLookup };
}

function clearTwinOverlay() {
  if (!twinGroup) return;
  scene.remove(twinGroup);
  twinGroup.traverse((obj) => {
    obj.geometry?.dispose?.();
    obj.material?.dispose?.();
  });
  twinGroup = null;
}

async function loadTwinOverlay() {
  clearTwinOverlay();
  const twin = await fetchJson(`${DATA_ROOT}/${uuid}/twin_payload.json`);
  if (!twin) return;

  const group = new THREE.Group();
  group.userData.tierPreview = true;
  group.userData.calmTwinOverlay = true;
  const material = new THREE.MeshBasicMaterial({
    color: 0x4a6b8a,
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const faces = collectTwinFaces(twin);
  for (const corners of faces) {
    if (corners.length < 3) continue;
    const geometry = trianglesFromCorners(corners);
    if (!geometry) continue;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.tierPreview = true;
    mesh.userData.calmTwinOverlay = true;
    group.add(mesh);
  }

  scene.add(group);
  twinGroup = group;
}

function collectTwinFaces(twin) {
  // twin_payload.json shape varies; defensively pull anything that looks
  // like a polygon (corners with x/y/z). This is best-effort: the overlay
  // is decorative — it only needs to render *something* recognizable.
  const out = [];
  function walk(node) {
    if (!node) return;
    if (Array.isArray(node)) {
      const looksLikePolygon = node.length >= 3
        && node.every((p) => p && typeof p === "object" && ("x" in p || "0" in p));
      if (looksLikePolygon) {
        const corners = node.map((p) => ({
          x: Number(p.x ?? p[0] ?? 0),
          y: Number(p.y ?? p[1] ?? 0),
          z: Number(p.z ?? p[2] ?? 0),
        }));
        if (corners.every((c) => Number.isFinite(c.x) && Number.isFinite(c.y) && Number.isFinite(c.z))) {
          out.push(corners);
          return;
        }
      }
      node.forEach(walk);
      return;
    }
    if (typeof node === "object") {
      for (const key of Object.keys(node)) {
        if (key === "corners" && Array.isArray(node[key])) walk(node[key]);
        else walk(node[key]);
      }
    }
  }
  walk(twin);
  return out;
}

function trianglesFromCorners(corners) {
  if (corners.length < 3) return null;
  const positions = [];
  for (let i = 1; i < corners.length - 1; i++) {
    const a = corners[0];
    const b = corners[i];
    const c = corners[i + 1];
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return geo;
}

function refreshAuditAffordances() {
  sidebar.style.display = "";
  auditDeepLink.classList.toggle("hidden", mode !== "audit");
  screenshotButton.classList.toggle("hidden", mode !== "export");
  if (mode === "audit") {
    const queueStats = renderImpactSidebar(sidebar, signals.flagQueue, {
      onInspect(locator) {
        if (locator && flyToLocator(scene, camera, controls, locator)) requestRender();
      },
    });
    renderSilentFixToast(toast, queueStats?.hiddenCount || 0);
  } else {
    sidebar.classList.add("collapsed");
    sidebar.innerHTML = "";
    toast.classList.add("hidden");
  }
}

function refreshApproximateChip() {
  const items = signals.flagQueue?.items || [];
  const above = items.filter((item) => shouldShowDefect(item));
  if (above.length === 0) {
    approximateChip.classList.add("hidden");
    return;
  }
  approximateChip.classList.remove("hidden");
  const noun = above.length === 1 ? "approximate element" : "approximate elements";
  approximateChip.querySelector(".text").textContent = `${above.length} ${noun} · review in audit`;
}

async function setMode(next) {
  if (!currentPayload) return;
  mode = next;
  if (mode === "source") {
    applyStyle(scene, renderer, "source", currentPayload, signals);
    await loadTwinOverlay();
  } else {
    clearTwinOverlay();
    applyStyle(scene, renderer, mode, currentPayload, signals);
  }
  applyLod(scene, manualLod);
  refreshAuditAffordances();
  refreshApproximateChip();
  if (mode === "audit") {
    status.textContent = `Audit · ${currentPayload.address || uuid}`;
  } else if (mode === "source") {
    status.textContent = `Raw scan · ${currentPayload.address || uuid}`;
  } else if (mode === "export") {
    status.textContent = "";
  } else {
    status.textContent = currentPayload.address || uuid;
  }
  requestRender();
}

async function load() {
  status.textContent = "Loading…";
  currentPayload = await loadPayload();
  signals = await loadSignals(currentPayload);

  populateBuildingScene(scene, currentPayload, {
    style: "calm",
    confidenceLookup: signals.confidenceLookup,
  });
  frame = frameScene(
    scene,
    camera,
    controls,
    new THREE.Vector3(
      currentPayload.building_center?.x || 0,
      currentPayload.building_center?.y || 0,
      currentPayload.building_center?.z || 0,
    ),
  );
  addressEl.textContent = currentPayload.address || "(no address)";
  uuidEl.textContent = uuid;
  await setMode(initialMode === "audit" || initialMode === "source" || initialMode === "export" ? initialMode : "browse");

  const buildingConfidence = buildBuildingConfidence(currentPayload);
  console.info(
    `[calm] building=${buildingConfidence.toFixed(2)} · per-element shaded=${signals.confidenceLookup?.size ?? 0}`,
  );
}

document.querySelectorAll("#mode-pills button").forEach((button) => {
  button.addEventListener("click", () => {
    const next = button.dataset.mode;
    if (next && next !== mode) setMode(next);
  });
});

approximateChip.addEventListener("click", () => {
  if (mode !== "audit") setMode("audit");
});

projectionToggle.addEventListener("click", () => {
  useOrthographic = !useOrthographic;
  projectionToggle.setAttribute("aria-pressed", useOrthographic ? "true" : "false");
  projectionToggle.textContent = useOrthographic ? "Persp" : "Axon";
  swapProjection();
});

bindDetailSlider(slider, (level) => {
  manualLod = level;
  applyLod(scene, level);
  requestRender();
});

function bindControlsListeners() {
  controls.addEventListener("change", () => {
    const autoLod = lodFromCamera(camera, frame.center, frame.diagonal);
    if (Number(slider.value) !== autoLod) slider.value = String(autoLod);
    applyLod(scene, Number(slider.value));
    requestRender();
  });
}
bindControlsListeners();

auditDeepLink.addEventListener("click", () => {
  window.open(`/reconcile_tiers/web/viewer-tiers.html#${encodeURIComponent(uuid)}`, "_blank");
});

screenshotButton.addEventListener("click", () => {
  renderer.render(scene, camera);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${uuid}-calm-${mode}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, "image/png");
});

window.addEventListener("resize", resize);

window.addEventListener("contextmenu", async (event) => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const pointer = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(pointer, camera);
  const targets = [...(scene.userData.tierLocatorMap?.values() || [])];
  const hits = raycaster.intersectObjects(targets, false);
  const uid = hits[0]?.object?.userData?.elementUid;
  if (uid) {
    try {
      await navigator.clipboard?.writeText(uid);
      status.textContent = `Copied ${uid}`;
    } catch {
      status.textContent = uid;
    }
  }
});

resize();
load().catch((err) => {
  console.error(err);
  status.textContent = err.message || String(err);
});
