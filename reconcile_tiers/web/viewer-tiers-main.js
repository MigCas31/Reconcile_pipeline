import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { SMAAPass } from "three/addons/postprocessing/SMAAPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { addPascalLighting, clearBuildingMeshes, populateBuildingScene, setCeilingsVisible, setStoryExplode, setVisibleStory } from "./tier-preview.js";
import { HEATING_COLORS, HEATING_LABELS } from "./material-palette.js";
import { parseElementUid } from "./locator.js";
import { compareCaseRows, normalizeScanHealthMap } from "./case-ranking.js";
import { openContextMenu, resetPreviews } from "./context-menu.js";
import { payloadStoryOptions } from "./story-visibility.js";

const DATA_ROOT = "../../pipeline-outputs";
const canvas = document.querySelector("#view");
const list = document.querySelector("#list");
const search = document.querySelector("#search");
const sidebarStats = document.querySelector("#sidebar-stats");
const status = document.querySelector("#status");
const pill = document.querySelector("#pill");
const loading = document.querySelector("#loading");
const signals = document.querySelector("#signals");
const main = document.querySelector("main");
const currentAddress = document.querySelector("#current-address");
const currentMeta = document.querySelector("#current-meta");
const navPrev = document.querySelector("#nav-prev");
const navNext = document.querySelector("#nav-next");
const explodeSlider = document.querySelector("#explode-slider");
const explodeReadout = document.querySelector("#explode-readout");
const storySelect = document.querySelector("#story-select");
const heatingToggle = document.querySelector("#heating-toggle");
const heatingLegend = document.querySelector("#heating-legend");
const hideCeilingsToggle = document.querySelector("#hide-ceilings-toggle");
const filterTierEl = document.querySelector("#filter-tier");
const filterRatingEl = document.querySelector("#filter-rating");
const sortModeSelect = document.querySelector("#sort-mode");
let heatingMode = false;
let hideCeilings = false;
let viewExplodeM = 0;
let viewSelection = "all";
let roofRatingsByUuid = {};
let sortMode = "tier-asc";
// Multi-select filter state. Empty set means "match all" — same semantics as
// the previous "all" sentinel.
let filterTiers = new Set();
let filterRatings = new Set();
currentMeta.tabIndex = 0;
currentMeta.setAttribute("role", "button");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
const SSGI_PARAMS = {
  enabled: true,
  sliceCount: 1,
  stepCount: 4,
  radius: 1,
  expFactor: 1.5,
  thickness: 0.5,
  backfaceLighting: 0.5,
  aoIntensity: 1.5,
  giIntensity: 0,
  useLinearThickness: false,
  useScreenSpaceSampling: true,
  useTemporalFiltering: false,
};

function configureRenderer(renderer) {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  if (renderer.shadowMap) {
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
  }
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.9;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
}

function createWebGLRuntime() {
  scene.background = new THREE.Color(0xffffff);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  configureRenderer(renderer);
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.35;
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const smaa = new SMAAPass(1, 1);
  composer.addPass(smaa);
  composer.addPass(new OutputPass());
  return {
    renderer,
    mode: "webgl",
    resize(width, height) {
      renderer.setSize(width, height, false);
      composer.setSize(width, height);
      smaa.setSize(width, height);
    },
    render() {
      composer.render();
    },
  };
}

async function createSsgiPipeline(renderer) {
  const [
    webgpu,
    tsl,
    ssgiModule,
    denoiseModule,
  ] = await Promise.all([
    import("three/webgpu"),
    import("three/tsl"),
    import("three/addons/tsl/display/SSGINode.js"),
    import("three/addons/tsl/display/DenoiseNode.js"),
  ]);
  const {
    Color,
    RenderPipeline,
    UnsignedByteType,
  } = webgpu;
  const {
    add,
    colorToDirection,
    diffuseColor,
    directionToColor,
    float,
    mix,
    mrt,
    normalView,
    output,
    pass,
    sample,
    uniform,
    vec4,
  } = tsl;
  const { ssgi } = ssgiModule;
  const { denoise } = denoiseModule;

  const scenePass = pass(scene, camera);
  const scenePassColor = scenePass.getTextureNode("output");
  let sceneColor = scenePassColor;
  const contentAlpha = scenePassColor.a;

  if (SSGI_PARAMS.enabled) {
    scenePass.setMRT(
      mrt({
        output,
        diffuseColor,
        normal: directionToColor(normalView),
      }),
    );

    const scenePassDiffuse = scenePass.getTextureNode("diffuseColor");
    const scenePassDepth = scenePass.getTextureNode("depth");
    const scenePassNormal = scenePass.getTextureNode("normal");
    scenePass.getTexture("diffuseColor").type = UnsignedByteType;
    scenePass.getTexture("normal").type = UnsignedByteType;

    const sceneNormal = sample((uv) => colorToDirection(scenePassNormal.sample(uv)));
    const giPass = ssgi(scenePassColor, scenePassDepth, sceneNormal, camera);
    giPass.sliceCount.value = SSGI_PARAMS.sliceCount;
    giPass.stepCount.value = SSGI_PARAMS.stepCount;
    giPass.radius.value = SSGI_PARAMS.radius;
    giPass.expFactor.value = SSGI_PARAMS.expFactor;
    giPass.thickness.value = SSGI_PARAMS.thickness;
    giPass.backfaceLighting.value = SSGI_PARAMS.backfaceLighting;
    giPass.aoIntensity.value = SSGI_PARAMS.aoIntensity;
    giPass.giIntensity.value = SSGI_PARAMS.giIntensity;
    giPass.useLinearThickness.value = SSGI_PARAMS.useLinearThickness;
    giPass.useScreenSpaceSampling.value = SSGI_PARAMS.useScreenSpaceSampling;
    giPass.useTemporalFiltering = SSGI_PARAMS.useTemporalFiltering;

    const giTexture = giPass.getTextureNode();
    const aoAsRgb = vec4(giTexture.a, giTexture.a, giTexture.a, float(1));
    const denoisePass = denoise(aoAsRgb, scenePassDepth, sceneNormal, camera);
    denoisePass.index.value = 0;
    denoisePass.radius.value = 4;
    const ao = denoisePass.r;
    const gi = giPass.rgb;
    sceneColor = vec4(
      add(scenePassColor.rgb.mul(ao), scenePassDiffuse.rgb.mul(gi)),
      contentAlpha,
    );
  }

  const background = uniform(new Color(0xffffff));
  const finalOutput = vec4(mix(background, sceneColor.rgb, contentAlpha), float(1));
  const pipeline = new RenderPipeline(renderer);
  pipeline.outputNode = finalOutput;
  return pipeline;
}

async function createWebGPURuntime() {
  if (!("gpu" in navigator)) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter?.();
    if (!adapter) {
      console.warn("[tier-viewer] WebGPU adapter unavailable; falling back to WebGL.");
      return null;
    }
    const { WebGPURenderer } = await import("three/webgpu");
    const renderer = new WebGPURenderer({ canvas, antialias: true });
    configureRenderer(renderer);
    renderer.setClearColor?.(0xffffff, 1);
    scene.background = null;
    await renderer.init();
    if (renderer.backend?.isWebGLBackend) {
      console.warn("[tier-viewer] WebGPURenderer is using a WebGL backend; rendering without SSGI.");
      scene.background = new THREE.Color(0xffffff);
      return {
        renderer,
        mode: "webgpu-direct",
        resize(width, height) {
          renderer.setSize(width, height, false);
        },
        render() {
          renderer.setClearAlpha?.(1);
          renderer.setClearColor?.(0xffffff, 1);
          renderer.render(scene, camera);
        },
      };
    }
    let pipeline = null;
    let postFailed = false;
    try {
      pipeline = await createSsgiPipeline(renderer);
    } catch (error) {
      console.warn("[tier-viewer] WebGPU SSGI unavailable; rendering without post-processing.", error);
      postFailed = true;
    }
    return {
      renderer,
      mode: postFailed ? "webgpu-direct" : "webgpu-ssgi",
      resize(width, height) {
        renderer.setSize(width, height, false);
      },
      render() {
        if (!postFailed && pipeline) {
          try {
            renderer.setClearAlpha?.(0);
            pipeline.render();
            return;
          } catch (error) {
            console.warn("[tier-viewer] WebGPU SSGI render failed; falling back to direct rendering.", error);
            postFailed = true;
            pipeline.dispose?.();
            pipeline = null;
            this.mode = "webgpu-direct";
          }
        }
        renderer.setClearAlpha?.(1);
        renderer.setClearColor?.(0xffffff, 1);
        renderer.render(scene, camera);
      },
    };
  } catch (error) {
    console.warn("[tier-viewer] WebGPU renderer unavailable; falling back to WebGL.", error);
    return null;
  }
}

const renderRuntime = await createWebGPURuntime() ?? createWebGLRuntime();
const renderer = renderRuntime.renderer;
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
window.__tierViewer = { scene, camera, controls, renderer, renderRuntime, requestRender: null };
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let rows = [];
let activeUuid = null;
let activePayload = null;
let selectedHelper = null;
let scanHealth = {};
let renderQueued = false;
let lastRenderWidth = 0;
let lastRenderHeight = 0;
let copyNoticeTimer = null;
const SELECTION_MATERIAL = new THREE.LineBasicMaterial({
  color: 0xffcc00,
  depthTest: false,
  depthWrite: false,
});
const FLAG_AUTO_MATERIAL = new THREE.LineBasicMaterial({
  color: 0xdc2626,
  depthTest: false,
  depthWrite: false,
  transparent: true,
  opacity: 0.95,
});
const FLAG_MANUAL_MATERIAL = new THREE.LineBasicMaterial({
  color: 0x06b6d4,
  depthTest: false,
  depthWrite: false,
  transparent: true,
  opacity: 0.95,
});
// NOTE: deliberately not flagged as `tierPreview` so it survives
// `clearBuildingMeshes` on building swap. Children are managed by
// `syncFlagOverlays`, which fully clears + repopulates from the queue.
const flagOverlayGroup = new THREE.Group();
flagOverlayGroup.renderOrder = 999;
scene.add(flagOverlayGroup);
const MULTI_SELECT_MATERIAL = new THREE.LineBasicMaterial({
  color: 0x10b981,
  depthTest: false,
  depthWrite: false,
  transparent: true,
  opacity: 0.9,
});
const multiSelectGroup = new THREE.Group();
multiSelectGroup.renderOrder = 998;
scene.add(multiSelectGroup);
const multiSelection = new Set();
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

addPascalLighting(scene);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[char]));
}

function classification(row) {
  return row.classification || row.payload?.classification || {};
}

function rowLabel(row) {
  return row.address || row.payload?.address || row.uuid;
}

function rowSearchText(row) {
  const cls = classification(row);
  return [
    row.uuid,
    row.address,
    row.payload?.address,
    cls.tier_label,
    cls.roof_type,
    cls.tier,
  ].filter(Boolean).join(" ").toLowerCase();
}

function normalizeRow(entry) {
  return typeof entry === "string" ? { uuid: entry } : entry;
}

function mergeRows(primary, discovered) {
  const merged = [];
  const seen = new Set();
  for (const row of [...primary, ...discovered]) {
    if (!row?.uuid || seen.has(row.uuid)) continue;
    seen.add(row.uuid);
    merged.push(row);
  }
  return merged;
}

function rowTierValue(row) {
  const tier = Number(classification(row).tier);
  return Number.isFinite(tier) ? tier : null;
}

function rowRatingValue(row) {
  const rec = row?.uuid ? roofRatingsByUuid[row.uuid] : null;
  if (!rec || !Object.prototype.hasOwnProperty.call(rec, "rating")) return null;
  return rec.rating;
}

function ratingGroup(value) {
  if (typeof value === "number" && Number.isFinite(value)) return 0;
  if (value === "upstream_error") return 1;
  return 2;
}

function compareByRating(a, b, dir) {
  const ra = rowRatingValue(a);
  const rb = rowRatingValue(b);
  const ga = ratingGroup(ra);
  const gb = ratingGroup(rb);
  if (ga !== gb) return ga - gb;
  if (ga === 0) {
    const numericDelta = ra - rb;
    if (numericDelta) return dir === "asc" ? numericDelta : -numericDelta;
  }
  return compareCaseRows(a, b, scanHealth);
}

function rowPredictedRating(row) {
  const value = row?.payload?.roof_quality?.predicted_rating;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function compareByPredicted(a, b, dir) {
  const ra = rowPredictedRating(a);
  const rb = rowPredictedRating(b);
  if (ra === null && rb === null) return compareCaseRows(a, b, scanHealth);
  if (ra === null) return 1;
  if (rb === null) return -1;
  const delta = ra - rb;
  if (delta) return dir === "asc" ? delta : -delta;
  return compareCaseRows(a, b, scanHealth);
}

function sortRows() {
  if (sortMode === "tier-desc") {
    rows.sort((a, b) => compareCaseRows(b, a, scanHealth));
  } else if (sortMode === "rating-asc") {
    rows.sort((a, b) => compareByRating(a, b, "asc"));
  } else if (sortMode === "rating-desc") {
    rows.sort((a, b) => compareByRating(a, b, "desc"));
  } else if (sortMode === "quality-asc") {
    rows.sort((a, b) => compareByPredicted(a, b, "asc"));
  } else if (sortMode === "quality-desc") {
    rows.sort((a, b) => compareByPredicted(a, b, "desc"));
  } else {
    rows.sort((a, b) => compareCaseRows(a, b, scanHealth));
  }
}

function rowMatchesTierFilter(row) {
  if (filterTiers.size === 0) return true;
  const tier = rowTierValue(row);
  if (tier === null) return filterTiers.has("unknown");
  return filterTiers.has(String(tier));
}

function rowMatchesRatingFilter(row) {
  if (filterRatings.size === 0) return true;
  const rating = rowRatingValue(row);
  if (rating === null) return filterRatings.has("unrated");
  if (rating === "upstream_error") return filterRatings.has("upstream_error");
  return typeof rating === "number" && filterRatings.has(String(rating));
}

async function hasTierPayload(uuid) {
  try {
    const response = await fetch(`${DATA_ROOT}/${uuid}/`, { cache: "no-store" });
    if (!response.ok) return false;
    const doc = new DOMParser().parseFromString(await response.text(), "text/html");
    return [...doc.querySelectorAll("a")]
      .some((link) => decodeURIComponent(link.getAttribute("href") || "") === "tier_payload.json");
  } catch (_error) {
    return false;
  }
}

async function discoverRowsFromDirectoryListing() {
  try {
    const response = await fetch(`${DATA_ROOT}/`, { cache: "no-store" });
    if (!response.ok) return [];
    const doc = new DOMParser().parseFromString(await response.text(), "text/html");
    const uuids = [...doc.querySelectorAll("a")]
      .map((link) => decodeURIComponent(link.getAttribute("href") || "").replace(/\/$/, ""))
      .filter((href) => UUID_RE.test(href));
    const candidates = await Promise.all(uuids.map(async (uuid) => (
      await hasTierPayload(uuid) ? { uuid } : null
    )));
    return candidates.filter(Boolean);
  } catch (error) {
    console.debug("tier payload directory discovery failed", error);
    return [];
  }
}

async function loadRows() {
  const response = await fetch(`${DATA_ROOT}/tier_index.json`, { cache: "no-store" });
  const index = response.ok ? await response.json() : { buildings: [] };
  const indexedRows = (index.buildings || []).map(normalizeRow);
  if (indexedRows.length > 1) return indexedRows;
  const discoveredRows = await discoverRowsFromDirectoryListing();
  return discoveredRows.length > indexedRows.length ? mergeRows(indexedRows, discoveredRows) : indexedRows;
}

function shortUuid(uuid) {
  return uuid ? uuid.slice(0, 8) : "";
}

async function copyText(value) {
  const text = String(value || "");
  if (!text) return false;
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_err) {
      // Fall back to the legacy path below for browsers that gate Clipboard API writes.
    }
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch (_fallbackErr) {
    copied = false;
  }
  textarea.remove();
  return copied;
}

function copyToast() {
  let toast = document.querySelector("#copy-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "copy-toast";
    main.appendChild(toast);
  }
  return toast;
}

function showCopyNotice(label, value, copied) {
  const toast = copyToast();
  const prefix = copied ? "Copied" : "Could not copy";
  const message = `${prefix} ${label}: ${value}`;
  status.classList.add("locator");
  status.classList.toggle("copy-failed", !copied);
  status.textContent = message;
  toast.textContent = message;
  toast.classList.toggle("copy-failed", !copied);
  toast.classList.add("visible");
  clearTimeout(copyNoticeTimer);
  copyNoticeTimer = setTimeout(() => {
    toast.classList.remove("visible");
  }, 1800);
}

function renderBadges(row) {
  const cls = classification(row);
  const badges = [];
  if (cls.tier != null) badges.push(`<span class="badge badge-tier">T${escapeHtml(cls.tier)}</span>`);
  if (cls.n_stories != null) badges.push(`<span class="badge">${escapeHtml(cls.n_stories)}st</span>`);
  if (cls.has_half_height) badges.push('<span class="badge badge-half">1/2</span>');
  if (cls.has_gable) badges.push('<span class="badge badge-gable">gable</span>');
  if (cls.n_oblique > 0) badges.push(`<span class="badge badge-oblique">${escapeHtml(cls.n_oblique)}ob</span>`);
  if (cls.n_flat > 0) badges.push(`<span class="badge">${escapeHtml(cls.n_flat)}fl</span>`);
  const rating = rowRatingValue(row);
  if (typeof rating === "number") {
    badges.push(
      `<span class="badge badge-rating-${escapeHtml(rating)}" title="Roof rating ${escapeHtml(rating)}/5">★${escapeHtml(rating)}</span>`
    );
  } else if (rating === "upstream_error") {
    badges.push('<span class="badge badge-rating-error" title="Roof: upstream error">upstream</span>');
  }
  const health = scanHealth[row.uuid];
  if (health) {
    badges.push(
      `<span class="badge badge-scan-issue" title="${escapeHtml(health.summary)}">⚠ scan</span>`
    );
  }
  return badges.join("");
}

function updateSidebarStats() {
  const loaded = rows.filter((row) => row.address || row.classification || row.payload).length;
  const visible = list.querySelectorAll(".row").length;
  sidebarStats.textContent = `${visible} shown · ${rows.length} buildings · ${loaded} with loaded metadata`;
}

function updateNavButtons() {
  const canNavigate = rows.length > 1;
  navPrev.disabled = !canNavigate;
  navNext.disabled = !canNavigate;
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  if (width === lastRenderWidth && height === lastRenderHeight) return;
  lastRenderWidth = width;
  lastRenderHeight = height;
  renderRuntime.resize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function renderFrame() {
  renderQueued = false;
  resize();
  controls.update();
  renderRuntime.render();
}

function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(renderFrame);
}
if (window.__tierViewer) window.__tierViewer.requestRender = requestRender;

function populateStorySelect(payload) {
  while (storySelect.options.length > 1) storySelect.remove(1);
  for (const story of payloadStoryOptions(payload)) {
    const opt = document.createElement("option");
    opt.value = story.value;
    opt.textContent = story.label;
    storySelect.appendChild(opt);
  }
  // try to keep the prior selection if still valid; otherwise reset
  const valid = [...storySelect.options].some((o) => o.value === viewSelection);
  storySelect.value = valid ? viewSelection : "all";
  viewSelection = storySelect.value;
}

function renderHeatingLegend() {
  if (!heatingMode) {
    heatingLegend.classList.add("hidden");
    heatingLegend.innerHTML = "";
    return;
  }
  heatingLegend.classList.remove("hidden");
  const order = ["radiators", "floorHeating", "radiatorsAndFloorHeating", "unheated", "heatedUnknown"];
  heatingLegend.innerHTML = order
    .map((k) => {
      const hex = HEATING_COLORS[k].toString(16).padStart(6, "0");
      return `<div class="legend-row"><span class="swatch" style="background:#${hex}"></span><span>${HEATING_LABELS[k]}</span></div>`;
    })
    .join("");
}

explodeSlider.addEventListener("input", () => {
  viewExplodeM = Number(explodeSlider.value) || 0;
  explodeReadout.textContent = `${viewExplodeM.toFixed(1)} m`;
  setStoryExplode(scene, viewExplodeM);
  requestRender();
});

storySelect.addEventListener("change", () => {
  viewSelection = storySelect.value;
  clearSelection();
  clearMultiSelect();
  setVisibleStory(scene, viewSelection);
  requestRender();
});

heatingToggle.addEventListener("change", () => {
  heatingMode = heatingToggle.checked;
  renderHeatingLegend();
  if (activePayload) {
    populateBuildingScene(scene, activePayload, { heatingMode });
    if (viewSelection !== "all") setVisibleStory(scene, viewSelection);
    if (viewExplodeM > 0) setStoryExplode(scene, viewExplodeM);
    if (hideCeilings) setCeilingsVisible(scene, false);
  }
  requestRender();
});

if (hideCeilingsToggle) {
  hideCeilingsToggle.addEventListener("change", () => {
    hideCeilings = hideCeilingsToggle.checked;
    setCeilingsVisible(scene, !hideCeilings);
    requestRender();
  });
}

function framePayload(payload) {
  const box = new THREE.Box3();
  scene.traverse((obj) => {
    if (obj.userData?.tierPreview && obj.isMesh && !obj.userData.pickOnly && !obj.userData.framingIgnore) box.expandByObject(obj);
  });
  const center = box.isEmpty() ? new THREE.Vector3(payload.building_center.x, payload.building_center.y, payload.building_center.z) : box.getCenter(new THREE.Vector3());
  const size = box.isEmpty() ? new THREE.Vector3(8, 4, 8) : box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 4);
  controls.target.copy(center);
  camera.position.set(center.x + radius * 1.2, center.y + radius * 0.9, center.z + radius * 1.2);
  camera.near = Math.max(0.05, radius / 200);
  camera.far = radius * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

async function loadPayload(uuid) {
  activeUuid = uuid;
  applyRoofRatingToPanel(uuid);
  loading.classList.remove("hidden");
  currentAddress.textContent = "Loading...";
  currentMeta.textContent = uuid;
  currentMeta.dataset.uuid = uuid;
  const hashParams = new URLSearchParams(location.hash.slice(1));
  const sourceFile =
    hashParams.get("source") === "twin" ? "twin_payload.json" : "tier_payload.json";
  const response = await fetch(`${DATA_ROOT}/${uuid}/${sourceFile}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${sourceFile} missing for ${uuid}`);
  activePayload = await response.json();
  const row = rows.find((entry) => entry.uuid === uuid);
  if (row) {
    row.address = row.address || activePayload.address;
    row.classification = activePayload.classification;
    row.payload = activePayload;
    sortRows();
  }
  clearSelection();
  populateBuildingScene(scene, activePayload, { heatingMode });
  populateStorySelect(activePayload);
  if (viewSelection !== "all") {
    setVisibleStory(scene, viewSelection);
  }
  if (viewExplodeM > 0) {
    setStoryExplode(scene, viewExplodeM);
  }
  if (hideCeilings) {
    setCeilingsVisible(scene, false);
  }
  window.__tierState = {
    activeUuid: uuid,
    locatorCount: scene.userData.tierLocatorMap?.size || 0,
    firstLocator: scene.userData.tierLocatorMap ? [...scene.userData.tierLocatorMap.keys()][0] : null,
    selectedLocator: null,
  };
  framePayload(activePayload);
  status.classList.remove("locator");
  status.classList.remove("copy-failed");
  status.textContent = "Click a mesh to inspect it. Right-click to copy its locator ID. Click the UUID above to copy it.";
  const cls = activePayload.classification || {};
  const title = activePayload.address || activePayload.uuid;
  currentAddress.textContent = title;
  currentMeta.textContent = `${activePayload.uuid} · Tier ${cls.tier ?? "-"} · ${cls.tier_label || "Unclassified"}`;
  currentMeta.dataset.uuid = activePayload.uuid;
  currentMeta.title = "Click to copy building UUID";
  const health = scanHealth[uuid];
  const baseLabel = `${cls.tier_label || "Unclassified"} · ${cls.roof_type || "unknown roof"}`;
  if (health) {
    pill.textContent = `⚠ ${health.summary} · ${baseLabel}`;
    pill.title = `Scan-time AR / pipeline issues observed:\n${health.summary}`;
    pill.classList.add("pill-scan-issue");
  } else {
    pill.textContent = baseLabel;
    pill.title = "";
    pill.classList.remove("pill-scan-issue");
  }
  const storyLabels = activePayload.story_labels;
  const storeyDisplay = (storyLabels && storyLabels.length > 0)
    ? storyLabels.join(" · ")
    : (cls.n_stories ?? "-");
  const rq = activePayload.roof_quality;
  const rqDisplay = rq
    ? `<span title="${escapeHtml(rq.quality_version || "")}">${rq.predicted_rating.toFixed(1)}/5 · ${rq.score.toFixed(2)}</span>`
    : "-";
  signals.innerHTML = `
    <div class="signal-row"><span class="key">Storeys</span><span class="value">${escapeHtml(storeyDisplay)}</span></div>
    <div class="signal-row"><span class="key">Rooms</span><span class="value">${escapeHtml(cls.n_rooms ?? "-")}</span></div>
    <div class="signal-row"><span class="key">Oblique</span><span class="value">${escapeHtml(cls.n_oblique ?? "-")}</span></div>
    <div class="signal-row"><span class="key">Flat</span><span class="value">${escapeHtml(cls.n_flat ?? "-")}</span></div>
    <div class="signal-row"><span class="key">Half-height</span><span class="value">${cls.has_half_height ? "yes" : "no"}</span></div>
    <div class="signal-row"><span class="key">Gable</span><span class="value">${cls.has_gable ? "yes" : "no"}</span></div>
    <div class="signal-row"><span class="key">Predicted rating</span><span class="value">${rqDisplay}</span></div>
  `;
  const preservedHash = new URLSearchParams(location.hash.slice(1));
  preservedHash.set("b", uuid);
  history.replaceState(null, "", `${location.pathname}${location.search}#${preservedHash.toString()}`);
  renderList(search.value.includes("::") ? "" : search.value);
  const activeRow = list.querySelector(`.row[data-uuid="${CSS.escape(uuid)}"]`);
  if (activeRow) {
    activeRow.classList.add("active");
    activeRow.scrollIntoView({ block: "nearest" });
  }
  loading.classList.add("hidden");
  requestRender();
  clearMultiSelect();
  renderFlagPanel();
  loadAutoScanForBuilding(uuid);
  resetPreviews(scene);
  window.dispatchEvent(new CustomEvent("tier:building-loaded", { detail: { uuid } }));
}

async function hydrateSidebarMetadata(limit = rows.length) {
  const candidates = rows.filter((row) => !row.address && !row.classification).slice(0, limit);
  let updated = false;
  for (const row of candidates) {
    if (row.uuid === activeUuid && activePayload) continue;
    try {
      const response = await fetch(`${DATA_ROOT}/${row.uuid}/tier_payload.json`, { cache: "force-cache" });
      if (!response.ok) continue;
      const payload = await response.json();
      row.address = payload.address;
      row.classification = payload.classification;
      row.payload = payload;
      updated = true;
    } catch (error) {
      console.debug("sidebar metadata fetch failed", row.uuid, error);
    }
  }
  if (updated) {
    sortRows();
    if (!search.value.includes("::")) renderList(search.value);
  }
}

function renderList(filter = "") {
  const needle = filter.trim().toLowerCase();
  list.textContent = "";
  for (const row of rows) {
    if (needle && !rowSearchText(row).includes(needle)) continue;
    if (!rowMatchesTierFilter(row)) continue;
    if (!rowMatchesRatingFilter(row)) continue;
    const button = document.createElement("button");
    button.className = `row${row.uuid === activeUuid ? " active" : ""}`;
    button.dataset.uuid = row.uuid;
    const meta = renderBadges(row);
    button.innerHTML = `
      <span class="label">${escapeHtml(rowLabel(row))}</span>
      <span class="uuid">${escapeHtml(shortUuid(row.uuid))} · ${escapeHtml(row.uuid)}</span>
      <span class="row-meta">${meta}</span>
    `;
    button.addEventListener("click", () => loadPayload(row.uuid).catch(showError));
    list.appendChild(button);
  }
  updateSidebarStats();
  updateNavButtons();
}

function showError(error) {
  console.error(error);
  loading.classList.add("hidden");
  status.classList.remove("copy-failed");
  status.textContent = error.message || String(error);
  currentAddress.textContent = "Unable to load building";
  currentMeta.textContent = error.message || String(error);
  currentMeta.dataset.uuid = "";
}

function meshesUnderPointer(event) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  raycaster.setFromCamera(pointer, camera);
  return raycaster.intersectObjects(scene.children, true);
}

function locatorHit(event) {
  return meshesUnderPointer(event).find((item) => item.object.userData?.elementLocator);
}

function clearSelection() {
  if (!selectedHelper) return;
  selectedHelper.removeFromParent();
  selectedHelper.geometry?.dispose?.();
  selectedHelper.material?.dispose?.();
  selectedHelper = null;
}

function showLocator(uid, options = {}) {
  status.classList.add("locator");
  status.classList.remove("copy-failed");
  status.textContent = uid;
  if (window.__tierState) window.__tierState.selectedLocator = uid;
  if (options.copyState) {
    showCopyNotice("element UUID", uid, options.copyState === "copied");
  }
}

function createSelectionHelper(mesh) {
  const selectionBox = mesh.userData?.selectionBox;
  if (!selectionBox) return new THREE.BoxHelper(mesh, 0xffcc00);
  const boxGeometry = new THREE.BoxGeometry(
    selectionBox.size.x,
    selectionBox.size.y,
    selectionBox.size.z,
  );
  const edges = new THREE.EdgesGeometry(boxGeometry);
  boxGeometry.dispose();
  const helper = new THREE.LineSegments(edges, SELECTION_MATERIAL.clone());
  helper.renderOrder = 1000;
  const basis = selectionBox.basis;
  const matrix = new THREE.Matrix4().makeBasis(basis.u, basis.v, basis.n);
  helper.quaternion.setFromRotationMatrix(matrix);
  helper.position.copy(selectionBox.center);
  return helper;
}

function selectLocator(uid, options = {}) {
  const mesh = scene.userData.tierLocatorMap?.get(uid);
  if (!mesh) return false;
  clearSelection();
  selectedHelper = createSelectionHelper(mesh);
  selectedHelper.userData.tierPreview = true;
  selectedHelper.userData.selectionHelper = true;
  scene.add(selectedHelper);
  const box = new THREE.Box3().setFromObject(mesh);
  const center = box.getCenter(new THREE.Vector3());
  if (options.focus) {
    controls.target.copy(center);
    camera.position.lerp(new THREE.Vector3(center.x + 6, center.y + 4, center.z + 6), 0.8);
    controls.update();
  }
  showLocator(uid, options);
  requestRender();
  return true;
}

async function copyAndSelectLocator(uid, options = {}) {
  const copied = await copyText(uid);
  selectLocator(uid, { ...options, copyState: copied ? "copied" : "failed" });
}

canvas.addEventListener("click", (event) => {
  const hit = locatorHit(event);
  if (!hit) {
    if (!event.metaKey && !event.ctrlKey) clearMultiSelect();
    return;
  }
  const uid = hit.object.userData.elementLocator;
  if (event.metaKey || event.ctrlKey) {
    toggleMultiSelect(uid);
    selectLocator(uid);
    return;
  }
  clearMultiSelect();
  selectLocator(uid);
});

canvas.addEventListener("contextmenu", async (event) => {
  event.preventDefault();
  const hit = locatorHit(event);
  if (hit && event.shiftKey) {
    const uid = hit.object.userData.elementLocator;
    selectLocator(uid);
    const targets =
      multiSelection.size >= 2 && multiSelection.has(uid)
        ? Array.from(multiSelection)
        : [uid];
    openFlagPopover(targets, event.clientX, event.clientY);
    return;
  }
  if (hit && (event.metaKey || event.ctrlKey)) {
    // Preserve existing copy-on-modifier shortcut.
    const uid = hit.object.userData.elementLocator;
    await copyAndSelectLocator(uid);
    return;
  }

  scene.userData.selectLocator = (uid, options) => selectLocator(uid, options);
  scene.userData.requestRender = requestRender;

  if (hit) {
    const uid = hit.object.userData.elementLocator;
    selectLocator(uid);
    const targets =
      multiSelection.size >= 2 && multiSelection.has(uid)
        ? Array.from(multiSelection)
        : [uid];
    openContextMenu("element", {
      scene,
      uuid: activeUuid,
      targets,
      x: event.clientX,
      y: event.clientY,
      requestRender,
    });
  } else {
    openContextMenu("building", {
      scene,
      uuid: activeUuid,
      targets: [],
      x: event.clientX,
      y: event.clientY,
      requestRender,
    });
  }
});

currentMeta.addEventListener("click", async () => {
  const uuid = currentMeta.dataset.uuid || activeUuid;
  if (!uuid) return;
  showCopyNotice("building UUID", uuid, await copyText(uuid));
});

currentMeta.addEventListener("keydown", async (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  const uuid = currentMeta.dataset.uuid || activeUuid;
  if (!uuid) return;
  showCopyNotice("building UUID", uuid, await copyText(uuid));
});

search.addEventListener("keydown", async (event) => {
  if (event.key !== "Enter") return;
  const value = search.value.trim();
  const parsed = parseElementUid(value);
  if (parsed) {
    if (parsed.uuid !== activeUuid) await loadPayload(parsed.uuid);
    if (!selectLocator(value, { focus: true })) {
      status.classList.remove("locator");
      status.textContent = `No mesh for ${value}`;
    }
    return;
  }
  const found = rows.find((row) => rowSearchText(row).includes(value.toLowerCase()));
  if (found) await loadPayload(found.uuid);
});
search.addEventListener("input", () => renderList(search.value));

function attachPillFilter(rootEl, applyFn) {
  if (!rootEl) return;
  rootEl.querySelectorAll("button.pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      pill.classList.toggle("active");
      const selected = new Set(
        Array.from(rootEl.querySelectorAll("button.pill.active")).map(
          (el) => el.dataset.value,
        ),
      );
      applyFn(selected);
      renderList(search.value);
    });
  });
}

attachPillFilter(filterTierEl, (s) => {
  filterTiers = s;
});
attachPillFilter(filterRatingEl, (s) => {
  filterRatings = s;
});
sortModeSelect?.addEventListener("change", () => {
  sortMode = sortModeSelect.value;
  sortRows();
  renderList(search.value);
});

function visibleRows() {
  if (search.value.includes("::")) return rows;
  const needle = search.value.trim().toLowerCase();
  return rows.filter((row) => {
    if (needle && !rowSearchText(row).includes(needle)) return false;
    if (!rowMatchesTierFilter(row)) return false;
    if (!rowMatchesRatingFilter(row)) return false;
    return true;
  });
}

function step(delta) {
  const visible = visibleRows();
  if (!visible.length) return;
  const currentIndex = Math.max(0, visible.findIndex((row) => row.uuid === activeUuid));
  const nextIndex = (currentIndex + delta + visible.length) % visible.length;
  loadPayload(visible[nextIndex].uuid).catch(showError);
}

navPrev.addEventListener("click", () => step(-1));
navNext.addEventListener("click", () => step(1));
window.addEventListener("keydown", (event) => {
  const plainKey = !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey;
  const isTypingTarget = event.target instanceof HTMLInputElement
    || event.target instanceof HTMLTextAreaElement
    || event.target?.isContentEditable;
  if (plainKey && event.key === "ArrowUp") {
    event.preventDefault();
    step(-1);
  } else if (plainKey && event.key === "ArrowDown") {
    event.preventDefault();
    step(1);
  } else if (!isTypingTarget && (event.key === "k" || event.key === "j")) {
    event.preventDefault();
    step(event.key === "k" ? -1 : 1);
  }
}, { capture: true });

// ---- flag queue (handoff to triage) ----------------------------------------

const flagPanel = document.querySelector("#flag-panel");
const flagList = document.querySelector("#flag-list");
const flagCount = document.querySelector("#flag-panel-count");
const flagToggle = document.querySelector("#flag-panel-toggle");
const flagSendBtn = document.querySelector("#flag-send");
const flagClearBtn = document.querySelector("#flag-clear");
const flagStatus = document.querySelector("#flag-status");

// Per-building queue: { [uuid]: [{ id, locator, kind, parts, rule, note, severity, evidence, dismissed }] }
const flagQueueByUuid = new Map();

function currentQueue() {
  if (!activeUuid) return [];
  if (!flagQueueByUuid.has(activeUuid)) flagQueueByUuid.set(activeUuid, []);
  return flagQueueByUuid.get(activeUuid);
}

function flagId() {
  return Math.random().toString(36).slice(2, 14);
}

function addManualFlag(locator, note = null) {
  if (!activeUuid || !locator) return;
  const queue = currentQueue();
  const existing = queue.find((item) => item.locator === locator && !item.rule && !item.dismissed);
  if (existing) {
    if (note && note !== existing.note) {
      existing.note = note;
      renderFlagPanel();
      flashFlagStatus(`updated note for ${locator}`);
      return;
    }
    flashFlagStatus(`already flagged: ${locator}`);
    return;
  }
  const parsed = parseElementUid(locator) || { uuid: activeUuid, scope: "", parts: [] };
  queue.push({
    id: flagId(),
    locator,
    kind: parsed.scope ? `tier-${parsed.scope}` : "",
    parts: parsed.parts || [],
    rule: null,
    note: note || null,
    severity: null,
    evidence: null,
    dismissed: false,
  });
  renderFlagPanel();
  flashFlagStatus(`added (${queue.filter((it) => !it.dismissed).length} in queue)`);
  if (flagPanel.classList.contains("collapsed")) flagPanel.classList.remove("collapsed");
}

function syncFlagOverlays() {
  while (flagOverlayGroup.children.length) {
    const child = flagOverlayGroup.children[0];
    flagOverlayGroup.remove(child);
    child.geometry?.dispose?.();
  }
  if (!activeUuid) {
    requestRender();
    return;
  }
  const locatorMap = scene.userData.tierLocatorMap;
  if (!locatorMap) {
    requestRender();
    return;
  }
  for (const item of currentQueue()) {
    if (item.dismissed) continue;
    const mesh = locatorMap.get(item.locator);
    if (!mesh) continue;
    const helper = createFlagHelper(mesh, item.rule ? FLAG_AUTO_MATERIAL : FLAG_MANUAL_MATERIAL);
    if (!helper) continue;
    helper.userData.tierPreview = true;
    helper.userData.flagHelper = true;
    helper.userData.flagId = item.id;
    flagOverlayGroup.add(helper);
  }
  requestRender();
}

function createFlagHelper(mesh, material) {
  const selectionBox = mesh.userData?.selectionBox;
  if (selectionBox) {
    const geometry = new THREE.BoxGeometry(selectionBox.size.x, selectionBox.size.y, selectionBox.size.z);
    const edges = new THREE.EdgesGeometry(geometry);
    geometry.dispose();
    const helper = new THREE.LineSegments(edges, material);
    const matrix = new THREE.Matrix4().makeBasis(selectionBox.basis.u, selectionBox.basis.v, selectionBox.basis.n);
    helper.quaternion.setFromRotationMatrix(matrix);
    helper.position.copy(selectionBox.center);
    return helper;
  }
  // Fallback: world-aligned bounding box.
  const box = new THREE.Box3().setFromObject(mesh);
  if (box.isEmpty()) return null;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const geometry = new THREE.BoxGeometry(size.x || 0.05, size.y || 0.05, size.z || 0.05);
  const edges = new THREE.EdgesGeometry(geometry);
  geometry.dispose();
  const helper = new THREE.LineSegments(edges, material);
  helper.position.copy(center);
  return helper;
}

// ---- multi-select (Cmd/Ctrl+click) -----------------------------------------

function toggleMultiSelect(uid) {
  if (multiSelection.has(uid)) multiSelection.delete(uid);
  else multiSelection.add(uid);
  syncMultiSelectOverlays();
  flashFlagStatus(
    multiSelection.size > 0
      ? `${multiSelection.size} selected — Shift+right-click to flag the group`
      : "selection cleared",
  );
}

function clearMultiSelect() {
  if (multiSelection.size === 0) return;
  multiSelection.clear();
  syncMultiSelectOverlays();
}

function syncMultiSelectOverlays() {
  while (multiSelectGroup.children.length) {
    const child = multiSelectGroup.children[0];
    multiSelectGroup.remove(child);
    child.geometry?.dispose?.();
  }
  if (!activeUuid) {
    requestRender();
    return;
  }
  const locatorMap = scene.userData.tierLocatorMap;
  if (!locatorMap) {
    requestRender();
    return;
  }
  for (const uid of multiSelection) {
    const mesh = locatorMap.get(uid);
    if (!mesh) continue;
    const helper = createFlagHelper(mesh, MULTI_SELECT_MATERIAL);
    if (helper) multiSelectGroup.add(helper);
  }
  requestRender();
}

// ---- flag popover (Shift+right-click → flag with note) ---------------------

const flagPopover = document.querySelector("#flag-popover");
const flagPopoverLocator = document.querySelector("#flag-popover-locator");
const flagPopoverNote = document.querySelector("#flag-popover-note");
const flagPopoverAdd = document.querySelector("#flag-popover-add");
const flagPopoverCancel = document.querySelector("#flag-popover-cancel");
let popoverLocators = [];

function openFlagPopover(locators, screenX, screenY) {
  popoverLocators = Array.isArray(locators) ? [...locators] : [locators];
  if (!popoverLocators.length) return;
  const count = popoverLocators.length;
  if (count === 1) {
    flagPopoverLocator.textContent = popoverLocators[0];
  } else {
    const preview = popoverLocators.slice(0, 3).map(escapeAttr).join("<br>");
    const more = count > 3 ? `<br>… +${count - 3} more` : "";
    flagPopoverLocator.innerHTML = `<strong>${count} elements</strong> (one shared note)<br>${preview}${more}`;
  }
  flagPopoverAdd.textContent = count > 1 ? `Add ${count} to queue` : "Add to queue";
  flagPopoverNote.value = "";
  flagPopoverNote.placeholder =
    count > 1
      ? "Why do these look wrong, or what's missing? (shared note for all)"
      : "Why does this look wrong? (optional — Enter to add, Esc to cancel)";
  flagPopover.classList.remove("hidden");
  const rect = main.getBoundingClientRect();
  const popWidth = 280;
  const popHeight = 160;
  let x = screenX - rect.left + 12;
  let y = screenY - rect.top + 12;
  if (x + popWidth > rect.width - 8) x = rect.width - popWidth - 8;
  if (y + popHeight > rect.height - 8) y = rect.height - popHeight - 8;
  if (x < 8) x = 8;
  if (y < 8) y = 8;
  flagPopover.style.left = `${x}px`;
  flagPopover.style.top = `${y}px`;
  setTimeout(() => flagPopoverNote.focus(), 0);
}

function closeFlagPopover() {
  flagPopover.classList.add("hidden");
  popoverLocators = [];
}

function commitFlagPopover() {
  if (!popoverLocators.length) return;
  const note = flagPopoverNote.value.trim() || null;
  const locators = popoverLocators;
  closeFlagPopover();
  for (const locator of locators) {
    addManualFlag(locator, note);
  }
  if (locators.length > 1) clearMultiSelect();
}

flagPopoverAdd.addEventListener("click", commitFlagPopover);
flagPopoverCancel.addEventListener("click", closeFlagPopover);
flagPopoverNote.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    commitFlagPopover();
  } else if (event.key === "Escape") {
    event.preventDefault();
    closeFlagPopover();
  }
});
document.addEventListener("mousedown", (event) => {
  if (flagPopover.classList.contains("hidden")) return;
  if (flagPopover.contains(event.target)) return;
  closeFlagPopover();
});
window.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!flagPopover.classList.contains("hidden")) return; // popover handles its own Esc
  const isTypingTarget =
    event.target instanceof HTMLInputElement ||
    event.target instanceof HTMLTextAreaElement ||
    event.target?.isContentEditable;
  if (isTypingTarget) return;
  if (multiSelection.size > 0) {
    clearMultiSelect();
    flashFlagStatus("selection cleared");
  }
});

const diffByUuid = new Map(); // uuid → diff object from latest auto-scan
const diffFilterByUuid = new Map(); // uuid → "fixed" | "persisting" | "new" | null

async function loadAutoScanForBuilding(uuid) {
  if (!uuid) return;
  // Only fetch the first time we see this uuid; manual flags are kept across visits.
  if (flagQueueByUuid.has(uuid) && flagQueueByUuid.get(uuid).some((it) => it.rule)) return;
  try {
    const response = await fetch(`/flag-queues?uuid=${encodeURIComponent(uuid)}`, { cache: "no-store" });
    if (!response.ok) return;
    const data = await response.json();
    const items = Array.isArray(data?.items) ? data.items : [];
    if (data?.diff) diffByUuid.set(uuid, data.diff);
    if (!items.length) {
      if (uuid === activeUuid) renderFlagPanel();
      return;
    }
    if (!flagQueueByUuid.has(uuid)) flagQueueByUuid.set(uuid, []);
    const queue = flagQueueByUuid.get(uuid);
    const seen = new Set(queue.map((it) => `${it.locator}::${it.rule || ""}`));
    for (const item of items) {
      const key = `${item.locator}::${item.rule || ""}`;
      if (seen.has(key)) continue;
      queue.push({ ...item, dismissed: !!item.dismissed });
      seen.add(key);
    }
    if (uuid === activeUuid) renderFlagPanel();
  } catch (error) {
    console.debug("auto-scan fetch failed", uuid, error);
  }
}

function currentDiff() {
  return activeUuid ? diffByUuid.get(activeUuid) || null : null;
}

function currentDiffFilter() {
  return activeUuid ? diffFilterByUuid.get(activeUuid) || null : null;
}

function setDiffFilter(bucket) {
  if (!activeUuid) return;
  const current = diffFilterByUuid.get(activeUuid) || null;
  diffFilterByUuid.set(activeUuid, current === bucket ? null : bucket);
  renderFlagPanel();
}

function diffBucketFor(item, diff) {
  if (!diff || !item.rule || !item.locator) return null;
  const key = (it) => it.locator === item.locator && it.rule === item.rule;
  if (diff.new?.some(key)) return "new";
  if (diff.fixed?.some(key)) return "fixed";
  if (diff.persisting?.some(key)) return "persisting";
  return null;
}

function renderDiffBanner(diff) {
  const banner = document.querySelector("#flag-diff-banner");
  if (!banner) return;
  if (!diff) {
    banner.classList.add("hidden");
    banner.innerHTML = "";
    return;
  }
  const fixedN = diff.fixed?.length || 0;
  const persistingN = diff.persisting?.length || 0;
  const newN = diff.new?.length || 0;
  if (fixedN === 0 && persistingN === 0 && newN === 0) {
    banner.classList.add("hidden");
    banner.innerHTML = "";
    return;
  }
  const filter = currentDiffFilter();
  const stripe = (bucket, count, label) => {
    const empty = count === 0;
    const active = filter === bucket;
    return `<button type="button" class="diff-stripe ${bucket}${active ? " active" : ""}" data-bucket="${bucket}" data-empty="${empty}" ${empty ? "tabindex=\"-1\"" : ""}>${count} ${label}</button>`;
  };
  banner.classList.remove("hidden");
  const headline = diff.compared_to
    ? `vs <span title="${escapeAttr(diff.compared_to)}">${escapeAttr(diff.compared_to_created || diff.compared_to)}</span>`
    : "vs prior scan";
  banner.innerHTML = `
    <div class="diff-headline">${headline}</div>
    <div class="diff-stripes">
      ${stripe("fixed", fixedN, "fixed")}
      ${stripe("persisting", persistingN, "persisting")}
      ${stripe("new", newN, "new")}
    </div>
    ${filter ? `<button type="button" class="diff-clear">show all</button>` : ""}
  `;
}

document.querySelector("#flag-diff-banner").addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  if (target.classList.contains("diff-clear")) {
    if (activeUuid) diffFilterByUuid.set(activeUuid, null);
    renderFlagPanel();
    return;
  }
  const stripe = target.closest(".diff-stripe");
  if (!stripe) return;
  if (stripe.dataset.empty === "true") return;
  setDiffFilter(stripe.dataset.bucket);
});

function escapeAttr(value) {
  return String(value || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function evidenceSummary(item) {
  if (!item.evidence) return "";
  const ev = item.evidence;
  const pick = (key) => (ev[key] !== undefined ? `${key}=${typeof ev[key] === "number" ? ev[key].toFixed(3) : ev[key]}` : null);
  const parts = [
    pick("normal_y"),
    pick("delta_below_m"),
    pick("outside_ratio"),
    pick("outside_area_xz_m2"),
    pick("coverage_ratio"),
    pick("delta_m"),
    pick("nearest_oblique_distance_m"),
  ].filter(Boolean);
  return parts.slice(0, 2).join(" · ");
}

function whyPanelInner(item) {
  if (item.rule && item.hypothesis) {
    const files = (item.hypothesis.files || [])
      .map((f) => `<button type="button" class="why-file" data-action="copy-file">${escapeAttr(f)}</button>`)
      .join("");
    return `
      <div class="why-summary">${escapeAttr(item.hypothesis.summary || "")}</div>
      ${item.hypothesis.step ? `<div class="why-step">likely step: ${escapeAttr(item.hypothesis.step)}</div>` : ""}
      ${files ? `<div class="why-files">${files}</div>` : ""}
    `;
  }
  // Manual flag — no hypothesis. Use the note above as the why.
  return `<div class="why-summary">Manual flag — kind <code>${escapeAttr(item.kind || "?")}</code>. Use the note above to record what looks wrong.</div>`;
}

function renderFlagPanel() {
  const queue = currentQueue();
  const visibleCount = queue.filter((it) => !it.dismissed).length;
  flagCount.textContent = String(visibleCount);
  flagCount.classList.toggle("empty", visibleCount === 0);
  // Enable Send whenever the queue has any entries — dismissals also commit.
  flagSendBtn.disabled = queue.length === 0;
  syncFlagOverlays();
  const diff = currentDiff();
  renderDiffBanner(diff);
  const filter = currentDiffFilter();
  flagList.innerHTML = "";
  for (const item of queue) {
    const row = document.createElement("li");
    const bucket = diffBucketFor(item, diff);
    const classes = ["flag-row"];
    if (item.dismissed) classes.push("dismissed");
    if (bucket) classes.push(`diff-${bucket}`);
    if (filter && bucket !== filter && item.rule) classes.push("diff-hidden");
    row.className = classes.join(" ");
    row.dataset.flagId = item.id;
    const ruleBadge = item.rule
      ? `<span class="flag-rule">${escapeAttr(item.rule)}</span>`
      : `<span class="flag-rule manual">manual</span>`;
    const evidence = evidenceSummary(item);
    const hasWhy = !!(item.rule && item.hypothesis) || !item.rule;
    row.innerHTML = `
      <div class="flag-locator">${ruleBadge}<span>${escapeAttr(item.locator)}</span></div>
      <div class="flag-actions-row">
        ${hasWhy ? `<button type="button" class="flag-why-toggle" data-action="why">why?</button>` : ""}
        <button type="button" class="flag-mini" data-action="goto" title="Select in viewer">↗</button>
        <button type="button" class="flag-mini" data-action="toggle" title="${item.dismissed ? "Restore" : "Dismiss"}">${item.dismissed ? "↺" : "✕"}</button>
      </div>
      ${evidence ? `<div class="flag-evidence">${escapeAttr(evidence)}</div>` : ""}
      <textarea data-action="note" placeholder="note (optional)">${escapeAttr(item.note || "")}</textarea>
      <div class="flag-why hidden">${whyPanelInner(item)}</div>
    `;
    flagList.appendChild(row);
  }
}

flagList.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const row = target.closest(".flag-row");
  if (!row) return;
  const id = row.dataset.flagId;
  const item = currentQueue().find((it) => it.id === id);
  if (!item) return;
  const action = target.dataset.action;
  if (action === "goto") {
    selectLocator(item.locator, { focus: true });
  } else if (action === "toggle") {
    item.dismissed = !item.dismissed;
    renderFlagPanel();
  } else if (action === "why") {
    const panel = row.querySelector(".flag-why");
    if (panel) panel.classList.toggle("hidden");
  } else if (action === "copy-file") {
    const path = target.textContent || "";
    const ok = await copyText(path);
    flashFlagStatus(ok ? `copied ${path}` : `copy failed`);
  }
});

flagList.addEventListener("input", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLTextAreaElement)) return;
  if (target.dataset.action !== "note") return;
  const row = target.closest(".flag-row");
  if (!row) return;
  const item = currentQueue().find((it) => it.id === row.dataset.flagId);
  if (!item) return;
  item.note = target.value || null;
});

flagToggle.addEventListener("click", () => {
  flagPanel.classList.toggle("collapsed");
});

flagClearBtn.addEventListener("click", () => {
  if (!activeUuid) return;
  flagQueueByUuid.set(activeUuid, []);
  renderFlagPanel();
  flashFlagStatus("cleared");
});

flagSendBtn.addEventListener("click", async () => {
  if (!activeUuid) return;
  const queue = currentQueue();
  // Send the full queue — including dismissed entries — so the server can fold
  // dismissals into the calibration file. Dismissed items are filtered out
  // when the queue file is written; only active flags reach the triage queue.
  const items = queue;
  if (!items.length) return;
  flagSendBtn.disabled = true;
  flagStatus.classList.remove("error", "success");
  flagStatus.textContent = "sending...";
  let screenshot = null;
  try {
    requestRender();
    await new Promise((r) => requestAnimationFrame(r));
    screenshot = canvas.toDataURL("image/png");
  } catch (error) {
    console.debug("screenshot capture failed", error);
  }
  try {
    const response = await fetch("/flag-queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        building_uuid: activeUuid,
        items,
        screenshot_data_url: screenshot,
        source: items.some((it) => it.rule) && items.some((it) => !it.rule) ? "merged" : (items.every((it) => it.rule) ? "merged" : "viewer"),
      }),
    });
    if (!response.ok) throw new Error(`server returned ${response.status}`);
    const result = await response.json();
    flagQueueByUuid.set(activeUuid, []);
    renderFlagPanel();
    flagStatus.classList.add("success");
    flagStatus.textContent = `sent → ${result.queue_path}`;
  } catch (error) {
    console.error("flag-queue send failed", error);
    flagStatus.classList.add("error");
    flagStatus.textContent = `send failed: ${error.message}`;
    flagSendBtn.disabled = false;
  }
});

let flashTimer = null;
function flashFlagStatus(message) {
  flagStatus.classList.remove("error", "success");
  flagStatus.textContent = message;
  if (flashTimer) clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {
    if (flagStatus.textContent === message) flagStatus.textContent = "";
  }, 2400);
}

const RATING_REASON_VALUES = new Set([
  "fragments",
  "missing_dormer",
  "wrong_ridge",
  "splayed_wing",
  "daylight_through",
  "half_shell",
  "other",
]);
const UPSTREAM_REASON_VALUES = new Set([
  "viewer_broken_render",
  "extraction_incomplete",
  "not_my_building",
  "other",
]);

function applyRoofRatingToPanel(uuid) {
  const panel = document.getElementById("roof-rating-panel");
  if (!panel) return;
  const record = uuid ? roofRatingsByUuid[uuid] : null;
  const current =
    record && Object.prototype.hasOwnProperty.call(record, "rating") ? record.rating : null;
  panel.querySelectorAll("button.rating-btn").forEach((btn) => {
    const raw = btn.dataset.rating;
    const value = raw === "upstream_error" ? "upstream_error" : Number(raw);
    btn.classList.toggle("active", current !== null && current === value);
  });
  const statusEl = document.getElementById("roof-rating-status");
  if (statusEl) {
    if (current === null) {
      statusEl.textContent = "";
      statusEl.title = "";
    } else if (current === "upstream_error") {
      statusEl.textContent = "upstream";
      statusEl.title = `Saved ${record?.updated_at || ""}`;
    } else {
      statusEl.textContent = `${current}/5`;
      statusEl.title = `Saved ${record?.updated_at || ""}`;
    }
  }

  const reasonSelect = document.getElementById("roof-rating-reason");
  const errorReasonSelect = document.getElementById("roof-rating-error-reason");
  const savedReason = record && typeof record.reason === "string" ? record.reason : "";
  const lowRating = typeof current === "number" && current >= 1 && current <= 3;
  const isUpstream = current === "upstream_error";
  if (reasonSelect) {
    reasonSelect.classList.toggle("hidden", !lowRating);
    reasonSelect.value = lowRating && RATING_REASON_VALUES.has(savedReason) ? savedReason : "";
  }
  if (errorReasonSelect) {
    errorReasonSelect.classList.toggle("hidden", !isUpstream);
    errorReasonSelect.value =
      isUpstream && UPSTREAM_REASON_VALUES.has(savedReason) ? savedReason : "";
  }
}

function refreshSidebarAfterRatingChange() {
  if (sortMode === "rating-asc" || sortMode === "rating-desc") sortRows();
  renderList(search.value);
}

async function submitRoofRating(rating, options = {}) {
  const uuid = activeUuid;
  if (!uuid) return;
  const prior = roofRatingsByUuid[uuid] || null;
  const priorRating = prior ? prior.rating : null;
  const explicitReason = Object.prototype.hasOwnProperty.call(options, "reason");
  const nextRating = !explicitReason && priorRating === rating ? null : rating;
  let reason = null;
  if (nextRating !== null) {
    if (explicitReason) {
      reason = options.reason || null;
    } else if (typeof nextRating === "number" && nextRating >= 1 && nextRating <= 3) {
      const sel = document.getElementById("roof-rating-reason");
      reason = sel && sel.value ? sel.value : null;
    } else if (nextRating === "upstream_error") {
      const sel = document.getElementById("roof-rating-error-reason");
      reason = sel && sel.value ? sel.value : null;
    }
  }

  if (nextRating === null) {
    delete roofRatingsByUuid[uuid];
  } else {
    const optimistic = { rating: nextRating, updated_at: "..." };
    if (reason) optimistic.reason = reason;
    roofRatingsByUuid[uuid] = optimistic;
  }
  applyRoofRatingToPanel(uuid);
  refreshSidebarAfterRatingChange();
  try {
    const body = { uuid, rating: nextRating };
    if (reason) body.reason = reason;
    const resp = await fetch("/roof-rating", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (resp.ok) {
      const respBody = await resp.json().catch(() => ({}));
      if (respBody && respBody.record) {
        roofRatingsByUuid[uuid] = respBody.record;
      } else {
        delete roofRatingsByUuid[uuid];
      }
    } else if (prior) {
      roofRatingsByUuid[uuid] = prior;
    } else {
      delete roofRatingsByUuid[uuid];
    }
  } catch (_err) {
    if (prior) roofRatingsByUuid[uuid] = prior;
    else delete roofRatingsByUuid[uuid];
  }
  applyRoofRatingToPanel(uuid);
  refreshSidebarAfterRatingChange();
}

const roofRatingPanel = document.getElementById("roof-rating-panel");
roofRatingPanel?.addEventListener("pointerdown", (event) => event.stopPropagation());
roofRatingPanel?.addEventListener("click", (event) => event.stopPropagation());
roofRatingPanel?.addEventListener("contextmenu", (event) => event.stopPropagation());

document.querySelectorAll("#roof-rating-panel button.rating-btn").forEach((btn) => {
  btn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const raw = btn.dataset.rating;
    const value = raw === "upstream_error" ? "upstream_error" : Number(raw);
    submitRoofRating(value);
  });
});

const reasonSelectEl = document.getElementById("roof-rating-reason");
if (reasonSelectEl) {
  reasonSelectEl.addEventListener("change", () => {
    const uuid = activeUuid;
    if (!uuid) return;
    const record = roofRatingsByUuid[uuid];
    if (!record || typeof record.rating !== "number") return;
    submitRoofRating(record.rating, { reason: reasonSelectEl.value || null });
  });
}
const errorReasonSelectEl = document.getElementById("roof-rating-error-reason");
if (errorReasonSelectEl) {
  errorReasonSelectEl.addEventListener("change", () => {
    const uuid = activeUuid;
    if (!uuid) return;
    const record = roofRatingsByUuid[uuid];
    if (!record || record.rating !== "upstream_error") return;
    submitRoofRating("upstream_error", { reason: errorReasonSelectEl.value || null });
  });
}

async function init() {
  rows = await loadRows();
  try {
    const ratingsResp = await fetch("/roof-rating", { cache: "no-store" });
    if (ratingsResp.ok) {
      const data = await ratingsResp.json();
      if (data && typeof data === "object") roofRatingsByUuid = data;
    }
  } catch (error) {
    console.debug("/roof-rating fetch failed", error);
  }
  try {
    const healthResp = await fetch(`${DATA_ROOT}/scan_health.json`, { cache: "no-store" });
    if (healthResp.ok) scanHealth = normalizeScanHealthMap(await healthResp.json());
  } catch (error) {
    console.debug("scan_health.json fetch failed", error);
  }
  sortRows();
  renderList();
  const hash = new URLSearchParams(location.hash.slice(1));
  const uuid = hash.get("b") || rows[0]?.uuid;
  if (uuid) {
    await loadPayload(uuid);
    hydrateSidebarMetadata().catch((error) => console.debug("metadata hydration failed", error));
  }
  else {
    status.textContent = "No tier payloads found";
    currentAddress.textContent = "No tier payloads found";
    currentMeta.textContent = "";
    pill.textContent = "-";
  }
  requestRender();
}

const sourceTwinToggle = document.getElementById("source-twin-toggle");
function syncSourceToggleFromHash() {
  if (!sourceTwinToggle) return;
  const hash = new URLSearchParams(location.hash.slice(1));
  sourceTwinToggle.checked = hash.get("source") === "twin";
}
sourceTwinToggle?.addEventListener("change", (event) => {
  const hash = new URLSearchParams(location.hash.slice(1));
  if (event.target.checked) hash.set("source", "twin");
  else hash.delete("source");
  if (activeUuid) hash.set("b", activeUuid);
  history.replaceState(null, "", `${location.pathname}${location.search}#${hash.toString()}`);
  if (activeUuid) loadPayload(activeUuid).catch(showError);
});

init().then(syncSourceToggleFromHash).catch(showError);
controls.addEventListener("change", requestRender);
window.addEventListener("resize", requestRender);
window.addEventListener("hashchange", () => {
  const hash = new URLSearchParams(location.hash.slice(1));
  const uuid = hash.get("b");
  if (uuid && uuid !== activeUuid) {
    loadPayload(uuid).catch(showError);
  } else if (uuid) {
    // Same uuid; user may have toggled &source=... — reload the payload.
    loadPayload(uuid).catch(showError);
  }
});
