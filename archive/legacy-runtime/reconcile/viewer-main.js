import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const VIEWER_MODULE_VERSION = '20260422d';
import {
  STORY_COLORS, STORY_WALL_COLORS, ROOM_COLORS,
  DOOR_COLOR, DOOR_EDGE, WINDOW_COLOR, WINDOW_EDGE, MERGED_COLOR, MERGED_EDGE,
  RAW_CEILING_COLOR, RAW_CEILING_EDGE,
  RAW_CEILING_ROLE_COLORS, CEILING_RECON_DORMER_COLORS, CEILING_RECON_WING_COLORS,
  COMPUTED_OVEREXTEND_COLORS,
  RAW_DISAGREEMENT_COLORS,
  RAW_CEILING_SPLIT_COLORS,
  CEILING_REPLACEMENT_COLORS,
  ROOF_CLUSTER_COLORS,
  SOURCE_COLORS, SOURCE_LABELS,
  LAYER_CONTROL_IDS, LAYER_KEYS, PIPELINE_STEPS,
  EMPTY_MAP_STYLE,
} from './viewer-modules/constants.js?v=20260422d';
import {
  createPolygonMesh, createEdgeLoop, createLine, createPolyline3,
  createTriangleMesh, createTriangleBoundaryEdges,
  disposeGroup, polygonPlaneBasis, projectToPlane2,
  collectWallCutoutHoles, orientedStructureCorners,
} from './viewer-modules/geometry.js?v=20260418newell';
import { renderRoofFromPythonResult } from './viewer-modules/roof-python.js?v=20260420c';
import {
  renderOntologySemantics,
  renderOntologyContinuationDiagnostics,
  renderOntologyExact,
} from './viewer-modules/ontology-cells.js?v=20260420c';
import { renderOntologyEnhancedFullModel } from './viewer-modules/full-model-ontology.js?v=20260420c';
import { renderV3Model, renderV3RoofProposals, renderCandidateFaces, renderReconstruction, renderRidgeEaveScoring, proposalColor } from './viewer-modules/v3-model.js?v=20260420c';
import { createOrthoMapController } from './viewer-modules/map-ortho.js?v=20260420c';
import { bindUIEventHandlers } from './viewer-modules/ui-bindings.js?v=20260418m';

const viewport = document.getElementById('viewport');
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setClearColor(0x171717);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;

function resizeRenderer() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
camera.position.set(10, 15, 10);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

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

const grid = new THREE.GridHelper(60, 60, 0x2d2d2d, 0x232323);
grid.visible = false;
scene.add(grid);

let roofClusterData = [];
let colorByStory = true;
let colorBySource = false;
let DATA = [];
let currentBuilding = 0;
let orthoEnabled = false;
let modelMapEnabled = true;
let modelMapRotationDeg = 0;
let modelMapOffsetEastM = 0;
let modelMapOffsetNorthM = 0;
let alignmentByUuid = {};
let alignmentSaveTimer = null;
let roofRatingsByUuid = {};
let anchorModeEnabled = false;
let pyRoofByUuid = {};
// Phase-2 raw-ceiling prototype sidecar. Shape:
//   { thresholds, planes: {element_id -> {role, archetype, ...}},
//     rooms: {"<uuid>::room::<story>:<ri>" -> {archetype, features, ...}},
//     reconstructions: {"<uuid>": {"dormer": [...], "wing": [...]}} }
let rawCeilingPrototype = null;
let rawCeilingPrototypePromise = null;
// Computed-surface overextend sidecar from
// scripts/audit_computed_surface_extent_vs_raw.py. Shape:
//   { buildings: {"<uuid>": [{overextend_element_id, corners, ...}]} }
let computedOverextend = null;
let computedOverextendPromise = null;
// Raw-ceiling orientation-disagreement sidecar from
// scripts/audit_raw_orientation_disagreement.py. Shape:
//   { buildings: {"<uuid>": [{element_id, corners, angle_deg, ...}]} }
let rawDisagreement = null;
let rawDisagreementPromise = null;
// Raw-eave-supported split sidecars (versioned):
//   v1: legacy scorer output
//   v2: relation-first scorer output
// Shape:
//   { buildings: {"<uuid>": [{piece_id, piece_role, corners, holes, ...}] } }
const RAW_CEILING_SPLIT_VERSIONS = ['v1', 'v2'];
const rawCeilingPlaneSplitsByVersion = { v1: null, v2: null };
const rawCeilingPlaneSplitsPromiseByVersion = { v1: null, v2: null };

const RAW_CEILING_SPLIT_VERSION_LABELS = {
  v1: 'V1',
  v2: 'V2',
};

const RAW_CEILING_SPLIT_VERSION_COLORS = {
  v1: {
    final: 0xf59e0b,
    candidate: 0x94a3b8,
    candidateEdge: 0x0f172a,
    edge: 0xf59e0b,
  },
  v2: {
    final: 0x22c55e,
    candidate: 0x94a3b8,
    candidateEdge: 0x3f3f46,
    edge: 0x22c55e,
  },
};
// Clean-ceiling replacement sidecar from
// scripts/audit_noisy_slanted_ceiling_replacement.py. Shape:
//   { buildings: {"<uuid>": [{element_id, piece_role, corners, ...}]} }
let ceilingReplacement = null;
let ceilingReplacementPromise = null;
let ontologySummaryByUuid = {};
let ontologySummaryPromiseByUuid = {};
let ontologyPartDetailsByUuid = {};
let ontologyPartPromiseByUuid = {};
let ontologyLoadStateByUuid = {};
let selectedOntologyPartByUuid = {};
let fullModelEnhancementByUuid = {};
let fullModelPayloadByUuid = {};
let fullModelPayloadPromiseByUuid = {};
let fullModelDiffModeEnabled = false;
let buildingIndexByUuid = new Map();
const elementMeshByUid = new Map();
let pendingElementUid = getElementUidFromHash();
let pendingBuildingUuid = getBuildingUuidFromHash();
let buildingInfoBaseHtml = '';

window.__viewerDebug = () => ({
  summaryKeys: Object.keys(ontologySummaryByUuid),
  fullModelKeys: Object.keys(fullModelPayloadByUuid),
  fullModelPromiseKeys: Object.keys(fullModelPayloadPromiseByUuid),
  enhancementKeys: Object.keys(fullModelEnhancementByUuid),
  fullModelVisible: groups.fullModelOntology?.visible,
  fullModelChildCount: groups.fullModelOntology?.children?.length,
  checkboxOn: document.getElementById('show-full-model')?.checked,
  current: DATA[currentBuilding]?.uuid,
  ridgeEave: {
    visible: groups.ridgeEave?.visible,
    childCount: groups.ridgeEave?.children?.length,
    scoresKeys: Object.keys(ridgeEaveScoresByUuid),
    candidateKeys: Object.keys(candidateFacesByUuid),
  },
});
window.__viewerForceRidgeEave = () => {
  const bldg = DATA[currentBuilding];
  if (!bldg?.uuid) return 'no current building';
  const ctl = document.getElementById('show-ridge-eave');
  if (ctl) ctl.checked = true;
  setLayerVisibility('ridgeEave', true, true);
  return Promise.all([
    ensureCandidateFaces(bldg.uuid),
    ensureRidgeEaveScores(bldg.uuid),
  ]).then(() => {
    renderRidgeEaveForBuilding(bldg);
    renderLegend();
    return {
      candidates: !!candidateFacesByUuid[bldg.uuid],
      scores: !!ridgeEaveScoresByUuid[bldg.uuid],
      visible: groups.ridgeEave?.visible,
      childCount: groups.ridgeEave?.children?.length,
    };
  });
};
window.__viewerForceLoad = () => {
  const bldg = DATA[currentBuilding];
  if (!bldg?.uuid) return 'no current building';
  return Promise.all([
    ensureOntologySummary(bldg.uuid),
    ensureOntologyFullModel(bldg.uuid),
  ]).then(([sum, full]) => {
    renderFullModelOntologyForBuilding(bldg);
    updateOntologyStatusInfo();
    return {
      summary: !!sum,
      fullModel: !!full,
      childCount: groups.fullModelOntology?.children?.length,
    };
  });
};

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function getVisiblePickRoots() {
  if (typeof document !== 'undefined' && document.body?.dataset?.mode === 'labeling') {
    return [groups.v3Proposals].filter((g) => g && g.visible);
  }
  return [
    groups.merged,
    groups.computed,
    groups.doors,
    groups.windows,
    groups.floors,
    groups.rawCeilings,
    groups.rawCeilingsRoles,
    groups.rawCeilingsReconstructions,
    groups.rawCeilingPlaneSplits,
    groups.rawCeilingPlaneSplitCandidates,
    groups.gaps,
    groups.crossStory,
    groups.extensions,
    groups.overlaps,
    groups.wallClips,
    groups.extGaps,
    groups.fullModel,
    groups.ceilings,
    groups.ceilingReplacement,
    groups.thermalCeilings,
    groups.roofClusters,
    groups.fullModelHeuristicRoof,
    groups.fullModelOntology,
    groups.ontologySemantics,
    groups.ontologyContinuation,
    groups.ontologyCells,
    groups.v3Model,
    groups.v3Proposals,
    groups.candidateFaces,
    groups.reconstruction,
    groups.ridgeEave,
    groups.gableExtension,
  ].filter(g => g.visible);
}

function pickElementIntersection(intersections) {
  if (!Array.isArray(intersections) || intersections.length === 0) return null;
  const withUid = intersections.filter((it) => it.object?.userData?.elementUid);
  if (withUid.length === 0) return null;
  const rawSplitKinds = new Set(['raw-eave-split', 'raw-eave-split-v1', 'raw-eave-split-v2']);
  const finalRawEaveHit = withUid.find(
    (it) => rawSplitKinds.has(it.object?.userData?.elementLocator?.kind)
      && it.object?.userData?.elementLocator?.rawEaveSplitLayer === 'final'
  );
  if (finalRawEaveHit) return finalRawEaveHit;
  const anyRawEaveHit = withUid.find(
    (it) => rawSplitKinds.has(it.object?.userData?.elementLocator?.kind)
  );
  if (anyRawEaveHit) return anyRawEaveHit;
  const rawCeilingHit = withUid.find(
    (it) => it.object?.userData?.elementLocator?.kind === 'ceiling-raw'
  );
  return rawCeilingHit || withUid[0];
}

function ontologyPartIdFromLocator(locator) {
  if (!locator) return null;
  if (locator.partId) return String(locator.partId);
  if (Array.isArray(locator.partIds) && locator.partIds.length > 0) return String(locator.partIds[0]);
  return null;
}

function getBuildingUuid() {
  return DATA[currentBuilding]?.uuid || "";
}

function makeElementUid(locator) {
  const buildingUuid = locator?.buildingUuid || getBuildingUuid();
  if (!buildingUuid || !locator?.kind || !locator?.id) return null;
  return `${buildingUuid}::${locator.kind}::${locator.id}`;
}

function attachLocator(mesh, locator) {
  if (!mesh || !locator) return;
  const uid = makeElementUid(locator);
  if (!uid) return;
  mesh.userData.elementLocator = locator;
  mesh.userData.elementUid = uid;
  if (!elementMeshByUid.has(uid)) elementMeshByUid.set(uid, mesh);
}

function cornersCenter(corners) {
  if (!Array.isArray(corners) || corners.length === 0) return null;
  let sx = 0;
  let sy = 0;
  let sz = 0;
  for (const c of corners) {
    sx += Number(c?.[0] || 0);
    sy += Number(c?.[1] || 0);
    sz += Number(c?.[2] || 0);
  }
  return [sx / corners.length, sy / corners.length, sz / corners.length];
}

// Least-squares plane fit y = a*x + b*z + c over ≥3 corners. Rejects fits
// whose max residual exceeds 0.10 m (input is not planar enough to clip
// against safely). Returns [a, b, c] or null.
function fitObliquePlaneYXZ(corners) {
  if (!Array.isArray(corners) || corners.length < 3) return null;
  let sx = 0, sz = 0, sy = 0;
  let sxx = 0, szz = 0, sxz = 0, sxy = 0, szy = 0;
  const n = corners.length;
  for (const c of corners) {
    const x = Number(c[0]), y = Number(c[1]), z = Number(c[2]);
    sx += x; sz += z; sy += y;
    sxx += x * x; szz += z * z; sxz += x * z;
    sxy += x * y; szy += z * y;
  }
  const m00 = sxx, m01 = sxz, m02 = sx;
  const m10 = sxz, m11 = szz, m12 = sz;
  const m20 = sx,  m21 = sz,  m22 = n;
  const det = (
    m00 * (m11 * m22 - m12 * m21)
    - m01 * (m10 * m22 - m12 * m20)
    + m02 * (m10 * m21 - m11 * m20)
  );
  if (Math.abs(det) < 1e-10) return null;
  const invDet = 1 / det;
  const a = invDet * (
    sxy * (m11 * m22 - m12 * m21)
    - m01 * (szy * m22 - m12 * sy)
    + m02 * (szy * m21 - m11 * sy)
  );
  const b = invDet * (
    m00 * (szy * m22 - m12 * sy)
    - sxy * (m10 * m22 - m12 * m20)
    + m02 * (m10 * sy  - szy * m20)
  );
  const c = invDet * (
    m00 * (m11 * sy  - szy * m21)
    - m01 * (m10 * sy  - szy * m20)
    + sxy * (m10 * m21 - m11 * m20)
  );
  let maxResid = 0;
  for (const corner of corners) {
    const resid = Math.abs(corner[1] - (a * corner[0] + b * corner[2] + c));
    if (resid > maxResid) maxResid = resid;
  }
  if (maxResid > 0.10) return null;
  return [a, b, c];
}

// Point-in-polygon test on the xz plane with a boundary buffer (ray cast
// plus a distance-to-edge fallback so walls that share vertices with the
// oblique atom's footprint aren't excluded by numeric slack).
function pointInXZPolygon(x, z, poly, boundaryTol = 0.05) {
  if (!Array.isArray(poly) || poly.length < 3) return false;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], zi = poly[i][1];
    const xj = poly[j][0], zj = poly[j][1];
    const denom = (zj - zi) || 1e-12;
    const hit = ((zi > z) !== (zj > z))
      && (x < (xj - xi) * (z - zi) / denom + xi);
    if (hit) inside = !inside;
  }
  if (inside) return true;
  const tol2 = boundaryTol * boundaryTol;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const dx = poly[i][0] - poly[j][0];
    const dz = poly[i][1] - poly[j][1];
    const len2 = dx * dx + dz * dz;
    if (len2 < 1e-12) continue;
    let t = ((x - poly[j][0]) * dx + (z - poly[j][1]) * dz) / len2;
    if (t < 0) t = 0; else if (t > 1) t = 1;
    const px = poly[j][0] + t * dx;
    const pz = poly[j][1] + t * dz;
    const d2 = (x - px) * (x - px) + (z - pz) * (z - pz);
    if (d2 <= tol2) return true;
  }
  return false;
}

// Build a Map<`${story}:${roomIndex}`, Array<{plane,footprint,atomId}>> from
// pyResult.ceiling_partitions.oblique. Oblique atoms whose corners don't
// admit a planar fit are skipped.
function buildObliqueCeilingPlaneIndex(pyResult) {
  const index = new Map();
  const atoms = pyResult?.ceiling_partitions?.oblique || [];
  for (const atom of atoms) {
    const ri = atom?.room_index;
    const st = atom?.story;
    if (ri == null || st == null) continue;
    const corners = atom.poly || atom.corners || [];
    if (!Array.isArray(corners) || corners.length < 3) continue;
    const plane = fitObliquePlaneYXZ(corners);
    if (!plane) continue;
    const footprint = corners.map(c => [Number(c[0]), Number(c[2])]);
    const key = `${st}:${ri}`;
    let bucket = index.get(key);
    if (!bucket) { bucket = []; index.set(key, bucket); }
    bucket.push({ plane, footprint, atomId: atom.id });
  }
  return index;
}

// Lower any corner whose xz falls under an overhead oblique ceiling plane
// so the corner sits on the plane instead of poking through it. Never
// raises a corner, and never drops a corner below the polygon's original
// min-y (which would invert a wall quad if the slope dips below the floor).
// Returns new array (shallow) if any corner changed, else input unchanged.
function clipCornersToObliqueCeilings(corners, planes) {
  if (!planes || planes.length === 0) return corners;
  if (!Array.isArray(corners) || corners.length === 0) return corners;
  let minY = Infinity;
  for (const c of corners) {
    const y = Number(c[1]);
    if (y < minY) minY = y;
  }
  let changed = false;
  const out = new Array(corners.length);
  for (let i = 0; i < corners.length; i++) {
    const c = corners[i];
    const x = Number(c[0]), y = Number(c[1]), z = Number(c[2]);
    let clippedY = y;
    for (const { plane, footprint } of planes) {
      if (!pointInXZPolygon(x, z, footprint)) continue;
      const py = plane[0] * x + plane[1] * z + plane[2];
      if (py < clippedY) clippedY = py;
    }
    if (clippedY < minY) clippedY = minY;
    if (clippedY !== y) {
      changed = true;
      out[i] = [x, clippedY, z];
    } else {
      out[i] = c;
    }
  }
  return changed ? out : corners;
}

function parseElementUid(text) {
  const match = String(text || "").trim().match(
    /^([0-9a-fA-F-]{36})::([a-zA-Z0-9_-]+)::(.+)$/
  );
  if (!match) return null;
  return {
    buildingUuid: match[1],
    kind: match[2],
    id: match[3],
    uid: `${match[1]}::${match[2]}::${match[3]}`,
  };
}

function updateElementHash(uid) {
  if (!uid) return;
  const parsed = parseElementUid(uid);
  if (!parsed) return;
  const encoded = encodeURIComponent(uid);
  history.replaceState(null, "", `#eid=${encoded}`);
}

function getElementUidFromHash() {
  const hash = window.location.hash || "";
  const match = hash.match(/eid=([^&]+)/);
  if (!match) return null;
  try {
    return decodeURIComponent(match[1]);
  } catch (_err) {
    return null;
  }
}

function getBuildingUuidFromHash() {
  const hash = window.location.hash || "";
  const match = hash.match(/(?:^#|&)b=([0-9a-fA-F-]{36})/);
  if (!match) return null;
  return match[1];
}

function getLayerPresetFromHash() {
  // #b=<uuid>&layers=<comma-list>   — explicitly set layer toggles for
  // scripted loads (screenshot harness). Unknown keys are ignored.
  const hash = window.location.hash || "";
  const match = hash.match(/(?:^#|&)layers=([^&]+)/);
  if (!match) return null;
  try {
    return decodeURIComponent(match[1])
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  } catch (_err) {
    return null;
  }
}

function getCameraPresetFromHash() {
  const hash = window.location.hash || "";
  const match = hash.match(/(?:^#|&)cam=([A-Za-z_-]+)/);
  if (!match) return null;
  return match[1];
}

// Multi-select set (shift-click). The "primary" is the last-clicked uid —
// its features drive the panel. Labels apply to every entry on submit.
const multiSelectedUids = new Set();

function resolveMeshByUid(uid) {
  let mesh = elementMeshByUid.get(uid);
  if (!mesh) {
    // Fallback: the queue still carries parent ids, but the renderer emits
    // pieces under `<parent>#...` — `#side-<bits>` for opposing-seam splits
    // and `#<n>` for user-drawn manual splits. Match any `#`-suffixed piece
    // so navigation lands on something visible.
    const piecePrefix = `${uid}#`;
    for (const [candidateUid, candidateMesh] of elementMeshByUid) {
      if (typeof candidateUid === 'string' && candidateUid.startsWith(piecePrefix)) {
        mesh = candidateMesh;
        break;
      }
    }
  }
  return mesh || null;
}

function addSelectionHighlightFor(mesh) {
  if (!mesh || !mesh.geometry) return;
  const highlightMesh = new THREE.Mesh(
    mesh.geometry.clone(),
    new THREE.MeshBasicMaterial({
      color: 0xffd166,
      transparent: true,
      opacity: 0.32,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  highlightMesh.position.copy(mesh.position);
  highlightMesh.quaternion.copy(mesh.quaternion);
  highlightMesh.scale.copy(mesh.scale);
  highlightMesh.renderOrder = 1000;
  groups.selection.add(highlightMesh);

  const edgeLines = new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry),
    new THREE.LineBasicMaterial({ color: 0xffef99, transparent: true, opacity: 0.95 }),
  );
  edgeLines.position.copy(mesh.position);
  edgeLines.quaternion.copy(mesh.quaternion);
  edgeLines.scale.copy(mesh.scale);
  edgeLines.renderOrder = 1001;
  groups.selection.add(edgeLines);
}

function rebuildSelectionHighlights() {
  disposeGroup(groups.selection);
  for (const uid of multiSelectedUids) {
    addSelectionHighlightFor(resolveMeshByUid(uid));
  }
}

function selectElementByUid(uid, { focus = true, updateHash = true, additive = false } = {}) {
  const parsed = parseElementUid(uid);
  if (!parsed) return false;
  const mesh = resolveMeshByUid(parsed.uid);
  if (!mesh) return false;

  if (additive) {
    if (multiSelectedUids.has(parsed.uid)) {
      multiSelectedUids.delete(parsed.uid);
    } else {
      multiSelectedUids.add(parsed.uid);
    }
  } else {
    multiSelectedUids.clear();
    multiSelectedUids.add(parsed.uid);
  }
  rebuildSelectionHighlights();

  if (multiSelectedUids.size === 0) {
    hideProposalPanel();
    setMapStatus('Cleared selection');
    return false;
  }

  // "Primary" = the clicked uid if still selected, otherwise an arbitrary
  // remaining entry. It drives the panel (features, current label, ...).
  const primaryUid = multiSelectedUids.has(parsed.uid)
    ? parsed.uid
    : multiSelectedUids.values().next().value;
  const primaryMesh = resolveMeshByUid(primaryUid);
  const primaryParsed = primaryUid === parsed.uid ? parsed : parseElementUid(primaryUid);
  if (!primaryMesh || !primaryParsed) return false;

  const bldg = DATA[currentBuilding];
  const addr = bldg?.address || bldg?.uuid || "Building";
  const locator = primaryMesh.userData?.elementLocator;
  const roomPart = locator?.roomId ? ` room ${locator.roomId}` : "";
  const storyPart = Number.isFinite(locator?.story) ? ` story ${locator.story}` : "";
  const sourcePart = locator?.source ? ` (${locator.source})` : "";
  const multiSuffix = multiSelectedUids.size > 1 ? ` (+${multiSelectedUids.size - 1} more)` : "";
  setMapStatus(`Selected ${primaryParsed.kind} ${primaryParsed.id}${roomPart}${storyPart}${sourcePart}${multiSuffix}`);

  if (focus) {
    const center = cornersCenter(locator?.corners);
    if (center) {
      controls.target.set(center[0], center[1], center[2]);
      controls.update();
    }
  }

  showLineagePanel(primaryParsed, locator);
  if (primaryParsed.kind === 'v3-roof-proposal' || primaryParsed.kind === 'v3-merged-roof-segment') {
    showProposalPanel(primaryParsed, locator, primaryMesh);
    refreshQueueIndexFromSelection(primaryParsed);
  } else {
    hideProposalPanel();
  }

  if (updateHash) updateElementHash(primaryParsed.uid);
  document.title = `${addr} - ${primaryParsed.kind} ${primaryParsed.id} - 3D Viewer`;
  return true;
}

const PROPOSAL_REASON_CHIPS = [
  'wrong-azimuth',
  'wrong-footprint',
  'should-be-flat',
  'not-a-roof',
  'covers-wrong-room',
];

let proposalPanelState = null;

function formatProposalFeatureValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(3);
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}

function showProposalPanel(parsed, locator, mesh) {
  const panel = document.getElementById('proposal-panel');
  if (!panel) return;
  const kv = document.getElementById('proposal-kv');
  const heuristic = document.getElementById('proposal-heuristic');
  const reasonsEl = document.getElementById('proposal-reasons');
  const statusEl = document.getElementById('proposal-status');
  if (!kv || !heuristic || !reasonsEl || !statusEl) return;

  const features = locator?.features || {};
  const heuristicLabel = locator?.heuristicLabel || 'not_evaluated';
  const proposalId = locator?.proposalId || parsed.uid;
  const buildingUuid = parsed.buildingUuid;
  const currentLabel = (roofProposalLabelsByUuid[buildingUuid] || {})[proposalId] || 'unlabeled';

  const h3 = panel.querySelector('h3');
  if (h3) {
    h3.textContent = multiSelectedUids.size > 1
      ? `V3 Roof Proposal — ${multiSelectedUids.size} selected`
      : 'V3 Roof Proposal';
  }

  const keys = Object.keys(features).sort();
  let kvHtml = '';
  for (const k of keys) {
    kvHtml += `<div class="k">${k}</div><div class="v">${formatProposalFeatureValue(features[k])}</div>`;
  }
  kv.innerHTML = kvHtml || '<div class="k">No features</div><div class="v">—</div>';

  // Phase 7: surface the model's score and autonomy verdict alongside the
  // legacy heuristic. Looking up in `proposalQueue` (loaded from the
  // scored mirror JSON) keeps this decoupled from the in-memory V3 payload.
  const rawId = rawProposalIdOf(proposalId);
  const qEntry = proposalQueue.find((p) => p.proposal_id === rawId);
  const autoLabel = qEntry?.autonomy_label;
  const score = typeof qEntry?.score === 'number' ? qEntry.score : null;
  let autoHtml = '';
  if (autoLabel) {
    const color = autoLabel === 'auto_accept' ? '#4a9' : autoLabel === 'auto_reject' ? '#c66' : '#d90';
    const scoreTxt = score !== null ? score.toFixed(3) : '—';
    const ruleTxt = qEntry?.rule_fires ? ' <em>(rule)</em>' : '';
    autoHtml = ` &middot; <span style="color:${color}">model: <strong>${autoLabel}</strong> (${scoreTxt})${ruleTxt}</span>`;
  }
  heuristic.innerHTML = `heuristic: <strong>${heuristicLabel}</strong> &middot; current: <strong>${currentLabel}</strong>${autoHtml}`;

  const activeReasons = new Set();
  let reasonsHtml = '';
  for (const r of PROPOSAL_REASON_CHIPS) {
    reasonsHtml += `<span class="reason-chip" data-reason="${r}">${r}</span>`;
  }
  reasonsEl.innerHTML = reasonsHtml;
  reasonsEl.querySelectorAll('.reason-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const r = chip.dataset.reason;
      if (activeReasons.has(r)) {
        activeReasons.delete(r);
        chip.classList.remove('on');
      } else {
        activeReasons.add(r);
        chip.classList.add('on');
      }
    });
  });

  statusEl.textContent = '';

  proposalPanelState = {
    parsed, locator, mesh, features, heuristicLabel, proposalId, buildingUuid, activeReasons,
  };
  if (splitModeState && splitModeState.proposalId !== proposalId) {
    exitSplitMode();
  }
  panel.classList.add('visible');
}

function hideProposalPanel() {
  const panel = document.getElementById('proposal-panel');
  if (panel) panel.classList.remove('visible');
  proposalPanelState = null;
  if (splitModeState) exitSplitMode();
}

async function submitLabelForMesh(mesh, parsed, label, activeReasons) {
  const locator = mesh.userData?.elementLocator || {};
  const buildingUuid = parsed.buildingUuid;
  const proposalId = locator.proposalId || parsed.uid;
  const features = locator.features || {};
  const heuristicLabel = locator.heuristicLabel || 'not_evaluated';

  // Walk sibling meshes on the same cluster-side so the label captures the
  // multi-polygon of post-split 3D corners contributed by each cluster
  // member. Dedupe by parent proposal id (fill Mesh + edge Line share a
  // locator) so each member contributes exactly one piece polygon.
  const sidePieces = [];
  const seenMemberIds = new Set();
  const parent = mesh.parent;
  if (parent) {
    for (const child of parent.children) {
      const loc = child.userData?.elementLocator;
      if (!loc || loc.proposalId !== proposalId) continue;
      if (!Array.isArray(loc.corners) || loc.corners.length < 3) continue;
      if (child.type !== 'Mesh') continue;
      const memberId = loc.parentProposalId;
      if (memberId && seenMemberIds.has(memberId)) continue;
      if (memberId) seenMemberIds.add(memberId);
      sidePieces.push({
        parent_proposal_id: memberId ?? null,
        piece_corners_xyz: loc.corners.map((c) => [...c]),
      });
    }
  }

  const body = {
    building_uuid: buildingUuid,
    proposal_id: proposalId,
    label,
    reasons: Array.from(activeReasons || []),
    features_snapshot: features,
    heuristic_label: heuristicLabel,
    // Rich snapshot so the JSONL is self-contained for reverse engineering:
    // downstream code can rebuild the merged plane, its members, the seams
    // that clipped this piece (rain intersection with opposing planes +
    // room/gap boundary split + building-boundary clip), and the merge
    // thresholds — without re-running the proposer or viewer.
    merge_mode: !!locator.mergeMode,
    kind: locator.kind || null,
    cluster_canonical_id: locator.clusterCanonicalId || null,
    part_index: locator.partIndex ?? 0,
    part_count: locator.partCount ?? 1,
    merged_plane: locator.mergedPlane || locator.plane || null,
    cluster_members: locator.clusterMembers || [],
    member_proposal_ids: locator.memberProposalIds || [],
    cluster_params: locator.clusterParams || null,
    opposing_cluster_canonicals: locator.opposingClusterCanonicals || [],
    opposing_planes: locator.opposingPlanes || [],
    room_boundary_refs: locator.roomBoundaryRefs || [],
    building_boundary_xz: locator.buildingBoundaryXz || null,
    segment_corners_xyz: Array.isArray(locator.corners)
      ? locator.corners.map((c) => [...c])
      : null,
    side_pieces: sidePieces,
  };

  try {
    const resp = await fetch('v3-roof-proposal-label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`label save failed: ${resp.status}`);
    if (!roofProposalLabelsByUuid[buildingUuid]) {
      roofProposalLabelsByUuid[buildingUuid] = {};
    }
    roofProposalLabelsByUuid[buildingUuid][proposalId] = label;
    const qIdx = findQueueIndexByProposalId(proposalId);
    if (qIdx >= 0) proposalQueue[qIdx].label = label;
    recolorProposalMesh(mesh, label);
    return true;
  } catch (err) {
    console.warn('Failed to save proposal label', err);
    return false;
  }
}

async function submitProposalLabel(label, { autoAdvance = false } = {}) {
  const state = proposalPanelState;
  const statusEl = document.getElementById('proposal-status');

  // Collect targets from the multi-select set. Fall back to the primary
  // when the set is unexpectedly empty but a panel is open.
  const uids = Array.from(multiSelectedUids);
  if (uids.length === 0 && state?.parsed?.uid) uids.push(state.parsed.uid);
  if (uids.length === 0) return;

  if (statusEl) {
    statusEl.textContent = uids.length > 1
      ? `Saving ${label} to ${uids.length} segments…`
      : `Saving ${label}…`;
  }

  const reasons = state?.activeReasons || new Set();
  let saved = 0;
  for (const uid of uids) {
    const parsed = parseElementUid(uid);
    const mesh = resolveMeshByUid(uid);
    if (!parsed || !mesh) continue;
    const ok = await submitLabelForMesh(mesh, parsed, label, reasons);
    if (ok) saved += 1;
  }
  updateQueueBarCounter();

  if (uids.length > 1) {
    if (statusEl) statusEl.textContent = `Saved ${label} to ${saved}/${uids.length}.`;
    multiSelectedUids.clear();
    rebuildSelectionHighlights();
    const h3 = document.getElementById('proposal-panel')?.querySelector('h3');
    if (h3) h3.textContent = 'V3 Roof Proposal';
  } else {
    if (statusEl) statusEl.textContent = saved ? `Saved ${label}.` : `Error saving ${label}.`;
    const heuristic = document.getElementById('proposal-heuristic');
    if (heuristic && state) {
      const rawId = rawProposalIdOf(state.proposalId);
      const qEntry = proposalQueue.find((p) => p.proposal_id === rawId);
      const autoLabel = qEntry?.autonomy_label;
      const score = typeof qEntry?.score === 'number' ? qEntry.score : null;
      let autoHtml = '';
      if (autoLabel) {
        const color = autoLabel === 'auto_accept' ? '#4a9' : autoLabel === 'auto_reject' ? '#c66' : '#d90';
        const scoreTxt = score !== null ? score.toFixed(3) : '—';
        const ruleTxt = qEntry?.rule_fires ? ' <em>(rule)</em>' : '';
        autoHtml = ` &middot; <span style="color:${color}">model: <strong>${autoLabel}</strong> (${scoreTxt})${ruleTxt}</span>`;
      }
      heuristic.innerHTML = `heuristic: <strong>${state.heuristicLabel}</strong> &middot; current: <strong>${label}</strong>${autoHtml}`;
    }
  }

  if (autoAdvance) stepQueue(1, { onlyUnlabeled: true });
}

let splitModeState = null;

function enterSplitMode() {
  const state = proposalPanelState;
  if (!state) return;
  splitModeState = {
    buildingUuid: state.buildingUuid,
    proposalId: state.proposalId,
    points: [],
  };
  const statusEl = document.getElementById('proposal-status');
  if (statusEl) statusEl.textContent = 'Split mode: click 2 points on the polygon (Esc to cancel).';
  const btn = document.getElementById('proposal-split');
  if (btn) btn.classList.add('active');
  const canvas = document.getElementById('c');
  if (canvas) canvas.style.cursor = 'crosshair';
}

function exitSplitMode(message = '') {
  splitModeState = null;
  const btn = document.getElementById('proposal-split');
  if (btn) btn.classList.remove('active');
  const canvas = document.getElementById('c');
  if (canvas) canvas.style.cursor = '';
  if (message) {
    const statusEl = document.getElementById('proposal-status');
    if (statusEl) statusEl.textContent = message;
  }
}

function toggleSplitMode() {
  if (splitModeState) {
    exitSplitMode('Split cancelled.');
  } else {
    enterSplitMode();
  }
}

async function submitProposalSplit(p1, p2) {
  const state = splitModeState;
  if (!state) return;
  const statusEl = document.getElementById('proposal-status');
  if (statusEl) statusEl.textContent = 'Splitting…';
  try {
    const resp = await fetch('v3-roof-proposal-split', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        building_uuid: state.buildingUuid,
        parent_proposal_id: state.proposalId,
        split_line: [p1, p2],
      }),
    });
    if (!resp.ok) {
      const errText = await resp.text().catch(() => '');
      throw new Error(`split failed: ${resp.status} ${errText}`);
    }
    const data = await resp.json();
    const children = Array.isArray(data?.children) ? data.children : [];
    if (children.length < 2) throw new Error('split returned no children');

    const splitRecord = {
      parent_id: state.proposalId,
      split_line: [p1, p2],
      children: children.map((c) => ({ id: c.id, corners: c.corners })),
    };
    if (!Array.isArray(roofProposalSplitsByUuid[state.buildingUuid])) {
      roofProposalSplitsByUuid[state.buildingUuid] = [];
    }
    roofProposalSplitsByUuid[state.buildingUuid].push(splitRecord);

    const parentIdx = findQueueIndexByProposalId(state.proposalId);
    if (parentIdx >= 0) {
      const addr = proposalQueue[parentIdx].address || '';
      const replacement = children.map((c) => ({
        building_uuid: state.buildingUuid,
        address: addr,
        proposal_id: c.id,
        heuristic_label: c.heuristic_label,
        label: 'unlabeled',
      }));
      proposalQueue.splice(parentIdx, 1, ...replacement);
      proposalQueueIndex = parentIdx;
    }

    const bldg = DATA[currentBuilding];
    if (bldg && bldg.uuid === state.buildingUuid) {
      renderV3ForBuilding(bldg);
    }
    exitSplitMode(`Split into ${children.length} pieces.`);
    updateQueueBarCounter();
    await selectElementByUid(children[0].id, { focus: true, updateHash: true });
  } catch (err) {
    console.warn('Split failed', err);
    exitSplitMode(`Split error: ${err.message || err}`);
  }
}

function recolorProposalMesh(mesh, label) {
  if (!mesh) return;
  const color = proposalColor(label);
  if (mesh.material && mesh.material.color) {
    mesh.material.color.setHex(color);
    mesh.material.needsUpdate = true;
  }
  const parent = mesh.parent;
  if (!parent) return;
  // Labels apply per rain-exposed segment (one proposal = one label), so
  // only recolor siblings that share this exact elementUid — i.e. the
  // same piece's fill Mesh and edge Line. Sibling proposals that happen
  // to be coplanar keep their own color.
  const locatorUid = mesh.userData?.elementUid;
  if (!locatorUid) return;
  for (const child of parent.children) {
    if (child === mesh) continue;
    if (child.userData?.elementUid !== locatorUid) continue;
    if (child.material && child.material.color) {
      child.material.color.setHex(color);
      child.material.needsUpdate = true;
    }
  }
}

function showLineagePanel(parsed, locator) {
  const panel = document.getElementById("lineage-panel");
  const content = document.getElementById("lineage-content");
  if (!panel || !content) return;
  const lineage = locator?.lineage;
  if (!lineage || lineage.length === 0) {
    panel.classList.remove("visible");
    return;
  }
  let html = `<div class="lineage-element-info"><strong>${parsed.kind}</strong> ${parsed.id}`;
  if (locator.source) html += ` <span style="color:#666">(${locator.source})</span>`;
  html += `</div>`;
  for (const entry of lineage) {
    html += `<div class="lineage-entry ${entry.action}">`;
    html += `<span class="lineage-step">${entry.step}</span>`;
    html += `<span class="lineage-action">${entry.action}</span>`;
    if (entry.detail) html += `<div class="lineage-detail">${entry.detail}</div>`;
    html += `</div>`;
  }
  content.innerHTML = html;
  panel.classList.add("visible");
}

function hideLineagePanel() {
  const panel = document.getElementById("lineage-panel");
  if (panel) panel.classList.remove("visible");
}

document.querySelectorAll('#proposal-panel .action').forEach((btn) => {
  btn.addEventListener('click', () => {
    const label = btn.dataset.label;
    if (label) submitProposalLabel(label);
  });
});

let proposalQueue = [];
let proposalQueueIndex = -1;

async function ensureProposalQueue() {
  if (proposalQueue.length > 0) return proposalQueue;
  try {
    const resp = await fetch('v3-roof-proposal-queue', { cache: 'no-store' });
    if (!resp.ok) return [];
    const data = await resp.json();
    proposalQueue = Array.isArray(data?.proposals) ? data.proposals : [];
  } catch (err) {
    console.warn('Failed to load proposal queue', err);
    proposalQueue = [];
  }
  return proposalQueue;
}

function updateQueueBarCounter() {
  const counter = document.getElementById('queue-counter');
  if (!counter) return;
  if (proposalQueue.length === 0) {
    counter.innerHTML = 'No proposals';
    return;
  }
  const labeled = proposalQueue.filter((p) => p.label && p.label !== 'unlabeled').length;
  const idx = proposalQueueIndex >= 0 ? proposalQueueIndex + 1 : 0;
  const review = proposalQueue.filter((p) => p.autonomy_label === 'review').length;
  const autoAcc = proposalQueue.filter((p) => p.autonomy_label === 'auto_accept').length;
  const autoRej = proposalQueue.filter((p) => p.autonomy_label === 'auto_reject').length;
  const scored = review + autoAcc + autoRej > 0;
  if (scored) {
    counter.innerHTML = (
      `<strong>${idx}</strong> / ${proposalQueue.length} &middot; ` +
      `${labeled} labeled &middot; ` +
      `<span title="review queue">${review} review</span> &middot; ` +
      `<span title="auto-accept" style="color:#4a9">${autoAcc}✓</span> ` +
      `<span title="auto-reject" style="color:#c66">${autoRej}✗</span>`
    );
  } else {
    counter.innerHTML = `<strong>${idx}</strong> / ${proposalQueue.length} &middot; ${labeled} labeled`;
  }
}

function setQueueBarVisible(visible) {
  const bar = document.getElementById('queue-bar');
  if (!bar) return;
  if (visible) {
    bar.classList.add('visible');
    ensureProposalQueue().then(() => updateQueueBarCounter());
  } else {
    bar.classList.remove('visible');
  }
}

async function navigateToProposal(proposalId, { focus = true, updateHash = true } = {}) {
  const parsed = parseElementUid(proposalId);
  if (!parsed) return false;
  const targetIdx = buildingIndexByUuid.get(parsed.buildingUuid);
  if (!Number.isInteger(targetIdx)) return false;
  if (currentBuilding !== targetIdx) {
    loadBuilding(targetIdx, { resetPipeline: false });
  }
  const proposalsToggle = document.getElementById('show-v3-roof-proposals');
  if (proposalsToggle && !proposalsToggle.checked) {
    proposalsToggle.checked = true;
    proposalsToggle.dispatchEvent(new Event('change'));
  }
  const bldg = DATA[currentBuilding];
  if (!bldg) return false;
  await Promise.all([
    ensureV3Payload(bldg.uuid),
    ensureRoofProposalLabels(bldg.uuid),
    ensureRoofProposalSplits(bldg.uuid),
  ]);
  if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return false;
  renderV3ForBuilding(bldg);
  return selectElementByUid(proposalId, { focus, updateHash });
}

// Strip any `#...` piece suffix (opposing-seam `#side-<bits>` and user-drawn
// manual-split `#<n>`) so piece ids map back to their parent proposal id for
// queue indexing.
function rawProposalIdOf(proposalId) {
  if (typeof proposalId !== 'string') return proposalId;
  const hash = proposalId.indexOf('#');
  return hash < 0 ? proposalId : proposalId.slice(0, hash);
}

function findQueueIndexByProposalId(proposalId) {
  const rawId = rawProposalIdOf(proposalId);
  for (let i = 0; i < proposalQueue.length; i += 1) {
    if (proposalQueue[i].proposal_id === rawId) return i;
  }
  return -1;
}

// A proposal counts as labeled for queue-advance purposes if its raw id
// OR any `<id>#part-...` piece (disjoint rain-exposed sub-region) has a
// non-unlabeled entry. Labels are per rain-exposed segment; no cluster
// aggregation.
function isProposalLabeledInMap(buildingUuid, rawId) {
  const labels = roofProposalLabelsByUuid[buildingUuid] || {};
  const direct = labels[rawId];
  if (direct && direct !== 'unlabeled') return true;
  const prefix = `${rawId}#part-`;
  for (const key of Object.keys(labels)) {
    if (key.startsWith(prefix) && labels[key] && labels[key] !== 'unlabeled') return true;
  }
  return false;
}

async function stepQueue(delta, { onlyUnlabeled = false } = {}) {
  await ensureProposalQueue();
  if (proposalQueue.length === 0) return;
  let start = proposalQueueIndex;
  if (start < 0) start = delta >= 0 ? -1 : proposalQueue.length;
  const n = proposalQueue.length;
  for (let step = 1; step <= n; step += 1) {
    const idx = ((start + delta * step) % n + n) % n;
    const p = proposalQueue[idx];
    if (onlyUnlabeled) {
      if (isProposalLabeledInMap(p.building_uuid, p.proposal_id)) continue;
      if (p.label && p.label !== 'unlabeled') continue;
      // Phase 7: proposals the classifier is confident about are treated
      // as already handled — surfacing them for a human decision is the
      // opposite of what autonomy is supposed to achieve. They remain in
      // the queue so a user can still jump to them by ID for audit.
      if (p.autonomy_label === 'auto_accept' || p.autonomy_label === 'auto_reject') continue;
    }
    proposalQueueIndex = idx;
    updateQueueBarCounter();
    await navigateToProposal(p.proposal_id);
    return;
  }
  const status = document.getElementById('proposal-status');
  if (status) status.textContent = 'No more unlabeled proposals.';
}

function refreshQueueIndexFromSelection(parsed) {
  if (!parsed) return;
  if (parsed.kind !== 'v3-roof-proposal' && parsed.kind !== 'v3-merged-roof-segment') return;
  const idx = findQueueIndexByProposalId(parsed.uid);
  if (idx >= 0) {
    proposalQueueIndex = idx;
    updateQueueBarCounter();
  }
}

async function stepBuilding(delta) {
  await ensureProposalQueue();
  if (!Array.isArray(DATA) || DATA.length === 0) return;
  const n = DATA.length;
  const start = Number.isInteger(currentBuilding) ? currentBuilding : 0;
  for (let step = 1; step <= n; step += 1) {
    const idx = ((start + delta * step) % n + n) % n;
    const bldg = DATA[idx];
    if (!bldg?.uuid) continue;
    const unlabeled = proposalQueue.find((p) => {
      if (p.building_uuid !== bldg.uuid) return false;
      if (isProposalLabeledInMap(p.building_uuid, p.proposal_id)) return false;
      return !p.label || p.label === 'unlabeled';
    });
    const target = unlabeled || proposalQueue.find((p) => p.building_uuid === bldg.uuid);
    if (target) {
      const qIdx = findQueueIndexByProposalId(target.proposal_id);
      if (qIdx >= 0) proposalQueueIndex = qIdx;
      updateQueueBarCounter();
      await navigateToProposal(target.proposal_id);
      return;
    }
    loadBuilding(idx, { resetPipeline: false });
    const proposalsToggle = document.getElementById('show-v3-roof-proposals');
    if (proposalsToggle && !proposalsToggle.checked) {
      proposalsToggle.checked = true;
      proposalsToggle.dispatchEvent(new Event('change'));
    }
    return;
  }
}

document.getElementById('queue-prev')?.addEventListener('click', () => stepQueue(-1));
document.getElementById('queue-next')?.addEventListener('click', () => stepQueue(1));
document.getElementById('queue-next-unlabeled')?.addEventListener('click', () => stepQueue(1, { onlyUnlabeled: true }));

const PROPOSAL_KEY_LABELS = { a: 'accepted', r: 'rejected', s: 'skipped' };

document.addEventListener('keydown', (event) => {
  const t = event.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === 'escape' && splitModeState) {
    event.preventDefault();
    exitSplitMode('Split cancelled.');
    return;
  }
  if (proposalPanelState && key === 'x') {
    event.preventDefault();
    toggleSplitMode();
    return;
  }
  if (proposalPanelState && key in PROPOSAL_KEY_LABELS) {
    event.preventDefault();
    submitProposalLabel(PROPOSAL_KEY_LABELS[key], { autoAdvance: true });
    return;
  }
  const bar = document.getElementById('queue-bar');
  const queueActive = bar && bar.classList.contains('visible');
  if (!queueActive) return;
  if (key === 'n') { event.preventDefault(); stepQueue(1, { onlyUnlabeled: true }); }
  else if (key === 'j' || key === 'arrowleft') { event.preventDefault(); stepQueue(-1); }
  else if (key === 'k' || key === 'arrowright') { event.preventDefault(); stepQueue(1); }
  else if (key === 'arrowdown') { event.preventDefault(); stepBuilding(1); }
  else if (key === 'arrowup') { event.preventDefault(); stepBuilding(-1); }
});

document.getElementById('proposal-split')?.addEventListener('click', () => toggleSplitMode());

function applyRoofRatingToPanel(uuid) {
  const panel = document.getElementById('roof-rating-panel');
  if (!panel) return;
  const record = uuid ? roofRatingsByUuid[uuid] : null;
  const current = record && Object.prototype.hasOwnProperty.call(record, 'rating') ? record.rating : null;
  panel.querySelectorAll('button.rating-btn').forEach((btn) => {
    const raw = btn.dataset.rating;
    const value = raw === 'upstream_error' ? 'upstream_error' : Number(raw);
    btn.classList.toggle('active', current !== null && current === value);
  });
  const status = document.getElementById('roof-rating-status');
  if (status) {
    if (current === null) {
      status.textContent = '';
      status.title = '';
    } else if (current === 'upstream_error') {
      status.textContent = 'upstream';
      status.title = `Saved ${record?.updated_at || ''}`;
    } else {
      status.textContent = `${current}/5`;
      status.title = `Saved ${record?.updated_at || ''}`;
    }
  }
}

async function submitRoofRating(rating) {
  const uuid = getBuildingUuid();
  if (!uuid) return;
  const prior = roofRatingsByUuid[uuid] || null;
  const priorRating = prior?.rating ?? null;
  const nextRating = priorRating === rating ? null : rating;
  if (nextRating === null) {
    delete roofRatingsByUuid[uuid];
  } else {
    roofRatingsByUuid[uuid] = { rating: nextRating, updated_at: '...' };
  }
  applyRoofRatingToPanel(uuid);
  try {
    const resp = await fetch('/roof-rating', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uuid, rating: nextRating }),
    });
    if (resp.ok) {
      const body = await resp.json().catch(() => ({}));
      if (body?.record) {
        roofRatingsByUuid[uuid] = body.record;
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
}

document.querySelectorAll('#roof-rating-panel button.rating-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const raw = btn.dataset.rating;
    const value = raw === 'upstream_error' ? 'upstream_error' : Number(raw);
    submitRoofRating(value);
  });
});

function jumpToElementUid(uid, { focus = true, updateHash = true, additive = false } = {}) {
  const parsed = parseElementUid(uid);
  if (!parsed) return false;
  const targetIndex = buildingIndexByUuid.get(parsed.buildingUuid);
  if (!Number.isInteger(targetIndex)) return false;

  if (currentBuilding !== targetIndex) {
    loadBuilding(targetIndex, { resetPipeline: false });
  }
  return selectElementByUid(parsed.uid, { focus, updateHash, additive });
}

async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (_err) {
    // Fall through to textarea fallback.
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch (_err) {
    return false;
  }
}

let groups = {
  merged: new THREE.Group(),
  computed: new THREE.Group(),
  doors: new THREE.Group(),
  windows: new THREE.Group(),
  floors: new THREE.Group(),
  gaps: new THREE.Group(),
  extensions: new THREE.Group(),
  overlaps: new THREE.Group(),
  wallClips: new THREE.Group(),
  extGaps: new THREE.Group(),
  ceilings: new THREE.Group(),
  rawCeilings: new THREE.Group(),
  rawCeilingsRoles: new THREE.Group(),
  rawCeilingsReconstructions: new THREE.Group(),
  computedOverextend: new THREE.Group(),
  rawDisagreement: new THREE.Group(),
  rawCeilingPlaneSplits: new THREE.Group(),
  rawCeilingPlaneSplitCandidates: new THREE.Group(),
  ceilingReplacement: new THREE.Group(),
  thermalCeilings: new THREE.Group(),
  crossStory: new THREE.Group(),
  roofClusters: new THREE.Group(),
  fullModelHeuristicRoof: new THREE.Group(),
  fullModelOntology: new THREE.Group(),
  ontologySemantics: new THREE.Group(),
  ontologyContinuation: new THREE.Group(),
  ontologyCells: new THREE.Group(),
  fullModel: new THREE.Group(),
  v3Model: new THREE.Group(),
  v3Proposals: new THREE.Group(),
  gableExtension: new THREE.Group(),
  candidateFaces: new THREE.Group(),
  reconstruction: new THREE.Group(),
  ridgeEave: new THREE.Group(),
  selection: new THREE.Group(),
};
scene.add(groups.merged);
scene.add(groups.computed);
scene.add(groups.doors);
scene.add(groups.windows);
scene.add(groups.floors);
scene.add(groups.gaps);
scene.add(groups.extensions);
scene.add(groups.overlaps);
scene.add(groups.wallClips);
scene.add(groups.extGaps);
scene.add(groups.ceilings);
scene.add(groups.rawCeilings);
scene.add(groups.rawCeilingsRoles);
scene.add(groups.rawCeilingsReconstructions);
scene.add(groups.computedOverextend);
scene.add(groups.rawDisagreement);
scene.add(groups.rawCeilingPlaneSplits);
scene.add(groups.rawCeilingPlaneSplitCandidates);
scene.add(groups.ceilingReplacement);
scene.add(groups.thermalCeilings);
scene.add(groups.crossStory);
scene.add(groups.roofClusters);
scene.add(groups.fullModelHeuristicRoof);
scene.add(groups.fullModelOntology);
scene.add(groups.ontologySemantics);
scene.add(groups.ontologyContinuation);
scene.add(groups.ontologyCells);
scene.add(groups.fullModel);
scene.add(groups.v3Model);
scene.add(groups.v3Proposals);
scene.add(groups.gableExtension);
scene.add(groups.candidateFaces);
scene.add(groups.reconstruction);
scene.add(groups.ridgeEave);
scene.add(groups.selection);
groups.overlaps.visible = false;
groups.fullModel.visible = false;
groups.wallClips.visible = false;
groups.merged.visible = false;
groups.gaps.visible = false;
groups.crossStory.visible = false;
groups.extGaps.visible = false;
groups.ceilings.visible = false;
groups.thermalCeilings.visible = false;
groups.rawCeilingsRoles.visible = false;
groups.rawCeilingsReconstructions.visible = false;
groups.computedOverextend.visible = false;
groups.rawDisagreement.visible = false;
groups.rawCeilingPlaneSplits.visible = false;
groups.rawCeilingPlaneSplitCandidates.visible = false;
groups.ceilingReplacement.visible = false;
groups.fullModelHeuristicRoof.visible = false;
groups.fullModelOntology.visible = false;
groups.ontologySemantics.visible = false;
groups.ontologyContinuation.visible = false;
groups.ontologyCells.visible = false;
groups.v3Model.visible = false;
groups.v3Proposals.visible = false;
groups.gableExtension.visible = false;
groups.candidateFaces.visible = false;
groups.reconstruction.visible = false;
groups.ridgeEave.visible = false;
groups.selection.visible = true;

let pipelineStepIndex = 0;
let currentStorySet = new Set();

function buildPipelineVisibility(stepIndex) {
  const vis = {};
  for (const key of LAYER_KEYS) vis[key] = false;

  for (let i = 0; i <= stepIndex; i++) {
    const step = PIPELINE_STEPS[i];
    if (!step) continue;
    if (step.exclusive) {
      for (const key of LAYER_KEYS) vis[key] = false;
      for (const key of step.exclusive) vis[key] = true;
      continue;
    }
    for (const key of (step.adds || [])) vis[key] = true;
  }
  return vis;
}

function setLayerVisibility(layer, visible, syncControls = true) {
  if (groups[layer]) groups[layer].visible = visible;
  if (layer === 'fullModel') {
    groups.fullModelHeuristicRoof.visible = visible;
    groups.fullModelOntology.visible = visible;
  }
  if (syncControls) {
    const id = LAYER_CONTROL_IDS[layer];
    const ctl = id ? document.getElementById(id) : null;
    if (ctl) ctl.checked = visible;
  }
  if (layer === 'ontologySemantics' || layer === 'ontologyContinuation' || layer === 'ontologyCells') {
    updateOntologyStatusInfo();
  }
  if (visible && (layer === 'ontologySemantics' || layer === 'ontologyContinuation' || layer === 'ontologyCells' || layer === 'fullModel')) {
    maybeLoadOntologyForCurrentBuilding();
  }
  if (visible && (layer === 'v3Model' || layer === 'v3Proposals')) {
    maybeLoadV3ForCurrentBuilding();
  }
  if (visible && layer === 'candidateFaces') {
    maybeLoadCandidateFacesForCurrentBuilding();
  }
  if (visible && layer === 'reconstruction') {
    maybeLoadReconstructionForCurrentBuilding();
  }
  if (visible && layer === 'ridgeEave') {
    maybeLoadRidgeEaveForCurrentBuilding();
  }
  if (visible && (layer === 'rawCeilingsRoles' || layer === 'rawCeilingsReconstructions')) {
    maybeLoadRawCeilingPrototypeForCurrentBuilding();
  }
  if (visible && layer === 'computedOverextend') {
    maybeLoadComputedOverextendForCurrentBuilding();
  }
  if (visible && layer === 'rawDisagreement') {
    maybeLoadRawDisagreementForCurrentBuilding();
  }
  if (visible && layer === 'rawCeilingPlaneSplits') {
    maybeLoadRawCeilingPlaneSplitsForCurrentBuilding();
  }
  if (visible && layer === 'rawCeilingPlaneSplitCandidates') {
    maybeLoadRawCeilingPlaneSplitsForCurrentBuilding();
  }
  if (visible && layer === 'ceilingReplacement') {
    maybeLoadCeilingReplacementForCurrentBuilding();
  }
  if (layer === 'v3Proposals') {
    setQueueBarVisible(visible);
  }
}

function renderLegend() {
  const legendBox = document.getElementById('legend-box');
  let html = '';
  if (colorBySource) {
    for (const [src, label] of Object.entries(SOURCE_LABELS)) {
      const c = SOURCE_COLORS[src].fill.toString(16).padStart(6, '0');
      html += `<span style="background:#${c}"></span>${label} `;
    }
  } else if (colorByStory) {
    for (const s of [...currentStorySet].sort()) {
      const c = STORY_COLORS[s % STORY_COLORS.length].toString(16).padStart(6, '0');
      html += `<span style="background:#${c}"></span>S${s} `;
    }
  }
  html += `<span style="background:#c73"></span>Door `;
  html += `<span style="background:#3ad"></span>Window `;
  if (document.getElementById('show-full-model').checked) {
    html += `<span style="background:#be8ad8"></span>Opening `;
  }
  if (document.getElementById('show-overlaps').checked) {
    html += `<span style="background:#f22"></span>Overlap `;
  }
  if (document.getElementById('show-wall-clips').checked) {
    html += `<span style="background:#f66;opacity:0.3"></span>Clipped wall `;
  }
  if (document.getElementById('show-full-model').checked) {
    html += `<span style="background:#d2d8e0;opacity:0.45"></span>Heuristic shell reference `;
    html += `<span style="background:#8eb8d6"></span>Ontology roof replacement `;
    html += `<span style="background:#e5ded4"></span>Ontology wall replacement `;
    html += `<span style="background:#c8c2b7"></span>Ontology floor replacement `;
    html += `<span style="background:#d5cbc0"></span>Exact semantic ceiling `;
    html += `<span style="background:#f59e0b"></span>Fallback ceiling diagnostic `;
    html += `<span style="background:#e3bf96"></span>Knee wall `;
    html += `<span style="background:#f97316"></span>Unresolved coverage `;
    if (document.getElementById('show-full-model-diff')?.checked) {
      html += `<span style="background:#22d3ee"></span>Ontology diff roof `;
      html += `<span style="background:#a3e635"></span>Ontology diff exterior wall `;
      html += `<span style="background:#fb7185"></span>Ontology diff ceiling `;
      html += `<span style="background:#ef4444"></span>Ontology diff knee wall `;
    }
  }
  if (document.getElementById('show-thermal-ceilings')?.checked) {
    html += `<span style="background:#e85d04"></span>Thermal flat `;
    html += `<span style="background:#dc2f02"></span>Thermal slant `;
    html += `<span style="background:#f48c06"></span>Thermal cap `;
    html += `<span style="background:#ff006e"></span>Knee wall `;
    html += `<span style="background:#cc8844"></span>Dormer cheek `;
    html += `<span style="background:#ccaa44"></span>Dormer header `;
    html += `<span style="background:#e85d04"></span>Gap ceiling `;
  }
  if (document.getElementById('show-raw-ceiling-plane-splits')?.checked || document.getElementById('show-raw-ceiling-plane-split-candidates')?.checked) {
    const mode = getRawCeilingSplitVersionMode();
    const showV1 = mode === 'v1' || mode === 'both';
    const showV2 = mode === 'v2' || mode === 'both';
    html += '<br>';
    if (document.getElementById('show-raw-ceiling-plane-splits')?.checked) {
      if (showV1) {
        html += `<span style="background:#f59e0b"></span>V1 final splits `;
      }
      if (showV2) {
        html += `<span style="background:#22c55e"></span>V2 final splits `;
      }
    }
    if (document.getElementById('show-raw-ceiling-plane-split-candidates')?.checked) {
      if (showV1) {
        html += `<span style="background:#94a3b8;opacity:0.88;border:1px solid #0f172a"></span>V1 not-final splits `;
      }
      if (showV2) {
        html += `<span style="background:#94a3b8;opacity:0.88;border:1px solid #3f3f46"></span>V2 not-final splits `;
      }
    }
    html += `<span style="background:#94a3b8;opacity:0.7"></span>Residual `;
  }
  if (document.getElementById('show-roof-clusters').checked && roofClusterData.length > 0) {
    html += '<br>';
    const compassDir = (deg) => {
      const dirs = ['N','NE','E','SE','S','SW','W','NW'];
      return dirs[Math.round(deg / 45) % 8];
    };
    for (const cl of roofClusterData) {
      const c = cl.color.toString(16).padStart(6, '0');
      if (cl.flat) {
        html += `<span style="background:#${c}"></span>Flat +${cl.height}m (${cl.count}) `;
      } else {
        html += `<span style="background:#${c}"></span>${cl.avgIncl.toFixed(0)}&deg; ${compassDir(cl.avgAzimuth)} (${cl.count}) `;
      }
    }
  }
  if (document.getElementById('show-ontology-semantics')?.checked) {
    html += '<br>';
    html += `<span style="background:#e5e7eb"></span>Building part `;
    html += `<span style="background:#0f766e"></span>Sloped roof coverage `;
    html += `<span style="background:#f59e0b"></span>Slope subpart `;
    html += `<span style="background:#dc2626"></span>Gable subpart `;
    html += `<span style="background:#7c3aed"></span>L/T branch `;
    html += `<span style="background:#0f766e"></span>Sloped atom `;
    html += `<span style="background:#c2410c"></span>Attic atom `;
    html += `<span style="background:#2563eb"></span>Flat cap `;
    html += `<span style="background:#dc2626"></span>Unresolved `;
    html += `<span style="background:#eab308"></span>Dormer `;
  }
  if (document.getElementById('show-ontology-continuation')?.checked) {
    html += '<br>';
    html += `<span style="background:#22d3ee"></span>Arrangement-face continuation `;
    html += `<span style="background:#38bdf8"></span>Continuation candidate `;
  }
  if (document.getElementById('show-ontology-cells')?.checked) {
    html += '<br>';
    html += `<span style="background:#c2410c"></span>Attic roof `;
    html += `<span style="background:#0f766e"></span>Upper-void roof `;
    html += `<span style="background:#60a5fa"></span>Slab faces `;
    html += `<span style="background:#e11d48"></span>Perimeter walls `;
    html += `<span style="background:#ff006e"></span>Knee walls `;
  }
  if (document.getElementById('show-v3-model')?.checked) {
    html += '<br>';
    html += `<span style="background:#fbbf24"></span>V3 slab `;
    html += `<span style="background:#7da8d6"></span>V3 flat ceiling `;
    html += `<span style="background:#9ec5e3"></span>V3 slanted roof `;
    html += `<span style="background:#a78bfa"></span>V3 wall extension `;
    html += `<span style="background:#34d399"></span>V3 gap (closed) `;
    html += `<span style="background:#f472b6"></span>V3 dormer `;
    html += `<span style="background:#ff00ff"></span>V3 unresolved `;
  }
  if (document.getElementById('show-v3-roof-proposals')?.checked) {
    html += '<br>';
    html += `<span style="background:#94a3b8"></span>Roof proposal (unlabeled) `;
    html += `<span style="background:#22c55e"></span>Accepted `;
    html += `<span style="background:#ef4444"></span>Rejected `;
    html += `<span style="background:#64748b"></span>Skipped `;
  }
  if (document.getElementById('show-candidate-faces')?.checked) {
    html += '<br>';
    html += `<span style="background:#38bdf8"></span>Candidate face (original) `;
    html += `<span style="background:#fb923c"></span>Candidate face (ridge-extended) `;
  }
  if (document.getElementById('show-reconstruction')?.checked) {
    html += '<br>';
    html += `<span style="background:#22c55e"></span>BIP envelope (auto-accept) `;
    html += `<span style="background:#f59e0b"></span>BIP envelope (review) `;
  }
  if (document.getElementById('show-ridge-eave')?.checked) {
    html += '<br>';
    html += `<span style="background:#ef4444"></span>Score 0 `;
    html += `<span style="background:#facc15"></span>Score 0.5 `;
    html += `<span style="background:#22c55e"></span>Score 1 `;
    html += `<span style="background:#ffd700"></span>Ridge `;
    html += `<span style="background:#fb923c"></span>Eaves `;
    html += `<span style="background:#6366f1"></span>Part OBB `;
  }
  legendBox.innerHTML = html;
}

function setGhostStateOnMaterial(material, ghosted, opacityScale, minOpacity = 0.04) {
  if (!material || material.isLineBasicMaterial || material.isLineDashedMaterial) return;
  material.userData = material.userData || {};
  if (material.userData._ghostBaseOpacity === undefined) {
    material.userData._ghostBaseOpacity = Number.isFinite(material.opacity) ? material.opacity : 1.0;
    material.userData._ghostBaseTransparent = !!material.transparent;
    material.userData._ghostBaseDepthWrite = material.depthWrite !== false;
  }
  if (ghosted) {
    const baseOpacity = Number(material.userData._ghostBaseOpacity ?? 1.0);
    material.opacity = Math.max(minOpacity, baseOpacity * opacityScale);
    material.transparent = true;
    material.depthWrite = false;
  } else {
    material.opacity = Number(material.userData._ghostBaseOpacity ?? 1.0);
    material.transparent = !!material.userData._ghostBaseTransparent;
    material.depthWrite = material.userData._ghostBaseDepthWrite !== false;
  }
  material.needsUpdate = true;
}

function setGroupGhostState(group, ghosted, opacityScale, minOpacity = 0.04) {
  if (!group) return;
  group.traverse((obj) => {
    if (!obj?.isMesh || !obj.material) return;
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    materials.forEach((material) => setGhostStateOnMaterial(material, ghosted, opacityScale, minOpacity));
  });
}

function updateFullModelReferencePresentation(bldg) {
  if (!bldg?.uuid) return;
  const showFullModel = !!document.getElementById('show-full-model')?.checked;
  const enhancement = fullModelEnhancementByUuid[bldg.uuid] || null;
  // The heuristic shell (walls, floors, 3D windows, 3D doors) stays visible —
  // it carries gap closures and wall-extension geometry that the simplified
  // ontology base_* surfaces drop. The ontology layer only overlays ceilings,
  // roofs, attic floors, and knee walls. If the ontology delivers roof surfaces
  // we hide the heuristic roof to avoid a double-render.
  groups.fullModel.visible = showFullModel;
  groups.fullModelHeuristicRoof.visible = showFullModel && !enhancement?.hasRoofReplacement;
  groups.fullModelOntology.visible = showFullModel && !!enhancement?.hasEnhancement;
  setGroupGhostState(groups.fullModel, false, 1.0, 0.035);
  setGroupGhostState(groups.fullModelHeuristicRoof, false, 1.0, 0.05);
}

function getOntologyLoadState(uuid) {
  if (!ontologyLoadStateByUuid[uuid]) {
    ontologyLoadStateByUuid[uuid] = {
      streamToken: 0,
      streaming: false,
      requestedPartIds: new Set(),
      loadedPartIds: new Set(),
    };
  }
  return ontologyLoadStateByUuid[uuid];
}

function getLoadedOntologyParts(uuid) {
  return Object.values(ontologyPartDetailsByUuid[uuid] || {});
}

function getAllOntologyPartIds(summary) {
  const parts = Array.isArray(summary?.building_parts) ? summary.building_parts : [];
  return parts
    .map((part) => String(part?.id || ''))
    .filter(Boolean);
}

function getDefaultOntologyPartId(summary) {
  const parts = Array.isArray(summary?.building_parts) ? summary.building_parts : [];
  const fullBuilding = parts.find((part) => part?.synthetic_role === 'full_building');
  if (fullBuilding?.id) return String(fullBuilding.id);
  const real = parts.find((part) => !part?.synthetic);
  return String((real || parts[0] || {}).id || '') || null;
}

function getSelectedOntologyPartId(uuid) {
  return selectedOntologyPartByUuid[uuid] || null;
}

function setSelectedOntologyPart(uuid, partId, { announce = false } = {}) {
  if (!uuid || !partId) return;
  selectedOntologyPartByUuid[uuid] = partId;
  const bldg = DATA[currentBuilding];
  if (bldg?.uuid === uuid) {
    renderOntologySemanticsForBuilding(bldg);
    renderOntologyContinuationForBuilding(bldg);
    renderOntologyExactForBuilding(bldg);
    updateOntologyStatusInfo();
    if (announce) setMapStatus(`Selected ontology part ${partId}`);
    if (document.getElementById('show-ontology-cells')?.checked) {
      ensureOntologyPart(uuid, partId).then(() => {
        if (DATA[currentBuilding]?.uuid !== uuid) return;
        renderOntologyExactForBuilding(bldg);
        renderLegend();
      });
    }
  }
}

function updateOntologyStatusInfo() {
  const bldg = DATA[currentBuilding];
  if (!bldg) return;
  const showSemantics = !!document.getElementById('show-ontology-semantics')?.checked;
  const showContinuation = !!document.getElementById('show-ontology-continuation')?.checked;
  const showCells = !!document.getElementById('show-ontology-cells')?.checked;
  const showFullModel = !!document.getElementById('show-full-model')?.checked;
  if (!showSemantics && !showContinuation && !showCells && !showFullModel) {
    document.getElementById('building-info').innerHTML = buildingInfoBaseHtml;
    return;
  }
  const summary = ontologySummaryByUuid[bldg.uuid];
  const loadState = getOntologyLoadState(bldg.uuid);
  const partCount = Array.isArray(summary?.building_parts) ? summary.building_parts.length : 0;
  const selectedPartId = getSelectedOntologyPartId(bldg.uuid) || getDefaultOntologyPartId(summary);
  const enhancement = fullModelEnhancementByUuid[bldg.uuid] || null;
  const fullModelPayload = fullModelPayloadByUuid[bldg.uuid] || null;
  const summaryLine = summary
    ? `Ontology summary: ${summary.metadata?.oblique_coverage_patch_count || 0} slope patches, ${summary.metadata?.roof_continuation_region_count || 0} continuation regions, ${summary.metadata?.coverage_subpart_count || 0} subparts, ${summary.metadata?.unresolved_region_count || 0} unresolved`
    : 'Ontology summary: not loaded';
  const continuationLine = showContinuation
    ? `Continuation diagnostics: ${summary?.metadata?.roof_continuation_region_count || 0} arrangement-face regions`
    : 'Continuation diagnostics: layer hidden';
  const exactLine = showCells
    ? `Exact part: ${selectedPartId || 'none'}${selectedPartId && loadState.loadedPartIds.has(selectedPartId) ? ' loaded' : ''}`
    : 'Exact part payloads: layer hidden';
  const fullModelLine = showFullModel
    ? (enhancement
      ? `Full model ontology: ${enhancement.hasEnhancement ? 'active' : 'no exact replacement'}${enhancement.hasEnhancement ? ` (${enhancement.roofFaceCount} roof faces, ${enhancement.baseWallCount} base walls, ${enhancement.baseFloorCount} base floors, ${enhancement.fenestrationCount} fenestration surfaces, ${enhancement.exteriorWallCount} scaffold exterior walls, ${enhancement.occupiedSurfaceCount} scaffold occupied-room surfaces, ${enhancement.ceilingSurfaceCount} exact semantic ceilings${enhancement.fallbackCeilingCount ? `, ${enhancement.fallbackCeilingCount} fallback ceilings` : ''}, ${enhancement.kneeWallCount} knee walls${enhancement.unresolvedCount ? `, ${enhancement.unresolvedCount} unresolved` : ''}${fullModelDiffModeEnabled ? ', diff mode' : ', heuristic shell ghosted'})` : ''}`
      : fullModelPayload
        ? 'Full model ontology: loaded, rendering'
        : 'Full model ontology: loading committed payload')
    : 'Full model ontology: layer hidden';
  document.getElementById('building-info').innerHTML =
    `${buildingInfoBaseHtml}<br><span style="color:#67e8f9">${summaryLine}</span><br><span style="color:#22d3ee">${continuationLine}</span><br><span style="color:#93c5fd">${exactLine}</span><br><span style="color:#c4b5fd">${fullModelLine}</span>`;
}

function renderOntologySemanticsForBuilding(bldg) {
  disposeGroup(groups.ontologySemantics);
  const summary = ontologySummaryByUuid[bldg.uuid] || null;
  if (!summary) {
    updateOntologyStatusInfo();
    return;
  }
  renderOntologySemantics({
    ontologySummary: summary,
    selectedPartId: getSelectedOntologyPartId(bldg.uuid) || getDefaultOntologyPartId(summary),
    groups,
    createPolygonMesh,
    createEdgeLoop,
    attachLocator,
    buildingUuid: bldg.uuid,
  });
  updateOntologyStatusInfo();
}

function renderOntologyContinuationForBuilding(bldg) {
  disposeGroup(groups.ontologyContinuation);
  const summary = ontologySummaryByUuid[bldg.uuid] || null;
  if (!summary) {
    updateOntologyStatusInfo();
    return;
  }
  renderOntologyContinuationDiagnostics({
    ontologySummary: summary,
    selectedPartId: getSelectedOntologyPartId(bldg.uuid) || getDefaultOntologyPartId(summary),
    groups,
    createPolygonMesh,
    createEdgeLoop,
    attachLocator,
    buildingUuid: bldg.uuid,
  });
  updateOntologyStatusInfo();
}

function renderOntologyExactForBuilding(bldg) {
  disposeGroup(groups.ontologyCells);
  const selectedPartId = getSelectedOntologyPartId(bldg.uuid) || getDefaultOntologyPartId(ontologySummaryByUuid[bldg.uuid]);
  const parts = getLoadedOntologyParts(bldg.uuid).filter((payload) => payload?.part_id === selectedPartId);
  renderOntologyExact({
    ontologyParts: parts,
    groups,
    createPolygonMesh,
    createEdgeLoop,
    attachLocator,
    buildingUuid: bldg.uuid,
  });
  updateOntologyStatusInfo();
}

function renderFullModelOntologyForBuilding(bldg) {
  disposeGroup(groups.fullModelOntology);
  fullModelEnhancementByUuid[bldg.uuid] = null;
  const payload = fullModelPayloadByUuid[bldg.uuid] || null;
  if (!payload) {
    ensureOntologyFullModel(bldg.uuid).then(() => {
      if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
      renderFullModelOntologyForBuilding(bldg);
      renderLegend();
      updateOntologyStatusInfo();
    });
    updateFullModelReferencePresentation(bldg);
    updateOntologyStatusInfo();
    return null;
  }
  const enhancement = renderOntologyEnhancedFullModel({
    ontologySummary: null,
    ontologyParts: [payload],
    groups,
    createPolygonMesh,
    createEdgeLoop,
    attachLocator,
    buildingUuid: bldg.uuid,
    diffMode: fullModelDiffModeEnabled,
  });
  fullModelEnhancementByUuid[bldg.uuid] = enhancement;
  updateFullModelReferencePresentation(bldg);
  updateOntologyStatusInfo();
  return enhancement;
}

function renderAncillaryBuildingLayers(bldg, pyResult) {
  try {
    roofClusterData = renderRoofFromPythonResult({
      pyResult,
      THREE,
      groups,
      createLine,
      createPolygonMesh,
      createEdgeLoop,
      roofClusterColors: ROOF_CLUSTER_COLORS,
      attachLocator,
      buildingUuid: bldg.uuid,
    });
  } catch (err) {
    roofClusterData = [];
    console.error('Roof render failed', bldg?.uuid, err);
  }
  try {
    renderOntologySemanticsForBuilding(bldg);
  } catch (err) {
    console.error('Ontology semantics render failed', bldg?.uuid, err);
  }
  try {
    renderOntologyContinuationForBuilding(bldg);
  } catch (err) {
    console.error('Ontology continuation render failed', bldg?.uuid, err);
  }
  try {
    renderOntologyExactForBuilding(bldg);
  } catch (err) {
    console.error('Ontology exact render failed', bldg?.uuid, err);
  }
  try {
    maybeLoadOntologyForCurrentBuilding();
  } catch (err) {
    console.error('Ontology load trigger failed', bldg?.uuid, err);
  }
  try {
    maybeLoadCandidateFacesForCurrentBuilding();
  } catch (err) {
    console.error('Candidate-faces load trigger failed', bldg?.uuid, err);
  }
  try {
    maybeLoadReconstructionForCurrentBuilding();
  } catch (err) {
    console.error('Reconstruction load trigger failed', bldg?.uuid, err);
  }
  try {
    maybeLoadRidgeEaveForCurrentBuilding();
  } catch (err) {
    console.error('Ridge/eave load trigger failed', bldg?.uuid, err);
  }
  try {
    maybeLoadRawCeilingPrototypeForCurrentBuilding();
  } catch (err) {
    console.error('Raw-ceiling prototype load trigger failed', bldg?.uuid, err);
  }
  try {
    maybeLoadComputedOverextendForCurrentBuilding();
  } catch (err) {
    console.error('Computed-overextend load trigger failed', bldg?.uuid, err);
  }
  try {
    maybeLoadRawDisagreementForCurrentBuilding();
  } catch (err) {
    console.error('Raw-disagreement load trigger failed', bldg?.uuid, err);
  }
  try {
    maybeLoadRawCeilingPlaneSplitsForCurrentBuilding();
  } catch (err) {
    console.error('Raw-eave-split load trigger failed', bldg?.uuid, err);
  }
  try {
    maybeLoadCeilingReplacementForCurrentBuilding();
  } catch (err) {
    console.error('Ceiling-replacement load trigger failed', bldg?.uuid, err);
  }
}

function ensureOntologySummary(uuid) {
  if (ontologySummaryByUuid[uuid]) return Promise.resolve(ontologySummaryByUuid[uuid]);
  if (ontologySummaryPromiseByUuid[uuid]) return ontologySummaryPromiseByUuid[uuid];
  ontologySummaryPromiseByUuid[uuid] = fetch(
    `ontology-artifacts?uuid=${encodeURIComponent(uuid)}&view=summary`,
    { cache: 'no-store' },
  )
    .then((resp) => {
      if (!resp.ok) throw new Error(`Ontology summary fetch failed: ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      ontologySummaryByUuid[uuid] = data;
      return data;
    })
    .catch((err) => {
      console.warn('Failed to load ontology summary', uuid, err);
      return null;
    })
    .finally(() => {
      delete ontologySummaryPromiseByUuid[uuid];
    });
  return ontologySummaryPromiseByUuid[uuid];
}

function ensureOntologyPart(uuid, partId) {
  ontologyPartDetailsByUuid[uuid] = ontologyPartDetailsByUuid[uuid] || {};
  ontologyPartPromiseByUuid[uuid] = ontologyPartPromiseByUuid[uuid] || {};
  if (ontologyPartDetailsByUuid[uuid][partId]) return Promise.resolve(ontologyPartDetailsByUuid[uuid][partId]);
  if (ontologyPartPromiseByUuid[uuid][partId]) return ontologyPartPromiseByUuid[uuid][partId];
  const loadState = getOntologyLoadState(uuid);
  loadState.requestedPartIds.add(partId);
  ontologyPartPromiseByUuid[uuid][partId] = fetch(
    `ontology-artifacts?uuid=${encodeURIComponent(uuid)}&view=part&part_id=${encodeURIComponent(partId)}`,
    { cache: 'no-store' },
  )
    .then((resp) => {
      if (!resp.ok) throw new Error(`Ontology part fetch failed: ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      ontologyPartDetailsByUuid[uuid][partId] = data;
      loadState.loadedPartIds.add(partId);
      return data;
    })
    .catch((err) => {
      console.warn('Failed to load ontology part', uuid, partId, err);
      return null;
    })
    .finally(() => {
      delete ontologyPartPromiseByUuid[uuid][partId];
      updateOntologyStatusInfo();
    });
  return ontologyPartPromiseByUuid[uuid][partId];
}

let v3PayloadByUuid = {};
let v3PayloadPromiseByUuid = {};
let candidateFacesByUuid = {};
let candidateFacesPromiseByUuid = {};
let reconstructionByUuid = {};
let reconstructionPromiseByUuid = {};
let ridgeEaveScoresByUuid = {};
let ridgeEaveScoresPromiseByUuid = {};
let roofProposalLabelsByUuid = {};
let roofProposalLabelsPromiseByUuid = {};
let roofProposalSplitsByUuid = {};
let roofProposalSplitsPromiseByUuid = {};

function ensureRoofProposalSplits(uuid) {
  if (roofProposalSplitsByUuid[uuid] !== undefined) {
    return Promise.resolve(roofProposalSplitsByUuid[uuid]);
  }
  if (roofProposalSplitsPromiseByUuid[uuid]) return roofProposalSplitsPromiseByUuid[uuid];
  roofProposalSplitsPromiseByUuid[uuid] = fetch(
    `v3-roof-proposal-splits?building_uuid=${encodeURIComponent(uuid)}`,
    { cache: 'no-store' },
  )
    .then((resp) => {
      if (resp.status === 404) return { splits: [] };
      if (!resp.ok) throw new Error(`splits fetch failed: ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      const list = Array.isArray(data?.splits) ? data.splits : [];
      roofProposalSplitsByUuid[uuid] = list;
      return list;
    })
    .catch((err) => {
      console.warn('Failed to load roof-proposal splits', uuid, err);
      roofProposalSplitsByUuid[uuid] = [];
      return [];
    })
    .finally(() => {
      delete roofProposalSplitsPromiseByUuid[uuid];
    });
  return roofProposalSplitsPromiseByUuid[uuid];
}

function expandProposalsWithSplits(v3Data, splits) {
  if (!v3Data) return v3Data;
  const childrenByParent = new Map();
  const cornersById = new Map();
  for (const rec of splits || []) {
    const pid = rec?.parent_id;
    const kids = rec?.children || [];
    if (!pid || !Array.isArray(kids) || kids.length < 2) continue;
    childrenByParent.set(pid, kids.map((c) => c.id));
    for (const c of kids) {
      if (c?.id && Array.isArray(c?.corners)) cornersById.set(c.id, c.corners);
    }
  }
  const leavesOf = (pid) => {
    const kids = childrenByParent.get(pid);
    if (!kids) return [pid];
    return kids.flatMap((k) => leavesOf(k));
  };
  const expandList = (items) => {
    const out = [];
    for (const p of items || []) {
      for (const leafId of leavesOf(p.id)) {
        if (leafId === p.id) {
          out.push(p);
        } else {
          const corners = cornersById.get(leafId);
          if (!corners) continue;
          out.push({ ...p, id: leafId, corners });
        }
      }
    }
    return out;
  };
  return {
    ...v3Data,
    roof_proposals: expandList(v3Data.roof_proposals),
    merged_roof_segments: expandList(v3Data.merged_roof_segments),
  };
}

function ensureRoofProposalLabels(uuid) {
  if (roofProposalLabelsByUuid[uuid] !== undefined) {
    return Promise.resolve(roofProposalLabelsByUuid[uuid]);
  }
  if (roofProposalLabelsPromiseByUuid[uuid]) return roofProposalLabelsPromiseByUuid[uuid];
  roofProposalLabelsPromiseByUuid[uuid] = fetch(
    `v3-roof-proposal-labels?building_uuid=${encodeURIComponent(uuid)}`,
    { cache: 'no-store' },
  )
    .then((resp) => {
      if (resp.status === 404) return {};
      if (!resp.ok) throw new Error(`label fetch failed: ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      const serverMap = (data && data.labels) || {};
      // Merge, with any locally-written entries (from a POST that raced the
      // initial GET) taking precedence. Without this, a late-arriving GET
      // silently wipes the label the user just saved, which then disappears
      // from the re-render triggered by auto-advance.
      const local = roofProposalLabelsByUuid[uuid] || {};
      roofProposalLabelsByUuid[uuid] = { ...serverMap, ...local };
      return roofProposalLabelsByUuid[uuid];
    })
    .catch((err) => {
      console.warn('Failed to load roof-proposal labels', uuid, err);
      if (roofProposalLabelsByUuid[uuid] === undefined) {
        roofProposalLabelsByUuid[uuid] = {};
      }
      return roofProposalLabelsByUuid[uuid];
    })
    .finally(() => {
      delete roofProposalLabelsPromiseByUuid[uuid];
    });
  return roofProposalLabelsPromiseByUuid[uuid];
}

function ensureV3Payload(uuid) {
  if (v3PayloadByUuid[uuid] !== undefined) {
    return Promise.resolve(v3PayloadByUuid[uuid]);
  }
  if (v3PayloadPromiseByUuid[uuid]) return v3PayloadPromiseByUuid[uuid];
  v3PayloadPromiseByUuid[uuid] = fetch(
    `v3?uuid=${encodeURIComponent(uuid)}`,
    { cache: 'no-store' },
  )
    .then((resp) => {
      if (resp.status === 404) return null;
      if (!resp.ok) throw new Error(`v3 payload fetch failed: ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      v3PayloadByUuid[uuid] = data;
      return data;
    })
    .catch((err) => {
      console.warn('Failed to load v3 payload', uuid, err);
      v3PayloadByUuid[uuid] = null;
      return null;
    })
    .finally(() => {
      delete v3PayloadPromiseByUuid[uuid];
    });
  return v3PayloadPromiseByUuid[uuid];
}

function renderV3ForBuilding(bldg) {
  if (!bldg?.uuid) return;
  const v3Prefix = `${bldg.uuid}::v3-`;
  for (const uid of Array.from(elementMeshByUid.keys())) {
    if (typeof uid === 'string' && uid.startsWith(v3Prefix)) {
      elementMeshByUid.delete(uid);
    }
  }
  disposeGroup(groups.v3Model);
  disposeGroup(groups.gableExtension);
  disposeGroup(groups.v3Proposals);
  const payload = v3PayloadByUuid[bldg.uuid];
  if (!payload) return;
  renderV3Model({
    v3Data: payload,
    groups,
    createPolygonMesh,
    createEdgeLoop,
    createLine,
    attachLocator,
    buildingUuid: bldg.uuid,
  });
  const expanded = expandProposalsWithSplits(
    payload,
    roofProposalSplitsByUuid[bldg.uuid] || [],
  );
  renderV3RoofProposals({
    v3Data: expanded,
    groups,
    createPolygonMesh,
    createEdgeLoop,
    attachLocator,
    buildingUuid: bldg.uuid,
    labelsByProposalId: roofProposalLabelsByUuid[bldg.uuid] || {},
    mergeSimilarPlanes: !!document.getElementById('merge-v3-roof-proposal-planes')?.checked,
  });
}

function maybeLoadV3ForCurrentBuilding() {
  const bldg = DATA[currentBuilding];
  if (!bldg?.uuid) return;
  const showModel = !!document.getElementById('show-v3-model')?.checked;
  const showProposals = !!document.getElementById('show-v3-roof-proposals')?.checked;
  if (!showModel && !showProposals) return;
  Promise.all([
    ensureV3Payload(bldg.uuid),
    ensureRoofProposalLabels(bldg.uuid),
    ensureRoofProposalSplits(bldg.uuid),
  ]).then(() => {
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    renderV3ForBuilding(bldg);
  });
}

function ensureCandidateFaces(uuid) {
  if (candidateFacesByUuid[uuid] !== undefined) {
    return Promise.resolve(candidateFacesByUuid[uuid]);
  }
  if (candidateFacesPromiseByUuid[uuid]) return candidateFacesPromiseByUuid[uuid];
  candidateFacesPromiseByUuid[uuid] = fetch(
    `candidate-faces?uuid=${encodeURIComponent(uuid)}`,
    { cache: 'no-store' },
  )
    .then((resp) => {
      if (resp.status === 404) return null;
      if (!resp.ok) throw new Error(`candidate-faces fetch failed: ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      candidateFacesByUuid[uuid] = data;
      return data;
    })
    .catch((err) => {
      console.warn('Failed to load candidate faces', uuid, err);
      candidateFacesByUuid[uuid] = null;
      return null;
    })
    .finally(() => {
      delete candidateFacesPromiseByUuid[uuid];
    });
  return candidateFacesPromiseByUuid[uuid];
}

function renderCandidateFacesForBuilding(bldg) {
  if (!bldg?.uuid) return;
  const prefix = `${bldg.uuid}::candidate-face::`;
  for (const uid of Array.from(elementMeshByUid.keys())) {
    if (typeof uid === 'string' && uid.startsWith(prefix)) {
      elementMeshByUid.delete(uid);
    }
  }
  disposeGroup(groups.candidateFaces);
  const payload = candidateFacesByUuid[bldg.uuid];
  if (!payload) return;
  renderCandidateFaces({
    candidatesData: payload,
    groups,
    createPolygonMesh,
    createEdgeLoop,
    attachLocator,
    buildingUuid: bldg.uuid,
  });
}

function maybeLoadCandidateFacesForCurrentBuilding() {
  const bldg = DATA[currentBuilding];
  if (!bldg?.uuid) return;
  if (!document.getElementById('show-candidate-faces')?.checked) return;
  ensureCandidateFaces(bldg.uuid).then(() => {
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    renderCandidateFacesForBuilding(bldg);
    renderLegend();
  });
}

function ensureReconstruction(uuid) {
  if (reconstructionByUuid[uuid] !== undefined) {
    return Promise.resolve(reconstructionByUuid[uuid]);
  }
  if (reconstructionPromiseByUuid[uuid]) return reconstructionPromiseByUuid[uuid];
  reconstructionPromiseByUuid[uuid] = fetch(
    `reconstruction?uuid=${encodeURIComponent(uuid)}`,
    { cache: 'no-store' },
  )
    .then((resp) => {
      if (resp.status === 404) return null;
      if (!resp.ok) throw new Error(`reconstruction fetch failed: ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      reconstructionByUuid[uuid] = data;
      return data;
    })
    .catch((err) => {
      console.warn('Failed to load reconstruction', uuid, err);
      reconstructionByUuid[uuid] = null;
      return null;
    })
    .finally(() => {
      delete reconstructionPromiseByUuid[uuid];
    });
  return reconstructionPromiseByUuid[uuid];
}

function renderReconstructionForBuilding(bldg) {
  if (!bldg?.uuid) return;
  const prefix = `${bldg.uuid}::reconstruction-face::`;
  for (const uid of Array.from(elementMeshByUid.keys())) {
    if (typeof uid === 'string' && uid.startsWith(prefix)) {
      elementMeshByUid.delete(uid);
    }
  }
  disposeGroup(groups.reconstruction);
  const payload = reconstructionByUuid[bldg.uuid];
  if (!payload) return;
  renderReconstruction({
    reconstructionData: payload,
    groups,
    createPolygonMesh,
    createEdgeLoop,
    attachLocator,
    buildingUuid: bldg.uuid,
  });
}

function maybeLoadReconstructionForCurrentBuilding() {
  const bldg = DATA[currentBuilding];
  if (!bldg?.uuid) return;
  if (!document.getElementById('show-reconstruction')?.checked) return;
  ensureReconstruction(bldg.uuid).then(() => {
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    renderReconstructionForBuilding(bldg);
    renderLegend();
  });
}

function ensureRidgeEaveScores(uuid) {
  if (ridgeEaveScoresByUuid[uuid] !== undefined) {
    return Promise.resolve(ridgeEaveScoresByUuid[uuid]);
  }
  if (ridgeEaveScoresPromiseByUuid[uuid]) return ridgeEaveScoresPromiseByUuid[uuid];
  ridgeEaveScoresPromiseByUuid[uuid] = fetch(
    `ridge-eave-scores?uuid=${encodeURIComponent(uuid)}`,
    { cache: 'no-store' },
  )
    .then((resp) => {
      if (resp.status === 404) return null;
      if (!resp.ok) throw new Error(`ridge-eave-scores fetch failed: ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      ridgeEaveScoresByUuid[uuid] = data;
      return data;
    })
    .catch((err) => {
      console.warn('Failed to load ridge/eave scores', uuid, err);
      ridgeEaveScoresByUuid[uuid] = null;
      return null;
    })
    .finally(() => {
      delete ridgeEaveScoresPromiseByUuid[uuid];
    });
  return ridgeEaveScoresPromiseByUuid[uuid];
}

function renderRidgeEaveForBuilding(bldg) {
  if (!bldg?.uuid) return;
  const prefix = `${bldg.uuid}::ridge-eave-candidate::`;
  for (const uid of Array.from(elementMeshByUid.keys())) {
    if (typeof uid === 'string' && uid.startsWith(prefix)) {
      elementMeshByUid.delete(uid);
    }
  }
  disposeGroup(groups.ridgeEave);
  const scoresPayload = ridgeEaveScoresByUuid[bldg.uuid];
  const candsPayload = candidateFacesByUuid[bldg.uuid];
  if (!scoresPayload || !candsPayload) return;
  renderRidgeEaveScoring({
    scoresData: scoresPayload,
    candidatesData: candsPayload,
    groups,
    createPolygonMesh,
    createEdgeLoop,
    createLine,
    attachLocator,
    buildingUuid: bldg.uuid,
  });
}

function maybeLoadRidgeEaveForCurrentBuilding() {
  const bldg = DATA[currentBuilding];
  if (!bldg?.uuid) return;
  if (!document.getElementById('show-ridge-eave')?.checked) return;
  Promise.all([
    ensureCandidateFaces(bldg.uuid),
    ensureRidgeEaveScores(bldg.uuid),
  ]).then(() => {
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    renderRidgeEaveForBuilding(bldg);
    renderLegend();
  });
}

function ensureOntologyFullModel(uuid) {
  if (fullModelPayloadByUuid[uuid]) return Promise.resolve(fullModelPayloadByUuid[uuid]);
  if (fullModelPayloadPromiseByUuid[uuid]) return fullModelPayloadPromiseByUuid[uuid];
  fullModelPayloadPromiseByUuid[uuid] = fetch(
    `ontology-artifacts?uuid=${encodeURIComponent(uuid)}&view=full-model`,
    { cache: 'no-store' },
  )
    .then((resp) => {
      if (!resp.ok) throw new Error(`Full model payload fetch failed: ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      fullModelPayloadByUuid[uuid] = data;
      return data;
    })
    .catch((err) => {
      console.warn('Failed to load full-model payload', uuid, err);
      return null;
    })
    .finally(() => {
      delete fullModelPayloadPromiseByUuid[uuid];
      updateOntologyStatusInfo();
    });
  return fullModelPayloadPromiseByUuid[uuid];
}

function ensureOntologyAllParts(uuid, summary = null) {
  return Promise.resolve(summary || ensureOntologySummary(uuid)).then((resolvedSummary) => {
    if (!resolvedSummary) return [];
    const partIds = getAllOntologyPartIds(resolvedSummary);
    if (partIds.length === 0) return [];
    return Promise.all(partIds.map((partId) => ensureOntologyPart(uuid, partId)));
  });
}

function maybeLoadOntologyForCurrentBuilding() {
  const bldg = DATA[currentBuilding];
  if (!bldg?.uuid) return;
  const showSemantics = !!document.getElementById('show-ontology-semantics')?.checked;
  const showContinuation = !!document.getElementById('show-ontology-continuation')?.checked;
  const showCells = !!document.getElementById('show-ontology-cells')?.checked;
  const showFullModel = !!document.getElementById('show-full-model')?.checked;
  if (!showSemantics && !showContinuation && !showCells && !showFullModel) {
    updateOntologyStatusInfo();
    return;
  }
  ensureOntologySummary(bldg.uuid).then((summary) => {
    if (!summary) return;
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    if (!getSelectedOntologyPartId(bldg.uuid)) {
      const defaultPartId = getDefaultOntologyPartId(summary);
      if (defaultPartId) selectedOntologyPartByUuid[bldg.uuid] = defaultPartId;
    }
    renderOntologySemanticsForBuilding(bldg);
    renderOntologyContinuationForBuilding(bldg);
    renderLegend();
    if (showFullModel) {
      ensureOntologyFullModel(bldg.uuid).then(() => {
        if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
        renderFullModelOntologyForBuilding(bldg);
        renderLegend();
        updateOntologyStatusInfo();
      });
    }
    if (!showCells) {
      renderOntologyExactForBuilding(bldg);
      return;
    }
    const partId = getSelectedOntologyPartId(bldg.uuid);
    if (!partId) {
      updateOntologyStatusInfo();
      return;
    }
    ensureOntologyPart(bldg.uuid, partId).then(() => {
      if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
      renderOntologyExactForBuilding(bldg);
      renderLegend();
      updateOntologyStatusInfo();
    });
  });
}

function renderPipelineStatus() {
  const current = PIPELINE_STEPS[pipelineStepIndex];
  const currentLabel = current ? current.label : 'Unknown';
  document.getElementById('pipeline-current').textContent =
    `Step ${pipelineStepIndex + 1}/${PIPELINE_STEPS.length}: ${currentLabel}`;

  const prevBtn = document.getElementById('pipeline-prev');
  const nextBtn = document.getElementById('pipeline-next');
  prevBtn.disabled = pipelineStepIndex <= 0;
  nextBtn.disabled = pipelineStepIndex >= PIPELINE_STEPS.length - 1;

  const holder = document.getElementById('pipeline-steps');
  holder.innerHTML = PIPELINE_STEPS.map((step, i) => {
    const cls = i < pipelineStepIndex ? 'pipeline-step-pill done' : (i === pipelineStepIndex ? 'pipeline-step-pill active' : 'pipeline-step-pill');
    return `<span class="${cls}" data-step="${i}">${i + 1}. ${step.label}</span>`;
  }).join('');
}

function applyPipelineStep(stepIndex) {
  pipelineStepIndex = Math.max(0, Math.min(stepIndex, PIPELINE_STEPS.length - 1));
  const vis = buildPipelineVisibility(pipelineStepIndex);
  for (const key of LAYER_KEYS) setLayerVisibility(key, !!vis[key], true);
  renderPipelineStatus();
  renderLegend();
}

function normalizeDeg(d) {
  let x = d % 360;
  if (x > 180) x -= 360;
  if (x < -180) x += 360;
  return x;
}

function setMapAlignLabel(id, value) {
  document.getElementById(id).textContent = String(Number(value));
}

function applyAlignmentStateToControls(state) {
  const rot = Number(state?.rotation_deg ?? 0);
  const east = Number(state?.offset_east_m ?? 0);
  const north = Number(state?.offset_north_m ?? 0);
  document.getElementById('map-rot').value = String(rot);
  document.getElementById('map-east').value = String(east);
  document.getElementById('map-north').value = String(north);
  setMapAlignLabel('map-rot-val', rot);
  setMapAlignLabel('map-east-val', east);
  setMapAlignLabel('map-north-val', north);
  modelMapRotationDeg = rot;
  modelMapOffsetEastM = east;
  modelMapOffsetNorthM = north;
}

function queueSaveAlignmentForCurrentBuilding() {
  const bldg = DATA[currentBuilding];
  if (!bldg?.uuid) return;
  if (alignmentSaveTimer) clearTimeout(alignmentSaveTimer);
  alignmentSaveTimer = setTimeout(async () => {
    const payload = {
      uuid: bldg.uuid,
      rotation_deg: modelMapRotationDeg,
      offset_east_m: modelMapOffsetEastM,
      offset_north_m: modelMapOffsetNorthM,
    };
    alignmentByUuid[bldg.uuid] = {
      rotation_deg: payload.rotation_deg,
      offset_east_m: payload.offset_east_m,
      offset_north_m: payload.offset_north_m,
    };
    try {
      await fetch('/alignment-calibration', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch (_err) {
      // Keep local state; save can be retried on next adjustment.
    }
  }, 220);
}

function computeModelFootprintGeoJSON(bldg) {
  const gps = bldg.gps;
  if (!gps || typeof gps.lng !== 'number' || typeof gps.lat !== 'number') return null;

  const roomPolys = [];
  const allPts = [];
  for (const room of (bldg.rooms || [])) {
    const fp = room.floor_polygon;
    if (!Array.isArray(fp) || fp.length < 3) continue;
    const pts = fp.map(c => [c[0], c[2]]);
    roomPolys.push(pts);
    allPts.push(...pts);
  }
  if (allPts.length < 3 || roomPolys.length === 0) return null;

  let cx = 0, cz = 0;
  for (const p of allPts) { cx += p[0]; cz += p[1]; }
  cx /= allPts.length;
  cz /= allPts.length;

  const theta = modelMapRotationDeg * Math.PI / 180;
  const ct = Math.cos(theta), st = Math.sin(theta);
  const lat0 = gps.lat;
  const lng0 = gps.lng;
  const metersPerDegLat = 111320;
  const metersPerDegLng = 111320 * Math.cos(lat0 * Math.PI / 180);

  function toLngLat(x, z) {
    const dx = x - cx;
    const dz = z - cz;
    const dn = -dz;
    const e = (ct * dx - st * dn) + modelMapOffsetEastM;
    const n = (st * dx + ct * dn) + modelMapOffsetNorthM;
    return [lng0 + (e / metersPerDegLng), lat0 + (n / metersPerDegLat)];
  }

  const features = roomPolys.map(poly => {
    const ring = poly.map(([x, z]) => toLngLat(x, z));
    ring.push(ring[0]);
    return { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [ring] } };
  });
  return { type: 'FeatureCollection', features };
}

const orthoController = createOrthoMapController({
  maplibregl: window.maplibregl,
  emptyMapStyle: EMPTY_MAP_STYLE,
  getCurrentBuilding: () => DATA[currentBuilding],
  getAllData: () => DATA,
  getCurrentBuildingIndex: () => currentBuilding,
  getAnchorModeEnabled: () => anchorModeEnabled,
  setAnchorModeEnabled: (v) => { anchorModeEnabled = v; },
  getModelAlignment: () => ({
    rotationDeg: modelMapRotationDeg,
    eastM: modelMapOffsetEastM,
    northM: modelMapOffsetNorthM,
    enabled: modelMapEnabled,
  }),
  setModelOffsets: (eastM, northM) => { modelMapOffsetEastM = eastM; modelMapOffsetNorthM = northM; },
  applyAlignmentStateToControls,
  queueSaveAlignmentForCurrentBuilding,
  computeModelFootprintGeoJSON,
  onSetStatus: (txt) => {
    const statusEl = document.getElementById('map-panel-status');
    statusEl.textContent = txt || '';
  },
});

function setMapStatus(text) {
  orthoController.setMapStatus(text);
}

function updateModelMapOverlay(bldg) {
  orthoController.updateModelMapOverlay(bldg);
}

function updateOrthoPanelForBuilding(bldg) {
  orthoController.updateOrthoPanelForBuilding(bldg, orthoEnabled);
}

function getOrthoMap() {
  return orthoController.getMap();
}

function loadBuilding(index, { resetPipeline = true } = {}) {
  currentBuilding = index;
  elementMeshByUid.clear();
  [groups.merged, groups.computed, groups.doors, groups.windows, groups.floors, groups.gaps, groups.crossStory, groups.extensions, groups.overlaps, groups.wallClips, groups.extGaps, groups.ceilings, groups.rawCeilings, groups.rawCeilingsRoles, groups.rawCeilingsReconstructions, groups.rawCeilingPlaneSplits, groups.rawCeilingPlaneSplitCandidates, groups.thermalCeilings, groups.roofClusters, groups.fullModelHeuristicRoof, groups.fullModelOntology, groups.ontologySemantics, groups.ontologyContinuation, groups.ontologyCells, groups.fullModel, groups.v3Model, groups.v3Proposals, groups.gableExtension, groups.selection].forEach(disposeGroup);
  roofClusterData = [];

  const bldg = DATA[index];
  fullModelEnhancementByUuid[bldg.uuid] = null;
  groups.fullModelHeuristicRoof.visible = groups.fullModel.visible;
  groups.fullModelOntology.visible = groups.fullModel.visible;
  applyAlignmentStateToControls(alignmentByUuid[bldg.uuid] || {});
  applyRoofRatingToPanel(bldg.uuid);

  const allCorners = bldg.rooms.flatMap(r =>
    (r.walls_computed.length > 0 ? r.walls_computed : r.walls_merged).flatMap(w => w.corners)
  );
  let cx = 0, cy = 0, cz = 0;
  for (const c of allCorners) { cx += c[0]; cy += c[1]; cz += c[2]; }
  if (allCorners.length > 0) { cx /= allCorners.length; cy /= allCorners.length; cz /= allCorners.length; }

  const storySet = new Set(bldg.rooms.map(r => r.story));
  currentStorySet = storySet;
  let hasRoomCeilingFill = false;

  bldg.rooms.forEach((room, ri) => {
    const story = room.story || 0;
    const floorColor = colorByStory ? STORY_COLORS[story % STORY_COLORS.length] : ROOM_COLORS[ri % ROOM_COLORS.length];
    const wallColor = colorByStory ? STORY_WALL_COLORS[story % STORY_WALL_COLORS.length] : 0x44aa88;
    const wallEdge = colorByStory ? STORY_COLORS[story % STORY_COLORS.length] : 0x66ccaa;

    for (const w of room.walls_merged) {
      if (w.corners.length < 3) continue;
      const mesh = createPolygonMesh(w.corners, MERGED_COLOR, 0.4);
      if (mesh) {
        attachLocator(mesh, {
          buildingUuid: bldg.uuid,
          kind: "wall-merged",
          id: String(w.id || ""),
          roomId: `${story}:${ri}`,
          story,
          source: w.source || "merged",
          corners: w.corners,
          lineage: w.lineage || [],
        });
        groups.merged.add(mesh);
      }
      groups.merged.add(createEdgeLoop(w.corners, MERGED_EDGE));
    }

    // Computed walls: color by source provenance or by story/room
    if (colorBySource) {
      for (const w of room.walls_computed) {
        if (w.corners.length < 3) continue;
        const src = SOURCE_COLORS[w.source] || SOURCE_COLORS['merged-room'];
        const mesh = createPolygonMesh(w.corners, src.fill, 0.5);
        if (mesh) {
          attachLocator(mesh, {
            buildingUuid: bldg.uuid,
            kind: "wall-computed",
            id: String(w.id || ""),
            roomId: `${story}:${ri}`,
            story,
            source: w.source || "",
            corners: w.corners,
            lineage: w.lineage || [],
          });
          groups.computed.add(mesh);
        }
        groups.computed.add(createEdgeLoop(w.corners, src.edge));
      }
    } else {
      for (const w of room.walls_computed) {
        if (w.corners.length < 3) continue;
        const mesh = createPolygonMesh(w.corners, wallColor, 0.5);
        if (mesh) {
          attachLocator(mesh, {
            buildingUuid: bldg.uuid,
            kind: "wall-computed",
            id: String(w.id || ""),
            roomId: `${story}:${ri}`,
            story,
            source: w.source || "",
            corners: w.corners,
            lineage: w.lineage || [],
          });
          groups.computed.add(mesh);
        }
        groups.computed.add(createEdgeLoop(w.corners, wallEdge));
      }
    }

    // Render wall extension strips (computed portions closing floor gaps)
    // extension_strip is a list of quads (each quad is 4 corners)
    for (const w of room.walls_computed) {
      if (w.extension_strip && w.extension_strip.length > 0) {
        const strips = Array.isArray(w.extension_strip[0]?.[0]) ? w.extension_strip : [w.extension_strip];
        for (const quad of strips) {
          if (quad.length >= 3) {
            const mesh = createPolygonMesh(quad, 0xffaa44, 0.35);
            if (mesh) {
              attachLocator(mesh, {
                buildingUuid: bldg.uuid,
                kind: "wall-extension",
                id: `${w.id || "wall"}:${story}:${ri}`,
                roomId: `${story}:${ri}`,
                story,
                source: w.source || "",
                corners: quad,
                lineage: w.lineage || [],
              });
              groups.extensions.add(mesh);
            }
            groups.extensions.add(createEdgeLoop(quad, 0xff8800));
          }
        }
      }
    }

    // Doors
    for (const d of (room.doors || [])) {
      if (d.corners.length >= 3) {
        const mesh = createPolygonMesh(d.corners, DOOR_COLOR, 0.7);
        if (mesh) {
          attachLocator(mesh, {
            buildingUuid: bldg.uuid,
            kind: "door",
            id: String(d.id || ""),
            roomId: `${story}:${ri}`,
            story,
            source: d.source || "",
            corners: d.corners,
            lineage: d.lineage || [],
          });
          groups.doors.add(mesh);
        }
        groups.doors.add(createEdgeLoop(d.corners, DOOR_EDGE));
      }
    }

    // Windows
    for (const w of (room.windows || [])) {
      if (w.corners.length >= 3) {
        const mesh = createPolygonMesh(w.corners, WINDOW_COLOR, 0.6);
        if (mesh) {
          attachLocator(mesh, {
            buildingUuid: bldg.uuid,
            kind: "window",
            id: String(w.id || ""),
            roomId: `${story}:${ri}`,
            story,
            source: w.source || "",
            corners: w.corners,
            lineage: w.lineage || [],
          });
          groups.windows.add(mesh);
        }
        groups.windows.add(createEdgeLoop(w.corners, WINDOW_EDGE));
      }
    }

    if (room.floor_polygon && room.floor_polygon.length >= 3) {
      const floorMesh = createPolygonMesh(room.floor_polygon, floorColor, 0.4);
      if (floorMesh) {
        attachLocator(floorMesh, {
          buildingUuid: bldg.uuid,
          kind: "floor",
          id: `${story}:${ri}`,
          roomId: `${story}:${ri}`,
          story,
          source: "room-floor",
          corners: room.floor_polygon,
          lineage: [],
        });
        groups.floors.add(floorMesh);
      }
      groups.floors.add(createEdgeLoop(room.floor_polygon, floorColor));
    }

    const rawCeilingPlanes = room.raw_ceiling_planes || [];
    rawCeilingPlanes.forEach((plane, planeIdx) => {
      const corners = plane && plane.corners;
      if (!corners || corners.length < 3) return;
      const rawCeilingMesh = createPolygonMesh(corners, RAW_CEILING_COLOR, 0.4);
      if (rawCeilingMesh) {
        attachLocator(rawCeilingMesh, {
          buildingUuid: bldg.uuid,
          kind: "ceiling-raw",
          id: `${story}:${ri}:${planeIdx}`,
          roomId: `${story}:${ri}`,
          story,
          source: room.raw_ceiling_source || "scan",
          corners,
          lineage: [],
        });
        groups.rawCeilings.add(rawCeilingMesh);
      }
      groups.rawCeilings.add(createEdgeLoop(corners, RAW_CEILING_EDGE));
    });

    // Floor overlap regions (clipped area visualization)
    if (room.floor_overlap_region && room.floor_overlap_region.length >= 3) {
      const overlapMesh = createPolygonMesh(room.floor_overlap_region, 0xff2222, 0.5);
      if (overlapMesh) {
        attachLocator(overlapMesh, {
          buildingUuid: bldg.uuid,
          kind: "floor-overlap",
          id: `${story}:${ri}`,
          roomId: `${story}:${ri}`,
          story,
          source: "overlap",
          corners: room.floor_overlap_region,
          lineage: [],
        });
        groups.overlaps.add(overlapMesh);
      }
      groups.overlaps.add(createEdgeLoop(room.floor_overlap_region, 0xff4444));
    }

    // Wall clip ghosts (original extent of clipped staircase walls)
    for (const w of room.walls_computed) {
      if (w.wall_clipped && w.corners_original && w.corners_original.length >= 3) {
        const ghost = createPolygonMesh(w.corners_original, 0xff6666, 0.15);
        if (ghost) {
          attachLocator(ghost, {
            buildingUuid: bldg.uuid,
            kind: "wall-clipped-original",
            id: String(w.id || ""),
            roomId: `${story}:${ri}`,
            story,
            source: w.source || "",
            corners: w.corners_original,
            lineage: w.lineage || [],
          });
          groups.wallClips.add(ghost);
        }
        groups.wallClips.add(createEdgeLoop(w.corners_original, 0xff4444));
      }
    }

    // OLD ceiling rendering commented out — replaced by cluster-based construction below
    // if (room.ceiling_mesh_triangles && room.ceiling_mesh_triangles.length > 0) { ... }
    // else if (room.ceiling_polygon && room.ceiling_polygon.length >= 3) { ... }

  });

  // OLD building-level roof envelope — commented out
  // if (bldg.roof_mesh_triangles && bldg.roof_mesh_triangles.length > 0) { ... }

  const pyResult = bldg.roof_surfaces ? bldg : pyRoofByUuid[bldg.uuid];
  renderAncillaryBuildingLayers(bldg, pyResult);

  // Thermal ceiling surfaces over detected gaps.
  // Skip cross_story gap ceilings when the roof pipeline produced thermal
  // ceilings — the thermal system (thermal-knee, thermal-cap, etc.) already
  // handles cross-story coverage and avoids extending into building extensions.
  const THERMAL_GAP_COLOR = 0xe85d04;
  const THERMAL_GAP_OPACITY = 0.5;
  const thermalCeilings = Array.isArray(pyResult?.ceiling?.thermal) ? pyResult.ceiling.thermal : [];
  const hasThermalCeilings = thermalCeilings.length > 0;
  // a) cross_floor_gaps (within_story + cross_story horizontal gap polygons)
  for (let gi = 0; gi < (bldg.cross_floor_gaps || []).length; gi++) {
    const gap = bldg.cross_floor_gaps[gi];
    // When thermal ceilings exist, cross_story gaps are redundant — those
    // areas are already covered by thermal-knee surfaces.
    if (hasThermalCeilings && gap.type === 'cross_story') continue;
    // Use ceiling_corners (raised to wall-top Y) when available; fall back to floor-level corners
    const ceilCorners = gap.ceiling_corners || gap.corners;
    if (!ceilCorners || ceilCorners.length < 3) continue;
    const mesh = createPolygonMesh(ceilCorners, THERMAL_GAP_COLOR, THERMAL_GAP_OPACITY);
    if (mesh) {
      mesh.renderOrder = 55;
      mesh.material.depthTest = true;
      mesh.material.depthWrite = false;
      if (attachLocator) attachLocator(mesh, {
        buildingUuid: bldg.uuid,
        kind: 'thermal-ceiling',
        id: `thermal:gap:${gi}`,
        corners: ceilCorners,
      });
      groups.thermalCeilings.add(mesh);
    }
    groups.thermalCeilings.add(createEdgeLoop(ceilCorners, THERMAL_GAP_COLOR, 0.6));
  }
  // b) gap_closures with type=ceiling
  for (let ci = 0; ci < (bldg.gap_closures || []).length; ci++) {
    const gc = bldg.gap_closures[ci];
    if (gc.type !== 'ceiling' || !gc.corners || gc.corners.length < 3) continue;
    const mesh = createPolygonMesh(gc.corners, THERMAL_GAP_COLOR, THERMAL_GAP_OPACITY);
    if (mesh) {
      mesh.renderOrder = 55;
      mesh.material.depthTest = true;
      mesh.material.depthWrite = false;
      if (attachLocator) attachLocator(mesh, {
        buildingUuid: bldg.uuid,
        kind: 'thermal-ceiling',
        id: `thermal:closure-ceil:${ci}`,
        corners: gc.corners,
      });
      groups.thermalCeilings.add(mesh);
    }
    groups.thermalCeilings.add(createEdgeLoop(gc.corners, THERMAL_GAP_COLOR, 0.6));
  }

  // Cross-floor gap regions
  // within_story = strips between rooms (magenta), cross_story = coverage differences (orange)
  const WITHIN_COLORS = { high: 0xff00ff, medium: 0xcc44ff, low: 0x8866ff };
  const CROSS_COLORS = { high: 0xff8800, medium: 0xffaa44, low: 0xffcc88 };
  const GAP_OPACITIES = { high: 0.55, medium: 0.4, low: 0.25 };
  const gapList = bldg.cross_floor_gaps || [];
  for (const gap of gapList) {
    if (gap.corners && gap.corners.length >= 3) {
      const isCross = gap.type === 'cross_story';
      const palette = isCross ? CROSS_COLORS : WITHIN_COLORS;
      const col = palette[gap.confidence] || 0xff2222;
      const opa = GAP_OPACITIES[gap.confidence] || 0.3;
      const targetGroup = isCross ? groups.crossStory : groups.gaps;
      const gapMesh = createPolygonMesh(gap.corners, col, opa);
      if (gapMesh) {
        attachLocator(gapMesh, {
          buildingUuid: bldg.uuid,
          kind: isCross ? "gap-cross-story" : "gap-within-story",
          id: String(gap.id || `${gap.type || "gap"}:${gap.confidence || "unknown"}:${targetGroup.children.length}`),
          roomId: "",
          story: Number.isFinite(gap.story) ? gap.story : null,
          source: gap.confidence || "",
          corners: gap.corners,
          lineage: gap.lineage || [],
        });
        targetGroup.add(gapMesh);
      }
      targetGroup.add(createEdgeLoop(gap.corners, col));
    }
  }

  // Stitch walls (fill gaps between disconnected wall endpoints)
  const STITCH_COLOR = 0x44aa88;
  const STITCH_EDGE = 0x66ccaa;
  for (const sw of (bldg.stitch_walls || [])) {
    if (sw.corners && sw.corners.length >= 3) {
      const sCol = colorByStory ? STORY_WALL_COLORS[(sw.story || 0) % STORY_WALL_COLORS.length] : STITCH_COLOR;
      const sEdge = colorByStory ? STORY_COLORS[(sw.story || 0) % STORY_COLORS.length] : STITCH_EDGE;
      const mesh = createPolygonMesh(sw.corners, sCol, 0.5);
      if (mesh) {
        if (!sw.id) {
          console.warn("stitch entry missing id; locator will not round-trip", sw);
        }
        attachLocator(mesh, {
          buildingUuid: bldg.uuid,
          kind: "wall-stitch",
          id: String(sw.id || `stitch:unknown:${groups.computed.children.length}`),
          roomId: "",
          story: Number(sw.story || 0),
          source: "stitch",
          corners: sw.corners,
          lineage: sw.lineage || [],
        });
        groups.computed.add(mesh);
      }
      groups.computed.add(createEdgeLoop(sw.corners, sEdge));
    }
  }

  // Gap walls (vertical walls + floor/ceiling quads along cross-floor gap edges)
  const GAP_WALL_COLORS = {
    wall:    { fill: 0xcc44ff, edge: 0xaa22dd },
    floor:   { fill: 0x88bb44, edge: 0x669933 },
    ceiling: { fill: 0x4488dd, edge: 0x3366bb },
  };
  for (const gw of (bldg.gap_walls || [])) {
    if (gw.corners && gw.corners.length >= 3) {
      const palette = gw.type === 'gap_floor' ? GAP_WALL_COLORS.floor
                    : gw.type === 'gap_ceiling' ? GAP_WALL_COLORS.ceiling
                    : GAP_WALL_COLORS.wall;
      const mesh = createPolygonMesh(gw.corners, palette.fill, 0.35);
      if (mesh) {
        attachLocator(mesh, {
          buildingUuid: bldg.uuid,
          kind: "gap-wall",
          id: String(gw.id || `${gw.type || "wall"}:${groups.gaps.children.length}`),
          roomId: "",
          story: Number.isFinite(gw.story) ? gw.story : null,
          source: gw.type || "",
          corners: gw.corners,
          lineage: gw.lineage || [],
        });
        groups.gaps.add(mesh);
      }
      groups.gaps.add(createEdgeLoop(gw.corners, palette.edge));
    }
  }

  // Exterior gap indicators (door/opening + parallel wall pairs)
  const extGapList = bldg.exterior_gap_indicators || [];
  const EXT_GAP_COLOR = 0xff44ff;
  const EXT_GAP_EDGE = 0xff66ff;
  for (const ind of extGapList) {
    if (ind.element_corners && ind.element_corners.length >= 3) {
      const mesh = createPolygonMesh(ind.element_corners, EXT_GAP_COLOR, 0.7);
      if (mesh) {
        attachLocator(mesh, {
          buildingUuid: bldg.uuid,
          kind: "exterior-gap-element",
          id: String(ind.id || `element:${groups.extGaps.children.length}`),
          roomId: "",
          story: Number.isFinite(ind.story) ? ind.story : null,
          source: "exterior-gap",
          corners: ind.element_corners,
          lineage: ind.lineage || [],
        });
        groups.extGaps.add(mesh);
      }
      groups.extGaps.add(createEdgeLoop(ind.element_corners, EXT_GAP_EDGE));
    }
    if (ind.wall_corners && ind.wall_corners.length >= 3) {
      const mesh = createPolygonMesh(ind.wall_corners, EXT_GAP_COLOR, 0.4);
      if (mesh) {
        attachLocator(mesh, {
          buildingUuid: bldg.uuid,
          kind: "exterior-gap-wall",
          id: String(ind.id || `wall:${groups.extGaps.children.length}`),
          roomId: "",
          story: Number.isFinite(ind.story) ? ind.story : null,
          source: "exterior-gap",
          corners: ind.wall_corners,
          lineage: ind.lineage || [],
        });
        groups.extGaps.add(mesh);
      }
      groups.extGaps.add(createEdgeLoop(ind.wall_corners, EXT_GAP_EDGE));
    }
  }

  // Gap closures (side/floor/ceiling quads filling detected gaps)
  const GAP_CLOSURE_COLORS = {
    side:    { fill: 0x44ddaa, edge: 0x33bb88 },
    floor:   { fill: 0x88bb44, edge: 0x669933 },
    ceiling: { fill: 0x4488dd, edge: 0x3366bb },
  };
  const GAP_CLOSURE_DEFAULT = { fill: 0x44ddaa, edge: 0x33bb88 };
  for (const gc of (bldg.gap_closures || [])) {
    if (gc.corners && gc.corners.length >= 3) {
      const palette = GAP_CLOSURE_COLORS[gc.type] || GAP_CLOSURE_DEFAULT;
      const mesh = createPolygonMesh(gc.corners, palette.fill, 0.5);
      if (mesh) {
        attachLocator(mesh, {
          buildingUuid: bldg.uuid,
          kind: "gap-closure",
          id: String(gc.id || `${gc.type || "closure"}:${groups.extGaps.children.length}`),
          roomId: "",
          story: Number.isFinite(gc.story) ? gc.story : null,
          source: gc.type || "",
          corners: gc.corners,
          lineage: gc.lineage || [],
        });
        groups.extGaps.add(mesh);
      }
      groups.extGaps.add(createEdgeLoop(gc.corners, palette.edge));
    }
  }

  // Full model: neutral structure + accented openings for readability
  const SHOW_FULL_MODEL_EDGES = false;
  const FULL_EDGE_OPACITY = 0.18;
  const fullPalette = {
    structure: { fill: 0xf3f1ee, edge: 0xa1a1aa, opacity: 1.0, roughness: 0.9, metalness: 0.0 },
    floor:     { fill: 0xc8c2b7, edge: 0x9f978a, opacity: 1.0, roughness: 0.55, metalness: 0.0 },
    ceiling:   { fill: 0xd8d2c7, edge: 0xafa89b, opacity: 1.0, roughness: 0.6, metalness: 0.0 },
    // Inspired by pascalorg/editor material presets: wood + glass
    door:      { fill: 0xc49a6c, edge: 0x8a5f3a, opacity: 0.96, roughness: 0.72, metalness: 0.0 },
    window:    { fill: 0x87ceeb, edge: 0xe6f6ff, opacity: 0.28, roughness: 0.08, metalness: 0.12 },
    // Opening shown as "void-like" cut marker (dark fill, bright edge)
    opening:   { fill: 0x0a0e1b, edge: 0xfafafa, opacity: 0.92, roughness: 0.6, metalness: 0.0 },
    // Roof surfaces — light blue, semi-transparent
    roof:      { fill: 0x88aacc, edge: 0x6688aa, opacity: 0.3, roughness: 0.5, metalness: 0.0 },
    // Dormer cheeks/headers — tinted to stand out from plain walls
    dormer:    { fill: 0xcc8844, edge: 0x996633, opacity: 0.85, roughness: 0.7, metalness: 0.0 },
  };
  const fullMaterials = {
    structure: new THREE.MeshStandardMaterial({
      color: fullPalette.structure.fill, opacity: fullPalette.structure.opacity, transparent: false,
      side: THREE.DoubleSide, depthWrite: true, roughness: fullPalette.structure.roughness, metalness: fullPalette.structure.metalness,
      flatShading: false,
    }),
    floor: new THREE.MeshStandardMaterial({
      color: fullPalette.floor.fill, opacity: fullPalette.floor.opacity, transparent: false,
      side: THREE.DoubleSide, depthWrite: true, roughness: fullPalette.floor.roughness, metalness: fullPalette.floor.metalness,
      emissive: 0x16120c, emissiveIntensity: 0.03,
    }),
    ceiling: new THREE.MeshStandardMaterial({
      color: fullPalette.ceiling.fill, opacity: fullPalette.ceiling.opacity, transparent: false,
      side: THREE.DoubleSide, depthWrite: true, roughness: fullPalette.ceiling.roughness, metalness: fullPalette.ceiling.metalness,
    }),
    door: new THREE.MeshStandardMaterial({
      color: fullPalette.door.fill, opacity: fullPalette.door.opacity, transparent: true,
      side: THREE.DoubleSide, depthWrite: false, roughness: fullPalette.door.roughness, metalness: fullPalette.door.metalness,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    }),
    window: new THREE.MeshStandardMaterial({
      color: fullPalette.window.fill, opacity: fullPalette.window.opacity, transparent: true,
      side: THREE.DoubleSide, depthWrite: false, roughness: fullPalette.window.roughness, metalness: fullPalette.window.metalness,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    }),
    opening: new THREE.MeshStandardMaterial({
      color: fullPalette.opening.fill, opacity: fullPalette.opening.opacity, transparent: true,
      side: THREE.DoubleSide, depthWrite: false, roughness: fullPalette.opening.roughness, metalness: fullPalette.opening.metalness,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    }),
    roof: new THREE.MeshStandardMaterial({
      color: fullPalette.roof.fill, opacity: fullPalette.roof.opacity, transparent: true,
      side: THREE.DoubleSide, depthWrite: false, roughness: fullPalette.roof.roughness, metalness: fullPalette.roof.metalness,
    }),
    dormer: new THREE.MeshStandardMaterial({
      color: fullPalette.dormer.fill, opacity: fullPalette.dormer.opacity, transparent: true,
      side: THREE.DoubleSide, depthWrite: false, roughness: fullPalette.dormer.roughness, metalness: fullPalette.dormer.metalness,
    }),
    frame: new THREE.MeshStandardMaterial({
      color: '#f2f0ed', opacity: 1, transparent: false,
      side: THREE.DoubleSide, depthWrite: true, roughness: 0.5, metalness: 0,
    }),
    frameDark: new THREE.MeshStandardMaterial({
      color: '#8a5f3a', opacity: 1, transparent: false,
      side: THREE.DoubleSide, depthWrite: true, roughness: 0.65, metalness: 0,
    }),
    doorLeaf: new THREE.MeshStandardMaterial({
      color: '#c8a27a', opacity: 1, transparent: false,
      side: THREE.DoubleSide, depthWrite: true, roughness: 0.7, metalness: 0,
    }),
    doorPanel: new THREE.MeshStandardMaterial({
      color: '#b88e65', opacity: 1, transparent: false,
      side: THREE.DoubleSide, depthWrite: true, roughness: 0.72, metalness: 0,
    }),
    handleMetal: new THREE.MeshStandardMaterial({
      color: '#c0c0c0', opacity: 1, transparent: false,
      side: THREE.DoubleSide, depthWrite: true, roughness: 0.3, metalness: 0.9,
    }),
  };
  function addOrientedBox(group, basis, cx, cy, cz, sx, sy, sz, material, renderOrder = 3, castShadow = true, receiveShadow = true, locator = null) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), material);
    const rot = new THREE.Matrix4().makeBasis(basis.u, basis.v, basis.n);
    mesh.quaternion.setFromRotationMatrix(rot);
    mesh.position.copy(basis.origin)
      .addScaledVector(basis.u, cx)
      .addScaledVector(basis.v, cy)
      .addScaledVector(basis.n, cz);
    mesh.renderOrder = renderOrder;
    mesh.castShadow = castShadow && material !== fullMaterials.window;
    mesh.receiveShadow = receiveShadow;
    if (locator) attachLocator(mesh, locator);
    group.add(mesh);
  }

  function openingBounds(corners, basis) {
    const pts = corners.map(p => projectToPlane2(new THREE.Vector3(...p), basis));
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
  }

  function addWindowModel(corners, locator) {
    const basis = polygonPlaneBasis(corners);
    if (!basis) return;
    const b = openingBounds(corners, basis);
    if (b.width < 0.05 || b.height < 0.05) return;
    const frameT = Math.max(0.03, Math.min(0.08, Math.min(b.width, b.height) * 0.1));
    const depth = 0.07;
    const glassDepth = 0.012;
    const cx = (b.minX + b.maxX) * 0.5;
    const cy = (b.minY + b.maxY) * 0.5;

    // Outer frame — first piece gets the locator so right-click picks up the window
    addOrientedBox(groups.fullModel, basis, cx, b.maxY - frameT * 0.5, 0, b.width, frameT, depth, fullMaterials.frame, 4, true, false, locator);
    addOrientedBox(groups.fullModel, basis, cx, b.minY + frameT * 0.5, 0, b.width, frameT, depth, fullMaterials.frame, 4, true, false);
    addOrientedBox(groups.fullModel, basis, b.minX + frameT * 0.5, cy, 0, frameT, b.height - 2 * frameT, depth, fullMaterials.frame, 4, true, false);
    addOrientedBox(groups.fullModel, basis, b.maxX - frameT * 0.5, cy, 0, frameT, b.height - 2 * frameT, depth, fullMaterials.frame, 4, true, false);

    // Vertical mullion
    const innerW = Math.max(0.02, b.width - 2 * frameT);
    const innerH = Math.max(0.02, b.height - 2 * frameT);
    const mullionW = Math.max(0.015, frameT * 0.7);
    addOrientedBox(groups.fullModel, basis, cx, cy, 0, mullionW, innerH, depth * 0.96, fullMaterials.frame, 5, true, false);

    // Glass panes — also attach locator so clicking glass works
    const paneW = Math.max(0.01, (innerW - mullionW) * 0.5);
    addOrientedBox(groups.fullModel, basis, cx - (mullionW + paneW) * 0.5, cy, 0, paneW, innerH, glassDepth, fullMaterials.window, 6, false, false, locator);
    addOrientedBox(groups.fullModel, basis, cx + (mullionW + paneW) * 0.5, cy, 0, paneW, innerH, glassDepth, fullMaterials.window, 6, false, false, locator);
  }

  function addDoorModel(corners, locator) {
    const basis = polygonPlaneBasis(corners);
    if (!basis) return;
    const b = openingBounds(corners, basis);
    if (b.width < 0.05 || b.height < 0.05) return;
    const frameT = Math.max(0.035, Math.min(0.09, Math.min(b.width, b.height) * 0.11));
    const depth = 0.08;
    const leafDepth = 0.04;
    const cx = (b.minX + b.maxX) * 0.5;

    // Frame: sides + head (Pascal-like door frame)
    addOrientedBox(groups.fullModel, basis, b.minX + frameT * 0.5, (b.minY + b.maxY) * 0.5, 0, frameT, b.height, depth, fullMaterials.frame, 4, true, false, locator);
    addOrientedBox(groups.fullModel, basis, b.maxX - frameT * 0.5, (b.minY + b.maxY) * 0.5, 0, frameT, b.height, depth, fullMaterials.frame, 4, true, false);
    addOrientedBox(groups.fullModel, basis, cx, b.maxY - frameT * 0.5, 0, b.width, frameT, depth, fullMaterials.frame, 4, true, false);

    // Leaf fills opening below top frame — attach locator since it's the biggest clickable surface
    const leafW = Math.max(0.02, b.width - 2 * frameT);
    const leafH = Math.max(0.02, b.height - frameT);
    const leafCy = b.minY + leafH * 0.5;
    addOrientedBox(groups.fullModel, basis, cx, leafCy, 0, leafW, leafH, leafDepth, fullMaterials.doorLeaf, 5, true, false, locator);

    // Inset panel
    const insetX = Math.max(0.02, leafW * 0.12);
    const insetY = Math.max(0.02, leafH * 0.12);
    addOrientedBox(groups.fullModel, basis, cx, leafCy, leafDepth * 0.38, leafW - 2 * insetX, leafH - 2 * insetY, 0.01, fullMaterials.doorPanel, 6, false, false);

    // Handle (right side)
    const hx = b.maxX - frameT - leafW * 0.08;
    const hy = b.minY + leafH * 0.55;
    addOrientedBox(groups.fullModel, basis, hx, hy, leafDepth * 0.62, 0.02, 0.10, 0.03, fullMaterials.handleMetal, 7, false, false);
  }

  function addOpeningModel(corners) {
    const basis = polygonPlaneBasis(corners);
    if (!basis) return;
    const b = openingBounds(corners, basis);
    if (b.width < 0.03 || b.height < 0.03) return;
    // Thin slab slightly in front of the wall plane to avoid coplanar jitter.
    addOrientedBox(
      groups.fullModel,
      basis,
      (b.minX + b.maxX) * 0.5,
      (b.minY + b.maxY) * 0.5,
      0.03,
      b.width,
      b.height,
      0.02,
      fullMaterials.opening,
      6
    );
  }

  const fullMeshRenderProps = {
    structure: { renderOrder: 1, castShadow: true, receiveShadow: true },
    floor:     { renderOrder: 0, castShadow: true, receiveShadow: true },
    ceiling:   { renderOrder: 0, castShadow: false, receiveShadow: true },
  };

  function addFullMesh(corners, kind = 'structure', holes = [], locator = null, targetGroup = groups.fullModel) {
    if (!corners || corners.length < 3) return;
    if (kind === 'door') { addDoorModel(corners, locator); return; }
    if (kind === 'window') { addWindowModel(corners, locator); return; }
    if (kind === 'opening') { return; }
    if (kind === 'structure') {
      corners = orientedStructureCorners(corners, [cx, cy, cz]);
    }
    const palette = fullPalette[kind] || fullPalette.structure;
    const mat = fullMaterials[kind] || fullMaterials.structure;
    const mesh = createPolygonMesh(corners, palette.fill, palette.opacity, holes);
    if (mesh) {
      mesh.material = mat;
      const props = fullMeshRenderProps[kind] || { renderOrder: 3, castShadow: false, receiveShadow: false };
      mesh.renderOrder = props.renderOrder;
      mesh.castShadow = props.castShadow;
      mesh.receiveShadow = props.receiveShadow;
      if (locator) attachLocator(mesh, locator);
      targetGroup.add(mesh);
    }
    // Keep structural/floor edges off for a clean mass.
    const showKindEdge = SHOW_FULL_MODEL_EDGES || (kind !== 'structure' && kind !== 'floor' && kind !== 'ceiling');
    if (showKindEdge) {
      const edgeOpacity = kind === 'structure' ? FULL_EDGE_OPACITY : 0.55;
      targetGroup.add(createEdgeLoop(corners, palette.edge, edgeOpacity));
    }

  }
  // Index overhead oblique ceiling planes per (story, roomIndex). Used to
  // clip V1 walls + stitches down to the roof pipeline's sloped ceiling so
  // walls don't poke above a slanted roof in the full-model view.
  const obliqueCeilingIndex = buildObliqueCeilingPlaneIndex(pyResult);

  // a) Computed walls + extension strips
  bldg.rooms.forEach((room, ri) => {
    const story = room.story || 0;
    const roomId = `${story}:${ri}`;
    // Physical rule for final model:
    // only windows create light-through wall holes.
    const openingsForCutout = [
      ...(room.windows || []).map(w => ({ corners: w.corners })),
    ];
    const roomObliquePlanes = obliqueCeilingIndex.get(`${story}:${ri}`) || [];
    for (const w of room.walls_computed) {
      const clippedCorners = clipCornersToObliqueCeilings(w.corners, roomObliquePlanes);
      const loc = {
        buildingUuid: bldg.uuid, kind: 'wall-computed',
        id: `${w.id || 'wall'}:${story}:${ri}`, roomId, story, corners: clippedCorners,
      };
      const holes = collectWallCutoutHoles(clippedCorners, openingsForCutout);
      addFullMesh(clippedCorners, 'structure', holes, loc);
      // Extension strips as continuation (same color = no seam)
      if (w.extension_strip && w.extension_strip.length > 0) {
        const strips = Array.isArray(w.extension_strip[0]?.[0]) ? w.extension_strip : [w.extension_strip];
        for (const quad of strips) {
          const clippedQuad = clipCornersToObliqueCeilings(quad, roomObliquePlanes);
          addFullMesh(clippedQuad, 'structure', [], {
            buildingUuid: bldg.uuid, kind: 'wall-extension',
            id: `${w.id || 'wall'}:${story}:${ri}`, roomId, story, corners: clippedQuad,
          });
        }
      }
    }
  });
  // b) Gap walls (including gap floor/ceiling quads)
  for (let gi = 0; gi < (bldg.gap_walls || []).length; gi++) {
    const gw = bldg.gap_walls[gi];
    const kind = gw.type === 'gap_floor' ? 'floor' : gw.type === 'gap_ceiling' ? 'ceiling' : 'structure';
    addFullMesh(gw.corners, kind, [], {
      buildingUuid: bldg.uuid, kind: 'gap-wall', id: String(gw.id || `gw:${gi}`), corners: gw.corners,
    });
  }
  // b2) Floors (included in final full model)
  bldg.rooms.forEach((room, ri) => {
    const story = room.story || 0;
    if (room.floor_polygon && room.floor_polygon.length >= 3) {
      addFullMesh(room.floor_polygon, 'floor', [], {
        buildingUuid: bldg.uuid, kind: 'floor', id: `${story}:${ri}`, corners: room.floor_polygon,
      });
    }
  });
  // c) Gap closures (side/floor/ceiling quads filling exterior gaps)
  for (let ci = 0; ci < (bldg.gap_closures || []).length; ci++) {
    const gc = bldg.gap_closures[ci];
    const kind = gc.type === 'floor' ? 'floor' : gc.type === 'ceiling' ? 'ceiling' : 'structure';
    addFullMesh(gc.corners, kind, [], {
      buildingUuid: bldg.uuid, kind: 'gap-closure', id: String(gc.id || `gc:${ci}`), corners: gc.corners,
    });
  }
  // d) Stitch walls
  for (let si = 0; si < (bldg.stitch_walls || []).length; si++) {
    const sw = bldg.stitch_walls[si];
    // Collect the overhead oblique planes for every room this stitch touches;
    // stitches span two rooms, so a slope in either one needs to clip it.
    const stitchStory = sw.story ?? 0;
    const stitchRooms = Array.isArray(sw.room_indices) && sw.room_indices.length
      ? sw.room_indices
      : (sw.room_index != null ? [sw.room_index] : []);
    const stitchPlanes = [];
    for (const ri of stitchRooms) {
      const planes = obliqueCeilingIndex.get(`${stitchStory}:${ri}`);
      if (planes && planes.length) stitchPlanes.push(...planes);
    }
    const stitchCorners = clipCornersToObliqueCeilings(sw.corners, stitchPlanes);
    addFullMesh(stitchCorners, 'structure', [], {
      buildingUuid: bldg.uuid, kind: 'wall-stitch', id: String(sw.id || `sw:${si}`), corners: stitchCorners,
    });
  }
  // d2) Dormer cheeks and headers — added individually (not merged) so locators work.
  // Dormer + oblique-roof geometry is produced by the python roof pipeline and
  // delivered via pyRoofByUuid; buildings_3d.json does not carry these fields.
  const fullModelRoofSource = (bldg?.dormers || bldg?.roof_surfaces) ? bldg : (pyResult || {});
  const dormerList = Array.isArray(fullModelRoofSource.dormers) ? fullModelRoofSource.dormers : [];
  for (let di = 0; di < dormerList.length; di++) {
    const d = dormerList[di];
    for (let ci = 0; ci < (d.cheeks || []).length; ci++) {
      const cheek = d.cheeks[ci];
      if (!cheek?.corners || cheek.corners.length < 3 || cheek.source === 'existing') continue;
      const mesh = createPolygonMesh(cheek.corners, fullPalette.dormer.fill, fullPalette.dormer.opacity);
      if (mesh) {
        mesh.material = fullMaterials.dormer.clone();
        mesh.renderOrder = 3;
        mesh.castShadow = false;
        attachLocator(mesh, {
          buildingUuid: bldg.uuid,
          kind: 'dormer-cheek',
          id: `dormer-cheek:${di}:${ci}`,
          corners: cheek.corners,
        });
        groups.fullModel.add(mesh);
      }
    }
    if (d.header && d.header.corners && d.header.corners.length >= 3 && d.header.source !== 'existing') {
      const mesh = createPolygonMesh(d.header.corners, fullPalette.dormer.fill, fullPalette.dormer.opacity);
      if (mesh) {
        mesh.material = fullMaterials.dormer.clone();
        mesh.renderOrder = 3;
        attachLocator(mesh, {
          buildingUuid: bldg.uuid,
          kind: 'dormer-header',
          id: `dormer-header:${di}`,
          corners: d.header.corners,
        });
        groups.fullModel.add(mesh);
      }
    }
  }
  // d3) Oblique roof surfaces (with dormer cutout holes)
  const obliqueRoofSurfaces = Array.isArray(fullModelRoofSource?.roof_surfaces?.oblique)
    ? fullModelRoofSource.roof_surfaces.oblique
    : [];
  for (let ri = 0; ri < obliqueRoofSurfaces.length; ri++) {
    const s = obliqueRoofSurfaces[ri];
    if (!s.corners || s.corners.length < 3) continue;
    const holes = Array.isArray(s.cutout_holes) ? s.cutout_holes : [];
    addFullMesh(s.corners, 'roof', holes, {
      buildingUuid: bldg.uuid, kind: 'roof-oblique', id: `oblique:${ri}`, corners: s.corners,
    }, groups.fullModelHeuristicRoof);
  }
  // (Structure/floor/ceiling meshes are added individually in addFullMesh for right-click locators.)
  // e) Openings + windows/doors so final stage includes all cutout geometry context
  bldg.rooms.forEach((room, ri) => {
    const story = room.story || 0;
    const roomId = `${story}:${ri}`;
    for (const d of (room.doors || [])) {
      addFullMesh(d.corners, 'door', [], {
        buildingUuid: bldg.uuid, kind: 'door', id: String(d.id || ''),
        roomId, story, corners: d.corners,
      });
    }
    for (const w of (room.windows || [])) {
      addFullMesh(w.corners, 'window', [], {
        buildingUuid: bldg.uuid, kind: 'window', id: String(w.id || ''),
        roomId, story, corners: w.corners,
      });
    }
  });
  // Exterior indicators are diagnostic; skip in final model to avoid overlap/jitter with doors.

  // Legend
  renderLegend();

  // Info
  const computedCount = bldg.rooms.reduce((s, r) => s + r.walls_computed.length, 0);
  const mergedCount = bldg.rooms.reduce((s, r) => s + r.walls_merged.length, 0);
  const doorCount = bldg.rooms.reduce((s, r) => s + (r.doors || []).length, 0);
  const windowCount = bldg.rooms.reduce((s, r) => s + (r.windows || []).length, 0);
  const addr = bldg.address || bldg.uuid;
  const gapHigh = gapList.filter(g => g.confidence === 'high').length;
  const gapMed = gapList.filter(g => g.confidence === 'medium').length;
  const gapInfo = gapList.length > 0 ? ` &middot; gaps: ${gapHigh}h/${gapMed}m/${gapList.length - gapHigh - gapMed}l` : '';
  const extCount = bldg.rooms.reduce((s, r) => s + r.walls_computed.filter(w => w.extension_strip).length, 0);
  const extInfo = extCount > 0 ? ` &middot; ${extCount} extended` : '';
  const om = bldg.overlap_metrics || {};
  const overlapInfo = om.floor_overlap_count > 0 ? ` &middot; <span style="color:#f44">${om.floor_overlap_count} overlaps (${om.total_floor_overlap_area_m2}m&sup2;)</span>` : '';
  const wallClipInfo = om.walls_clipped > 0 ? ` &middot; <span style="color:#f66">${om.walls_clipped} wall clips</span>` : '';
  const extGapInfo = extGapList.length > 0 ? ` &middot; <span style="color:#f4f">${extGapList.length} ext.gaps</span>` : '';
  const openingsInfo = (doorCount + windowCount) > 0 ? ` &middot; <span style="color:#c73">${doorCount}d</span> <span style="color:#3ad">${windowCount}w</span>` : '';
  // Source breakdown for computed walls
  const srcCounts = {};
  bldg.rooms.forEach(r => r.walls_computed.forEach(w => { srcCounts[w.source] = (srcCounts[w.source] || 0) + 1; }));
  const srcInfo = Object.entries(srcCounts).map(([k, v]) => `${v} ${k}`).join(', ');
  const uuidChip = `<span class="uuid-chip" title="Click to copy UUID" data-copy-uuid="${bldg.uuid}" style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#1a1a2e;color:#cde;padding:1px 6px;border-radius:3px;border:1px solid #333;cursor:pointer;user-select:all;">${bldg.uuid}</span>`;
  buildingInfoBaseHtml =
    `${uuidChip} &middot; ${addr} &middot; ${bldg.rooms.length} rooms &middot; ` +
    `${bldg.stories_found} stories &middot; ${computedCount} computed / ${mergedCount} merged walls${openingsInfo}${gapInfo}${extInfo}${overlapInfo}${wallClipInfo}${extGapInfo}` +
    `<br><span style="color:#555">Sources: ${srcInfo}</span>` +
    `<br><span style="color:#888">Right-click any element to copy its shareable ID</span>`;
  const infoEl = document.getElementById('building-info');
  infoEl.innerHTML = buildingInfoBaseHtml;
  const chip = infoEl.querySelector('.uuid-chip');
  if (chip && !chip.dataset.copyBound) {
    chip.dataset.copyBound = '1';
    chip.addEventListener('click', async () => {
      const ok = await copyText(chip.dataset.copyUuid || '');
      setMapStatus(ok ? `Copied UUID: ${chip.dataset.copyUuid}` : `UUID: ${chip.dataset.copyUuid}`);
    });
  }
  updateOntologyStatusInfo();

  // Camera — honor the `#cam=<preset>` URL param when present so the
  // screenshot harness can capture deterministic views. Defaults to the
  // original iso view.
  let maxDist = 5;
  for (const p of allCorners) {
    const d = Math.hypot(p[0]-cx, p[1]-cy, p[2]-cz);
    if (d > maxDist) maxDist = d;
  }
  const cd = maxDist * 1.8;
  applyCameraPreset(getCameraPresetFromHash() || 'iso', cx, cy, cz, cd);

  // Highlight sidebar (only scroll for keyboard nav, not clicks)
  document.querySelectorAll('.bldg-item').forEach(el => {
    el.classList.toggle('active', Number(el.dataset.index) === index);
  });
  document.title = `${addr} - 3D Viewer`;
  updateOrthoPanelForBuilding(bldg);

  if (resetPipeline) {
    applyPipelineStep(0);
  } else {
    applyPipelineStep(pipelineStepIndex);
  }

  // `#layers=<comma-list>` overrides all layer toggles AFTER the pipeline
  // step (which normally sets them) — lets scripted loads capture an
  // arbitrary layer combination regardless of which pipeline step a user
  // previously parked at.
  const layerPreset = getLayerPresetFromHash();
  if (Array.isArray(layerPreset)) applyLayerPreset(layerPreset);

  // Stamp a render-complete marker so the screenshot harness can wait for
  // rendering to settle before snapping. (`wait_for` in the
  // chrome-devtools MCP polls DOM.)
  stampRenderComplete();
}

function applyCameraPreset(preset, cx, cy, cz, cd) {
  switch (preset) {
    case 'overhead':
      // Top-down view, roughly orthographic-feeling — useful for
      // footprint comparison against the scan footprint.
      camera.position.set(cx, cy + cd * 1.6, cz + 0.01);
      break;
    case 'south':
      camera.position.set(cx, cy + cd * 0.4, cz + cd * 1.2);
      break;
    case 'east':
      camera.position.set(cx + cd * 1.2, cy + cd * 0.4, cz);
      break;
    case 'iso':
    default:
      camera.position.set(cx + cd * 0.7, cy + cd * 0.9, cz + cd * 0.7);
  }
  controls.target.set(cx, cy, cz);
  controls.update();
}

function applyLayerPreset(layers) {
  const wanted = new Set(layers);
  for (const key of LAYER_KEYS) {
    const visible = wanted.has(key);
    setLayerVisibility(key, visible);
  }
}

function stampRenderComplete() {
  let marker = document.getElementById('render-complete');
  if (!marker) {
    marker = document.createElement('div');
    marker.id = 'render-complete';
    marker.style.display = 'none';
    document.body.appendChild(marker);
  }
  // Two rAF frames guarantees the last frame committed to the canvas has
  // flushed — then we flip the marker's data attribute the harness polls.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      marker.dataset.stamp = String(Date.now());
    });
  });
}

bindUIEventHandlers({
  documentRef: document,
  LAYER_CONTROL_IDS,
  LAYER_KEYS,
  groups,
  getData: () => DATA,
  getCurrentBuilding: () => currentBuilding,
  getPipelineStepIndex: () => pipelineStepIndex,
  applyPipelineStep,
  setLayerVisibility,
  onLayerVisibilityChanged: (layer, visible) => {
    if (layer === 'fullModel' && visible) {
      // Full model view is the unified final render. Hide every other layer
      // so debug / per-story / per-cluster / diagnostic overlays don't tint
      // the same surfaces with conflicting colors.
      for (const other of LAYER_KEYS) {
        if (other === 'fullModel') continue;
        if (groups[other]?.visible) setLayerVisibility(other, false);
      }
      const diffCtl = document.getElementById('show-full-model-diff');
      if (diffCtl && diffCtl.checked) {
        diffCtl.checked = false;
        diffCtl.dispatchEvent(new Event('change'));
      }
    }
    if (layer === 'ontologySemantics' || layer === 'ontologyContinuation' || layer === 'ontologyCells' || layer === 'fullModel') {
      const bldg = DATA[currentBuilding];
      if (bldg && layer === 'ontologyCells' && !visible) {
        const loadState = getOntologyLoadState(bldg.uuid);
        loadState.streamToken += 1;
        loadState.streaming = false;
      }
      if (bldg) {
        if (layer === 'ontologySemantics') renderOntologySemanticsForBuilding(bldg);
        if (layer === 'ontologyContinuation') renderOntologyContinuationForBuilding(bldg);
        if (layer === 'ontologyCells') renderOntologyExactForBuilding(bldg);
        if (layer === 'fullModel') renderFullModelOntologyForBuilding(bldg);
      }
      if (visible) maybeLoadOntologyForCurrentBuilding();
      else updateOntologyStatusInfo();
    }
  },
  renderLegend,
  loadBuilding,
  setColorByStory: (v) => { colorByStory = v; },
  setColorBySource: (v) => { colorBySource = v; },
  setOrthoEnabled: (v) => { orthoEnabled = v; },
  updateOrthoPanelForBuilding,
  getOrthoMap,
  setModelMapEnabled: (v) => { modelMapEnabled = v; },
  setModelMapRotationDeg: (v) => { modelMapRotationDeg = v; },
  setModelMapOffsetEastM: (v) => { modelMapOffsetEastM = v; },
  setModelMapOffsetNorthM: (v) => { modelMapOffsetNorthM = v; },
  getModelMapRotationDeg: () => modelMapRotationDeg,
  getModelMapOffsetEastM: () => modelMapOffsetEastM,
  getModelMapOffsetNorthM: () => modelMapOffsetNorthM,
  applyAlignmentStateToControls,
  updateModelMapOverlay,
  queueSaveAlignmentForCurrentBuilding,
  normalizeDeg,
  setMapStatus,
  setAnchorModeEnabled: (v) => { anchorModeEnabled = v; },
  getAnchorModeEnabled: () => anchorModeEnabled,
});

document.getElementById('merge-v3-roof-proposal-planes')?.addEventListener('change', () => {
  const bldg = DATA[currentBuilding];
  if (!bldg) return;
  renderV3ForBuilding(bldg);
  renderLegend();
});

document.getElementById('show-full-model-diff')?.addEventListener('change', (event) => {
  fullModelDiffModeEnabled = !!event.target?.checked;
  const bldg = DATA[currentBuilding];
  if (bldg && document.getElementById('show-full-model')?.checked) {
    renderFullModelOntologyForBuilding(bldg);
  }
  renderLegend();
  updateOntologyStatusInfo();
});

document.getElementById('raw-split-version-mode')?.addEventListener('change', () => {
  const bldg = DATA[currentBuilding];
  if (!bldg?.uuid) return;
  maybeLoadRawCeilingPlaneSplitsForCurrentBuilding();
  renderLegend();
});

canvas.addEventListener("contextmenu", async (event) => {
  event.preventDefault();
  if (!DATA[currentBuilding]) return;

  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  const intersections = raycaster.intersectObjects(getVisiblePickRoots(), true);
  const hit = pickElementIntersection(intersections);
  if (!hit) {
    setMapStatus("Right-click a rendered element to copy a shareable ID");
    hideLineagePanel();
    return;
  }

  const uid = hit.object.userData.elementUid;
  const locator = hit.object.userData?.elementLocator || null;
  const ok = await copyText(uid);

  let suffix = '';
  if (locator && (locator.kind === 'ridge-eave-candidate' || locator.kind === 'candidate-face')) {
    const az = locator.azimuthDeg != null ? `az=${Number(locator.azimuthDeg).toFixed(0)}°` : '';
    const inc = locator.inclinationDeg != null ? `inc=${Number(locator.inclinationDeg).toFixed(0)}°` : '';
    const score = locator.bestScore != null ? `score=${Number(locator.bestScore).toFixed(2)}` : '';
    const pg = locator.planeGroupId ? `pg=${String(locator.planeGroupId).slice(-12)}` : '';
    suffix = ` · ${[az, inc, score, pg].filter(Boolean).join(' ')}`;
  }

  if (!ok) {
    setMapStatus(`Element ID: ${uid}${suffix}`);
    return;
  }

  const selected = jumpToElementUid(uid, { focus: false, updateHash: true });
  setMapStatus(`Copied: ${uid}${suffix}`);
});

canvas.addEventListener("click", (event) => {
  if (!DATA[currentBuilding]) return;
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  if (splitModeState) {
    const hits = raycaster.intersectObject(groups.v3Proposals, true);
    const hit = hits.find((it) => it.object?.userData?.elementUid === splitModeState.proposalId);
    if (!hit) {
      const statusEl = document.getElementById('proposal-status');
      if (statusEl) statusEl.textContent = 'Click inside the selected proposal polygon.';
      return;
    }
    splitModeState.points.push([hit.point.x, hit.point.z]);
    if (splitModeState.points.length === 1) {
      const statusEl = document.getElementById('proposal-status');
      if (statusEl) statusEl.textContent = 'One point captured. Click second point.';
      return;
    }
    if (splitModeState.points.length >= 2) {
      const [p1, p2] = splitModeState.points;
      submitProposalSplit(p1, p2);
    }
    return;
  }

  const intersections = raycaster.intersectObjects(getVisiblePickRoots(), true);
  const hit = pickElementIntersection(intersections);
  if (!hit) return;
  const uid = hit.object.userData.elementUid;
  const locator = hit.object.userData?.elementLocator || null;
  // Shift-click multi-select only for label-eligible kinds; other kinds
  // (walls, floors, …) keep single-select behavior.
  const parsedClicked = parseElementUid(uid);
  const additive = !!event.shiftKey
    && parsedClicked
    && (parsedClicked.kind === 'v3-roof-proposal' || parsedClicked.kind === 'v3-merged-roof-segment');
  jumpToElementUid(uid, { focus: false, updateHash: !additive, additive });
  const partId = ontologyPartIdFromLocator(locator);
  if (partId) {
    const uuid = locator?.buildingUuid || DATA[currentBuilding]?.uuid;
    if (uuid) setSelectedOntologyPart(uuid, partId, { announce: true });
  }
});

document.getElementById("search").addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  const uid = String(event.target?.value || "").trim();
  if (!uid.includes("::")) return;
  if (jumpToElementUid(uid)) {
    event.preventDefault();
    setMapStatus("Jumped to shared element ID");
  } else {
    setMapStatus("Could not resolve that element ID");
  }
});

window.addEventListener('resize', resizeRenderer);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

renderPipelineStatus();
renderLegend();

async function fetchJsonWithLastModified(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  const lastModifiedHeader = response.headers.get('last-modified');
  const lastModifiedMs = lastModifiedHeader ? Date.parse(lastModifiedHeader) : NaN;
  return {
    data: await response.json(),
    lastModifiedMs: Number.isFinite(lastModifiedMs) ? lastModifiedMs : null,
  };
}

function ensureRawCeilingPrototype() {
  if (rawCeilingPrototype !== null) return Promise.resolve(rawCeilingPrototype);
  if (rawCeilingPrototypePromise) return rawCeilingPrototypePromise;
  rawCeilingPrototypePromise = fetch('/raw-ceiling-prototype')
    .then(r => (r.ok ? r.json() : null))
    .then(data => {
      rawCeilingPrototype = data && typeof data === 'object'
        ? data
        : { planes: {}, rooms: {}, reconstructions: {} };
      return rawCeilingPrototype;
    })
    .catch(() => {
      rawCeilingPrototype = { planes: {}, rooms: {}, reconstructions: {} };
      return rawCeilingPrototype;
    });
  return rawCeilingPrototypePromise;
}

function renderRawCeilingPrototypeForBuilding(bldg) {
  if (!bldg?.uuid || !rawCeilingPrototype) return;

  const rolesPrefix = `${bldg.uuid}::ceiling-raw-role::`;
  const reconPrefixDormer = `${bldg.uuid}::ceiling-reconstruction-dormer::`;
  const reconPrefixWing = `${bldg.uuid}::ceiling-reconstruction-wing::`;
  for (const uid of Array.from(elementMeshByUid.keys())) {
    if (typeof uid === 'string' && (
      uid.startsWith(rolesPrefix) ||
      uid.startsWith(reconPrefixDormer) ||
      uid.startsWith(reconPrefixWing)
    )) {
      elementMeshByUid.delete(uid);
    }
  }
  disposeGroup(groups.rawCeilingsRoles);
  disposeGroup(groups.rawCeilingsReconstructions);

  const planesMap = rawCeilingPrototype.planes || {};
  bldg.rooms.forEach((room, ri) => {
    const story = room.story || 0;
    const planes = room.raw_ceiling_planes || [];
    planes.forEach((plane, planeIdx) => {
      const corners = plane && plane.corners;
      if (!corners || corners.length < 3) return;
      const elementId = `${bldg.uuid}::ceiling-raw::${story}:${ri}:${planeIdx}`;
      const entry = planesMap[elementId];
      const role = entry?.role || 'unclassified';
      const color = RAW_CEILING_ROLE_COLORS[role] ?? RAW_CEILING_ROLE_COLORS.unclassified;
      const mesh = createPolygonMesh(corners, color, 0.55);
      if (mesh) {
        mesh.renderOrder = 56;
        attachLocator(mesh, {
          buildingUuid: bldg.uuid,
          kind: 'ceiling-raw',
          id: `${story}:${ri}:${planeIdx}`,
          roomId: `${story}:${ri}`,
          story,
          source: entry?.archetype || room.raw_ceiling_source || 'scan',
          corners,
          lineage: [],
        });
        groups.rawCeilingsRoles.add(mesh);
      }
      groups.rawCeilingsRoles.add(createEdgeLoop(corners, color, 0.85));
    });
  });

  const reconBucket = (rawCeilingPrototype.reconstructions || {})[bldg.uuid] || {};
  const dormerPieces = Array.isArray(reconBucket.dormer) ? reconBucket.dormer : [];
  const wingPieces = Array.isArray(reconBucket.wing) ? reconBucket.wing : [];
  const pieceIdFromElementId = (elementId, story, roomIndex, spi) => {
    if (typeof elementId === 'string') {
      const idx = elementId.lastIndexOf('::');
      if (idx >= 0) return elementId.slice(idx + 2);
    }
    return `${story}:${roomIndex}:${spi ?? 0}`;
  };
  for (const piece of dormerPieces) {
    const corners = piece?.corners;
    if (!Array.isArray(corners) || corners.length < 3) continue;
    const color = CEILING_RECON_DORMER_COLORS[piece.piece_role] ?? CEILING_RECON_DORMER_COLORS.slope;
    const mesh = createPolygonMesh(corners, color, 0.7);
    if (mesh) {
      mesh.renderOrder = 60;
      attachLocator(mesh, {
        buildingUuid: bldg.uuid,
        kind: 'ceiling-reconstruction-dormer',
        id: pieceIdFromElementId(piece.element_id, piece.story, piece.room_index, piece.source_plane_index),
        roomId: `${piece.story}:${piece.room_index}`,
        story: piece.story,
        source: piece.piece_role || 'dormer',
        corners,
        lineage: [],
      });
      groups.rawCeilingsReconstructions.add(mesh);
    }
    groups.rawCeilingsReconstructions.add(createEdgeLoop(corners, color, 0.95));
  }
  for (const piece of wingPieces) {
    const corners = piece?.corners;
    if (!Array.isArray(corners) || corners.length < 3) continue;
    const color = CEILING_RECON_WING_COLORS[piece.piece_role] ?? CEILING_RECON_WING_COLORS.slope;
    const mesh = createPolygonMesh(corners, color, 0.7);
    if (mesh) {
      mesh.renderOrder = 60;
      attachLocator(mesh, {
        buildingUuid: bldg.uuid,
        kind: 'ceiling-reconstruction-wing',
        id: pieceIdFromElementId(piece.element_id, piece.story, piece.room_index, piece.source_plane_index),
        roomId: `${piece.story}:${piece.room_index}`,
        story: piece.story,
        source: piece.piece_role || 'wing',
        corners,
        lineage: [],
      });
      groups.rawCeilingsReconstructions.add(mesh);
    }
    groups.rawCeilingsReconstructions.add(createEdgeLoop(corners, color, 0.95));
  }
}

function maybeLoadRawCeilingPrototypeForCurrentBuilding() {
  const bldg = DATA?.[currentBuilding];
  if (!bldg?.uuid) return;
  const wantRoles = !!document.getElementById('show-raw-ceilings-roles')?.checked;
  const wantRecon = !!document.getElementById('show-raw-ceilings-reconstructions')?.checked;
  if (!wantRoles && !wantRecon) return;
  ensureRawCeilingPrototype().then(() => {
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    renderRawCeilingPrototypeForBuilding(bldg);
  });
}

function ensureComputedOverextend() {
  if (computedOverextend !== null) return Promise.resolve(computedOverextend);
  if (computedOverextendPromise) return computedOverextendPromise;
  computedOverextendPromise = fetch('/computed-overextend')
    .then(r => (r.ok ? r.json() : null))
    .then(data => {
      computedOverextend = data && typeof data === 'object' ? data : { buildings: {} };
      return computedOverextend;
    })
    .catch(() => {
      computedOverextend = { buildings: {} };
      return computedOverextend;
    });
  return computedOverextendPromise;
}

function overextendColor(fraction) {
  if (fraction <= 0.1) return COMPUTED_OVEREXTEND_COLORS.low;
  if (fraction <= 0.4) return COMPUTED_OVEREXTEND_COLORS.mid;
  return COMPUTED_OVEREXTEND_COLORS.high;
}

function renderComputedOverextendForBuilding(bldg) {
  if (!bldg?.uuid || !computedOverextend) return;
  const prefix = `${bldg.uuid}::roof-overextend::`;
  for (const uid of Array.from(elementMeshByUid.keys())) {
    if (typeof uid === 'string' && uid.startsWith(prefix)) {
      elementMeshByUid.delete(uid);
    }
  }
  disposeGroup(groups.computedOverextend);

  const pieces = (computedOverextend.buildings || {})[bldg.uuid] || [];
  for (const piece of pieces) {
    const topCorners = piece?.corners;
    if (!Array.isArray(topCorners) || topCorners.length < 3) continue;
    const fraction = Number(piece.overextend_fraction_xz) || 0;
    const color = overextendColor(fraction);
    // Top face — the overextend polygon lifted to the computed surface's plane.
    const topMesh = createPolygonMesh(topCorners, color, 0.55);
    if (topMesh) {
      topMesh.renderOrder = 58;
      attachLocator(topMesh, {
        buildingUuid: bldg.uuid,
        kind: 'roof-overextend',
        id: `${piece.surface_kind}:${piece.surface_index}`,
        roomId: null,
        story: piece.story ?? null,
        source: `${piece.surface_kind} overextend ${(fraction * 100).toFixed(1)}% Δy=${(piece.overextend_y_m ?? 0).toFixed(2)}m`,
        corners: topCorners,
        lineage: [piece.source_element_id],
      });
      groups.computedOverextend.add(topMesh);
    }
    groups.computedOverextend.add(createEdgeLoop(topCorners, color, 0.95));

    // 3D drop from the computed surface down to the highest raw-ceiling corner,
    // so the overlay visibly encloses the air volume the pipeline extrapolated
    // beyond scan evidence. Skip when raw_y_max is missing (no overlap) or the
    // gap is trivially small (flat roofs whose Y matches raw already).
    const rawYMax = piece.raw_y_max;
    const gap = Number(piece.overextend_y_m);
    if (typeof rawYMax === 'number' && Number.isFinite(gap) && gap > 0.05) {
      const bottomCorners = topCorners.map(c => [c[0], rawYMax, c[2]]);
      // Side walls as explicit triangles — each side quad is non-planar (sloped
      // top edge, horizontal bottom) so it must be split, otherwise
      // createPolygonMesh's fit-plane step flattens the whole ring into a
      // single bbox-sized quad.
      const wallTris = [];
      for (let i = 0; i < topCorners.length; i++) {
        const j = (i + 1) % topCorners.length;
        wallTris.push([topCorners[i], topCorners[j], bottomCorners[j]]);
        wallTris.push([topCorners[i], bottomCorners[j], bottomCorners[i]]);
      }
      const walls = createTriangleMesh(wallTris, color, 0.25);
      if (walls) {
        walls.renderOrder = 57;
        groups.computedOverextend.add(walls);
      }
      // Raw-max floor outline so the datum is visible.
      groups.computedOverextend.add(createEdgeLoop(bottomCorners, color, 0.6));
    }
  }
}

function maybeLoadComputedOverextendForCurrentBuilding() {
  const bldg = DATA?.[currentBuilding];
  if (!bldg?.uuid) return;
  if (!document.getElementById('show-computed-overextend')?.checked) return;
  ensureComputedOverextend().then(() => {
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    renderComputedOverextendForBuilding(bldg);
  });
}

function ensureRawDisagreement() {
  if (rawDisagreement !== null) return Promise.resolve(rawDisagreement);
  if (rawDisagreementPromise) return rawDisagreementPromise;
  rawDisagreementPromise = fetch('/raw-disagreement')
    .then(r => (r.ok ? r.json() : null))
    .then(data => {
      rawDisagreement = data && typeof data === 'object' ? data : { buildings: {} };
      return rawDisagreement;
    })
    .catch(() => {
      rawDisagreement = { buildings: {} };
      return rawDisagreement;
    });
  return rawDisagreementPromise;
}

function getRawCeilingSplitVersionMode() {
  const value = String(document.getElementById('raw-split-version-mode')?.value || 'both').toLowerCase();
  if (value === 'v1' || value === 'v2' || value === 'both') return value;
  return 'both';
}

function getRawCeilingSplitVersionsToRender() {
  const mode = getRawCeilingSplitVersionMode();
  return mode === 'both' ? RAW_CEILING_SPLIT_VERSIONS : [mode];
}

function ensureRawCeilingPlaneSplits(version = 'v1') {
  const key = RAW_CEILING_SPLIT_VERSIONS.includes(version) ? version : 'v1';
  if (rawCeilingPlaneSplitsByVersion[key] !== null) return Promise.resolve(rawCeilingPlaneSplitsByVersion[key]);
  if (rawCeilingPlaneSplitsPromiseByVersion[key]) return rawCeilingPlaneSplitsPromiseByVersion[key];
  rawCeilingPlaneSplitsPromiseByVersion[key] = fetch(`/raw-ceiling-plane-splits?version=${encodeURIComponent(key)}`)
    .then(r => (r.ok ? r.json() : null))
    .then(data => {
      rawCeilingPlaneSplitsByVersion[key] = data && typeof data === 'object' ? data : { buildings: {}, available: false };
      return rawCeilingPlaneSplitsByVersion[key];
    })
    .catch(() => {
      rawCeilingPlaneSplitsByVersion[key] = { buildings: {}, available: false };
      return rawCeilingPlaneSplitsByVersion[key];
    });
  return rawCeilingPlaneSplitsPromiseByVersion[key];
}

function rawCeilingSplitColor(piece, { finalLayer = false, version = 'v1' } = {}) {
  if (piece?.piece_role === 'intersection_seam') return RAW_CEILING_SPLIT_COLORS.intersection_seam;
  if (piece?.piece_role === 'residual') return RAW_CEILING_SPLIT_COLORS.residual;
  const support = Number(piece?.support_score) || 0;
  if (!finalLayer) {
    // Keep not-final pieces intentionally muted so they read as diagnostics,
    // not final roof surfaces.
    return support >= 0.85 ? 0x64748b : support >= 0.7 ? 0x94a3b8 : 0xcbd5e1;
  }
  if (version === 'v1') {
    return support >= 0.85 ? 0xea580c : support >= 0.7 ? 0xf97316 : 0xf59e0b;
  }
  return support >= 0.85 ? 0x16a34a : support >= 0.7 ? 0x22c55e : 0x84cc16;
}

function rawCeilingSplitOpacity(piece, { finalLayer = false } = {}) {
  if (piece?.piece_role === 'intersection_seam') return finalLayer ? 0.36 : 0.22;
  if (piece?.piece_role === 'residual') return finalLayer ? 0.18 : 0.12;
  return finalLayer ? 0.62 : 0.18;
}

function rawCeilingSplitEdgeColor(piece, fillColor, { version = 'v1', compareMode = false, finalLayer = false } = {}) {
  if (piece?.piece_role === 'intersection_seam') return RAW_CEILING_SPLIT_COLORS.intersection_seam;
  if (piece?.piece_role === 'residual') return fillColor;
  if (piece?.provenance_relevance_flag === 'suspect_interior_slice') {
    return RAW_CEILING_SPLIT_COLORS.suspect_interior_slice;
  }
  const competitorLoss = Number(piece?.local_competitor_loss_fraction);
  if (Number.isFinite(competitorLoss) && competitorLoss > 0.15) {
    return RAW_CEILING_SPLIT_COLORS.ownership_competitor_loss;
  }
  const throughRatio = Number(piece?.through_ratio);
  const hasMirror = typeof piece?.mirror_partner_plane_group_id === 'string'
    ? piece.mirror_partner_plane_group_id.length > 0
    : !!piece?.mirror_partner_plane_group_id;
  if (!hasMirror && Number.isFinite(throughRatio) && throughRatio > 1.0) {
    return RAW_CEILING_SPLIT_COLORS.ownership_unpaired_through;
  }
  if (!finalLayer) {
    return RAW_CEILING_SPLIT_VERSION_COLORS[version]?.candidateEdge || 0x1f2937;
  }
  if (compareMode) {
    return RAW_CEILING_SPLIT_VERSION_COLORS[version]?.edge || fillColor;
  }
  return fillColor;
}

function rawCeilingSplitSource(piece, { version = 'v1' } = {}) {
  const chainIds = Array.isArray(piece?.chain_ids) ? piece.chain_ids : [];
  const support = (Number(piece?.support_score) || 0).toFixed(2);
  const throughRatio = Number(piece?.through_ratio);
  const competitorLoss = Number(piece?.local_competitor_loss_fraction);
  const competitorIds = Array.isArray(piece?.local_top_competitor_ids)
    ? piece.local_top_competitor_ids.filter(Boolean)
    : [];
  const mirrorPartner = piece?.mirror_partner_plane_group_id || null;
  const chainResidual = Number(piece?.best_supported_chain_height_residual_m);
  const suspectReasons = Array.isArray(piece?.provenance_relevance_reasons)
    ? piece.provenance_relevance_reasons.join(', ')
    : '';
  const versionLabel = RAW_CEILING_SPLIT_VERSION_LABELS[version] || version.toUpperCase();

  if (piece?.piece_role === 'residual') {
    return `${versionLabel} raw eave split residual from ${piece.target_element_id}`;
  }
  if (piece?.piece_role === 'intersection_seam') {
    const partner = piece?.pair_partner_target_id || 'unknown partner';
    const disputed = Number(piece?.pair_disputed_overlap_in_piece_m2);
    const partnerEvidence = Number(piece?.pair_partner_evidence_area_m2);
    const overlap = Number(piece?.pair_overlap_m2);
    const parts = [`vs ${partner}`];
    if (Number.isFinite(disputed)) parts.push(`disputed in piece ${disputed.toFixed(2)}m²`);
    if (Number.isFinite(overlap)) parts.push(`pair overlap ${overlap.toFixed(2)}m²`);
    if (Number.isFinite(partnerEvidence)) parts.push(`partner evidence ${partnerEvidence.toFixed(2)}m²`);
    return `${versionLabel} intersection seam (${parts.join(', ')})`;
  }

  const details = [
    `support ${support}`,
    `${chainIds.length} chain(s)`,
  ];
  if (Number.isFinite(throughRatio)) details.push(`through ${throughRatio.toFixed(2)}x`);
  if (Number.isFinite(competitorLoss)) details.push(`competitor loss ${competitorLoss.toFixed(2)}`);
  if (competitorIds.length) details.push(`top competitor ${competitorIds[0]}`);
  if (mirrorPartner) {
    details.push(`mirror ${mirrorPartner}`);
  } else if (Number.isFinite(throughRatio)) {
    details.push('no mirror');
  }
  if (Number.isFinite(chainResidual)) details.push(`chain height residual ${chainResidual.toFixed(2)}m`);

  if (piece?.provenance_relevance_flag === 'suspect_interior_slice') {
    details.push(`creator rain area ${(Number(piece.creator_rain_area_fraction) || 0).toFixed(2)}`);
    details.push(
      `covered/rain creators ${Number(piece.creator_covered_segment_count) || 0}/${Number(piece.creator_rain_segment_count) || 0}`,
    );
    if (suspectReasons) details.push(suspectReasons);
    return `${versionLabel} raw eave split suspect ${details.join(', ')}`;
  }
  return `${versionLabel} raw eave split ${details.join(', ')}`;
}

function isFinalRawCeilingSplitPiece(piece) {
  if (piece?.piece_role === 'intersection_seam') return true;
  if (typeof piece?.final_layer === 'boolean') {
    return piece.final_layer;
  }
  return piece?.target_kind === 'committed_oblique';
}

function splitRawCeilingPiecesForLayers(pieces) {
  const visiblePieces = pieces.filter((piece) => !piece?.overlay_suppressed);
  const finalPieces = visiblePieces.filter(isFinalRawCeilingSplitPiece);
  const candidatePieces = pieces.filter((piece) => {
    if (piece?.overlay_suppressed) return false;
    if (isFinalRawCeilingSplitPiece(piece)) return false;
    return true;
  });
  return { finalPieces, candidatePieces };
}

function clearLocatorEntriesForGroup(group) {
  if (!group) return;
  group.traverse((child) => {
    const uid = child?.userData?.elementUid;
    if (typeof uid === 'string') elementMeshByUid.delete(uid);
  });
}

function renderRawCeilingPlaneSplitGroup(
  bldg,
  group,
  pieces,
  { finalLayer = false, version = 'v1', compareMode = false, append = false } = {},
) {
  if (!append) {
    clearLocatorEntriesForGroup(group);
    disposeGroup(group);
  }
  const seenRenderKeys = new Set();
  for (const piece of pieces) {
    const renderKey = `${version}|${finalLayer ? 'final' : 'candidate'}|${String(piece?.piece_id || piece?.target_element_id || '')}`;
    if (seenRenderKeys.has(renderKey)) continue;
    seenRenderKeys.add(renderKey);
    const corners = piece?.corners;
    if (!Array.isArray(corners) || corners.length < 3) continue;
    const holes = Array.isArray(piece?.holes) ? piece.holes.filter(h => Array.isArray(h) && h.length >= 3) : [];
    const color = rawCeilingSplitColor(piece, { finalLayer, version });
    const edgeColor = rawCeilingSplitEdgeColor(piece, color, { version, compareMode, finalLayer });
    const baseOpacity = rawCeilingSplitOpacity(piece, { finalLayer });
    const opacity = compareMode
      ? (piece?.piece_role === 'residual' ? 0.14 : (finalLayer ? 0.34 : 0.22))
      : baseOpacity;
    const mesh = createPolygonMesh(corners, color, opacity, holes);
    if (mesh) {
      const versionBump = version === 'v2' ? 1 : 0;
      mesh.renderOrder = piece?.piece_role === 'residual'
        ? (57 + versionBump)
        : (finalLayer ? (59 + versionBump) : (58 + versionBump));
      const chainIds = Array.isArray(piece?.chain_ids) ? piece.chain_ids : [];
      const versionLabel = RAW_CEILING_SPLIT_VERSION_LABELS[version] || version.toUpperCase();
      const layerLabel = finalLayer ? 'final-layer roof-plane split' : 'not-final roof-plane split';
      const locatorKind = version === 'v2' ? 'raw-eave-split-v2' : 'raw-eave-split-v1';
      attachLocator(mesh, {
        buildingUuid: bldg.uuid,
        kind: locatorKind,
        id: String(piece.piece_id || piece.target_element_id || 'piece'),
        targetKind: piece.target_kind || null,
        rawEaveSplitLayer: finalLayer ? 'final' : 'candidate',
        rawEaveSplitVersion: version,
        roomId: null,
        story: piece.story ?? null,
        source: `${versionLabel} ${layerLabel} (${piece.target_kind || 'unknown'}) — ${rawCeilingSplitSource(piece, { version })}`,
        corners,
        lineage: [piece.target_element_id, ...chainIds].filter(Boolean),
      });
      group.add(mesh);
    }
    const edgeOpacity = compareMode
      ? (finalLayer ? 0.95 : 0.72)
      : (finalLayer ? (piece?.piece_role === 'residual' ? 0.35 : 0.9) : 0.6);
    group.add(createEdgeLoop(corners, edgeColor, edgeOpacity));
  }
}

function updateRawSplitCompareStatusForBuilding(bldg) {
  const statusEl = document.getElementById('raw-split-compare-status');
  if (!statusEl) return;
  if (!bldg?.uuid) {
    statusEl.textContent = 'V1/V2: -';
    return;
  }

  function countsFor(version) {
    const payload = rawCeilingPlaneSplitsByVersion[version];
    if (!payload) return { loading: true, final: 0, candidate: 0 };
    const pieces = ((payload.buildings || {})[bldg.uuid] || []).filter((piece) => !piece?.overlay_suppressed);
    const split = splitRawCeilingPiecesForLayers(pieces);
    return {
      loading: false,
      available: payload.available !== false,
      final: split.finalPieces.length,
      candidate: split.candidatePieces.length,
    };
  }

  const mode = getRawCeilingSplitVersionMode();
  const v1 = countsFor('v1');
  const v2 = countsFor('v2');
  const fmt = (label, c) => (c.loading ? `${label} …` : `${label} F${c.final}/C${c.candidate}${c.available === false ? ' (n/a)' : ''}`);
  statusEl.textContent = mode === 'both'
    ? `${fmt('V1', v1)} | ${fmt('V2', v2)}`
    : (mode === 'v1' ? fmt('V1', v1) : fmt('V2', v2));
}

function renderRawCeilingPlaneSplitsForBuilding(bldg) {
  if (!bldg?.uuid) return;
  const mode = getRawCeilingSplitVersionMode();
  const versions = getRawCeilingSplitVersionsToRender();
  const compareMode = mode === 'both';

  clearLocatorEntriesForGroup(groups.rawCeilingPlaneSplits);
  disposeGroup(groups.rawCeilingPlaneSplits);
  clearLocatorEntriesForGroup(groups.rawCeilingPlaneSplitCandidates);
  disposeGroup(groups.rawCeilingPlaneSplitCandidates);

  let appendedFinal = false;
  let appendedCandidate = false;
  for (const version of versions) {
    const payload = rawCeilingPlaneSplitsByVersion[version];
    if (!payload) continue;
    const pieces = (payload.buildings || {})[bldg.uuid] || [];
    const { finalPieces, candidatePieces } = splitRawCeilingPiecesForLayers(pieces);
    renderRawCeilingPlaneSplitGroup(
      bldg,
      groups.rawCeilingPlaneSplits,
      finalPieces,
      { finalLayer: true, version, compareMode, append: appendedFinal },
    );
    renderRawCeilingPlaneSplitGroup(
      bldg,
      groups.rawCeilingPlaneSplitCandidates,
      candidatePieces,
      { finalLayer: false, version, compareMode, append: appendedCandidate },
    );
    appendedFinal = true;
    appendedCandidate = true;
  }
  updateRawSplitCompareStatusForBuilding(bldg);
}

function maybeLoadRawCeilingPlaneSplitsForCurrentBuilding() {
  const bldg = DATA?.[currentBuilding];
  if (!bldg?.uuid) return;
  const showFinal = document.getElementById('show-raw-ceiling-plane-splits')?.checked;
  const showCandidates = document.getElementById('show-raw-ceiling-plane-split-candidates')?.checked;
  const versions = getRawCeilingSplitVersionsToRender();
  if (!showFinal && !showCandidates) {
    updateRawSplitCompareStatusForBuilding(bldg);
    return;
  }
  updateRawSplitCompareStatusForBuilding(bldg);
  Promise.all(versions.map((version) => ensureRawCeilingPlaneSplits(version))).then(() => {
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    renderRawCeilingPlaneSplitsForBuilding(bldg);
  });
}

function disagreementColor(angleDeg) {
  if (angleDeg <= 30) return RAW_DISAGREEMENT_COLORS.low;
  if (angleDeg <= 60) return RAW_DISAGREEMENT_COLORS.mid;
  return RAW_DISAGREEMENT_COLORS.high;
}

function renderRawDisagreementForBuilding(bldg) {
  if (!bldg?.uuid || !rawDisagreement) return;
  const prefix = `${bldg.uuid}::raw-disagreement::`;
  for (const uid of Array.from(elementMeshByUid.keys())) {
    if (typeof uid === 'string' && uid.startsWith(prefix)) {
      elementMeshByUid.delete(uid);
    }
  }
  disposeGroup(groups.rawDisagreement);

  const pieces = (rawDisagreement.buildings || {})[bldg.uuid] || [];
  for (const piece of pieces) {
    const corners = piece?.corners;
    if (!Array.isArray(corners) || corners.length < 3) continue;
    const angle = Number(piece.angle_deg) || 0;
    const color = disagreementColor(angle);
    const mesh = createPolygonMesh(corners, color, 0.7);
    if (mesh) {
      mesh.renderOrder = 60;
      attachLocator(mesh, {
        buildingUuid: bldg.uuid,
        kind: 'raw-disagreement',
        id: String(piece.pair_index),
        roomId: null,
        story: piece.story ?? null,
        source: `raw pair Δ=${angle.toFixed(0)}° (az ${piece.azimuth_i}°/${piece.inclination_i}° vs ${piece.azimuth_j}°/${piece.inclination_j}°) overlap ${piece.overlap_area_m2}m²`,
        corners,
        lineage: [piece.plane_i_element_id, piece.plane_j_element_id].filter(Boolean),
      });
      groups.rawDisagreement.add(mesh);
    }
    groups.rawDisagreement.add(createEdgeLoop(corners, color, 1.0));
  }
}

function maybeLoadRawDisagreementForCurrentBuilding() {
  const bldg = DATA?.[currentBuilding];
  if (!bldg?.uuid) return;
  if (!document.getElementById('show-raw-disagreement')?.checked) return;
  ensureRawDisagreement().then(() => {
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    renderRawDisagreementForBuilding(bldg);
  });
}

function ensureCeilingReplacement() {
  if (ceilingReplacement !== null) return Promise.resolve(ceilingReplacement);
  if (ceilingReplacementPromise) return ceilingReplacementPromise;
  ceilingReplacementPromise = fetch('/ceiling-replacement')
    .then(r => (r.ok ? r.json() : null))
    .then(data => {
      ceilingReplacement = data && typeof data === 'object' ? data : { buildings: {} };
      return ceilingReplacement;
    })
    .catch(() => {
      ceilingReplacement = { buildings: {} };
      return ceilingReplacement;
    });
  return ceilingReplacementPromise;
}

function replacementColor(density) {
  if (density <= 0.7) return CEILING_REPLACEMENT_COLORS.low;
  if (density <= 1.5) return CEILING_REPLACEMENT_COLORS.mid;
  return CEILING_REPLACEMENT_COLORS.high;
}

function renderCeilingReplacementForBuilding(bldg) {
  if (!bldg?.uuid || !ceilingReplacement) return;
  const prefix = `${bldg.uuid}::clean-ceiling::`;
  for (const uid of Array.from(elementMeshByUid.keys())) {
    if (typeof uid === 'string' && uid.startsWith(prefix)) {
      elementMeshByUid.delete(uid);
    }
  }
  disposeGroup(groups.ceilingReplacement);

  const pieces = (ceilingReplacement.buildings || {})[bldg.uuid] || [];
  for (const piece of pieces) {
    const corners = piece?.corners;
    if (!Array.isArray(corners) || corners.length < 3) continue;
    const density = Number(piece.max_density_planes_per_m2 ?? piece.density_planes_per_m2) || 0;
    const color = replacementColor(density);
    const pieceRole = piece.piece_role || 'oblique';
    const opacity = pieceRole === 'flat-cap' ? 0.58 : 0.72;
    const mesh = createPolygonMesh(corners, color, opacity);
    if (mesh) {
      mesh.renderOrder = 59;
      const elementId = String(piece.element_id || `${bldg.uuid}::clean-ceiling::${piece.story}:${piece.room_index}`);
      const idTail = elementId.split('::').slice(2).join('::') || `${piece.story}:${piece.room_index}`;
      const componentDensity = Number(piece.component_density_planes_per_m2) || 0;
      const mode = piece.replacement_mode || 'single-oblique';
      const mixedFit = Number(piece.mixed_fit_iou) || 0;
      const gapArea = Number(piece.gap_extension_area_m2) || 0;
      const selectionMode = String(piece.replacement_selection_mode || 'overlap-only');
      const azDelta = Number(piece.replacement_azimuth_delta_deg);
      const noMeshPromoted = Number(piece.promote_nomesh_computed) === 1;
      const topologyApplied = Number(piece.topology_filter_applied) === 1;
      const topologyAction = String(piece.topology_filter_action || '');
      const topExposedArea = Number(piece.top_exposed_area_m2) || 0;
      const roomArea = Number(piece.floor_area_m2) || 0;
      let topologyNote = '';
      if (topologyApplied && topologyAction === 'clip-to-top-exposed') {
        topologyNote = `, topology clip ${topExposedArea.toFixed(2)}m² exposed / ${roomArea.toFixed(2)}m² room`;
      } else if (topologyApplied && topologyAction) {
        topologyNote = `, topology ${topologyAction.replaceAll('-', ' ')}`;
      }
      attachLocator(mesh, {
        buildingUuid: bldg.uuid,
        kind: 'clean-ceiling',
        id: idTail,
        roomId: null,
        story: piece.story ?? null,
        source: `clean ceiling (${pieceRole}, ${mode}, ${selectionMode}${noMeshPromoted ? ', noMesh computed-backed' : ''}) — ${piece.n_raw_planes} raw planes, room ${Number(piece.density_planes_per_m2 || 0).toFixed(2)}/m², local ${componentDensity.toFixed(2)}/m², fit ${mixedFit.toFixed(2)}, gap +${gapArea.toFixed(2)}m², az Δ ${Number.isFinite(azDelta) ? azDelta.toFixed(1) : '-'}°, max slant ${piece.max_incl_deg}°, room floor ${piece.floor_area_m2}m²${topologyNote}`,
        corners,
        lineage: piece.raw_plane_element_ids || [],
      });
      groups.ceilingReplacement.add(mesh);
    }
    groups.ceilingReplacement.add(createEdgeLoop(corners, color, 1.0));
  }
}

function maybeLoadCeilingReplacementForCurrentBuilding() {
  const bldg = DATA?.[currentBuilding];
  if (!bldg?.uuid) return;
  if (!document.getElementById('show-ceiling-replacement')?.checked) return;
  ensureCeilingReplacement().then(() => {
    if (!DATA[currentBuilding] || DATA[currentBuilding].uuid !== bldg.uuid) return;
    renderCeilingReplacementForBuilding(bldg);
  });
}

Promise.all([
  fetchJsonWithLastModified(`buildings_3d.json?v=${Date.now()}`),
  fetchJsonWithLastModified(`roof_algorithms_py_results.json?v=${Date.now()}`).catch(() => ({
    data: {},
    lastModifiedMs: null,
  })),
])
  .then(([buildingPayload, roofPayload]) => {
    const buildingsMtime = buildingPayload.lastModifiedMs;
    const roofMtime = roofPayload.lastModifiedMs;
    const roofFreshEnough = (
      roofMtime == null ||
      buildingsMtime == null ||
      roofMtime >= buildingsMtime
    );
    if (!roofFreshEnough) {
      console.warn(
        'Ignoring stale roof_algorithms_py_results.json because it is older than buildings_3d.json',
        { roofMtime, buildingsMtime },
      );
      pyRoofByUuid = {};
    } else {
      pyRoofByUuid = (roofPayload.data && typeof roofPayload.data === 'object') ? roofPayload.data : {};
    }
    return Promise.all([
      fetch('/alignment-calibration').then(r => (r.ok ? r.json() : {})).catch(() => ({})),
      fetch('/roof-rating').then(r => (r.ok ? r.json() : {})).catch(() => ({})),
    ]).then(([calib, ratings]) => {
      alignmentByUuid = (calib && typeof calib === 'object') ? calib : {};
      roofRatingsByUuid = (ratings && typeof ratings === 'object') ? ratings : {};
      return buildingPayload.data;
    });
  })
  .then(data => {
    DATA = data;
    buildingIndexByUuid = new Map();
    data.forEach((b, i) => {
      if (b?.uuid) buildingIndexByUuid.set(b.uuid, i);
    });

    // Sort by address
    const indices = data.map((b, i) => i);
    indices.sort((a, b) => (data[a].address || '').localeCompare(data[b].address || ''));

    const list = document.getElementById('building-list');
    for (const i of indices) {
      const b = data[i];
      const el = document.createElement('div');
      el.className = 'bldg-item';
      const addr = b.address || b.uuid.slice(0, 12);
      const computedCount = b.rooms.reduce((s, r) => s + r.walls_computed.length, 0);
      const doorCount = b.rooms.reduce((s, r) => s + (r.doors || []).length, 0);
      const windowCount = b.rooms.reduce((s, r) => s + (r.windows || []).length, 0);
      const cls = b.classification || 'UNKNOWN';
      const storiesClass = b.stories_found > 1 ? 'multi' : '';
      const gapCount = (b.cross_floor_gaps || []).length;
      const gapBadge = gapCount > 0 ? `<span style="color:#f84">${gapCount}gaps</span>` : '';
      const extGapCount = (b.exterior_gap_indicators || []).length;
      const extGapBadge = extGapCount > 0 ? `<span style="color:#f4f">${extGapCount}eg</span>` : '';
      el.innerHTML =
        `<div class="bldg-addr">${addr}</div>` +
        `<div class="bldg-meta">` +
          `<span class="tag tag-${cls}">${cls}</span>` +
          `<span class="stories-badge ${storiesClass}">${b.stories_found}F</span>` +
          `<span>${b.rooms.length}r</span>` +
          `<span>${computedCount}w</span>` +
          `<span style="color:#c73">${doorCount}d</span>` +
          `<span style="color:#3ad">${windowCount}w</span>` +
          gapBadge +
          extGapBadge +
        `</div>`;
      el.dataset.search = `${addr} ${b.uuid} ${cls}`.toLowerCase();
      el.dataset.index = i;
      el.addEventListener('click', () => {
        document.querySelectorAll('.bldg-item').forEach(x => x.classList.remove('active'));
        el.classList.add('active');
        requestAnimationFrame(() => loadBuilding(i));
      });
      list.appendChild(el);
    }

    document.getElementById('sidebar-stats').textContent = `${data.length} buildings`;
    resizeRenderer();
    if (pendingElementUid) {
      const parsed = parseElementUid(pendingElementUid);
      if (parsed && buildingIndexByUuid.has(parsed.buildingUuid)) {
        const idx = buildingIndexByUuid.get(parsed.buildingUuid);
        const b = data[idx];
        applyAlignmentStateToControls(alignmentByUuid[b.uuid] || {});
        try {
          loadBuilding(idx);
        } catch (err) {
          console.error('Initial building load failed', b?.uuid, err);
          document.getElementById('building-info').textContent = `Initial building load failed for ${b?.uuid || 'unknown building'}`;
        }
        jumpToElementUid(pendingElementUid, { focus: true, updateHash: false });
      } else if (indices.length > 0) {
        const b = data[indices[0]];
        applyAlignmentStateToControls(alignmentByUuid[b.uuid] || {});
        try {
          loadBuilding(indices[0]);
        } catch (err) {
          console.error('Initial building load failed', b?.uuid, err);
          document.getElementById('building-info').textContent = `Initial building load failed for ${b?.uuid || 'unknown building'}`;
        }
      }
      pendingElementUid = null;
      pendingBuildingUuid = null;
    } else if (pendingBuildingUuid && buildingIndexByUuid.has(pendingBuildingUuid)) {
      const idx = buildingIndexByUuid.get(pendingBuildingUuid);
      const b = data[idx];
      applyAlignmentStateToControls(alignmentByUuid[b.uuid] || {});
      try {
        loadBuilding(idx);
      } catch (err) {
        console.error('Initial building load failed', b?.uuid, err);
        document.getElementById('building-info').textContent = `Initial building load failed for ${b?.uuid || 'unknown building'}`;
      }
      pendingBuildingUuid = null;
    } else if (indices.length > 0) {
      const b = data[indices[0]];
      applyAlignmentStateToControls(alignmentByUuid[b.uuid] || {});
      try {
        loadBuilding(indices[0]);
      } catch (err) {
        console.error('Initial building load failed', b?.uuid, err);
        document.getElementById('building-info').textContent = `Initial building load failed for ${b?.uuid || 'unknown building'}`;
      }
    }
    animate();
  })
  .catch(err => {
    console.error('Failed to load buildings_3d.json', err);
    document.getElementById('sidebar-stats').textContent = 'Failed to load buildings';
    document.getElementById('building-info').textContent = 'Could not load buildings_3d.json';
  });
