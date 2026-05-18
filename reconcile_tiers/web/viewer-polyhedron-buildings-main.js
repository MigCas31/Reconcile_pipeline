import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { populateBuildingScene } from "./tier-preview.js";

const params = new URLSearchParams(window.location.search);
const DEFAULT_TRACE_ROOT =
  "../../.context/polyhedron-envelope-roof-selection-safe-check";
const INDEX_URL = params.get("index") || `${DEFAULT_TRACE_ROOT}/index.json`;
const TRACE_ROOT = params.get("traceRoot") || traceRootForIndex(INDEX_URL);
const PIPELINE_ROOT = "../../pipeline-outputs";

function traceRootForIndex(indexUrl) {
  const slashIndex = indexUrl.lastIndexOf("/");
  return slashIndex >= 0 ? indexUrl.slice(0, slashIndex) : DEFAULT_TRACE_ROOT;
}

const viewport = document.querySelector("#viewport");
const canvas = document.querySelector("#view");
const buildingList = document.querySelector("#building-list");
const search = document.querySelector("#search");
const sidebarStats = document.querySelector("#sidebar-stats");
const currentTitle = document.querySelector("#current-title");
const currentMeta = document.querySelector("#current-meta");
const pill = document.querySelector("#pill");
const status = document.querySelector("#status");
const prevBuildingButton = document.querySelector("#prev-building");
const nextBuildingButton = document.querySelector("#next-building");
const edgesToggle = document.querySelector("#edges-toggle");
const ghostToggle = document.querySelector("#ghost-toggle");
const slopedToggle = document.querySelector("#sloped-toggle");
const roomsToggle = document.querySelector("#rooms-toggle");
const baseBuildingToggle = document.querySelector("#base-building-toggle");
const emittedOnlyToggle = document.querySelector("#emitted-only-toggle");
const roomAuditToggle = document.querySelector("#room-audit-toggle");
const cellsToggle = document.querySelector("#cells-toggle");
const droppedToggle = document.querySelector("#dropped-toggle");
const topCandidatesToggle = document.querySelector("#top-candidates-toggle");
const partPlanesToggle = document.querySelector("#part-planes-toggle");
const openOriginal = document.querySelector("#open-original");
const partCounts = document.querySelector("#part-counts");
const partBody = document.querySelector("#part-body");

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

const PEEK_OPACITY = 0.25;
const peekRaycaster = new THREE.Raycaster();
const peekPointer = new THREE.Vector2();
const pointerDown = { x: 0, y: 0, time: 0, button: -1 };

canvas.addEventListener("pointerdown", (event) => {
  pointerDown.x = event.clientX;
  pointerDown.y = event.clientY;
  pointerDown.time = performance.now();
  pointerDown.button = event.button;
});

canvas.addEventListener("pointerup", (event) => {
  if (event.button !== 0 || pointerDown.button !== 0) return;
  if (performance.now() - pointerDown.time > 350) return;
  const dx = event.clientX - pointerDown.x;
  const dy = event.clientY - pointerDown.y;
  if (dx * dx + dy * dy > 16) return;
  togglePeekAt(event);
});

function togglePeekAt(event) {
  const rect = canvas.getBoundingClientRect();
  peekPointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  peekPointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  peekRaycaster.setFromCamera(peekPointer, camera);
  const peekable = [];
  modelGroup.traverse((obj) => {
    if (obj.isMesh && obj.userData.peekable) peekable.push(obj);
  });
  const hits = peekRaycaster.intersectObjects(peekable, false);
  if (!hits.length) return;
  const mesh = hits[0].object;
  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  const peeked = materials.some((m) => m.userData.peeked);
  for (const material of materials) {
    if (peeked) {
      material.transparent = false;
      material.opacity = 1;
      material.depthWrite = true;
      material.userData.peeked = false;
    } else {
      material.transparent = true;
      material.opacity = PEEK_OPACITY;
      material.depthWrite = false;
      material.userData.peeked = true;
    }
    material.needsUpdate = true;
  }
  requestRender();
}

const wallPalette = [
  0x3b82f6, 0x4f46e5, 0x0891b2, 0x0f766e, 0x7c3aed, 0x0284c7,
  0x475569, 0x2563eb, 0x0e7490, 0x4338ca, 0x0369a1, 0x64748b,
];

const materials = new Map();
const edgeMaterial = new THREE.LineBasicMaterial({
  color: 0x17202a,
  transparent: true,
  opacity: 0.68,
});
const ghostMaterial = new THREE.MeshBasicMaterial({
  color: 0x64748b,
  transparent: true,
  opacity: 0.13,
  side: THREE.DoubleSide,
  depthWrite: false,
});
const roomFootprintMaterial = new THREE.LineBasicMaterial({
  color: 0xef4444,
  transparent: true,
  opacity: 0.9,
  depthTest: false,
});
const coveredRoomMaterial = new THREE.LineBasicMaterial({
  color: 0x16a34a,
  transparent: true,
  opacity: 0.95,
  depthTest: false,
});
const partialRoomMaterial = new THREE.LineBasicMaterial({
  color: 0xd97706,
  transparent: true,
  opacity: 0.95,
  depthTest: false,
});
const droppedRoomMaterial = new THREE.LineBasicMaterial({
  color: 0xdc2626,
  transparent: true,
  opacity: 1,
  depthTest: false,
});
const cellEdgeMaterial = new THREE.LineBasicMaterial({
  color: 0x0f766e,
  transparent: true,
  opacity: 0.95,
});
const planeEdgeMaterial = new THREE.LineBasicMaterial({
  color: 0x7c3aed,
  transparent: true,
  opacity: 0.9,
});
const candidateEdgeMaterial = new THREE.LineBasicMaterial({
  color: 0x0284c7,
  transparent: true,
  opacity: 0.82,
});
const flatMaterial = new THREE.MeshPhongMaterial({
  color: 0x22c55e,
  shininess: 18,
  transparent: false,
  opacity: 1,
  side: THREE.DoubleSide,
});
const slopedMaterial = new THREE.MeshPhongMaterial({
  color: 0xf97316,
  emissive: 0x3d1600,
  shininess: 22,
  transparent: false,
  opacity: 1,
  side: THREE.DoubleSide,
});

let indexData = null;
let groups = [];
let visibleGroups = [];
let activeGroup = null;
let activeParts = [];
let activePayload = null;
let activeAudit = null;
let renderQueued = false;

function wallMaterial(partIndex) {
  const key = `wall:${partIndex}`;
  if (!materials.has(key)) {
    materials.set(
      key,
      new THREE.MeshPhongMaterial({
        color: wallPalette[Math.abs(partIndex) % wallPalette.length],
        shininess: 16,
        transparent: false,
        opacity: 1,
        side: THREE.DoubleSide,
      }),
    );
  }
  return materials.get(key);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatCounts(counts) {
  if (!counts) return "0F 0V 0HE";
  return `${counts.faces}F ${counts.vertices}V ${counts.half_edges}HE`;
}

function sourceSummary(row) {
  const source = row.top_source || "";
  if (!source) return row.locator_id || `part ${row.part_index}`;
  const pieces = source.split(" + ");
  const first = pieces[0]?.split("::").slice(-2).join("::") || source;
  return pieces.length > 1 ? `${first} + ${pieces.length - 1} more` : first;
}

function groupMatches(group, query) {
  if (!query) return true;
  return `${group.uuid} ${group.rows.map((row) => `${row.locator_id} ${row.top_source}`).join(" ")}`
    .toLowerCase()
    .includes(query);
}

function buildGroups(records) {
  const byUuid = new Map();
  for (const row of records) {
    if (!row.trace) continue;
    if (!byUuid.has(row.uuid)) byUuid.set(row.uuid, []);
    byUuid.get(row.uuid).push(row);
  }
  return [...byUuid.entries()]
    .map(([uuid, rows]) => {
      rows.sort((a, b) => Number(a.part_index) - Number(b.part_index));
      const obliqueSourceCount = rows.filter((row) =>
        String(row.top_source || "").includes("oblique"),
      ).length;
      const stepCount = rows.reduce((sum, row) => sum + Number(row.step_count || 0), 0);
      const emittedCount = rows.filter((row) => rowEmitted(row)).length;
      const assembledCount = rows.filter((row) => rowAssembled(row)).length;
      const lowCoverageCount = rows.filter((row) => row.low_coverage).length;
      const assemblyCoverage = Math.max(
        ...rows.map((row) => Number(row.assembly_coverage_ratio || 0)),
      );
      const complete = selectorV2Mode()
        ? assemblyCoverage >= 0.95
        : false;
      const finalFaceCount = rows.reduce(
        (sum, row) => sum + Number(row.final_counts?.faces || 0),
        0,
      );
      return {
        uuid,
        rows,
        obliqueSourceCount,
        stepCount,
        emittedCount,
        assembledCount,
        assemblyCoverage,
        lowCoverageCount,
        complete,
        finalFaceCount,
      };
    })
    .sort(
      (a, b) =>
        b.obliqueSourceCount - a.obliqueSourceCount ||
        b.rows.length - a.rows.length ||
        b.stepCount - a.stepCount ||
        a.uuid.localeCompare(b.uuid),
    );
}

function renderBuildingList() {
  const query = search.value.trim().toLowerCase();
  visibleGroups = groups.filter((group) => groupMatches(group, query));
  buildingList.innerHTML = visibleGroups
    .map((group) => {
      const active = group === activeGroup ? " active" : "";
      return `
        <button class="building-row${active}" data-uuid="${escapeHtml(group.uuid)}">
          <span class="label">${escapeHtml(group.uuid)}</span>
          <span class="meta">
            <span class="badge">${group.rows.length} ${selectorV2Mode() ? "domains" : "parts"}</span>
            ${selectorV2Mode() ? `<span class="badge${group.complete ? "" : " warn"}">${group.complete ? "complete" : "partial"}</span>` : ""}
            ${selectorV2Mode() ? `<span class="badge">${Math.round(group.assemblyCoverage * 100)}% assembled</span>` : ""}
            ${selectorV2Mode() ? `<span class="badge">${group.assembledCount} selected</span>` : ""}
            ${selectorV2Mode() ? `<span class="badge">${group.emittedCount} emitted</span>` : ""}
            ${selectorV2Mode() && group.lowCoverageCount ? `<span class="badge warn">${group.lowCoverageCount} low coverage</span>` : ""}
            ${selectorV2Mode() ? "" : `<span class="badge">${group.obliqueSourceCount} oblique sources</span>`}
            <span class="badge">${selectorV2Mode() ? "selection trace" : `${group.stepCount} trace steps`}</span>
          </span>
        </button>
      `;
    })
    .join("");

  for (const button of buildingList.querySelectorAll(".building-row")) {
    button.addEventListener("click", () => {
      const group = groups.find((candidate) => candidate.uuid === button.dataset.uuid);
      if (group) void loadBuilding(group);
    });
  }
}

async function loadBuilding(group) {
  activeGroup = group;
  activeParts = [];
  activeAudit = null;
  renderBuildingList();
  setHash(group.uuid);
  status.textContent = "Loading trace-fixed envelope parts...";

  const loaded = await Promise.all(
    group.rows.map(async (row) => {
      const response = await fetch(`${TRACE_ROOT}/${row.trace}`);
      if (!response.ok) throw new Error(`trace fetch failed: ${row.trace}: ${response.status}`);
      return { row, trace: await response.json() };
    }),
  );
  const payloadResponse = await fetch(`${PIPELINE_ROOT}/${group.uuid}/tier_payload.json`);
  activePayload = payloadResponse.ok ? await payloadResponse.json() : null;
  const auditPath = group.rows.find((row) => row.audit_path)?.audit_path;
  if (auditPath) {
    const auditResponse = await fetch(`${TRACE_ROOT}/${auditPath}`);
    activeAudit = auditResponse.ok ? await auditResponse.json() : null;
  } else if (indexData?.domain !== "selector-v2") {
    const auditResponse = await fetch(`../../.context/polyhedron-room-audit/audit/${group.uuid}.json`);
    activeAudit = auditResponse.ok ? await auditResponse.json() : null;
  }
  activeParts = loaded;
  renderBuilding();
  focusModel();
}

function renderBuilding() {
  clearGroup(modelGroup);
  if (!activeGroup) return;

  let finalFaces = 0;
  let renderedFaces = 0;
  let slopedFaces = 0;
  let fixedParts = 0;
  const partRows = [];
  const visibleParts = activeParts.filter((part) => shouldShowPart(part.row));

  if (baseBuildingToggle.checked && activePayload) {
    populateBuildingScene(modelGroup, activePayload, { style: "calm" });
    softenBaseBuilding(modelGroup);
  }

  for (const [index, part] of visibleParts.entries()) {
    const frames = part.trace.frames || [];
    if (!frames.length) continue;
    const initialFrame = frames[0];
    const finalFrame = frames[frames.length - 1];
    const partMetrics = frameMetrics(finalFrame);
    finalFaces += partMetrics.faces;
    slopedFaces += partMetrics.slopedFaces;
    if ((part.trace.steps || []).length) fixedParts += 1;

    if (ghostToggle.checked && frames.length > 1) {
      addFrameMeshes(modelGroup, initialFrame, {
        ghost: true,
        partIndex: index,
      });
    }

    renderedFaces += addFrameMeshes(modelGroup, finalFrame, {
      ghost: false,
      partIndex: index,
    });
    partRows.push({ row: part.row, trace: part.trace, metrics: partMetrics });
  }

  if (roomsToggle.checked && activePayload) {
    addRoomFootprints(modelGroup, activePayload, activeParts);
  }
  if (activeAudit) {
    addAuditOverlays(modelGroup, activePayload, activeAudit);
  }

  updateHeader({
    finalFaces,
    renderedFaces,
    slopedFaces,
    fixedParts,
    partRows,
    visiblePartCount: visibleParts.length,
  });
  requestRender();
}

function selectorV2Mode() {
  return indexData?.domain === "selector-v2";
}

function hasAssemblyData(row) {
  return Object.prototype.hasOwnProperty.call(row, "assembly_candidate");
}

function rowEmitted(row) {
  return Boolean(row.emitted_candidate);
}

function rowAssembled(row) {
  return hasAssemblyData(row) ? Boolean(row.assembly_candidate) : rowEmitted(row);
}

function rowAccepted(row) {
  return !selectorV2Mode() || rowAssembled(row);
}

function shouldShowPart(row) {
  return !selectorV2Mode() || !emittedOnlyToggle.checked || rowAccepted(row);
}

function coverageLabel(row) {
  if (!selectorV2Mode()) return null;
  const value = Number(row.coverage_ratio);
  if (!Number.isFinite(value)) return "no coverage";
  return `${Math.round(value * 100)}% coverage`;
}

function addFrameMeshes(group, frame, { ghost, partIndex }) {
  let rendered = 0;
  for (const face of frame.faces || []) {
    if (!Array.isArray(face.corners) || face.corners.length < 3) continue;
    const classification = faceClass(face);
    if (!ghost && slopedToggle.checked && classification !== "sloped") continue;
    const geometry = polygonGeometry(face.corners);
    if (!geometry) continue;
    const material = ghost
      ? ghostMaterial
      : classification === "sloped"
        ? slopedMaterial
        : classification === "flat"
          ? flatMaterial
          : wallMaterial(partIndex);
    const meshMaterial = ghost ? material : material.clone();
    const mesh = new THREE.Mesh(geometry, meshMaterial);
    mesh.userData.faceId = face.id;
    mesh.userData.faceClass = classification;
    mesh.userData.peekable = !ghost;
    group.add(mesh);
    rendered += 1;

    if (edgesToggle.checked && !ghost) {
      const edge = edgeLoop(face.corners);
      if (edge) group.add(edge);
    }
  }
  return rendered;
}

function softenBaseBuilding(group) {
  group.traverse((obj) => {
    if (!obj.isMesh) return;
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    const cloned = materials.map((material) => {
      const copy = material.clone();
      copy.transparent = false;
      copy.opacity = 1;
      copy.depthWrite = true;
      copy.polygonOffset = true;
      copy.polygonOffsetFactor = 1;
      copy.polygonOffsetUnits = 1;
      return copy;
    });
    obj.material = Array.isArray(obj.material) ? cloned : cloned[0];
    obj.userData.peekable = true;
  });
}

function frameMetrics(frame) {
  let slopedFaces = 0;
  for (const face of frame.faces || []) {
    if (faceClass(face) === "sloped") slopedFaces += 1;
  }
  return {
    faces: frame.faces?.length || 0,
    slopedFaces,
    counts: frame.counts,
  };
}

function addRoomFootprints(group, payload, parts) {
  const roomsGroup = new THREE.Group();
  roomsGroup.name = "roomFootprints";
  const envelopeFloorPolygons = envelopeFloorPolygonsFromParts(parts);
  let roomCount = 0;
  for (const room of payload.rooms || []) {
    for (const floor of room.floor || []) {
      const corners = floor.corners || [];
      if (!Array.isArray(corners) || corners.length < 3) continue;
      const y = Math.min(...corners.map((corner) => Number(corner.y ?? corner[1] ?? 0))) + 0.045;
      const points = [...corners, corners[0]].map((corner) => {
        const x = Number(corner.x ?? corner[0]);
        const z = Number(corner.z ?? corner[2]);
        return new THREE.Vector3(x, y, z);
      });
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const polygon = corners.map((corner) => [
        Number(corner.x ?? corner[0]),
        Number(corner.z ?? corner[2]),
      ]);
      const isCovered =
        polygonCoverageRatio(polygon, envelopeFloorPolygons) >= 0.10;
      const line = new THREE.Line(
        geometry,
        isCovered ? coveredRoomMaterial : droppedRoomMaterial,
      );
      line.userData.title = isCovered ? "covered room footprint" : "unbuilt room footprint";
      line.renderOrder = 20;
      roomsGroup.add(line);
      roomCount += 1;
      break;
    }
  }
  group.add(roomsGroup);
  group.userData.roomFootprintCount = roomCount;
}

function envelopeFloorPolygonsFromParts(parts) {
  const polygons = [];
  for (const part of parts || []) {
    const frames = part.trace?.frames || [];
    const finalFrame = frames[frames.length - 1];
    for (const face of finalFrame?.faces || []) {
      const plane = face.plane || {};
      if (Number(plane.b) >= -0.5) continue;
      const polygon = (face.corners || []).map((corner) => [
        Number(corner.x ?? corner[0]),
        Number(corner.z ?? corner[2]),
      ]);
      if (polygon.length >= 3) polygons.push(polygon);
    }
  }
  return polygons;
}

function polygonCoverageRatio(polygon, candidates) {
  if (polygon.length < 3 || !candidates.length) return 0;
  const xs = polygon.map((point) => point[0]);
  const zs = polygon.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  const steps = 10;
  let insideRoom = 0;
  let insideEnvelope = 0;
  for (let ix = 0; ix < steps; ix += 1) {
    const x = minX + ((ix + 0.5) / steps) * (maxX - minX);
    for (let iz = 0; iz < steps; iz += 1) {
      const z = minZ + ((iz + 0.5) / steps) * (maxZ - minZ);
      const point = [x, z];
      if (!pointInPolygon(point, polygon)) continue;
      insideRoom += 1;
      if (candidates.some((candidate) => pointInPolygon(point, candidate))) {
        insideEnvelope += 1;
      }
    }
  }
  if (insideRoom > 0) return insideEnvelope / insideRoom;
  const centroid = polygonCentroid(polygon);
  return candidates.some((candidate) => pointInPolygon(centroid, candidate)) ? 1 : 0;
}

function polygonCentroid(polygon) {
  let x = 0;
  let z = 0;
  for (const point of polygon) {
    x += point[0];
    z += point[1];
  }
  return [x / polygon.length, z / polygon.length];
}

function pointInPolygon(point, polygon) {
  let inside = false;
  const x = point[0];
  const z = point[1];
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0];
    const zi = polygon[i][1];
    const xj = polygon[j][0];
    const zj = polygon[j][1];
    const intersects =
      zi > z !== zj > z &&
      x < ((xj - xi) * (z - zi)) / ((zj - zi) || 1e-12) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function addAuditOverlays(group, payload, audit) {
  if (partPlanesToggle.checked) addPlaneGroupFootprints(group, audit);
  if (topCandidatesToggle.checked) addTopCandidateFootprints(group, audit);
  if (cellsToggle.checked) addPlanCells(group, audit);
  if (roomAuditToggle.checked && payload) addRoomAuditFootprints(group, payload, audit);
}

function addPlaneGroupFootprints(group, audit) {
  const overlay = new THREE.Group();
  overlay.name = "partPlaneGroups";
  for (const plane of audit.plane_groups || []) {
    const polygon = plane.footprint || [];
    if (polygon.length < 3) continue;
    const y = supportOverlayY(plane.label);
    const mesh = xzPolygonMesh(polygon, y, planeMaterial(plane.label, 0.13));
    if (mesh) {
      mesh.userData.title = `${plane.label} ${Math.round((plane.support_ratio || 0) * 100)}%`;
      overlay.add(mesh);
    }
    const edge = xzEdgeLoop(polygon, y + 0.01, planeEdgeMaterial);
    if (edge) overlay.add(edge);
  }
  group.add(overlay);
}

function addTopCandidateFootprints(group, audit) {
  const overlay = new THREE.Group();
  overlay.name = "topCandidates";
  for (const room of audit.rooms || []) {
    for (const candidate of room.best_top_candidates || []) {
      // Candidate polygons are not duplicated per room in audit JSON; use the
      // part-plane layer for geometry and keep this layer as a searchable marker.
      const label = `${room.room_locator_id || room.room_index}: ${candidate.locator_id}`;
      overlay.userData[label] = candidate.overlap_ratio;
    }
  }
  for (const plane of audit.plane_groups || []) {
    const edge = xzEdgeLoop(plane.footprint || [], supportOverlayY(plane.label) + 0.04, candidateEdgeMaterial);
    if (edge) overlay.add(edge);
  }
  group.add(overlay);
}

function addPlanCells(group, audit) {
  const overlay = new THREE.Group();
  overlay.name = "planCells";
  for (const cell of audit.selected_cells || audit.cells || []) {
    const polygon = cell.polygon || [];
    if (polygon.length < 3) continue;
    const label = cell.top_label || "ambiguous-top";
    const y = Number(cell.floor_y || 0) + 0.035;
    const mesh = xzPolygonMesh(polygon, y, cellMaterial(label));
    if (mesh) {
      mesh.userData.title = `${cell.cell_id} ${label}`;
      overlay.add(mesh);
    }
    const edge = xzEdgeLoop(polygon, y + 0.01, cellEdgeMaterial);
    if (edge) overlay.add(edge);
  }
  group.add(overlay);
}

function addRoomAuditFootprints(group, payload, audit) {
  const roomsByIndex = new Map((audit.rooms || []).map((room) => [Number(room.room_index), room]));
  const overlay = new THREE.Group();
  overlay.name = "roomAudit";
  for (const [roomIndex, room] of (payload.rooms || []).entries()) {
    const auditRoom = roomsByIndex.get(roomIndex);
    if (!auditRoom) continue;
    if (!droppedToggle.checked && auditRoom.status === "dropped") continue;
    const material =
      auditRoom.status === "covered"
        ? coveredRoomMaterial
        : auditRoom.status === "partial"
          ? partialRoomMaterial
          : droppedRoomMaterial;
    for (const floor of room.floor || []) {
      const corners = floor.corners || [];
      if (!Array.isArray(corners) || corners.length < 3) continue;
      const y = Math.min(...corners.map((corner) => Number(corner.y ?? corner[1] ?? 0))) + 0.055;
      const polygon = corners.map((corner) => [Number(corner.x ?? corner[0]), Number(corner.z ?? corner[2])]);
      const edge = xzEdgeLoop(polygon, y, material);
      if (edge) {
        edge.userData.title = `${auditRoom.status}: ${auditRoom.reason}`;
        overlay.add(edge);
      }
      break;
    }
  }
  group.add(overlay);
}

function supportOverlayY(label) {
  return label === "flat-ceiling" ? 0.12 : 0.18;
}

function planeMaterial(label, opacity) {
  const key = `plane:${label}:${opacity}`;
  if (!materials.has(key)) {
    materials.set(
      key,
      new THREE.MeshBasicMaterial({
        color: label === "flat-ceiling" ? 0x22c55e : 0xf97316,
        transparent: true,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
  }
  return materials.get(key);
}

function cellMaterial(label) {
  const key = `cell:${label}`;
  if (!materials.has(key)) {
    const color =
      label === "flat-ceiling"
        ? 0x22c55e
        : label === "gable-pair"
          ? 0xf97316
          : label === "single-oblique"
            ? 0xf59e0b
            : 0x14b8a6;
    materials.set(
      key,
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.16,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
  }
  return materials.get(key);
}

function roomCount(payload) {
  return (payload?.rooms || []).filter((room) =>
    (room.floor || []).some((floor) => (floor.corners || []).length >= 3),
  ).length;
}

function faceClass(face) {
  const inclination = faceInclinationDeg(face);
  if (inclination > 5 && inclination < 80) return "sloped";
  if (inclination <= 5) return "flat";
  return "wall";
}

function faceInclinationDeg(face) {
  let normal = null;
  const plane = face.plane || {};
  if ([plane.a, plane.b, plane.c].every((value) => Number.isFinite(Number(value)))) {
    normal = new THREE.Vector3(Number(plane.a), Number(plane.b), Number(plane.c));
  } else {
    normal = newellNormal(face.corners || []);
  }
  if (!normal || normal.lengthSq() <= 1e-12) return 90;
  normal.normalize();
  const verticalNormal = Math.min(1, Math.max(0, Math.abs(normal.y)));
  return (Math.acos(verticalNormal) * 180) / Math.PI;
}

function updateHeader(metrics) {
  currentTitle.textContent = activeGroup.uuid;
  const auditSummary = activeAudit?.summary || null;
  const auditSuffix = auditSummary
    ? ` · ${auditSummary.rooms_ge80}/${auditSummary.rooms_total} rooms >=80%`
    : "";
  const roomSuffix = activePayload ? ` · ${roomCount(activePayload)} room footprints` : "";
  const emittedCount = activeGroup.rows.filter((row) => rowAccepted(row)).length;
  const hiddenSuffix =
    selectorV2Mode() && emittedOnlyToggle.checked
      ? ` · showing ${metrics.visiblePartCount}/${activeGroup.rows.length} assembled domains`
      : "";
  currentMeta.textContent = selectorV2Mode()
    ? `${activeGroup.rows.length} selector-v2 domains · ${activeGroup.assembledCount} assembled · ${activeGroup.emittedCount} emitted · ${Math.round(activeGroup.assemblyCoverage * 100)}% coverage${hiddenSuffix}${roomSuffix}`
    : `${activeGroup.rows.length} trace-fixed envelope parts · ${metrics.fixedParts} parts changed by trace steps${roomSuffix}`;
  pill.textContent = slopedToggle.checked
    ? `${metrics.renderedFaces}/${metrics.finalFaces} faces`
    : `${metrics.finalFaces} faces`;
  openOriginal.href = `./viewer-tiers.html#b=${encodeURIComponent(activeGroup.uuid)}`;
  partCounts.textContent = selectorV2Mode()
    ? `${activeGroup.assembledCount} assembled · ${activeGroup.emittedCount} emitted · ${activeGroup.rows.filter((row) => row.low_coverage).length} low coverage`
    : `${metrics.slopedFaces} sloped faces`;
  status.textContent = selectorV2Mode()
    ? `${metrics.renderedFaces} selected selector faces rendered${auditSuffix}`
    : `${metrics.renderedFaces} fixed final faces rendered${auditSuffix}`;
  partBody.innerHTML = `${auditPanel(activeAudit)}${metrics.partRows
    .map(({ row, trace, metrics: partMetrics }) => {
      const steps = trace.steps?.length || 0;
      const coverage = coverageLabel(row);
      const statusClass = selectorV2Mode()
        ? rowAccepted(row)
          ? " emitted"
          : " rejected"
        : "";
      return `
        <div class="part-line${statusClass}${row.low_coverage ? " low-coverage" : ""}">
          <span class="source" title="${escapeHtml(row.top_source || "")}">
            ${escapeHtml(row.locator_id || `part ${row.part_index}`)} · ${escapeHtml(sourceSummary(row))}
          </span>
          ${coverage ? `<span class="badge${row.low_coverage ? " warn" : ""}">${escapeHtml(coverage)}</span>` : ""}
          ${selectorV2Mode() ? `<span class="badge">${rowAssembled(row) ? "assembled" : rowEmitted(row) ? "emitted alt" : "diagnostic"}</span>` : ""}
          <span class="badge">${partMetrics.slopedFaces} sloped</span>
          <span class="badge">${selectorV2Mode() ? "selection" : `${steps} steps`}</span>
        </div>
      `;
    })
    .join("")}`;
}

function auditPanel(audit) {
  if (!audit?.summary) return "";
  const summary = audit.summary;
  const reasons = Object.entries(summary.reasons || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([reason, count]) => `${escapeHtml(reason)}: ${escapeHtml(count)}`)
    .join(" · ");
  const failedAttempts = (audit.build_attempts || []).filter(
    (attempt) => attempt.result !== "accepted",
  );
  const failures = failedAttempts
    .slice(0, 4)
    .map((attempt) => {
      const rooms = (attempt.room_indices || []).join(",");
      const finalError = attempt.final_candidate?.error || attempt.fallback_candidate?.error || "";
      return `
        <div class="audit-failure">
          <span>${escapeHtml(attempt.reason || attempt.result)} · rooms ${escapeHtml(rooms)}</span>
          <span title="${escapeHtml(finalError)}">${escapeHtml(finalError || attempt.top_label || "")}</span>
        </div>
      `;
    })
    .join("");
  const gableDiagnostics = (audit.build_attempts || [])
    .filter((attempt) => attempt.gable_footprint_coherence)
    .slice(0, 6)
    .map((attempt) => {
      const diagnostic = attempt.gable_footprint_coherence || {};
      const rooms = (attempt.room_indices || []).join(",");
      const parts = [
        `part ${attempt.part_index}`,
        `rooms ${rooms || "none"}`,
        diagnostic.status || "unknown",
      ];
      if (diagnostic.status === "ok") {
        parts.push(`split ${diagnostic.split_region_count}`);
        parts.push(
          `balance ${Number(diagnostic.side_area_balance || 0).toFixed(2)}`,
        );
        parts.push(`frag ${diagnostic.fragmented_side_count || 0}`);
      }
      return `
        <div class="audit-gable">
          <span>${escapeHtml(parts.join(" · "))}</span>
          <span>${escapeHtml(attempt.result || "")}</span>
        </div>
      `;
    })
    .join("");
  return `
    <section class="audit-panel">
      <div class="audit-summary">
        <span class="badge">${escapeHtml(summary.rooms_ge80)}/${escapeHtml(summary.rooms_total)} rooms >=80%</span>
        <span class="badge">${escapeHtml(summary.rooms_ge50)} rooms >=50%</span>
        <span class="badge">${escapeHtml(summary.dropped_rooms)} dropped</span>
      </div>
      ${reasons ? `<div class="audit-reasons">${reasons}</div>` : ""}
      ${failures ? `<div class="audit-failures">${failures}</div>` : ""}
      ${gableDiagnostics ? `<div class="audit-gables">${gableDiagnostics}</div>` : ""}
    </section>
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

function xzPolygonMesh(polygon, y, material) {
  const corners = polygon.map(([x, z]) => [Number(x), Number(y), Number(z)]);
  const geometry = polygonGeometry(corners);
  return geometry ? new THREE.Mesh(geometry, material) : null;
}

function xzEdgeLoop(polygon, y, material) {
  if (!Array.isArray(polygon) || polygon.length < 3) return null;
  const points = [...polygon, polygon[0]].map(
    ([x, z]) => new THREE.Vector3(Number(x), Number(y), Number(z)),
  );
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geometry, material);
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
      if (obj.material) {
        const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const material of materials) material.dispose?.();
      }
    });
  }
  group.clear();
}

function focusModel() {
  const box = new THREE.Box3().setFromObject(modelGroup);
  if (box.isEmpty()) {
    camera.position.set(4, 3, 5);
    controls.target.set(0, 0, 0);
    controls.update();
    requestRender();
    return;
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 1);
  camera.position.copy(center).add(new THREE.Vector3(radius * 1.25, radius * 0.95, radius * 1.4));
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
  const index = visibleGroups.indexOf(activeGroup);
  const next = visibleGroups[(index + delta + visibleGroups.length) % visibleGroups.length];
  void loadBuilding(next);
}

function hashUuid() {
  const params = new URLSearchParams(window.location.hash.slice(1));
  return params.get("b") || params.get("uuid");
}

function setHash(uuid) {
  const next = `#b=${encodeURIComponent(uuid)}`;
  if (window.location.hash !== next) {
    window.history.replaceState(null, "", next);
  }
}

async function init() {
  const response = await fetch(INDEX_URL);
  if (!response.ok) throw new Error(`index fetch failed: ${response.status}`);
  indexData = await response.json();
  groups = buildGroups(indexData.records || []);
  const recordCount = groups.reduce((sum, group) => sum + group.rows.length, 0);
  const builtCount =
    indexData.summary.built_parts ?? indexData.summary.built_rooms ?? indexData.summary.records ?? recordCount;
  const domainLabel =
    indexData.domain === "selector-v2" ? "selector-v2 domains" : "fixed envelope parts";
  sidebarStats.textContent = `${groups.length} buildings · ${builtCount} ${domainLabel}`;
  renderBuildingList();

  const requestedUuid = hashUuid();
  const requested = groups.find((group) => group.uuid === requestedUuid);
  const first = requested || groups[0];
  if (first) await loadBuilding(first);
  else status.textContent = "No trace-fixed buildings found.";
}

search.addEventListener("input", () => {
  renderBuildingList();
});
prevBuildingButton.addEventListener("click", () => moveBuilding(-1));
nextBuildingButton.addEventListener("click", () => moveBuilding(1));
edgesToggle.addEventListener("change", renderBuilding);
ghostToggle.addEventListener("change", renderBuilding);
slopedToggle.addEventListener("change", renderBuilding);
roomsToggle.addEventListener("change", renderBuilding);
baseBuildingToggle.addEventListener("change", renderBuilding);
emittedOnlyToggle.addEventListener("change", renderBuilding);
roomAuditToggle.addEventListener("change", renderBuilding);
cellsToggle.addEventListener("change", renderBuilding);
droppedToggle.addEventListener("change", renderBuilding);
topCandidatesToggle.addEventListener("change", renderBuilding);
partPlanesToggle.addEventListener("change", renderBuilding);
window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(viewport);

resize();
init().catch((err) => {
  console.error(err);
  status.textContent = String(err.message || err);
});
