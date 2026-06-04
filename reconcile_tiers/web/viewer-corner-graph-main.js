import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import cytoscape from "cytoscape";

const DATA_ROOT = "../../pipeline-outputs";
const GRAPH_API = "/room-postprocessing/graph";
const UUID_RE = /^[0-9a-fA-F-]{8,64}$/;
const FALLBACK_UUID = "016980bc-6762-4022-bfbf-17df4112e10c";

const params = new URLSearchParams(window.location.search);
let currentUuid = params.get("uuid") || "";

const buildingList = document.querySelector("#building-list");
const search = document.querySelector("#search");
const sidebarStats = document.querySelector("#sidebar-stats");
const currentTitle = document.querySelector("#current-title");
const currentMeta = document.querySelector("#current-meta");
const openTiers = document.querySelector("#open-tiers");
const status3d = document.querySelector("#status3d");
const graphPanel = document.querySelector("#graphPanel");
const canvas = document.querySelector("#view3d");
const prevBuilding = document.querySelector("#prev-building");
const nextBuilding = document.querySelector("#next-building");

const KIND_COLORS = {
  floor: 0x16a34a,
  wall: 0x3b82f6,
  ceiling: 0xf97316,
  visual_shell: 0xa855f7,
  gable_closure: 0x0f766e,
  knee_wall: 0xca8a04,
};

/** 3D highlight: gold = picked node, cyan = corner-sharing neighbors. */
const HIGHLIGHT_SELECTED = 0xfbbf24;
const HIGHLIGHT_NEIGHBOR = 0x22d3ee;
const EDGE_DIM = 0x1e293b;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a2129);

const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 500);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.addEventListener("change", requestRender);

scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(4, 8, 6);
scene.add(keyLight);

const modelGroup = new THREE.Group();
const edgeGroup = new THREE.Group();
const isolatedEdgeGroup = new THREE.Group();
scene.add(modelGroup);
scene.add(edgeGroup);
scene.add(isolatedEdgeGroup);

const meshById = new Map();
const edgeLinesById = new Map();
const isolatedLinesById = new Map();
const kindById = new Map();
let rows = [];
let graphData = null;
let cy = null;
let selectedId = null;
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
  const viewport = document.querySelector("#viewport3d");
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  if (cy) cy.resize();
  requestRender();
}

function materialForKind(kind, role = "default") {
  const base = KIND_COLORS[kind] ?? 0x64748b;
  if (role === "selected") {
    return new THREE.MeshStandardMaterial({
      color: HIGHLIGHT_SELECTED,
      transparent: true,
      opacity: 0.95,
      side: THREE.DoubleSide,
      emissive: HIGHLIGHT_SELECTED,
      emissiveIntensity: 0.45,
      depthWrite: true,
    });
  }
  if (role === "neighbor") {
    return new THREE.MeshStandardMaterial({
      color: HIGHLIGHT_NEIGHBOR,
      transparent: true,
      opacity: 0.82,
      side: THREE.DoubleSide,
      emissive: HIGHLIGHT_NEIGHBOR,
      emissiveIntensity: 0.28,
      depthWrite: true,
    });
  }
  if (role === "dim") {
    return new THREE.MeshStandardMaterial({
      color: base,
      transparent: true,
      opacity: 0.1,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
  }
  return new THREE.MeshStandardMaterial({
    color: base,
    transparent: true,
    opacity: 0.45,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
}

function neighborsOf(elementId) {
  const neighbors = new Set();
  if (!graphData?.edges) return neighbors;
  for (const edge of graphData.edges) {
    if (edge.source === elementId) neighbors.add(edge.target);
    if (edge.target === elementId) neighbors.add(edge.source);
  }
  return neighbors;
}

function highlightRole(elementId, selectedId, neighbors) {
  if (elementId === selectedId) return "selected";
  if (neighbors.has(elementId)) return "neighbor";
  if (selectedId) return "dim";
  return "default";
}

function trianglesFromCorners(corners) {
  if (corners.length < 3) return null;
  const positions = [];
  for (let i = 1; i < corners.length - 1; i += 1) {
    positions.push(corners[0].x, corners[0].y, corners[0].z);
    positions.push(corners[i].x, corners[i].y, corners[i].z);
    positions.push(corners[i + 1].x, corners[i + 1].y, corners[i + 1].z);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function cornerVec(c) {
  return new THREE.Vector3(Number(c.x), Number(c.y), Number(c.z));
}

function frameToNodes(nodes) {
  const box = new THREE.Box3();
  for (const node of nodes) {
    for (const c of node.corners || []) {
      box.expandByPoint(cornerVec(c));
    }
  }
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const dist = Math.max(size.x, size.y, size.z, 2) * 1.8;
  camera.position.set(center.x + dist * 0.6, center.y + dist * 0.5, center.z + dist * 0.7);
  controls.target.copy(center);
  controls.update();
}

function clearScene() {
  for (const group of [modelGroup, edgeGroup, isolatedEdgeGroup]) {
    while (group.children.length) {
      const child = group.children.pop();
      child.geometry?.dispose?.();
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else {
        child.material?.dispose?.();
      }
    }
  }
  meshById.clear();
  edgeLinesById.clear();
  isolatedLinesById.clear();
  kindById.clear();
}

function build3D(data) {
  clearScene();
  const wallEdgeMaterial = new THREE.LineBasicMaterial({
    color: 0x334155,
    transparent: true,
    opacity: 0.7,
  });
  const isolatedMaterial = new THREE.LineBasicMaterial({
    color: 0xff00ff,
    linewidth: 2,
  });

  for (const node of data.nodes) {
    const corners = (node.corners || []).map(cornerVec);
    const geometry = trianglesFromCorners(corners);
    if (!geometry) continue;
    kindById.set(node.id, node.kind);
    const mesh = new THREE.Mesh(geometry, materialForKind(node.kind));
    mesh.userData.elementId = node.id;
    modelGroup.add(mesh);
    meshById.set(node.id, mesh);

    if (node.kind === "wall") {
      const positions = [];
      for (let i = 0; i < corners.length; i += 1) {
        const a = corners[i];
        const b = corners[(i + 1) % corners.length];
        positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
      }
      const edgeGeo = new THREE.BufferGeometry();
      edgeGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      const lines = new THREE.LineSegments(edgeGeo, wallEdgeMaterial);
      lines.userData.elementId = node.id;
      edgeGroup.add(lines);
      edgeLinesById.set(node.id, lines);
    }
  }

  for (const seg of data.wall_edge_segments || []) {
    if (!seg.isolated) continue;
    const start = seg.start;
    const end = seg.end;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        [start.x, start.y, start.z, end.x, end.y, end.z],
        3,
      ),
    );
    const line = new THREE.Line(geometry, isolatedMaterial);
    line.userData.elementId = seg.element_id;
    isolatedEdgeGroup.add(line);
    const list = isolatedLinesById.get(seg.element_id) || [];
    list.push(line);
    isolatedLinesById.set(seg.element_id, list);
  }

  frameToNodes(data.nodes);
  requestRender();
}

function nodeLabel(node) {
  const room = node.room_index != null ? ` r${node.room_index}` : "";
  return `${node.kind}${room}`;
}

function buildGraph(data) {
  if (cy) {
    cy.destroy();
    cy = null;
  }
  const elements = [];
  for (const node of data.nodes) {
    elements.push({
      data: {
        id: node.id,
        label: nodeLabel(node),
        kind: node.kind,
        degree: node.degree ?? 0,
      },
    });
  }
  for (const edge of data.edges) {
    elements.push({
      data: {
        id: `${edge.source}--${edge.target}`,
        source: edge.source,
        target: edge.target,
      },
    });
  }

  cy = cytoscape({
    container: graphPanel,
    elements,
    style: [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "font-size": 9,
          "text-valign": "center",
          "text-halign": "center",
          width: 28,
          height: 28,
          "background-color": "#94a3b8",
          color: "#0f1419",
          "text-wrap": "wrap",
          "text-max-width": 80,
        },
      },
      {
        selector: "node[kind = 'floor']",
        style: { "background-color": "#16a34a" },
      },
      {
        selector: "node[kind = 'wall']",
        style: { "background-color": "#3b82f6" },
      },
      {
        selector: "node[kind = 'ceiling']",
        style: { "background-color": "#f97316" },
      },
      {
        selector: "node[degree = 0]",
        style: {
          "border-width": 3,
          "border-color": "#e11d48",
        },
      },
      {
        selector: "node.neighbor",
        style: {
          "border-width": 3,
          "border-color": "#22d3ee",
          "background-color": "#67e8f9",
        },
      },
      {
        selector: "node:selected",
        style: {
          "border-width": 4,
          "border-color": "#fbbf24",
          "background-color": "#fde047",
        },
      },
      {
        selector: "edge",
        style: {
          width: 2,
          "line-color": "#64748b",
          "curve-style": "bezier",
        },
      },
    ],
    layout: { name: "cose", animate: false, padding: 24 },
  });

  cy.on("tap", "node", (evt) => {
    selectElement(evt.target.id());
  });
  cy.on("tap", (evt) => {
    if (evt.target === cy) clearSelection();
  });
}

function apply3DHighlight(elementId) {
  const neighbors = neighborsOf(elementId);
  for (const [id, mesh] of meshById) {
    const role = highlightRole(id, elementId, neighbors);
    const mat = materialForKind(kindById.get(id) ?? "wall", role);
    mesh.material.dispose();
    mesh.material = mat;
  }
  for (const [id, lines] of edgeLinesById) {
    const role = highlightRole(id, elementId, neighbors);
    if (role === "selected") {
      lines.material.opacity = 1;
      lines.material.color.setHex(HIGHLIGHT_SELECTED);
    } else if (role === "neighbor") {
      lines.material.opacity = 0.9;
      lines.material.color.setHex(HIGHLIGHT_NEIGHBOR);
    } else {
      lines.material.opacity = role === "dim" ? 0.15 : 0.35;
      lines.material.color.setHex(EDGE_DIM);
    }
  }
  for (const [id, lines] of isolatedLinesById) {
    const role = highlightRole(id, elementId, neighbors);
    for (const line of lines) {
      if (role === "selected") {
        line.material.opacity = 1;
        line.material.color.setHex(0xff00ff);
      } else if (role === "neighbor") {
        line.material.opacity = 0.85;
        line.material.color.setHex(0xff66ff);
      } else {
        line.material.opacity = role === "dim" ? 0.12 : 0.4;
        line.material.color.setHex(0xff00ff);
      }
    }
  }
}

function selectElement(elementId) {
  selectedId = elementId;
  const neighbors = neighborsOf(elementId);
  apply3DHighlight(elementId);
  if (cy) {
    cy.nodes().removeClass("neighbor");
    cy.nodes().unselect();
    const node = cy.getElementById(elementId);
    if (node.length) node.select();
    for (const nid of neighbors) {
      cy.getElementById(nid).addClass("neighbor");
    }
  }
  const n = neighbors.size;
  status3d.textContent = n
    ? `${elementId} · ${n} connected`
    : elementId;
  requestRender();
}

function clearSelection() {
  selectedId = null;
  for (const [id, mesh] of meshById) {
    const mat = materialForKind(kindById.get(id) ?? "wall", "default");
    mesh.material.dispose();
    mesh.material = mat;
  }
  for (const [, lines] of edgeLinesById) {
    lines.material.opacity = 0.7;
    lines.material.color.setHex(0x334155);
  }
  for (const [, lines] of isolatedLinesById) {
    for (const line of lines) {
      line.material.opacity = 0.5;
      line.material.color.setHex(0xff00ff);
    }
  }
  if (cy) {
    cy.nodes().removeClass("neighbor");
    cy.nodes().unselect();
  }
  status3d.textContent = "Click a graph node to highlight";
  requestRender();
}

async function fetchGraph(uuid) {
  const url = `${GRAPH_API}?uuid=${encodeURIComponent(uuid)}&corner_tol=0.05`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Graph API ${response.status} for ${uuid}`);
  }
  return response.json();
}

function renderBuildingList() {
  buildingList.innerHTML = "";
  const query = (search.value || "").trim().toLowerCase();
  const filtered = rows.filter((r) => !query || r.uuid.toLowerCase().includes(query));
  for (const row of filtered) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `building-row${row.uuid === currentUuid ? " active" : ""}`;
    btn.textContent = row.uuid.slice(0, 8);
    btn.title = row.uuid;
    btn.addEventListener("click", () => loadBuilding(row.uuid));
    buildingList.appendChild(btn);
  }
  sidebarStats.textContent = `${filtered.length} / ${rows.length} buildings`;
}

async function hasTierPayload(uuid) {
  try {
    const response = await fetch(`${DATA_ROOT}/${uuid}/`, { cache: "no-store" });
    if (!response.ok) return false;
    const doc = new DOMParser().parseFromString(await response.text(), "text/html");
    return [...doc.querySelectorAll("a")]
      .some((link) => decodeURIComponent(link.getAttribute("href") || "") === "tier_payload.json");
  } catch (_err) {
    return false;
  }
}

async function discoverRows() {
  try {
    const response = await fetch(`${DATA_ROOT}/`, { cache: "no-store" });
    if (!response.ok) return [];
    const doc = new DOMParser().parseFromString(await response.text(), "text/html");
    const uuids = [...doc.querySelectorAll("a")]
      .map((link) => decodeURIComponent(link.getAttribute("href") || "").replace(/\/$/, ""))
      .filter((href) => UUID_RE.test(href));
    const candidates = await Promise.all(
      uuids.map(async (uuid) => (await hasTierPayload(uuid) ? { uuid } : null)),
    );
    return candidates.filter(Boolean);
  } catch (err) {
    console.debug("building discovery failed", err);
    return [];
  }
}

async function loadRows() {
  const response = await fetch(`${DATA_ROOT}/tier_index.json`, { cache: "no-store" });
  const index = response.ok ? await response.json() : { buildings: [] };
  const indexed = (index.buildings || [])
    .map((b) => ({ uuid: b.uuid || b }))
    .filter((r) => UUID_RE.test(r.uuid));
  if (indexed.length > 1) return indexed;
  const discovered = await discoverRows();
  return discovered.length > indexed.length ? discovered : indexed;
}

function updateUrl(uuid) {
  const url = new URL(window.location.href);
  url.searchParams.set("uuid", uuid);
  window.history.replaceState({}, "", url);
}

async function loadBuilding(uuid) {
  currentUuid = uuid;
  updateUrl(uuid);
  renderBuildingList();
  currentTitle.textContent = uuid.slice(0, 8);
  currentMeta.textContent = "Loading corner graph…";
  openTiers.href = `/reconcile_tiers/web/viewer-tiers.html#${encodeURIComponent(uuid)}`;
  status3d.textContent = "Loading…";

  try {
    graphData = await fetchGraph(uuid);
    currentMeta.textContent =
      `${graphData.element_count} elements · ${graphData.edges.length} edges · tol ${graphData.corner_tol} m`;
    build3D(graphData);
    buildGraph(graphData);
    status3d.textContent = "Click a graph node to highlight";
  } catch (err) {
    currentMeta.textContent = String(err.message || err);
    status3d.textContent = "Failed to load";
    console.error(err);
  }
}

function stepBuilding(delta) {
  if (!rows.length || !currentUuid) return;
  const idx = rows.findIndex((r) => r.uuid === currentUuid);
  const next = rows[(idx + delta + rows.length) % rows.length];
  loadBuilding(next.uuid);
}

search.addEventListener("input", renderBuildingList);
search.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  const value = search.value.trim();
  if (UUID_RE.test(value)) loadBuilding(value);
});

prevBuilding.addEventListener("click", () => stepBuilding(-1));
nextBuilding.addEventListener("click", () => stepBuilding(1));

window.addEventListener("resize", resize);

async function init() {
  rows = await loadRows();
  if (!currentUuid && rows.length) {
    currentUuid = rows[0].uuid;
  }
  if (!currentUuid && FALLBACK_UUID) {
    currentUuid = FALLBACK_UUID;
  }
  renderBuildingList();
  resize();
  if (currentUuid) {
    await loadBuilding(currentUuid);
  } else {
    currentTitle.textContent = "No buildings found";
    status3d.textContent = "Add tier_payload under pipeline-outputs/";
  }
}

init();
