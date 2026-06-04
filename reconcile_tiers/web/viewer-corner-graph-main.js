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
const graphModeInputs = document.querySelectorAll('input[name="graph-mode"]');
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

/** 3D highlight: gold = picked node, cyan = junction/approx segment group. */
const HIGHLIGHT_SELECTED = 0xf59e0b;
const HIGHLIGHT_CONNECTED = 0x0ea5e9;
const EDGE_DEFAULT = 0x475569;
const EDGE_DIM = 0x94a3b8;
const SEGMENT_RADIUS = 0.02;
const SEGMENT_RADIUS_CONNECTED = 0.05;
const SEGMENT_RADIUS_SELECTED = 0.07;
const SEGMENT_UP = new THREE.Vector3(0, 1, 0);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 500);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.addEventListener("change", requestRender);

scene.add(new THREE.HemisphereLight(0xffffff, 0xe2e8f0, 1.2));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
keyLight.position.set(4, 8, 6);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xf8fafc, 0.6);
fillLight.position.set(-5, 2, -4);
scene.add(fillLight);

const modelGroup = new THREE.Group();
const edgeGroup = new THREE.Group();
const isolatedEdgeGroup = new THREE.Group();
const roomFloorGroup = new THREE.Group();
scene.add(modelGroup);
scene.add(edgeGroup);
scene.add(isolatedEdgeGroup);
scene.add(roomFloorGroup);

const meshById = new Map();
const edgeLinesById = new Map();
const segmentLineById = new Map();
const isolatedLinesById = new Map();
const kindById = new Map();
const groupIdToSegmentIds = new Map();
const roomIdToWallIds = new Map();
const roomIdToSegmentIds = new Map();
let rows = [];
let graphData = null;
let cy = null;
let selectedId = null;
let graphMode = "all";
let renderQueued = false;

function activeGraph() {
  if (!graphData) return { nodes: [], edges: [] };
  if (graphMode === "rooms" && graphData.segment_room_graph) {
    return graphData.segment_room_graph;
  }
  if (graphMode === "segments" && graphData.wall_segment_graph) {
    return graphData.wall_segment_graph;
  }
  if (graphMode === "walls" && graphData.wall_graph) {
    return graphData.wall_graph;
  }
  return { nodes: graphData.nodes || [], edges: graphData.edges || [] };
}

function rebuildSegmentMaps() {
  groupIdToSegmentIds.clear();
  for (const node of graphData?.wall_segment_graph?.nodes || []) {
    groupIdToSegmentIds.set(node.id, node.segment_ids || []);
  }
}

function rebuildRoomMaps() {
  roomIdToWallIds.clear();
  roomIdToSegmentIds.clear();
  for (const node of graphData?.segment_room_graph?.nodes || []) {
    roomIdToWallIds.set(node.id, node.wall_ids || []);
    roomIdToSegmentIds.set(node.id, node.segment_ids || []);
  }
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
  if (role === "connected") {
    return new THREE.MeshStandardMaterial({
      color: HIGHLIGHT_CONNECTED,
      transparent: true,
      opacity: 0.88,
      side: THREE.DoubleSide,
      emissive: HIGHLIGHT_CONNECTED,
      emissiveIntensity: 0.22,
      depthWrite: true,
    });
  }
  if (role === "dim") {
    return new THREE.MeshStandardMaterial({
      color: base,
      transparent: true,
      opacity: 0.42,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
  }
  return new THREE.MeshStandardMaterial({
    color: base,
    transparent: true,
    opacity: 0.72,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
}

function _bfsComponent(elementId, edges, includeEdge) {
  const adj = new Map();
  for (const edge of edges) {
    if (!includeEdge(edge)) continue;
    if (!adj.has(edge.source)) adj.set(edge.source, []);
    adj.get(edge.source).push(edge.target);
    if (!adj.has(edge.target)) adj.set(edge.target, []);
    adj.get(edge.target).push(edge.source);
  }
  const component = new Set([elementId]);
  const queue = [elementId];
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    for (const next of adj.get(cur) || []) {
      if (!component.has(next)) {
        component.add(next);
        queue.push(next);
      }
    }
  }
  return component;
}

function connectedComponentOf(elementId) {
  const { edges } = activeGraph();
  return _bfsComponent(elementId, edges, () => true);
}

function highlightRole(elementId, selectedId, connected) {
  if (elementId === selectedId) return "selected";
  if (connected.has(elementId)) return "connected";
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
  segmentLineById.clear();
  isolatedLinesById.clear();
  kindById.clear();
}

function createSegmentCylinderMesh(start, end, radius, color, opacity) {
  const a = cornerVec(start);
  const b = cornerVec(end);
  const dir = b.clone().sub(a);
  const length = dir.length();
  if (length < 1e-6) return null;
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 8);
  const material = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthTest: true,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(SEGMENT_UP, dir.normalize());
  mesh.userData.segmentLength = length;
  mesh.userData.baseRadius = radius;
  return mesh;
}

function setSegmentCylinderRadius(mesh, radius) {
  const length = mesh.userData.segmentLength;
  if (!length) return;
  mesh.geometry.dispose();
  mesh.geometry = new THREE.CylinderGeometry(radius, radius, length, 8);
}

function buildVerticalSegmentLines(segmentGraph) {
  for (const seg of segmentGraph.segments || []) {
    const mesh = createSegmentCylinderMesh(
      seg.start,
      seg.end,
      SEGMENT_RADIUS,
      EDGE_DEFAULT,
      0.9,
    );
    if (!mesh) continue;
    mesh.userData.segmentId = seg.id;
    mesh.userData.wallId = seg.wall_id;
    edgeGroup.add(mesh);
    segmentLineById.set(seg.id, mesh);
  }
}

function build3D(data) {
  clearScene();
  const wallEdgeMaterial = new THREE.LineBasicMaterial({
    color: EDGE_DEFAULT,
    transparent: true,
    opacity: 0.85,
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

  if (graphData?.wall_segment_graph) {
    buildVerticalSegmentLines(graphData.wall_segment_graph);
  }

  frameToNodes(data.nodes);
  requestRender();
}

function nodeLabel(node) {
  if (graphMode === "rooms") {
    const w = node.wall_ids?.length ?? 0;
    const area = node.area_m2 != null ? `${node.area_m2.toFixed(0)}m²` : "";
    const short = node.id.split("::").pop();
    return area ? `${short} · ${w}w · ${area}` : `${short} · ${w}w`;
  }
  if (graphMode === "segments") {
    const n = node.segment_count ?? node.segment_ids?.length ?? 0;
    const w = node.wall_ids?.length ?? 0;
    return node.orphan ? `orphan · ${n} seg` : `${n} seg · ${w}w`;
  }
  if (graphMode === "walls") {
    const room = node.room_index != null ? `r${node.room_index}` : "wall";
    const short = node.id.includes("::") ? node.id.split("::").pop() : node.id;
    return `${room} ${short}`;
  }
  const room = node.room_index != null ? ` r${node.room_index}` : "";
  return `${node.kind}${room}`;
}

function cytoscapeStyles() {
  const styles = [
    {
      selector: "node",
      style: {
        label: "data(label)",
        "font-size": 9,
        "text-valign": "center",
        "text-halign": "center",
        width: graphMode === "rooms" ? 40 : graphMode === "segments" ? 32 : graphMode === "walls" ? 36 : 28,
        height: graphMode === "rooms" ? 40 : graphMode === "segments" ? 32 : graphMode === "walls" ? 36 : 28,
        "background-color": graphMode === "rooms" ? "#16a34a" : "#3b82f6",
        color: "#0f1419",
        "text-wrap": "wrap",
        "text-max-width": 100,
      },
    },
    {
      selector: "node[degree = 0]",
      style: {
        "border-width": 3,
        "border-color": "#e11d48",
      },
    },
      {
        selector: "node.connected",
        style: {
          "border-width": 3,
          "border-color": "#0ea5e9",
          "background-color": "#7dd3fc",
          opacity: 1,
        },
      },
      {
        selector: "node.faded",
        style: {
          opacity: 0.45,
          "background-color": "#cbd5e1",
          "border-width": 1,
          "border-color": "#94a3b8",
        },
      },
      {
        selector: "node:selected",
        style: {
          "border-width": 4,
          "border-color": "#f59e0b",
          "background-color": "#fde047",
          opacity: 1,
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
  ];
  if (graphMode === "segments") {
    styles.splice(1, 0, {
      selector: "node[orphan = 1]",
      style: {
        "background-color": "#dc2626",
        "border-color": "#991b1b",
        "border-width": 3,
        color: "#ffffff",
      },
    });
  }
  if (graphMode === "all") {
    styles.splice(1, 0,
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
        selector: "node[kind = 'visual_shell']",
        style: { "background-color": "#a855f7" },
      },
      {
        selector: "node[kind = 'gable_closure']",
        style: { "background-color": "#0f766e" },
      },
      {
        selector: "node[kind = 'knee_wall']",
        style: { "background-color": "#ca8a04" },
      },
    );
  }
  return styles;
}

function rebuildGraph() {
  if (!graphData) return;
  const { nodes, edges } = activeGraph();
  if (cy) {
    cy.destroy();
    cy = null;
  }
  const elements = [];
  for (const node of nodes) {
    elements.push({
      data: {
        id: node.id,
        label: nodeLabel(node),
        kind: node.kind,
        wall_id: node.wall_id,
        degree: node.degree ?? 0,
        orphan: node.orphan ? 1 : 0,
      },
    });
  }
  for (const edge of edges) {
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
    style: cytoscapeStyles(),
    layout: { name: "cose", animate: false, padding: 24 },
  });

  cy.on("tap", "node", (evt) => {
    selectElement(evt.target.id());
  });
  cy.on("tap", (evt) => {
    if (evt.target === cy) clearSelection();
  });

  if (selectedId && cy.getElementById(selectedId).length) {
    selectElement(selectedId);
  } else {
    clearSelection();
  }
}

function setGraphMode(mode) {
  if (mode !== "all" && mode !== "walls" && mode !== "segments" && mode !== "rooms") return;
  graphMode = mode;
  clearSelection();
  rebuildGraph();
  updateGraphMeta();
}

function updateGraphMeta() {
  if (!graphData || !currentUuid) return;
  const allN = graphData.nodes?.length ?? 0;
  const allE = graphData.edges?.length ?? 0;
  const wallN = graphData.wall_graph?.nodes?.length ?? 0;
  const wallE = graphData.wall_graph?.edges?.length ?? 0;
  const segCount = graphData.wall_segment_graph?.segments?.length ?? 0;
  const grpN = graphData.wall_segment_graph?.nodes?.length ?? 0;
  const segE = graphData.wall_segment_graph?.edges?.length ?? 0;
  const roomN = graphData.segment_room_graph?.nodes?.length ?? 0;
  const roomE = graphData.segment_room_graph?.edges?.length ?? 0;
  const orphanN = graphMode === "segments"
    ? (graphData.wall_segment_graph?.nodes || []).filter((n) => n.orphan).length
    : 0;
  const view = graphMode === "rooms"
    ? `${roomN} rooms · ${roomE} adjacency edges`
    : graphMode === "segments"
      ? `${segCount} segments · ${grpN} groups · ${orphanN} orphan · ${segE} edges`
      : graphMode === "walls"
        ? `walls ${wallN} nodes · ${wallE} edges`
        : `all ${allN} nodes · ${allE} edges`;
  const adj = graphData.adjacency_tol ?? 0.5;
  currentMeta.textContent =
    `${view} · corner ${graphData.corner_tol} m · adjacency ${adj} m`;
}

function wallMatchesSet(meshId, wallIds) {
  if (wallIds.has(meshId)) return true;
  for (const wid of wallIds) {
    if (meshId.startsWith(`${wid}::split::`)) return true;
  }
  return false;
}

function apply3DWallHighlightSets(selectedWallIds, connectedWallIds) {
  const hasSelection = selectedWallIds && selectedWallIds.size > 0;
  const connected = connectedWallIds || new Set();
  for (const [id, mesh] of meshById) {
    let role = "default";
    if (hasSelection && wallMatchesSet(id, selectedWallIds)) role = "selected";
    else if (wallMatchesSet(id, connected)) role = "connected";
    else if (hasSelection) role = "dim";
    const mat = materialForKind(kindById.get(id) ?? "wall", role);
    mesh.material.dispose();
    mesh.material = mat;
  }
  for (const [id, lines] of edgeLinesById) {
    let role = "default";
    if (hasSelection && wallMatchesSet(id, selectedWallIds)) role = "selected";
    else if (wallMatchesSet(id, connected)) role = "connected";
    else if (hasSelection) role = "dim";
    if (role === "selected") {
      lines.material.opacity = 1;
      lines.material.color.setHex(HIGHLIGHT_SELECTED);
    } else if (role === "connected") {
      lines.material.opacity = 0.95;
      lines.material.color.setHex(HIGHLIGHT_CONNECTED);
    } else {
      lines.material.opacity = role === "dim" ? 0.5 : 0.85;
      lines.material.color.setHex(role === "dim" ? EDGE_DIM : EDGE_DEFAULT);
    }
  }
  for (const [id, lines] of isolatedLinesById) {
    let role = "default";
    if (hasSelection && wallMatchesSet(id, selectedWallIds)) role = "selected";
    else if (wallMatchesSet(id, connected)) role = "connected";
    else if (hasSelection) role = "dim";
    for (const line of lines) {
      if (role === "selected") {
        line.material.opacity = 1;
        line.material.color.setHex(0xff00ff);
      } else if (role === "connected") {
        line.material.opacity = 0.9;
        line.material.color.setHex(0xff66ff);
      } else {
        line.material.opacity = role === "dim" ? 0.45 : 0.65;
        line.material.color.setHex(0xff00ff);
      }
    }
  }
}

function apply3DWallHighlight(selectedWallId, connectedWallIds) {
  const selected = selectedWallId ? new Set([selectedWallId]) : new Set();
  apply3DWallHighlightSets(selected, connectedWallIds);
}

function clearRoomFloorHighlight() {
  while (roomFloorGroup.children.length) {
    const child = roomFloorGroup.children.pop();
    child.geometry?.dispose?.();
    child.material?.dispose?.();
  }
}

function pointInPolygonXZ(px, pz, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const xi = poly[i].x;
    const zi = poly[i].z;
    const xj = poly[j].x;
    const zj = poly[j].z;
    const intersect =
      zi > pz !== zj > pz && px < ((xj - xi) * (pz - zi)) / (zj - zi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function roomFloorY(room) {
  const segmentsById = new Map(
    (graphData.wall_segment_graph?.segments || []).map((s) => [s.id, s]),
  );
  let floorY = Infinity;
  for (const segId of room.segment_ids || []) {
    const seg = segmentsById.get(segId);
    if (!seg) continue;
    floorY = Math.min(floorY, seg.start.y, seg.end.y);
  }
  if (!Number.isFinite(floorY)) {
    for (const node of graphData.nodes || []) {
      if (node.kind !== "floor" || node.story !== room.story) continue;
      for (const c of node.corners || []) {
        floorY = Math.min(floorY, c.y);
      }
    }
  }
  return Number.isFinite(floorY) ? floorY : 0;
}

function addRoomFloorFill(poly, floorY) {
  if (!poly || poly.length < 3) return;
  const positions = [];
  for (let i = 1; i < poly.length - 1; i += 1) {
    positions.push(poly[0].x, floorY + 0.02, poly[0].z);
    positions.push(poly[i].x, floorY + 0.02, poly[i].z);
    positions.push(poly[i + 1].x, floorY + 0.02, poly[i + 1].z);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      color: KIND_COLORS.floor,
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  roomFloorGroup.add(mesh);
}

function addRoomFloorOutline(poly, floorY) {
  if (!poly || poly.length < 3) return;
  const positions = [];
  for (const p of poly) {
    positions.push(p.x, floorY + 0.025, p.z);
  }
  positions.push(poly[0].x, floorY + 0.025, poly[0].z);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  const line = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({
      color: HIGHLIGHT_SELECTED,
      transparent: true,
      opacity: 0.9,
    }),
  );
  roomFloorGroup.add(line);
}

function apply3DRoomFloorMeshes(room, fillPoly) {
  if (!room || !fillPoly?.length) return;
  for (const [id, mesh] of meshById) {
    if (kindById.get(id) !== "floor") continue;
    const node = graphData.nodes?.find((n) => n.id === id);
    if (!node || node.story !== room.story) continue;
    const corners = node.corners || [];
    if (!corners.length) continue;
    let cx = 0;
    let cz = 0;
    for (const c of corners) {
      cx += c.x;
      cz += c.z;
    }
    cx /= corners.length;
    cz /= corners.length;
    if (!pointInPolygonXZ(cx, cz, fillPoly)) continue;
    mesh.material.dispose();
    mesh.material = materialForKind("floor", "connected");
  }
}

function updateRoomFloorHighlight(roomId) {
  clearRoomFloorHighlight();
  if (!roomId || !graphData?.segment_room_graph) return;
  const room = graphData.segment_room_graph.nodes?.find((n) => n.id === roomId);
  const fillPoly =
    room?.floor_polygon_xz?.length >= 3 ? room.floor_polygon_xz : room?.polygon_xz;
  if (!fillPoly || fillPoly.length < 3) return;

  const floorY = roomFloorY(room);
  addRoomFloorFill(fillPoly, floorY);
  addRoomFloorOutline(fillPoly, floorY);
  apply3DRoomFloorMeshes(room, fillPoly);
}

/** Segment mode: keep building context muted; only vertical segment lines carry selection. */
function apply3DBuildingMuted(muted) {
  for (const [id, mesh] of meshById) {
    const mat = materialForKind(kindById.get(id) ?? "wall", muted ? "dim" : "default");
    mesh.material.dispose();
    mesh.material = mat;
  }
  for (const [, lines] of edgeLinesById) {
    lines.material.opacity = muted ? 0.35 : 0.85;
    lines.material.color.setHex(muted ? EDGE_DIM : EDGE_DEFAULT);
  }
  for (const [, lines] of isolatedLinesById) {
    for (const line of lines) {
      line.material.opacity = muted ? 0.35 : 0.65;
      line.material.color.setHex(0xff00ff);
    }
  }
}

function apply3DSegmentLineHighlight(selectedSegmentIds, connectedSegmentIds) {
  const hasSelection = selectedSegmentIds && selectedSegmentIds.size > 0;
  const connected = connectedSegmentIds || new Set();
  for (const [segId, mesh] of segmentLineById) {
    let role = "default";
    if (hasSelection && selectedSegmentIds.has(segId)) role = "selected";
    else if (connected.has(segId)) role = "connected";
    else if (hasSelection) role = "dim";
    let radius = SEGMENT_RADIUS;
    let color = EDGE_DEFAULT;
    let opacity = 0.9;
    if (role === "selected") {
      radius = SEGMENT_RADIUS_SELECTED;
      color = HIGHLIGHT_SELECTED;
      opacity = 1;
    } else if (role === "connected") {
      radius = SEGMENT_RADIUS_CONNECTED;
      color = HIGHLIGHT_CONNECTED;
      opacity = 0.98;
    } else if (role === "dim") {
      radius = SEGMENT_RADIUS * 0.65;
      color = EDGE_DIM;
      opacity = 0.3;
    }
    setSegmentCylinderRadius(mesh, radius);
    mesh.material.opacity = opacity;
    mesh.material.color.setHex(color);
  }
}

function apply3DHighlight(elementId, connected) {
  if (graphMode === "rooms") {
    apply3DBuildingMuted(!!elementId);
    const selectedWalls = new Set(roomIdToWallIds.get(elementId) || []);
    const connectedWalls = new Set();
    const selectedSegs = new Set(roomIdToSegmentIds.get(elementId) || []);
    const connectedSegs = new Set();
    for (const rid of connected) {
      if (rid === elementId) continue;
      for (const wid of roomIdToWallIds.get(rid) || []) connectedWalls.add(wid);
      for (const sid of roomIdToSegmentIds.get(rid) || []) connectedSegs.add(sid);
    }
    apply3DWallHighlightSets(selectedWalls, connectedWalls);
    apply3DSegmentLineHighlight(selectedSegs, connectedSegs);
    updateRoomFloorHighlight(elementId);
    return;
  }
  clearRoomFloorHighlight();
  if (graphMode === "segments") {
    apply3DBuildingMuted(!!elementId);
    const selectedSegs = new Set(groupIdToSegmentIds.get(elementId) || []);
    const connectedSegs = new Set();
    for (const gid of connected) {
      if (gid === elementId) continue;
      for (const sid of groupIdToSegmentIds.get(gid) || []) {
        connectedSegs.add(sid);
      }
    }
    apply3DSegmentLineHighlight(selectedSegs, connectedSegs);
    return;
  }
  apply3DBuildingMuted(false);
  apply3DWallHighlight(elementId, connected);
  apply3DSegmentLineHighlight(null, new Set());
}

function selectElement(elementId) {
  selectedId = elementId;
  const connected = connectedComponentOf(elementId);
  apply3DHighlight(elementId, connected);
  if (cy) {
    cy.nodes().removeClass("connected faded");
    cy.nodes().unselect();
    for (const node of cy.nodes()) {
      const id = node.id();
      if (id === elementId) {
        node.select();
      } else if (connected.has(id)) {
        node.addClass("connected");
      } else {
        node.addClass("faded");
      }
    }
  }
  const n = connected.size;
  if (graphMode === "rooms") {
    const room = graphData?.segment_room_graph?.nodes?.find((node) => node.id === elementId);
    const w = room?.wall_ids?.length ?? 0;
    const area = room?.area_m2 != null ? `${room.area_m2.toFixed(1)} m²` : "";
    status3d.textContent = n > 1
      ? `room · ${w} walls · ${area} · ${n} rooms in component`
      : `room · ${w} walls · ${area}`;
  } else if (graphMode === "segments") {
    const grp = graphData?.wall_segment_graph?.nodes?.find((node) => node.id === elementId);
    const segCount = grp?.segment_count ?? grp?.segment_ids?.length ?? 0;
    status3d.textContent = n > 1
      ? `approx group · ${segCount} segments · ${n} groups in component`
      : `approx group · ${segCount} segments`;
  } else {
    status3d.textContent = n > 1
      ? `${elementId} · ${n} in component`
      : elementId;
  }
  requestRender();
}

function clearSelection() {
  selectedId = null;
  apply3DBuildingMuted(false);
  apply3DSegmentLineHighlight(null, new Set());
  clearRoomFloorHighlight();
  if (graphMode !== "rooms") {
    apply3DWallHighlightSets(new Set(), new Set());
  }
  if (cy) {
    cy.nodes().removeClass("connected faded");
    cy.nodes().unselect();
  }
  status3d.textContent = "Click a graph node to highlight";
  requestRender();
}

async function fetchGraph(uuid) {
  const url =
    `${GRAPH_API}?uuid=${encodeURIComponent(uuid)}&corner_tol=0.05&adjacency_tol=0.5`;
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
    rebuildSegmentMaps();
    rebuildRoomMaps();
    build3D(graphData);
    rebuildGraph();
    updateGraphMeta();
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

for (const input of graphModeInputs) {
  input.addEventListener("change", () => {
    if (input.checked) setGraphMode(input.value);
  });
}

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
