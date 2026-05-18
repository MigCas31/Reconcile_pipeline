import * as THREE from "three";
import { mergeGeometries, mergeVertices, toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

import { newellNormal, signedArea2 } from "./geometry.js";
import { attachLocator } from "./locator.js";
import { gapMaterial, HEATING_COLORS, MATERIAL_PALETTE } from "./material-palette.js";
import { CALM_MATERIAL_PALETTE, calmGapMaterial } from "./calm/material-palette-calm.js";
import { RENDER_TUNING } from "./render-tuning.js";
import { EXTERIOR_STORY_KEY, isStoryVisible } from "./story-visibility.js";

const SPLIT_LEVEL_DY_M = 2.0;
const HEATING_TINT_FACTORS = { floor: 0.6, structure: 0.35, structureFill: 0.5 };

const OUTLINE_NAMES = new Set(["roof", "roofAttic", "doorLeaf", "dormer", "gableClosure"]);
const CREASE_ANGLE_RAD = THREE.MathUtils.degToRad(RENDER_TUNING.creaseAngleDeg);
const EDGE_THRESHOLD_DEG = RENDER_TUNING.edgeThresholdDeg;
const WINDOW_FRAME_THICKNESS_M = 0.05;
const WINDOW_FRAME_DEPTH_M = 0.07;
const DOOR_FRAME_THICKNESS_M = 0.05;
const DOOR_FRAME_DEPTH_M = 0.07;
const DOOR_LEAF_DEPTH_M = 0.04;
const CEILING_GRID_STEP_M = 0.2;
const CEILING_GRID_OFFSET_M = 0.003;

const OUTLINE_MATERIAL = new THREE.LineBasicMaterial({
  color: 0x2a2a2a,
  transparent: true,
  opacity: 0.35,
  depthTest: true,
  depthWrite: false,
});
const CALM_OUTLINE_MATERIAL = new THREE.LineBasicMaterial({
  color: 0x3a3a3a,
  transparent: true,
  opacity: 0.55,
  depthTest: true,
  depthWrite: false,
});
const CEILING_GRID_MATERIAL = new THREE.LineBasicMaterial({
  color: 0x999999,
  transparent: true,
  opacity: 0.35,
  depthTest: true,
  depthWrite: false,
});
const PICK_MATERIAL = new THREE.MeshBasicMaterial({
  colorWrite: false,
  depthWrite: false,
  depthTest: false,
  side: THREE.DoubleSide,
});
const GROUND_MATERIAL = new THREE.MeshStandardMaterial({
  color: 0xfafafa,
  roughness: 1.0,
  metalness: 0,
  side: THREE.FrontSide,
  depthWrite: true,
  polygonOffset: true,
  polygonOffsetFactor: 1,
  polygonOffsetUnits: 1,
});
const CALM_GROUND_MATERIAL = new THREE.MeshLambertMaterial({
  color: 0xf4f1ec,
  side: THREE.FrontSide,
  depthWrite: true,
});
const SELECTION_PADDING_M = 0.03;
const SELECTION_MIN_DIM_M = 0.05;

function makeMaterial(def, style = "tiers") {
  const isCeilingMaterial = def.name === "ceilingTop" || def.name === "ceilingSloped" || def.name === "ceilingRaw" || def.name === "gapCeiling";
  const renderBothSides = def.name === "window" || def.name === "structure" || def.name === "structureFill" || isCeilingMaterial;
  const options = {
    color: def.fill,
    side: renderBothSides ? THREE.DoubleSide : THREE.FrontSide,
    depthWrite: true,
  };
  if (style !== "calm" && def.name !== "window") {
    options.roughness = def.roughness;
    options.metalness = def.metalness;
  }
  if (def.name === "roof" || def.name === "roofAttic" || def.name === "dormer" || def.name === "structureFill" || def.name === "floor" || def.name === "gableClosure" || isCeilingMaterial) {
    options.polygonOffset = true;
    options.polygonOffsetFactor = -1;
    options.polygonOffsetUnits = -1;
  }
  if (def.name === "window") {
    options.transparent = true;
    options.opacity = 0.35;
    options.depthWrite = false;
  }
  if (isCeilingMaterial) {
    options.transparent = true;
    options.opacity = def.opacity ?? 0.35;
    options.depthWrite = false;
  }
  if (def.name === "roofAttic" || def.name === "gableClosure") {
    options.transparent = true;
    options.opacity = def.opacity ?? 0.25;
    options.depthWrite = false;
  }
  if (style === "calm") {
    options.vertexColors = true;
    options.transparent = true;
    if (options.opacity == null) options.opacity = def.opacity ?? 1;
    options.depthWrite = options.depthWrite !== false;
  }
  const material = def.name === "window" || isCeilingMaterial
    ? new THREE.MeshBasicMaterial(options)
    : style === "calm" ? new THREE.MeshLambertMaterial(options) : new THREE.MeshStandardMaterial(options);
  if (def.name === "window") material.forceSinglePass = true;
  material.name = def.name;
  return material;
}

const MATERIALS = Object.fromEntries(
  Object.entries(MATERIAL_PALETTE).map(([key, value]) => [key, makeMaterial(value)]),
);
const CALM_MATERIALS = Object.fromEntries(
  Object.entries(CALM_MATERIAL_PALETTE).map(([key, value]) => [key, makeMaterial(value, "calm")]),
);

function dormerFaceMaterial(face) {
  if (face?.kind === "dormer_header") {
    return { material: MATERIALS.roof, materialName: "roof" };
  }
  return { material: MATERIALS.structure, materialName: "structure" };
}

function vec(c) {
  return new THREE.Vector3(c.x ?? c[0], c.y ?? c[1], c.z ?? c[2]);
}

function cornerArray(c) {
  return [c.x ?? c[0], c.y ?? c[1], c.z ?? c[2]];
}

function basis(corners) {
  const n0 = newellNormal(corners.map(cornerArray));
  const n = new THREE.Vector3(n0?.x ?? 0, n0?.y ?? 1, n0?.z ?? 0).normalize();
  let u = new THREE.Vector3(1, 0, 0);
  let best = 0;
  for (let i = 0; i < corners.length; i += 1) {
    const a = vec(corners[i]);
    const b = vec(corners[(i + 1) % corners.length]);
    const edge = b.sub(a);
    edge.addScaledVector(n, -edge.dot(n));
    const len = edge.lengthSq();
    if (len > best) {
      best = len;
      u = edge.clone().normalize();
    }
  }
  const v = new THREE.Vector3().crossVectors(n, u).normalize();
  const origin = vec(corners[0]);
  return { origin, u, v, n };
}

function to2(loop, frame) {
  return loop.map((corner) => {
    const p = vec(corner).sub(frame.origin);
    return new THREE.Vector2(p.dot(frame.u), p.dot(frame.v));
  });
}

function normalizeLoop(loop, points2, wantPositive) {
  const positive = signedArea2(points2) > 0;
  if (positive === wantPositive) return { loop, points2 };
  return { loop: [...loop].reverse(), points2: [...points2].reverse() };
}

function dedupeLoop(loop) {
  const out = [];
  for (const corner of loop || []) {
    const prev = out[out.length - 1];
    if (!prev) {
      out.push(corner);
      continue;
    }
    const a = cornerArray(corner);
    const b = cornerArray(prev);
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    if (dx * dx + dy * dy + dz * dz > 1e-8) out.push(corner);
  }
  if (out.length >= 2) {
    const a = cornerArray(out[0]);
    const b = cornerArray(out[out.length - 1]);
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    if (dx * dx + dy * dy + dz * dz <= 1e-8) out.pop();
  }
  return out;
}

function segmentsCross(a, b, c, d) {
  const ccw = (p, q, r) => (r.y - p.y) * (q.x - p.x) - (q.y - p.y) * (r.x - p.x);
  const d1 = ccw(c, d, a);
  const d2 = ccw(c, d, b);
  const d3 = ccw(a, b, c);
  const d4 = ccw(a, b, d);
  return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
    && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}

function isSimplePolygon(points2) {
  const n = points2.length;
  if (n < 4) return true;
  for (let i = 0; i < n; i += 1) {
    const a = points2[i];
    const b = points2[(i + 1) % n];
    for (let j = i + 2; j < n; j += 1) {
      if (i === 0 && j === n - 1) continue;
      const c = points2[j];
      const d = points2[(j + 1) % n];
      if (segmentsCross(a, b, c, d)) return false;
    }
  }
  return true;
}

function repairSimpleQuad(loop, points2) {
  if (isSimplePolygon(points2) || points2.length !== 4) return { loop, points2 };
  for (const [i, k] of [[1, 2], [2, 3]]) {
    const candidateLoop = loop.slice();
    const candidatePoints = points2.slice();
    [candidateLoop[i], candidateLoop[k]] = [candidateLoop[k], candidateLoop[i]];
    [candidatePoints[i], candidatePoints[k]] = [candidatePoints[k], candidatePoints[i]];
    if (isSimplePolygon(candidatePoints)) return { loop: candidateLoop, points2: candidatePoints };
  }
  return null;
}

function polygonArea3(corners) {
  let nx = 0;
  let ny = 0;
  let nz = 0;
  const arr = corners.map(cornerArray);
  for (let i = 0; i < arr.length; i += 1) {
    const a = arr[i];
    const b = arr[(i + 1) % arr.length];
    nx += (a[1] - b[1]) * (a[2] + b[2]);
    ny += (a[2] - b[2]) * (a[0] + b[0]);
    nz += (a[0] - b[0]) * (a[1] + b[1]);
  }
  return 0.5 * Math.hypot(nx, ny, nz);
}

function polygonGeometry(corners, holes = []) {
  if (!corners || corners.length < 3) return null;
  if (polygonArea3(corners) < RENDER_TUNING.minPolygonAreaM2) return null;
  const outerLoop = dedupeLoop(corners);
  if (outerLoop.length < 3) return null;
  const frame = basis(outerLoop);
  let outer = normalizeLoop(outerLoop, to2(outerLoop, frame), true);
  outer = repairSimpleQuad(outer.loop, outer.points2);
  if (!outer) return null;
  const holeLoops = [];
  const holePoints = [];
  for (const hole of holes || []) {
    const holeLoop = dedupeLoop(hole);
    if (holeLoop.length < 3) continue;
    const normalized = normalizeLoop(holeLoop, to2(holeLoop, frame), false);
    holeLoops.push(normalized.loop);
    holePoints.push(normalized.points2);
  }
  let triangles = null;
  try {
    triangles = THREE.ShapeUtils.triangulateShape(outer.points2, holePoints);
  } catch (_err) {
    return null;
  }
  if (!triangles.length) return null;
  const loops = [outer.loop, ...holeLoops];
  const vertices = loops.flat();
  const positions = new Float32Array(vertices.length * 3);
  vertices.forEach((corner, idx) => {
    const p = vec(corner);
    positions[idx * 3] = p.x;
    positions[idx * 3 + 1] = p.y;
    positions[idx * 3 + 2] = p.z;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(triangles.flat());
  attachConfidenceAttribute(geometry, 1);
  return geometry;
}

/**
 * Attach a per-vertex RGBA color attribute (rgb=white, a=alpha) used by the
 * calm viewer's confidence shading. Tier-style materials don't enable
 * `vertexColors`, so the attribute is harmless overhead in audit mode.
 * Calm materials enable it and modulate alpha per-vertex.
 */
function attachConfidenceAttribute(geometry, alpha) {
  const count = geometry.attributes.position?.count ?? 0;
  if (!count) return;
  const colors = new Float32Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    colors[i * 4] = 1;
    colors[i * 4 + 1] = 1;
    colors[i * 4 + 2] = 1;
    colors[i * 4 + 3] = alpha;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 4));
}

function setGeometryConfidence(geometry, alpha) {
  const attr = geometry.attributes?.color;
  if (!attr || attr.itemSize !== 4) return;
  const arr = attr.array;
  for (let i = 3; i < arr.length; i += 4) arr[i] = alpha;
  attr.needsUpdate = true;
}

function addToAabb(aabb, corners) {
  for (const corner of corners || []) aabb.expandByPoint(vec(corner));
}

function selectionBoxFromCorners(corners, depth = SELECTION_MIN_DIM_M) {
  const loop = dedupeLoop(corners);
  if (loop.length < 3) return null;
  const frame = basis(loop);
  let minU = Infinity;
  let minV = Infinity;
  let minN = Infinity;
  let maxU = -Infinity;
  let maxV = -Infinity;
  let maxN = -Infinity;
  for (const corner of loop) {
    const p = vec(corner).sub(frame.origin);
    const u = p.dot(frame.u);
    const v = p.dot(frame.v);
    const n = p.dot(frame.n);
    minU = Math.min(minU, u);
    minV = Math.min(minV, v);
    minN = Math.min(minN, n);
    maxU = Math.max(maxU, u);
    maxV = Math.max(maxV, v);
    maxN = Math.max(maxN, n);
  }
  const size = new THREE.Vector3(
    Math.max(maxU - minU + SELECTION_PADDING_M * 2, SELECTION_MIN_DIM_M),
    Math.max(maxV - minV + SELECTION_PADDING_M * 2, SELECTION_MIN_DIM_M),
    Math.max(maxN - minN + depth + SELECTION_PADDING_M * 2, SELECTION_MIN_DIM_M),
  );
  const center = frame.origin.clone()
    .addScaledVector(frame.u, (minU + maxU) * 0.5)
    .addScaledVector(frame.v, (minV + maxV) * 0.5)
    .addScaledVector(frame.n, (minN + maxN) * 0.5);
  return {
    center,
    size,
    basis: {
      u: frame.u.clone(),
      v: frame.v.clone(),
      n: frame.n.clone(),
    },
  };
}

function selectionBoxFromOpening(frame, bounds, depth) {
  return {
    center: frame.origin.clone()
      .addScaledVector(frame.u, (bounds.minX + bounds.maxX) * 0.5)
      .addScaledVector(frame.v, (bounds.minY + bounds.maxY) * 0.5),
    size: new THREE.Vector3(
      Math.max(bounds.width + SELECTION_PADDING_M * 2, SELECTION_MIN_DIM_M),
      Math.max(bounds.height + SELECTION_PADDING_M * 2, SELECTION_MIN_DIM_M),
      Math.max(depth + SELECTION_PADDING_M * 2, SELECTION_MIN_DIM_M),
    ),
    basis: {
      u: frame.u.clone(),
      v: frame.v.clone(),
      n: frame.n.clone(),
    },
  };
}

function addLocatorMesh(scene, geometry, uid, locatorMap, selectionBox = null, story = null, heating = null) {
  if (!uid || !geometry) return;
  const pickGeometry = geometry.clone();
  const mesh = new THREE.Mesh(pickGeometry, PICK_MATERIAL);
  mesh.userData.tierPreview = true;
  mesh.userData.pickOnly = true;
  mesh.userData.story = story;
  mesh.userData.heating = heating;
  if (selectionBox) mesh.userData.selectionBox = selectionBox;
  attachLocator(mesh, uid);
  locatorMap.set(uid, mesh);
  scene.add(mesh);
}

function addOutline(parent, geometry, story, style = "tiers") {
  if (!geometry) return;
  const edges = new THREE.EdgesGeometry(geometry, EDGE_THRESHOLD_DEG);
  const line = new THREE.LineSegments(edges, style === "calm" ? CALM_OUTLINE_MATERIAL : OUTLINE_MATERIAL);
  line.userData.tierPreview = true;
  line.userData.story = story;
  parent.add(line);
}

function batchKey(materialName, story, heating) {
  return `${materialName}::${story}::${heating ?? "none"}`;
}

function batchGeometry(batches, materialName, story, heating, geometry) {
  if (!geometry) return;
  const key = batchKey(materialName, story, heating);
  const bucket = batches.get(key);
  if (bucket) bucket.geometries.push(geometry);
  else batches.set(key, { materialName, story, heating, geometries: [geometry] });
}

function addPolygon(scene, state, corners, materialName, uid, holes = [], outlineName = "", story = null, heating = null) {
  const geometry = polygonGeometry(corners, holes);
  if (!geometry) return;
  addToAabb(state.aabb, corners);
  const alpha = state.confidenceLookup ? state.confidenceLookup(uid) : 1;
  if (alpha < 1) setGeometryConfidence(geometry, alpha);
  batchGeometry(state.batches, materialName, story, heating, geometry);
  addLocatorMesh(scene, geometry, uid, state.locatorMap, selectionBoxFromCorners(corners), story, heating);
  if (OUTLINE_NAMES.has(outlineName)) {
    const outlineGeometry = geometry.clone();
    const outlineParent = state.storyGroup(story);
    addOutline(outlineParent, outlineGeometry, story, state.style);
  }
}

function gridIntersections(points, axis, value) {
  const hits = [];
  const aKey = axis === "x" ? "x" : "y";
  const bKey = axis === "x" ? "y" : "x";
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    const av = a[aKey];
    const bv = b[aKey];
    if (Math.abs(av - bv) < 1e-9) continue;
    if (value < Math.min(av, bv) || value >= Math.max(av, bv)) continue;
    const t = (value - av) / (bv - av);
    hits.push(a[bKey] + (b[bKey] - a[bKey]) * t);
  }
  hits.sort((a, b) => a - b);
  return hits;
}

function addGridSegment(positions, frame, axis, fixed, a, b) {
  if (Math.abs(b - a) < 1e-4) return;
  const p0 = axis === "x"
    ? frame.origin.clone().addScaledVector(frame.u, fixed).addScaledVector(frame.v, a)
    : frame.origin.clone().addScaledVector(frame.u, a).addScaledVector(frame.v, fixed);
  const p1 = axis === "x"
    ? frame.origin.clone().addScaledVector(frame.u, fixed).addScaledVector(frame.v, b)
    : frame.origin.clone().addScaledVector(frame.u, b).addScaledVector(frame.v, fixed);
  p0.addScaledVector(frame.n, CEILING_GRID_OFFSET_M);
  p1.addScaledVector(frame.n, CEILING_GRID_OFFSET_M);
  positions.push(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
}

function addCeilingGrid(parent, corners, holes, story) {
  const outerLoop = dedupeLoop(corners);
  if (outerLoop.length < 3) return;
  const frame = basis(outerLoop);
  const loops = [outerLoop, ...(holes || []).map(dedupeLoop).filter((loop) => loop.length >= 3)];
  const loops2 = loops.map((loop) => to2(loop, frame));
  const xs = loops2.flatMap((loop) => loop.map((p) => p.x));
  const ys = loops2.flatMap((loop) => loop.map((p) => p.y));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const positions = [];

  for (let x = Math.ceil(minX / CEILING_GRID_STEP_M) * CEILING_GRID_STEP_M; x <= maxX; x += CEILING_GRID_STEP_M) {
    const hits = loops2.flatMap((loop) => gridIntersections(loop, "x", x)).sort((a, b) => a - b);
    for (let i = 0; i + 1 < hits.length; i += 2) addGridSegment(positions, frame, "x", x, hits[i], hits[i + 1]);
  }
  for (let y = Math.ceil(minY / CEILING_GRID_STEP_M) * CEILING_GRID_STEP_M; y <= maxY; y += CEILING_GRID_STEP_M) {
    const hits = loops2.flatMap((loop) => gridIntersections(loop, "y", y)).sort((a, b) => a - b);
    for (let i = 0; i + 1 < hits.length; i += 2) addGridSegment(positions, frame, "y", y, hits[i], hits[i + 1]);
  }
  if (!positions.length) return;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const grid = new THREE.LineSegments(geometry, CEILING_GRID_MATERIAL);
  grid.userData.tierPreview = true;
  grid.userData.story = story;
  parent.add(grid);
}

function addCeilingSurface(scene, state, corners, uid, holes = [], story = null) {
  const geometry = polygonGeometry(corners, holes);
  if (!geometry) return;
  addToAabb(state.aabb, corners);
  const alpha = state.confidenceLookup ? state.confidenceLookup(uid) : 1;
  if (alpha < 1) setGeometryConfidence(geometry, alpha);
  batchGeometry(state.batches, "ceilingTop", story, null, geometry);
  addLocatorMesh(scene, geometry, uid, state.locatorMap, selectionBoxFromCorners(corners), story, null);
  addCeilingGrid(state.storyGroup(story), corners, holes, story);
}

function weldTolerance(materialName) {
  if (materialName === "roof" || materialName === "roofAttic" || materialName === "gableClosure" || materialName === "ceilingSloped" || materialName === "ceilingRaw" || materialName === "gapCeiling") return RENDER_TUNING.weldTol.roof;
  if (materialName === "structureFill") return RENDER_TUNING.weldTol.structureFill;
  return RENDER_TUNING.weldTol.default;
}

const HEATING_MATERIAL_CACHE = new Map();

function tintColor(baseHex, tintHex, factor) {
  const base = new THREE.Color(baseHex);
  const tint = new THREE.Color(tintHex);
  return base.lerp(tint, factor);
}

function getHeatingMaterial(materialName, heating, style = "tiers") {
  if (style === "calm") return CALM_MATERIALS[materialName];
  const baseDef = MATERIAL_PALETTE[materialName];
  if (!baseDef) return MATERIALS[materialName];
  const tintFactor = HEATING_TINT_FACTORS[materialName];
  if (tintFactor == null) return MATERIALS[materialName];
  const tintHex = HEATING_COLORS[heating ?? "heatedUnknown"] ?? HEATING_COLORS.heatedUnknown;
  const cacheKey = `${materialName}::${heating ?? "heatedUnknown"}`;
  const cached = HEATING_MATERIAL_CACHE.get(cacheKey);
  if (cached) return cached;
  const tinted = makeMaterial({ ...baseDef, fill: tintColor(baseDef.fill, tintHex, tintFactor).getHex() });
  HEATING_MATERIAL_CACHE.set(cacheKey, tinted);
  return tinted;
}

function resolveMaterial(materialName, heating, heatingMode, style = "tiers") {
  if (style === "calm") return CALM_MATERIALS[materialName];
  if (heatingMode && HEATING_TINT_FACTORS[materialName] != null) {
    return getHeatingMaterial(materialName, heating, style);
  }
  return MATERIALS[materialName];
}

function flushBatches(state, heatingMode) {
  for (const { materialName, story, heating, geometries } of state.batches.values()) {
    if (!geometries.length) continue;
    let merged = null;
    try {
      merged = mergeGeometries(geometries, false);
    } catch (_err) {
      merged = null;
    }
    if (!merged) continue;
    let welded = merged;
    try {
      welded = mergeVertices(merged, weldTolerance(materialName));
    } catch (_err) {
      welded = merged;
    }
    const creased = toCreasedNormals(welded, CREASE_ANGLE_RAD);
    if (welded !== merged) welded.dispose();
    const material = resolveMaterial(materialName, heating, heatingMode, state.style);
    const mesh = new THREE.Mesh(creased, material);
    mesh.userData.tierPreview = true;
    mesh.userData.story = story;
    mesh.userData.heating = heating;
    mesh.castShadow =
      state.style !== "calm"
      && materialName !== "structureFill"
      && materialName !== "window"
      && materialName !== "roofAttic"
      && materialName !== "gableClosure";
    mesh.receiveShadow = state.style !== "calm" && materialName !== "window" && materialName !== "roofAttic" && materialName !== "gableClosure";
    state.storyGroup(story).add(mesh);
    if (materialName === "roof" || materialName === "roofAttic" || materialName === "dormer" || materialName === "gableClosure") {
      addOutline(state.storyGroup(story), creased, story, state.style);
    }
    for (const geometry of geometries) {
      if (geometry !== merged && geometry !== welded && geometry !== creased) geometry.dispose();
    }
    if (merged !== creased) merged.dispose();
  }
}

function openingBounds(corners, frame) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const corner of corners || []) {
    const p = vec(corner).sub(frame.origin);
    const x = p.dot(frame.u);
    const y = p.dot(frame.v);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

function addOpeningBox(scene, state, quad, materialName, uid, depth, story) {
  if (!quad?.corners || quad.corners.length < 3) return;
  const frame = basis(quad.corners);
  const bounds = openingBounds(quad.corners, frame);
  if (bounds.width < RENDER_TUNING.opening.minDim || bounds.height < RENDER_TUNING.opening.minDim) return;
  addToAabb(state.aabb, quad.corners);
  const material = state.style === "calm" ? CALM_MATERIALS[materialName] : MATERIALS[materialName];
  const mesh = createOpeningBoxMesh(frame, bounds, 0, bounds.width, bounds.height, depth, material);
  mesh.userData.tierPreview = true;
  mesh.userData.story = story;
  mesh.castShadow = state.style !== "calm" && materialName !== "window";
  mesh.receiveShadow = state.style !== "calm" && materialName !== "window";
  state.storyGroup(story).add(mesh);
  mesh.updateMatrixWorld(true);
  const worldGeometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
  addLocatorMesh(
    scene,
    worldGeometry,
    uid,
    state.locatorMap,
    selectionBoxFromOpening(frame, bounds, depth),
    story,
    null,
  );
  worldGeometry.dispose();
  if (materialName !== "window") {
    addOutline(state.storyGroup(story), mesh.geometry.clone().applyMatrix4(mesh.matrixWorld), story, state.style);
  }
}

function createOpeningBoxMesh(frame, bounds, z, width, height, depth, material) {
  const geometry = new THREE.BoxGeometry(width, height, depth);
  const matrix = new THREE.Matrix4().makeBasis(frame.u, frame.v, frame.n);
  const center = frame.origin.clone()
    .addScaledVector(frame.u, (bounds.minX + bounds.maxX) * 0.5)
    .addScaledVector(frame.v, (bounds.minY + bounds.maxY) * 0.5)
    .addScaledVector(frame.n, z);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.quaternion.setFromRotationMatrix(matrix);
  mesh.position.copy(center);
  return mesh;
}

function addOpeningPartBox(parent, frame, x, y, z, width, height, depth, material) {
  const localBounds = {
    minX: x - width * 0.5,
    maxX: x + width * 0.5,
    minY: y - height * 0.5,
    maxY: y + height * 0.5,
  };
  const mesh = createOpeningBoxMesh(frame, localBounds, z, width, height, depth, material);
  mesh.userData.tierPreview = true;
  mesh.castShadow = material.name !== "window";
  mesh.receiveShadow = material.name !== "window";
  parent.add(mesh);
}

function addOpeningLocator(scene, state, frame, bounds, depth, uid, story) {
  const selectionMesh = createOpeningBoxMesh(frame, bounds, 0, bounds.width, bounds.height, depth, PICK_MATERIAL);
  selectionMesh.updateMatrixWorld(true);
  addLocatorMesh(
    scene,
    selectionMesh.geometry.clone().applyMatrix4(selectionMesh.matrixWorld),
    uid,
    state.locatorMap,
    selectionBoxFromOpening(frame, bounds, depth),
    story,
    null,
  );
  selectionMesh.geometry.dispose();
}

function addWindowModel(scene, state, quad, uid, story) {
  if (!quad?.corners || quad.corners.length < 3) return;
  const frame = basis(quad.corners);
  const bounds = openingBounds(quad.corners, frame);
  if (bounds.width < RENDER_TUNING.opening.minDim || bounds.height < RENDER_TUNING.opening.minDim) return;
  addToAabb(state.aabb, quad.corners);
  const parent = state.storyGroup(story);
  const frameMaterial = state.style === "calm" ? CALM_MATERIALS.structure : MATERIALS.structure;
  const glassMaterial = state.style === "calm" ? CALM_MATERIALS.window : MATERIALS.window;
  const frameThickness = Math.min(
    WINDOW_FRAME_THICKNESS_M,
    bounds.width * 0.2,
    bounds.height * 0.2,
  );
  const innerW = bounds.width - frameThickness * 2;
  const innerH = bounds.height - frameThickness * 2;
  const cx = (bounds.minX + bounds.maxX) * 0.5;
  const cy = (bounds.minY + bounds.maxY) * 0.5;
  const topY = bounds.maxY - frameThickness * 0.5;
  const bottomY = bounds.minY + frameThickness * 0.5;
  const leftX = bounds.minX + frameThickness * 0.5;
  const rightX = bounds.maxX - frameThickness * 0.5;

  addOpeningPartBox(parent, frame, cx, topY, 0, bounds.width, frameThickness, WINDOW_FRAME_DEPTH_M, frameMaterial);
  addOpeningPartBox(parent, frame, cx, bottomY, 0, bounds.width, frameThickness, WINDOW_FRAME_DEPTH_M, frameMaterial);
  if (innerH > RENDER_TUNING.opening.minDim) {
    addOpeningPartBox(parent, frame, leftX, cy, 0, frameThickness, innerH, WINDOW_FRAME_DEPTH_M, frameMaterial);
    addOpeningPartBox(parent, frame, rightX, cy, 0, frameThickness, innerH, WINDOW_FRAME_DEPTH_M, frameMaterial);
  }
  if (innerW > RENDER_TUNING.opening.minDim && innerH > RENDER_TUNING.opening.minDim) {
    const glassDepth = Math.max(0.004, WINDOW_FRAME_DEPTH_M * 0.08);
    addOpeningPartBox(parent, frame, cx, cy, 0, innerW, innerH, glassDepth, glassMaterial);
  }

  addOpeningLocator(scene, state, frame, bounds, WINDOW_FRAME_DEPTH_M, uid, story);
}

function addDoorModel(scene, state, quad, uid, story) {
  if (!quad?.corners || quad.corners.length < 3) return;
  const frame = basis(quad.corners);
  const bounds = openingBounds(quad.corners, frame);
  if (bounds.width < RENDER_TUNING.opening.minDim || bounds.height < RENDER_TUNING.opening.minDim) return;
  addToAabb(state.aabb, quad.corners);
  const parent = state.storyGroup(story);
  const frameMaterial = state.style === "calm" ? CALM_MATERIALS.structure : MATERIALS.structure;
  const leafMaterial = state.style === "calm" ? CALM_MATERIALS.doorLeaf : MATERIALS.doorLeaf;
  const frameThickness = Math.min(
    DOOR_FRAME_THICKNESS_M,
    bounds.width * 0.2,
    bounds.height * 0.2,
  );
  const leafW = Math.max(bounds.width - frameThickness * 2, RENDER_TUNING.opening.minDim);
  const leafH = Math.max(bounds.height - frameThickness, RENDER_TUNING.opening.minDim);
  const cx = (bounds.minX + bounds.maxX) * 0.5;
  const cy = (bounds.minY + bounds.maxY) * 0.5;
  const topY = bounds.maxY - frameThickness * 0.5;
  const leftX = bounds.minX + frameThickness * 0.5;
  const rightX = bounds.maxX - frameThickness * 0.5;
  const leafCenterY = cy - frameThickness * 0.5;

  addOpeningPartBox(parent, frame, leftX, cy, 0, frameThickness, bounds.height, DOOR_FRAME_DEPTH_M, frameMaterial);
  addOpeningPartBox(parent, frame, rightX, cy, 0, frameThickness, bounds.height, DOOR_FRAME_DEPTH_M, frameMaterial);
  addOpeningPartBox(parent, frame, cx, topY, 0, bounds.width, frameThickness, DOOR_FRAME_DEPTH_M, frameMaterial);
  addOpeningPartBox(parent, frame, cx, leafCenterY, 0, leafW, leafH, DOOR_LEAF_DEPTH_M, leafMaterial);

  const handleY = bounds.minY + bounds.height * 0.5;
  const handleX = bounds.maxX - frameThickness - 0.045;
  addOpeningPartBox(parent, frame, handleX, handleY, DOOR_LEAF_DEPTH_M * 0.5 + 0.005, 0.028, 0.14, 0.01, frameMaterial);
  addOpeningPartBox(parent, frame, handleX, handleY, DOOR_LEAF_DEPTH_M * 0.5 + 0.025, 0.022, 0.10, 0.035, frameMaterial);

  addOpeningLocator(scene, state, frame, bounds, DOOR_FRAME_DEPTH_M, uid, story);
}

// Render a reconstructed Stair as a stepped 3D primitive at the StairNode's
// position + rotation. Each segment renders in sequence; segments with
// `attachment_side: 'left'|'right'` apply a 90° turn around +Y at the segment
// boundary so switchback geometry chains correctly.
function addStairModel(scene, state, stair, story) {
  if (!stair?.position || !Array.isArray(stair.segments) || stair.segments.length === 0) return;
  const stairMat = state.style === "calm" ? CALM_MATERIALS.stair : MATERIALS.stair;
  const landingMat = state.style === "calm" ? CALM_MATERIALS.stairLanding : MATERIALS.stairLanding;
  const parent = state.storyGroup(story);

  const root = new THREE.Group();
  root.position.set(stair.position.x, stair.position.y, stair.position.z);
  root.rotation.y = stair.rotation ?? 0;
  root.userData.tierPreview = true;
  parent.add(root);

  // Cumulative cursor: position + rotation in stair-local frame, advanced
  // segment by segment. Each segment is rendered at the cursor's transform.
  const cursor = new THREE.Object3D();
  cursor.position.set(0, 0, 0);
  cursor.rotation.y = 0;

  for (let i = 0; i < stair.segments.length; i++) {
    const seg = stair.segments[i];
    if (i > 0 && seg.attachment_side === "left") cursor.rotation.y += Math.PI / 2;
    else if (i > 0 && seg.attachment_side === "right") cursor.rotation.y -= Math.PI / 2;

    const segGroup = new THREE.Group();
    segGroup.position.copy(cursor.position);
    segGroup.rotation.y = cursor.rotation.y;
    segGroup.userData.tierPreview = true;
    root.add(segGroup);

    const width = Math.max(seg.width ?? 1, 0.05);
    const length = Math.max(seg.length ?? 1, 0.05);
    const height = Math.max(seg.height ?? 0, 0);

    if (seg.segment_type === "stair" && seg.step_count > 0) {
      const n = seg.step_count;
      const treadDepth = length / n;
      const stepHeight = height / n;
      for (let s = 0; s < n; s++) {
        const stepBox = new THREE.BoxGeometry(width, stepHeight * (s + 1), treadDepth);
        const mesh = new THREE.Mesh(stepBox, stairMat);
        mesh.position.set(0, stepHeight * (s + 1) * 0.5, treadDepth * (s + 0.5));
        mesh.userData.tierPreview = true;
        segGroup.add(mesh);
      }
      const uid = `${stair.locator_id}/seg:${i}`;
      const selectionBox = {
        center: [length * 0.5 * Math.sin(segGroup.rotation.y) + segGroup.position.x,
                 segGroup.position.y + height * 0.5,
                 length * 0.5 * Math.cos(segGroup.rotation.y) + segGroup.position.z],
        half: [Math.max(width, length) * 0.5 + 0.05, Math.max(height * 0.5, 0.5), Math.max(width, length) * 0.5 + 0.05],
      };
      // Locators are anchored to the root group's world space; just record uid → selection bounds.
      state.locatorMap.set(uid, { story, heating: null, selectionBox });
    } else {
      // Landing: thin slab the full width × length, ~0.05 m thick, at cursor Y.
      const slab = new THREE.BoxGeometry(width, 0.05, length);
      const mesh = new THREE.Mesh(slab, landingMat);
      mesh.position.set(0, 0.025, length * 0.5);
      mesh.userData.tierPreview = true;
      segGroup.add(mesh);
    }

    // Advance cursor by this segment's length along its current heading,
    // accumulate vertical rise.
    const dx = Math.sin(cursor.rotation.y) * length;
    const dz = Math.cos(cursor.rotation.y) * length;
    cursor.position.x += dx;
    cursor.position.z += dz;
    cursor.position.y += height;
  }

  // Optional slab opening polygon: render a thin translucent disc at the upper
  // floor's Y. Not part of the stair body — it's a hint about where the upper
  // slab should be cut.
  if (Array.isArray(stair.slab_opening_polygon) && stair.slab_opening_polygon.length >= 3) {
    addPolygon(
      scene,
      state,
      stair.slab_opening_polygon,
      "ceilingTop",
      `${stair.locator_id}/slab_opening`,
      [],
      "ceilingTop",
      story,
      null,
    );
  }
}

export function clearBuildingMeshes(scene) {
  const doomed = [];
  scene.traverse((obj) => {
    if (obj.userData?.tierPreview) doomed.push(obj);
  });
  for (const obj of doomed) {
    obj.removeFromParent();
    obj.geometry?.dispose?.();
    if (obj.userData?.disposeMaterialOnClear) obj.material?.dispose?.();
  }
  scene.userData.tierLocatorMap = new Map();
}

export function addPascalLighting(scene) {
  if (scene.userData.tierLighting) return;
  scene.add(new THREE.AmbientLight(0xffffff, RENDER_TUNING.light.ambient));
  const key = new THREE.DirectionalLight(0xffffff, RENDER_TUNING.light.key);
  key.position.set(10, 10, 10);
  key.castShadow = true;
  key.shadow.bias = RENDER_TUNING.shadow.bias;
  key.shadow.normalBias = RENDER_TUNING.shadow.normalBias;
  key.shadow.radius = RENDER_TUNING.shadow.radius;
  key.shadow.mapSize.set(RENDER_TUNING.shadow.mapSize, RENDER_TUNING.shadow.mapSize);
  if ("intensity" in key.shadow) key.shadow.intensity = 0.6;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 100;
  key.shadow.camera.left = -RENDER_TUNING.shadow.halfExtent;
  key.shadow.camera.right = RENDER_TUNING.shadow.halfExtent;
  key.shadow.camera.top = RENDER_TUNING.shadow.halfExtent;
  key.shadow.camera.bottom = -RENDER_TUNING.shadow.halfExtent;
  scene.add(key);
  const fill1 = new THREE.DirectionalLight(0xffffff, RENDER_TUNING.light.fill1);
  fill1.position.set(-10, 10, -10);
  scene.add(fill1);
  const fill2 = new THREE.DirectionalLight(0xffffff, RENDER_TUNING.light.fill2);
  fill2.position.set(-10, 10, 10);
  scene.add(fill2);
  scene.userData.tierLighting = true;
}

export function addCalmLighting(scene) {
  if (scene.userData.calmLighting) return;
  scene.add(new THREE.AmbientLight(0xfff8ee, 1.25));
  const key = new THREE.DirectionalLight(0xfff0dc, 0.65);
  key.position.set(8, 12, 10);
  key.castShadow = false;
  scene.add(key);
  scene.userData.calmLighting = true;
}

function avgCorner(corners, key) {
  if (!corners?.length) return 0;
  let sum = 0;
  for (const c of corners) sum += c[key] ?? c[key === "x" ? 0 : key === "y" ? 1 : 2];
  return sum / corners.length;
}

function roomFloorPieces(room) {
  // Schema migration: room.floor used to be a single HorizontalLid; it is now
  // list[HorizontalLid] (one piece per uniform-adjacency region). Tolerate
  // both shapes so older payloads still render.
  if (Array.isArray(room?.floor)) return room.floor;
  if (room?.floor?.corners) return [room.floor];
  return [];
}

function buildRoomLookup(payload) {
  const stories = new Map();
  for (const room of payload.rooms || []) {
    const pieces = roomFloorPieces(room);
    if (!pieces.length) continue;
    const allCorners = pieces.flatMap((p) => p.corners ?? []);
    if (!allCorners.length) continue;
    const cx = avgCorner(allCorners, "x");
    const cy = avgCorner(allCorners, "y");
    const cz = avgCorner(allCorners, "z");
    const entry = { story: room.story ?? 0, heating: room.heating ?? null, cx, cy, cz };
    if (!stories.has(entry.story)) stories.set(entry.story, []);
    stories.get(entry.story).push(entry);
  }
  const storyMeanY = new Map();
  for (const [story, list] of stories) {
    const meanY = list.reduce((a, b) => a + b.cy, 0) / list.length;
    storyMeanY.set(story, meanY);
  }
  return { stories, storyMeanY };
}

function nearestRoom(rooms, cx, cz) {
  let best = null;
  for (const r of rooms) {
    const dx = r.cx - cx;
    const dz = r.cz - cz;
    const d = dx * dx + dz * dz;
    if (best === null || d < best.d) best = { d, room: r };
  }
  return best?.room ?? null;
}

function inferStoryFromY(storyMeanY, y) {
  let bestStory = 0;
  let bestDist = Infinity;
  for (const [story, meanY] of storyMeanY) {
    const d = Math.abs(meanY - y);
    if (d < bestDist) {
      bestDist = d;
      bestStory = story;
    }
  }
  return bestStory;
}

function storyBelow(storyMeanY, y) {
  // Return the story whose meanY is the highest value still ≤ y (the storey directly under the point)
  let best = null;
  let bestMeanY = -Infinity;
  for (const [story, meanY] of storyMeanY) {
    if (meanY <= y + 0.01 && meanY > bestMeanY) {
      bestMeanY = meanY;
      best = story;
    }
  }
  return best;
}

function gapStoryAndHeating(gap, lookup) {
  const cx = avgCorner(gap.corners, "x");
  const cy = avgCorner(gap.corners, "y");
  const cz = avgCorner(gap.corners, "z");
  if (gap.scope === "exterior") {
    return { story: EXTERIOR_STORY_KEY, heating: null };
  }
  if (gap.scope === "inter_story" || gap.scope === "junction") {
    // Cross-storey gap: visualise with the storey directly below it (and inherit that storey's heating)
    const lowerStory = storyBelow(lookup.storyMeanY, cy);
    const targetStory = lowerStory ?? inferStoryFromY(lookup.storyMeanY, cy);
    const rooms = lookup.stories.get(targetStory) ?? [];
    const nearest = nearestRoom(rooms, cx, cz);
    return { story: targetStory, heating: nearest?.heating ?? null };
  }
  // intra_story or stitch: pick nearest same-story room
  const sameStory = [];
  for (const [story, list] of lookup.stories) {
    if (Math.abs((lookup.storyMeanY.get(story) ?? 0) - cy) < SPLIT_LEVEL_DY_M) sameStory.push(...list);
  }
  const pool = sameStory.length ? sameStory : [...lookup.stories.values()].flat();
  const nearest = nearestRoom(pool, cx, cz);
  if (!nearest) return { story: inferStoryFromY(lookup.storyMeanY, cy), heating: null };
  return { story: nearest.story, heating: nearest.heating };
}

// Map a CeilingPiece.source string to a material name. Sloped and raw ceiling
// evidence stays visually distinct from exterior roof/shell geometry.
function ceilingMaterialName(source) {
  if (source === "roof_arrangement_attic") return "roofAttic";
  if (source === "flat_ceiling" || source === "flat_emit" || source === "thermal_cap" || source === "flat_gap_closure" || source === "attic_flat_lid") {
    return "ceilingTop";
  }
  if (source === "raw_scan" || source === "raw_fallback") return "ceilingRaw";
  return "ceilingSloped";
}

function ceilingStoryFromCellId(cellId, fallbackY, lookup) {
  if (typeof cellId === "string") {
    const parts = cellId.split(":");
    if (parts.length >= 2) {
      const n = Number(parts[1]);
      if (Number.isInteger(n)) return n;
    }
  }
  return inferStoryFromY(lookup.storyMeanY, fallbackY);
}

function makeStoryGroupFactory(scene) {
  const groups = new Map();
  return {
    groups,
    get(story) {
      const key = story == null ? "null" : String(story);
      if (groups.has(key)) return groups.get(key);
      const group = new THREE.Group();
      group.userData.tierPreview = true;
      group.userData.tierStoryGroup = true;
      group.userData.story = story;
      groups.set(key, group);
      scene.add(group);
      return group;
    },
  };
}

export function populateBuildingScene(scene, payload, options = {}) {
  const heatingMode = !!options.heatingMode;
  const style = options.style === "calm" ? "calm" : "tiers";
  clearBuildingMeshes(scene);
  const factory = makeStoryGroupFactory(scene);
  const state = {
    locatorMap: new Map(),
    batches: new Map(),
    aabb: new THREE.Box3(),
    storyGroup: factory.get,
    style,
  };
  scene.userData.tierLocatorMap = state.locatorMap;
  scene.userData.tierStoryGroups = factory.groups;
  state.aabb.makeEmpty();
  state.confidenceLookup = typeof options.confidenceLookup === "function" ? options.confidenceLookup : null;

  const lookup = buildRoomLookup(payload);

  for (const room of payload.rooms || []) {
    const story = room.story ?? 0;
    const heating = room.heating ?? null;
    const floorPieces = roomFloorPieces(room);
    floorPieces.forEach((piece, idx) => {
      if (!piece?.corners?.length) return;
      const locatorSuffix = floorPieces.length > 1 ? `${room.locator_id}/${idx}` : room.locator_id;
      addPolygon(scene, state, piece.corners, "floor", locatorSuffix, [], "floor", story, heating);
    });
    for (const wall of room.walls || []) {
      addPolygon(
        scene,
        state,
        wall.corners,
        "structure",
        wall.locator_id,
        wall.cutouts?.map((quad) => quad.corners) ?? [],
        "structure",
        story,
        heating,
      );
      if (wall.descent_strip) {
        addPolygon(scene, state, wall.descent_strip, "structure", `${wall.locator_id}:descent`, [], "structure", story, heating);
      }
      if (wall.uplift_strip) {
        addPolygon(scene, state, wall.uplift_strip, "structure", `${wall.locator_id}:uplift`, [], "structure", story, heating);
      }
    }
    room.doors?.forEach((door, index) => {
      addDoorModel(scene, state, door, `${room.locator_id}:door:${index}`, story);
    });
    room.windows?.forEach((window, index) => {
      addWindowModel(scene, state, window, `${room.locator_id}:window:${index}`, story);
    });
    room.openings?.forEach((opening, index) => {
      addOpeningBox(scene, state, opening, "opening", `${room.locator_id}:opening:${index}`, RENDER_TUNING.opening.doorDepth, story);
    });
  }

  for (const gap of payload.gaps || []) {
    const materialDef = style === "calm" ? calmGapMaterial(gap.kind) : gapMaterial(gap.kind);
    const { story, heating } = gapStoryAndHeating(gap, lookup);
    addPolygon(scene, state, gap.corners, materialDef.name, gap.locator_id, [], materialDef.name, story, heating);
  }
  for (const piece of payload.ceiling || []) {
    const materialName = ceilingMaterialName(piece.source);
    const cy = avgCorner(piece.corners, "y");
    const story = ceilingStoryFromCellId(piece.arrangement_cell_id, cy, lookup);
    if (materialName === "ceilingTop") {
      addCeilingSurface(scene, state, piece.corners, piece.locator_id, piece.holes ?? [], story);
    } else {
      addPolygon(scene, state, piece.corners, materialName, piece.locator_id, piece.holes ?? [], materialName, story, null);
    }
  }
  for (const shell of payload.visual_shells || []) {
    const cy = avgCorner(shell.corners, "y");
    const story = inferStoryFromY(lookup.storyMeanY, cy);
    addPolygon(scene, state, shell.corners, "roofAttic", shell.locator_id, [], "roofAttic", story, null);
  }
  for (const wall of payload.knee_walls || []) {
    const cy = avgCorner(wall.corners, "y");
    const story = inferStoryFromY(lookup.storyMeanY, cy);
    addPolygon(scene, state, wall.corners, "dormer", wall.locator_id, [], "dormer", story, null);
  }
  for (const face of payload.dormer_faces || []) {
    const { materialName } = dormerFaceMaterial(face);
    const cy = avgCorner(face.corners, "y");
    const story = inferStoryFromY(lookup.storyMeanY, cy);
    addPolygon(
      scene,
      state,
      face.corners,
      materialName,
      face.locator_id,
      face.cutouts?.map((quad) => quad.corners) ?? [],
      materialName,
      story,
      null,
    );
  }
  for (const stair of payload.stairs || []) {
    const story = stair.from_story ?? null;
    addStairModel(scene, state, stair, story);
  }
  for (const closure of payload.gable_closures || []) {
    const cy = avgCorner(closure.corners, "y");
    const story = inferStoryFromY(lookup.storyMeanY, cy);
    addPolygon(scene, state, closure.corners, "gableClosure", closure.locator_id, [], "gableClosure", story, null);
  }

  flushBatches(state, heatingMode);

  const center = state.aabb.isEmpty()
    ? new THREE.Vector3(payload.building_center?.x ?? 0, payload.building_center?.y ?? 0, payload.building_center?.z ?? 0)
    : state.aabb.getCenter(new THREE.Vector3());
  if (!state.aabb.isEmpty()) {
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(RENDER_TUNING.ground.size, RENDER_TUNING.ground.size),
      style === "calm" ? CALM_GROUND_MATERIAL : GROUND_MATERIAL,
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(center.x, state.aabb.min.y - RENDER_TUNING.ground.dropM, center.z);
    ground.receiveShadow = style !== "calm";
    ground.castShadow = false;
    ground.userData.tierPreview = true;
    ground.userData.framingIgnore = true;
    scene.add(ground);
  }
  return { center, locatorMap: state.locatorMap, storyGroups: factory.groups, storyMeanY: lookup.storyMeanY };
}

export function setStoryExplode(scene, separationM) {
  const groups = scene.userData.tierStoryGroups;
  if (!groups) return;
  for (const [key, group] of groups) {
    if (key === "null" || key === EXTERIOR_STORY_KEY) {
      group.position.y = 0;
      continue;
    }
    const idx = Number(key);
    group.position.y = Number.isFinite(idx) ? idx * separationM : 0;
  }
  // Apply same offset to picker meshes (which sit directly on scene root)
  scene.traverse((obj) => {
    if (obj.userData?.pickOnly) {
      const story = obj.userData.story;
      if (story == null || story === EXTERIOR_STORY_KEY) {
        obj.position.y = 0;
      } else if (Number.isFinite(story)) {
        obj.position.y = story * separationM;
      }
    }
  });
}

export { setCeilingsVisible } from "./ceiling-visibility.js";

export function setVisibleStory(scene, selection) {
  const groups = scene.userData.tierStoryGroups;
  if (!groups) return;
  for (const [key, group] of groups) {
    group.visible = isStoryVisible(key, selection, { isKey: true });
  }
  scene.traverse((obj) => {
    if (obj.userData?.pickOnly) {
      obj.visible = isStoryVisible(obj.userData.story, selection);
    }
  });
}
