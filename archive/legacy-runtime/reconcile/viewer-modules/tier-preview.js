// Tier viewer: builds a THREE.Scene for one building using the same palette
// and 3D door/window/roof/dormer treatment as the main viewer's "Full model"
// toggle (viewer-main.js:3066–3226), so the tier page looks identical to what
// the user already knows.
//
// Source data per building (single endpoint):
//   - /building-merged?uuid=<u>  → walls_computed + extension_strip, floors,
//     gap_walls, gap_closures, stitch_walls, cross_floor_gaps, doors/windows,
//     thermal-knee walls + dormer cheeks/headers (from the roof pipeline's
//     ceiling.thermal output), raw-ceiling fallback scraps, and the V2 raw
//     eave-supported final-layer `slanted_pieces` — all with dormer cutouts
//     subtracted in XZ server-side, so the tier preview gets a real hole
//     through the roof beneath each dormer for free.

import * as THREE from 'three';
import { mergeGeometries, mergeVertices, toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import {
  polygonPlaneBasis,
  projectToPlane2,
  collectWallCutoutHoles,
  orientedStructureCorners,
} from './geometry.js';

// Palette copied religiously from Pascal editor:
//   - walls / frames / structural: `baseMaterial` in
//     `.context/pascal-editor/packages/core/src/materials.ts:6` → `#f2f0ed`,
//     roughness 0.5
//   - windows: `glassMaterial` in same file → lightblue, roughness 0.05,
//     metalness 0.1, opacity 0.35, double-side, depthWrite false
//   - slabs / floors / ceilings: Pascal roof-materials.ts deck → `#e5e5e5`,
//     roughness ~1.0 (matches `DEFAULT_SLAB_MATERIAL`)
//   - roof shingle: Pascal roof-materials.ts shingle → `#e5e5e5`,
//     roughness 0.9 (production values from
//     `.context/pascal-editor/packages/viewer/src/components/renderers/roof/roof-materials.ts`)
//   - doors: `DEFAULT_DOOR_MATERIAL` → `#8b4513`, roughness 0.7
//   - default ceiling: `DEFAULT_CEILING_MATERIAL` → `#f5f5dc`, roughness 0.95
const PALETTE = {
  structure: { fill: 0xf2f0ed, roughness: 0.5,  metalness: 0.0 },
  floor:     { fill: 0xe5e5e5, roughness: 1.0,  metalness: 0.0 },
  ceiling:   { fill: 0xf5f5dc, roughness: 0.95, metalness: 0.0 },
  door:      { fill: 0x8b4513, roughness: 0.7,  metalness: 0.0 },
  window:    { fill: 0xadd8e6, roughness: 0.05, metalness: 0.1 },
  opening:   { fill: 0x1a1a1a, roughness: 0.6,  metalness: 0.0 },
  roof:      { fill: 0x808080, roughness: 0.85, metalness: 0.0 },
  dormer:    { fill: 0xf2f0ed, roughness: 0.5,  metalness: 0.0 },
};

const MATERIALS = {
  // Structural walls — FrontSide matches Pascal's WallRenderer
  // (`.context/pascal-editor/packages/viewer/src/components/renderers/wall/
  // wall-renderer.tsx`) and avoids the back-face z-fighting that shows up
  // as a "tile mosaic" when adjacent walls nearly touch. The corner winding
  // must be consistent (see `orientedStructureCorners`).
  structure: new THREE.MeshStandardMaterial({
    color: PALETTE.structure.fill, side: THREE.FrontSide,
    roughness: PALETTE.structure.roughness, metalness: PALETTE.structure.metalness,
    depthWrite: true,
  }),
  floor: new THREE.MeshStandardMaterial({
    color: PALETTE.floor.fill, side: THREE.DoubleSide,
    roughness: PALETTE.floor.roughness, metalness: PALETTE.floor.metalness,
    depthWrite: true,
  }),
  ceiling: new THREE.MeshStandardMaterial({
    color: PALETTE.ceiling.fill, side: THREE.DoubleSide,
    roughness: PALETTE.ceiling.roughness, metalness: PALETTE.ceiling.metalness,
    depthWrite: true,
  }),
  roof: new THREE.MeshStandardMaterial({
    color: PALETTE.roof.fill, side: THREE.DoubleSide,
    roughness: PALETTE.roof.roughness, metalness: PALETTE.roof.metalness,
    depthWrite: true,
  }),
  dormer: new THREE.MeshStandardMaterial({
    color: PALETTE.dormer.fill, side: THREE.DoubleSide,
    roughness: PALETTE.dormer.roughness, metalness: PALETTE.dormer.metalness,
    depthWrite: true,
  }),
  // Interior fill — gap walls, gap closures (side quads), stitch walls, knee
  // walls. Same pigment as structure and `FrontSide` so the back-face is
  // culled like a real wall (user spec: vertical gap-closing pieces should
  // face outwards). Kept in its own bucket with weldTolerance=0 so
  // `mergeVertices` can't average a flipped-winding fill triangle against
  // walls_computed (welding averages normals across reversed faces →
  // near-zero normal → black shading along the seam). castShadow is also
  // disabled for this bucket in `flushBatches` — thin gap walls (~30 cm
  // wide) alias the 1024² shadow map into 1-to-3-texel bands that look
  // like black stripes on adjacent floors.
  structureFill: new THREE.MeshStandardMaterial({
    color: PALETTE.structure.fill, side: THREE.FrontSide,
    roughness: PALETTE.structure.roughness, metalness: PALETTE.structure.metalness,
    depthWrite: true,
  }),
  // Glass — Pascal's `glassMaterial` exactly (core/src/materials.ts:15).
  window: new THREE.MeshStandardMaterial({
    color: PALETTE.window.fill, transparent: true, opacity: 0.35,
    side: THREE.DoubleSide, depthWrite: false,
    roughness: PALETTE.window.roughness, metalness: PALETTE.window.metalness,
  }),
  // Door — Pascal's `DEFAULT_DOOR_MATERIAL` exactly (viewer/lib/materials.ts:287).
  doorLeaf: new THREE.MeshStandardMaterial({
    color: PALETTE.door.fill, side: THREE.DoubleSide,
    roughness: PALETTE.door.roughness, metalness: PALETTE.door.metalness,
    depthWrite: true,
  }),
};

// Triangulate a planar polygon (with optional holes) and return a mesh using
// the given material. Algorithm from viewer-modules/geometry.js:createPolygonMesh,
// trimmed for opaque materials used by the tier preview.

// Only these materials produce outline lines. See `addPoly`.
const _OUTLINE_MATERIALS = new Set([MATERIALS.roof, MATERIALS.doorLeaf, MATERIALS.dormer]);

function makePolyGeometry(corners, holes) {
  if (!Array.isArray(corners) || corners.length < 3) return null;

  const p0 = new THREE.Vector3(...corners[0]);
  const n = new THREE.Vector3();
  for (let i = 0; i < corners.length; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    n.x += (a[1] - b[1]) * (a[2] + b[2]);
    n.y += (a[2] - b[2]) * (a[0] + b[0]);
    n.z += (a[0] - b[0]) * (a[1] + b[1]);
  }
  if (n.lengthSq() < 1e-10) n.set(0, 1, 0);
  else n.normalize();

  let u = new THREE.Vector3();
  let bestLen = 0;
  for (let i = 0; i < corners.length; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    const e = new THREE.Vector3(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    e.addScaledVector(n, -e.dot(n));
    const l2 = e.lengthSq();
    if (l2 > bestLen) { bestLen = l2; u.copy(e); }
  }
  if (bestLen < 1e-10) u.set(1, 0, 0);
  else u.normalize();
  const v = new THREE.Vector3().crossVectors(n, u).normalize();

  const toUV = (c) => {
    const p = new THREE.Vector3(c[0], c[1], c[2]).sub(p0);
    return new THREE.Vector2(p.dot(u), p.dot(v));
  };
  const dedupe = (loop) => {
    const out = [];
    for (const p of loop) {
      const prev = out[out.length - 1];
      if (!prev) { out.push(p); continue; }
      const dx = p[0] - prev[0], dy = p[1] - prev[1], dz = p[2] - prev[2];
      if ((dx * dx + dy * dy + dz * dz) > 1e-8) out.push(p);
    }
    if (out.length >= 2) {
      const a = out[0], b = out[out.length - 1];
      const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
      if ((dx * dx + dy * dy + dz * dz) <= 1e-8) out.pop();
    }
    return out;
  };
  const outer = dedupe(corners);
  const innerLoops = (holes || [])
    .filter((h) => Array.isArray(h) && h.length >= 3)
    .map(dedupe)
    .filter((h) => h.length >= 3);
  if (outer.length < 3) return null;

  let contour = outer.map(toUV);
  let holeContours = innerLoops.map((h) => h.map(toUV));

  // Bow-tie repair: same logic as createPolygonMesh in geometry.js. Earcut
  // on a self-intersecting quad produces the X-pattern the user sees for
  // wall-computed / ceiling-raw on scans with out-of-order corners.
  const segmentsCross = (a, b, c, d) => {
    const ccw = (p, q, r) => (r.y - p.y) * (q.x - p.x) - (q.y - p.y) * (r.x - p.x);
    const d1 = ccw(c, d, a), d2 = ccw(c, d, b);
    const d3 = ccw(a, b, c), d4 = ccw(a, b, d);
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
        && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
  };
  const isSimplePolygon = (pts) => {
    const n = pts.length;
    if (n < 4) return true;
    for (let i = 0; i < n; i++) {
      const a = pts[i], b = pts[(i + 1) % n];
      for (let j = i + 2; j < n; j++) {
        if (i === 0 && j === n - 1) continue;
        const c = pts[j], d = pts[(j + 1) % n];
        if (segmentsCross(a, b, c, d)) return false;
      }
    }
    return true;
  };
  if (!isSimplePolygon(contour)) {
    if (contour.length === 4) {
      const swapAt = (arr, i, k) => { const c = arr.slice(); [c[i], c[k]] = [c[k], c[i]]; return c; };
      let repairedUV = null;
      for (const [i, k] of [[1, 2], [2, 3]]) {
        const candidateUV = swapAt(contour, i, k);
        if (isSimplePolygon(candidateUV)) { repairedUV = candidateUV; break; }
      }
      if (repairedUV) contour = repairedUV;
      else return null;
    } else {
      return null;
    }
  }

  if (!THREE.ShapeUtils.isClockWise(contour)) contour = contour.slice().reverse();
  holeContours = holeContours.map((h) => (THREE.ShapeUtils.isClockWise(h) ? h.slice().reverse() : h));

  let indices;
  try {
    indices = THREE.ShapeUtils.triangulateShape(contour, holeContours);
  } catch (_err) {
    return null;
  }
  if (!indices || indices.length === 0) return null;

  const flat = [contour, ...holeContours];
  const allUV = flat.flat();
  const verts = new Float32Array(allUV.length * 3);
  for (let i = 0; i < allUV.length; i++) {
    const uv = allUV[i];
    const p = new THREE.Vector3().copy(p0).addScaledVector(u, uv.x).addScaledVector(v, uv.y);
    verts[i * 3] = p.x; verts[i * 3 + 1] = p.y; verts[i * 3 + 2] = p.z;
  }
  const idx = [];
  for (const tri of indices) idx.push(tri[0], tri[1], tri[2]);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setIndex(idx);
  // Leave normals to be computed post-merge. `flushBatches` runs
  // `toCreasedNormals` after `mergeVertices`, which averages only inside
  // coplanar groups and splits vertices at real creases (≥ CREASE_ANGLE_RAD)
  // so facades stay smooth and corners/ridges stay crisp.
  return geo;
}

function expandAabb(aabb, corners) {
  if (!Array.isArray(corners)) return;
  for (const c of corners) {
    aabb.min.x = Math.min(aabb.min.x, c[0]);
    aabb.min.y = Math.min(aabb.min.y, c[1]);
    aabb.min.z = Math.min(aabb.min.z, c[2]);
    aabb.max.x = Math.max(aabb.max.x, c[0]);
    aabb.max.y = Math.max(aabb.max.y, c[1]);
    aabb.max.z = Math.max(aabb.max.z, c[2]);
  }
}

// Subtle dark line material for architectural edge outlines. Matches the
// "outline pass" in Pascal's post-processing.tsx, which draws crisp edges
// over the composited GI output to give buildings a drafted feel.
const OUTLINE_MATERIAL = new THREE.LineBasicMaterial({
  color: 0x2a2a2a,
  transparent: true,
  opacity: 0.35,
  depthTest: true,
  depthWrite: false,
});

// 18° threshold angle — only mark edges between faces whose normals differ
// by ≥ 18°, so T-junctions on co-planar triangulated polygons don't emit
// visible lines. Matches the default in three.js `EdgesGeometry`.
const EDGE_THRESHOLD_DEG = 18;

function addOutline(scene, mesh) {
  if (!mesh?.geometry) return;
  const edges = new THREE.EdgesGeometry(mesh.geometry, EDGE_THRESHOLD_DEG);
  const lines = new THREE.LineSegments(edges, OUTLINE_MATERIAL);
  lines.position.copy(mesh.position);
  lines.quaternion.copy(mesh.quaternion);
  lines.scale.copy(mesh.scale);
  lines.name = 'outline';
  // Render on top of the AO-darkened mesh for a crisp drafted edge, but
  // respect depth so walls don't bleed through geometry in front of them.
  lines.renderOrder = 1;
  scene.add(lines);
}

// Walls, floors, and interior ceilings are rendered as many small coplanar
// pieces (walls_computed + extensions + stitch_walls + gap_walls + …). If
// each piece is its own mesh, adjacent pieces' vertex positions don't
// exactly match at the seams (the scan-derived corners are independently
// quantised), so at close zoom you see hairline gaps + z-fight flicker
// running down the facade. To fix that, we batch every polygon into
// per-material `BufferGeometry` lists, then at the end of
// `populateBuildingScene` we call `mergeGeometries` + `mergeVertices` with
// a 1 cm tolerance to weld close-neighbour vertices. The result is one
// continuous mesh per material — no seams, no z-fighting, smooth shared
// normals across the whole facade.
//
// `_OUTLINE_MATERIALS` still restricts outline passes to the roof, door,
// and dormer materials so the facade reads smooth and only high-contrast
// boundaries carry a drafted edge.
// Polygons below 1 cm² are producer noise (74 `within_story` gap_walls in
// the corpus are degenerate at that threshold). Rendering them adds nothing
// a human can see but contributes slivers that GTAO over-darkens.
const MIN_POLYGON_AREA_M2 = 1e-4;

// Split vertex normals where adjacent triangle face normals differ by more
// than this angle. Plain `computeVertexNormals` after `mergeVertices` averages
// the two meeting walls' face normals into a 45° diagonal at every building
// corner, which Gouraud-interpolates into the curvy per-facade shading we
// were rendering before. `toCreasedNormals` keeps smooth shading inside each
// coplanar group (room-to-room walls on the same facade differ by <2° after
// plane fits) and produces a hard crease at corners, hips, ridges, and
// eaves (≥30° between faces). 20° matches the 18° `EDGE_THRESHOLD_DEG`
// used for the outline pass, so shading creases and drafted edges agree.
const CREASE_ANGLE_RAD = THREE.MathUtils.degToRad(20);

function polygonArea3(corners) {
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < corners.length; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    nx += (a[1] - b[1]) * (a[2] + b[2]);
    ny += (a[2] - b[2]) * (a[0] + b[0]);
    nz += (a[0] - b[0]) * (a[1] + b[1]);
  }
  return 0.5 * Math.hypot(nx, ny, nz);
}

function addPoly(batches, aabb, corners, material, holes = null) {
  if (!Array.isArray(corners) || corners.length < 3) return;
  if (polygonArea3(corners) < MIN_POLYGON_AREA_M2) return;
  const geo = makePolyGeometry(corners, holes || []);
  if (!geo) return;
  expandAabb(aabb, corners);
  const bucket = batches.get(material);
  if (bucket) bucket.push(geo);
  else batches.set(material, [geo]);
}

// Weld tolerance per material, in metres. Walls and floors keep a tight
// 1 cm tolerance so we don't accidentally fuse door-reveal corners into
// adjacent wall vertices. Roof uses 3 cm — loose enough to close the
// hairline seams between adjacent-room plane fits, tight enough that real
// Y-step jogs between ceilings (≥ few cm) stay separate and render a clean
// crease under `toCreasedNormals` instead of a fused-and-bent normal ridge
// we used to hide under smooth shading at 10 cm. Interior fill
// (`structureFill`) never welds across polygons: the producer's gap
// geometry has inconsistent windings, and averaging normals across
// opposite-winding triangles drives vertex normals toward zero — the exact
// cause of the "black seam" look.
function weldTolerance(material) {
  if (material === MATERIALS.roof) return 0.03;
  if (material === MATERIALS.structureFill) return 0;
  return 0.01;
}

function flushBatches(scene, batches) {
  for (const [material, geos] of batches) {
    if (!geos.length) continue;
    let merged;
    try {
      merged = mergeGeometries(geos, false);
    } catch (_err) {
      merged = null;
    }
    if (!merged) {
      for (const g of geos) g.dispose();
      continue;
    }
    let welded;
    try {
      welded = mergeVertices(merged, weldTolerance(material));
    } catch (_err) {
      welded = merged;
    }
    // `toCreasedNormals` computes per-vertex normals and splits vertices
    // at edges where adjacent face normals differ by more than
    // `CREASE_ANGLE_RAD`, producing a new non-indexed geometry. This keeps
    // smooth shading inside each coplanar group (one per facade, one per
    // roof face) and a hard crease at corners/ridges/eaves — the thing
    // plain `computeVertexNormals` after a weld averages away.
    const creased = toCreasedNormals(welded, CREASE_ANGLE_RAD);
    // Dispose the post-mergeVertices intermediate unless it's actually
    // `merged` itself (the caught-error fallback path), which the loop
    // below disposes once.
    if (welded !== merged) welded.dispose();
    welded = creased;
    const mesh = new THREE.Mesh(welded, material);
    // Interior fill (gap walls, stitches, knee walls, closure sides) is
    // often thin vertical slivers < 30 cm wide. At 1024² shadow map over
    // 100 m × 100 m each texel covers ~10 cm, so thin fill casts aliased
    // 1-to-3-texel shadow bands across adjacent floors — exactly the dark
    // stripes users read as "black gap walls". Keep them as shadow
    // receivers so they still pick up AO-like darkening from true walls.
    mesh.castShadow = material !== MATERIALS.structureFill;
    mesh.receiveShadow = true;
    scene.add(mesh);
    if (_OUTLINE_MATERIALS.has(material)) addOutline(scene, mesh);
    // Dispose the pre-merge intermediates — their positions are now owned
    // by the merged buffer.
    for (const g of geos) {
      if (g !== merged && g !== welded) g.dispose();
    }
    if (merged !== welded) merged.dispose();
  }
}

// Placement of an oriented box on a wall plane. Mirrors
// viewer-main.js:3142:addOrientedBox.
function addOrientedBox(scene, basis, cx, cy, cz, sx, sy, sz, material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), material);
  const rot = new THREE.Matrix4().makeBasis(basis.u, basis.v, basis.n);
  mesh.quaternion.setFromRotationMatrix(rot);
  mesh.position.copy(basis.origin)
    .addScaledVector(basis.u, cx)
    .addScaledVector(basis.v, cy)
    .addScaledVector(basis.n, cz);
  // Glass panes skip shadow-casting (match viewer-main:3151).
  mesh.castShadow = material !== MATERIALS.window;
  mesh.receiveShadow = material !== MATERIALS.window;
  scene.add(mesh);
  // Outline the door slab so reveals are drafted. Skip glass — outlining
  // it just doubles the frame line visible behind it.
  if (material !== MATERIALS.window) addOutline(scene, mesh);
}

function openingBounds(corners, basis) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const c of corners) {
    const p = projectToPlane2(new THREE.Vector3(...c), basis);
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

// Pascal's WindowRenderer / DoorRenderer render each opening as a single
// mesh with one material (`.context/pascal-editor/packages/viewer/src/
// components/renderers/window/window-renderer.tsx` uses
// `DEFAULT_WINDOW_MATERIAL`, door-renderer.tsx uses `DEFAULT_DOOR_MATERIAL`).
// The elaborate frame/leaf/handle tree lives in Pascal's `DoorSystem`
// (core/src/systems/door/door-system.tsx) which drives a schema-rich node
// (width, frameThickness, segments, handle, …) we don't have from the
// reconcile scan. So we render one thin oriented box per opening at the
// opening polygon's plane, with the Pascal default material.
function addOpeningBox(scene, aabb, corners, material, depth) {
  const basis = polygonPlaneBasis(corners);
  if (!basis) return;
  const b = openingBounds(corners, basis);
  if (b.width < 0.05 || b.height < 0.05) return;
  expandAabb(aabb, corners);
  const cx = (b.minX + b.maxX) * 0.5;
  const cy = (b.minY + b.maxY) * 0.5;
  // Center the box on the wall plane (cz=0) — the wall polygon has a hole
  // at the opening so there's nothing coplanar to z-fight with.
  addOrientedBox(scene, basis, cx, cy, 0, b.width, b.height, depth, material);
}

function addWindowModel(scene, aabb, corners) {
  // Thin glass pane — 1.2 cm so it reads as flush glazing, not a solid slab.
  addOpeningBox(scene, aabb, corners, MATERIALS.window, 0.012);
}

function addDoorModel(scene, aabb, corners) {
  // 4 cm leaf — matches Pascal `DoorSystem`'s `leafDepth = 0.04`.
  addOpeningBox(scene, aabb, corners, MATERIALS.doorLeaf, 0.04);
}

// Gap segments get their material from the `type` field — the heuristic
// full-model emits `gap_floor` / `gap_ceiling` / `within_story` etc.
// `gap_ceiling` segments sit at the top of cross-storey gaps and inter-room
// voids; when the building's top storey has one, that surface is exposed to
// weather — part of the thermal envelope, not an interior finish — so it
// should read as roof (gray) rather than interior ceiling (cream).
// Vertical fill (`within_story` gap walls) goes to `structureFill` rather
// than `structure` so FrontSide culling can't blank out polygons whose
// winding flipped during the producer's interior reconstruction.
function gapMaterial(type) {
  const t = String(type || '').toLowerCase();
  if (t.includes('floor')) return MATERIALS.floor;
  if (t.includes('ceiling')) return MATERIALS.roof;
  return MATERIALS.structureFill;
}

// Floors are horizontal. Ceiling caps, however, are built from wall-top /
// sloped-ceiling snapped Y per vertex; even a few centimetres of Y change
// across a narrow gap is a real oblique roof strip, not noise. Only normalize
// ceiling caps that are flat to millimetre-level tolerance.
const FLAT_CEILING_CAP_MAX_Y_SPAN_M = 0.005;

function ySpan(corners) {
  if (!Array.isArray(corners) || !corners.length) return 0;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const c of corners) {
    minY = Math.min(minY, c[1]);
    maxY = Math.max(maxY, c[1]);
  }
  return maxY - minY;
}

function flattenToMeanY(corners) {
  if (!Array.isArray(corners) || !corners.length) return corners;
  let sum = 0;
  for (const c of corners) sum += c[1];
  const meanY = sum / corners.length;
  return corners.map((c) => [c[0], meanY, c[2]]);
}

function flattenToMaxY(corners) {
  if (!Array.isArray(corners) || !corners.length) return corners;
  let maxY = -Infinity;
  for (const c of corners) maxY = Math.max(maxY, c[1]);
  return corners.map((c) => [c[0], maxY, c[2]]);
}

// Horizontal ceiling/floor lids are produced with arbitrary vertex order
// (Shapely's exterior ring direction, earlier ear-clip output, etc.). That
// propagates into `computeVertexNormals` so a lid's normal can end up
// pointing DOWN instead of UP. The DoubleSide material still draws the
// surface, but the directional light uses `shadow.normalBias = 0.3` m —
// pushing the shadow-sample point 30 cm along the wrong-direction normal
// drops it into the floor below, so the lid self-shadows and reads as a
// solid black patch from above. Force lids to wind CW in XZ (in Three.js
// right-handed coords, cross(+X, +Z) = -Y, so CW-in-XZ gives a +Y normal).
function orientHorizontalLidUp(corners) {
  if (!Array.isArray(corners) || corners.length < 3) return corners;
  let signed2 = 0;
  for (let i = 0; i < corners.length; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    signed2 += a[0] * b[2] - b[0] * a[2];
  }
  // Newell's y-component for a horizontal polygon works out to -signed2.
  // signed2 < 0 (CW in XZ) → n.y > 0 (normal up) → keep. Else reverse.
  return signed2 < 0 ? corners : corners.slice().reverse();
}

function prepareLidCorners(corners, type) {
  const t = String(type || '').toLowerCase();
  if (t.includes('ceiling') && ySpan(corners) > FLAT_CEILING_CAP_MAX_Y_SPAN_M) {
    return corners;
  }
  return orientHorizontalLidUp(flattenToMeanY(corners));
}


// Building centroid from all wall corners — used as a fallback reference
// when a room has no floor polygon.  Mirrors viewer-main.js:2622-2627.
function computeBuildingCenter(merged) {
  const rooms = merged?.rooms || [];
  let x = 0, y = 0, z = 0, n = 0;
  for (const room of rooms) {
    for (const wall of room.walls_computed || []) {
      for (const c of wall.corners || []) {
        x += c[0]; y += c[1]; z += c[2]; n += 1;
      }
    }
  }
  if (!n) return [0, 0, 0];
  return [x / n, y / n, z / n];
}

// Room centroid from its floor polygon — a more accurate reference than
// the building center for orienting individual room walls outward, since
// each room's walls should face away from their own room interior.
function floorCentroid(floor_polygon) {
  if (!floor_polygon?.length) return null;
  let x = 0, y = 0, z = 0;
  for (const c of floor_polygon) { x += c[0]; y += c[1]; z += c[2]; }
  const n = floor_polygon.length;
  return [x / n, y / n, z / n];
}

// Populate `scene` with the Pascal-styled building geometry and return
// `{ aabb, center, radius }` so the caller can frame the orbit camera.
// Caller is responsible for disposing any previous geometry.
export function populateBuildingScene(scene, { merged }) {
  const aabb = new THREE.Box3(
    new THREE.Vector3(Infinity, Infinity, Infinity),
    new THREE.Vector3(-Infinity, -Infinity, -Infinity),
  );
  const batches = new Map();

  if (merged) {
    const bc = computeBuildingCenter(merged);

    // Walls (computed) + extension strips. Windows AND doors cut out of
    // the wall polygon so the thin opening boxes don't z-fight with a
    // solid wall plane behind them — without this, the jitter + tile
    // mosaic appears wherever an opening meets the wall.
    for (const room of merged.rooms || []) {
      // Use the room's own floor centroid as the orientation reference so
      // each room's walls face away from their own interior.  Fall back to
      // the building center for rooms without a floor polygon.
      const rc = floorCentroid(room.floor_polygon) ?? bc;
      const openings = [
        ...((room.windows || []).filter((w) => w.corners).map((w) => ({ corners: w.corners }))),
        ...((room.doors || []).filter((d) => d.corners).map((d) => ({ corners: d.corners }))),
      ];
      for (const wall of room.walls_computed || []) {
        if (!wall.corners) continue;
        const oriented = orientedStructureCorners(wall.corners, rc);
        const holes = collectWallCutoutHoles(oriented, openings);
        addPoly(batches, aabb, oriented, MATERIALS.structure, holes);
        const strips = wall.extension_strip || [];
        const stripList = Array.isArray(strips[0]?.[0]) ? strips : [strips];
        for (const quad of stripList) {
          if (Array.isArray(quad) && quad.length >= 3) {
            addPoly(batches, aabb, orientedStructureCorners(quad, rc), MATERIALS.structure);
          }
        }
      }
      // Match viewer.html's "Full model" toggle: render only walls_computed.
      // walls_merged (Apple RoomPlan raw walls) sit a few cm away from the
      // computed plane, so drawing both stacks two parallel surfaces and the
      // 1 cm weld doesn't catch them — producing horizontal-band z-fighting
      // across the facade.
      addPoly(batches, aabb, room.floor_polygon, MATERIALS.floor);
      for (const d of room.doors || []) addDoorModel(scene, aabb, d.corners);
      for (const w of room.windows || []) addWindowModel(scene, aabb, w.corners);
    }

    // Gap walls, stitches, gap closures, knee walls — interior fill
    // geometry. Vertical pieces go through `orientedStructureCorners` so
    // their front face points outwards from the building centroid (user
    // spec: vertical gap closures should face outwards). Near-horizontal lids
    // flatten to their mean Y and are re-wound so their normal points +Y (see
    // `prepareLidCorners`); sloped lids keep their snapped top profile and
    // render as oblique ceiling/roof strips. The `structureFill` bucket is FrontSide +
    // non-shadow-casting + non-welding so back-faces are hidden like real
    // walls but thin interior fill can't poison walls_computed normals.
    for (const g of merged.gap_walls || []) {
      const isHorizontal = /floor|ceiling/.test(String(g.type || '').toLowerCase());
      const corners = isHorizontal
        ? prepareLidCorners(g.corners, g.type)
        : orientedStructureCorners(g.corners, bc);
      addPoly(batches, aabb, corners, gapMaterial(g.type));
    }
    for (const g of merged.gap_closures || []) {
      const isHorizontal = /floor|ceiling/.test(String(g.type || '').toLowerCase());
      const corners = isHorizontal
        ? prepareLidCorners(g.corners, g.type)
        : orientedStructureCorners(g.corners, bc);
      addPoly(batches, aabb, corners, gapMaterial(g.type));
    }
    for (const s of merged.stitch_walls || []) {
      addPoly(batches, aabb, orientedStructureCorners(s.corners, bc), MATERIALS.structureFill);
    }
    // Cross-floor gaps can describe either a floor-side lid or a ceiling-side
    // lid. Cross-story gaps that were draped onto the lower story wall tops
    // are ceiling closures; flattening those to mean-Y draws a fake plate
    // through the occupied room volume below.
    for (const g of merged.cross_floor_gaps || []) {
      const isCrossStory = String(g.type || '').toLowerCase() === 'cross_story';
      if (isCrossStory && g.ceiling_corners) {
        addPoly(batches, aabb, orientHorizontalLidUp(flattenToMaxY(g.ceiling_corners)), MATERIALS.roof);
        continue;
      }
      if (isCrossStory && ySpan(g.corners) > FLAT_CEILING_CAP_MAX_Y_SPAN_M) {
        addPoly(batches, aabb, orientHorizontalLidUp(flattenToMaxY(g.corners)), MATERIALS.roof);
        continue;
      }
      addPoly(batches, aabb, orientHorizontalLidUp(flattenToMeanY(g.corners)), MATERIALS.floor);
      if (g.ceiling_corners) {
        addPoly(batches, aabb, prepareLidCorners(g.ceiling_corners, 'ceiling'), MATERIALS.roof);
      }
    }
    // Knee walls + dormer cheeks/headers. Plain knee walls are vertical
    // interior fill and should face outwards like other gap-closing walls.
    for (const k of merged.knee_walls || []) {
      const isDormer = String(k.kind || '').includes('dormer');
      const mat = isDormer ? MATERIALS.dormer : MATERIALS.structureFill;
      const poly = isDormer ? k.poly : orientedStructureCorners(k.poly, bc);
      addPoly(batches, aabb, poly, mat);
    }
    // Raw-scanned ceilings, clipped server-side to the XZ regions V2 pieces
    // don't cover. The 3 cm weld in the roof bucket (see `weldTolerance`)
    // merges adjacent raw-ceiling pieces with cross_floor_gap ceiling lids
    // and gap_ceiling lids; they all need the same +Y winding so the
    // subsequent `toCreasedNormals` pass sees coplanar face normals on
    // both sides of the seam (instead of 180°-opposed normals, which would
    // be split into a crease and render as a visible stripe). Without
    // this, raw-ceiling pieces wound down (46% of the corpus) fight
    // cross_floor_gap lids wound up and the overlap region reads as an
    // unclosed gap.
    for (const r of merged.raw_ceiling_fallback || []) {
      addPoly(batches, aabb, orientHorizontalLidUp(r.poly), MATERIALS.roof, r.holes || []);
    }
  }

  // Slanted ceilings — V2 final-layer pieces with dormer cutouts already
  // subtracted server-side. Share the roof material + outline with
  // raw-ceiling fallback and gap-ceiling lids.
  for (const p of (merged?.slanted_pieces || [])) {
    addPoly(batches, aabb, p.corners, MATERIALS.roof, p.holes || []);
  }

  // Merge all batches into one mesh per material — weld vertices with a
  // 1 cm tolerance so adjacent coplanar wall pieces share a single seam
  // vertex and no hairline gap / z-fight shows through at close zoom.
  flushBatches(scene, batches);

  let center = new THREE.Vector3(0, 0, 0);
  let radius = 10;
  if (Number.isFinite(aabb.min.x) && Number.isFinite(aabb.max.x)) {
    aabb.getCenter(center);
    const size = new THREE.Vector3();
    aabb.getSize(size);
    radius = Math.max(size.x, size.y, size.z, 1) * 0.6;
    // Ground plane under the building — Pascal's GroundOccluder
    // (.context/pascal-editor/packages/viewer/src/components/viewer/
    // ground-occluder.tsx). Catches cast shadows so the building grounds
    // itself visually.
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.MeshStandardMaterial({
        color: 0xfafafa, roughness: 1.0, metalness: 0, side: THREE.FrontSide,
        depthWrite: true,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(center.x, aabb.min.y - 0.05, center.z);
    ground.receiveShadow = true;
    ground.castShadow = false;
    ground.name = 'pascal-ground';
    scene.add(ground);
  }
  return { aabb, center, radius };
}

// Lighting copied from Pascal editor
// `.context/pascal-editor/packages/viewer/src/components/viewer/lights.tsx`:
// three directional lights + ambient in light mode. Intensities are much
// higher than Three.js defaults because Pascal's palette (white walls,
// light-gray slabs, medium-gray roof) is tuned for a bright, airy render.
// The caller must enable shadows on the renderer (see viewer-tiers.html).
export function addPascalLighting(scene) {
  // Intensities are halved from Pascal's light-mode values (4.0 / 0.75 / 1.0
  // / 0.5) because Pascal's scene brightness is anchored by its WebGPU SSGI
  // post-process (which adds AO + GI for contrast). Without SSGI under
  // plain WebGL the original values blow the scene out to near-white, so
  // the ratios are preserved but the absolute floor is lower.
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));

  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(10, 10, 10);
  key.castShadow = true;
  key.shadow.bias = -0.002;
  key.shadow.normalBias = 0.3;
  key.shadow.radius = 3;
  key.shadow.mapSize.set(1024, 1024);
  if ('intensity' in key.shadow) key.shadow.intensity = 0.6;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 100;
  key.shadow.camera.left = -50;
  key.shadow.camera.right = 50;
  key.shadow.camera.top = 50;
  key.shadow.camera.bottom = -50;
  scene.add(key);

  const fill1 = new THREE.DirectionalLight(0xffffff, 0.45);
  fill1.position.set(-10, 10, -10);
  scene.add(fill1);

  const fill2 = new THREE.DirectionalLight(0xffffff, 0.6);
  fill2.position.set(-10, 10, 10);
  scene.add(fill2);
}

// Remove all meshes + outline lines from the scene, keeping lights intact.
export function clearBuildingMeshes(scene) {
  const toRemove = [];
  for (const obj of scene.children) {
    if (obj.isMesh || obj.isLineSegments) toRemove.push(obj);
  }
  for (const m of toRemove) {
    if (m.geometry) m.geometry.dispose();
    scene.remove(m);
  }
}
