import * as THREE from 'three';

function newellNormal(corners, eps = 1e-10) {
  if (!corners || corners.length < 3) return null;
  const n = new THREE.Vector3();
  for (let i = 0; i < corners.length; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    n.x += (a[1] - b[1]) * (a[2] + b[2]);
    n.y += (a[2] - b[2]) * (a[0] + b[0]);
    n.z += (a[0] - b[0]) * (a[1] + b[1]);
  }
  if (n.lengthSq() < eps) return null;
  return n.normalize();
}

export function createPolygonMesh(corners, color, opacity, holes = []) {
  if (corners.length < 3) return null;

  const p0 = new THREE.Vector3(...corners[0]);
  // Newell's method: normal summed over all edges (v_i - v_{i+1}) * (v_i + v_{i+1}).
  // Robust to near-duplicate vertices and collinear runs — picking only the
  // first 3 corners breaks when Shapely intersections emit sub-cm noise near
  // p0/p1/p2 (the cross product is then dominated by FP noise, and the mesh
  // is flattened onto a wrong plane).
  const n = newellNormal(corners) ?? new THREE.Vector3(0, 1, 0);
  // Choose an in-plane ``u`` direction that's well-separated from ``n`` so
  // the UV basis is well-conditioned. Walk the corners to find the longest
  // edge projected into the plane.
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

  function dedupeLoop(loop3) {
    const out = [];
    for (const p of loop3) {
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
  }

  let outer3 = dedupeLoop(corners);
  const validHoles3 = holes
    .filter(h => h && h.length >= 3)
    .map(dedupeLoop)
    .filter(h => h.length >= 3);

  function toUV(c) {
    const p = new THREE.Vector3(c[0], c[1], c[2]).sub(p0);
    return new THREE.Vector2(p.dot(u), p.dot(v));
  }
  let contour = outer3.map(toUV);
  let holeContours = validHoles3.map(h => h.map(toUV));

  // Bow-tie repair: for quads whose UV contour self-intersects (two edges
  // cross each other), try the two alternate cyclic orderings. Earcut on a
  // self-intersecting contour silently produces overlapping triangles that
  // render as an X pattern inside the quad — see viewer screenshots on
  // wall-computed / ceiling-raw of buildings with out-of-order scan corners.
  // Leaves simple polygons untouched; for n>4 we log and skip rather than
  // attempt a larger permutation search.
  function segmentsCross(a, b, c, d) {
    const ccw = (p, q, r) => (r.y - p.y) * (q.x - p.x) - (q.y - p.y) * (r.x - p.x);
    const d1 = ccw(c, d, a), d2 = ccw(c, d, b);
    const d3 = ccw(a, b, c), d4 = ccw(a, b, d);
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
        && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
  }
  function isSimplePolygon(pts) {
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
  }
  if (!isSimplePolygon(contour)) {
    if (contour.length === 4) {
      const swap = (arr, i, k) => { const c = arr.slice(); [c[i], c[k]] = [c[k], c[i]]; return c; };
      const attempts = [[1, 2], [2, 3]];
      let repaired = null;
      for (const [i, k] of attempts) {
        const candidateUV = swap(contour, i, k);
        if (isSimplePolygon(candidateUV)) { repaired = { i, k, uv: candidateUV }; break; }
      }
      if (repaired) {
        outer3 = swap(outer3, repaired.i, repaired.k);
        contour = repaired.uv;
      } else {
        console.warn('createPolygonMesh: quad is self-intersecting in all permutations, skipping mesh');
        return null;
      }
    } else {
      console.warn('createPolygonMesh: polygon is self-intersecting (n=' + contour.length + '), skipping mesh');
      return null;
    }
  }

  // Remove collinear vertices that create zero-width arms — these arise from
  // Shapely difference operations and break ear-clip triangulation.
  function removeCollinear(pts, eps = 1e-4) {
    let changed = true;
    while (changed) {
      changed = false;
      const next = [];
      const n = pts.length;
      for (let i = 0; i < n; i++) {
        const prev = pts[(i - 1 + n) % n];
        const curr = pts[i];
        const nxt = pts[(i + 1) % n];
        const cross = (curr.x - prev.x) * (nxt.y - prev.y) - (curr.y - prev.y) * (nxt.x - prev.x);
        if (Math.abs(cross) > eps) next.push(curr);
        else changed = true;
      }
      if (next.length < 3) return pts;
      pts = next;
    }
    return pts;
  }
  contour = removeCollinear(contour);
  if (contour.length < 3) return null;

  if (!THREE.ShapeUtils.isClockWise(contour)) contour = contour.slice().reverse();
  holeContours = holeContours.map(h => THREE.ShapeUtils.isClockWise(h) ? h.slice().reverse() : h);
  // A fan-from-vertex-0 fallback here renders a giant triangle spanning the
  // polygon's bounding box whenever triangulation fails on a non-convex or
  // self-intersecting ring — far worse than rendering nothing.
  let indices;
  try {
    indices = THREE.ShapeUtils.triangulateShape(contour, holeContours);
  } catch (e) {
    console.warn('createPolygonMesh: triangulation failed, skipping mesh', e);
    return null;
  }
  if (!indices || indices.length === 0) {
    console.warn('createPolygonMesh: triangulation returned no triangles, skipping mesh');
    return null;
  }

  const flat2 = [contour, ...holeContours];
  const allUV = flat2.flat();
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
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color, opacity, transparent: true, side: THREE.DoubleSide, depthWrite: false,
  }));
}

export function createEdgeLoop(corners, color, opacity = 1) {
  const pts = corners.map(c => new THREE.Vector3(c[0], c[1], c[2]));
  pts.push(pts[0]);
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity })
  );
}

export function createLine(points, color, opacity = 1) {
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity })
  );
}

export function createPolyline3(points3, color, opacity = 1, opts = {}) {
  if (!points3 || points3.length < 2) return null;
  const pts = points3.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const mat = new THREE.LineBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    depthTest: opts.depthTest ?? false,
    depthWrite: opts.depthWrite ?? false,
  });
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
  if (opts.renderOrder !== undefined) line.renderOrder = opts.renderOrder;
  return line;
}

export function polygonPlaneBasis(corners) {
  if (!corners || corners.length < 3) return null;
  const p0 = new THREE.Vector3(...corners[0]);
  const n = newellNormal(corners);
  if (!n) return null;
  const worldUp = new THREE.Vector3(0, 1, 0);
  let v = worldUp.clone().sub(n.clone().multiplyScalar(worldUp.dot(n)));
  if (v.lengthSq() < 1e-10) v = new THREE.Vector3(1, 0, 0).sub(n.clone().multiplyScalar(n.x));
  v.normalize();
  const u = new THREE.Vector3().crossVectors(v, n).normalize();
  return { origin: p0, u, v, n };
}

export function projectToPlane2(point3, basis) {
  const p = point3.clone().sub(basis.origin);
  return { x: p.dot(basis.u), y: p.dot(basis.v) };
}

export function createTriangleMesh(triangles, color, opacity, opts = {}) {
  if (!triangles || triangles.length === 0) return null;
  const verts = [];
  const idx = [];
  let vi = 0;
  for (const tri of triangles) {
    if (!tri || tri.length !== 3) continue;
    verts.push(
      tri[0][0], tri[0][1], tri[0][2],
      tri[1][0], tri[1][1], tri[1][2],
      tri[2][0], tri[2][1], tri[2][2],
    );
    idx.push(vi, vi + 1, vi + 2);
    vi += 3;
  }
  if (idx.length === 0) return null;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color,
    opacity,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: opts.depthWrite ?? false,
    depthTest: opts.depthTest ?? true,
    polygonOffset: !!opts.polygonOffset,
    polygonOffsetFactor: opts.polygonOffsetFactor ?? 0,
    polygonOffsetUnits: opts.polygonOffsetUnits ?? 0,
  });
  const mesh = new THREE.Mesh(geo, mat);
  if (opts.renderOrder !== undefined) mesh.renderOrder = opts.renderOrder;
  return mesh;
}

export function createTriangleBoundaryEdges(triangles, color = 0xffffff, opacity = 0.9, opts = {}) {
  if (!triangles || triangles.length === 0) return null;
  const edgeMap = new Map();
  const quant = (v) => `${Math.round(v * 10000)}`;
  const keyPt = (p) => `${quant(p[0])},${quant(p[1])},${quant(p[2])}`;
  const edgeKey = (a, b) => {
    const ka = keyPt(a), kb = keyPt(b);
    return ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
  };
  for (const tri of triangles) {
    if (!tri || tri.length !== 3) continue;
    const edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
    for (const [a, b] of edges) {
      const k = edgeKey(a, b);
      const rec = edgeMap.get(k);
      if (rec) rec.count += 1;
      else edgeMap.set(k, { count: 1, a, b });
    }
  }
  const verts = [];
  for (const rec of edgeMap.values()) {
    if (rec.count !== 1) continue;
    verts.push(rec.a[0], rec.a[1], rec.a[2], rec.b[0], rec.b[1], rec.b[2]);
  }
  if (verts.length === 0) return null;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  return new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      depthTest: opts.depthTest ?? false,
      depthWrite: opts.depthWrite ?? false,
    }),
  );
}

export function disposeGroup(g) {
  while (g.children.length) {
    const c = g.children[0];
    if (c.geometry) c.geometry.dispose();
    if (c.material) c.material.dispose();
    g.remove(c);
  }
}

export function addWalls(group, walls, color, edgeColor, opacity) {
  for (const w of walls) {
    if (w.corners.length >= 3) {
      const mesh = createPolygonMesh(w.corners, color, opacity);
      if (mesh) group.add(mesh);
      group.add(createEdgeLoop(w.corners, edgeColor));
    }
  }
}

function polygonCentroid3(corners) {
  let x = 0, y = 0, z = 0;
  for (const c of corners) { x += c[0]; y += c[1]; z += c[2]; }
  const n = Math.max(1, corners.length);
  return [x / n, y / n, z / n];
}

function getProjectionAxesForCorners(corners) {
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < corners.length; i++) {
    const cur = corners[i], next = corners[(i + 1) % corners.length];
    nx += (cur[1] - next[1]) * (cur[2] + next[2]);
    ny += (cur[2] - next[2]) * (cur[0] + next[0]);
    nz += (cur[0] - next[0]) * (cur[1] + next[1]);
  }
  const anx = Math.abs(nx), any = Math.abs(ny), anz = Math.abs(nz);
  if (any >= anx && any >= anz) return [0, 2];
  if (anx >= any && anx >= anz) return [1, 2];
  return [0, 1];
}

function pointInPolygon2(point, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y;
    const xj = poly[j].x, yj = poly[j].y;
    const intersect = ((yi > point.y) !== (yj > point.y))
      && (point.x < (xj - xi) * (point.y - yi) / ((yj - yi) || 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function distancePointToSegment2(p, a, b) {
  const vx = b.x - a.x, vy = b.y - a.y;
  const wx = p.x - a.x, wy = p.y - a.y;
  const vv = vx * vx + vy * vy;
  if (vv <= 1e-12) return Math.hypot(wx, wy);
  const t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / vv));
  const px = a.x + t * vx, py = a.y + t * vy;
  return Math.hypot(p.x - px, p.y - py);
}

function wallPlane(corners) {
  if (!corners || corners.length < 3) return null;
  const a = new THREE.Vector3(...corners[0]);
  const normal = newellNormal(corners, 1e-12);
  if (!normal) return null;
  const d = -normal.dot(a);
  return { normal, d };
}

function distanceToPlane(plane, p) {
  return Math.abs(plane.normal.x * p[0] + plane.normal.y * p[1] + plane.normal.z * p[2] + plane.d);
}

export function collectWallCutoutHoles(wallCorners, openings) {
  const plane = wallPlane(wallCorners);
  if (!plane) return [];
  const [axis0, axis1] = getProjectionAxesForCorners(wallCorners);
  const outer2 = wallCorners.map(c => new THREE.Vector2(c[axis0], c[axis1]));
  const holes = [];
  const PLANE_EPS = 0.05;
  const EDGE_MARGIN = 0.01;

  for (const op of openings || []) {
    const oc = op?.corners;
    if (!oc || oc.length < 3) continue;
    if (oc.length !== 4) continue;
    if (oc.some(p => distanceToPlane(plane, p) > PLANE_EPS)) continue;
    const centroid = polygonCentroid3(oc);
    const centroid2 = new THREE.Vector2(centroid[axis0], centroid[axis1]);
    if (!pointInPolygon2(centroid2, outer2)) continue;
    const hole2 = oc.map(c => new THREE.Vector2(c[axis0], c[axis1]));
    let ok = true;
    for (const hv of hole2) {
      if (!pointInPolygon2(hv, outer2)) { ok = false; break; }
      let minDist = Infinity;
      for (let i = 0, j = outer2.length - 1; i < outer2.length; j = i++) {
        minDist = Math.min(minDist, distancePointToSegment2(hv, outer2[j], outer2[i]));
      }
      if (minDist < EDGE_MARGIN) { ok = false; break; }
    }
    if (!ok) continue;
    holes.push(oc);
  }
  return holes;
}

export function orientedStructureCorners(corners, buildingCenter) {
  if (!corners || corners.length < 3) return corners;
  const n = newellNormal(corners);
  if (!n) return corners;
  const cc = polygonCentroid3(corners);
  const out = new THREE.Vector3(cc[0] - buildingCenter[0], cc[1] - buildingCenter[1], cc[2] - buildingCenter[2]);
  if (n.dot(out) < 0) return corners.slice().reverse();
  return corners;
}
