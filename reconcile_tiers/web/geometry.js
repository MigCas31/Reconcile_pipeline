// JS twin of reconcile_tiers/_core/newell.py — keep in sync.

export function newellNormal(corners, eps = 1e-10) {
  if (!corners || corners.length < 3) return null;
  let x = 0;
  let y = 0;
  let z = 0;
  for (let i = 0; i < corners.length; i += 1) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    x += (a[1] - b[1]) * (a[2] + b[2]);
    y += (a[2] - b[2]) * (a[0] + b[0]);
    z += (a[0] - b[0]) * (a[1] + b[1]);
  }
  const len = Math.hypot(x, y, z);
  if (len <= eps) return null;
  return { x: x / len, y: y / len, z: z / len };
}

export function getProjectionAxesForCorners(corners) {
  const n = newellNormal(corners);
  if (!n) return [0, 2];
  const ax = Math.abs(n.x);
  const ay = Math.abs(n.y);
  const az = Math.abs(n.z);
  if (ay >= ax && ay >= az) return [0, 2];
  if (ax >= ay && ax >= az) return [1, 2];
  return [0, 1];
}

export function pointInPolygon2(point, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;
    const intersects = ((yi > point.y) !== (yj > point.y))
      && (point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-9) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

export function distancePointToSegment2(point, a, b) {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const wx = point.x - a.x;
  const wy = point.y - a.y;
  const vv = vx * vx + vy * vy;
  if (vv <= 1e-12) return Math.hypot(wx, wy);
  const t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / vv));
  const px = a.x + t * vx;
  const py = a.y + t * vy;
  return Math.hypot(point.x - px, point.y - py);
}

function vector(a, b) {
  return { x: b[0] - a[0], y: b[1] - a[1], z: b[2] - a[2] };
}

function dot(a, b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function normalize(v) {
  const len = Math.hypot(v.x, v.y, v.z);
  if (len <= 1e-12) return { x: 1, y: 0, z: 0 };
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

export function projectToPlane2(corners) {
  if (!corners || corners.length === 0) return [];
  const n = newellNormal(corners) ?? { x: 0, y: 1, z: 0 };
  let best = { x: 1, y: 0, z: 0 };
  let bestLen = 0;
  for (let i = 0; i < corners.length; i += 1) {
    const edge = vector(corners[i], corners[(i + 1) % corners.length]);
    const alongNormal = dot(edge, n);
    const inPlane = {
      x: edge.x - n.x * alongNormal,
      y: edge.y - n.y * alongNormal,
      z: edge.z - n.z * alongNormal,
    };
    const len = dot(inPlane, inPlane);
    if (len > bestLen) {
      bestLen = len;
      best = inPlane;
    }
  }
  const u = normalize(best);
  const v = normalize(cross(n, u));
  const p0 = corners[0];
  return corners.map((point) => {
    const rel = vector(p0, point);
    return { x: dot(rel, u), y: dot(rel, v) };
  });
}

export function signedArea2(points) {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    area += a.x * b.y - b.x * a.y;
  }
  return area * 0.5;
}
