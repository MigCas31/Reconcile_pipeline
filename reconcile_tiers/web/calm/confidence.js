/**
 * Confidence helpers for the calm viewer.
 *
 * The report's "killer feature" is per-element confidence shading
 * (low-confidence elements render translucent, signalling "approximate"
 * rather than "defect"). True per-element shading requires baking
 * confidence into vertex colors during the batched scene build in
 * tier-preview.js — that's a larger refactor than fits the v1 slice and
 * is tracked as a follow-up.
 *
 * For v1 we provide:
 *   - buildBuildingConfidence(payload): a single 0..1 score derived from
 *     ceiling.support_quality and the synthetic-wall ratio. Drives a
 *     scene-wide opacity nudge in `applyBuildingConfidence`.
 *   - indexFlagSignals({ flagQueue, audit }): a Map<elementUid, severity>
 *     for the impact sidebar to display, and reserved for the future
 *     per-element shading pass.
 */

const MIN_BUILDING_OPACITY_FACTOR = 0.85;

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

export function buildBuildingConfidence(payload) {
  const ceilings = payload?.ceiling || [];
  const support = ceilings.length
    ? ceilings.reduce((sum, p) => sum + Number(p.support_quality ?? 1), 0) / ceilings.length
    : 1;
  const walls = (payload?.rooms || []).flatMap((r) => r.walls || []);
  const synthRatio = walls.length
    ? walls.filter((w) => w.synthetic).length / walls.length
    : 0;
  return clamp01(support * (1 - synthRatio));
}

export function indexFlagSignals({ flagQueue, audit } = {}) {
  const index = new Map();
  for (const item of flagQueue?.items || []) {
    if (!item?.locator) continue;
    const sev = clamp01(Number(item.severity_score ?? item.severity ?? 0.6));
    const prev = index.get(item.locator) ?? 0;
    if (sev > prev) index.set(item.locator, sev);
  }
  for (const item of audit?.flags || []) {
    const locator = item?.locator || item?.element_uid;
    if (!locator) continue;
    const sev = clamp01(Number(item.severity ?? item.score ?? 0.7));
    const prev = index.get(locator) ?? 0;
    if (sev > prev) index.set(locator, sev);
  }
  return index;
}

/**
 * Apply a scene-wide opacity nudge proportional to building confidence.
 * Buildings with many synthetic walls or low ceiling support_quality
 * render slightly more translucent overall — a calibrated humility
 * signal at the model level. Per-element granularity is the follow-up.
 */
export function applyBuildingConfidence(scene, confidence) {
  const factor = MIN_BUILDING_OPACITY_FACTOR + (1 - MIN_BUILDING_OPACITY_FACTOR) * clamp01(confidence);
  if (factor >= 1.0) return;
  scene.traverse((obj) => {
    if (!obj.userData?.tierPreview || !obj.isMesh || obj.userData?.pickOnly) return;
    if (!obj.material || obj.userData?.calmConfidenceApplied) return;
    obj.material = obj.material.clone();
    obj.material.transparent = true;
    obj.material.opacity = clamp01((obj.material.opacity ?? 1) * factor);
    obj.userData.calmConfidenceApplied = true;
  });
}

export function applyConfidenceShading(scene, payload, _signals = {}) {
  applyBuildingConfidence(scene, buildBuildingConfidence(payload));
}

/**
 * Build a per-element confidence Map<elementUid, 0..1> combining structural
 * priors (ceiling support_quality, wall.synthetic, gap.adjacency) with the
 * live flag-queue / inspect-building severity index. Intended to be passed
 * to populateBuildingScene as `options.confidenceLookup`.
 */
export function buildPerElementConfidence(payload, signals = {}) {
  const map = new Map();
  const flagIndex = signals.flagIndex || indexFlagSignals(signals);

  for (const piece of payload?.ceiling || []) {
    if (!piece?.locator_id) continue;
    const sq = clamp01(Number(piece.support_quality ?? 1));
    if (sq < 1) map.set(piece.locator_id, sq);
  }
  for (const room of payload?.rooms || []) {
    for (const wall of room?.walls || []) {
      if (!wall?.locator_id) continue;
      if (wall.synthetic) map.set(wall.locator_id, 0.4);
    }
  }
  for (const gap of payload?.gaps || []) {
    if (!gap?.locator_id) continue;
    const adj = gap.adjacency || gap.scope || "";
    if (adj === "unknown" || gap.kind === "stitch" || (gap.kind || "").startsWith("exterior")) {
      const prev = map.get(gap.locator_id) ?? 1;
      map.set(gap.locator_id, Math.min(prev, 0.5));
    }
  }
  for (const [uid, severity] of flagIndex.entries()) {
    const prev = map.get(uid) ?? 1;
    map.set(uid, Math.min(prev, 1 - severity));
  }

  const lookup = (uid) => {
    if (!uid) return 1;
    const exact = map.get(uid);
    if (exact != null) return exact;
    const colon = uid.lastIndexOf("/");
    if (colon > -1) {
      const stem = uid.slice(0, colon);
      const stemValue = map.get(stem);
      if (stemValue != null) return stemValue;
    }
    return 1;
  };
  lookup.size = map.size;
  return lookup;
}
