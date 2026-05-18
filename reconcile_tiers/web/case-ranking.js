function rowClassification(row) {
  return row?.classification || row?.payload?.classification || {};
}

function rowTier(row) {
  const tier = Number(rowClassification(row).tier);
  return Number.isFinite(tier) ? tier : Number.POSITIVE_INFINITY;
}

function rowLabel(row) {
  return row?.address || row?.payload?.address || row?.uuid || "";
}

function count(value) {
  const n = Number(value || 0);
  return Number.isFinite(n) ? n : 0;
}

function plural(n, singular, pluralForm = `${singular}s`) {
  return n === 1 ? singular : pluralForm;
}

export function effectiveScanHealth(health) {
  if (!health) return null;
  const hasStructuredSignals = [
    "broken_rooms",
    "merge_errors",
    "upload_failed",
    "pipeline_broken",
    "multi_session",
  ].some((key) => Object.prototype.hasOwnProperty.call(health, key));

  const brokenRooms = count(health.broken_rooms);
  const mergeErrors = count(health.merge_errors);
  const uploadFailed = count(health.upload_failed);
  const multiSession = count(health.multi_session);
  const userStoreys = count(health.user_storeys);
  const extraSessions = userStoreys > 0
    ? Math.max(0, multiSession - userStoreys)
    : multiSession;

  const parts = [];
  if (health.pipeline_broken) parts.push("no merged.json");
  if (brokenRooms) parts.push(`${brokenRooms} broken ${plural(brokenRooms, "room")}`);
  if (mergeErrors) parts.push(`${mergeErrors} mergeRoomsError`);
  if (uploadFailed) parts.push(`${uploadFailed} scanUploadFailed`);
  if (extraSessions) {
    const context = userStoreys > 0
      ? ` (${multiSession} for ${userStoreys} user ${plural(userStoreys, "storey", "storeys")})`
      : "";
    parts.push(`${extraSessions} extra ARKit ${plural(extraSessions, "session")}${context}`);
  }

  if (!parts.length && !hasStructuredSignals && health.summary) {
    parts.push(String(health.summary));
  }

  if (!parts.length) return null;
  return {
    ...health,
    multi_session_excess: extraSessions,
    summary: parts.join(" · "),
  };
}

export function normalizeScanHealthMap(scanHealth = {}) {
  return Object.fromEntries(
    Object.entries(scanHealth)
      .map(([uuid, health]) => [uuid, effectiveScanHealth(health)])
      .filter(([, health]) => Boolean(health))
  );
}

export function hasScanWarning(row, scanHealth = {}) {
  return Boolean(row?.uuid && effectiveScanHealth(scanHealth?.[row.uuid]));
}

export function compareCaseRows(a, b, scanHealth = {}) {
  const tierDelta = rowTier(a) - rowTier(b);
  if (tierDelta) return tierDelta;

  const scanDelta = Number(hasScanWarning(a, scanHealth)) - Number(hasScanWarning(b, scanHealth));
  if (scanDelta) return scanDelta;

  const labelDelta = rowLabel(a).localeCompare(rowLabel(b), undefined, {
    numeric: true,
    sensitivity: "base",
  });
  if (labelDelta) return labelDelta;

  return String(a?.uuid || "").localeCompare(String(b?.uuid || ""));
}
