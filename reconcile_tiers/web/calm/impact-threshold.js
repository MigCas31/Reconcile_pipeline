export const DEFAULT_IMPACT_POLICY = {
  kwhThreshold: 100,
  pctThreshold: 0.01,
};

export function shouldShowDefect(item, policy = DEFAULT_IMPACT_POLICY) {
  const impact = item?.impact;
  if (!impact) return item?.severity === "high";
  return Math.abs(Number(impact.kwh_delta || 0)) > policy.kwhThreshold
    || Number(impact.pct_of_total || 0) > policy.pctThreshold;
}

export function impactQuadrant(item) {
  const confidence = Number(item?.impact?.confidence ?? 0);
  if (confidence < 0.6) return "needs-look";
  if (confidence < 0.85) return "glance";
  return "verified";
}

