import { impactQuadrant, shouldShowDefect } from "./impact-threshold.js";

const LABELS = {
  "needs-look": "High impact, low confidence",
  glance: "Medium confidence",
  verified: "High confidence",
};

export function renderImpactSidebar(el, queue, { onInspect } = {}) {
  const items = queue?.items || [];
  const above = items.filter((item) => shouldShowDefect(item));
  const hidden = Math.max(0, items.length - above.length);
  el.classList.toggle("collapsed", above.length === 0);
  if (!above.length) {
    el.innerHTML = "";
    return { visibleCount: 0, hiddenCount: hidden };
  }
  el.innerHTML = `
    <header>
      <strong>Approximate elements to review</strong>
      <span>${above.length}</span>
    </header>
    <ol>
      ${above.map((item, index) => {
        const quadrant = impactQuadrant(item);
        const impact = item.impact || {};
        const delta = Number(impact.kwh_delta || 0);
        const confidence = Number(impact.confidence ?? 0);
        return `
          <li data-index="${index}" class="${quadrant}">
            <div class="impact-title"><span></span>${LABELS[quadrant]}</div>
            <div class="impact-rule">${item.rule || item.kind || "Geometry issue"}</div>
            <div class="impact-meta">${Math.round(delta)} kWh/yr · conf ${confidence.toFixed(2)}</div>
            <button type="button" data-action="inspect" data-locator="${item.locator || ""}">Inspect</button>
          </li>
        `;
      }).join("")}
    </ol>
    ${hidden ? `<footer>${hidden} below-threshold hidden</footer>` : ""}
  `;
  el.querySelectorAll("[data-action='inspect']").forEach((button) => {
    button.addEventListener("click", () => onInspect?.(button.dataset.locator));
  });
  return { visibleCount: above.length, hiddenCount: hidden };
}

