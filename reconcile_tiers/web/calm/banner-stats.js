export function renderBannerStats(el, energy, confidence) {
  const kwh = Number(energy?.annual_kwh_proxy);
  const kwhText = Number.isFinite(kwh) ? `${Math.round(kwh).toLocaleString()} kWh/yr` : "-";
  const confidenceText = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : "-";
  el.innerHTML = `
    <div><span>Estimated heat loss</span><strong>${kwhText}</strong></div>
    <div><span>Geometry confidence</span><strong>${confidenceText}</strong></div>
  `;
}

