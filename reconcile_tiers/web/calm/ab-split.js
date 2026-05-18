export function maybeRenderAbSplit() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("mode") !== "ab" || params.get("embedded") === "1") return false;
  const uuid = params.get("uuid") || "";
  document.body.classList.add("ab-mode");
  document.body.innerHTML = `
    <div id="ab-shell">
      <iframe title="Expert viewer" src="/viewer.html?uuid=${encodeURIComponent(uuid)}"></iframe>
      <iframe title="Calm viewer" src="/calm-viewer?uuid=${encodeURIComponent(uuid)}&embedded=1"></iframe>
    </div>
  `;
  return true;
}

