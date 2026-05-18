export function bindExpertToggle(button, uuidProvider) {
  button.addEventListener("click", () => {
    const uuid = uuidProvider();
    if (!uuid) return;
    const target = `/reconcile_tiers/web/viewer-tiers.html#${encodeURIComponent(uuid)}`;
    window.open(target, "_blank");
  });
}
