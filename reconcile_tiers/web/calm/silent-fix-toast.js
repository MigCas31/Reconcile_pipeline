export function renderSilentFixToast(el, count) {
  if (!count) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.classList.remove("hidden");
  el.innerHTML = `<span>${count} small gaps closed automatically.</span><button type="button" aria-label="Dismiss">x</button>`;
  el.querySelector("button")?.addEventListener("click", () => el.classList.add("hidden"));
  window.setTimeout(() => el.classList.add("hidden"), 8000);
}

