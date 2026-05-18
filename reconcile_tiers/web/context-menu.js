// Right-click context menu for the tier viewer.
//
// Two surfaces:
//   1. Element-scoped quick actions (per tier scope) — pure, no LLM.
//   2. Building-scoped visibility / screenshot helpers.
//   3. (Gated by ?gemini=1) free-form Gemini chat at the bottom.
//
// The element-scoped buttons hit `POST /context-action` with names from
// `reconcile_tiers/quick_actions/REGISTRY`. Building-scoped operations
// (hide/show/screenshot) are pure client-side and don't touch the server.

import * as THREE from "three";

import { parseElementUid } from "./locator.js";

// ---- DOM lookup --------------------------------------------------------------

const root = document.querySelector("#context-menu");
const targetEl = document.querySelector("#context-menu-target");
const actionsEl = document.querySelector("#context-menu-actions");
const infoEl = document.querySelector("#context-menu-info");
const geminiPanel = document.querySelector("#context-menu-gemini");
const geminiPrompt = document.querySelector("#context-menu-prompt");
const geminiOutput = document.querySelector("#context-menu-gemini-output");
const geminiSendBtn = document.querySelector("#context-menu-gemini-send");
const resetBtn = document.querySelector("#context-menu-reset");
const closeBtn = document.querySelector("#context-menu-close");

const GEMINI_ENABLED = /[?&]gemini=1\b/.test(window.location.search);

// ---- preview overlay state ---------------------------------------------------
//
// previewState: locator -> { mesh, overlay }
//   mesh    — the original mesh from `scene.userData.tierLocatorMap`
//   overlay — Three.Object3D added to scene to show the previewed shape
//             (null for delete-style previews where we just hide).

const previewState = new Map();

const PREVIEW_FILL = new THREE.MeshBasicMaterial({
  color: 0xffcc00,
  transparent: true,
  opacity: 0.45,
  side: THREE.DoubleSide,
  depthWrite: false,
});
const PREVIEW_EDGE = new THREE.LineBasicMaterial({
  color: 0xb37b00,
  depthTest: true,
  depthWrite: false,
});

function makePreviewOverlay(corners) {
  const group = new THREE.Group();
  group.userData.tierPreview = true;
  if (!Array.isArray(corners) || corners.length < 3) return group;

  // triangle fan from corner[0]
  const positions = [];
  const v0 = corners[0];
  for (let i = 1; i < corners.length - 1; i += 1) {
    const a = corners[i];
    const b = corners[i + 1];
    positions.push(v0.x, v0.y, v0.z, a.x, a.y, a.z, b.x, b.y, b.z);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geom.computeVertexNormals();
  const mesh = new THREE.Mesh(geom, PREVIEW_FILL);
  group.add(mesh);

  // outline
  const edgePositions = [];
  for (let i = 0; i < corners.length; i += 1) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    edgePositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(edgePositions, 3),
  );
  group.add(new THREE.LineSegments(edgeGeom, PREVIEW_EDGE));
  return group;
}

function applyPreview(scene, locator, result) {
  const mesh = scene.userData.tierLocatorMap?.get(locator);
  if (!mesh) return false;
  // Cache prior preview if any so toggling is reversible.
  releasePreview(scene, locator, /*hideOriginal=*/ true);

  let overlay = null;
  if (Array.isArray(result?.corners) && result.corners.length >= 3) {
    overlay = makePreviewOverlay(result.corners);
    scene.add(overlay);
  }
  mesh.visible = false;
  previewState.set(locator, { mesh, overlay });
  updateResetButton();
  return true;
}

function releasePreview(scene, locator, hideOriginal) {
  const prior = previewState.get(locator);
  if (!prior) return;
  if (prior.overlay) {
    prior.overlay.removeFromParent();
    prior.overlay.traverse?.((obj) => {
      obj.geometry?.dispose?.();
    });
  }
  if (!hideOriginal && prior.mesh) prior.mesh.visible = true;
  previewState.delete(locator);
  updateResetButton();
}

function resetAllPreviews(scene) {
  for (const [locator] of [...previewState]) {
    releasePreview(scene, locator, /*hideOriginal=*/ false);
  }
}

function updateResetButton() {
  if (!resetBtn) return;
  resetBtn.classList.toggle("hidden", previewState.size === 0);
}

// ---- visibility helpers (building-scoped) -----------------------------------

function meshScope(mesh) {
  const uid = mesh?.userData?.elementLocator;
  if (!uid) return null;
  return parseElementUid(uid)?.scope ?? null;
}

function scopeFamily(scope) {
  if (!scope) return null;
  if (scope === "wall" || scope.startsWith("wall-")) return "walls";
  if (scope.startsWith("ceiling")) return "ceilings";
  if (scope.startsWith("gap")) return "gaps";
  if (scope === "knee-wall" || scope === "dormer-face" || scope === "gable-closure")
    return "roof";
  if (scope === "room") return "rooms";
  return null;
}

const FAMILY_LABELS = {
  walls: "walls",
  ceilings: "ceilings",
  gaps: "gaps",
  roof: "roof / dormers / knee walls",
  rooms: "rooms",
};

function setFamilyVisible(scene, family, visible) {
  scene.userData.tierLocatorMap?.forEach((mesh) => {
    const fam = scopeFamily(meshScope(mesh));
    if (fam === family) {
      // Don't trample preview-hidden meshes.
      if (previewState.has(mesh.userData.elementLocator)) return;
      mesh.visible = visible;
    }
  });
}

function setOnlyFamilyVisible(scene, family) {
  scene.userData.tierLocatorMap?.forEach((mesh) => {
    const fam = scopeFamily(meshScope(mesh));
    if (previewState.has(mesh.userData.elementLocator)) return;
    mesh.visible = fam === family;
  });
}

function resetAllVisibility(scene) {
  scene.userData.tierLocatorMap?.forEach((mesh) => {
    if (previewState.has(mesh.userData.elementLocator)) return;
    mesh.visible = true;
  });
}

function takeScreenshot(uuid) {
  const canvas = document.querySelector("#view");
  if (!canvas) return;
  const url = canvas.toDataURL("image/png");
  const link = document.createElement("a");
  link.href = url;
  link.download = `${uuid || "viewer"}-${Date.now()}.png`;
  link.click();
}

// ---- network helpers ---------------------------------------------------------

async function callContextAction(action, params) {
  const resp = await fetch("/context-action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, params }),
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(body?.error || `${action} failed (${resp.status})`);
  }
  return body.result;
}

async function callGemini(prompt, targets) {
  const resp = await fetch("/gemini/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, targets }),
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(body?.error || `gemini chat failed (${resp.status})`);
  }
  return body;
}

// ---- menu rendering ----------------------------------------------------------

function clearActions() {
  actionsEl.textContent = "";
  infoEl.classList.add("hidden");
  infoEl.textContent = "";
}

function addAction(label, handler, { primary = false } = {}) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = label;
  if (primary) btn.classList.add("primary");
  btn.addEventListener("click", async (event) => {
    event.preventDefault();
    btn.disabled = true;
    try {
      await handler();
    } catch (err) {
      console.error("[context-menu] action failed", err);
      showInfo(String(err?.message || err));
    } finally {
      btn.disabled = false;
    }
  });
  actionsEl.appendChild(btn);
  return btn;
}

function showInfo(text) {
  infoEl.textContent = text;
  infoEl.classList.remove("hidden");
}

function showInfoJSON(obj) {
  infoEl.textContent = JSON.stringify(obj, null, 2);
  infoEl.classList.remove("hidden");
}

function positionMenu(screenX, screenY) {
  // The menu is absolutely positioned inside <main>, while mouse events are
  // reported in viewport coordinates. Clamp in the same coordinate space as
  // the positioned element so the sidebar cannot push the menu out of view.
  const margin = 8;
  const container = root.offsetParent || document.documentElement;
  const rect = container.getBoundingClientRect();
  const width = root.offsetWidth || 320;
  const height = root.offsetHeight || 320;
  const maxX = Math.max(margin, rect.width - width - margin);
  const maxY = Math.max(margin, rect.height - height - margin);
  const x = screenX - rect.left;
  const y = screenY - rect.top;
  root.style.left = `${Math.min(Math.max(margin, x), maxX)}px`;
  root.style.top = `${Math.min(Math.max(margin, y), maxY)}px`;
}

// ---- public open() -----------------------------------------------------------

let activeCtx = null;

function focusMesh(scene, locator, opts = {}) {
  const select = scene.userData.selectLocator;
  if (typeof select === "function") {
    select(locator, { focus: true, ...opts });
  }
}

function buildElementMenu(ctx) {
  const targets = ctx.targets;
  targetEl.textContent =
    targets.length === 1
      ? targets[0]
      : `${targets.length} elements selected`;

  const single = targets.length === 1 ? targets[0] : null;
  const parsed = single ? parseElementUid(single) : null;
  const scope = parsed?.scope || null;

  // Per-scope preview actions
  if (scope?.startsWith("ceiling-flat")) {
    addAction("Preview as slanted (20°)", async () => {
      const result = await callContextAction("preview_make_slanted", {
        locator: single,
        slope_deg: 20,
        azimuth_deg: 0,
      });
      applyPreview(ctx.scene, single, result);
      ctx.requestRender();
    });
    addAction("Preview delete", async () => {
      const result = await callContextAction("preview_delete", {
        locator: single,
      });
      applyPreview(ctx.scene, single, result);
      ctx.requestRender();
    });
  } else if (scope?.startsWith("ceiling-slanted")) {
    addAction("Preview as flat", async () => {
      const result = await callContextAction("preview_make_flat", {
        locator: single,
      });
      applyPreview(ctx.scene, single, result);
      ctx.requestRender();
    });
    addAction("Preview delete", async () => {
      const result = await callContextAction("preview_delete", {
        locator: single,
      });
      applyPreview(ctx.scene, single, result);
      ctx.requestRender();
    });
  } else if (
    scope === "knee-wall" ||
    scope === "dormer-face" ||
    scope === "gable-closure"
  ) {
    addAction("Preview delete", async () => {
      const result = await callContextAction("preview_delete", {
        locator: single,
      });
      applyPreview(ctx.scene, single, result);
      ctx.requestRender();
    });
  } else if (scope?.startsWith("gap")) {
    addAction("Toggle inclusion (preview)", async () => {
      const result = await callContextAction("preview_toggle_gap", {
        locator: single,
      });
      applyPreview(ctx.scene, single, result);
      ctx.requestRender();
    });
  }

  // Universal actions
  if (single) {
    addAction("Show info", async () => {
      const info = await callContextAction("element_info", { locator: single });
      showInfoJSON(info);
    });
    addAction("Show neighbors", async () => {
      const adj = await callContextAction("neighbors", { locator: single });
      const targetMap = ctx.scene.userData.tierLocatorMap;
      let highlit = 0;
      adj.neighbors?.forEach((loc) => {
        if (targetMap?.has(loc)) {
          focusMesh(ctx.scene, loc);
          highlit += 1;
        }
      });
      showInfo(
        `${adj.neighbors?.length || 0} neighbor(s); ` +
          `${highlit} present in current scene`,
      );
    });
    addAction("Focus camera", async () => {
      focusMesh(ctx.scene, single);
    });
    addAction("Hide", async () => {
      const mesh = ctx.scene.userData.tierLocatorMap?.get(single);
      if (mesh) {
        mesh.visible = false;
        ctx.requestRender();
      }
    });
    addAction("Copy ID", async () => {
      try {
        await navigator.clipboard.writeText(single);
        showInfo("Copied to clipboard.");
      } catch (err) {
        showInfo("Copy failed: " + (err?.message || err));
      }
    });
  } else if (targets.length > 1) {
    addAction("Copy IDs", async () => {
      const text = targets.join("\n");
      try {
        await navigator.clipboard.writeText(text);
        showInfo(`Copied ${targets.length} IDs.`);
      } catch (err) {
        showInfo("Copy failed: " + (err?.message || err));
      }
    });
    addAction("Hide all selected", async () => {
      let n = 0;
      for (const loc of targets) {
        const mesh = ctx.scene.userData.tierLocatorMap?.get(loc);
        if (mesh) {
          mesh.visible = false;
          n += 1;
        }
      }
      ctx.requestRender();
      showInfo(`Hid ${n} element(s).`);
    });
  }
}

function buildBuildingMenu(ctx) {
  targetEl.textContent = ctx.uuid
    ? `Building ${ctx.uuid}`
    : "(no building loaded)";
  actionsEl.classList.add("full");

  const families = ["walls", "ceilings", "gaps", "roof"];
  for (const family of families) {
    addAction(`Hide all ${FAMILY_LABELS[family]}`, async () => {
      setFamilyVisible(ctx.scene, family, false);
      ctx.requestRender();
    });
  }
  for (const family of families) {
    addAction(`Show only ${FAMILY_LABELS[family]}`, async () => {
      setOnlyFamilyVisible(ctx.scene, family);
      ctx.requestRender();
    });
  }
  addAction("Reset visibility", async () => {
    resetAllVisibility(ctx.scene);
    ctx.requestRender();
  });
  addAction("Take screenshot", async () => {
    takeScreenshot(ctx.uuid);
  });
  if (ctx.uuid) {
    addAction("Copy building UUID", async () => {
      try {
        await navigator.clipboard.writeText(ctx.uuid);
        showInfo(`Copied ${ctx.uuid}.`);
      } catch (err) {
        showInfo("Copy failed: " + (err?.message || err));
      }
    });
  }
}

function buildGeminiPanel(ctx) {
  if (!GEMINI_ENABLED) {
    geminiPanel.classList.add("hidden");
    return;
  }
  geminiPanel.classList.remove("hidden");
  geminiPrompt.value = "";
  geminiOutput.textContent = "";

  const handler = async () => {
    const prompt = geminiPrompt.value.trim();
    if (!prompt) return;
    geminiSendBtn.disabled = true;
    geminiOutput.textContent = "Thinking…";
    try {
      const body = await callGemini(prompt, ctx.targets);
      const calls = (body.tool_calls || [])
        .map(
          (c) =>
            `<div class="ctx-call">→ ${c.name}(${JSON.stringify(c.args || {})})</div>`,
        )
        .join("");
      geminiOutput.innerHTML =
        calls + `<div>${escapeHtml(body.final_text || "")}</div>`;
    } catch (err) {
      geminiOutput.textContent = String(err?.message || err);
    } finally {
      geminiSendBtn.disabled = false;
    }
  };

  geminiSendBtn.onclick = handler;
  geminiPrompt.onkeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handler();
    }
  };
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[
      char
    ]),
  );
}

export function openContextMenu(scope, ctx) {
  activeCtx = ctx;
  clearActions();
  actionsEl.classList.remove("full");

  if (scope === "element") {
    buildElementMenu(ctx);
  } else {
    buildBuildingMenu(ctx);
  }

  root.classList.remove("hidden");
  buildGeminiPanel(ctx);
  positionMenu(ctx.x ?? 100, ctx.y ?? 100);
}

export function closeContextMenu() {
  root.classList.add("hidden");
  activeCtx = null;
}

export function hasActivePreviews() {
  return previewState.size > 0;
}

export function resetPreviews(scene) {
  resetAllPreviews(scene);
  if (scene?.userData?.requestRender) scene.userData.requestRender();
}

// ---- global wiring -----------------------------------------------------------

closeBtn?.addEventListener("click", () => closeContextMenu());
resetBtn?.addEventListener("click", () => {
  if (activeCtx) {
    resetAllPreviews(activeCtx.scene);
    activeCtx.requestRender();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !root.classList.contains("hidden")) {
    closeContextMenu();
  }
});

document.addEventListener(
  "mousedown",
  (event) => {
    if (root.classList.contains("hidden")) return;
    if (root.contains(event.target)) return;
    // Don't dismiss when clicking the canvas with shift/meta/ctrl — those
    // are reserved for the existing flag / multi-select shortcuts.
    closeContextMenu();
  },
  true,
);

// On building swap, drop preview overlays so they don't leak across loads.
window.addEventListener("tier:building-loaded", () => {
  if (activeCtx?.scene) resetAllPreviews(activeCtx.scene);
});
