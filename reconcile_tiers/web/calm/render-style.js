import * as THREE from "three";

import { addCalmLighting, addPascalLighting, populateBuildingScene } from "../tier-preview.js";
import { applySketchyEdges } from "./sketchy-edges.js";

export const CALM_BACKGROUND = new THREE.Color(0xf4f1ec);
export const AUDIT_BACKGROUND = new THREE.Color(0x1a1a1a);
export const EXPORT_BACKGROUND = new THREE.Color(0xfafafa);

const CONTRACT_LABELS = {
  browse: { primary: "Schematic reconstruction", secondary: "· approximately LOD 200 · scan-derived" },
  audit:  { primary: "Detailed audit view",       secondary: "· LOD 300 · photoreal · review approximate elements" },
  source: { primary: "Raw scan source",            secondary: "· LOD 100 · pre-reconstruction twin" },
  export: { primary: "Schematic reconstruction", secondary: "· approximately LOD 200 · for presentation" },
};

function clearLighting(scene) {
  const toRemove = [];
  scene.traverse((obj) => {
    if (obj.isLight) toRemove.push(obj);
  });
  toRemove.forEach((light) => {
    light.parent?.remove(light);
    light.dispose?.();
  });
  delete scene.userData.tierLighting;
  delete scene.userData.calmLighting;
}

function applyLighting(scene, mode) {
  clearLighting(scene);
  if (mode === "audit") addPascalLighting(scene);
  else addCalmLighting(scene);
}

function applyBackground(scene, renderer, mode) {
  const bg = mode === "audit" ? AUDIT_BACKGROUND : mode === "export" ? EXPORT_BACKGROUND : CALM_BACKGROUND;
  scene.background = bg;
  if (renderer) {
    renderer.toneMapping = mode === "audit" ? THREE.ACESFilmicToneMapping : THREE.LinearToneMapping;
    renderer.toneMappingExposure = mode === "audit" ? 0.9 : 1.0;
  }
}

function applyContractLabel(mode) {
  const el = document.querySelector("#contract-label");
  if (!el) return;
  const labels = CONTRACT_LABELS[mode] ?? CONTRACT_LABELS.browse;
  el.querySelector("strong").textContent = labels.primary;
  el.querySelector("span").textContent = labels.secondary;
}

function applyModePills(mode) {
  document.querySelectorAll("#mode-pills button").forEach((button) => {
    const isActive = button.dataset.mode === mode;
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });
}

/**
 * Switch the viewer between Browse / Audit / Source / Export modes.
 *
 * Re-runs populateBuildingScene with the appropriate `style`, swaps the
 * lighting rig and background, refreshes the contract label, and applies
 * per-element confidence shading. The renderer object is kept (we do not
 * tear down the WebGLRenderer between mode swaps); only scene contents
 * are rebuilt.
 *
 * The Source-mode twin overlay is layered on top by the caller after this
 * function returns, since it requires a separate JSON fetch.
 */
export function applyStyle(scene, renderer, mode, payload, signals = {}) {
  const style = mode === "audit" ? "tiers" : "calm";
  const confidenceLookup = mode === "audit" ? null : signals.confidenceLookup;
  populateBuildingScene(scene, payload, { style, confidenceLookup });
  applyLighting(scene, mode);
  applyBackground(scene, renderer, mode);

  if (mode !== "audit") {
    applySketchyEdges(scene);
  }

  if (mode === "source") {
    fadeBaseForSource(scene);
  }

  document.body.dataset.mode = mode;
  applyContractLabel(mode);
  applyModePills(mode);
}

function fadeBaseForSource(scene) {
  scene.traverse((obj) => {
    if (!obj.userData?.tierPreview || obj.userData?.pickOnly) return;
    if (!obj.material || obj.userData?.calmTwinOverlay) return;
    obj.material = obj.material.clone();
    obj.material.transparent = true;
    obj.material.opacity = (obj.material.opacity ?? 1) * 0.35;
  });
}
