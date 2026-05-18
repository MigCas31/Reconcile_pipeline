import test from "node:test";
import assert from "node:assert/strict";

import { setCeilingsVisible } from "../../../../reconcile_tiers/web/ceiling-visibility.js";

// Minimal stand-in for a THREE scene: anything with a `traverse(cb)` method
// that walks every "mesh" the function might inspect. The function only
// reads `obj.isMesh`, `obj.material?.name`, and writes `obj.visible`.
function fakeScene(meshes) {
  return {
    traverse(cb) {
      for (const m of meshes) cb(m);
    },
  };
}

function fakeMesh(materialName) {
  return { isMesh: true, material: { name: materialName }, visible: true };
}

const CEILING_MATERIALS = [
  "roof",
  "roofAttic",
  "ceilingTop",
  "ceilingSloped",
  "ceilingRaw",
  "gapCeiling",
];

test("setCeilingsVisible(false) hides every ceiling-class material", () => {
  const meshes = CEILING_MATERIALS.map(fakeMesh);
  const scene = fakeScene(meshes);
  setCeilingsVisible(scene, false);
  for (const m of meshes) assert.equal(m.visible, false, `${m.material.name} should be hidden`);
});

test("setCeilingsVisible(true) restores every ceiling-class material", () => {
  const meshes = CEILING_MATERIALS.map(fakeMesh);
  for (const m of meshes) m.visible = false;
  const scene = fakeScene(meshes);
  setCeilingsVisible(scene, true);
  for (const m of meshes) assert.equal(m.visible, true, `${m.material.name} should be visible`);
});

test("setCeilingsVisible never touches floor-class meshes", () => {
  // The slab between two stories has both a ceiling face (rendered with a
  // ceiling material) and a floor face (rendered with `floor`). Hiding
  // ceilings must keep the `floor` face visible — the user stands on it.
  const floorMesh = fakeMesh("floor");
  const ceilingMesh = fakeMesh("roof");
  const scene = fakeScene([floorMesh, ceilingMesh]);
  setCeilingsVisible(scene, false);
  assert.equal(floorMesh.visible, true, "floor must remain visible after hide-ceilings");
  assert.equal(ceilingMesh.visible, false, "ceiling should be hidden");
});

test("setCeilingsVisible never touches walls / structural elements", () => {
  const walls = [
    fakeMesh("structure"),
    fakeMesh("structureFill"),
    fakeMesh("dormer"),
    fakeMesh("gableClosure"), // triangular gable-end wall — a wall, not a ceiling
    fakeMesh("doorLeaf"),
    fakeMesh("window"),
    fakeMesh("opening"),
    fakeMesh("stair"),
    fakeMesh("stairLanding"),
  ];
  const scene = fakeScene(walls);
  setCeilingsVisible(scene, false);
  for (const m of walls) assert.equal(m.visible, true, `${m.material.name} should not be hidden by hide-ceilings`);
});
