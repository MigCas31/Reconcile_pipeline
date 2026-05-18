// Toggle visibility of "ceiling-class" meshes in a Three.js scene without
// depending on Three.js itself — only `scene.traverse(cb)`, `obj.isMesh`,
// `obj.material?.name`, and `obj.visible` are read/written. Lives in its
// own file so it can be unit-tested with plain object stubs.

// "Ceiling-class" materials cover the surfaces that visually close a story
// from above:
// - `roof` — solid ceilings / roof underside
// - `roofAttic` — translucent attic shell
// - `ceilingTop` — translucent flat-ceiling lid
// - `ceilingSloped` — sloped ceiling surfaces (kink ceilings, etc.)
// - `ceilingRaw` — raw_scan ceiling pieces from the scanner
// - `gapCeiling` — ceiling-kind gap fills (gap_ceiling / stitch_ceiling /
//   exterior_ceiling)
//
// Deliberately NOT included:
// - `floor` — the slab between two stories has BOTH a ceiling face (rendered
//   with a ceiling material) and a floor face (rendered with `floor`). When
//   the user hides ceilings to peek down into story N, the floor of story N
//   must stay visible — that's what they stand on. Hiding `floor` would
//   mix up the two faces of the same physical slab.
// - `gableClosure` — triangular gable-end wall sections. Vertical, not
//   overhead; they're walls, not ceilings.
// - `dormer`, `structure`, `structureFill` — walls / structural elements.
export const CEILING_MATERIAL_NAMES = new Set([
  "roof",
  "roofAttic",
  "ceilingTop",
  "ceilingSloped",
  "ceilingRaw",
  "gapCeiling",
]);
export const FLOOR_MATERIAL_NAMES = new Set(["floor"]);

export function setCeilingsVisible(scene, visible) {
  scene.traverse((obj) => {
    if (!obj.isMesh && !obj.isLineSegments) return;
    const name = obj.material?.name;
    if (FLOOR_MATERIAL_NAMES.has(name)) return; // never touch floor-class meshes
    if (CEILING_MATERIAL_NAMES.has(name)) obj.visible = visible;
  });
}
