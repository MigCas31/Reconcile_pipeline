import test from "node:test";
import assert from "node:assert/strict";

import { MATERIAL_PALETTE, gapMaterial } from "../../../../reconcile_tiers/web/material-palette.js";

test("gapMaterial uses typed gap kinds", () => {
  // Ceiling-kind gaps render with the dedicated translucent gapCeiling
  // material so the hide-ceilings toggle catches them.
  assert.equal(gapMaterial("gap_ceiling"), MATERIAL_PALETTE.gapCeiling);
  assert.equal(gapMaterial("stitch_ceiling"), MATERIAL_PALETTE.gapCeiling);
  assert.equal(gapMaterial("exterior_ceiling"), MATERIAL_PALETTE.gapCeiling);
  assert.equal(gapMaterial("gap_floor"), MATERIAL_PALETTE.floor);
  assert.equal(gapMaterial("side"), MATERIAL_PALETTE.structure);
  assert.equal(gapMaterial("stitch"), MATERIAL_PALETTE.structure);
  assert.equal(gapMaterial("exterior_side"), MATERIAL_PALETTE.structure);
});

test("unknown gap kind falls back to structure fill", () => {
  assert.equal(gapMaterial("unknown"), MATERIAL_PALETTE.structureFill);
});
