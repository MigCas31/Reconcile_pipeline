import test from "node:test";
import assert from "node:assert/strict";

import { newellNormal, projectToPlane2 } from "../../../../reconcile_tiers/web/geometry.js";

test("newellNormal matches the Python +Y horizontal winding", () => {
  const normal = newellNormal([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 0, 0],
  ]);

  assert.ok(normal.y > 0);
  assert.equal(Math.round(normal.x * 1e9), 0);
  assert.equal(Math.round(normal.z * 1e9), 0);
});

test("projectToPlane2 chooses a stable longest-edge basis", () => {
  const projected = projectToPlane2([
    [0, 0, 0],
    [3, 0, 0],
    [3, 0, -1],
    [0, 0, -1],
  ]);

  assert.deepEqual(projected.map((p) => [Math.round(p.x), Math.round(p.y)]), [
    [0, 0],
    [3, 0],
    [3, 1],
    [0, 1],
  ]);
});
