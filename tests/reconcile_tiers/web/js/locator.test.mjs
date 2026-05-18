import test from "node:test";
import assert from "node:assert/strict";

import { attachLocator, makeElementUid, parseElementUid } from "../../../../reconcile_tiers/web/locator.js";

test("locator round-trips uuid, scope, and parts", () => {
  const uid = makeElementUid("uuid-1", "wall", 0, 3);

  assert.equal(uid, "uuid-1::tier-wall::0:3");
  assert.deepEqual(parseElementUid(uid), {
    uuid: "uuid-1",
    scope: "wall",
    parts: ["0", "3"],
  });
});

test("attachLocator stores the selectable uid on userData", () => {
  const mesh = {};

  attachLocator(mesh, "uuid-1::tier-wall::0");

  assert.equal(mesh.userData.elementLocator, "uuid-1::tier-wall::0");
});
