import test from "node:test";
import assert from "node:assert/strict";

import {
  compareCaseRows,
  effectiveScanHealth,
  hasScanWarning,
  normalizeScanHealthMap,
} from "../../../../reconcile_tiers/web/case-ranking.js";

test("case ranking sorts by tier, then scan warning last, then label", () => {
  const scanHealth = {
    "t2-warning": { summary: "broken room" },
    "t1-warning": { summary: "merge warning" },
  };
  const rows = [
    { uuid: "t2-clean", address: "Zeta", classification: { tier: 2 } },
    { uuid: "unknown-warning", address: "Alpha" },
    { uuid: "t1-clean", address: "Beta", classification: { tier: 1 } },
    { uuid: "t1-warning", address: "Gamma", classification: { tier: 1 } },
    { uuid: "t2-warning", address: "Alpha", classification: { tier: 2 } },
  ];

  rows.sort((a, b) => compareCaseRows(a, b, scanHealth));

  assert.deepEqual(rows.map((row) => row.uuid), [
    "t1-clean",
    "t1-warning",
    "t2-clean",
    "t2-warning",
    "unknown-warning",
  ]);
});

test("scan warnings use the same truthy scan-health check as the viewer badges", () => {
  assert.equal(hasScanWarning({ uuid: "a" }, { a: { summary: "issue" } }), true);
  assert.equal(hasScanWarning({ uuid: "b" }, { a: { summary: "issue" } }), false);
});

test("scan warnings ignore expected one-session-per-user-storey counts", () => {
  const raw = {
    expected: {
      broken_rooms: 0,
      merge_errors: 0,
      upload_failed: 0,
      pipeline_broken: false,
      multi_session: 3,
      user_storeys: 3,
      summary: "3 ARKit sessions",
    },
    extra: {
      broken_rooms: 0,
      merge_errors: 0,
      upload_failed: 0,
      pipeline_broken: false,
      multi_session: 4,
      user_storeys: 3,
      summary: "4 ARKit sessions",
    },
    broken: {
      broken_rooms: 1,
      merge_errors: 0,
      upload_failed: 0,
      pipeline_broken: false,
      multi_session: 3,
      user_storeys: 3,
      summary: "1 broken room · 3 ARKit sessions",
    },
  };

  const normalized = normalizeScanHealthMap(raw);
  assert.equal(normalized.expected, undefined);
  assert.equal(normalized.extra.multi_session_excess, 1);
  assert.equal(normalized.broken.summary, "1 broken room");
  assert.equal(hasScanWarning({ uuid: "expected" }, raw), false);
  assert.equal(hasScanWarning({ uuid: "extra" }, raw), true);
  assert.equal(effectiveScanHealth(raw.extra).summary, "1 extra ARKit session (4 for 3 user storeys)");
});
