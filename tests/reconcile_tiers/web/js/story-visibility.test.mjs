import test from "node:test";
import assert from "node:assert/strict";

import {
  EXTERIOR_STORY_KEY,
  isStoryVisible,
  payloadHasExteriorStory,
  payloadStoryOptions,
  storyKey,
  storySelectionTarget,
} from "../../../../reconcile_tiers/web/story-visibility.js";

test("storyKey normalizes numbered, exterior, and null stories", () => {
  assert.equal(storyKey(0), "0");
  assert.equal(storyKey(2), "2");
  assert.equal(storyKey(EXTERIOR_STORY_KEY), "exterior");
  assert.equal(storyKey(null), "null");
  assert.equal(storyKey(undefined), "null");
});

test("storySelectionTarget treats all and null as unfiltered selections", () => {
  assert.equal(storySelectionTarget("all"), null);
  assert.equal(storySelectionTarget(null), null);
  assert.equal(storySelectionTarget(1), "1");
  assert.equal(storySelectionTarget(EXTERIOR_STORY_KEY), "exterior");
});

test("numbered storey selection hides exterior geometry", () => {
  assert.equal(isStoryVisible(1, "1"), true);
  assert.equal(isStoryVisible(0, "1"), false);
  assert.equal(isStoryVisible(EXTERIOR_STORY_KEY, "1"), false);
  assert.equal(isStoryVisible("1", "1", { isKey: true }), true);
  assert.equal(isStoryVisible(EXTERIOR_STORY_KEY, EXTERIOR_STORY_KEY), true);
});

test("all storeys selection shows every story bucket", () => {
  assert.equal(isStoryVisible(0, "all"), true);
  assert.equal(isStoryVisible(3, "all"), true);
  assert.equal(isStoryVisible(EXTERIOR_STORY_KEY, "all"), true);
  assert.equal(isStoryVisible(null, "all"), true);
});

test("payloadHasExteriorStory only counts renderable exterior gaps", () => {
  assert.equal(payloadHasExteriorStory({ gaps: [] }), false);
  assert.equal(payloadHasExteriorStory({ gaps: [{ scope: "exterior", corners: [[0], [1]] }] }), false);
  assert.equal(payloadHasExteriorStory({ gaps: [{ scope: "intra_story", corners: [[0], [1], [2]] }] }), false);
  assert.equal(payloadHasExteriorStory({ gaps: [{ scope: "exterior", corners: [[0], [1], [2]] }] }), true);
});

test("payloadStoryOptions uses actual payload story ids before classification counts", () => {
  assert.deepEqual(
    payloadStoryOptions({
      classification: { n_stories: 2 },
      story_labels: ["Ground", "First"],
      rooms: [{ story: 0 }, { story: 2 }],
      stairs: [{ from_story: 2, to_story: 3 }],
    }),
    [
      { value: "0", label: "Ground" },
      { value: "1", label: "First" },
      { value: "2", label: "Storey 2" },
      { value: "3", label: "Storey 3" },
    ],
  );
});

test("payloadStoryOptions adds exterior as an explicit selectable bucket", () => {
  assert.deepEqual(
    payloadStoryOptions({
      rooms: [{ story: 0 }],
      gaps: [{ scope: "exterior", corners: [[0], [1], [2]] }],
    }),
    [
      { value: "0", label: "Storey 0" },
      { value: "exterior", label: "Exterior" },
    ],
  );
});
