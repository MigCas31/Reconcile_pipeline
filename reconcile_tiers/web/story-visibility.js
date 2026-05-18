export const EXTERIOR_STORY_KEY = "exterior";

export function storyKey(story) {
  return story == null ? "null" : String(story);
}

export function storySelectionTarget(selection) {
  return selection === "all" || selection == null ? null : String(selection);
}

export function isStoryVisible(story, selection, { isKey = false } = {}) {
  const target = storySelectionTarget(selection);
  if (target === null) return true;
  const key = isKey ? String(story) : storyKey(story);
  return key === target;
}

export function payloadHasExteriorStory(payload) {
  return (payload?.gaps || []).some((gap) => (
    gap?.scope === EXTERIOR_STORY_KEY
    && Array.isArray(gap?.corners)
    && gap.corners.length >= 3
  ));
}

export function payloadStoryOptions(payload) {
  const labels = Array.isArray(payload?.story_labels) ? payload.story_labels : [];
  const stories = new Set();
  for (const room of payload?.rooms || []) {
    if (Number.isFinite(room?.story)) stories.add(room.story);
  }
  for (const stair of payload?.stairs || []) {
    if (Number.isFinite(stair?.from_story)) stories.add(stair.from_story);
    if (Number.isFinite(stair?.to_story)) stories.add(stair.to_story);
  }
  for (let i = 0; i < labels.length; i += 1) stories.add(i);
  if (stories.size === 0) {
    const n = Number(payload?.classification?.n_stories);
    if (Number.isFinite(n)) {
      for (let i = 0; i < n; i += 1) stories.add(i);
    }
  }
  const options = [...stories]
    .filter(Number.isFinite)
    .sort((a, b) => a - b)
    .map((story) => ({
      value: String(story),
      label: labels[story] || `Storey ${story}`,
    }));
  if (payloadHasExteriorStory(payload)) {
    options.push({ value: EXTERIOR_STORY_KEY, label: "Exterior" });
  }
  return options;
}
