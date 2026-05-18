export const CALM_MATERIAL_PALETTE = {
  structure: { name: "structure", fill: 0xe5e0d5, opacity: 1.0 },
  structureFill: { name: "structureFill", fill: 0xe5e0d5, opacity: 0.85 },
  floor: { name: "floor", fill: 0xd9d2c4, opacity: 1.0 },
  roof: { name: "roof", fill: 0xe5e5e5, opacity: 1.0 },
  roofAttic: { name: "roofAttic", fill: 0xe5e5e5, opacity: 0.25 },
  ceilingTop: { name: "ceilingTop", fill: 0x999999, opacity: 0.35 },
  ceilingSloped: { name: "ceilingSloped", fill: 0xb8bec4, opacity: 0.55 },
  ceilingRaw: { name: "ceilingRaw", fill: 0xc7a36f, opacity: 0.38 },
  gapCeiling: { name: "gapCeiling", fill: 0xb8bec4, opacity: 0.55 },
  gableClosure: { name: "gableClosure", fill: 0xe5e0d5, opacity: 0.35 },
  doorLeaf: { name: "doorLeaf", fill: 0x7c5a3f, opacity: 0.85 },
  window: { name: "window", fill: 0xa9c9d9, opacity: 0.4 },
  opening: { name: "opening", fill: 0xc75d5d, opacity: 0.4 },
  dormer: { name: "dormer", fill: 0xe5e0d5, opacity: 1.0 },
  stair: { name: "stair", fill: 0xc8a878, opacity: 1.0 },
  stairLanding: { name: "stairLanding", fill: 0xb89868, opacity: 1.0 },
};

export const CALM_MATERIAL_BY_GAP_KIND = {
  gap_floor: CALM_MATERIAL_PALETTE.floor,
  gap_ceiling: CALM_MATERIAL_PALETTE.gapCeiling,
  side: CALM_MATERIAL_PALETTE.structureFill,
  stitch: CALM_MATERIAL_PALETTE.structureFill,
  stitch_floor: CALM_MATERIAL_PALETTE.floor,
  stitch_ceiling: CALM_MATERIAL_PALETTE.gapCeiling,
  exterior_side: CALM_MATERIAL_PALETTE.structureFill,
  exterior_floor: CALM_MATERIAL_PALETTE.floor,
  exterior_ceiling: CALM_MATERIAL_PALETTE.gapCeiling,
};

export function calmGapMaterial(kind) {
  return CALM_MATERIAL_BY_GAP_KIND[kind] ?? CALM_MATERIAL_PALETTE.structureFill;
}
