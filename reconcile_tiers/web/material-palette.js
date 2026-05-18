// Pascal-derived palette. Source reference: .context/pascal-editor snapshot available in this workspace on 2026-04-26.

export const MATERIAL_PALETTE = {
  structure: { name: "structure", fill: 0xf2f0ed, roughness: 0.5, metalness: 0.0 },
  structureFill: { name: "structureFill", fill: 0xf2f0ed, roughness: 0.5, metalness: 0.0 },
  floor: { name: "floor", fill: 0xe5e5e5, roughness: 1.0, metalness: 0.0 },
  roof: { name: "roof", fill: 0xe5e5e5, roughness: 0.9, metalness: 0.0 },
  roofAttic: { name: "roofAttic", fill: 0xe5e5e5, roughness: 0.9, metalness: 0.0, opacity: 0.25 },
  ceilingTop: { name: "ceilingTop", fill: 0x999999, opacity: 0.35 },
  ceilingSloped: { name: "ceilingSloped", fill: 0xb0b8c0, opacity: 0.55, roughness: 0.9, metalness: 0.0 },
  ceilingRaw: { name: "ceilingRaw", fill: 0xc7a36f, opacity: 0.38, roughness: 0.9, metalness: 0.0 },
  gapCeiling: { name: "gapCeiling", fill: 0xaeb6bd, opacity: 0.55, roughness: 0.9, metalness: 0.0 },
  gableClosure: { name: "gableClosure", fill: 0xf2f0ed, roughness: 0.5, metalness: 0.0, opacity: 0.25 },
  doorLeaf: { name: "doorLeaf", fill: 0x8b4513, roughness: 0.7, metalness: 0.0 },
  window: { name: "window", fill: 0xadd8e6, roughness: 0.05, metalness: 0.1 },
  opening: { name: "opening", fill: 0x1a1a1a, roughness: 0.6, metalness: 0.0 },
  dormer: { name: "dormer", fill: 0xf2f0ed, roughness: 0.5, metalness: 0.0 },
  stair: { name: "stair", fill: 0xc8a878, roughness: 0.6, metalness: 0.0 },
  stairLanding: { name: "stairLanding", fill: 0xb89868, roughness: 0.6, metalness: 0.0 },
};

export const MATERIAL_BY_GAP_KIND = {
  gap_floor: MATERIAL_PALETTE.floor,
  gap_ceiling: MATERIAL_PALETTE.gapCeiling,
  side: MATERIAL_PALETTE.structure,
  stitch: MATERIAL_PALETTE.structure,
  stitch_floor: MATERIAL_PALETTE.floor,
  stitch_ceiling: MATERIAL_PALETTE.gapCeiling,
  exterior_side: MATERIAL_PALETTE.structure,
  exterior_floor: MATERIAL_PALETTE.floor,
  exterior_ceiling: MATERIAL_PALETTE.gapCeiling,
};

export function gapMaterial(kind) {
  return MATERIAL_BY_GAP_KIND[kind] ?? MATERIAL_PALETTE.structureFill;
}

// `heatedUnknown` is the bucket for `heating == null`. Calor convention:
// nil heating means "heated, type unspecified" (not unheated). The warm tan
// keeps it visually with the other heated tints rather than calling it out
// as a defect.
export const HEATING_COLORS = {
  radiators: 0xf4a261,
  floorHeating: 0xe76f51,
  radiatorsAndFloorHeating: 0xc44536,
  unheated: 0x2a9d8f,
  heatedUnknown: 0xf2cc8f,
};

export const HEATING_LABELS = {
  radiators: "Radiators",
  floorHeating: "Floor heating",
  radiatorsAndFloorHeating: "Radiators + floor heating",
  unheated: "Unheated",
  heatedUnknown: "Heated (type unknown)",
};
