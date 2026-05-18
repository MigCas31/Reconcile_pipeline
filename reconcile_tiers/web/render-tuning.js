export const RENDER_TUNING = {
  minPolygonAreaM2: 1e-4,
  creaseAngleDeg: 20,
  edgeThresholdDeg: 18,
  weldTol: { roof: 0.005, structureFill: 0, default: 0.01 },
  opening: { minDim: 0.05, glassDepth: 0.012, doorDepth: 0.04 },
  shadow: { mapSize: 1024, halfExtent: 50, normalBias: 0.3, bias: -0.002, radius: 3 },
  light: { ambient: 0.35, key: 2.2, fill1: 0.45, fill2: 0.6 },
  ground: { size: 200, dropM: 0.05 },
};
