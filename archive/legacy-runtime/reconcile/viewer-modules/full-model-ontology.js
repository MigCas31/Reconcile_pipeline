function addFace({
  corners,
  holes = [],
  color,
  opacity,
  group,
  createPolygonMesh,
  createEdgeLoop,
  attachLocator,
  locator,
  renderOrder = 5,
  edgeOpacity = null,
  depthTest = true,
  depthWrite = false,
  polygonOffset = false,
  polygonOffsetFactor = 0,
  polygonOffsetUnits = 0,
}) {
  if (!Array.isArray(corners) || corners.length < 3) return false;
  const mesh = createPolygonMesh(corners, color, opacity, holes);
  if (mesh) {
    mesh.renderOrder = renderOrder;
    mesh.material.depthTest = depthTest;
    mesh.material.depthWrite = depthWrite;
    mesh.material.transparent = opacity < 0.999;
    mesh.material.polygonOffset = polygonOffset;
    mesh.material.polygonOffsetFactor = polygonOffsetFactor;
    mesh.material.polygonOffsetUnits = polygonOffsetUnits;
    if (attachLocator && locator) attachLocator(mesh, locator);
    group.add(mesh);
  }
  group.add(createEdgeLoop(corners, color, edgeOpacity ?? Math.min(opacity + 0.1, 0.95)));
  return true;
}

function surfaceStyle(surface) {
  const category = String(surface?.category || '');
  if (category === 'exterior_roof') return { color: 0x9ec5e3, opacity: 0.42, renderOrder: 6, edgeOpacity: 0.55 };
  // The heuristic full-model already renders walls / floors / windows / doors
  // with gap closures, wall extensions, and 3D window & door detail — the
  // ontology base_* surfaces are simplified duplicates that lose those details,
  // so the heuristic owns the base shell. Ceiling overlays (base_room_ceiling,
  // room_ceiling_sloped, room_ceiling_flat) are intentionally hidden here: the
  // full-model view shows only roofs, knee walls, and attic floors on top.
  if (category === 'base_exterior_wall') return null;
  if (category === 'base_interior_wall') return null;
  if (category === 'base_room_floor') return null;
  if (category === 'base_room_ceiling') return null;
  if (category === 'fallback_room_ceiling') return null;
  if (category === 'base_window') return null;
  if (category === 'base_door') return null;
  if (category === 'base_opening') return null;
  if (category === 'exterior_wall') return null;
  if (category === 'occupied_room_wall') return null;
  if (category === 'occupied_room_floor') return null;
  if (category === 'occupied_room_ceiling') return null;
  if (category === 'room_ceiling_sloped') return null;
  if (category === 'attic_floor') return {
    color: 0x7da8d6, opacity: 0.82, renderOrder: 8, edgeOpacity: 0.9,
    polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3,
  };
  if (category === 'room_ceiling_flat') return null;
  if (category === 'knee_wall') return { color: 0xe3bf96, opacity: 0.84, renderOrder: 5 };
  if (category === 'unresolved_region') return null;
  return null;
}

function diffSurfaceStyle(surface) {
  const category = String(surface?.category || '');
  if (category === 'exterior_roof') return { color: 0x22d3ee, opacity: 0.82, renderOrder: 8, edgeOpacity: 0.95 };
  if (category === 'base_exterior_wall') return { color: 0x34d399, opacity: 0.48, renderOrder: 7, edgeOpacity: 0.9 };
  if (category === 'base_interior_wall') return { color: 0xf59e0b, opacity: 0.28, renderOrder: 6, edgeOpacity: 0.76 };
  if (category === 'base_room_floor') return { color: 0xfbbf24, opacity: 0.36, renderOrder: 6, edgeOpacity: 0.78 };
  if (category === 'base_room_ceiling') return {
    color: 0xfb7185, opacity: 0.38, renderOrder: 7, edgeOpacity: 0.82,
    depthTest: false, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
  };
  if (category === 'fallback_room_ceiling') return {
    color: 0xf59e0b, opacity: 0.18, renderOrder: 7, edgeOpacity: 0.95,
    depthTest: false, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
  };
  if (category === 'base_window') return { color: 0x38bdf8, opacity: 0.48, renderOrder: 8, edgeOpacity: 0.9 };
  if (category === 'base_door') return { color: 0xfb923c, opacity: 0.56, renderOrder: 8, edgeOpacity: 0.9 };
  if (category === 'base_opening') return { color: 0xffffff, opacity: 0.22, renderOrder: 8, edgeOpacity: 0.95 };
  if (category === 'exterior_wall') return { color: 0xa3e635, opacity: 0.42, renderOrder: 7, edgeOpacity: 0.9 };
  if (category === 'occupied_room_wall') return { color: 0xf472b6, opacity: 0.22, renderOrder: 6, edgeOpacity: 0.72 };
  if (category === 'occupied_room_floor') return { color: 0xfacc15, opacity: 0.28, renderOrder: 6, edgeOpacity: 0.72 };
  if (category === 'occupied_room_ceiling') return { color: 0xfb7185, opacity: 0.28, renderOrder: 6, edgeOpacity: 0.72 };
  if (category === 'room_ceiling_sloped') return {
    color: 0x38bdf8, opacity: 0.64, renderOrder: 8, edgeOpacity: 0.9,
    depthTest: false, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3,
  };
  if (category === 'attic_floor') return {
    color: 0x818cf8, opacity: 0.64, renderOrder: 8, edgeOpacity: 0.92,
    depthTest: false, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3,
  };
  if (category === 'room_ceiling_flat') return {
    color: 0x818cf8, opacity: 0.58, renderOrder: 8, edgeOpacity: 0.9,
    depthTest: false, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3,
  };
  if (category === 'knee_wall') return { color: 0xef4444, opacity: 0.62, renderOrder: 8, edgeOpacity: 0.9 };
  if (category === 'unresolved_region') return { color: 0xf97316, opacity: 0.28, renderOrder: 9, edgeOpacity: 0.95 };
  return null;
}

function surfaceLocatorKind(surface) {
  const category = String(surface?.category || '');
  if (category === 'exterior_roof') return 'ontology-renderable-roof';
  if (category === 'base_exterior_wall') return 'ontology-base-exterior-wall';
  if (category === 'base_interior_wall') return 'ontology-base-interior-wall';
  if (category === 'base_room_floor') return 'ontology-base-floor';
  if (category === 'base_room_ceiling') return 'ontology-base-ceiling';
  if (category === 'fallback_room_ceiling') return 'ontology-fallback-ceiling';
  if (category === 'base_window') return 'ontology-base-window';
  if (category === 'base_door') return 'ontology-base-door';
  if (category === 'base_opening') return 'ontology-base-opening';
  if (category === 'exterior_wall') return 'ontology-renderable-wall';
  if (category === 'occupied_room_wall') return 'ontology-renderable-room-wall';
  if (category === 'occupied_room_floor') return 'ontology-renderable-floor';
  if (category === 'occupied_room_ceiling') return 'ontology-renderable-ceiling';
  if (category === 'knee_wall') return 'ontology-knee-wall';
  if (category === 'unresolved_region') return 'ontology-unresolved-coverage';
  return 'ontology-renderable-ceiling';
}

export function renderOntologyEnhancedFullModel({
  ontologySummary,
  ontologyParts,
  groups,
  createPolygonMesh,
  createEdgeLoop,
  attachLocator,
  buildingUuid,
  diffMode = false,
}) {
  const partPayloads = Array.isArray(ontologyParts) ? ontologyParts : [];
  const partSurfaces = partPayloads.flatMap((payload) =>
    Array.isArray(payload?.renderable_surfaces) ? payload.renderable_surfaces : []
  );
  const surfaces = partSurfaces;

  const counts = {
    roofFaceCount: 0,
    baseWallCount: 0,
    baseFloorCount: 0,
    fenestrationCount: 0,
    exteriorWallCount: 0,
    occupiedSurfaceCount: 0,
    ceilingSurfaceCount: 0,
    fallbackCeilingCount: 0,
    kneeWallCount: 0,
    unresolvedCount: 0,
  };

  for (const surface of surfaces) {
    const style = diffMode ? diffSurfaceStyle(surface) : surfaceStyle(surface);
    if (!style) continue;
    const category = String(surface?.category || '');
    if (addFace({
      corners: surface?.corners,
      holes: Array.isArray(surface?.holes) ? surface.holes : [],
      color: style.color,
      opacity: style.opacity,
      group: groups.fullModelOntology,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: surfaceLocatorKind(surface),
        id: String(surface?.id || `${category}:${Math.random()}`),
        corners: surface?.corners,
        partId: surface?.part_id || null,
        roomId: surface?.room_id || null,
        role: surface?.role || null,
        category,
      },
      renderOrder: style.renderOrder,
      edgeOpacity: style.edgeOpacity ?? null,
      depthTest: style.depthTest ?? true,
      depthWrite: style.depthWrite ?? false,
      polygonOffset: style.polygonOffset ?? false,
      polygonOffsetFactor: style.polygonOffsetFactor ?? 0,
      polygonOffsetUnits: style.polygonOffsetUnits ?? 0,
    })) {
      if (category === 'exterior_roof') counts.roofFaceCount += 1;
      else if (category === 'base_exterior_wall' || category === 'base_interior_wall') counts.baseWallCount += 1;
      else if (category === 'base_room_floor') counts.baseFloorCount += 1;
      else if (category === 'base_room_ceiling') counts.ceilingSurfaceCount += 1;
      else if (category === 'fallback_room_ceiling') counts.fallbackCeilingCount += 1;
      else if (category === 'base_window' || category === 'base_door' || category === 'base_opening') counts.fenestrationCount += 1;
      else if (category === 'exterior_wall') counts.exteriorWallCount += 1;
      else if (category === 'knee_wall') counts.kneeWallCount += 1;
      else if (category === 'unresolved_region') counts.unresolvedCount += 1;
      else if (category.startsWith('occupied_room_')) counts.occupiedSurfaceCount += 1;
      else counts.ceilingSurfaceCount += 1;
    }
  }

  return {
    ...counts,
    hasEnhancement:
      counts.roofFaceCount > 0 ||
      counts.baseWallCount > 0 ||
      counts.baseFloorCount > 0 ||
      counts.fenestrationCount > 0 ||
      counts.exteriorWallCount > 0 ||
      counts.occupiedSurfaceCount > 0 ||
      counts.ceilingSurfaceCount > 0 ||
      counts.fallbackCeilingCount > 0 ||
      counts.kneeWallCount > 0 ||
      counts.unresolvedCount > 0,
    hasRoofReplacement: counts.roofFaceCount > 0,
    hasBaseShellReplacement: counts.baseWallCount > 0 || counts.baseFloorCount > 0,
  };
}
