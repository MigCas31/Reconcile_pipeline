function addFace({
  face,
  color,
  opacity,
  group,
  createPolygonMesh,
  createEdgeLoop,
  attachLocator,
  locator,
  renderOrder = 60,
  edgeOpacity = null,
  drawFill = true,
}) {
  const corners = face?.corners || face?.vertices || face?.poly || face?.polygon;
  if (!Array.isArray(corners) || corners.length < 3) return;
  if (drawFill) {
    const mesh = createPolygonMesh(corners, color, opacity);
    if (mesh) {
      mesh.renderOrder = renderOrder;
      mesh.material.depthTest = true;
      mesh.material.depthWrite = false;
      if (attachLocator && locator) attachLocator(mesh, locator);
      group.add(mesh);
    }
  }
  group.add(createEdgeLoop(corners, color, edgeOpacity ?? Math.min(opacity + 0.22, 0.95)));
}

function semanticAtomStyle(atom) {
  const role = String(atom?.role || '');
  const kind = String(atom?.kind || '');
  if (role === 'sloped_ceiling' || kind === 'oblique') return { color: 0x0f766e, opacity: 0.34 };
  if (role === 'attic_floor') return { color: 0xc2410c, opacity: 0.3 };
  if (role === 'attic_floor_inferred') return { color: 0xea580c, opacity: 0.22 };
  if (role === 'attic_floor_candidate') return { color: 0xfb923c, opacity: 0.12 };
  if (role === 'flat_transition_cap') return { color: 0x2563eb, opacity: 0.26 };
  if (role === 'flat_transition_cap_inferred') return { color: 0x3b82f6, opacity: 0.18 };
  if (role === 'flat_transition_cap_candidate') return { color: 0x60a5fa, opacity: 0.1 };
  return { color: 0x6b7280, opacity: 0.06 };
}

function subpartStyle(subpart) {
  const kind = String(subpart?.semantic_kind || 'slope_run');
  if (kind === 'gable_run') return { color: 0xdc2626, opacity: 0.18 };
  if (kind === 'l_t_branch') return { color: 0x7c3aed, opacity: 0.16 };
  return { color: 0xf59e0b, opacity: 0.14 };
}

function coveragePatchStyle(patch, selectedPartId) {
  const selected = selectedPartId && (patch?.effective_part_ids || []).includes(selectedPartId);
  return {
    color: 0x0f766e,
    opacity: selected ? 0.34 : 0.18,
    edgeOpacity: selected ? 0.8 : 0.5,
  };
}

function unresolvedStyle(region) {
  const high = Number(region?.roof_evidence_score || 0) >= 6;
  return {
    color: high ? 0xdc2626 : 0xf59e0b,
    opacity: high ? 0.28 : 0.22,
  };
}

function continuationDiagnosticStyle(region, selectedPartId) {
  const selected = selectedPartId && (region?.effective_part_ids || []).includes(selectedPartId);
  const exact = Number(region?.exact_incidence_pair_count || 0) > 0;
  return {
    color: exact ? 0x22d3ee : 0x38bdf8,
    opacity: selected ? (exact ? 0.34 : 0.28) : (exact ? 0.2 : 0.14),
    edgeOpacity: selected ? 0.9 : 0.55,
  };
}

function shouldRenderSemanticAtom(atom) {
  const role = String(atom?.role || '');
  if (!role || role === 'flat_ceiling') return false;
  return [
    'sloped_ceiling',
    'attic_floor',
    'attic_floor_inferred',
    'attic_floor_candidate',
    'flat_transition_cap',
    'flat_transition_cap_inferred',
    'flat_transition_cap_candidate',
  ].includes(role);
}

function roofFaceStyle(cell, face) {
  const role = String(face?.role || '');
  const attic = cell?.cell_kind === 'attic';
  if (role === 'roof') return { color: attic ? 0xc2410c : 0x0f766e, opacity: 0.32 };
  if (role === 'slab') return { color: 0x60a5fa, opacity: 0.16 };
  if (role === 'wall') return { color: 0xe11d48, opacity: 0.22 };
  return { color: 0x64748b, opacity: 0.08 };
}

export function renderOntologySemantics({
  ontologySummary,
  selectedPartId = null,
  groups,
  createPolygonMesh,
  createEdgeLoop,
  attachLocator,
  buildingUuid,
}) {
  const buildingParts = Array.isArray(ontologySummary?.building_parts) ? ontologySummary.building_parts : [];
  const subparts = Array.isArray(ontologySummary?.coverage_subparts) ? ontologySummary.coverage_subparts : [];
  const coveragePatches = Array.isArray(ontologySummary?.oblique_coverage_patches) ? ontologySummary.oblique_coverage_patches : [];
  const atoms = Array.isArray(ontologySummary?.semantic_atoms) ? ontologySummary.semantic_atoms : [];
  const unresolvedRegions = Array.isArray(ontologySummary?.unresolved_regions) ? ontologySummary.unresolved_regions : [];
  const dormers = Array.isArray(ontologySummary?.dormers) ? ontologySummary.dormers : [];

  for (const part of buildingParts) {
    const corners = part?.polygon;
    if (!Array.isArray(corners) || corners.length < 3) continue;
    const selected = selectedPartId && part?.id === selectedPartId;
    addFace({
      face: { corners },
      color: selected ? 0xf8fafc : 0xe5e7eb,
      opacity: selected ? 0.03 : 0.02,
      edgeOpacity: selected ? 0.95 : 0.45,
      group: groups.ontologySemantics,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'ontology-building-part',
        id: String(part.id || 'part'),
        corners,
        partId: part.id,
      },
      renderOrder: 52,
      drawFill: false,
    });
  }

  for (const patch of coveragePatches) {
    const style = coveragePatchStyle(patch, selectedPartId);
    addFace({
      face: { corners: patch?.polygon },
      color: style.color,
      opacity: style.opacity,
      edgeOpacity: style.edgeOpacity,
      group: groups.ontologySemantics,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'ontology-oblique-coverage',
        id: String(patch?.id || 'coverage-patch'),
        corners: patch?.polygon,
        partIds: patch?.effective_part_ids || [],
        partId: (patch?.effective_part_ids || [])[0] || null,
        roofHypothesisId: patch?.roof_hypothesis_id,
      },
      renderOrder: 54,
    });
  }

  for (const subpart of subparts) {
    const style = subpartStyle(subpart);
    const selected = selectedPartId && (subpart?.effective_part_ids || []).includes(selectedPartId);
    addFace({
      face: { corners: subpart?.polygon },
      color: style.color,
      opacity: selected ? Math.min(style.opacity + 0.06, 0.28) : style.opacity,
      group: groups.ontologySemantics,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'ontology-coverage-subpart',
        id: String(subpart?.id || 'subpart'),
        corners: subpart?.polygon,
        semanticKind: subpart?.semantic_kind,
        partIds: subpart?.effective_part_ids || [],
        partId: (subpart?.effective_part_ids || [])[0] || null,
      },
      renderOrder: 55,
      drawFill: false,
    });
  }

  for (const atom of atoms) {
    if (!shouldRenderSemanticAtom(atom)) continue;
    const style = semanticAtomStyle(atom);
    const selected = selectedPartId && atom?.effective_part_id === selectedPartId;
    addFace({
      face: { corners: atom?.poly },
      color: style.color,
      opacity: selected ? Math.min(style.opacity + 0.08, 0.36) : style.opacity,
      group: groups.ontologySemantics,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'ontology-top-boundary-atom',
        id: String(atom?.id || 'atom'),
        corners: atom?.poly,
        role: atom?.role,
        partId: atom?.effective_part_id,
      },
      renderOrder: 58,
    });
  }

  for (const region of unresolvedRegions) {
    const style = unresolvedStyle(region);
    addFace({
      face: { corners: region?.polygon },
      color: style.color,
      opacity: style.opacity,
      group: groups.ontologySemantics,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'ontology-unresolved-coverage',
        id: String(region?.id || 'unresolved'),
        corners: region?.polygon,
        partIds: region?.effective_part_ids || [],
        partId: (region?.effective_part_ids || [])[0] || null,
        roomId: region?.room_id,
      },
      renderOrder: 59,
    });
  }

  for (let di = 0; di < dormers.length; di++) {
    const dormer = dormers[di];
    for (let ci = 0; ci < (dormer?.cheeks || []).length; ci++) {
      addFace({
        face: { corners: dormer.cheeks[ci]?.corners },
        color: 0xcc8844,
        opacity: 0.28,
        group: groups.ontologySemantics,
        createPolygonMesh,
        createEdgeLoop,
        attachLocator,
        locator: {
          buildingUuid,
          kind: 'ontology-dormer-cheek',
          id: `${dormer?.id || `dormer:${di}`}:cheek:${ci}`,
          corners: dormer.cheeks[ci]?.corners,
        },
        renderOrder: 61,
      });
    }
    addFace({
      face: { corners: dormer?.header?.corners },
      color: 0xeab308,
      opacity: 0.3,
      group: groups.ontologySemantics,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'ontology-dormer-header',
        id: String(dormer?.id || `dormer:${di}`),
        corners: dormer?.header?.corners,
      },
      renderOrder: 61,
    });
  }

  return {
    buildingPartCount: buildingParts.length,
    coveragePatchCount: coveragePatches.length,
    coverageSubpartCount: subparts.length,
    semanticAtomCount: atoms.length,
    unresolvedRegionCount: unresolvedRegions.length,
    dormerCount: dormers.length,
  };
}

export function renderOntologyContinuationDiagnostics({
  ontologySummary,
  selectedPartId = null,
  groups,
  createPolygonMesh,
  createEdgeLoop,
  attachLocator,
  buildingUuid,
}) {
  const continuationRegions = Array.isArray(ontologySummary?.roof_continuation_diagnostics)
    ? ontologySummary.roof_continuation_diagnostics
    : [];
  let continuationRegionCount = 0;

  for (const region of continuationRegions) {
    const style = continuationDiagnosticStyle(region, selectedPartId);
    addFace({
      face: { corners: region?.polygon },
      color: style.color,
      opacity: style.opacity,
      edgeOpacity: style.edgeOpacity,
      group: groups.ontologyContinuation,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'ontology-roof-continuation',
        id: String(region?.id || 'continuation-region'),
        corners: region?.polygon,
        partIds: region?.effective_part_ids || [],
        partId: (region?.effective_part_ids || [])[0] || null,
        roofHypothesisId: region?.roof_hypothesis_id,
        roomId: region?.room_id,
      },
      renderOrder: 57,
    });
    continuationRegionCount += 1;
  }

  return {
    continuationRegionCount,
  };
}

export function renderOntologyExact({
  ontologyParts,
  groups,
  createPolygonMesh,
  createEdgeLoop,
  attachLocator,
  buildingUuid,
}) {
  const partPayloads = Array.isArray(ontologyParts) ? ontologyParts : [];
  let roofCellCount = 0;
  let kneeWallCount = 0;

  for (const payload of partPayloads) {
    const roofCells = Array.isArray(payload?.roof_cells) ? payload.roof_cells : [];
    const kneeWalls = Array.isArray(payload?.knee_walls) ? payload.knee_walls : [];

    roofCellCount += roofCells.length;
    kneeWallCount += kneeWalls.length;

    for (const cell of roofCells) {
      const faces = Array.isArray(cell?.faces) ? cell.faces : [];
      for (const face of faces) {
        const style = roofFaceStyle(cell, face);
        addFace({
          face,
          color: style.color,
          opacity: style.opacity,
          group: groups.ontologyCells,
          createPolygonMesh,
          createEdgeLoop,
          attachLocator,
          locator: {
            buildingUuid,
            kind: 'ontology-roof-face',
            id: `${cell.id}:${face.id || face.kind || 'face'}`,
            corners: face.corners,
            cellKind: cell.cell_kind,
            partId: payload?.part_id,
          },
        });
      }
    }

    for (let i = 0; i < kneeWalls.length; i++) {
      const wall = kneeWalls[i];
      addFace({
        face: wall,
        color: 0xff006e,
        opacity: 0.28,
        group: groups.ontologyCells,
        createPolygonMesh,
        createEdgeLoop,
        attachLocator,
        locator: {
          buildingUuid,
          kind: 'ontology-knee-wall',
          id: String(wall?.id || `knee:${payload?.part_id || 'part'}:${i}`),
          corners: wall?.corners,
          partId: payload?.part_id,
        },
        renderOrder: 62,
      });
    }
  }

  return {
    loadedPartCount: partPayloads.length,
    roofCellCount,
    kneeWallCount,
  };
}
