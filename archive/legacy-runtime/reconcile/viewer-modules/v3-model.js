// Renders reconcile_v3_results.json payloads in the viewer.
// Mirrors full-model-ontology.js structure, but each element is already a
// first-class dataclass with an element id — no renderable_surfaces stitching.

import {
  GABLE_STATUS_COLORS,
  GABLE_RIDGE_COLOR,
  GABLE_EXTENSION_TARGET_COLOR,
  CANDIDATE_FACE_COLOR_ORIGINAL,
  CANDIDATE_FACE_COLOR_EXTENDED,
  RECONSTRUCTION_COLOR_ACCEPT,
  RECONSTRUCTION_COLOR_REVIEW,
  RIDGE_EAVE_RIDGE_COLOR,
  RIDGE_EAVE_EAVE_COLOR,
  RIDGE_EAVE_MEDIAL_COLOR,
  RIDGE_EAVE_SCORE_LOW,
  RIDGE_EAVE_SCORE_MID,
  RIDGE_EAVE_SCORE_HIGH,
} from './constants.js';

function _lerp(a, b, t) { return a + (b - a) * t; }
function _lerpColor(c0, c1, t) {
  const r0 = (c0 >> 16) & 0xff, g0 = (c0 >> 8) & 0xff, b0 = c0 & 0xff;
  const r1 = (c1 >> 16) & 0xff, g1 = (c1 >> 8) & 0xff, b1 = c1 & 0xff;
  const r = Math.round(_lerp(r0, r1, t));
  const g = Math.round(_lerp(g0, g1, t));
  const b = Math.round(_lerp(b0, b1, t));
  return (r << 16) | (g << 8) | b;
}
function scoreColor(score) {
  if (score == null || !Number.isFinite(score)) return 0x64748b;
  const t = Math.max(0, Math.min(1, score));
  return t < 0.5
    ? _lerpColor(RIDGE_EAVE_SCORE_LOW, RIDGE_EAVE_SCORE_MID, t / 0.5)
    : _lerpColor(RIDGE_EAVE_SCORE_MID, RIDGE_EAVE_SCORE_HIGH, (t - 0.5) / 0.5);
}

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
  renderOrder = 6,
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

function parseInnerId(token) {
  if (typeof token !== 'string') return '';
  const parts = token.split('::');
  return parts.length === 3 ? parts[2] : token;
}

// Color palette for proposal labels. unlabeled = muted teal so labeled
// ones (green/red) pop visually.
export const PROPOSAL_COLORS = {
  unlabeled: 0x94a3b8,
  accepted: 0x22c55e,
  rejected: 0xef4444,
  skipped: 0x64748b,
};

export function proposalColor(label) {
  return PROPOSAL_COLORS[label] ?? PROPOSAL_COLORS.unlabeled;
}

export function renderV3Model({
  v3Data,
  groups,
  createPolygonMesh,
  createEdgeLoop,
  createLine,
  attachLocator,
  buildingUuid,
}) {
  const counts = {
    slabCount: 0,
    flatCeilingCount: 0,
    slantedRoofCount: 0,
    wallExtensionCount: 0,
    gapCount: 0,
    dormerCount: 0,
    unresolvedCount: 0,
    gableStatusCount: 0,
    gableRidgeCount: 0,
    gableExtensionTargetCount: 0,
  };

  if (!v3Data || !groups?.v3Model) {
    return { ...counts, hasAny: false };
  }

  const render = (items, kind, style, polygonKey) => {
    for (const item of items || []) {
      const corners = item?.[polygonKey];
      if (!Array.isArray(corners) || corners.length < 3) continue;
      if (addFace({
        corners,
        color: style.color,
        opacity: style.opacity,
        group: groups.v3Model,
        createPolygonMesh,
        createEdgeLoop,
        attachLocator,
        locator: {
          buildingUuid,
          kind,
          id: parseInnerId(item.id),
          corners,
        },
        renderOrder: style.renderOrder,
        edgeOpacity: style.edgeOpacity ?? null,
        depthTest: style.depthTest ?? true,
        depthWrite: style.depthWrite ?? false,
        polygonOffset: style.polygonOffset ?? false,
        polygonOffsetFactor: style.polygonOffsetFactor ?? 0,
        polygonOffsetUnits: style.polygonOffsetUnits ?? 0,
      })) {
        counts[style.counter] += 1;
      }
    }
  };

  render(v3Data.slabs, 'v3-slab', {
    color: 0xfbbf24, opacity: 0.5, renderOrder: 5, edgeOpacity: 0.75,
    counter: 'slabCount',
  }, 'polygon');

  render(v3Data.flat_ceilings, 'v3-flat-ceiling', {
    color: 0x7da8d6, opacity: 0.7, renderOrder: 8, edgeOpacity: 0.88,
    polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3,
    counter: 'flatCeilingCount',
  }, 'footprint_xz');

  render(v3Data.slanted_roofs, 'v3-slanted-roof', {
    color: 0x9ec5e3, opacity: 0.45, renderOrder: 7, edgeOpacity: 0.65,
    counter: 'slantedRoofCount',
  }, 'corners');

  render(v3Data.wall_extensions, 'v3-wall-extension', {
    color: 0xa78bfa, opacity: 0.55, renderOrder: 6, edgeOpacity: 0.8,
    counter: 'wallExtensionCount',
  }, 'strip_corners');

  render(v3Data.gaps, 'v3-gap', {
    color: 0x34d399, opacity: 0.45, renderOrder: 5, edgeOpacity: 0.8,
    counter: 'gapCount',
  }, 'footprint_xz');

  render(v3Data.dormers, 'v3-dormer', {
    color: 0xf472b6, opacity: 0.55, renderOrder: 7, edgeOpacity: 0.8,
    counter: 'dormerCount',
  }, 'corners');

  // Unresolved regions — magenta, deliberately loud so failures are obvious.
  render(v3Data.unresolved, 'v3-unresolved', {
    color: 0xff00ff, opacity: 0.35, renderOrder: 9, edgeOpacity: 0.95,
    depthTest: false, depthWrite: false,
    counter: 'unresolvedCount',
  }, 'footprint_xz');

  if (groups.gableExtension) {
    renderGableDecorations({
      parts: v3Data.parts,
      group: groups.gableExtension,
      createPolygonMesh,
      createEdgeLoop,
      createLine,
      attachLocator,
      buildingUuid,
      counts,
    });
  }

  return { ...counts, hasAny: Object.values(counts).some((v) => v > 0) };
}

/**
 * Render roof proposals into ``groups.v3Proposals``, color-coded by label
 * status. Labels are looked up in ``labelsByProposalId``; missing ids render
 * as ``unlabeled``.
 *
 * When ``mergeSimilarPlanes`` is true, renders ``v3Data.merged_roof_segments``
 * — each one a labelable ``v3-merged-roof-segment`` mesh clipped and split
 * in the backend proposer (plane cluster × rain-exposed piece × opposing
 * seams × building boundary). Otherwise renders raw per-proposal corners.
 */
export function renderV3RoofProposals({
  v3Data,
  groups,
  createPolygonMesh,
  createEdgeLoop,
  attachLocator,
  buildingUuid,
  labelsByProposalId = {},
  mergeSimilarPlanes = false,
}) {
  let proposalCount = 0;
  if (!v3Data || !groups?.v3Proposals) {
    return { proposalCount, hasAny: false };
  }

  if (mergeSimilarPlanes) {
    const segments = v3Data.merged_roof_segments || [];
    for (const seg of segments) {
      if (!Array.isArray(seg.corners) || seg.corners.length < 3) continue;
      const segLabel = labelsByProposalId[seg.id] || 'unlabeled';
      const color = proposalColor(segLabel);
      const innerId = parseInnerId(seg.id);
      const ok = addFace({
        corners: seg.corners,
        color,
        opacity: 0.5,
        group: groups.v3Proposals,
        createPolygonMesh,
        createEdgeLoop,
        attachLocator,
        locator: {
          buildingUuid,
          kind: 'v3-merged-roof-segment',
          id: innerId,
          corners: seg.corners,
          plane: seg.merged_plane,
          features: seg.features || {},
          heuristicLabel: seg.heuristic_label ?? 'not_evaluated',
          proposalId: seg.id,
          mergeMode: true,
          mergedPlane: Array.isArray(seg.merged_plane) ? [...seg.merged_plane] : null,
          clusterCanonicalId: seg.cluster_canonical_id,
          clusterMembers: seg.member_snapshots || [],
          memberProposalIds: seg.member_proposal_ids || [],
          clusterParams: seg.cluster_params || null,
          opposingClusterCanonicals: seg.opposing_cluster_canonicals || [],
          opposingPlanes: seg.opposing_planes || [],
          roomBoundaryRefs: seg.room_boundary_refs || [],
          buildingBoundaryXz: seg.building_boundary_xz || null,
          labelStatus: segLabel,
        },
        renderOrder: 9,
        edgeOpacity: 0.9,
        depthTest: true,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -2,
        polygonOffsetUnits: -2,
      });
      if (ok) proposalCount += 1;
    }
    return { proposalCount, hasAny: proposalCount > 0 };
  }

  // Non-merge: render each raw proposal as its own mesh, corners as-is.
  const proposals = v3Data.roof_proposals || [];
  for (const p of proposals) {
    if (!Array.isArray(p.corners) || p.corners.length < 3) continue;
    const label = labelsByProposalId[p.id] || 'unlabeled';
    const color = proposalColor(label);
    const innerId = parseInnerId(p.id);
    const ok = addFace({
      corners: p.corners,
      color,
      opacity: 0.5,
      group: groups.v3Proposals,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'v3-roof-proposal',
        id: innerId,
        corners: p.corners,
        plane: p.plane,
        features: p.features || {},
        heuristicLabel: p.heuristic_label ?? 'not_evaluated',
        proposalId: p.id,
        mergeMode: false,
        segmentIndex: p.segment_index,
        sourceRoomId: p.source_room_id,
        sourceWallId: p.source_wall_id,
        slabRoomId: p.slab_room_id,
        slabId: p.slab_id,
        labelStatus: label,
      },
      renderOrder: 9,
      edgeOpacity: 0.85,
      depthTest: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
    if (ok) proposalCount += 1;
  }
  return { proposalCount, hasAny: proposalCount > 0 };
}

/**
 * Render Phase-A candidate faces (reconstruction track) into
 * ``groups.candidateFaces``. Each candidate carries a plane equation
 * ``a·x + b·y + c·z + d = 0`` and a 2D clipped footprint (XZ). We lift the
 * footprint to 3D via ``y = -(a·x + c·z + d)/b`` so the face sits on its
 * plane, then color by whether the face was ridge-extended beyond its
 * parent segment.
 */
export function renderCandidateFaces({
  candidatesData,
  groups,
  createPolygonMesh,
  createEdgeLoop,
  attachLocator,
  buildingUuid,
}) {
  let faceCount = 0;
  if (!candidatesData || !groups?.candidateFaces) {
    return { faceCount, hasAny: false };
  }
  const faces = Array.isArray(candidatesData.candidates)
    ? candidatesData.candidates
    : [];
  for (const face of faces) {
    const plane = face.plane;
    const fpXZ = face.footprint_xz;
    if (!Array.isArray(plane) || plane.length !== 4) continue;
    if (!Array.isArray(fpXZ) || fpXZ.length < 3) continue;
    const corners = fpXZ
      .map((pt) => {
        if (!Array.isArray(pt) || pt.length < 2) return null;
        const [x, z] = pt;
        return [x, planeYAt(plane, x, z), z];
      })
      .filter(Boolean);
    if (corners.length < 3) continue;
    const color = face.extended
      ? CANDIDATE_FACE_COLOR_EXTENDED
      : CANDIDATE_FACE_COLOR_ORIGINAL;
    const innerId = parseInnerId(face.id);
    const ok = addFace({
      corners,
      color,
      opacity: 0.45,
      group: groups.candidateFaces,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'candidate-face',
        id: innerId,
        corners,
        plane,
        parentSegmentId: face.parent_segment_id,
        extended: !!face.extended,
        areaM2: face.area_m2,
        azimuthDeg: face.azimuth_deg,
        inclinationDeg: face.inclination_deg,
        neighbors: face.neighbors || [],
        supportM2: face.support_m2,
        gbmPrior: face.gbm_prior,
      },
      renderOrder: 10,
      edgeOpacity: 0.9,
      depthTest: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
    if (ok) faceCount += 1;
  }
  return { faceCount, hasAny: faceCount > 0 };
}

/**
 * Render the BIP solver's selected envelope (Phase B.1). Expects the
 * payload shape emitted by ``/reconstruction`` — ``selected_faces`` is a
 * list of candidate-face dicts already filtered down to the selection.
 * Renders solid green for auto-accept, amber for review.
 */
export function renderReconstruction({
  reconstructionData,
  groups,
  createPolygonMesh,
  createEdgeLoop,
  attachLocator,
  buildingUuid,
}) {
  let faceCount = 0;
  if (!reconstructionData || !groups?.reconstruction) {
    return { faceCount, hasAny: false };
  }
  const decision = reconstructionData.decision || 'review';
  const color = decision === 'auto_accept'
    ? RECONSTRUCTION_COLOR_ACCEPT
    : RECONSTRUCTION_COLOR_REVIEW;
  const faces = Array.isArray(reconstructionData.selected_faces)
    ? reconstructionData.selected_faces
    : [];
  for (const face of faces) {
    const plane = face.plane;
    const fpXZ = face.footprint_xz;
    if (!Array.isArray(plane) || plane.length !== 4) continue;
    if (!Array.isArray(fpXZ) || fpXZ.length < 3) continue;
    const corners = fpXZ
      .map((pt) => {
        if (!Array.isArray(pt) || pt.length < 2) return null;
        const [x, z] = pt;
        return [x, planeYAt(plane, x, z), z];
      })
      .filter(Boolean);
    if (corners.length < 3) continue;
    const innerId = parseInnerId(face.id);
    const ok = addFace({
      corners,
      color,
      opacity: 0.55,
      group: groups.reconstruction,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'reconstruction-face',
        id: innerId,
        corners,
        plane,
        parentSegmentId: face.parent_segment_id,
        extended: !!face.extended,
        areaM2: face.area_m2,
        azimuthDeg: face.azimuth_deg,
        inclinationDeg: face.inclination_deg,
        neighbors: face.neighbors || [],
        supportM2: face.support_m2,
        gbmPrior: face.gbm_prior,
        decision,
        status: reconstructionData.status,
      },
      renderOrder: 12,
      edgeOpacity: 0.95,
      depthTest: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -3,
      polygonOffsetUnits: -3,
    });
    if (ok) faceCount += 1;
  }
  return { faceCount, hasAny: faceCount > 0 };
}

/**
 * Render ridge/eave scoring overlay. ``scoresData`` is the per-building
 * payload from ``/ridge-eave-scores``; ``candidatesData`` is the matching
 * ``/candidate-faces`` response used to look up plane + footprint for each
 * scored candidate (so we can recolor the candidate footprint by its
 * best-pair score). For the best pair of each candidate we also draw the
 * ridge (yellow), eaves (orange), part OBB (indigo), and medial axis
 * (dashed white).
 */
export function renderRidgeEaveScoring({
  scoresData,
  candidatesData,
  groups,
  createPolygonMesh,
  createEdgeLoop,
  createLine,
  attachLocator,
  buildingUuid,
}) {
  const group = groups?.ridgeEave;
  if (!group || !scoresData) return { faceCount: 0, hasAny: false };

  const candById = new Map();
  const faces = Array.isArray(candidatesData?.candidates)
    ? candidatesData.candidates
    : [];
  for (const f of faces) candById.set(f.id, f);

  let faceCount = 0;
  const scoredCandidates = scoresData.candidates || [];
  const renderedPairs = new Set();

  // For SELECTED plane-groups, render ONE unified face per plane-group using
  // its union_xz polygon lifted to the rep plane — this unfragments pieces
  // that Phase A had split against (now dropped) competing planes. Track
  // which plane-groups we rendered so we skip their individual pieces.
  //
  // When the scorer has split the union at the best-partner ridge we draw
  // the downslope half (physical roof surface) in score color and the
  // upslope half (extrapolation past the ridge into the partner's domain)
  // in light blue, so the gable extrapolation is visually distinct. Falls
  // back to the full union when no ridge-split is available.
  const ABOVE_RIDGE_COLOR = 0x88ccff;
  const planeGroups = Array.isArray(scoresData.plane_groups)
    ? scoresData.plane_groups
    : [];
  const renderedPlaneGroupIds = new Set();

  function liftRing(ring, plane) {
    if (!Array.isArray(ring) || ring.length < 3) return null;
    const corners = ring
      .map((pt) => {
        if (!Array.isArray(pt) || pt.length < 2) return null;
        const [x, z] = pt;
        return [x, planeYAt(plane, x, z), z];
      })
      .filter(Boolean);
    return corners.length >= 3 ? corners : null;
  }

  function renderPgRing({ pg, ring, color, opacity, side }) {
    const corners = liftRing(ring, pg.plane);
    if (!corners) return false;
    const innerBase = `plane-group::${(pg.id || '').split('::').pop()}`;
    const innerId = side ? `${innerBase}::${side}` : innerBase;
    return addFace({
      corners,
      color,
      opacity,
      group,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'ridge-eave-candidate',
        id: innerId,
        corners,
        plane: pg.plane,
        bestScore: pg.best_score,
        bestPartnerPlaneGroupId: pg.best_partner_plane_group_id,
        bestComponents: pg.best_components,
        planeGroupId: pg.id,
        isPlaneGroupRep: true,
        ridgeSide: side || 'union',
        areaM2: pg.total_area,
        azimuthDeg: pg.azimuth_deg,
        inclinationDeg: pg.inclination_deg,
        memberIds: pg.member_ids,
      },
      renderOrder: 11,
      edgeOpacity: 0.9,
      depthTest: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
  }

  for (const pg of planeGroups) {
    if (pg.selected === false) continue;
    if (!Array.isArray(pg.plane) || pg.plane.length !== 4) continue;

    const scoreCol = scoreColor(pg.best_score);
    const below = Array.isArray(pg.union_below_ridge_xz) ? pg.union_below_ridge_xz : [];
    const above = Array.isArray(pg.union_above_ridge_xz) ? pg.union_above_ridge_xz : [];

    let rendered = false;
    if (below.length > 0 || above.length > 0) {
      for (const ring of below) {
        if (renderPgRing({ pg, ring, color: scoreCol, opacity: 0.55, side: 'below-ridge' })) {
          faceCount += 1;
          rendered = true;
        }
      }
      for (const ring of above) {
        if (renderPgRing({ pg, ring, color: ABOVE_RIDGE_COLOR, opacity: 0.55, side: 'above-ridge' })) {
          faceCount += 1;
          rendered = true;
        }
      }
    } else if (Array.isArray(pg.union_xz) && pg.union_xz.length >= 3) {
      if (renderPgRing({ pg, ring: pg.union_xz, color: scoreCol, opacity: 0.55 })) {
        faceCount += 1;
        rendered = true;
      }
    }
    if (rendered) renderedPlaneGroupIds.add(pg.id);
  }

  for (const sc of scoredCandidates) {
    // Skip individual pieces belonging to plane-groups we already drew as a
    // single union face above — otherwise we'd double-render and re-show
    // the Phase A clipping artifacts the union was meant to erase.
    if (renderedPlaneGroupIds.has(sc.plane_group_id)) continue;
    const base = candById.get(sc.id);
    if (!base) continue;
    const plane = base.plane;
    const fpXZ = base.footprint_xz;
    if (!Array.isArray(plane) || plane.length !== 4) continue;
    if (!Array.isArray(fpXZ) || fpXZ.length < 3) continue;
    const corners = fpXZ
      .map((pt) => {
        if (!Array.isArray(pt) || pt.length < 2) return null;
        const [x, z] = pt;
        return [x, planeYAt(plane, x, z), z];
      })
      .filter(Boolean);
    if (corners.length < 3) continue;

    const score = sc.best_score;
    const color = scoreColor(score);
    const innerId = parseInnerId(sc.id);
    const isSelected = sc.selected !== false;
    const ok = addFace({
      corners,
      color,
      opacity: isSelected ? 0.55 : 0.12,
      group,
      createPolygonMesh,
      createEdgeLoop,
      attachLocator,
      locator: {
        buildingUuid,
        kind: 'ridge-eave-candidate',
        id: innerId,
        corners,
        plane,
        bestScore: score,
        bestPartnerId: sc.best_partner_id,
        bestPartnerPlaneGroupId: sc.best_partner_plane_group_id,
        bestComponents: sc.best_components,
        planeGroupId: sc.plane_group_id,
        isPlaneGroupRep: !!sc.is_plane_group_rep,
        areaM2: sc.area_m2,
        azimuthDeg: sc.azimuth_deg,
        inclinationDeg: sc.inclination_deg,
        extended: !!sc.extended,
        parentSegmentId: sc.parent_segment_id,
      },
      renderOrder: 11,
      edgeOpacity: 0.9,
      depthTest: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
    if (ok) faceCount += 1;
  }

  const pairs = Array.isArray(scoresData.pairs) ? scoresData.pairs : [];
  // Build a partner-lookup — only draw the top-score pair each candidate
  // points to, and dedupe by (min_id, max_id).
  const bestPartnerOf = new Map();
  for (const sc of scoredCandidates) {
    if (sc.best_partner_id) bestPartnerOf.set(sc.id, sc.best_partner_id);
  }
  for (const p of pairs) {
    const pairKey = [p.a_id, p.b_id].sort().join('|');
    if (renderedPairs.has(pairKey)) continue;
    const aPartner = bestPartnerOf.get(p.a_id);
    const bPartner = bestPartnerOf.get(p.b_id);
    // Only render if this pair is the best-pair for at least one side.
    if (aPartner !== p.b_id && bPartner !== p.a_id) continue;
    renderedPairs.add(pairKey);

    const candA = candById.get(p.a_id);
    const candB = candById.get(p.b_id);
    const planeA = candA?.plane;
    const planeB = candB?.plane;

    // Ridge (3D — has explicit y).
    if (Array.isArray(p.ridge_xz) && p.ridge_xz.length === 2
        && Array.isArray(p.ridge_y) && p.ridge_y.length === 2) {
      const pts3 = [
        [p.ridge_xz[0][0], p.ridge_y[0], p.ridge_xz[0][1]],
        [p.ridge_xz[1][0], p.ridge_y[1], p.ridge_xz[1][1]],
      ];
      const line = createLine(pts3, RIDGE_EAVE_RIDGE_COLOR, 0.95);
      if (line) {
        line.renderOrder = 13;
        line.material.depthTest = true;
        line.material.linewidth = 2;
        group.add(line);
      }
    }

    // Eaves — each side is a list of polyline segments (XZ).
    // Lift every vertex to the plane of its candidate.
    const yMid = Array.isArray(p.ridge_y)
      ? 0.5 * (p.ridge_y[0] + p.ridge_y[1])
      : 0;
    const liftPolyline = (ptsXZ, plane) => {
      if (!Array.isArray(ptsXZ) || ptsXZ.length < 2) return null;
      return ptsXZ.map(([x, z]) => {
        const y = Array.isArray(plane) ? planeYAt(plane, x, z) : yMid;
        return [x, y, z];
      });
    };
    const drawEaveGroup = (lines, plane) => {
      if (!Array.isArray(lines)) return;
      for (const pts of lines) {
        const lifted = liftPolyline(pts, plane);
        if (!lifted) continue;
        const line = createLine(lifted, RIDGE_EAVE_EAVE_COLOR, 0.95);
        if (line) {
          line.renderOrder = 13;
          line.material.depthTest = true;
          if ('linewidth' in line.material) line.material.linewidth = 2;
          group.add(line);
        }
      }
    };
    drawEaveGroup(p.eave_a_xz, planeA);
    drawEaveGroup(p.eave_b_xz, planeB);

    // Side polygons — outline only, at ridge-midpoint elevation.
    const drawSideOutline = (ring) => {
      if (!Array.isArray(ring) || ring.length < 3) return;
      const pts3 = ring.map(([x, z]) => [x, yMid, z]);
      if (pts3.length > 0
          && (pts3[0][0] !== pts3[pts3.length - 1][0]
              || pts3[0][2] !== pts3[pts3.length - 1][2])) {
        pts3.push(pts3[0]);
      }
      const line = createLine(pts3, RIDGE_EAVE_MEDIAL_COLOR, 0.4);
      if (line) {
        line.renderOrder = 12;
        line.material.depthTest = true;
        group.add(line);
      }
    };
    drawSideOutline(p.side_a_xz);
    drawSideOutline(p.side_b_xz);
  }

  return { faceCount, hasAny: faceCount > 0, pairCount: renderedPairs.size };
}

function normalizedPlane(plane) {
  if (!Array.isArray(plane) || plane.length !== 4) return null;
  const [a, b, c, d] = plane;
  const n = Math.hypot(a, b, c);
  if (!(n > 1e-9)) return null;
  const s = b >= 0 ? 1 / n : -1 / n; // orient upward so (n·n') and d compare consistently
  return [a * s, b * s, c * s, d * s];
}

function clusterProposalsByPlane(proposals, { normalDotMin, dAbsMax }) {
  const clusters = [];
  for (const p of proposals) {
    const planeN = normalizedPlane(p.plane);
    if (!planeN) continue;
    let best = null;
    for (const cluster of clusters) {
      const [ax, ay, az, ad] = cluster.plane;
      const dot = ax * planeN[0] + ay * planeN[1] + az * planeN[2];
      if (dot >= normalDotMin && Math.abs(ad - planeN[3]) <= dAbsMax) {
        best = cluster;
        break;
      }
    }
    if (best) {
      best.members.push(p);
      const n = best.members.length;
      // Running mean for the cluster plane (re-normalize after each add).
      const [bx, by, bz, bd] = best.plane;
      const mx = ((n - 1) * bx + planeN[0]) / n;
      const my = ((n - 1) * by + planeN[1]) / n;
      const mz = ((n - 1) * bz + planeN[2]) / n;
      const md = ((n - 1) * bd + planeN[3]) / n;
      const norm = Math.hypot(mx, my, mz) || 1;
      best.plane = [mx / norm, my / norm, mz / norm, md / norm];
    } else {
      clusters.push({ plane: planeN, members: [p] });
    }
  }
  return clusters;
}

function planeYAt(plane, x, z) {
  const [a, b, c, d] = plane;
  if (Math.abs(b) < 1e-6) return 0;
  return -(a * x + c * z + d) / b;
}

function extractPolyXZ(corners) {
  if (!Array.isArray(corners)) return [];
  const out = [];
  for (const c of corners) {
    if (Array.isArray(c) && c.length >= 3) out.push([c[0], c[2]]);
  }
  return out;
}

function bboxOfPolyXZ(polyXZ) {
  if (!polyXZ.length) return null;
  let minX = polyXZ[0][0];
  let maxX = minX;
  let minZ = polyXZ[0][1];
  let maxZ = minZ;
  for (let i = 1; i < polyXZ.length; i += 1) {
    const [x, z] = polyXZ[i];
    if (x < minX) minX = x; else if (x > maxX) maxX = x;
    if (z < minZ) minZ = z; else if (z > maxZ) maxZ = z;
  }
  return [minX, minZ, maxX, maxZ];
}

function bboxesOverlapXZ(a, b, pad = 0) {
  if (!a || !b) return false;
  return a[2] >= b[0] - pad && b[2] >= a[0] - pad && a[3] >= b[1] - pad && b[3] >= a[1] - pad;
}

// Cosine between the horizontal (XZ) components of two upward-normalized plane
// normals. ~1 = same slope, 0 = perpendicular (hip), -1 = anti-parallel (gable
// ridge). Returns 1 for any near-flat plane (no meaningful azimuth).
function cosAzimuthBetweenPlanes(planeA, planeB) {
  const aA = planeA[0];
  const cA = planeA[2];
  const aB = planeB[0];
  const cB = planeB[2];
  const hA = Math.hypot(aA, cA);
  const hB = Math.hypot(aB, cB);
  if (hA < 1e-6 || hB < 1e-6) return 1;
  return (aA * aB + cA * cB) / (hA * hB);
}

// Returns [A, B, C] for the XZ half-plane A*x + B*z + C >= 0 where
// ownPlane.y(x,z) <= otherPlane.y(x,z). Both planes must be upward-normalized
// (b >= 0); that keeps the "lower envelope" side consistent across clips.
function halfPlaneForOpposing(ownPlane, otherPlane) {
  const [aI, bI, cI, dI] = ownPlane;
  const [aJ, bJ, cJ, dJ] = otherPlane;
  return [bJ * aI - bI * aJ, bJ * cI - bI * cJ, bJ * dI - bI * dJ];
}

// Split a (possibly non-convex) simple polygon by a single line
// A*x + B*z + C = 0 into its positive-side and negative-side halves. Vertices
// in the eps band belong to both halves so a boundary-hugging polygon still
// produces a usable piece on each side. Edges that cross the line get a
// shared intersection vertex on both output polygons.
function splitPolyXZByLine(polyXZ, A, B, C, eps = 1e-9) {
  if (polyXZ.length < 3) return { pos: [], neg: [] };
  const pos = [];
  const neg = [];
  const n = polyXZ.length;
  for (let i = 0; i < n; i += 1) {
    const p = polyXZ[i];
    const q = polyXZ[(i + 1) % n];
    const vp = A * p[0] + B * p[1] + C;
    const vq = A * q[0] + B * q[1] + C;
    if (vp >= -eps) pos.push(p);
    if (vp <= eps) neg.push(p);
    const crosses = (vp > eps && vq < -eps) || (vp < -eps && vq > eps);
    if (crosses) {
      const denom = vp - vq;
      if (Math.abs(denom) > 1e-12) {
        const t = vp / denom;
        const xi = p[0] + t * (q[0] - p[0]);
        const zi = p[1] + t * (q[1] - p[1]);
        pos.push([xi, zi]);
        neg.push([xi, zi]);
      }
    }
  }
  return { pos, neg };
}

function renderGableDecorations({
  parts,
  group,
  createPolygonMesh,
  createEdgeLoop,
  createLine,
  attachLocator,
  buildingUuid,
  counts,
}) {
  for (const part of parts || []) {
    const gable = part?.gable_extension;
    if (!gable) continue;
    const innerId = parseInnerId(part.id);
    const statusColor = GABLE_STATUS_COLORS[gable.status] ?? GABLE_STATUS_COLORS.not_gable;

    // Part footprint colored by status (thin overlay just above the floor).
    if (Array.isArray(part.footprint_xz) && part.footprint_xz.length >= 3) {
      const ok = addFace({
        corners: part.footprint_xz,
        color: statusColor,
        opacity: 0.22,
        group,
        createPolygonMesh,
        createEdgeLoop,
        attachLocator,
        locator: {
          buildingUuid,
          kind: 'v3-gable-status',
          id: innerId,
          corners: part.footprint_xz,
        },
        renderOrder: 10,
        edgeOpacity: 0.75,
        depthTest: false,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -4,
        polygonOffsetUnits: -4,
      });
      if (ok) counts.gableStatusCount += 1;
    }

    // Ridge line (red) — Tier 1 pass only.
    if (gable.ridge_line && createLine) {
      const [a, b] = gable.ridge_line;
      if (Array.isArray(a) && Array.isArray(b)) {
        const line = createLine([a, b], GABLE_RIDGE_COLOR, 0.95);
        if (line) {
          line.renderOrder = 12;
          if (line.material) {
            line.material.depthTest = false;
            line.material.transparent = true;
          }
          attachLocator(line, {
            buildingUuid,
            kind: 'v3-gable-ridge',
            id: innerId,
            corners: [a, b],
          });
          group.add(line);
          counts.gableRidgeCount += 1;
        }
      }
    }

    // Extension target polygon (orange) — where the gable can be extended to.
    if (Array.isArray(gable.uncovered_region_xz) && gable.uncovered_region_xz.length >= 3) {
      const ok = addFace({
        corners: gable.uncovered_region_xz,
        color: GABLE_EXTENSION_TARGET_COLOR,
        opacity: 0.45,
        group,
        createPolygonMesh,
        createEdgeLoop,
        attachLocator,
        locator: {
          buildingUuid,
          kind: 'v3-gable-extension-target',
          id: innerId,
          corners: gable.uncovered_region_xz,
        },
        renderOrder: 11,
        edgeOpacity: 0.85,
        depthTest: false,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -5,
        polygonOffsetUnits: -5,
      });
      if (ok) counts.gableExtensionTargetCount += 1;
    }
  }
}
