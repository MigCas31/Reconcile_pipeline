export function renderRoofFromPythonResult({
  pyResult,
  THREE,
  groups,
  createLine,
  createPolygonMesh,
  createEdgeLoop,
  roofClusterColors,
  attachLocator,
  buildingUuid,
}) {
  if (!pyResult || typeof pyResult !== 'object') return [];

  const validClusters = Array.isArray(pyResult.valid_clusters) ? pyResult.valid_clusters : [];
  const flatSurfaces = Array.isArray(pyResult?.roof_surfaces?.flat) ? pyResult.roof_surfaces.flat : [];
  const obliqueSurfaces = Array.isArray(pyResult?.roof_surfaces?.oblique) ? pyResult.roof_surfaces.oblique : [];
  const flatCeilings = Array.isArray(pyResult?.ceiling?.flat) ? pyResult.ceiling.flat : [];
  // ceiling.oblique is now merged into roof_surfaces.oblique
  const simpleSlantCeilings = Array.isArray(pyResult?.ceiling?.simple_slant) ? pyResult.ceiling.simple_slant : [];
  const roomPartitions = Array.isArray(pyResult?.ceiling?.room_partitions) ? pyResult.ceiling.room_partitions : [];
  const partitionedRoomIndices = new Set(
    roomPartitions
      .filter((room) => Number.isInteger(room?.room_index) && Number(room?.partition_count || 0) > 0)
      .map((room) => room.room_index)
  );

  const roofClusterData = Array.isArray(pyResult.roof_cluster_data)
    ? pyResult.roof_cluster_data.map((e, i) => ({
        color: (typeof e.color === 'number') ? e.color : roofClusterColors[i % roofClusterColors.length],
        ...e,
      }))
    : [];

  const colorForCluster = (cluster, fallbackIndex) => {
    if (!cluster) return roofClusterColors[fallbackIndex % roofClusterColors.length];
    let bestIdx = -1;
    let bestScore = Infinity;
    for (let i = 0; i < validClusters.length; i++) {
      const vc = validClusters[i];
      const da = Math.abs((vc.avgAzimuth ?? 0) - (cluster.avgAzimuth ?? 0));
      const di = Math.abs((vc.avgIncl ?? 0) - (cluster.avgIncl ?? 0));
      const dc = Math.abs((vc.segs?.length ?? 0) - (cluster.segs?.length ?? 0));
      const score = da + di + dc;
      if (score < bestScore) { bestScore = score; bestIdx = i; }
    }
    const idx = bestIdx >= 0 ? bestIdx : fallbackIndex;
    return roofClusterColors[idx % roofClusterColors.length];
  };

  for (let ci = 0; ci < validClusters.length; ci++) {
    const cl = validClusters[ci];
    const color = roofClusterColors[ci % roofClusterColors.length];
    for (const seg of (cl.segs || [])) {
      const a = seg.a, b = seg.b;
      if (!a || !b) continue;
      const line = createLine([new THREE.Vector3(a[0], a[1], a[2]), new THREE.Vector3(b[0], b[1], b[2])], color, 1.0);
      line.renderOrder = 101;
      line.material.depthTest = false;
      groups.roofClusters.add(line);
    }
  }

  for (let i = 0; i < obliqueSurfaces.length; i++) {
    const s = obliqueSurfaces[i];
    const corners = s.corners;
    if (!Array.isArray(corners) || corners.length < 3) continue;
    const color = colorForCluster(s.cluster, i);
    const holeArrays = Array.isArray(s.cutout_holes) ? s.cutout_holes : [];
    const mesh = createPolygonMesh(corners, color, 0.15, holeArrays);
    if (mesh) {
      mesh.renderOrder = 50;
      mesh.material.depthTest = true;
      mesh.material.depthWrite = false;
      if (attachLocator) attachLocator(mesh, {
        buildingUuid,
        kind: "roof-oblique",
        id: `oblique:${i}`,
        corners,
      });
      groups.roofClusters.add(mesh);
    }
    groups.roofClusters.add(createEdgeLoop(corners, color, 0.5));
  }

  // Render dormer cheeks and headers
  const dormers = Array.isArray(pyResult?.dormers) ? pyResult.dormers : [];
  for (let di = 0; di < dormers.length; di++) {
    const d = dormers[di];
    for (let ci = 0; ci < (d.cheeks || []).length; ci++) {
      const cheek = d.cheeks[ci];
      if (!cheek?.corners || cheek.corners.length < 3 || cheek.source === 'existing') continue;
      const cheekMesh = createPolygonMesh(cheek.corners, 0xcc8844, 0.3);
      if (cheekMesh) {
        cheekMesh.renderOrder = 50;
        cheekMesh.material.depthTest = true;
        cheekMesh.material.depthWrite = false;
        if (attachLocator) attachLocator(cheekMesh, {
          buildingUuid,
          kind: 'dormer-cheek',
          id: `dormer-cheek:${di}:${ci}`,
          corners: cheek.corners,
        });
        groups.roofClusters.add(cheekMesh);
      }
      groups.roofClusters.add(createEdgeLoop(cheek.corners, 0xcc8844, 0.5));
    }
    if (d.header && d.header.corners && d.header.corners.length >= 3 && d.header.source !== 'existing') {
      const headerMesh = createPolygonMesh(d.header.corners, 0xccaa44, 0.3);
      if (headerMesh) {
        headerMesh.renderOrder = 50;
        headerMesh.material.depthTest = true;
        headerMesh.material.depthWrite = false;
        if (attachLocator) attachLocator(headerMesh, {
          buildingUuid,
          kind: 'dormer-header',
          id: `dormer-header:${di}`,
          corners: d.header.corners,
        });
        groups.roofClusters.add(headerMesh);
      }
      groups.roofClusters.add(createEdgeLoop(d.header.corners, 0xccaa44, 0.5));
    }
  }

  const flatColorOffset = validClusters.length;
  for (let i = 0; i < flatSurfaces.length; i++) {
    const s = flatSurfaces[i];
    const corners = s.corners;
    if (!Array.isArray(corners) || corners.length < 3) continue;
    const color = roofClusterColors[(flatColorOffset + i) % roofClusterColors.length];
    const mesh = createPolygonMesh(corners, color, 0.15);
    if (mesh) {
      mesh.renderOrder = 50;
      mesh.material.depthTest = true;
      mesh.material.depthWrite = false;
      if (attachLocator) attachLocator(mesh, {
        buildingUuid,
        kind: "roof-flat",
        id: `flat:${i}`,
        corners,
      });
      groups.roofClusters.add(mesh);
    }
    const edge = createEdgeLoop(corners, color, 0.5);
    edge.renderOrder = 101;
    edge.material.depthTest = false;
    groups.roofClusters.add(edge);

    for (const seg of (s.segments || [])) {
      const a = seg.a, b = seg.b;
      if (!a || !b) continue;
      const line = createLine([new THREE.Vector3(a[0], a[1], a[2]), new THREE.Vector3(b[0], b[1], b[2])], color, 1.0);
      line.renderOrder = 101;
      line.material.depthTest = false;
      groups.roofClusters.add(line);
    }
  }

  for (let i = 0; i < flatCeilings.length; i++) {
    const c = flatCeilings[i];
    if (!Array.isArray(c.poly) || c.poly.length < 3) continue;
    const ceilMesh = createPolygonMesh(c.poly, 0x88aacc, 0.25);
    if (ceilMesh) {
      if (attachLocator) attachLocator(ceilMesh, {
        buildingUuid,
        kind: "ceiling-flat",
        id: `ceiling-flat:${i}`,
        corners: c.poly,
      });
      groups.ceilings.add(ceilMesh);
    }
    groups.ceilings.add(createEdgeLoop(c.poly, 0x88aacc));
  }

  // ceiling.oblique rendering removed — merged into roof_surfaces.oblique

  for (let i = 0; i < simpleSlantCeilings.length; i++) {
    const c = simpleSlantCeilings[i];
    if (Number.isInteger(c?.room_index) && partitionedRoomIndices.has(c.room_index)) continue;
    if (!Array.isArray(c.poly) || c.poly.length < 3) continue;
    const ceilMesh = createPolygonMesh(c.poly, 0xccaa88, 0.3);
    if (ceilMesh) {
      if (attachLocator) attachLocator(ceilMesh, {
        buildingUuid,
        kind: "ceiling-simple-slant",
        id: `ceiling-slant:${i}`,
        corners: c.poly,
      });
      groups.ceilings.add(ceilMesh);
    }
    groups.ceilings.add(createEdgeLoop(c.poly, 0xccaa88));
  }

  // --- Thermal-envelope ceilings ---
  const thermalCeilings = Array.isArray(pyResult?.ceiling?.thermal) ? pyResult.ceiling.thermal : [];
  const THERMAL_COLORS = {
    'thermal-flat':           0xe85d04,
    'thermal-slant':          0xdc2f02,
    'thermal-cap':            0xf48c06,
    'thermal-knee':           0xff006e,
    'thermal-dormer-cheek':   0xcc8844,
    'thermal-dormer-header':  0xccaa44,
  };
  for (let i = 0; i < thermalCeilings.length; i++) {
    const tc = thermalCeilings[i];
    if (!Array.isArray(tc.poly) || tc.poly.length < 3) continue;
    const color = THERMAL_COLORS[tc.kind] || 0xe85d04;
    const opacity = tc.kind === 'thermal-slant' ? 0.45 : 0.5;
    const holeArrays = Array.isArray(tc.cutout_holes) ? tc.cutout_holes : [];
    const mesh = createPolygonMesh(tc.poly, color, opacity, holeArrays);
    if (mesh) {
      mesh.renderOrder = 55;
      mesh.material.depthTest = true;
      mesh.material.depthWrite = false;
      if (attachLocator) attachLocator(mesh, {
        buildingUuid,
        kind: 'thermal-ceiling',
        id: `thermal:${tc.kind}:${i}`,
        corners: tc.poly,
      });
      groups.thermalCeilings.add(mesh);
    }
    groups.thermalCeilings.add(createEdgeLoop(tc.poly, color, 0.6));
  }

  return roofClusterData;
}
