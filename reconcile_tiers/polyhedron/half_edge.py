"""Half-edge data structure with face-plane geometry.

Following Geniet, Brédif & Vallet (ISPRS 2024). The defining property is
that vertex coordinates are *derived* from the intersection of the three
adjacent face planes, not stored. This makes face shifts (`face.plane.d +=
delta`) automatically propagate to every vertex on the face's boundary
without any explicit per-vertex update.

Phase 1: data structure + three_plane_intersection + face_shift +
make_cube fixture. No topology operators yet.

Phase 1.5: build_from_planar_polygons constructor (build a polyhedron
from a list of (corners, plane) pairs by clustering shared vertices and
twin-matching directed edges) + make_gable_house fixture for realistic
tests that approximate the tirana pipeline's roof + walls + floor shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from reconcile_tiers._core.plane import Plane


@dataclass
class Vertex:
    """A vertex identified by id; its coordinates are derived from the three
    incident face planes via :func:`three_plane_intersection`.
    """

    id: int
    outgoing: HalfEdge | None = None  # one of the half-edges leaving this vertex

    def __repr__(self) -> str:
        return f"Vertex(id={self.id})"


@dataclass
class HalfEdge:
    """One side of an edge. Bounds exactly one face. Has a twin (`opposite`)
    on the other side of the edge that bounds the adjacent face.

    Per the paper's Figure 2: (origin, opposite, next, face).
    """

    id: int
    origin: Vertex
    opposite: HalfEdge | None = None
    next: HalfEdge | None = None
    face: Face | None = None

    def __repr__(self) -> str:
        return f"HalfEdge(id={self.id}, origin={self.origin.id})"


@dataclass
class Face:
    """A planar face. The plane fully determines the face's geometry; the
    bounded polygon is recovered by walking ``half_edge.next`` until we
    return to ``half_edge``.
    """

    id: int
    plane: Plane
    half_edge: HalfEdge | None = None  # one of the bounding half-edges

    def __repr__(self) -> str:
        return f"Face(id={self.id}, plane={self.plane})"


@dataclass
class HalfEdgePolyhedron:
    """A watertight manifold polyhedron stored as faces (planes) plus a
    half-edge connectivity graph.
    """

    vertices: list[Vertex] = field(default_factory=list)
    half_edges: list[HalfEdge] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)

    # ---- geometry derivations -----------------------------------------------

    def vertex_position(self, v: Vertex) -> np.ndarray:
        """Return the 3D coordinates of ``v`` as the intersection of its three
        incident face planes.

        Raises ValueError if the vertex has fewer than three distinct
        incident faces (degenerate / boundary; not allowed in a watertight
        manifold polyhedron).
        """
        faces = self.incident_faces(v)
        if len(faces) < 3:
            raise ValueError(
                f"Vertex {v.id} has only {len(faces)} incident faces; "
                "need 3 to define a position"
            )
        for face_triplet in combinations(faces, 3):
            try:
                return three_plane_intersection(
                    face_triplet[0].plane,
                    face_triplet[1].plane,
                    face_triplet[2].plane,
                )
            except np.linalg.LinAlgError:
                continue
        raise ValueError(
            f"Vertex {v.id} has no three independent incident face planes"
        )

    def incident_faces(self, v: Vertex) -> list[Face]:
        """All faces incident to vertex ``v``, in the order encountered when
        rotating around the vertex via ``half_edge.opposite.next``.
        """
        if v.outgoing is None:
            return []
        seen: list[Face] = []
        h = v.outgoing
        for _ in range(64):  # bounded safety; manifold valence is always small
            if h.face is not None and h.face not in seen:
                seen.append(h.face)
            if h.opposite is None or h.opposite.next is None:
                break
            h = h.opposite.next
            if h is v.outgoing:
                break
        return seen

    def face_polygon(self, face: Face) -> list[np.ndarray]:
        """Return the ordered list of vertex positions bounding ``face``."""
        if face.half_edge is None:
            return []
        coords: list[np.ndarray] = []
        h = face.half_edge
        for _ in range(256):
            coords.append(self.vertex_position(h.origin))
            h = h.next
            if h is None or h is face.half_edge:
                break
        return coords

    # ---- editing operators --------------------------------------------------

    def face_shift(self, face: Face, delta: float) -> None:
        """Shift ``face``'s plane along its normal by ``delta``.

        Equivalent to ``plane.d += delta`` because the convention here is
        ``a*x + b*y + c*z = d`` with (a, b, c) a unit normal — increasing d
        moves the plane outward along the normal direction.

        Phase 1: no topology updates. The polyhedron may become invalid
        (concave / self-intersecting) for large shifts; Phase 2 will add
        the topology operators that fix that.
        """
        p = face.plane
        face.plane = Plane(a=p.a, b=p.b, c=p.c, d=p.d + float(delta))

    # ---- invariants (cheap checks for tests) --------------------------------

    def is_watertight(self) -> bool:
        """Every half-edge has a twin pointing the other way."""
        return all(
            h.opposite is not None and h.opposite.opposite is h for h in self.half_edges
        )

    def faces_close(self) -> bool:
        """Following h.next around each face must come back to h after a
        finite number of steps.
        """
        for face in self.faces:
            if face.half_edge is None:
                return False
            h = face.half_edge
            for _ in range(256):
                if h.next is None:
                    return False
                h = h.next
                if h is face.half_edge:
                    break
            else:
                return False
        return True


# ---- helpers ----------------------------------------------------------------


def three_plane_intersection(p1: Plane, p2: Plane, p3: Plane) -> np.ndarray:
    """Return the (x, y, z) point where three planes meet.

    Planes use convention ``a*x + b*y + c*z = d`` (the same as
    :class:`reconcile_tiers._core.plane.Plane`). Raises ``np.linalg.LinAlgError``
    if the three normals are coplanar (degenerate).
    """
    A = np.array(
        [[p1.a, p1.b, p1.c], [p2.a, p2.b, p2.c], [p3.a, p3.b, p3.c]],
        dtype=float,
    )
    b = np.array([p1.d, p2.d, p3.d], dtype=float)
    return np.linalg.solve(A, b)


# ---- fixtures ---------------------------------------------------------------


def make_cube(size: float = 1.0) -> HalfEdgePolyhedron:
    """Build an axis-aligned cube of edge length ``size`` centered at origin.

    6 faces (±X, ±Y, ±Z), 8 vertices, 12 edges = 24 half-edges. Outward
    normals. Each face is a quad walked counterclockwise when viewed from
    outside.

    This is the canonical test fixture for the half-edge structure: every
    vertex has exactly 3 incident faces, every edge has exactly 2 incident
    half-edges (one per side), and every face has exactly 4 boundary
    half-edges.
    """
    s = float(size) / 2.0

    # The 6 face planes. Convention: a*x + b*y + c*z = d, with (a,b,c)
    # the outward unit normal and d = signed offset from origin.
    plane_specs = [
        ("+X", 1.0, 0.0, 0.0, s),
        ("-X", -1.0, 0.0, 0.0, s),
        ("+Y", 0.0, 1.0, 0.0, s),
        ("-Y", 0.0, -1.0, 0.0, s),
        ("+Z", 0.0, 0.0, 1.0, s),
        ("-Z", 0.0, 0.0, -1.0, s),
    ]
    poly = HalfEdgePolyhedron()
    face_by_label: dict[str, Face] = {}
    for i, (label, a, b, c, d) in enumerate(plane_specs):
        face = Face(id=i, plane=Plane(a=a, b=b, c=c, d=d))
        poly.faces.append(face)
        face_by_label[label] = face

    # 8 vertices labelled by sign tuple (sx, sy, sz) ∈ {-1, +1}^3.
    # Vertex id = bitmask: (sx>0)<<2 | (sy>0)<<1 | (sz>0).
    def vid(sx: int, sy: int, sz: int) -> int:
        return ((sx > 0) << 2) | ((sy > 0) << 1) | (sz > 0)

    vert_by_sig: dict[tuple[int, int, int], Vertex] = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                v = Vertex(id=vid(sx, sy, sz))
                vert_by_sig[(sx, sy, sz)] = v
                poly.vertices.append(v)
    poly.vertices.sort(key=lambda v: v.id)

    # For each face, list the 4 boundary vertices in CCW order seen from
    # outside (i.e., right-hand rule with the outward normal).
    #
    # +X face (normal +x): viewed from +x, "up" is +Y and "right" is -Z;
    # CCW order: (+1,-1,-1) -> (+1,+1,-1) -> (+1,+1,+1) -> (+1,-1,+1).
    face_vertices: dict[str, list[tuple[int, int, int]]] = {
        "+X": [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
        "-X": [(-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1)],
        "+Y": [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)],
        "-Y": [(-1, -1, 1), (-1, -1, -1), (1, -1, -1), (1, -1, 1)],
        "+Z": [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
        "-Z": [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)],
    }

    # Build half-edges face by face. Each face contributes 4 half-edges,
    # `next`-linked into a cycle. Pair them across faces (twins) by
    # matching the (origin, destination) ↔ (destination, origin) directed-edge
    # pair.
    # Key directed-edge pairs by vertex id (Vertex itself is mutable / unhashable).
    he_by_directed: dict[tuple[int, int], HalfEdge] = {}
    he_id = 0
    for label, sigs in face_vertices.items():
        face = face_by_label[label]
        ring: list[HalfEdge] = []
        for src_sig in sigs:
            v = vert_by_sig[src_sig]
            he = HalfEdge(id=he_id, origin=v, face=face)
            he_id += 1
            poly.half_edges.append(he)
            ring.append(he)
        # Link `next` and record directed-edge pairs.
        for i, he in enumerate(ring):
            he.next = ring[(i + 1) % 4]
            dst = ring[(i + 1) % 4].origin
            he_by_directed[(he.origin.id, dst.id)] = he
        face.half_edge = ring[0]

    # Twin pairing: each (a -> b) directed edge has a partner (b -> a).
    for (src_id, dst_id), he in he_by_directed.items():
        twin = he_by_directed.get((dst_id, src_id))
        if twin is None:
            raise RuntimeError(
                f"missing twin for half-edge {he.id} ({src_id}->{dst_id})"
            )
        he.opposite = twin

    # Set each vertex's outgoing pointer to one of its outgoing half-edges.
    # Choose deterministically: the lowest-id half-edge that originates at
    # this vertex.
    by_origin: dict[int, list[HalfEdge]] = {}
    for he in poly.half_edges:
        by_origin.setdefault(he.origin.id, []).append(he)
    for v in poly.vertices:
        outs = by_origin.get(v.id, [])
        if outs:
            v.outgoing = min(outs, key=lambda h: h.id)

    return poly


# ---- general-purpose constructor --------------------------------------------


def build_from_planar_polygons(
    polygons: list[tuple[list[list[float]], Plane]],
    *,
    coord_tol: float = 1e-3,
) -> HalfEdgePolyhedron:
    """Build a half-edge polyhedron from a list of (corners, plane) pairs.

    Each input polygon is one face of the polyhedron, given as its boundary
    corners (in counterclockwise order seen from outside, right-hand rule with
    the face's outward normal) plus its plane equation. Vertices shared
    between faces are identified by coordinate proximity (within `coord_tol`).
    Twin half-edges are paired by matching directed edges (src_vid, dst_vid)
    against (dst_vid, src_vid).

    Raises ValueError if the input is not a watertight closed manifold —
    i.e. if any directed edge has no twin, or if a vertex ends up with
    fewer than three incident faces.

    The corner coordinates supplied by the caller are used only to identify
    which vertex each corner belongs to. After construction, calling
    ``vertex_position(v)`` derives the coordinates from the three incident
    face planes. If the input is consistent (corners actually lie at the
    correct plane intersections), the derived coordinates will match the
    input within ~coord_tol.
    """
    # 1. Cluster corners across all polygons into Vertex IDs.
    centroids: list[np.ndarray] = []  # one per assigned vertex id

    def assign_vertex(pt: list[float]) -> int:
        coord = np.asarray(pt, dtype=float)
        for vid, c in enumerate(centroids):
            if float(np.linalg.norm(coord - c)) <= coord_tol:
                return vid
        centroids.append(coord)
        return len(centroids) - 1

    polygon_vids: list[list[int]] = []
    for corners, _plane in polygons:
        if len(corners) < 3:
            raise ValueError(f"face has only {len(corners)} corners; need 3+")
        vids = [assign_vertex(c) for c in corners]
        # Reject degenerate self-loops (consecutive corners that snap to the
        # same vertex id within tolerance). They would produce zero-length
        # edges that have no valid twin.
        for i in range(len(vids)):
            if vids[i] == vids[(i + 1) % len(vids)]:
                raise ValueError(
                    f"face polygon has duplicate consecutive vertices "
                    f"(vid {vids[i]} at corners {i} and {(i + 1) % len(vids)})"
                )
        polygon_vids.append(vids)

    poly = HalfEdgePolyhedron()
    for vid in range(len(centroids)):
        poly.vertices.append(Vertex(id=vid))

    # 2. Create faces + half-edge rings.
    he_by_directed: dict[tuple[int, int], HalfEdge] = {}
    he_id = 0
    for face_idx, (vids, (_corners, plane)) in enumerate(
        zip(polygon_vids, polygons, strict=True)
    ):
        face = Face(id=face_idx, plane=plane)
        poly.faces.append(face)
        ring: list[HalfEdge] = []
        for vid in vids:
            he = HalfEdge(id=he_id, origin=poly.vertices[vid], face=face)
            he_id += 1
            poly.half_edges.append(he)
            ring.append(he)
        for i, he in enumerate(ring):
            he.next = ring[(i + 1) % len(ring)]
            dst_vid = ring[(i + 1) % len(ring)].origin.id
            key = (he.origin.id, dst_vid)
            if key in he_by_directed:
                previous_face = he_by_directed[key].face
                previous_face_id = (
                    previous_face.id if previous_face is not None else "?"
                )
                raise ValueError(
                    f"duplicate directed edge {key}: faces {face_idx} and "
                    f"{previous_face_id} "
                    "share orientation — not a manifold"
                )
            he_by_directed[key] = he
        face.half_edge = ring[0]

    # 3. Twin-pair directed edges. A watertight manifold has every (a, b)
    # paired with exactly one (b, a) on the other side of the same edge.
    unpaired: list[tuple[int, int]] = []
    for (src, dst), he in he_by_directed.items():
        twin = he_by_directed.get((dst, src))
        if twin is None:
            unpaired.append((src, dst))
            continue
        he.opposite = twin
    if unpaired:
        raise ValueError(
            f"non-watertight: {len(unpaired)} half-edges have no twin "
            f"(first few: {unpaired[:5]})"
        )

    # 4. Pick an outgoing half-edge per vertex.
    by_origin: dict[int, list[HalfEdge]] = {}
    for he in poly.half_edges:
        by_origin.setdefault(he.origin.id, []).append(he)
    for v in poly.vertices:
        outs = by_origin.get(v.id, [])
        if not outs:
            raise ValueError(f"vertex {v.id} has no outgoing half-edge")
        v.outgoing = min(outs, key=lambda h: h.id)

    # 5. Verify every vertex has at least 3 incident faces (manifold).
    for v in poly.vertices:
        if len(poly.incident_faces(v)) < 3:
            raise ValueError(
                f"vertex {v.id} has only {len(poly.incident_faces(v))} "
                "incident faces; need 3 for a 3-manifold"
            )

    return poly


def make_gable_house(
    width: float = 6.0,
    depth: float = 8.0,
    eave_height: float = 2.5,
    ridge_rise: float = 1.5,
) -> HalfEdgePolyhedron:
    """Build a simple gable-roofed house: rectangular footprint, two side
    walls, two gable end walls (pentagons), two roof slopes meeting at a
    ridge along Z. 7 faces, 10 vertices, 15 edges.

    Used as a realistic test fixture for tirana-shaped buildings — exercises
    pentagon faces and a non-axis-aligned roof, both absent from the cube.
    """
    w, d, h, r = float(width), float(depth), float(eave_height), float(ridge_rise)

    # Vertices (caller-supplied coords; will be re-derived from face planes).
    floor_nn = [0.0, 0.0, 0.0]  # -X, -Z corner of the floor
    floor_pn = [w, 0.0, 0.0]
    floor_pp = [w, 0.0, d]
    floor_np = [0.0, 0.0, d]
    eave_nn = [0.0, h, 0.0]
    eave_pn = [w, h, 0.0]
    eave_pp = [w, h, d]
    eave_np = [0.0, h, d]
    ridge_n = [w / 2.0, h + r, 0.0]
    ridge_p = [w / 2.0, h + r, d]

    # Plane equations (a*x + b*y + c*z = d). Outward-normal convention.
    # Floor faces -Y (down).
    plane_floor = Plane(a=0.0, b=-1.0, c=0.0, d=0.0)
    # +Z gable end wall (faces +Z).
    plane_gable_pz = Plane(a=0.0, b=0.0, c=1.0, d=d)
    # -Z gable end wall (faces -Z).
    plane_gable_nz = Plane(a=0.0, b=0.0, c=-1.0, d=0.0)
    # +X side wall (faces +X). Wait — this is more nuanced. Side walls in a
    # standard gable end at the eave; the gable triangle is part of the gable
    # end wall, not the side wall. So +X side wall is just a rectangle from
    # floor to eave.
    plane_side_px = Plane(a=1.0, b=0.0, c=0.0, d=w)
    plane_side_nx = Plane(a=-1.0, b=0.0, c=0.0, d=0.0)
    # +X roof slope: rises from +X eave to ridge. Normal points "outward and
    # up". Compute from two edge vectors.
    # Edge 1: ridge_n - eave_pn = (-w/2, +r, 0)
    # Edge 2: ridge_p - ridge_n = (0, 0, d) (actually a face direction)
    # Outward normal = cross(edge along z, edge from eave to ridge) normalized,
    # pointing away from the interior. For the +X slope, normal has +X and +Y.
    nx_pos = r
    ny_pos = w / 2.0
    nrm = (nx_pos**2 + ny_pos**2) ** 0.5
    nx_pos /= nrm
    ny_pos /= nrm
    plane_roof_px = Plane(
        a=nx_pos, b=ny_pos, c=0.0, d=nx_pos * w + ny_pos * h
    )
    plane_roof_nx = Plane(
        a=-nx_pos, b=ny_pos, c=0.0, d=-nx_pos * 0.0 + ny_pos * h
    )

    # Winding orders verified by cross-product to give the stated outward
    # normal. Each pair of adjacent faces traverses their shared edge in
    # opposite directions, which is required for build_from_planar_polygons
    # to twin-pair the half-edges.
    polygons: list[tuple[list[list[float]], Plane]] = [
        # Floor (outward -Y).
        ([floor_nn, floor_pn, floor_pp, floor_np], plane_floor),
        # +X side wall (outward +X).
        ([floor_pn, eave_pn, eave_pp, floor_pp], plane_side_px),
        # -X side wall (outward -X).
        ([floor_nn, floor_np, eave_np, eave_nn], plane_side_nx),
        # -Z gable end pentagon (outward -Z).
        ([floor_nn, eave_nn, ridge_n, eave_pn, floor_pn], plane_gable_nz),
        # +Z gable end pentagon (outward +Z).
        ([floor_pp, eave_pp, ridge_p, eave_np, floor_np], plane_gable_pz),
        # +X roof slope (outward +X, +Y).
        ([eave_pn, ridge_n, ridge_p, eave_pp], plane_roof_px),
        # -X roof slope (outward -X, +Y).
        ([eave_nn, eave_np, ridge_p, ridge_n], plane_roof_nx),
    ]
    return build_from_planar_polygons(polygons)
