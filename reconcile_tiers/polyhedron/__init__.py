"""Half-edge polyhedron with face-plane geometry.

Experimental. Implements the representation from Geniet, Brédif & Vallet,
*Editing Watertight Manifold Polyhedra using Face Shifts with Automatic
Topological Updates and Edge Flips* (ISPRS 2024). Faces carry plane
equations; vertex coordinates are derived from the intersection of
adjacent face planes. The structure guarantees watertightness and
manifoldness by construction — corners cannot drift below the floor or
beyond the ridge unless the floor / ridge faces are explicitly removed.

Phase 1 (this commit): half-edge data structure, three-plane vertex
derivation, and face_shift (no topology updates yet). Phase 2 will add
face_creation, face_collapse, and edge_flip per the paper's Section 3.2.

Not wired into reconcile_tiers/build.py. Lives alongside the existing
selector for parallel evaluation.
"""

from reconcile_tiers.polyhedron.face_fit import (
    CoordinateDescentResult,
    FaceEvidence,
    ScanPointEvidence,
    coordinate_descent_face_shifts,
    evidence_from_face_corners,
    evidence_from_scan_points,
    signed_distance_cost,
)
from reconcile_tiers.polyhedron.face_selection import (
    CandidateFace,
    FaceSelectionResult,
    assemble_polyhedron,
    build_edge_incidence,
    face_selection_trace,
    generate_candidates,
    solve_face_selection_ilp,
)
from reconcile_tiers.polyhedron.half_edge import (
    Face,
    HalfEdge,
    HalfEdgePolyhedron,
    Vertex,
    build_from_planar_polygons,
    make_cube,
    make_gable_house,
    three_plane_intersection,
)
from reconcile_tiers.polyhedron.kinetic_partition import (
    BoundingPrism,
    ConvexCell,
    KineticEvent,
    cells_to_candidate_faces,
    kinetic_partition,
    label_cells_inside_outside,
)
from reconcile_tiers.polyhedron.payload_adapter import (
    EnvelopeCandidate,
    PayloadFace,
    build_envelope_polyhedra_from_tier_payload,
    build_polyhedron_from_tier_payload,
    build_room_shell_from_tier_payload,
    payload_envelope_candidates_from_tier_payload,
    payload_faces_for_room_shell,
    payload_faces_from_plane_evidence,
    payload_faces_from_tier_payload,
)
from reconcile_tiers.polyhedron.priors import DEFAULT, Priors, SelectionWeights
from reconcile_tiers.polyhedron.refinement import RefinementResult, refine_polyhedron
from reconcile_tiers.polyhedron.topology_events import (
    CoplanarFaceMergeResult,
    EdgeCollapseResult,
    EdgeFlipResult,
    FaceCollapseResult,
    FaceCreationResult,
    TopologyCounts,
    TopologyResolutionError,
    TopologyResolutionStep,
    TopologyResolutionTrace,
    resolve_single_adjacent_coplanar_face_merge,
    resolve_single_edge_collapse,
    resolve_single_edge_flip,
    resolve_single_face_creation,
    resolve_single_triangle_face_collapse,
    resolve_supported_topology_events,
)
from reconcile_tiers.polyhedron.trace_export import (
    export_topology_resolution,
    polyhedron_snapshot,
)
from reconcile_tiers.polyhedron.validity import (
    TopologicalEvent,
    ValidityIssue,
    detect_topological_events,
    validate_polyhedron,
)

__all__ = [
    "DEFAULT",
    "BoundingPrism",
    "CandidateFace",
    "ConvexCell",
    "CoordinateDescentResult",
    "CoplanarFaceMergeResult",
    "EdgeCollapseResult",
    "EdgeFlipResult",
    "EnvelopeCandidate",
    "Face",
    "FaceCollapseResult",
    "FaceCreationResult",
    "FaceEvidence",
    "FaceSelectionResult",
    "HalfEdge",
    "HalfEdgePolyhedron",
    "KineticEvent",
    "PayloadFace",
    "Priors",
    "RefinementResult",
    "ScanPointEvidence",
    "SelectionWeights",
    "TopologicalEvent",
    "TopologyCounts",
    "TopologyResolutionError",
    "TopologyResolutionStep",
    "TopologyResolutionTrace",
    "ValidityIssue",
    "Vertex",
    "assemble_polyhedron",
    "build_edge_incidence",
    "build_envelope_polyhedra_from_tier_payload",
    "build_from_planar_polygons",
    "build_polyhedron_from_tier_payload",
    "build_room_shell_from_tier_payload",
    "cells_to_candidate_faces",
    "coordinate_descent_face_shifts",
    "detect_topological_events",
    "evidence_from_face_corners",
    "evidence_from_scan_points",
    "export_topology_resolution",
    "face_selection_trace",
    "generate_candidates",
    "kinetic_partition",
    "label_cells_inside_outside",
    "make_cube",
    "make_gable_house",
    "payload_envelope_candidates_from_tier_payload",
    "payload_faces_for_room_shell",
    "payload_faces_from_plane_evidence",
    "payload_faces_from_tier_payload",
    "polyhedron_snapshot",
    "refine_polyhedron",
    "resolve_single_adjacent_coplanar_face_merge",
    "resolve_single_edge_collapse",
    "resolve_single_edge_flip",
    "resolve_single_face_creation",
    "resolve_single_triangle_face_collapse",
    "resolve_supported_topology_events",
    "signed_distance_cost",
    "solve_face_selection_ilp",
    "three_plane_intersection",
    "validate_polyhedron",
]
