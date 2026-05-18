"""Serve reconcile viewer assets and proxy Datafordeleren orthophoto WMTS tiles.

Usage:
  DATAFORDELEREN_API_KEY=... python reconcile/viewer_server.py

Then open:
  http://localhost:8765/viewer.html
"""

from __future__ import annotations

import sys
from pathlib import Path

# `reconcile/` is a symlink into archive/legacy-runtime/. The editable install's
# MAPPING covers `reconcile` and `reconcile_v2` but not `reconcile_tiers`, so we
# ensure the workspace root is on sys.path before importing across packages.
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import itertools
import json
import math
import os
import posixpath
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import split as shapely_split
from shapely.ops import unary_union
from shapely.validation import make_valid

from reconcile.extract3d.builder import extract_building
from reconcile.roof_algorithms_py import run_roof_algorithms
from reconcile.roof_algorithms_py.math_utils import plane_normal, plane_y_at
from reconcile_v2.graph_builder import build_topology_graph

HOST = "127.0.0.1"
PORT = int(os.environ.get("VIEWER_PORT", "8080"))
WMTS_BASE = "https://wmts.datafordeler.dk/GeoDanmarkOrto/orto_foraar_webm/1.0.0/WMTS"
SECRET_NAME = "datafordeler-graphql-api-key"
ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
CALIBRATION_PATH = ROOT_DIR.parent / ".context" / "alignment_calibration.json"
ROOF_PROPOSAL_LABELS_PATH = (
    ROOT_DIR.parent / ".context" / "v3_roof_proposal_labels.jsonl"
)
ROOF_PROPOSAL_SPLITS_PATH = (
    ROOT_DIR.parent / ".context" / "v3_roof_proposal_splits.jsonl"
)
PIPELINE_ROOT = ROOT_DIR.parent / "pipeline-outputs"
SCAN_CACHE_ROOT = ROOT_DIR.parent / ".scan-cache"
# `reconcile/` may be a symlink into archive/. Use the unresolved __file__ so
# flag queues always land in the workspace .context, alongside the cohort
# scanner's writes.
WORKSPACE_ROOT = Path(__file__).parent.parent
FLAG_QUEUES_ROOT = WORKSPACE_ROOT / ".context" / "flag-queues"
FLAG_CALIBRATION_ROOT = WORKSPACE_ROOT / ".context" / "flag-calibration"
ROOF_RATINGS_PATH = WORKSPACE_ROOT / ".context" / "roof_ratings.json"
ONTOLOGY_CACHE: dict[str, dict] = {}
V3_RESULTS_PATH = ROOT_DIR / "reconcile_v3_results.json"
V3_CACHE: dict[str, dict] = {}
V3_CACHE_MTIME: float = 0.0
# Phase A candidate faces — produced by `scripts/build_candidate_faces.py`.
# The viewer loads per-building slices on demand.
CANDIDATE_FACES_PATH_CANDIDATES: tuple[Path, ...] = (
    REPO_ROOT / ".context" / "candidate_faces_zone_v2_remainderfix" / "candidates.json",
    REPO_ROOT / ".context" / "candidate_faces_zone_v2_dedup" / "candidates.json",
    REPO_ROOT / "reports" / "candidate_faces" / "candidates.json",
    REPO_ROOT / "reports" / "candidate_faces_20260419" / "candidates.json",
)
CANDIDATE_FACES_CACHE: dict[str, dict] = {}
CANDIDATE_FACES_CACHE_MTIME: float = 0.0
# Phase B.1 reconstruction-solver selections — produced by
# `scripts/run_reconstruction_solver.py`. Joined with candidate faces at
# request time so the viewer can color the selected subset.
RECONSTRUCTION_PATH_CANDIDATES: tuple[Path, ...] = (
    REPO_ROOT / ".context" / "reconstruction_zone_v2_remainderfix" / "selections.json",
    REPO_ROOT / ".context" / "reconstruction_zone_v2" / "selections.json",
    REPO_ROOT / "reports" / "reconstruction_20260423" / "selections.json",
    REPO_ROOT / "reports" / "reconstruction_20260419_topologyfix" / "selections.json",
    REPO_ROOT / "reports" / "reconstruction_20260419" / "selections.json",
)
RECONSTRUCTION_CACHE: dict[str, dict] = {}
RECONSTRUCTION_CACHE_MTIME: float = 0.0
# Ridge/eave topology scoring — produced by
# `scripts/score_candidates_ridge_eave.py`. Per-building slice: best-pair
# scores for each candidate plus the pair geometry (ridge, medial axis,
# eaves, part OBB) for visual inspection.
RIDGE_EAVE_SCORES_PATH_CANDIDATES: tuple[Path, ...] = (
    REPO_ROOT / ".context" / "ridge_eave_scores_zone_v2_remainderfix" / "scores.json",
    REPO_ROOT / "reports" / "ridge_eave_scores_20260420" / "scores.json",
)
RIDGE_EAVE_CACHE: dict[str, dict] = {}
RIDGE_EAVE_CACHE_MTIME: float = 0.0
# Phase-2 raw-ceiling prototype — per-plane role + per-room archetype labels
# plus dormer/wing reconstruction geometry, produced by
# scripts/prototype_raw_ceiling_roles.py / prototype_dormer_reconstruction.py /
# prototype_wing_reconstruction.py. Served at /raw-ceiling-prototype as a
# combined payload the viewer fetches once on startup.
RAW_CEILING_ROLES_PATH = (
    ROOT_DIR.parent / "reports" / "raw_ceiling_prototype" / "roles.json"
)
RAW_CEILING_RECON_PATH = (
    ROOT_DIR.parent / "reports" / "raw_ceiling_prototype" / "reconstructions.json"
)
RAW_CEILING_PROTOTYPE_CACHE: dict = {}
RAW_CEILING_PROTOTYPE_CACHE_MTIME: tuple[float, float] = (0.0, 0.0)
# Per-surface overextend polygons produced by
# scripts/audit_computed_surface_extent_vs_raw.py — one polygon per
# roof-oblique/roof-flat surface showing the XZ region where the computed
# surface reaches beyond the union of overlapping raw ceiling planes,
# lifted to the surface's Y plane.
COMPUTED_OVEREXTEND_PATH = (
    ROOT_DIR.parent / "reports" / "computed_extent_vs_raw" / "overextend_polygons.json"
)
COMPUTED_OVEREXTEND_CACHE: dict = {}
COMPUTED_OVEREXTEND_CACHE_MTIME: float = 0.0
# Raw-ceiling orientation disagreement polygons produced by
# scripts/audit_raw_orientation_disagreement.py — one polygon per pair of
# raw ceiling planes that overlap in XZ but carry different fitted normals,
# i.e. a slope-split the pipeline may have flattened.
RAW_DISAGREEMENT_PATH = (
    ROOT_DIR.parent
    / "reports"
    / "raw_orientation_disagreement"
    / "disagreement_polygons.json"
)
RAW_DISAGREEMENT_CACHE: dict = {}
RAW_DISAGREEMENT_CACHE_MTIME: float = 0.0
# Raw-ceiling plane split overlays. The viewer can request versioned payloads:
#   /raw-ceiling-plane-splits?version=v1  (legacy scorer output)
#   /raw-ceiling-plane-splits?version=v2  (relation-first scorer output)
# V1 keeps the historical path under reports/, V2 is loaded from the migration
# workspace sidecar under .context/.
RAW_CEILING_PLANE_SPLITS_PATHS: dict[str, Path] = {
    "v1": ROOT_DIR.parent
    / "reports"
    / "raw_ceiling_plane_scorer"
    / "plane_extent_splits.json",
    "v2": ROOT_DIR.parent
    / ".context"
    / "raw_ceiling_plane_scorer_v2_full"
    / "plane_extent_splits.json",
}
RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION: dict[str, dict] = {"v1": {}, "v2": {}}
RAW_CEILING_PLANE_SPLITS_CACHE_MTIME_BY_VERSION: dict[str, float] = {
    "v1": 0.0,
    "v2": 0.0,
}
# Clean-ceiling replacement polygons produced by
# scripts/audit_noisy_slanted_ceiling_replacement.py — one polygon per
# noisy-slanted room, showing the computed oblique roof surface clipped
# to the room footprint (the "clean plane" that replaces the fragmented
# raw scan).
CEILING_REPLACEMENT_PATH = (
    ROOT_DIR.parent / "reports" / "noisy_slanted_ceilings" / "replacement_polygons.json"
)
CEILING_REPLACEMENT_CACHE: dict = {}
CEILING_REPLACEMENT_CACHE_MTIME: float = 0.0
# Phase 6/7: optional scored mirror of V3_RESULTS_PATH. When present, the
# queue endpoint filters to ``autonomy_label == "review"`` and sorts by
# uncertainty (|score - 0.5| asc).
V3_SCORED_PATH = ROOT_DIR / "reconcile_v3_results_scored.json"
# Map of proposal_id → {"score": float, "autonomy_label": str}
V3_SCORES: dict[str, dict] = {}
V3_SCORES_MTIME: float = 0.0
UNASSIGNED_PART_ID = "building-part:unassigned"
FULL_BUILDING_PART_ID = "building-part:full-building"

# Roof-focused viewer (viewer-roof.html). Loads the full roof pipeline results
# plus the scan-ceiling audit report once and serves compact per-building
# payloads. Sources:
#   * reconcile/roof_algorithms_py_results.json  (segments, clusters,
#     pre-selection candidate ceiling planes, committed roof surfaces)
#   * reconcile/buildings_3d.json                (raw scanned ceilings,
#     address/metadata, room floors)
#   * reports/scan_ceiling_support.json          (per-surface audit rows)
ROOF_RESULTS_PATH = ROOT_DIR / "roof_algorithms_py_results.json"
BUILDINGS_3D_PATH = ROOT_DIR / "buildings_3d.json"
ROOF_AUDIT_PATH = ROOT_DIR.parent / "reports" / "scan_ceiling_support.json"
ROOF_RESULTS_CACHE: dict[str, dict] = {}
ROOF_RESULTS_CACHE_MTIME: float = 0.0
BUILDINGS_3D_CACHE: dict[str, dict] = {}
BUILDINGS_3D_CACHE_MTIME: float = 0.0
ROOF_AUDIT_CACHE: dict[str, list[dict]] = {}
ROOF_AUDIT_CACHE_MTIME: float = 0.0
ROOF_INDEX_CACHE: list[dict] | None = None
ROOF_INDEX_CACHE_KEY: tuple[float, float, float] = (0.0, 0.0, 0.0)
TIER_INDEX_CACHE: list[dict] | None = None
TIER_INDEX_CACHE_KEY: tuple[float, float] = (0.0, 0.0)


# Coverage / Y-gap thresholds for `_gate_unsupported_v2_pieces`. A piece
# with no raw-scan support is dropped only when ≥50% of its XZ footprint
# lies under the union of supported pieces sitting ≥0.30 m higher. The
# 0.30 m gap protects intersection seams and side-by-side oblique faces
# at a shared ridge from being treated as "interior".
SLANTED_PIECE_DOMINATION_COVERAGE = 0.50
SLANTED_PIECE_DOMINATION_Y_GAP_M = 0.30

# Flat fallback pieces in viewer-tiers are allowed to coexist with slanted
# pieces in the same room. They are suppressed only when an exterior slanted
# final-layer piece is physically above the flat patch at the overlapping XZ
# region.
FLAT_PATCH_MAX_INCL_DEG = 5.0
FLAT_PATCH_MAX_Y_SPAN_M = 0.10
FLAT_PATCH_SLANTED_ABOVE_CLEARANCE_M = 0.05
FLAT_ROOM_RAW_NOISE_BELOW_TOL_M = 0.30
MIN_CEILING_FALLBACK_AREA_M2 = 0.05
FLAT_PARTITION_ROLES = {"roof_flat", "ceiling_cap"}


def _piece_has_no_support(piece: dict) -> bool:
    """Drop-candidate predicate. A piece has zero raw-scan support when
    all three indicators are zero: no `chain_ids`, no
    `piece_anchor_chain_count`, and `creator_rain_area_fraction == 0`.
    These are pure plane extrapolations the scorer's own provenance
    fields flag as suspect — e.g. `provenance_relevance_flag =
    'suspect_interior_slice'`.
    """
    if piece.get("chain_ids"):
        return False
    if int(piece.get("piece_anchor_chain_count") or 0) > 0:
        return False
    if float(piece.get("creator_rain_area_fraction") or 0.0) > 0.0:
        return False
    return True


def _piece_has_chain_or_anchor(piece: dict) -> bool:
    """Dominator predicate. A piece can dominate another only if it has
    direct alignment to a raw_ceiling_plane chain or a piece-level anchor.
    `creator_rain_area_fraction` is intentionally excluded — it describes
    the source segments' weather exposure, not whether the piece's plane
    fits the raw scan, so a rain-only piece is too weak to kick out
    another piece.
    """
    if piece.get("chain_ids"):
        return True
    if int(piece.get("piece_anchor_chain_count") or 0) > 0:
        return True
    return False


def _gate_unsupported_v2_pieces(pieces: list[dict]) -> list[dict]:
    """Drop V2 final pieces with no raw-scan support that are XZ-shadowed
    by a chain-or-anchor-supported sibling sitting strictly above them.

    A non-seam piece P is dropped iff:
      (1) P has zero scan support (`_piece_has_no_support` is True)
      (2) the union of pieces with chain or anchor support whose mean_y
          is at least `SLANTED_PIECE_DOMINATION_Y_GAP_M` above P covers
          ≥ `SLANTED_PIECE_DOMINATION_COVERAGE` of P's XZ area

    Intersection seams have no chain support by construction (they're
    geometric stitches between two parent pieces). They inherit their
    parent's decision via `target_element_id`: a seam is dropped iff its
    parent target has zero kept non-seam pieces in the cohort.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return pieces

    def _xz_poly(corners):
        if not corners or len(corners) < 3:
            return None
        try:
            poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1e-4:
                return None
            return poly
        except Exception:
            return None

    def _mean_y(corners):
        return sum(float(c[1]) for c in corners) / len(corners) if corners else None

    polys = [_xz_poly(p.get("corners") or []) for p in pieces]
    ys = [_mean_y(p.get("corners") or []) for p in pieces]
    keep_flags = [True] * len(pieces)

    # First pass: gate non-seam pieces. Seams are deferred to the second
    # pass because their drop decision depends on parent results.
    for i, piece in enumerate(pieces):
        if (piece.get("piece_role") or "") == "intersection_seam":
            continue
        if not _piece_has_no_support(piece):
            continue
        self_poly = polys[i]
        self_y = ys[i]
        if self_poly is None or self_y is None:
            continue
        dom_polys = []
        for j, other in enumerate(pieces):
            if j == i:
                continue
            if not _piece_has_chain_or_anchor(other):
                continue
            other_poly = polys[j]
            other_y = ys[j]
            if other_poly is None or other_y is None:
                continue
            if other_y < self_y + SLANTED_PIECE_DOMINATION_Y_GAP_M:
                continue
            if self_poly.intersects(other_poly):
                dom_polys.append(other_poly)
        if not dom_polys:
            continue
        try:
            covered = self_poly.intersection(unary_union(dom_polys)).area
        except Exception:
            covered = 0.0
        if (
            self_poly.area > 0
            and covered / self_poly.area >= SLANTED_PIECE_DOMINATION_COVERAGE
        ):
            keep_flags[i] = False

    # Build parent → kept-non-seam-count, used to inherit drops onto seams.
    parent_kept_count: dict[str, int] = {}
    for i, piece in enumerate(pieces):
        if (piece.get("piece_role") or "") == "intersection_seam":
            continue
        target = str(piece.get("target_element_id") or "")
        if not target:
            continue
        if keep_flags[i]:
            parent_kept_count[target] = parent_kept_count.get(target, 0) + 1
        else:
            parent_kept_count.setdefault(target, 0)

    # Second pass: seams whose parent target has zero kept non-seam
    # pieces inherit the drop. Seams whose parent is absent from the
    # cohort default to keep (they are standalone seams from a target
    # that wasn't admitted at all and shouldn't have been emitted, but
    # we don't have signal to discriminate them here).
    for i, piece in enumerate(pieces):
        if (piece.get("piece_role") or "") != "intersection_seam":
            continue
        target = str(piece.get("target_element_id") or "")
        if target in parent_kept_count and parent_kept_count[target] == 0:
            keep_flags[i] = False

    return [p for p, keep in zip(pieces, keep_flags, strict=True) if keep]


def _load_v2_final_pieces(uuid: str, *, gate_unsupported: bool = False) -> list[dict]:
    """Return the V2 raw-split final-layer pieces for one building.

    Loads the V2 splits sidecar on demand via the same cache the
    `/raw-ceiling-plane-splits` handler uses. Returns an empty list when
    the sidecar is unavailable or the building has no final-layer pieces.

    By default this matches the `viewer.html` v2 final-layer overlay exactly:
    final-layer pieces, committed oblique pieces, and intersection seams are
    included; `overlay_suppressed` pieces are excluded. Callers that explicitly
    want the older stricter tier-view gate can pass `gate_unsupported=True`.
    """
    global RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION
    global RAW_CEILING_PLANE_SPLITS_CACHE_MTIME_BY_VERSION
    version = "v2"
    path = RAW_CEILING_PLANE_SPLITS_PATHS[version]
    mtime = path.stat().st_mtime if path.exists() else 0.0
    if not RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION.get(
        version
    ) or mtime != RAW_CEILING_PLANE_SPLITS_CACHE_MTIME_BY_VERSION.get(version, 0.0):
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except Exception:
                data = {}
        RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION[version] = {
            "buildings": data.get("buildings") or {},
            "available": bool(data),
            "version": version,
        }
        RAW_CEILING_PLANE_SPLITS_CACHE_MTIME_BY_VERSION[version] = mtime
    buildings = (
        RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION[version].get("buildings") or {}
    )
    pieces = buildings.get(uuid) or []
    filtered = [
        p
        for p in pieces
        if (
            not p.get("overlay_suppressed")
            and (
                p.get("final_layer") is True
                or p.get("target_kind") == "committed_oblique"
                or p.get("piece_role") == "intersection_seam"
            )
        )
    ]
    if gate_unsupported:
        return _gate_unsupported_v2_pieces(filtered)
    return filtered


def _load_v2_full_model_pieces(uuid: str) -> list[dict]:
    """Return V2 pieces suitable for the tier/full-model roof surface.

    The raw split overlay intentionally includes diagnostic seam and committed
    oblique context. Rendering those in the final preview stacks overlapping
    planes over the selected roof surface, so the full-model path only uses
    actual final-layer pieces.
    """
    return [
        p
        for p in _load_v2_final_pieces(uuid)
        if p.get("final_layer") is True
        and (p.get("piece_role") or "") != "intersection_seam"
    ]


def _v2_final_pieces_xz_union(uuid: str):
    """XZ union of V2 raw-split final-layer pieces for one building.

    Returns a Shapely geometry (possibly empty) or None if the sidecar is
    unavailable or no pieces fit.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return None
    polys = []
    for p in _load_v2_full_model_pieces(uuid):
        corners = p.get("corners") or []
        if len(corners) < 3:
            continue
        try:
            poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            polys.append(poly)
        except Exception:
            continue
    if not polys:
        return None
    return unary_union(polys)


def _polygon_xz_from_corners(corners):
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return None
    if len(corners or []) < 3:
        return None
    try:
        poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 1e-6:
            return None
        return poly
    except Exception:
        return None


def _slanted_pieces_xz_union(slanted_pieces: list[dict]):
    try:
        from shapely.ops import unary_union
    except ImportError:
        return None
    polys = []
    for piece in slanted_pieces:
        poly = _polygon_xz_from_corners(piece.get("corners") or [])
        if poly is not None:
            polys.append(poly)
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return None


def _ceiling_cap_is_covered_by_slanted_roof(
    corners, slanted_roof_xz, *, min_coverage: float = 0.50
) -> bool:
    if slanted_roof_xz is None:
        return False
    cap = _polygon_xz_from_corners(corners or [])
    if cap is None:
        return False
    try:
        covered = cap.intersection(slanted_roof_xz).area
    except Exception:
        return False
    return covered / cap.area >= min_coverage


def _ceiling_cap_is_covered_by_nearby_slanted_roof(
    corners,
    slanted_pieces: list[dict],
    *,
    min_coverage: float = 0.50,
    max_vertical_gap_m: float = 0.50,
) -> bool:
    """Return True only when a slanted roof overlaps in XZ and height.

    Cross-story gap ceilings can occur far below the roof on split-level
    buildings. XZ-only suppression removes those legitimate intermediate lids
    and leaves their paired floor diagnostic to be rendered through rooms.
    """
    try:
        from shapely.ops import unary_union
    except ImportError:
        return False
    if not corners:
        return False
    cap_max_y = max(float(c[1]) for c in corners)
    nearby = []
    for piece in slanted_pieces or []:
        piece_corners = piece.get("corners") or []
        if not piece_corners:
            continue
        piece_min_y = min(float(c[1]) for c in piece_corners)
        if piece_min_y > cap_max_y + max_vertical_gap_m:
            continue
        poly = _polygon_xz_from_corners(piece_corners)
        if poly is not None:
            nearby.append(poly)
    if not nearby:
        return False
    try:
        return _ceiling_cap_is_covered_by_slanted_roof(
            corners, unary_union(nearby), min_coverage=min_coverage
        )
    except Exception:
        return False


def _filter_gap_ceiling_caps_for_full_model(
    records: list[dict], slanted_roof_xz
) -> list[dict]:
    """Remove gap ceiling lids covered by real slanted roof pieces.

    Gap ceilings are closure artifacts. In the full-model/tier preview, V2
    slanted pieces are the roof source for sloped areas; rendering covered gap
    lids as extra roof polygons creates horizontal/striped planes through the
    same footprint.
    """
    out = []
    for record in records:
        typ = str(record.get("type") or "").lower()
        if "ceiling" in typ and _ceiling_cap_is_covered_by_slanted_roof(
            record.get("corners"), slanted_roof_xz
        ):
            continue
        out.append(record)
    return out


def _filter_cross_floor_gap_ceiling_lids_for_full_model(
    records: list[dict], slanted_pieces: list[dict]
) -> list[dict]:
    out = []
    for record in records:
        item = dict(record)
        if item.get(
            "ceiling_corners"
        ) and _ceiling_cap_is_covered_by_nearby_slanted_roof(
            item.get("ceiling_corners"), slanted_pieces
        ):
            item["ceiling_corners"] = None
        out.append(item)
    return out


def _dormer_cutouts_for_uuid(uuid: str) -> list[tuple]:
    """Per-cutout `(xz_polygon, plane_coeffs)` for a building.

    Each entry of `roof_surfaces.oblique[i].cutout_holes` is a 4-point
    quad sitting on the oblique cluster plane (produced by
    `roof_algorithms_py/dormer_geometry.build_dormer_geometry`). The XZ
    polygon is what we subtract from ceiling pieces; the plane coeffs
    are fit through the original 3D corners so we can recover the
    dormer-plane Y at any (x, z) inside the cutout. We need that Y when
    lifting interior hole rings back to 3D — using the V2 piece's own
    fitted plane drifts a few cm against the cluster plane, leaving a
    visible slit where the cheek bottom meets the cutout edge.
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return []
    roof = ROOF_RESULTS_CACHE.get(uuid) or {}
    surfaces = (roof.get("roof_surfaces") or {}).get("oblique") or []
    out: list[tuple] = []
    for surf in surfaces:
        for quad in surf.get("cutout_holes") or []:
            if not isinstance(quad, list) or len(quad) < 3:
                continue
            try:
                poly = Polygon([(float(c[0]), float(c[2])) for c in quad])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
            except Exception:
                continue
            coeffs = _fit_plane_coeffs(quad)
            if coeffs is None:
                continue
            out.append((poly, coeffs))
    return out


def _dormer_cutouts_xz_for_uuid(uuid: str):
    """XZ union of all dormer cutout footprints (or None)."""
    try:
        from shapely.ops import unary_union
    except ImportError:
        return None
    cutouts = _dormer_cutouts_for_uuid(uuid)
    if not cutouts:
        return None
    return unary_union([c[0] for c in cutouts])


def _fit_plane_coeffs(corners):
    """Fit a plane a*x + b*y + c*z = d through corners. Returns (a,b,c,d).

    Uses SVD on the centered points; the smallest singular vector is the
    plane normal. Returns None if fitting fails or the plane is vertical
    (no Y can be solved at a given (x,z)).
    """
    import numpy as np

    try:
        pts = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in corners])
    except Exception:
        return None
    if pts.shape[0] < 3:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    a, b, c = float(normal[0]), float(normal[1]), float(normal[2])
    if abs(b) < 1e-6:
        return None
    d = float(normal @ centroid)
    return (a, b, c, d)


def _ring_to_3d_on_plane(ring, plane_coeffs):
    """Lift a Shapely 2D ring to 3D via plane a*x+b*y+c*z=d (solve y)."""
    a, b, c, d = plane_coeffs
    pts = list(ring)
    # Shapely closes rings; drop the repeated last point.
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    return [[x, (d - a * x - c * z) / b, z] for (x, z) in pts]


def _shapely_difference_to_3d_pieces(
    xz_poly,
    subtract_union,
    plane_coeffs,
    min_area,
    interior_planes=None,
):
    """Subtract `subtract_union` from `xz_poly`, lift remaining parts to 3D.

    Exterior rings are lifted via `plane_coeffs` (the surface this piece
    came from). Interior rings — i.e. dormer cutouts punched into the
    surface — are lifted via the matching cutout's source plane when
    `interior_planes=[(xz_poly, plane_coeffs), …]` is provided. That
    keeps the hole edge coplanar with the cheek bottoms (built on the
    same cluster plane) instead of drifting onto this piece's fitted
    plane.

    Returns a list of `{poly, holes}` dicts ready for the viewer.
    """
    try:
        from shapely.geometry import MultiPolygon
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        return []
    diff = xz_poly.difference(subtract_union) if subtract_union is not None else xz_poly
    if diff.is_empty:
        return []
    parts = diff.geoms if isinstance(diff, MultiPolygon) else [diff]
    out: list[dict] = []
    for part in parts:
        if part.is_empty or part.area < min_area:
            continue
        poly3 = _ring_to_3d_on_plane(part.exterior.coords, plane_coeffs)
        if poly3 is None:
            continue
        holes3 = []
        for interior in getattr(part, "interiors", []) or []:
            hole_plane = plane_coeffs
            if interior_planes:
                # representative_point() lies inside the hole, so the cutout
                # whose XZ poly contains it is the one to source Y from.
                probe = ShapelyPolygon(interior).representative_point()
                for ip_poly, ip_coeffs in interior_planes:
                    if ip_poly.contains(probe):
                        hole_plane = ip_coeffs
                        break
            hring = _ring_to_3d_on_plane(interior.coords, hole_plane)
            if hring is not None:
                holes3.append(hring)
        out.append({"poly": poly3, "holes": holes3})
    return out


def _backend_arrangement_pieces_for_uuid(uuid: str) -> list[dict]:
    roof = ROOF_RESULTS_CACHE.get(uuid) or {}
    roof_surfaces = roof.get("roof_surfaces") or {}
    arranged = roof_surfaces.get("oblique") or roof_surfaces.get("oblique_split") or []
    return [
        {
            "corners": piece["corners"],
            "holes": piece.get("holes") or [],
            "source": "roof_arrangement",
            "arrangement_cell_id": piece.get("arrangement_cell_id"),
            "roof_hypothesis_id": piece.get("roof_hypothesis_id"),
            "intersection_kind": piece.get("intersection_kind"),
        }
        for piece in arranged
        if len(piece.get("corners") or []) >= 3
    ]


def _use_backend_arrangement_for_slanted(uuid: str) -> bool:
    return bool(_backend_arrangement_pieces_for_uuid(uuid)) and (
        _roof_results_newer_than_v2_sidecar()
    )


def _active_slanted_source_pieces_for_uuid(uuid: str) -> list[dict]:
    if _use_backend_arrangement_for_slanted(uuid):
        return _backend_arrangement_pieces_for_uuid(uuid)
    return _load_v2_full_model_pieces(uuid)


def _active_slanted_pieces_xz_union(uuid: str):
    try:
        from shapely.ops import unary_union
    except ImportError:
        return None
    polys = []
    for piece in _active_slanted_source_pieces_for_uuid(uuid):
        corners = piece.get("corners") or piece.get("poly") or []
        poly = _polygon_xz_from_corners(corners)
        if poly is not None:
            polys.append(poly)
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return None


def _combined_ceiling_subtraction(uuid: str):
    """Active slanted XZ union plus dormer-cutout XZ union (or None)."""
    try:
        from shapely.ops import unary_union
    except ImportError:
        return None
    parts = [
        g
        for g in (
            _active_slanted_pieces_xz_union(uuid),
            _dormer_cutouts_xz_for_uuid(uuid),
        )
        if g is not None
    ]
    if not parts:
        return None
    return unary_union(parts)


def _plane_y_from_coeffs(coeffs, x: float, z: float) -> float | None:
    if coeffs is None:
        return None
    a, b, c, d = coeffs
    if abs(b) < 1e-9:
        return None
    return float((d - a * x - c * z) / b)


def _plane_inclination_deg(coeffs) -> float | None:
    if coeffs is None:
        return None
    a, b, c, _d = coeffs
    if abs(b) < 1e-9:
        return None
    grad_x = -float(a) / float(b)
    grad_z = -float(c) / float(b)
    return math.degrees(math.atan(math.hypot(grad_x, grad_z)))


def _y_span(corners) -> float:
    ys = [float(c[1]) for c in corners or [] if len(c) >= 2]
    if not ys:
        return 0.0
    return max(ys) - min(ys)


def _is_flat_patch(corners, coeffs) -> bool:
    incl = _plane_inclination_deg(coeffs)
    return (incl is not None and incl <= FLAT_PATCH_MAX_INCL_DEG) or _y_span(
        corners
    ) <= FLAT_PATCH_MAX_Y_SPAN_M


def _room_flat_reference_y(room: dict) -> float | None:
    poly = room.get("ceiling_polygon") or []
    if len(poly) >= 3 and _y_span(poly) <= FLAT_PATCH_MAX_Y_SPAN_M:
        return sum(float(p[1]) for p in poly) / len(poly)
    ridge = room.get("ceiling_ridge_height")
    eave = room.get("ceiling_eave_height")
    try:
        ridge_f = float(ridge)
        eave_f = float(eave)
    except (TypeError, ValueError):
        return None
    if abs(ridge_f - eave_f) <= FLAT_PATCH_MAX_Y_SPAN_M:
        return (ridge_f + eave_f) * 0.5
    return None


def _raw_plane_allowed_for_flat_room(room: dict, corners) -> bool:
    if room.get("ceiling_type") != "flat":
        return True
    ref_y = _room_flat_reference_y(room)
    if ref_y is None:
        return True
    ys = [float(c[1]) for c in corners or [] if len(c) >= 2]
    if not ys:
        return False
    # Flat-room raw ceiling planes sometimes contain wall faces from noMesh
    # ingest. A real ceiling patch should not dip far below the flat ceiling
    # consensus; high slanted patches are still allowed so mixed rooms survive.
    return (ref_y - min(ys)) <= FLAT_ROOM_RAW_NOISE_BELOW_TOL_M


def _flat_partition_patches_by_room(uuid: str) -> dict[int, list[dict]]:
    roof = ROOF_RESULTS_CACHE.get(uuid) or {}
    patches = (roof.get("ceiling_partitions") or {}).get("flat") or []
    out: dict[int, list[dict]] = defaultdict(list)
    for patch in patches:
        if patch.get("flat_role") not in FLAT_PARTITION_ROLES:
            continue
        room_index = patch.get("room_index")
        if not isinstance(room_index, int):
            continue
        poly = patch.get("poly") or []
        if len(poly) < 3:
            continue
        out[room_index].append(patch)
    return out


def _polygon_xz_with_holes(corners, holes=None):
    if len(corners or []) < 3:
        return None
    try:
        shell = [(float(c[0]), float(c[2])) for c in corners]
        hole_rings = [
            [(float(c[0]), float(c[2])) for c in hole]
            for hole in (holes or [])
            if isinstance(hole, list) and len(hole) >= 3
        ]
        poly = Polygon(shell, hole_rings)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 1e-6:
            return None
        return poly
    except Exception:
        return None


def _slanted_roof_xz_above_flat_patch(uuid: str, flat_xz, flat_coeffs):
    polys = []
    use_backend = _use_backend_arrangement_for_slanted(uuid)
    for piece in _active_slanted_source_pieces_for_uuid(uuid):
        corners = piece.get("corners") or piece.get("poly") or []
        slanted_xz = _polygon_xz_from_corners(corners)
        if slanted_xz is None:
            continue
        try:
            overlap = flat_xz.intersection(slanted_xz)
        except Exception:
            continue
        if overlap.is_empty or overlap.area <= 1e-6:
            continue
        if use_backend:
            polys.append(slanted_xz)
            continue
        slanted_coeffs = _fit_plane_coeffs(corners)
        if slanted_coeffs is None:
            continue
        probe = overlap.representative_point()
        flat_y = _plane_y_from_coeffs(flat_coeffs, probe.x, probe.y)
        slanted_y = _plane_y_from_coeffs(slanted_coeffs, probe.x, probe.y)
        if flat_y is None or slanted_y is None:
            continue
        if slanted_y >= flat_y + FLAT_PATCH_SLANTED_ABOVE_CLEARANCE_M:
            polys.append(slanted_xz)
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return None


def _flat_patch_subtraction(uuid: str, flat_xz, flat_coeffs):
    parts = [
        _slanted_roof_xz_above_flat_patch(uuid, flat_xz, flat_coeffs),
        _dormer_cutouts_xz_for_uuid(uuid),
    ]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    try:
        return unary_union(parts)
    except Exception:
        return None


def _append_ceiling_fallback_pieces(
    out: list[dict],
    *,
    xz_poly,
    plane_coeffs,
    subtract_union,
    cutouts,
    source: str,
    room_index: int | None = None,
    plane_index: int | None = None,
    partition_id: str | None = None,
) -> int:
    pieces = _shapely_difference_to_3d_pieces(
        xz_poly,
        subtract_union,
        plane_coeffs,
        min_area=MIN_CEILING_FALLBACK_AREA_M2,
        interior_planes=cutouts,
    )
    for piece in pieces:
        piece["source"] = source
        if room_index is not None:
            piece["room_index"] = room_index
        if plane_index is not None:
            piece["plane_index"] = plane_index
        if partition_id:
            piece["partition_id"] = partition_id
        out.append(piece)
    return len(pieces)


def _raw_ceiling_fallback_for_uuid(uuid: str, building: dict) -> list[dict]:
    """Ceiling fallback polygons clipped to the XZ regions V2 doesn't cover.

    Raw ceiling planes are the preferred source. Flat room-level ceiling
    polygons are only a final fallback; mixed rooms may legitimately contain
    both flat and slanted patches, so flat fallback suppression is patch- and
    height-aware rather than room-wide.

    For each source patch:
      1. Fit a plane through the corners so we can solve Y at any (x, z).
      2. Project corners to the XZ plane (drop Y) → Shapely polygon.
      3. Subtract the relevant V2 final-piece XZ union and dormer cutouts.
         Flat patches subtract only slanted pieces that are physically above
         them; non-flat raw patches keep the older full V2 subtraction.
      4. For each remaining polygon, lift each (x, z) back to 3D via the
         plane equation. Skip fragments that are too small to care about.
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return []

    rooms = building.get("rooms") or []
    if not rooms:
        return []

    subtract_union = _combined_ceiling_subtraction(uuid)
    cutouts = _dormer_cutouts_for_uuid(uuid)
    flat_partitions_by_room = _flat_partition_patches_by_room(uuid)
    raw_flat_source_rooms: set[int] = set()
    partition_flat_source_rooms: set[int] = set(flat_partitions_by_room)
    out: list[dict] = []
    for room_idx, room in enumerate(rooms):
        for plane_index, plane in enumerate(room.get("raw_ceiling_planes") or []):
            corners = plane.get("corners") or []
            if len(corners) < 3:
                continue
            coeffs = _fit_plane_coeffs(corners)
            if coeffs is None:
                continue
            is_flat = _is_flat_patch(corners, coeffs)
            if is_flat and not _raw_plane_allowed_for_flat_room(room, corners):
                continue
            try:
                xz_poly = Polygon([(float(p[0]), float(p[2])) for p in corners])
                if not xz_poly.is_valid:
                    xz_poly = xz_poly.buffer(0)
                if xz_poly.is_empty:
                    continue
            except Exception:
                continue
            if is_flat:
                raw_flat_source_rooms.add(room_idx)
            patch_subtract = (
                _flat_patch_subtraction(uuid, xz_poly, coeffs)
                if is_flat
                else subtract_union
            )
            _append_ceiling_fallback_pieces(
                out,
                xz_poly=xz_poly,
                plane_coeffs=coeffs,
                subtract_union=patch_subtract,
                cutouts=cutouts,
                source="raw_flat_ceiling" if is_flat else "raw_ceiling",
                room_index=room_idx,
                plane_index=plane_index,
            )

    for room_idx, patches in flat_partitions_by_room.items():
        if room_idx in raw_flat_source_rooms:
            continue
        for patch in patches:
            poly = patch.get("poly") or []
            coeffs = _fit_plane_coeffs(poly)
            if coeffs is None:
                continue
            xz_poly = _polygon_xz_with_holes(poly, patch.get("holes") or [])
            if xz_poly is None:
                continue
            _append_ceiling_fallback_pieces(
                out,
                xz_poly=xz_poly,
                plane_coeffs=coeffs,
                subtract_union=_flat_patch_subtraction(uuid, xz_poly, coeffs),
                cutouts=cutouts,
                source="ceiling_partition_flat",
                room_index=room_idx,
                partition_id=patch.get("id"),
            )

    for room_idx, room in enumerate(rooms):
        if room_idx in raw_flat_source_rooms or room_idx in partition_flat_source_rooms:
            continue
        poly = room.get("ceiling_polygon") or []
        if room.get("ceiling_type") != "flat" or len(poly) < 3:
            continue
        coeffs = _fit_plane_coeffs(poly)
        if coeffs is None or not _is_flat_patch(poly, coeffs):
            continue
        xz_poly = _polygon_xz_with_holes(poly)
        if xz_poly is None:
            continue
        _append_ceiling_fallback_pieces(
            out,
            xz_poly=xz_poly,
            plane_coeffs=coeffs,
            subtract_union=_flat_patch_subtraction(uuid, xz_poly, coeffs),
            cutouts=cutouts,
            source="wall_top_flat_ceiling",
            room_index=room_idx,
        )
    return out


def _roof_results_newer_than_v2_sidecar() -> bool:
    try:
        roof_mtime = ROOF_RESULTS_PATH.stat().st_mtime
        sidecar_path = RAW_CEILING_PLANE_SPLITS_PATHS["v2"]
        sidecar_mtime = sidecar_path.stat().st_mtime if sidecar_path.exists() else 0.0
    except OSError:
        return False
    return roof_mtime > sidecar_mtime


# When a single-extension shed slant has no opposing roof plane to bound it
# inward, the V2 supported-piece projection sweeps across the entire arrangement
# cell and crosses into rooms with confirmed flat ceilings. Clip such slants to
# the slabs of the rooms whose oblique walls actually seeded the cluster, plus
# this much eave overhang. 1.5 m matches typical Danish residential eaves.
EAVE_OVERHANG_M = 1.5


def _v3_has_room_flat_ceiling(uuid: str) -> bool:
    """True iff V3 has at least one flat_ceiling with `over == "room"`."""
    _ensure_v3_cache()
    bldg = V3_CACHE.get(uuid) or {}
    return any(fc.get("over") == "room" for fc in bldg.get("flat_ceilings") or [])


def _v3_slant_source_envelope_xz(uuid: str, buffer_m: float = EAVE_OVERHANG_M):
    """Buffered XZ union of slabs for rooms that seeded any V3 slanted_roofs cluster."""
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from shapely.validation import make_valid
    except ImportError:
        return None
    _ensure_v3_cache()
    bldg = V3_CACHE.get(uuid) or {}
    src_rooms: set[str] = set()
    for sr in bldg.get("slanted_roofs") or []:
        for rid in ((sr.get("trace") or {}).get("inputs") or {}).get("room_ids") or []:
            if isinstance(rid, str):
                src_rooms.add(rid)
    if not src_rooms:
        return None
    polys = []
    for slab in bldg.get("slabs") or []:
        if slab.get("room_id") not in src_rooms:
            continue
        pts = slab.get("polygon") or []
        if len(pts) < 3:
            continue
        try:
            poly = Polygon([(float(p[0]), float(p[2])) for p in pts])
            if not poly.is_valid:
                poly = make_valid(poly)
            if not poly.is_empty:
                polys.append(poly)
        except Exception:
            continue
    if not polys:
        return None
    try:
        env = unary_union(polys)
        if buffer_m > 0:
            env = env.buffer(buffer_m)
        return env if not env.is_empty else None
    except Exception:
        return None


def _building_has_gable(uuid: str) -> bool:
    """True iff the canonical complexity-tier classifier flags this as a gable."""
    bldg = BUILDINGS_3D_CACHE.get(uuid)
    if not bldg:
        return False
    try:
        try:
            from .complexity_tiers import classify_building
        except ImportError:
            from complexity_tiers import classify_building
        return bool(
            classify_building(bldg, ROOF_RESULTS_CACHE.get(uuid))["signals"].get(
                "has_gable"
            )
        )
    except Exception:
        return False


def _slant_envelope_clip_active(uuid: str):
    """Return the Shapely envelope to clip slanted pieces against, or None.

    Fires only when the building is not a gable, has at least one room-level
    flat_ceiling, and has a non-empty source-room slab envelope.
    """
    if _building_has_gable(uuid):
        return None
    if not _v3_has_room_flat_ceiling(uuid):
        return None
    return _v3_slant_source_envelope_xz(uuid)


def _envelope_clip_pieces(
    pieces: list[dict], envelope, *, source_label: str
) -> list[dict]:
    """
    Clip raw slanted pieces (corners-only) to a Shapely envelope, keeping the
    original metadata.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
    except ImportError:
        return pieces
    out: list[dict] = []
    for piece in pieces:
        corners = piece.get("corners") or []
        if len(corners) < 3:
            continue
        coeffs = _fit_plane_coeffs(corners)
        if coeffs is None:
            continue
        try:
            xz_poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
            if not xz_poly.is_valid:
                xz_poly = make_valid(xz_poly)
            if xz_poly.is_empty:
                continue
            xz_poly = xz_poly.intersection(envelope)
        except Exception:
            continue
        if xz_poly.is_empty or xz_poly.area < 0.05:
            continue
        for piece_out in _shapely_difference_to_3d_pieces(
            xz_poly,
            None,
            coeffs,
            min_area=0.05,
        ):
            out.append(
                {
                    **piece,
                    "corners": piece_out["poly"],
                    "holes": piece_out["holes"],
                    "source": source_label,
                    "clip_source": "slanted_wall_envelope",
                }
            )
    return out


def _slanted_pieces_for_uuid(uuid: str) -> list[dict]:
    """V2 final-layer slanted pieces for the tier/full-model preview.

    `viewer.html` renders raw eave-supported v2 final-layer splits directly
    from `/raw-ceiling-plane-splits?version=v2`. The tier preview should show
    the same slanted roof source, with only dormer cutouts subtracted for the
    full-model look. A freshly regenerated backend roof arrangement takes
    precedence over the sidecar during local single-building iteration.

    For non-gable buildings whose slant plane has no opposing plane, the V2
    supported piece can sweep over rooms with confirmed flat ceilings; in that
    case we additionally clip pieces to the slabs of the source rooms (plus an
    eave overhang). See `_slant_envelope_clip_active`.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return []

    envelope = _slant_envelope_clip_active(uuid)
    backend_pieces = _backend_arrangement_pieces_for_uuid(uuid)
    if _use_backend_arrangement_for_slanted(uuid):
        if envelope is None:
            return backend_pieces
        return _envelope_clip_pieces(
            backend_pieces, envelope, source_label="roof_arrangement"
        )

    source_pieces = _load_v2_full_model_pieces(uuid)
    source = "v2_sidecar"
    if not source_pieces:
        if envelope is None:
            return backend_pieces
        return _envelope_clip_pieces(
            backend_pieces, envelope, source_label="roof_arrangement"
        )

    cutouts = _dormer_cutouts_for_uuid(uuid)
    cutouts_xz = unary_union([c[0] for c in cutouts]) if cutouts else None

    out: list[dict] = []
    for piece in source_pieces:
        corners = piece.get("corners") or piece.get("poly") or []
        if len(corners) < 3:
            continue
        coeffs = _fit_plane_coeffs(corners)
        if coeffs is None:
            continue
        try:
            xz_poly = Polygon([(float(c[0]), float(c[2])) for c in corners])
            if not xz_poly.is_valid:
                xz_poly = xz_poly.buffer(0)
            if xz_poly.is_empty:
                continue
        except Exception:
            continue
        if envelope is not None:
            try:
                xz_poly = xz_poly.intersection(envelope)
            except Exception:
                continue
            if xz_poly.is_empty or xz_poly.area < 0.05:
                continue
        for piece_out in _shapely_difference_to_3d_pieces(
            xz_poly,
            cutouts_xz,
            coeffs,
            min_area=0.05,
            interior_planes=cutouts,
        ):
            entry = {
                "corners": piece_out["poly"],
                "holes": piece_out["holes"],
                "source": source,
                "piece_id": piece.get("piece_id"),
                "target_element_id": piece.get("target_element_id"),
                "piece_role": piece.get("piece_role"),
                "target_kind": piece.get("target_kind"),
                "arrangement_cell_id": piece.get("arrangement_cell_id"),
                "roof_hypothesis_id": piece.get("roof_hypothesis_id"),
                "intersection_kind": piece.get("intersection_kind"),
            }
            if envelope is not None:
                entry["clip_source"] = "slanted_wall_envelope"
            out.append(entry)
    return out


def _leaves_of(proposal_id: str, children_by_parent: dict[str, list[str]]) -> list[str]:
    kids = children_by_parent.get(proposal_id)
    if not kids:
        return [proposal_id]
    out: list[str] = []
    for k in kids:
        out.extend(_leaves_of(k, children_by_parent))
    return out


def _ensure_v3_cache() -> None:
    """Refresh V3_CACHE from disk when the results file has changed."""
    global V3_CACHE, V3_CACHE_MTIME
    if not V3_RESULTS_PATH.exists():
        return
    try:
        mtime = V3_RESULTS_PATH.stat().st_mtime
        if mtime != V3_CACHE_MTIME or not V3_CACHE:
            with open(V3_RESULTS_PATH) as handle:
                data = json.load(handle)
            V3_CACHE = {
                entry.get("building_uuid"): entry
                for entry in data
                if entry.get("building_uuid")
            }
            V3_CACHE_MTIME = mtime
    except Exception:
        return


def _resolve_artifact_path(
    *,
    env_var: str,
    default_candidates: tuple[Path, ...],
) -> Path:
    """Return the preferred on-disk artifact path for a viewer sidecar.

    Resolution order:
    1. explicit environment override
    2. first existing path in ``default_candidates``
    3. first candidate path (even if missing) for user-facing error messages
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    for candidate in default_candidates:
        if candidate.exists():
            return candidate
    return default_candidates[0]


def _ensure_v3_scores() -> None:
    """Refresh V3_SCORES from the scored mirror JSON when it has changed."""
    global V3_SCORES, V3_SCORES_MTIME
    if not V3_SCORED_PATH.exists():
        V3_SCORES = {}
        V3_SCORES_MTIME = 0.0
        return
    try:
        mtime = V3_SCORED_PATH.stat().st_mtime
        if mtime == V3_SCORES_MTIME and V3_SCORES:
            return
        scores: dict[str, dict] = {}
        import ijson  # local import; only needed when a scored file is present

        with V3_SCORED_PATH.open("rb") as handle:
            for b in ijson.items(handle, "item", use_float=True):
                for seg in b.get("merged_roof_segments") or []:
                    sid = seg.get("id")
                    if not isinstance(sid, str):
                        continue
                    scores[sid] = {
                        "score": seg.get("score"),
                        "autonomy_label": seg.get("autonomy_label"),
                        "rule_fires": bool(seg.get("rule_fires")),
                    }
        V3_SCORES = scores
        V3_SCORES_MTIME = mtime
    except Exception:
        # If scoring file is corrupt or mid-write, fall back to no scores.
        V3_SCORES = {}
        V3_SCORES_MTIME = 0.0


def _plane_y_at(x: float, z: float, plane: tuple[float, float, float, float]) -> float:
    a, b, c, d = plane
    if abs(b) < 1e-6:
        return 0.0
    return -(a * x + c * z + d) / b


def _split_proposal_polygon(
    corners_xyz: list[list[float]],
    plane: tuple[float, float, float, float],
    p1_xz: tuple[float, float],
    p2_xz: tuple[float, float],
) -> list[tuple[list[list[float]], tuple[float, float]]]:
    """Split the XZ footprint of a proposal polygon by an infinite line through
    p1/p2, then lift both halves back onto the plane.

    Returns a list of (corners_xyz, centroid_xz) tuples, ordered by signed
    distance from the split line (left half first, right half second).
    Raises ValueError if the split doesn't produce exactly two polygons.
    """
    import math

    if len(corners_xyz) < 3:
        raise ValueError("parent polygon has fewer than 3 corners")
    xz_ring = [(float(p[0]), float(p[2])) for p in corners_xyz]
    poly = Polygon(xz_ring)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty or poly.area <= 0:
        raise ValueError("parent polygon is empty or zero-area")

    dx, dz = p2_xz[0] - p1_xz[0], p2_xz[1] - p1_xz[1]
    length = math.hypot(dx, dz)
    if length < 1e-6:
        raise ValueError("split points coincide")
    ux, uz = dx / length, dz / length
    # Extend well beyond polygon bbox so shapely.split bisects reliably.
    minx, minz, maxx, maxz = poly.bounds
    diag = math.hypot(maxx - minx, maxz - minz)
    L = max(diag * 10.0, 100.0)
    ext_a = (p1_xz[0] - ux * L, p1_xz[1] - uz * L)
    ext_b = (p2_xz[0] + ux * L, p2_xz[1] + uz * L)
    line = LineString([ext_a, ext_b])

    try:
        result = shapely_split(poly, line)
    except Exception as exc:
        raise ValueError(f"shapely.split failed: {exc}") from exc

    pieces = [g for g in getattr(result, "geoms", [result]) if g.area > 1e-9]
    if len(pieces) < 2:
        raise ValueError("split line does not cross the polygon")

    # Left normal: 90° CCW from line direction.
    nx, nz = -uz, ux

    def side(poly2: Polygon) -> float:
        cx, cz = poly2.centroid.x, poly2.centroid.y
        return (cx - p1_xz[0]) * nx + (cz - p1_xz[1]) * nz

    if len(pieces) > 2:
        # Merge all pieces with side>0 into left, <=0 into right.
        left = [pp for pp in pieces if side(pp) > 0]
        right = [pp for pp in pieces if side(pp) <= 0]
        if not left or not right:
            raise ValueError("split produced degenerate halves")
        left_poly = unary_union(left)
        right_poly = unary_union(right)
        if left_poly.geom_type == "MultiPolygon":
            left_poly = max(left_poly.geoms, key=lambda g: g.area)
        if right_poly.geom_type == "MultiPolygon":
            right_poly = max(right_poly.geoms, key=lambda g: g.area)
        pieces_ordered = [left_poly, right_poly]
    else:
        pieces_ordered = sorted(pieces, key=side, reverse=True)

    out: list[tuple[list[list[float]], tuple[float, float]]] = []
    for piece in pieces_ordered:
        coords = list(piece.exterior.coords)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        corners = [
            [float(x), _plane_y_at(float(x), float(z), plane), float(z)]
            for x, z in coords
        ]
        out.append((corners, (float(piece.centroid.x), float(piece.centroid.y))))
    return out


def _room_key(room_index: int) -> str:
    return f"room:{int(room_index)}"


def _round6(value: float) -> float:
    return round(float(value), 6)


def _parse_room_index(room_id: str) -> int | None:
    if not isinstance(room_id, str) or not room_id.startswith("room:"):
        return None
    try:
        return int(room_id.split(":", 1)[1])
    except Exception:
        return None


def _parse_topology_room_index(source_id: str) -> int | None:
    if not isinstance(source_id, str) or not source_id:
        return None
    direct = _parse_room_index(source_id)
    if direct is not None:
        return direct
    marker = "merged_room_"
    if marker not in source_id:
        return None
    try:
        return int(source_id.rsplit(marker, 1)[1])
    except Exception:
        return None


def _room_indices_for_ids(
    room_ids: set[str] | list[str], room_indices_by_room_id: dict[str, int]
) -> list[int]:
    indices: set[int] = set()
    for room_id in room_ids or []:
        if room_id in room_indices_by_room_id:
            indices.add(int(room_indices_by_room_id[room_id]))
            continue
        parsed = _parse_room_index(str(room_id))
        if parsed is not None:
            indices.add(parsed)
    return sorted(indices)


def _poly_xz_from_3d(corners: list[Any]) -> Polygon | None:
    coords: list[tuple[float, float]] = []
    for corner in corners or []:
        if not isinstance(corner, (list, tuple)) or len(corner) < 3:
            continue
        coords.append((_round6(corner[0]), _round6(corner[2])))
    if len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda geom: geom.area, default=None)
        except Exception:
            return None
    if poly is None or poly.is_empty or not isinstance(poly, Polygon):
        return None
    if poly.area <= 1e-6:
        return None
    return poly


def _polygon_xz_to_3d(polygon_xz: list[list[float]], y: float) -> list[list[float]]:
    return [[_round6(x), _round6(y), _round6(z)] for x, z in polygon_xz]


def _serialize_poly_xz(poly: Polygon | None) -> list[list[float]]:
    if poly is None or poly.is_empty:
        return []
    coords = list(poly.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    return [[_round6(x), _round6(z)] for x, z, *_ in coords]


def _bbox_xz(poly: Polygon | None) -> list[float] | None:
    if poly is None or poly.is_empty:
        return None
    min_x, min_z, max_x, max_z = poly.bounds
    return [_round6(min_x), _round6(min_z), _round6(max_x), _round6(max_z)]


def _centroid_xz(poly: Polygon | None) -> list[float] | None:
    if poly is None or poly.is_empty:
        return None
    c = poly.centroid
    return [_round6(c.x), _round6(c.y)]


def _largest_polygon(geom: Any) -> Polygon | None:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, Polygon):
        return geom
    geoms = [
        g
        for g in getattr(geom, "geoms", [])
        if isinstance(g, Polygon) and not g.is_empty
    ]
    if not geoms:
        return None
    return max(geoms, key=lambda poly: poly.area)


def _decompose_polygons(geom: Any) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [
        poly
        for poly in getattr(geom, "geoms", [])
        if isinstance(poly, Polygon) and not poly.is_empty
    ]


def _surface_plane(surface: dict[str, Any]) -> dict[str, Any] | None:
    cluster = surface.get("cluster") or {}
    avg_azimuth = cluster.get("avgAzimuth")
    avg_incl = cluster.get("avgIncl")
    ref = cluster.get("refPt") or surface.get("center") or {}
    if avg_azimuth is not None and avg_incl is not None and ref:
        return {
            "n": plane_normal(float(avg_azimuth), float(avg_incl)),
            "ref": {
                "x": float(ref["x"]),
                "y": float(ref["y"]),
                "z": float(ref["z"]),
            },
        }
    corners = [
        corner
        for corner in (surface.get("corners") or [])
        if isinstance(corner, (list, tuple)) and len(corner) >= 3
    ]
    if len(corners) < 3:
        return None
    a, b, c = corners[0], corners[1], corners[2]
    ab = (
        float(b[0]) - float(a[0]),
        float(b[1]) - float(a[1]),
        float(b[2]) - float(a[2]),
    )
    ac = (
        float(c[0]) - float(a[0]),
        float(c[1]) - float(a[1]),
        float(c[2]) - float(a[2]),
    )
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
    if norm <= 1e-9 or abs(ny) <= 1e-9:
        return None
    nx /= norm
    ny /= norm
    nz /= norm
    if ny < 0.0:
        nx *= -1.0
        ny *= -1.0
        nz *= -1.0
    return {
        "n": {"x": nx, "y": ny, "z": nz},
        "ref": {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])},
    }


def _surface_is_oblique(surface: dict[str, Any]) -> bool:
    kind = str(surface.get("kind") or surface.get("surface_kind") or "")
    if kind == "oblique":
        return True
    if kind == "flat":
        return False
    hypothesis_id = str(surface.get("roof_hypothesis_id") or "")
    if hypothesis_id.startswith("roof-hypothesis:oblique:"):
        return True
    if hypothesis_id.startswith("roof-hypothesis:flat:"):
        return False
    cluster = surface.get("cluster") or {}
    try:
        return abs(float(cluster.get("avgIncl"))) > 5.0
    except Exception:
        return False


def _surface_y_at(surface: dict[str, Any], x: float, z: float) -> float:
    if _surface_is_oblique(surface):
        plane = _surface_plane(surface)
        if plane is not None:
            return _round6(plane_y_at(plane, float(x), float(z)))
    ys = [
        float(corner[1])
        for corner in (surface.get("corners") or [])
        if isinstance(corner, (list, tuple)) and len(corner) >= 3
    ]
    return _round6(sum(ys) / len(ys)) if ys else 0.0


def _lift_poly_on_surface(poly: Polygon, surface: dict[str, Any]) -> list[list[float]]:
    coords = list(poly.exterior.coords)
    if coords and coords[-1] == coords[0]:
        coords = coords[:-1]
    lifted: list[list[float]] = []
    for x, z, *_ in coords:
        lifted.append(
            [
                _round6(x),
                _surface_y_at(surface, float(x), float(z)),
                _round6(z),
            ]
        )
    return lifted


def _slice_topology_cells_for_graph_rooms(
    topology_cell_complex: dict[str, Any],
    graph_room_ids: set[str],
) -> list[dict[str, Any]]:
    cells = []
    for cell in topology_cell_complex.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        if str(cell.get("kind")) != "room":
            continue
        if str(cell.get("source_id") or "") not in graph_room_ids:
            continue
        cells.append(cell)
    return cells


def _filter_part_dormers(
    dormers: list[dict[str, Any]], room_indices: set[int]
) -> list[dict[str, Any]]:
    if not room_indices:
        return []
    kept: list[dict[str, Any]] = []
    for dormer in dormers or []:
        room_index = dormer.get("room_index")
        if isinstance(room_index, int) and room_index in room_indices:
            kept.append(dormer)
    return kept


def _room_partition_surfaces_by_room(
    roof: dict[str, Any] | None,
) -> dict[int, list[dict[str, Any]]]:
    surfaces_by_room: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for room_partition in ((roof or {}).get("ceiling_partitions") or {}).get(
        "room_partitions"
    ) or []:
        room_index = room_partition.get("room_index")
        if not isinstance(room_index, int):
            continue
        for partition in room_partition.get("partitions") or []:
            corners = partition.get("poly") or []
            if not isinstance(corners, list) or len(corners) < 3:
                continue
            surfaces_by_room[room_index].append(
                {
                    "id": partition.get("id"),
                    "kind": str(partition.get("kind") or "flat"),
                    "roof_hypothesis_id": partition.get("roof_hypothesis_id"),
                    "corners": corners,
                }
            )
    return surfaces_by_room


def _wall_vertical_pairs(
    corners: list[list[float]],
) -> tuple[list[int], dict[int, int]] | None:
    points = [
        [_round6(corner[0]), _round6(corner[1]), _round6(corner[2])]
        for corner in corners or []
        if isinstance(corner, (list, tuple)) and len(corner) >= 3
    ]
    if len(points) < 4:
        return None
    indexed = list(enumerate(points))
    indexed.sort(key=lambda item: (item[1][1], item[1][0], item[1][2]))
    bottom = [index for index, _ in indexed[:2]]
    top = [index for index, _ in indexed[-2:]]
    if len(set(bottom)) != 2 or len(set(top)) != 2:
        return None
    mapping: dict[int, int] = {}
    unused_top = set(top)
    for bottom_index in bottom:
        bx, _, bz = points[bottom_index]
        best_top = min(
            unused_top,
            key=lambda top_index: (
                (points[top_index][0] - bx) ** 2 + (points[top_index][2] - bz) ** 2,
                abs(points[top_index][1] - points[bottom_index][1]),
            ),
        )
        mapping[bottom_index] = best_top
        unused_top.remove(best_top)
    return bottom, mapping


def _decompose_lines(geom: Any) -> list[LineString]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, LineString):
        return [geom]
    return [
        line
        for line in getattr(geom, "geoms", [])
        if isinstance(line, LineString) and not line.is_empty
    ]


def _serialize_corners(corners: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for corner in corners or []:
        if len(corner) < 3:
            continue
        point = [_round6(corner[0]), _round6(corner[1]), _round6(corner[2])]
        if out and point == out[-1]:
            continue
        out.append(point)
    if len(out) >= 2 and out[0] == out[-1]:
        out.pop()
    return out


def _building_story_unions(building: dict[str, Any] | None) -> dict[int, Polygon]:
    if building is None:
        return {}
    polys_by_story: dict[int, list[Polygon]] = defaultdict(list)
    for room in building.get("rooms") or []:
        story = int(room.get("story", 0) or 0)
        poly = _poly_xz_from_3d(room.get("floor_polygon") or [])
        if poly is not None:
            polys_by_story[story].append(poly)
    unions: dict[int, Polygon] = {}
    for story, polys in polys_by_story.items():
        if not polys:
            continue
        try:
            union_poly = _largest_polygon(unary_union(polys))
        except Exception:
            union_poly = None
        if union_poly is not None:
            unions[story] = union_poly
    return unions


def _wall_ordered_profile(
    corners: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], LineString] | None:
    points = _serialize_corners(corners)
    pairing = _wall_vertical_pairs(points)
    if pairing is None:
        return None
    bottom_indices, top_by_bottom = pairing
    provisional = LineString(
        [(float(points[index][0]), float(points[index][2])) for index in bottom_indices]
    )
    if provisional.length <= 1e-6:
        return None
    ordered_bottom = sorted(
        bottom_indices,
        key=lambda index: float(
            provisional.project(Point(float(points[index][0]), float(points[index][2])))
        ),
    )
    b0, b1 = ordered_bottom
    t0 = top_by_bottom[b0]
    t1 = top_by_bottom[b1]
    bottom = [points[b0], points[b1]]
    top = [points[t0], points[t1]]
    wall_line = LineString([(bottom[0][0], bottom[0][2]), (bottom[1][0], bottom[1][2])])
    if wall_line.length <= 1e-6:
        return None
    return bottom, top, wall_line


def _interp_corner(left: list[float], right: list[float], t: float) -> list[float]:
    return [
        _round6(float(left[0]) + (float(right[0]) - float(left[0])) * t),
        _round6(float(left[1]) + (float(right[1]) - float(left[1])) * t),
        _round6(float(left[2]) + (float(right[2]) - float(left[2])) * t),
    ]


def _wall_surface_intervals(
    wall_line: LineString,
    room_surfaces: list[dict[str, Any]],
) -> list[tuple[float, float, dict[str, Any]]]:
    intervals: list[tuple[float, float, dict[str, Any]]] = []
    for surface in room_surfaces:
        poly = _poly_xz_from_3d(surface.get("corners") or [])
        if poly is None:
            continue
        try:
            intersection = poly.buffer(1e-6, cap_style=2, join_style=2).intersection(
                wall_line
            )
        except Exception:
            continue
        for line in _decompose_lines(intersection):
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            start = wall_line.project(Point(coords[0]))
            end = wall_line.project(Point(coords[-1]))
            if end < start:
                start, end = end, start
            if end - start <= 1e-6:
                continue
            intervals.append(
                (
                    start / wall_line.length,
                    end / wall_line.length,
                    surface,
                )
            )
    return intervals


def _is_exterior_wall_line(wall_line: LineString, story_union: Polygon | None) -> bool:
    if story_union is None:
        return False
    try:
        overlap = wall_line.intersection(story_union.boundary)
    except Exception:
        return False
    return not overlap.is_empty and float(getattr(overlap, "length", 0.0)) > 1e-6


def _polygon_centroid3(corners: list[list[float]]) -> list[float]:
    if not corners:
        return [0.0, 0.0, 0.0]
    sx = sum(float(corner[0]) for corner in corners)
    sy = sum(float(corner[1]) for corner in corners)
    sz = sum(float(corner[2]) for corner in corners)
    inv = 1.0 / max(1, len(corners))
    return [_round6(sx * inv), _round6(sy * inv), _round6(sz * inv)]


def _projection_axes_for_corners(corners: list[list[float]]) -> tuple[int, int]:
    nx = ny = nz = 0.0
    count = len(corners)
    for index, corner in enumerate(corners):
        next_corner = corners[(index + 1) % count]
        nx += (float(corner[1]) - float(next_corner[1])) * (
            float(corner[2]) + float(next_corner[2])
        )
        ny += (float(corner[2]) - float(next_corner[2])) * (
            float(corner[0]) + float(next_corner[0])
        )
        nz += (float(corner[0]) - float(next_corner[0])) * (
            float(corner[1]) + float(next_corner[1])
        )
    anx, any_, anz = abs(nx), abs(ny), abs(nz)
    if any_ >= anx and any_ >= anz:
        return (0, 2)
    if anx >= any_ and anx >= anz:
        return (1, 2)
    return (0, 1)


def _point_in_polygon_2(
    point: tuple[float, float], poly: list[tuple[float, float]]
) -> bool:
    inside = False
    for index in range(len(poly)):
        xi, yi = poly[index]
        xj, yj = poly[index - 1]
        crosses = ((yi > point[1]) != (yj > point[1])) and (
            point[0] < (xj - xi) * (point[1] - yi) / ((yj - yi) or 1e-9) + xi
        )
        if crosses:
            inside = not inside
    return inside


def _distance_point_to_segment_2(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    vx = float(b[0] - a[0])
    vy = float(b[1] - a[1])
    wx = float(point[0] - a[0])
    wy = float(point[1] - a[1])
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return ((wx * wx) + (wy * wy)) ** 0.5
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    px = float(a[0] + t * vx)
    py = float(a[1] + t * vy)
    dx = float(point[0] - px)
    dy = float(point[1] - py)
    return (dx * dx + dy * dy) ** 0.5


def _wall_plane(
    corners: list[list[float]],
) -> tuple[tuple[float, float, float], float] | None:
    if len(corners) < 3:
        return None
    a, b, c = corners[0], corners[1], corners[2]
    ab = (float(b[0] - a[0]), float(b[1] - a[1]), float(b[2] - a[2]))
    ac = (float(c[0] - a[0]), float(c[1] - a[1]), float(c[2] - a[2]))
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
    if norm <= 1e-12:
        return None
    nx /= norm
    ny /= norm
    nz /= norm
    d = -((nx * float(a[0])) + (ny * float(a[1])) + (nz * float(a[2])))
    return (nx, ny, nz), d


def _distance_to_plane(
    plane: tuple[tuple[float, float, float], float], point: list[float]
) -> float:
    (nx, ny, nz), d = plane
    return abs(nx * float(point[0]) + ny * float(point[1]) + nz * float(point[2]) + d)


def _collect_wall_cutout_holes(
    wall_corners: list[list[float]],
    openings: list[dict[str, Any]],
) -> list[list[list[float]]]:
    plane = _wall_plane(wall_corners)
    if plane is None:
        return []
    axis0, axis1 = _projection_axes_for_corners(wall_corners)
    outer2 = [(float(corner[axis0]), float(corner[axis1])) for corner in wall_corners]
    holes: list[list[list[float]]] = []
    plane_eps = 0.05
    edge_margin = 0.01
    for opening in openings or []:
        corners = opening.get("corners") or []
        if not isinstance(corners, list) or len(corners) != 4:
            continue
        normalized = _serialize_corners(corners)
        if any(_distance_to_plane(plane, corner) > plane_eps for corner in normalized):
            continue
        centroid = _polygon_centroid3(normalized)
        centroid2 = (float(centroid[axis0]), float(centroid[axis1]))
        if not _point_in_polygon_2(centroid2, outer2):
            continue
        hole2 = [(float(corner[axis0]), float(corner[axis1])) for corner in normalized]
        valid = True
        for point in hole2:
            if not _point_in_polygon_2(point, outer2):
                valid = False
                break
            min_dist = min(
                _distance_point_to_segment_2(point, outer2[index - 1], outer2[index])
                for index in range(len(outer2))
            )
            if min_dist < edge_margin:
                valid = False
                break
        if valid:
            holes.append(normalized)
    return holes


def _segment_wall_corners_by_room_surfaces(
    corners: list[list[float]],
    room_surfaces: list[dict[str, Any]],
) -> list[tuple[list[list[float]], dict[str, Any] | None]]:
    profile = _wall_ordered_profile(corners)
    if profile is None:
        return [(_serialize_corners(corners), None)]
    bottom, top, wall_line = profile
    intervals = _wall_surface_intervals(wall_line, room_surfaces)
    if not intervals:
        return [(_serialize_corners(corners), None)]
    split_ts = {0.0, 1.0}
    for start_t, end_t, _surface in intervals:
        split_ts.add(max(0.0, min(1.0, start_t)))
        split_ts.add(max(0.0, min(1.0, end_t)))
    ordered_ts = sorted(split_ts)
    segments: list[tuple[list[list[float]], dict[str, Any] | None]] = []
    for left_t, right_t in itertools.pairwise(ordered_ts):
        if right_t - left_t <= 1e-6:
            continue
        mid_t = (left_t + right_t) * 0.5
        surface = None
        for start_t, end_t, candidate in intervals:
            if start_t - 1e-6 <= mid_t <= end_t + 1e-6:
                surface = candidate
                break
        bottom_left = _interp_corner(bottom[0], bottom[1], left_t)
        bottom_right = _interp_corner(bottom[0], bottom[1], right_t)
        top_left = _interp_corner(top[0], top[1], left_t)
        top_right = _interp_corner(top[0], top[1], right_t)
        if surface is not None:
            top_left[1] = _round6(
                max(
                    bottom_left[1],
                    min(top_left[1], _surface_y_at(surface, top_left[0], top_left[2])),
                )
            )
            top_right[1] = _round6(
                max(
                    bottom_right[1],
                    min(
                        top_right[1], _surface_y_at(surface, top_right[0], top_right[2])
                    ),
                )
            )
        segment = [bottom_left, bottom_right, top_right, top_left]
        if (
            LineString(
                [
                    (bottom_left[0], bottom_left[2]),
                    (bottom_right[0], bottom_right[2]),
                ]
            ).length
            <= 1e-6
        ):
            continue
        segments.append((segment, surface))
    return segments or [(_serialize_corners(corners), None)]


def _renderable_surfaces_from_room_wall(
    wall: dict[str, Any],
    *,
    part_id: str,
    room_index: int,
    story: int,
    room_surfaces: list[dict[str, Any]],
    openings: list[dict[str, Any]],
    wall_index: int,
    story_union: Polygon | None,
    extension_index: int | None = None,
) -> list[dict[str, Any]]:
    raw_corners = wall.get("corners") or []
    if not isinstance(raw_corners, list) or len(raw_corners) < 3:
        return []
    profile = _wall_ordered_profile(raw_corners)
    if profile is None:
        return []
    _bottom, _top, wall_line = profile
    category = (
        "base_exterior_wall"
        if _is_exterior_wall_line(wall_line, story_union)
        else "base_interior_wall"
    )
    identifier = str(wall.get("id") or f"room-wall:{room_index}:{wall_index}")
    suffix = f":ext:{extension_index}" if extension_index is not None else ""
    renderable: list[dict[str, Any]] = []
    for segment_index, (segment_corners, surface) in enumerate(
        _segment_wall_corners_by_room_surfaces(raw_corners, room_surfaces)
    ):
        renderable.append(
            {
                "id": f"renderable:{category}:{identifier}{suffix}:seg:{segment_index}",
                "category": category,
                "source_kind": "raw_room_wall",
                "source_id": identifier,
                "corners": segment_corners,
                "holes": _collect_wall_cutout_holes(segment_corners, openings),
                "part_id": part_id,
                "room_id": _room_key(room_index),
                "room_index": room_index,
                "story": story,
                "roof_hypothesis_id": (surface or {}).get("roof_hypothesis_id"),
            }
        )
    return renderable


def _renderable_surface_from_room_floor(
    room: dict[str, Any], *, part_id: str, room_index: int, story: int
) -> dict[str, Any] | None:
    corners = room.get("floor_polygon") or []
    if not isinstance(corners, list) or len(corners) < 3:
        return None
    return {
        "id": f"renderable:base_room_floor:{room_index}",
        "category": "base_room_floor",
        "source_kind": "raw_room_floor",
        "source_id": f"room-floor:{room_index}",
        "corners": [
            [_round6(c[0]), _round6(c[1]), _round6(c[2])]
            for c in corners
            if len(c) >= 3
        ],
        "part_id": part_id,
        "room_id": _room_key(room_index),
        "room_index": room_index,
        "story": story,
    }


def _renderable_fenestration_surfaces(
    room: dict[str, Any],
    *,
    part_id: str,
    room_index: int,
    story: int,
) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    fenestration_specs = [
        ("window", "base_window", room.get("windows") or []),
        ("door", "base_door", room.get("doors") or []),
        ("opening", "base_opening", room.get("openings") or []),
    ]
    for kind, category, items in fenestration_specs:
        for index, item in enumerate(items):
            corners = _serialize_corners(item.get("corners") or [])
            if len(corners) < 3:
                continue
            surfaces.append(
                {
                    "id": f"renderable:{category}:{room_index}:{index}",
                    "category": category,
                    "source_kind": kind,
                    "source_id": item.get("id") or f"{kind}:{room_index}:{index}",
                    "corners": corners,
                    "part_id": part_id,
                    "room_id": _room_key(room_index),
                    "room_index": room_index,
                    "story": story,
                }
            )
    return surfaces


def _fenestration_by_room(
    building: dict[str, Any] | None,
) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    if building is None:
        return out
    for room_index, room in enumerate(building.get("rooms") or []):
        out[room_index] = [
            *(
                {"corners": item.get("corners") or []}
                for item in (room.get("windows") or [])
            ),
            *(
                {"corners": item.get("corners") or []}
                for item in (room.get("doors") or [])
            ),
            *(
                {"corners": item.get("corners") or []}
                for item in (room.get("openings") or [])
            ),
        ]
    return out


def _renderable_surface_from_occupied_face(
    face: dict[str, Any],
    cell: dict[str, Any],
    *,
    part_id: str,
    openings: list[dict[str, Any]],
    atoms_covering_ceiling: set[str] | None = None,
) -> dict[str, Any] | None:
    corners = _face_corners(face)
    if len(corners) < 3:
        return None
    metadata = face.get("metadata") or {}
    boundary_class = str(metadata.get("boundary_class") or "")
    category: str | None = None
    holes: list[list[list[float]]] = []
    exact_source_kind = str(cell.get("exact_source_kind") or "")
    if boundary_class == "floor":
        category = "base_room_floor"
    elif boundary_class == "ceiling":
        if exact_source_kind == "synthetic_top_boundary_atom":
            return None
        # Avoid double-rendering: when the cell's top boundary atom already
        # emits a committed ceiling surface (sloped/flat/transition cap), the
        # flat horizontal base_room_ceiling duplicates — and often floats
        # above — the atom's actual slope or cap. Suppress it here so the
        # atom's poly is the sole ceiling for that cell.
        top_atom_id = cell.get("top_boundary_atom_id")
        if (
            atoms_covering_ceiling is not None
            and isinstance(top_atom_id, str)
            and top_atom_id in atoms_covering_ceiling
        ):
            return None
        category = "base_room_ceiling"
    elif boundary_class == "exterior_wall":
        category = "base_exterior_wall"
        holes = _collect_wall_cutout_holes(corners, openings)
    elif boundary_class == "interior_wall":
        if (
            str(face.get("source_kind") or "") == "splitter"
            or str(face.get("role") or "") == "splitter"
        ):
            return None
        category = "base_interior_wall"
        holes = _collect_wall_cutout_holes(corners, openings)
    if category is None:
        return None
    return {
        "id": (
            f"renderable:{category}:{cell.get('id')}:{face.get('id') or boundary_class}"
        ),
        "category": category,
        "source_kind": "occupied_room_cell_face",
        "source_id": face.get("id") or boundary_class,
        "cell_id": cell.get("id"),
        "corners": corners,
        "holes": holes,
        "part_id": part_id,
        "room_id": cell.get("room_id"),
        "room_index": cell.get("room_index"),
        "story": cell.get("story"),
        "roof_hypothesis_id": cell.get("roof_hypothesis_id"),
        "top_boundary_atom_id": cell.get("top_boundary_atom_id"),
        "boundary_class": boundary_class,
        "exact_source_kind": exact_source_kind or None,
    }


def _unresolved_region_from_synthetic_occupied_ceiling(
    face: dict[str, Any],
    cell: dict[str, Any],
    *,
    part_id: str,
) -> dict[str, Any] | None:
    metadata = face.get("metadata") or {}
    if str(metadata.get("boundary_class") or "") != "ceiling":
        return None
    if str(cell.get("exact_source_kind") or "") != "synthetic_top_boundary_atom":
        return None
    corners = _face_corners(face)
    if len(corners) < 3:
        return None
    polygon_xz = [
        [_round6(corner[0]), _round6(corner[2])]
        for corner in corners
        if len(corner) >= 3
    ]
    if len(polygon_xz) < 3:
        return None
    cell_id = str(cell.get("id") or "")
    face_id = str(face.get("id") or "ceiling")
    return {
        "id": f"unresolved-occupied-fallback-top:{cell_id}:{face_id}",
        "room_id": cell.get("room_id"),
        "room_index": cell.get("room_index"),
        "story": cell.get("story"),
        "effective_part_ids": [part_id],
        "polygon": corners,
        "polygon_xz": polygon_xz,
        "roof_evidence_score": None,
        "fallback_source_kind": "occupied_room_synthetic_ceiling",
        "top_boundary_atom_id": cell.get("top_boundary_atom_id"),
        "exact_source_kind": cell.get("exact_source_kind"),
    }


def _renderable_surfaces_from_occupied_room_cells(
    *,
    occupied_room_cell_complex: dict[str, Any] | None,
    building: dict[str, Any] | None,
    room_indices: set[int],
    part_id: str,
    atoms_covering_ceiling: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if occupied_room_cell_complex is None:
        return [], []
    fenestration_by_room = _fenestration_by_room(building)
    rooms = building.get("rooms") or [] if building is not None else []
    renderable: list[dict[str, Any]] = []
    unresolved_regions: list[dict[str, Any]] = []
    include_all = part_id == FULL_BUILDING_PART_ID
    emitted_fenestration_rooms: set[int] = set()
    for cell in occupied_room_cell_complex.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        room_index = cell.get("room_index")
        if not include_all and (
            not isinstance(room_index, int) or room_index not in room_indices
        ):
            continue
        for face in cell.get("faces") or []:
            if not isinstance(face, dict):
                continue
            unresolved = _unresolved_region_from_synthetic_occupied_ceiling(
                face, cell, part_id=part_id
            )
            if unresolved is not None:
                unresolved_regions.append(unresolved)
                continue
            surface = _renderable_surface_from_occupied_face(
                face,
                cell,
                part_id=part_id,
                openings=fenestration_by_room.get(room_index, []),
                atoms_covering_ceiling=atoms_covering_ceiling,
            )
            if surface is not None:
                renderable.append(surface)
        if not isinstance(room_index, int):
            continue
        if room_index in emitted_fenestration_rooms:
            continue
        if room_index < 0 or room_index >= len(rooms):
            continue
        room = rooms[room_index] or {}
        story = int(room.get("story", 0) or 0)
        renderable.extend(
            _renderable_fenestration_surfaces(
                room,
                part_id=part_id,
                room_index=room_index,
                story=story,
            )
        )
        emitted_fenestration_rooms.add(room_index)
    return renderable, unresolved_regions


def _renderable_base_room_surfaces(
    *,
    building: dict[str, Any] | None,
    roof: dict[str, Any] | None,
    room_indices: set[int],
    primary_part_id_by_room_index: dict[int, str],
    part_id: str,
) -> list[dict[str, Any]]:
    if building is None or not room_indices:
        return []
    room_surfaces_by_room = _room_partition_surfaces_by_room(roof)
    story_unions = _building_story_unions(building)
    renderable: list[dict[str, Any]] = []
    rooms = building.get("rooms") or []
    for room_index in sorted(room_indices):
        if primary_part_id_by_room_index.get(room_index) != part_id:
            continue
        if room_index < 0 or room_index >= len(rooms):
            continue
        room = rooms[room_index] or {}
        story = int(room.get("story", 0) or 0)
        fenestration = [
            *(
                {"corners": item.get("corners") or []}
                for item in (room.get("windows") or [])
            ),
            *(
                {"corners": item.get("corners") or []}
                for item in (room.get("doors") or [])
            ),
            *(
                {"corners": item.get("corners") or []}
                for item in (room.get("openings") or [])
            ),
        ]
        floor_surface = _renderable_surface_from_room_floor(
            room, part_id=part_id, room_index=room_index, story=story
        )
        if floor_surface is not None:
            renderable.append(floor_surface)
        renderable.extend(
            _renderable_fenestration_surfaces(
                room,
                part_id=part_id,
                room_index=room_index,
                story=story,
            )
        )
        room_surfaces = room_surfaces_by_room.get(room_index) or []
        story_union = story_unions.get(story)
        for wall_index, wall in enumerate(room.get("walls_computed") or []):
            renderable.extend(
                _renderable_surfaces_from_room_wall(
                    wall,
                    part_id=part_id,
                    room_index=room_index,
                    story=story,
                    room_surfaces=room_surfaces,
                    openings=fenestration,
                    wall_index=wall_index,
                    story_union=story_union,
                )
            )
            extension_strip = wall.get("extension_strip")
            if not extension_strip:
                continue
            strips = (
                extension_strip
                if isinstance(extension_strip[0], list)
                and extension_strip
                and isinstance(extension_strip[0][0], list)
                else [extension_strip]
            )
            for extension_index, strip in enumerate(strips):
                renderable.extend(
                    _renderable_surfaces_from_room_wall(
                        {"id": wall.get("id"), "corners": strip},
                        part_id=part_id,
                        room_index=room_index,
                        story=story,
                        room_surfaces=room_surfaces,
                        openings=fenestration,
                        wall_index=wall_index,
                        story_union=story_union,
                        extension_index=extension_index,
                    )
                )
    return renderable


def _renderable_category_for_atom(
    atom: dict[str, Any],
    *,
    rooms_with_per_room_unresolved: set[str] | None = None,
) -> str | None:
    role = str(atom.get("role") or "")
    if role == "sloped_ceiling":
        return "room_ceiling_sloped"
    if role == "flat_ceiling":
        return "room_ceiling_flat"
    if role in {"attic_floor", "attic_floor_inferred"}:
        return "attic_floor"
    if role in {"flat_transition_cap", "flat_transition_cap_inferred"}:
        return "room_ceiling_flat"
    # Candidate atoms are evidence-supported but not strong enough to commit
    # as attic/transition caps in the Full model. Render them as
    # low-confidence unresolved-region overlays rather than leaving silent
    # holes where the Full model declines to commit. The user preference is
    # "low-confidence markers rather than silent holes" — this realises it
    # at per-atom granularity so regions adjacent to committed attic
    # surfaces (e.g. 5c557e06 room 0, where 2 exact-cell attic atoms sit
    # next to 1 demoted P1a atom) do not leave visible gaps. Suppress when
    # the room already emits a whole-room unresolved fallback so we don't
    # paint overlapping overlays.
    if role in {"attic_floor_candidate", "flat_transition_cap_candidate"}:
        room_id = str(atom.get("room_id") or "")
        if rooms_with_per_room_unresolved and room_id in rooms_with_per_room_unresolved:
            return None
        return "unresolved_region"
    return None


def _renderable_surface_from_atom(
    atom: dict[str, Any],
    *,
    rooms_with_per_room_unresolved: set[str] | None = None,
) -> dict[str, Any] | None:
    category = _renderable_category_for_atom(
        atom,
        rooms_with_per_room_unresolved=rooms_with_per_room_unresolved,
    )
    corners = atom.get("poly") or []
    if category is None or not isinstance(corners, list) or len(corners) < 3:
        return None
    raw_holes = atom.get("holes") or []
    holes = [ring for ring in raw_holes if isinstance(ring, list) and len(ring) >= 3]
    return {
        "id": f"renderable:{category}:{atom.get('id')}",
        "category": category,
        "source_kind": "semantic_atom",
        "source_id": atom.get("id"),
        "corners": corners,
        "holes": holes,
        "part_id": atom.get("effective_part_id") or UNASSIGNED_PART_ID,
        "room_id": atom.get("room_id"),
        "room_index": atom.get("room_index"),
        "story": atom.get("story"),
        "role": atom.get("role"),
        "roof_hypothesis_id": atom.get("roof_hypothesis_id"),
        "top_y_m": atom.get("top_y_m"),
    }


def _part_semantic_atoms(
    *,
    summary: dict[str, Any],
    part_id: str,
    room_indices: set[int],
    include_all_rooms: bool,
) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for atom in summary.get("semantic_atoms") or []:
        if not isinstance(atom, dict):
            continue
        atom_part_id = str(atom.get("effective_part_id") or UNASSIGNED_PART_ID)
        room_index = atom.get("room_index")
        if not include_all_rooms:
            if atom_part_id != part_id and (
                not isinstance(room_index, int) or room_index not in room_indices
            ):
                continue
            if part_id == UNASSIGNED_PART_ID and atom_part_id not in {
                UNASSIGNED_PART_ID,
                part_id,
            }:
                continue
        atoms.append(atom)
    return atoms


def _renderable_surface_from_unresolved_region(
    region: dict[str, Any],
) -> dict[str, Any] | None:
    corners = region.get("polygon") or []
    if not isinstance(corners, list) or len(corners) < 3:
        return None
    part_ids = [
        str(value) for value in (region.get("effective_part_ids") or []) if value
    ]
    return {
        "id": f"renderable:unresolved_region:{region.get('id')}",
        "category": "unresolved_region",
        "source_kind": "unresolved_region",
        "source_id": region.get("id"),
        "corners": corners,
        "part_id": part_ids[0] if part_ids else UNASSIGNED_PART_ID,
        "part_ids": part_ids or [UNASSIGNED_PART_ID],
        "room_id": region.get("room_id"),
        "room_index": region.get("room_index"),
        "story": region.get("story"),
        "roof_evidence_score": region.get("roof_evidence_score"),
    }


def _renderable_surface_from_roof_surface(
    surface: dict[str, Any],
    *,
    part_id: str,
    surface_id: str,
    source_kind: str,
) -> dict[str, Any] | None:
    corners = _serialize_corners(surface.get("corners") or [])
    if len(corners) < 3:
        return None
    room_index = surface.get("room_index")
    room_id = _room_key(room_index) if isinstance(room_index, int) else None
    return {
        "id": f"renderable:exterior_roof:{surface_id}",
        "category": "exterior_roof",
        "source_kind": source_kind,
        "source_id": surface.get("boundary_face_id") or surface_id,
        "corners": corners,
        "part_id": part_id,
        "room_id": room_id,
        "room_index": room_index,
        "story": surface.get("story", surface.get("dominant_story")),
        "roof_hypothesis_id": surface.get("roof_hypothesis_id"),
        "surface_kind": surface.get("surface_kind"),
        "flat_role": surface.get("flat_role"),
    }


def _renderable_surface_from_coverage_patch(
    patch: dict[str, Any],
    *,
    part_id: str,
) -> dict[str, Any] | None:
    corners = _serialize_corners(patch.get("polygon") or [])
    if len(corners) < 3:
        return None
    room_indices = [
        int(value)
        for value in (patch.get("room_indices") or [])
        if isinstance(value, int)
    ]
    room_ids = [_room_key(room_index) for room_index in room_indices]
    room_index = room_indices[0] if len(room_indices) == 1 else None
    room_id = room_ids[0] if len(room_ids) == 1 else None
    part_ids = [
        str(value) for value in (patch.get("effective_part_ids") or []) if value
    ]
    return {
        "id": f"renderable:exterior_roof:{patch.get('id')}",
        "category": "exterior_roof",
        "source_kind": "roof_coverage_patch",
        "source_id": patch.get("id"),
        "corners": corners,
        "part_id": part_id,
        "part_ids": part_ids or [part_id],
        "room_id": room_id,
        "room_index": room_index,
        "room_ids": room_ids,
        "room_indices": room_indices,
        "story": patch.get("story"),
        "roof_hypothesis_id": patch.get("roof_hypothesis_id"),
        "surface_kind": patch.get("surface_kind"),
        "coverage_subpart_id": patch.get("coverage_subpart_id"),
        "coverage_semantic_kind": patch.get("coverage_semantic_kind"),
        "continuation_source": patch.get("continuation_source"),
    }


def _fallback_unresolved_region_from_roof_surface(
    surface: dict[str, Any],
    *,
    part_id: str,
    surface_id: str,
) -> dict[str, Any] | None:
    polygon = _serialize_corners(surface.get("corners") or [])
    if len(polygon) < 3:
        return None
    room_index = surface.get("room_index")
    room_id = _room_key(room_index) if isinstance(room_index, int) else None
    polygon_xz = [
        [_round6(corner[0]), _round6(corner[2])]
        for corner in polygon
        if len(corner) >= 3
    ]
    return {
        "id": f"unresolved-fallback-roof:{surface_id}",
        "room_id": room_id,
        "room_index": room_index,
        "story": surface.get("story", surface.get("dominant_story")),
        "effective_part_ids": [part_id],
        "polygon": polygon,
        "polygon_xz": polygon_xz,
        "roof_evidence_score": 0,
        "fallback_source_kind": "roof_surface_fallback",
        "roof_hypothesis_id": surface.get("roof_hypothesis_id"),
    }


# Minimum residual area (m^2) for emitting a surface-level unresolved fallback.
# Below this, slivers along atom edges are suppressed as numerical noise rather
# than honest "uncovered" evidence.
ROOF_SURFACE_FALLBACK_MIN_AREA_M2 = 0.25


def _atoms_covering_roof_surface(
    surface: dict[str, Any],
    atoms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return atoms that share the surface's roof hypothesis (primary or sloped)."""
    hypothesis_id = str(surface.get("roof_hypothesis_id") or "")
    if not hypothesis_id:
        return []
    surface_kind = str(surface.get("surface_kind") or surface.get("kind") or "")
    matches: list[dict[str, Any]] = []
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        atom_roof_id = str(atom.get("roof_hypothesis_id") or "")
        atom_sloped_id = str(atom.get("sloped_hypothesis_id") or "")
        if atom_roof_id == hypothesis_id:
            matches.append(atom)
            continue
        # Oblique surface subtraction: flat atoms beneath that reference the same
        # sloped hypothesis (attic_floor / flat_transition_cap) cover the oblique
        # footprint from below, so they are valid subtraction sources.
        if surface_kind == "oblique" and atom_sloped_id == hypothesis_id:
            matches.append(atom)
    return matches


def _lift_xz_polygon_to_surface_plane(
    polygon_xz: Polygon,
    surface_corners: list[list[float]],
) -> list[list[float]] | None:
    if polygon_xz.is_empty or polygon_xz.area < ROOF_SURFACE_FALLBACK_MIN_AREA_M2:
        return None
    exterior = list(polygon_xz.exterior.coords)
    if exterior and exterior[-1] == exterior[0]:
        exterior = exterior[:-1]
    if len(exterior) < 3:
        return None
    points = [
        (float(c[0]), float(c[1]), float(c[2]))
        for c in surface_corners
        if isinstance(c, (list, tuple)) and len(c) >= 3
    ]
    ys = [p[1] for p in points]
    if not ys:
        return [[float(x), 0.0, float(z)] for (x, z) in exterior]
    if max(ys) - min(ys) < 1e-4:
        y_const = sum(ys) / len(ys)
        return [[float(x), float(y_const), float(z)] for (x, z) in exterior]
    try:
        import numpy as np

        a_mat = np.array([[p[0], p[2], 1.0] for p in points], dtype=float)
        b_vec = np.array(ys, dtype=float)
        coeffs, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        return [
            [float(x), a * float(x) + b * float(z) + c, float(z)] for (x, z) in exterior
        ]
    except Exception:
        y_mean = sum(ys) / len(ys)
        return [[float(x), float(y_mean), float(z)] for (x, z) in exterior]


def _split_polygon_holes(geom: Polygon, min_area: float) -> list[Polygon]:
    """Decompose a polygon (possibly with interior holes) into simple hole-free
    polygons by cutting vertical lines through each hole's centroid.

    Viewer meshes are simple ring polygons; dropping a holed polygon's interior
    ring silently back-fills covered regions, so we split instead.
    """
    if geom.geom_type != "Polygon" or geom.area < min_area:
        return []
    if not list(geom.interiors):
        return [geom]
    _minx, miny, _maxx, maxy = geom.bounds
    cut_xs = sorted({round(interior.centroid.x, 6) for interior in geom.interiors})
    lines = [LineString([(x, miny - 1.0), (x, maxy + 1.0)]) for x in cut_xs]
    pieces: list[Polygon] = []
    try:
        splitter = MultiLineString(lines) if len(lines) > 1 else lines[0]
        split_result = shapely_split(geom, splitter)
        for piece in getattr(split_result, "geoms", [split_result]):
            if piece.geom_type != "Polygon" or piece.area < min_area:
                continue
            if list(piece.interiors):
                pieces.extend(_split_polygon_holes(piece, min_area))
            else:
                pieces.append(piece)
    except Exception:
        pieces = []
    if pieces:
        return pieces
    try:
        tris = shapely.constrained_delaunay_triangles(geom)
        return [
            t
            for t in getattr(tris, "geoms", [])
            if t.geom_type == "Polygon" and t.area >= min_area
        ]
    except Exception:
        return []


def _residual_fallback_regions_from_roof_surface(
    surface: dict[str, Any],
    *,
    part_id: str,
    surface_id: str,
    covering_atoms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Emit unresolved_region fallbacks only for the parts of ``surface`` not
    already covered by atom-level committed geometry.

    If no atoms overlap, emits the full surface as a single fallback (existing
    behaviour). If atoms fully cover the surface, returns an empty list. If
    atoms partially cover, emits one fallback per residual polygon so the
    viewer paints orange only on genuinely uncovered regions.
    """
    surface_corners_raw = surface.get("corners") or []
    if len(surface_corners_raw) < 3:
        return []
    surface_corners = [
        [float(c[0]), float(c[1]), float(c[2])]
        for c in surface_corners_raw
        if isinstance(c, (list, tuple)) and len(c) >= 3
    ]
    if len(surface_corners) < 3:
        return []

    single_fallback = _fallback_unresolved_region_from_roof_surface(
        surface,
        part_id=part_id,
        surface_id=surface_id,
    )
    if single_fallback is None:
        return []
    if not covering_atoms:
        return [single_fallback]

    try:
        surface_poly_xz = Polygon([(c[0], c[2]) for c in surface_corners])
    except Exception:
        return [single_fallback]
    surface_poly_xz = make_valid(surface_poly_xz)
    if surface_poly_xz.is_empty:
        return []

    covering_polys: list[Polygon] = []
    for atom in covering_atoms:
        poly_coords = atom.get("poly") or []
        if len(poly_coords) < 3:
            continue
        try:
            poly = Polygon(
                [(float(c[0]), float(c[2])) for c in poly_coords if len(c) >= 3]
            )
        except Exception:
            continue
        cleaned = make_valid(poly.buffer(0.001))
        if not cleaned.is_empty:
            covering_polys.append(cleaned)
    if not covering_polys:
        return [single_fallback]

    covered = unary_union(covering_polys)
    try:
        residual = surface_poly_xz.difference(covered)
    except Exception:
        return [single_fallback]
    if residual.is_empty:
        return []

    room_index = surface.get("room_index")
    room_id = _room_key(room_index) if isinstance(room_index, int) else None
    story = surface.get("story", surface.get("dominant_story"))
    hypothesis_id = surface.get("roof_hypothesis_id")

    raw_geoms = (
        [residual]
        if residual.geom_type == "Polygon"
        else list(getattr(residual, "geoms", []))
    )
    simple_pieces: list[Polygon] = []
    for geom in raw_geoms:
        simple_pieces.extend(
            _split_polygon_holes(geom, ROOF_SURFACE_FALLBACK_MIN_AREA_M2)
        )
    regions: list[dict[str, Any]] = []
    for index, geom in enumerate(simple_pieces):
        lifted = _lift_xz_polygon_to_surface_plane(geom, surface_corners)
        if lifted is None:
            continue
        polygon_xz = [[_round6(p[0]), _round6(p[2])] for p in lifted]
        regions.append(
            {
                "id": f"unresolved-fallback-roof:{surface_id}:residual:{index}",
                "room_id": room_id,
                "room_index": room_index,
                "story": story,
                "effective_part_ids": [part_id],
                "polygon": lifted,
                "polygon_xz": polygon_xz,
                "roof_evidence_score": 0,
                "fallback_source_kind": "roof_surface_fallback",
                "roof_hypothesis_id": hypothesis_id,
            }
        )
    return regions


def _room_summary_for_room_index(
    summary: dict[str, Any], room_index: int | None
) -> dict[str, Any]:
    if not isinstance(room_index, int):
        return {}
    return (summary.get("room_summaries") or {}).get(_room_key(room_index)) or {}


def _is_exact_flat_roof_surface(
    *,
    surface: dict[str, Any],
    summary: dict[str, Any],
) -> bool:
    if str(surface.get("surface_kind") or surface.get("kind") or "") != "flat":
        return False
    if not isinstance(surface.get("room_index"), int):
        return False
    if str(surface.get("flat_role") or "") == "ambiguous_flat_over_sloped_part":
        return False
    room_summary = _room_summary_for_room_index(summary, surface.get("room_index"))
    if not room_summary:
        return True
    if bool(
        room_summary.get("has_attic_relation")
        or room_summary.get("has_upper_void_relation")
    ):
        return True
    roles = {str(value) for value in (room_summary.get("roles") or []) if value}
    has_sloped_semantics = bool(
        room_summary.get("has_oblique_atom")
        or room_summary.get("covered_by_sloped_roof")
        or room_summary.get("partially_covered_by_sloped_roof")
        or room_summary.get("strong_perimeter_sloped")
        or room_summary.get("strong_knee_wall_signal")
        or "sloped_ceiling" in roles
    )
    if has_sloped_semantics:
        return False
    if bool(room_summary.get("has_resolved_roof_relation")) and bool(
        room_summary.get("has_attic_relation")
        or room_summary.get("has_upper_void_relation")
    ):
        return True
    return not any(
        [
            bool(room_summary.get("has_candidate_attic_relation")),
            bool(room_summary.get("has_candidate_upper_void_relation")),
            int(room_summary.get("roof_evidence_score", 0) or 0) >= 4,
        ]
    )


def _drop_room_flat_fallback_over_sloped_semantics(
    *,
    surface: dict[str, Any],
    summary: dict[str, Any],
) -> bool:
    if str(surface.get("surface_kind") or surface.get("kind") or "") != "flat":
        return False
    room_index = surface.get("room_index")
    if not isinstance(room_index, int):
        return False
    room_summary = _room_summary_for_room_index(summary, room_index)
    if not room_summary:
        return False
    if bool(
        room_summary.get("has_attic_relation")
        or room_summary.get("has_upper_void_relation")
    ):
        return False
    roles = {str(value) for value in (room_summary.get("roles") or []) if value}
    if "sloped_ceiling" not in roles and not bool(room_summary.get("has_oblique_atom")):
        return False
    return True


def _skip_roomless_ambiguous_flat_surface(surface: dict[str, Any]) -> bool:
    if str(surface.get("surface_kind") or surface.get("kind") or "") != "flat":
        return False
    if isinstance(surface.get("room_index"), int):
        return False
    return str(surface.get("flat_role") or "") == "ambiguous_flat_over_sloped_part"


def _drop_roomless_flat_fallback_without_atoms(
    *,
    surface: dict[str, Any],
    roof: dict[str, Any] | None,
    summary: dict[str, Any],
) -> bool:
    if str(surface.get("surface_kind") or surface.get("kind") or "") != "flat":
        return False
    if isinstance(surface.get("room_index"), int):
        return False
    roof_hypothesis_id = str(surface.get("roof_hypothesis_id") or "")
    if not roof_hypothesis_id:
        return False

    semantic_atoms = [
        atom
        for atom in (summary.get("semantic_atoms") or [])
        if isinstance(atom, dict)
        and str(atom.get("kind") or "") == "flat"
        and str(atom.get("roof_hypothesis_id") or "") == roof_hypothesis_id
    ]
    if semantic_atoms:
        return False

    hypothesis_graph = (roof or {}).get("roof_hypothesis_graph") or {}
    selected_hypothesis_ids = {
        str(value)
        for value in (hypothesis_graph.get("selected_hypothesis_ids") or [])
        if value
    }
    if roof_hypothesis_id not in selected_hypothesis_ids:
        return True

    selected_cover_room_ids = [
        str(edge.get("to"))
        for edge in (hypothesis_graph.get("edges") or [])
        if isinstance(edge, dict)
        and str(edge.get("type") or "") == "COVERS_ROOM"
        and str(edge.get("from") or "") == roof_hypothesis_id
        and bool(edge.get("selected"))
        and isinstance(edge.get("to"), str)
        and str(edge.get("to")).startswith("room:")
    ]
    if not selected_cover_room_ids:
        return True

    room_summaries = summary.get("room_summaries") or {}
    for room_id in selected_cover_room_ids:
        room_summary = room_summaries.get(room_id) or {}
        if not bool(room_summary.get("has_resolved_roof_relation")):
            # No semantic atom and no resolved semantic room support. Reject the
            # legacy roomless flat fallback instead of inventing a payload-level
            # unresolved patch that the summary layer does not recognize.
            return True
    return True


def _roof_atom_patch_payload(
    *,
    summary: dict[str, Any],
    part_id: str,
    room_indices: set[int],
    roof_hypothesis_id: str,
    include_all_rooms: bool,
    surface_kind: str = "oblique",
) -> list[dict[str, Any]]:
    if not roof_hypothesis_id:
        return []
    renderable: list[dict[str, Any]] = []
    emitted_ids: set[str] = set()
    room_summaries = summary.get("room_summaries") or {}
    for atom in summary.get("semantic_atoms") or []:
        if not isinstance(atom, dict):
            continue
        if str(atom.get("kind") or "") != str(surface_kind):
            continue
        if str(atom.get("roof_hypothesis_id") or "") != roof_hypothesis_id:
            continue
        room_index = atom.get("room_index")
        room_summary = room_summaries.get(str(atom.get("room_id") or "")) or {}
        if surface_kind == "flat":
            if str(atom.get("flat_role") or "") == "ambiguous_flat_over_sloped_part":
                continue
            # Flat roof atom patches should only represent true flat roof exterior
            # surfaces, not interior top-boundary semantics like attic floors or
            # flat transition caps.
            if str(atom.get("role") or "") != "flat_ceiling":
                continue
            if str(atom.get("flat_role") or "") != "roof_flat":
                continue
            if bool(room_summary.get("mixed")):
                continue
            if bool(room_summary.get("has_oblique_atom")):
                continue
            if bool(room_summary.get("covered_by_sloped_roof")) or bool(
                room_summary.get("partially_covered_by_sloped_roof")
            ):
                continue
            if bool(room_summary.get("strong_perimeter_sloped")) or bool(
                room_summary.get("strong_knee_wall_signal")
            ):
                continue
        else:
            if str(atom.get("role") or "") != "sloped_ceiling":
                continue
            if room_summary:
                if str(atom.get("sloped_coverage_state") or "") != "confirmed":
                    continue
                clearance = atom.get("sloped_vertical_clearance_m")
                if isinstance(clearance, (int, float)) and float(clearance) < -0.05:
                    continue
        atom_part_id = str(atom.get("effective_part_id") or UNASSIGNED_PART_ID)
        if not include_all_rooms:
            if atom_part_id != part_id and (
                not isinstance(room_index, int) or room_index not in room_indices
            ):
                continue
        corners = _serialize_corners(atom.get("poly") or [])
        if len(corners) < 3:
            continue
        atom_id = str(atom.get("id") or "")
        renderable_id = (
            f"renderable:exterior_roof:roof-atom-patch:{surface_kind}:{atom_id}"
        )
        if renderable_id in emitted_ids:
            continue
        emitted_ids.add(renderable_id)
        renderable.append(
            {
                "id": renderable_id,
                "category": "exterior_roof",
                "source_kind": "roof_atom_patch",
                "source_id": atom_id,
                "corners": corners,
                "part_id": part_id if include_all_rooms else atom_part_id,
                "room_id": atom.get("room_id"),
                "room_index": room_index,
                "story": atom.get("story"),
                "roof_hypothesis_id": roof_hypothesis_id,
                "surface_kind": surface_kind,
            }
        )
    return renderable


def _room_flat_roof_atom_patch_payload(
    *,
    summary: dict[str, Any],
    part_id: str,
    room_indices: set[int],
    room_index: int,
    include_all_rooms: bool,
    emitted_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    renderable: list[dict[str, Any]] = []
    local_emitted_ids = emitted_ids if emitted_ids is not None else set()
    room_summaries = summary.get("room_summaries") or {}
    room_summary = room_summaries.get(_room_key(room_index)) or {}
    if bool(room_summary.get("mixed")):
        return renderable
    if bool(room_summary.get("has_oblique_atom")):
        return renderable
    if bool(room_summary.get("covered_by_sloped_roof")) or bool(
        room_summary.get("partially_covered_by_sloped_roof")
    ):
        return renderable
    if bool(room_summary.get("strong_perimeter_sloped")) or bool(
        room_summary.get("strong_knee_wall_signal")
    ):
        return renderable
    for atom in summary.get("semantic_atoms") or []:
        if not isinstance(atom, dict):
            continue
        if str(atom.get("kind") or "") != "flat":
            continue
        if str(atom.get("role") or "") != "flat_ceiling":
            continue
        if str(atom.get("flat_role") or "") != "roof_flat":
            continue
        if atom.get("room_index") != room_index:
            continue
        atom_part_id = str(atom.get("effective_part_id") or UNASSIGNED_PART_ID)
        if (
            not include_all_rooms
            and atom_part_id != part_id
            and room_index not in room_indices
        ):
            continue
        corners = _serialize_corners(atom.get("poly") or [])
        if len(corners) < 3:
            continue
        atom_id = str(atom.get("id") or "")
        renderable_id = f"renderable:exterior_roof:roof-atom-patch:flat:{atom_id}"
        if renderable_id in local_emitted_ids:
            continue
        local_emitted_ids.add(renderable_id)
        renderable.append(
            {
                "id": renderable_id,
                "category": "exterior_roof",
                "source_kind": "roof_atom_patch",
                "source_id": atom_id,
                "corners": corners,
                "part_id": part_id if include_all_rooms else atom_part_id,
                "room_id": atom.get("room_id"),
                "room_index": room_index,
                "story": atom.get("story"),
                "roof_hypothesis_id": atom.get("roof_hypothesis_id"),
                "surface_kind": "flat",
            }
        )
    return renderable


def _roof_surface_fallback_payload(
    *,
    roof: dict[str, Any] | None,
    summary: dict[str, Any],
    part_id: str,
    room_indices: set[int],
    exact_roof_room_indices: set[int],
    exact_roof_hypothesis_ids: set[str],
    include_all_rooms: bool,
    atoms_for_subtraction: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    if roof is None:
        return [], [], 0, 0
    roof_surfaces = roof.get("roof_surfaces") or {}
    atoms_for_subtraction = atoms_for_subtraction or []
    renderable: list[dict[str, Any]] = []
    unresolved_regions: list[dict[str, Any]] = []
    exact_flat_surface_count = 0
    coverage_patch_surface_count = 0
    emitted_room_flat_atom_ids: set[str] = set()
    emitted_renderable_ids: set[str] = set()

    def append_unique_surfaces(surfaces: list[dict[str, Any]]) -> int:
        appended = 0
        for item in surfaces:
            surface_id = str(item.get("id") or "")
            if surface_id and surface_id in emitted_renderable_ids:
                continue
            if surface_id:
                emitted_renderable_ids.add(surface_id)
            renderable.append(item)
            appended += 1
        return appended

    for surface_kind in ("oblique", "flat"):
        for index, surface in enumerate(roof_surfaces.get(surface_kind) or []):
            if not isinstance(surface, dict):
                continue
            if _skip_roomless_ambiguous_flat_surface(surface):
                continue
            room_index = surface.get("room_index")
            roof_hypothesis_id = str(surface.get("roof_hypothesis_id") or "")
            if isinstance(room_index, int):
                if not include_all_rooms and room_index not in room_indices:
                    continue
                if room_index in exact_roof_room_indices:
                    continue
            elif not include_all_rooms:
                continue
            elif roof_hypothesis_id and roof_hypothesis_id in exact_roof_hypothesis_ids:
                continue
            elif surface_kind == "oblique":
                atom_patches = _roof_atom_patch_payload(
                    summary=summary,
                    part_id=part_id,
                    room_indices=room_indices,
                    roof_hypothesis_id=roof_hypothesis_id,
                    include_all_rooms=include_all_rooms,
                    surface_kind="oblique",
                )
                if atom_patches:
                    append_unique_surfaces(atom_patches)
                    continue
                surface_id = str(
                    surface.get("boundary_face_id")
                    or surface.get("roof_hypothesis_id")
                    or f"{surface_kind}:{index}"
                )
                covering_atoms = _atoms_covering_roof_surface(
                    surface, atoms_for_subtraction
                )
                unresolved_regions.extend(
                    _residual_fallback_regions_from_roof_surface(
                        surface,
                        part_id=part_id,
                        surface_id=surface_id,
                        covering_atoms=covering_atoms,
                    )
                )
                continue
            if not isinstance(room_index, int):
                # Top-kind roomless flats are whole-building roof decks that
                # duplicate the per-room layer (intermediate flats, oblique
                # surfaces, ceiling atoms). Drop them to avoid double coverage.
                if str(surface.get("kind") or "") == "top":
                    continue
                atom_patches = _roof_atom_patch_payload(
                    summary=summary,
                    part_id=part_id,
                    room_indices=room_indices,
                    roof_hypothesis_id=roof_hypothesis_id,
                    include_all_rooms=include_all_rooms,
                    surface_kind="flat",
                )
                if atom_patches:
                    renderable.extend(atom_patches)
                    exact_flat_surface_count += len(atom_patches)
                    continue
                if _drop_roomless_flat_fallback_without_atoms(
                    surface=surface, roof=roof, summary=summary
                ):
                    continue
                surface_id = str(
                    surface.get("boundary_face_id")
                    or surface.get("roof_hypothesis_id")
                    or f"{surface_kind}:{index}"
                )
                covering_atoms = _atoms_covering_roof_surface(
                    surface, atoms_for_subtraction
                )
                unresolved_regions.extend(
                    _residual_fallback_regions_from_roof_surface(
                        surface,
                        part_id=part_id,
                        surface_id=surface_id,
                        covering_atoms=covering_atoms,
                    )
                )
                continue
            room_atom_patches = _room_flat_roof_atom_patch_payload(
                summary=summary,
                part_id=part_id,
                room_indices=room_indices,
                room_index=room_index,
                include_all_rooms=include_all_rooms,
                emitted_ids=emitted_room_flat_atom_ids,
            )
            if room_atom_patches:
                exact_flat_surface_count += append_unique_surfaces(room_atom_patches)
                continue
            surface_id = str(
                surface.get("boundary_face_id")
                or surface.get("roof_hypothesis_id")
                or f"{surface_kind}:{index}"
            )
            source_kind = "roof_surface_fallback"
            emit_unresolved = True
            if _drop_room_flat_fallback_over_sloped_semantics(
                surface=surface, summary=summary
            ):
                continue
            if _is_exact_flat_roof_surface(surface=surface, summary=summary):
                source_kind = "roof_surface_exact_flat"
                emit_unresolved = False
                exact_flat_surface_count += 1
            fallback_surface = _renderable_surface_from_roof_surface(
                surface,
                part_id=part_id,
                surface_id=surface_id,
                source_kind=source_kind,
            )
            if fallback_surface is None:
                continue
            append_unique_surfaces([fallback_surface])
            if emit_unresolved:
                covering_atoms = _atoms_covering_roof_surface(
                    surface, atoms_for_subtraction
                )
                unresolved_regions.extend(
                    _residual_fallback_regions_from_roof_surface(
                        surface,
                        part_id=part_id,
                        surface_id=surface_id,
                        covering_atoms=covering_atoms,
                    )
                )
    return (
        renderable,
        unresolved_regions,
        exact_flat_surface_count,
        coverage_patch_surface_count,
    )


def _part_unresolved_regions(
    *,
    summary: dict[str, Any],
    part_id: str,
    room_indices: set[int],
    include_all_rooms: bool,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for region in summary.get("unresolved_regions") or []:
        if not isinstance(region, dict):
            continue
        effective_part_ids = {
            str(value) for value in (region.get("effective_part_ids") or []) if value
        }
        room_index = region.get("room_index")
        if include_all_rooms:
            regions.append(dict(region))
            continue
        if part_id in effective_part_ids:
            regions.append(dict(region))
            continue
        if isinstance(room_index, int) and room_index in room_indices:
            regions.append(dict(region))
    return regions


def _renderable_surface_from_roof_face(
    face: dict[str, Any], cell: dict[str, Any], part_id: str
) -> dict[str, Any] | None:
    corners = face.get("corners") or []
    if not isinstance(corners, list) or len(corners) < 3:
        return None
    if str(face.get("role") or "") != "roof":
        return None
    return {
        "id": (
            f"renderable:exterior_roof:{cell.get('id')}:"
            f"{face.get('id') or face.get('kind') or 'face'}"
        ),
        "category": "exterior_roof",
        "source_kind": "roof_cell_face",
        "source_id": face.get("id") or face.get("kind") or "face",
        "cell_id": cell.get("id"),
        "cell_kind": cell.get("cell_kind"),
        "corners": corners,
        "part_id": part_id,
        "room_id": cell.get("room_id"),
        "room_index": cell.get("room_index"),
        "story": cell.get("story"),
        "roof_hypothesis_id": cell.get("roof_hypothesis_id")
        or (face.get("metadata") or {}).get("roof_hypothesis_id"),
    }


def _renderable_surface_from_knee_wall(
    wall: dict[str, Any], part_id: str
) -> dict[str, Any] | None:
    corners = wall.get("corners") or []
    if not isinstance(corners, list) or len(corners) < 3:
        return None
    return {
        "id": f"renderable:knee_wall:{wall.get('id')}",
        "category": "knee_wall",
        "source_kind": "knee_wall",
        "source_id": wall.get("id"),
        "corners": corners,
        "part_id": part_id,
        "room_id": wall.get("room_id"),
        "room_index": wall.get("room_index"),
        "story": wall.get("story"),
    }


def _surface_category_counts(surfaces: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for surface in surfaces:
        category = str(surface.get("category") or "")
        if category:
            counts[category] += 1
    return dict(sorted(counts.items()))


def _dedupe_renderable_surfaces(surfaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    emitted_ids: set[str] = set()
    for surface in surfaces:
        surface_id = str(surface.get("id") or "")
        if surface_id and surface_id in emitted_ids:
            continue
        if surface_id:
            emitted_ids.add(surface_id)
        deduped.append(surface)
    return deduped


def _resolve_full_model_surface_overlaps(
    surfaces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve competing ceiling interpretations for committed full-model output.

    Full model should show the selected geometry, not overlapping diagnostic
    alternatives. Keep the strongest ceiling interpretation per room + height
    band and drop lower-priority coplanar alternatives.
    """
    ceiling_priority = {
        "room_ceiling_sloped": 5,
        "room_ceiling_flat": 4,
        "attic_floor": 3,
        "base_room_ceiling": 2,
        "fallback_room_ceiling": 1,
    }
    passthrough: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)

    for surface in surfaces:
        category = str(surface.get("category") or "")
        room_index = surface.get("room_index")
        corners = surface.get("corners") or []
        if (
            category not in ceiling_priority
            or not isinstance(room_index, int)
            or not isinstance(corners, list)
            or len(corners) < 3
        ):
            passthrough.append(surface)
            continue
        y_values = sorted(
            {
                _round6(corner[1])
                for corner in corners
                if isinstance(corner, (list, tuple)) and len(corner) >= 3
            }
        )
        y_key = "|".join(f"{value:.3f}" for value in y_values)
        grouped[(room_index, y_key)].append(surface)

    resolved: list[dict[str, Any]] = list(passthrough)
    for group in grouped.values():
        best = max(
            ceiling_priority.get(str(surface.get("category") or ""), 0)
            for surface in group
        )
        resolved.extend(
            surface
            for surface in group
            if ceiling_priority.get(str(surface.get("category") or ""), 0) == best
        )
    return resolved


def _topology_cell_polygon(cell: dict[str, Any]) -> Polygon | None:
    footprint = (cell.get("properties") or {}).get("xz_footprint") or []
    if not isinstance(footprint, list) or len(footprint) < 3:
        return None
    coords: list[tuple[float, float]] = []
    for point in footprint:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        coords.append((_round6(point[0]), _round6(point[1])))
    if len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        try:
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda geom: geom.area, default=None)
        except Exception:
            return None
    if (
        poly is None
        or poly.is_empty
        or not isinstance(poly, Polygon)
        or poly.area <= 1e-6
    ):
        return None
    return poly


def _topology_story_unions(cells: list[dict[str, Any]]) -> dict[int, Polygon]:
    polys_by_story: dict[int, list[Polygon]] = defaultdict(list)
    for cell in cells:
        if str(cell.get("kind") or "") != "room":
            continue
        story = cell.get("story")
        if not isinstance(story, int):
            continue
        poly = _topology_cell_polygon(cell)
        if poly is not None:
            polys_by_story[story].append(poly)
    unions: dict[int, Polygon] = {}
    for story, polys in polys_by_story.items():
        if not polys:
            continue
        try:
            merged = unary_union(polys)
            merged = _largest_polygon(merged)
        except Exception:
            merged = None
        if merged is not None:
            unions[story] = merged
    return unions


def _face_corners(face: dict[str, Any]) -> list[list[float]]:
    vertices = face.get("corners") or face.get("vertices") or []
    corners: list[list[float]] = []
    for vertex in vertices:
        if isinstance(vertex, (list, tuple)) and len(vertex) >= 3:
            corners.append([_round6(vertex[0]), _round6(vertex[1]), _round6(vertex[2])])
    return corners


def _wall_face_xz_edge(face: dict[str, Any]) -> LineString | None:
    corners = _face_corners(face)
    unique: list[tuple[float, float]] = []
    for corner in corners:
        xz = (_round6(corner[0]), _round6(corner[2]))
        if xz not in unique:
            unique.append(xz)
    if len(unique) < 2:
        return None
    line = LineString([unique[0], unique[1]])
    return line if line.length > 1e-6 else None


def _is_exterior_wall_face(face: dict[str, Any], story_union: Polygon | None) -> bool:
    metadata = face.get("metadata") or {}
    if bool(metadata.get("perimeter_facing")):
        return True
    if story_union is None:
        return False
    wall_edge = _wall_face_xz_edge(face)
    if wall_edge is None:
        return False
    try:
        overlap = wall_edge.intersection(story_union.boundary)
    except Exception:
        return False
    return not overlap.is_empty and float(getattr(overlap, "length", 0.0)) > 1e-6


def _renderable_surface_from_topology_face(
    face: dict[str, Any],
    cell: dict[str, Any],
    *,
    part_id: str,
    is_exterior_wall: bool,
    include_ceiling: bool,
) -> dict[str, Any] | None:
    if str(cell.get("kind") or "") != "room":
        return None
    corners = _face_corners(face)
    if len(corners) < 3:
        return None
    metadata = face.get("metadata") or {}
    role = str(face.get("role") or "")
    face_kind = str(metadata.get("face_kind") or face.get("boundary_kind") or "")
    category: str | None = None
    if face_kind == "bottom":
        category = "occupied_room_floor"
    elif face_kind == "top":
        if include_ceiling:
            category = "occupied_room_ceiling"
    elif role == "wall":
        category = "exterior_wall" if is_exterior_wall else "occupied_room_wall"
    if category is None:
        return None
    return {
        "id": (
            f"renderable:{category}:{cell.get('id')}:"
            f"{face.get('id') or face_kind or role}"
        ),
        "category": category,
        "source_kind": "topology_room_face",
        "source_id": face.get("id") or face_kind or role,
        "cell_id": cell.get("id"),
        "corners": corners,
        "part_id": part_id,
        "room_id": cell.get("source_id"),
        "story": cell.get("story"),
        "face_kind": face_kind,
        "role": role,
    }


def _build_ontology_summary(
    *,
    uuid: str,
    roof: dict[str, Any],
    topology_cell_complex: dict[str, Any],
    building: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    building_part_graph = roof.get("building_part_graph") or {}
    roof_coverage_graph = roof.get("roof_coverage_graph") or {}
    top_boundary_graph = roof.get("top_boundary_graph") or {}
    roof_evidence_graph = roof.get("roof_evidence_graph") or {}
    roof_cell_complex = roof.get("roof_cell_complex") or {}
    roof_surfaces = roof.get("roof_surfaces") or {}
    roof_continuation_diagnostics = roof.get("roof_continuation_diagnostics") or {}
    room_partitions = (roof.get("ceiling_partitions") or {}).get(
        "room_partitions"
    ) or []
    dormers = roof.get("dormers") or []

    partition_by_id: dict[str, dict[str, Any]] = {}
    graph_room_by_room_id: dict[str, str] = {}
    room_partition_polys: dict[str, list[Polygon]] = defaultdict(list)
    room_partition_tops: dict[str, list[float]] = defaultdict(list)
    room_partition_count: dict[str, int] = defaultdict(int)
    room_indices_by_room_id: dict[str, int] = {}

    for room_partition in room_partitions:
        room_index = int(room_partition.get("room_index", 0))
        room_id = _room_key(room_index)
        room_indices_by_room_id[room_id] = room_index
        graph_room_id = room_partition.get("graph_room_id")
        if isinstance(graph_room_id, str) and graph_room_id:
            graph_room_by_room_id[room_id] = graph_room_id
        for partition in room_partition.get("partitions") or []:
            if not isinstance(partition, dict):
                continue
            atom_id = str(partition.get("id") or "")
            if atom_id:
                partition_by_id[atom_id] = partition
            poly = _poly_xz_from_3d(partition.get("poly") or [])
            if poly is not None:
                room_partition_polys[room_id].append(poly)
            top_y = partition.get("top_y_m")
            if isinstance(top_y, (int, float)):
                room_partition_tops[room_id].append(float(top_y))
            room_partition_count[room_id] += 1

    semantic_atoms: list[dict[str, Any]] = []
    atoms_by_id: dict[str, dict[str, Any]] = {}
    unassigned_room_ids: set[str] = set()
    for node in top_boundary_graph.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") != "TopBoundaryAtom":
            continue
        atom_id = str(node.get("id") or "")
        partition = partition_by_id.get(atom_id) or {}
        part_id = node.get("part_id")
        effective_part_id = str(part_id) if part_id else UNASSIGNED_PART_ID
        if not part_id and node.get("room_id"):
            unassigned_room_ids.add(str(node["room_id"]))
        record = dict(node)
        record["effective_part_id"] = effective_part_id
        record["poly"] = partition.get("poly") or []
        record["top_y_m"] = partition.get("top_y_m")
        record["supporting_roof_hypothesis_ids"] = list(
            partition.get("supporting_roof_hypothesis_ids") or []
        )
        record["flat_role_reason"] = partition.get("flat_role_reason")
        semantic_atoms.append(record)
        atoms_by_id[atom_id] = record

    part_nodes = [
        dict(node)
        for node in (building_part_graph.get("nodes") or [])
        if isinstance(node, dict)
    ]
    part_room_ids: dict[str, set[str]] = {
        str(node["id"]): set(str(room_id) for room_id in (node.get("room_ids") or []))
        for node in part_nodes
        if node.get("id")
    }

    for cell in roof_cell_complex.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        part_id = cell.get("part_id")
        room_id = cell.get("room_id")
        if not part_id and room_id:
            unassigned_room_ids.add(str(room_id))

    for knee_wall in roof_cell_complex.get("knee_walls") or []:
        if not isinstance(knee_wall, dict):
            continue
        part_id = knee_wall.get("part_id")
        room_index = knee_wall.get("room_index")
        if not part_id and isinstance(room_index, int):
            unassigned_room_ids.add(_room_key(room_index))

    coverage_part_ids: dict[str, set[str]] = {}
    atom_subpart_membership = roof_coverage_graph.get("atom_subpart_membership") or {}
    room_membership = building_part_graph.get("room_membership") or {}
    for subpart in roof_coverage_graph.get("subparts") or []:
        if not isinstance(subpart, dict):
            continue
        subpart_id = str(subpart.get("id") or "")
        part_ids = {str(value) for value in (subpart.get("part_ids") or []) if value}
        if not part_ids:
            room_indices = [
                int(v)
                for v in (subpart.get("room_indices") or [])
                if isinstance(v, int)
            ]
            for room_index in room_indices:
                part_ids.update(
                    str(part_id)
                    for part_id in (room_membership.get(_room_key(room_index)) or [])
                )
            if not part_ids:
                for atom_id, subpart_ids in atom_subpart_membership.items():
                    if subpart_id not in [str(value) for value in (subpart_ids or [])]:
                        continue
                    atom = atoms_by_id.get(str(atom_id))
                    if atom is None:
                        continue
                    part_ids.add(
                        str(atom.get("effective_part_id") or UNASSIGNED_PART_ID)
                    )
        if not part_ids:
            part_ids = {UNASSIGNED_PART_ID}
        coverage_part_ids[subpart_id] = part_ids

    building_parts: list[dict[str, Any]] = []
    part_graph_room_ids: dict[str, set[str]] = {}
    for node in part_nodes:
        part_id = str(node["id"])
        room_ids = part_room_ids.get(part_id, set())
        part_graph_room_ids[part_id] = {
            graph_room_by_room_id[room_id]
            for room_id in room_ids
            if graph_room_by_room_id.get(room_id)
        }
        polys: list[Polygon] = []
        top_values: list[float] = []
        for room_id in sorted(room_ids):
            polys.extend(room_partition_polys.get(room_id, []))
            top_values.extend(room_partition_tops.get(room_id, []))
        union_poly = _largest_polygon(unary_union(polys)) if polys else None
        avg_top_y = sum(top_values) / len(top_values) if top_values else 0.0
        polygon_xz = _serialize_poly_xz(union_poly)
        building_parts.append(
            {
                **node,
                "effective_part_id": part_id,
                "room_indices": _room_indices_for_ids(
                    room_ids, room_indices_by_room_id
                ),
                "polygon_xz": polygon_xz,
                "polygon": _polygon_xz_to_3d(polygon_xz, avg_top_y)
                if polygon_xz
                else [],
                "bbox_xz": _bbox_xz(union_poly),
                "centroid_xz": _centroid_xz(union_poly),
                "area_m2": _round6(
                    union_poly.area if isinstance(union_poly, Polygon) else 0.0
                ),
                "top_y_m": _round6(avg_top_y) if top_values else None,
            }
        )

    if unassigned_room_ids:
        polys: list[Polygon] = []
        top_values: list[float] = []
        for room_id in sorted(unassigned_room_ids):
            polys.extend(room_partition_polys.get(room_id, []))
            top_values.extend(room_partition_tops.get(room_id, []))
        union_poly = _largest_polygon(unary_union(polys)) if polys else None
        avg_top_y = sum(top_values) / len(top_values) if top_values else 0.0
        polygon_xz = _serialize_poly_xz(union_poly)
        part_graph_room_ids[UNASSIGNED_PART_ID] = {
            graph_room_by_room_id[room_id]
            for room_id in unassigned_room_ids
            if graph_room_by_room_id.get(room_id)
        }
        building_parts.append(
            {
                "id": UNASSIGNED_PART_ID,
                "type": "BuildingPart",
                "effective_part_id": UNASSIGNED_PART_ID,
                "room_ids": sorted(unassigned_room_ids),
                "room_indices": _room_indices_for_ids(
                    unassigned_room_ids, room_indices_by_room_id
                ),
                "hypothesis_ids": [],
                "oblique_hypothesis_ids": [],
                "flat_hypothesis_ids": [],
                "articulation_room_ids": [],
                "roof_family_guess": "unassigned",
                "polygon_xz": polygon_xz,
                "polygon": _polygon_xz_to_3d(polygon_xz, avg_top_y)
                if polygon_xz
                else [],
                "bbox_xz": _bbox_xz(union_poly),
                "centroid_xz": _centroid_xz(union_poly),
                "area_m2": _round6(
                    union_poly.area if isinstance(union_poly, Polygon) else 0.0
                ),
                "top_y_m": _round6(avg_top_y) if top_values else None,
                "synthetic": True,
            }
        )

    if building is not None:
        all_room_indices = list(range(len(building.get("rooms") or [])))
        all_room_ids = {_room_key(room_index) for room_index in all_room_indices}
        all_room_polys = [
            poly
            for room_index in all_room_indices
            for poly in [
                _poly_xz_from_3d(
                    (building.get("rooms") or [])[room_index].get("floor_polygon") or []
                )
            ]
            if poly is not None
        ]
        full_union = (
            _largest_polygon(unary_union(all_room_polys)) if all_room_polys else None
        )
        polygon_xz = _serialize_poly_xz(full_union)
        all_top_values = [
            float(value)
            for values in room_partition_tops.values()
            for value in values
            if isinstance(value, (int, float))
        ]
        avg_top_y = sum(all_top_values) / len(all_top_values) if all_top_values else 0.0
        part_graph_room_ids[FULL_BUILDING_PART_ID] = set(
            part_graph_room_ids.get(FULL_BUILDING_PART_ID) or set()
        )
        part_graph_room_ids[FULL_BUILDING_PART_ID].update(
            {
                source_id
                for cell in (topology_cell_complex.get("cells") or [])
                if isinstance(cell, dict)
                and str(cell.get("kind") or "") == "room"
                and isinstance((source_id := cell.get("source_id")), str)
                and source_id
            }
        )
        building_parts.append(
            {
                "id": FULL_BUILDING_PART_ID,
                "type": "BuildingPart",
                "effective_part_id": FULL_BUILDING_PART_ID,
                "room_ids": sorted(all_room_ids),
                "room_indices": all_room_indices,
                "hypothesis_ids": [],
                "oblique_hypothesis_ids": [],
                "flat_hypothesis_ids": [],
                "articulation_room_ids": [],
                "roof_family_guess": "full_building",
                "polygon_xz": polygon_xz,
                "polygon": _polygon_xz_to_3d(polygon_xz, avg_top_y)
                if polygon_xz
                else [],
                "bbox_xz": _bbox_xz(full_union),
                "centroid_xz": _centroid_xz(full_union),
                "area_m2": _round6(
                    full_union.area if isinstance(full_union, Polygon) else 0.0
                ),
                "top_y_m": _round6(avg_top_y) if all_top_values else None,
                "synthetic": True,
                "synthetic_role": "full_building",
            }
        )

    building_parts.sort(
        key=lambda part: (
            0 if part.get("synthetic_role") == "full_building" else 1,
            1 if part.get("synthetic") else 0,
            -float(part.get("area_m2", 0.0) or 0.0),
            str(part.get("id") or ""),
        )
    )

    coverage_subparts: list[dict[str, Any]] = []
    atom_tops_by_subpart: dict[str, list[float]] = defaultdict(list)
    for atom_id, subpart_ids in atom_subpart_membership.items():
        atom = atoms_by_id.get(str(atom_id))
        if atom is None:
            continue
        top_y = atom.get("top_y_m")
        if not isinstance(top_y, (int, float)):
            continue
        for subpart_id in subpart_ids or []:
            atom_tops_by_subpart[str(subpart_id)].append(float(top_y))
    for subpart in roof_coverage_graph.get("subparts") or []:
        if not isinstance(subpart, dict):
            continue
        subpart_id = str(subpart.get("id") or "")
        polygon_xz = [
            [_round6(point[0]), _round6(point[1])]
            for point in (subpart.get("polygon_xz") or [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        top_values = list(atom_tops_by_subpart.get(subpart_id, []))
        if not top_values:
            for room_index in [
                int(v)
                for v in (subpart.get("room_indices") or [])
                if isinstance(v, int)
            ]:
                top_values.extend(room_partition_tops.get(_room_key(room_index), []))
        top_y = sum(top_values) / len(top_values) if top_values else 0.0
        coverage_subparts.append(
            {
                **subpart,
                "effective_part_ids": sorted(
                    coverage_part_ids.get(subpart_id, {UNASSIGNED_PART_ID})
                ),
                "polygon": _polygon_xz_to_3d(polygon_xz, top_y) if polygon_xz else [],
                "top_y_m": _round6(top_y) if top_values else None,
            }
        )

    subparts_by_hypothesis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for subpart in coverage_subparts:
        hypothesis_id = str(subpart.get("roof_hypothesis_id") or "")
        if hypothesis_id:
            subparts_by_hypothesis[hypothesis_id].append(subpart)

    oblique_coverage_patches: list[dict[str, Any]] = []
    hypothesis_membership = building_part_graph.get("hypothesis_membership") or {}
    for index, surface in enumerate(roof_surfaces.get("oblique") or []):
        if not isinstance(surface, dict):
            continue
        hypothesis_id = str(
            surface.get("roof_hypothesis_id") or f"roof-hypothesis:oblique:{index}"
        )
        surface_poly = _poly_xz_from_3d(surface.get("corners") or [])
        if surface_poly is None:
            continue
        matching_subparts = subparts_by_hypothesis.get(hypothesis_id) or []
        if not matching_subparts:
            continue
        for subpart in matching_subparts:
            subpart_poly = Polygon(subpart.get("polygon_xz") or [])
            if subpart_poly.is_empty:
                continue
            try:
                clipped = surface_poly.intersection(subpart_poly)
            except Exception:
                continue
            for piece_index, piece in enumerate(_decompose_polygons(clipped)):
                if piece.is_empty or piece.area <= 1e-6:
                    continue
                effective_part_ids = list(subpart.get("effective_part_ids") or [])
                if not effective_part_ids:
                    effective_part_ids = [
                        str(part_id)
                        for part_id in (hypothesis_membership.get(hypothesis_id) or [])
                    ] or [UNASSIGNED_PART_ID]
                room_indices = [
                    int(value)
                    for value in (subpart.get("room_indices") or [])
                    if isinstance(value, int)
                ]
                oblique_coverage_patches.append(
                    {
                        "id": (
                            f"roof-coverage-patch:{hypothesis_id}:"
                            f"{subpart.get('id')}:{piece_index}"
                        ),
                        "roof_hypothesis_id": hypothesis_id,
                        "coverage_subpart_id": subpart.get("id"),
                        "effective_part_ids": effective_part_ids,
                        "room_indices": room_indices,
                        "room_ids": [
                            _room_key(room_index) for room_index in room_indices
                        ],
                        "polygon": _lift_poly_on_surface(piece, surface),
                        "polygon_xz": _serialize_poly_xz(piece),
                        "surface_kind": "oblique",
                        "story": surface.get("story", surface.get("dominant_story")),
                        "coverage_semantic_kind": subpart.get("semantic_kind"),
                        "continuation_source": surface.get("continuation_source"),
                    }
                )

    unresolved_regions: list[dict[str, Any]] = []
    room_summaries = top_boundary_graph.get("room_summaries") or {}
    for room_id, room_summary in room_summaries.items():
        if not isinstance(room_summary, dict):
            continue
        has_resolved = bool(room_summary.get("has_resolved_roof_relation"))
        if not has_resolved:
            has_resolved = bool(
                room_summary.get("has_attic_relation")
                or room_summary.get("has_upper_void_relation")
                or room_summary.get("has_oblique_atom")
            )
        should_be_covered = bool(
            room_summary.get("partially_covered_by_sloped_roof")
            or room_summary.get("strong_perimeter_sloped")
            or room_summary.get("strong_knee_wall_signal")
            or room_summary.get("has_candidate_attic_relation")
            or room_summary.get("has_candidate_upper_void_relation")
            or int(room_summary.get("roof_evidence_score", 0) or 0) >= 4
        )
        if has_resolved or not should_be_covered:
            continue
        polys = room_partition_polys.get(str(room_id), [])
        union_poly = _largest_polygon(unary_union(polys)) if polys else None
        if union_poly is None:
            continue
        polygon_xz = _serialize_poly_xz(union_poly)
        top_values = room_partition_tops.get(str(room_id), [])
        top_y = sum(top_values) / len(top_values) if top_values else 0.0
        unresolved_regions.append(
            {
                "id": f"unresolved-coverage:{room_id}",
                "room_id": room_id,
                "room_index": room_summary.get("room_index"),
                "story": room_summary.get("story"),
                "effective_part_ids": list(
                    room_summary.get("part_ids") or [UNASSIGNED_PART_ID]
                ),
                "polygon": _polygon_xz_to_3d(polygon_xz, top_y) if polygon_xz else [],
                "polygon_xz": polygon_xz,
                "slant_delta_m": room_summary.get("slant_delta_m"),
                "roof_evidence_score": room_summary.get("roof_evidence_score"),
                "has_candidate_attic_relation": bool(
                    room_summary.get("has_candidate_attic_relation")
                ),
                "has_candidate_upper_void_relation": bool(
                    room_summary.get("has_candidate_upper_void_relation")
                ),
            }
        )

    for dormer in dormers:
        if not isinstance(dormer, dict):
            continue
        room_index = dormer.get("room_index")
        room_id = _room_key(room_index) if isinstance(room_index, int) else None
        effective_part_ids = []
        if room_id is not None:
            effective_part_ids = [
                str(part_id) for part_id in (room_membership.get(room_id) or [])
            ] or ([UNASSIGNED_PART_ID] if room_id in unassigned_room_ids else [])
        dormer["effective_part_ids"] = effective_part_ids

    continuation_regions: list[dict[str, Any]] = []
    for region in roof_continuation_diagnostics.get("continuation_regions") or []:
        if not isinstance(region, dict):
            continue
        room_id = str(region.get("room_id") or "")
        room_index = region.get("room_index")
        effective_part_ids = [
            str(part_id) for part_id in (room_membership.get(room_id) or [])
        ]
        if not effective_part_ids and isinstance(room_index, int):
            effective_part_ids = [
                str(part_id)
                for part_id in (room_membership.get(_room_key(room_index)) or [])
            ]
        if not effective_part_ids:
            effective_part_ids = [UNASSIGNED_PART_ID]
        continuation_regions.append(
            {
                **region,
                "effective_part_ids": effective_part_ids,
            }
        )

    rooms_with_per_room_unresolved = {
        str(region.get("room_id"))
        for region in unresolved_regions
        if region.get("room_id")
    }
    renderable_surfaces = [
        surface
        for surface in (
            _renderable_surface_from_atom(
                atom,
                rooms_with_per_room_unresolved=rooms_with_per_room_unresolved,
            )
            for atom in semantic_atoms
        )
        if surface is not None
    ]
    renderable_surfaces.extend(
        surface
        for surface in (
            _renderable_surface_from_unresolved_region(region)
            for region in unresolved_regions
        )
        if surface is not None
    )
    renderable_surface_counts = _surface_category_counts(renderable_surfaces)

    summary = {
        "uuid": uuid,
        "view": "summary",
        "building_parts": building_parts,
        "coverage_subparts": coverage_subparts,
        "oblique_coverage_patches": oblique_coverage_patches,
        "roof_continuation_diagnostics": continuation_regions,
        "semantic_atoms": semantic_atoms,
        "unresolved_regions": unresolved_regions,
        "renderable_surfaces": renderable_surfaces,
        "room_summaries": top_boundary_graph.get("room_summaries") or {},
        "building_part_graph": building_part_graph,
        "roof_coverage_metadata": roof_coverage_graph.get("metadata") or {},
        "top_boundary_metadata": top_boundary_graph.get("metadata") or {},
        "roof_evidence_metadata": roof_evidence_graph.get("metadata") or {},
        "dormers": dormers,
        "metadata": {
            "building_part_count": len(building_parts),
            "semantic_atom_count": len(semantic_atoms),
            "coverage_subpart_count": len(coverage_subparts),
            "oblique_coverage_patch_count": len(oblique_coverage_patches),
            "roof_continuation_region_count": len(continuation_regions),
            "unresolved_region_count": len(unresolved_regions),
            "renderable_surface_count": len(renderable_surfaces),
            "renderable_surface_counts": renderable_surface_counts,
            "dormer_count": len(dormers),
            "topology_room_cell_count": len(
                [
                    cell
                    for cell in (topology_cell_complex.get("cells") or [])
                    if cell.get("kind") == "room"
                ]
            ),
            "roof_exact_cell_count": len(roof_cell_complex.get("cells") or []),
            "occupied_room_cell_count": len(
                (roof.get("occupied_room_cell_complex") or {}).get("cells") or []
            ),
            "knee_wall_count": len(roof_cell_complex.get("knee_walls") or []),
        },
    }
    return summary, part_graph_room_ids


def _build_ontology_part_payloads(
    *,
    uuid: str,
    summary: dict[str, Any],
    part_graph_room_ids: dict[str, set[str]],
    topology_cell_complex: dict[str, Any],
    roof_cell_complex: dict[str, Any],
    occupied_room_cell_complex: dict[str, Any] | None = None,
    building: dict[str, Any] | None = None,
    roof: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    part_details: dict[str, dict[str, Any]] = {}
    part_summaries = {
        str(part["id"]): part
        for part in (summary.get("building_parts") or [])
        if isinstance(part, dict) and part.get("id")
    }
    dormers = summary.get("dormers") or []
    topology_room_cells = [
        cell
        for cell in (topology_cell_complex.get("cells") or [])
        if isinstance(cell, dict) and str(cell.get("kind") or "") == "room"
    ]
    topology_room_unions = _topology_story_unions(topology_room_cells)
    room_to_part_id: dict[str, str] = {}
    part_ids_by_room_index: dict[int, str] = {}
    for mapped_part_id, room_ids in part_graph_room_ids.items():
        for room_id in room_ids:
            room_to_part_id.setdefault(str(room_id), str(mapped_part_id))
    top_graph_room_ids = set(room_to_part_id.keys())
    for part in summary.get("building_parts") or []:
        if not isinstance(part, dict):
            continue
        if str(part.get("synthetic_role") or "") == "full_building":
            continue
        part_id = str(part.get("id") or "")
        if not part_id:
            continue
        for room_index in part.get("room_indices") or []:
            if isinstance(room_index, int):
                part_ids_by_room_index.setdefault(room_index, part_id)
    for cell in topology_room_cells:
        source_id = str(cell.get("source_id") or "")
        if source_id in room_to_part_id:
            continue
        room_index = _parse_topology_room_index(source_id)
        if room_index is None:
            continue
        fallback_part_id = part_ids_by_room_index.get(room_index)
        if fallback_part_id:
            room_to_part_id[source_id] = fallback_part_id

    for part_id, part in part_summaries.items():
        room_indices = {
            int(value)
            for value in (part.get("room_indices") or [])
            if isinstance(value, int)
        }
        include_all_rooms = str(part.get("synthetic_role") or "") == "full_building"
        unresolved_regions = _part_unresolved_regions(
            summary=summary,
            part_id=part_id,
            room_indices=room_indices,
            include_all_rooms=include_all_rooms,
        )
        roof_cells = [
            cell
            for cell in (roof_cell_complex.get("cells") or [])
            if isinstance(cell, dict)
            and (
                include_all_rooms
                or (
                    str(cell.get("part_id") or "") == part_id
                    or cell.get("room_id") in set(part.get("room_ids") or [])
                    or (part_id == UNASSIGNED_PART_ID and not cell.get("part_id"))
                )
            )
        ]
        filtered_roof_cells: list[dict[str, Any]] = []
        renderable_surfaces: list[dict[str, Any]] = []
        exact_roof_room_indices: set[int] = set()
        exact_roof_hypothesis_ids: set[str] = set()
        topology_cells = [
            cell
            for cell in topology_room_cells
            if (
                str(
                    room_to_part_id.get(
                        str(cell.get("source_id") or ""), UNASSIGNED_PART_ID
                    )
                )
                == part_id
                or (
                    part_id == UNASSIGNED_PART_ID
                    and str(cell.get("source_id") or "") not in room_to_part_id
                )
            )
        ]
        for cell in roof_cells:
            viewer_faces = []
            for face in cell.get("faces") or []:
                if not isinstance(face, dict):
                    continue
                role = str(face.get("role") or "")
                perimeter_facing = bool(
                    (face.get("metadata") or {}).get("perimeter_facing")
                )
                if role in {"roof", "slab"}:
                    viewer_faces.append(face)
                elif role == "wall" and perimeter_facing:
                    viewer_faces.append(face)
                renderable = _renderable_surface_from_roof_face(face, cell, part_id)
                if renderable is not None:
                    renderable_surfaces.append(renderable)
                    room_index = cell.get("room_index")
                    if isinstance(room_index, int):
                        exact_roof_room_indices.add(room_index)
                    roof_hypothesis_id = str(cell.get("roof_hypothesis_id") or "")
                    if roof_hypothesis_id:
                        exact_roof_hypothesis_ids.add(roof_hypothesis_id)
            if not viewer_faces:
                continue
            filtered = dict(cell)
            filtered["faces"] = viewer_faces
            filtered_roof_cells.append(filtered)
        knee_walls = [
            wall
            for wall in (roof_cell_complex.get("knee_walls") or [])
            if isinstance(wall, dict)
            and (
                include_all_rooms
                or (
                    str(wall.get("part_id") or "") == part_id
                    or (
                        isinstance(wall.get("room_index"), int)
                        and wall.get("room_index") in room_indices
                    )
                    or (part_id == UNASSIGNED_PART_ID and not wall.get("part_id"))
                )
            )
        ]
        renderable_surfaces.extend(
            surface
            for surface in (
                _renderable_surface_from_knee_wall(wall, part_id) for wall in knee_walls
            )
            if surface is not None
        )
        ceiling_atom_roles = {
            "sloped_ceiling",
            "flat_ceiling",
            "flat_transition_cap",
            "flat_transition_cap_inferred",
        }
        atoms_covering_ceiling = {
            str(atom.get("id"))
            for atom in (summary.get("semantic_atoms") or [])
            if isinstance(atom, dict)
            and atom.get("id")
            and str(atom.get("role") or "") in ceiling_atom_roles
        }
        occupied_renderable_surfaces, occupied_unresolved_regions = (
            _renderable_surfaces_from_occupied_room_cells(
                occupied_room_cell_complex=occupied_room_cell_complex,
                building=building,
                room_indices=room_indices,
                part_id=part_id,
                atoms_covering_ceiling=atoms_covering_ceiling,
            )
        )
        if occupied_renderable_surfaces:
            renderable_surfaces.extend(occupied_renderable_surfaces)
        else:
            renderable_surfaces.extend(
                _renderable_base_room_surfaces(
                    building=building,
                    roof=roof,
                    room_indices=room_indices,
                    primary_part_id_by_room_index=part_ids_by_room_index,
                    part_id=part_id,
                )
            )
        unresolved_regions.extend(occupied_unresolved_regions)
        rooms_with_per_room_unresolved = {
            str(region.get("room_id"))
            for region in unresolved_regions
            if region.get("room_id")
        }
        part_semantic_atoms = _part_semantic_atoms(
            summary=summary,
            part_id=part_id,
            room_indices=room_indices,
            include_all_rooms=include_all_rooms,
        )
        renderable_surfaces.extend(
            surface
            for surface in (
                _renderable_surface_from_atom(
                    atom,
                    rooms_with_per_room_unresolved=rooms_with_per_room_unresolved,
                )
                for atom in part_semantic_atoms
            )
            if surface is not None
        )
        (
            fallback_roof_surfaces,
            fallback_unresolved_regions,
            exact_flat_surface_count,
            coverage_patch_surface_count,
        ) = _roof_surface_fallback_payload(
            roof=roof,
            summary=summary,
            part_id=part_id,
            room_indices=room_indices,
            exact_roof_room_indices=exact_roof_room_indices,
            exact_roof_hypothesis_ids=exact_roof_hypothesis_ids,
            include_all_rooms=include_all_rooms,
            atoms_for_subtraction=part_semantic_atoms,
        )
        renderable_surfaces.extend(fallback_roof_surfaces)
        unresolved_regions.extend(fallback_unresolved_regions)
        roof_fallback_surface_count = sum(
            1
            for surface in fallback_roof_surfaces
            if str(surface.get("source_kind") or "") == "roof_surface_fallback"
        )
        if building is None or roof is None:
            for cell in topology_cells:
                story_union = (
                    topology_room_unions.get(cell.get("story"))
                    if isinstance(cell.get("story"), int)
                    else None
                )
                include_ceiling = (
                    str(cell.get("source_id") or "") not in top_graph_room_ids
                )
                for face in cell.get("faces") or []:
                    if not isinstance(face, dict):
                        continue
                    renderable = _renderable_surface_from_topology_face(
                        face,
                        cell,
                        part_id=part_id,
                        is_exterior_wall=_is_exterior_wall_face(face, story_union),
                        include_ceiling=include_ceiling,
                    )
                    if renderable is not None:
                        renderable_surfaces.append(renderable)
        unresolved_renderable_surfaces = [
            surface
            for surface in (
                _renderable_surface_from_unresolved_region(region)
                for region in unresolved_regions
            )
            if surface is not None
        ]
        renderable_surfaces.extend(unresolved_renderable_surfaces)
        renderable_surfaces = _dedupe_renderable_surfaces(renderable_surfaces)
        renderable_surfaces = _resolve_full_model_surface_overlaps(renderable_surfaces)
        renderable_surface_counts = _surface_category_counts(renderable_surfaces)
        dormer_subset = _filter_part_dormers(dormers, room_indices)
        part_details[part_id] = {
            "uuid": uuid,
            "view": "part",
            "part_id": part_id,
            "part_summary": part,
            "roof_cells": filtered_roof_cells,
            "knee_walls": knee_walls,
            "unresolved_regions": unresolved_regions,
            "renderable_surfaces": renderable_surfaces,
            "dormers": dormer_subset,
            "metadata": {
                "roof_cell_count": len(filtered_roof_cells),
                "attic_cell_count": sum(
                    1
                    for cell in filtered_roof_cells
                    if cell.get("cell_kind") == "attic"
                ),
                "upper_void_cell_count": sum(
                    1
                    for cell in filtered_roof_cells
                    if cell.get("cell_kind") == "upper_void"
                ),
                "knee_wall_count": len(knee_walls),
                "occupied_room_cell_count": sum(
                    1
                    for cell in ((occupied_room_cell_complex or {}).get("cells") or [])
                    if isinstance(cell, dict)
                    and (
                        include_all_rooms
                        or (
                            isinstance(cell.get("room_index"), int)
                            and cell.get("room_index") in room_indices
                        )
                    )
                ),
                "roof_exact_flat_surface_count": exact_flat_surface_count,
                "roof_coverage_patch_surface_count": coverage_patch_surface_count,
                "roof_fallback_surface_count": roof_fallback_surface_count,
                "unresolved_region_count": len(unresolved_regions),
                "renderable_surface_count": len(renderable_surfaces),
                "renderable_surface_counts": renderable_surface_counts,
                "dormer_count": len(dormer_subset),
            },
        }
    return part_details


def _build_ontology_cache_entry(uuid: str) -> dict[str, Any]:
    merged_path = PIPELINE_ROOT / uuid / "merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"No merged.json for {uuid}")
    graph = build_topology_graph(
        merged_path=merged_path,
        scan_dir=SCAN_CACHE_ROOT,
        uuid=uuid,
    )
    building = extract_building(
        uuid=uuid,
        pipeline_dir=PIPELINE_ROOT,
        scan_cache_root=SCAN_CACHE_ROOT,
        load_topology_graph=False,
    )
    if not building:
        raise RuntimeError(f"extract_building returned no building for {uuid}")
    roof = run_roof_algorithms(building, graph=graph)
    topology_cell_complex = (graph.geometry_index or {}).get("cell_complex") or {}
    summary, part_graph_room_ids = _build_ontology_summary(
        uuid=uuid,
        roof=roof,
        topology_cell_complex=topology_cell_complex,
        building=building,
    )
    parts = _build_ontology_part_payloads(
        uuid=uuid,
        summary=summary,
        part_graph_room_ids=part_graph_room_ids,
        topology_cell_complex=topology_cell_complex,
        roof_cell_complex=roof.get("roof_cell_complex") or {},
        occupied_room_cell_complex=roof.get("occupied_room_cell_complex") or {},
        building=building,
        roof=roof,
    )
    return {
        "summary": summary,
        "parts": parts,
        "full_model": parts.get(FULL_BUILDING_PART_ID),
    }


def resolve_datafordeleren_api_key():
    env_key = os.environ.get("DATAFORDELEREN_API_KEY")
    if env_key:
        return env_key

    # Fallback: fetch from Secret Manager using local gcloud auth.
    try:
        out = subprocess.check_output(
            [
                "gcloud",
                "secrets",
                "versions",
                "access",
                "latest",
                f"--secret={SECRET_NAME}",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        ).strip()
        return out or None
    except Exception:
        return None


_SAFE_UUID_CHARS = set("0123456789abcdefABCDEF-")


def _is_safe_uuid(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if not (8 <= len(value) <= 64):
        return False
    return all(c in _SAFE_UUID_CHARS for c in value)


def _split_locator(locator: str) -> tuple[str, list[str]]:
    segments = locator.split("::", 2)
    if len(segments) != 3:
        return "", []
    kind = segments[1]
    parts = segments[2].split(":") if segments[2] else []
    return kind, parts


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        if directory is None:
            # Serve from workspace root so both the legacy V1 viewer at
            # /reconcile/viewer.html and the V2 tier viewer at
            # /reconcile_tiers/web/viewer-tiers.html resolve, alongside
            # /pipeline-outputs/<uuid>/* static data.
            directory = str(WORKSPACE_ROOT)
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        if parsed_path == "/" or parsed_path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/ortofoto":
            self._handle_ortofoto(parsed.query)
            return
        if parsed.path == "/ontology-artifacts":
            self._handle_ontology_artifacts(parsed.query)
            return
        if parsed.path == "/v3":
            self._handle_v3(parsed.query)
            return
        if parsed.path == "/candidate-faces":
            self._handle_candidate_faces(parsed.query)
            return
        if parsed.path == "/ridge-eave-scores":
            self._handle_ridge_eave_scores(parsed.query)
            return
        if parsed.path == "/raw-ceiling-prototype":
            self._handle_raw_ceiling_prototype()
            return
        if parsed.path == "/computed-overextend":
            self._handle_computed_overextend()
            return
        if parsed.path == "/raw-disagreement":
            self._handle_raw_disagreement()
            return
        if parsed.path == "/raw-ceiling-plane-splits":
            self._handle_raw_ceiling_plane_splits(parsed.query)
            return
        if parsed.path == "/ceiling-replacement":
            self._handle_ceiling_replacement()
            return
        if parsed.path == "/reconstruction":
            self._handle_reconstruction(parsed.query)
            return
        if parsed.path == "/alignment-calibration":
            self._handle_calibration_get()
            return
        if parsed.path == "/roof-rating":
            self._handle_roof_rating_get()
            return
        if parsed.path == "/v3-roof-proposal-labels":
            self._handle_roof_proposal_labels_get(parsed.query)
            return
        if parsed.path == "/v3-roof-proposal-splits":
            self._handle_roof_proposal_splits_get(parsed.query)
            return
        if parsed.path == "/v3-roof-proposal-queue":
            self._handle_roof_proposal_queue_get()
            return
        if parsed.path == "/roof-index":
            self._handle_roof_index()
            return
        if parsed.path == "/roof-detail":
            self._handle_roof_detail(parsed.query)
            return
        if parsed.path == "/tier-index":
            self._handle_tier_index()
            return
        if parsed.path == "/building-merged":
            self._handle_building_merged(parsed.query)
            return
        if parsed.path == "/flag-queues":
            self._handle_flag_queue_get(parsed.query)
            return
        if parsed.path == "/tier-payload":
            self._handle_tier_payload_get(parsed.query)
            return
        if parsed.path == "/energy-estimate":
            self._handle_energy_estimate_get(parsed.query)
            return

        if parsed.path in ("/", "/viewer-tiers.html"):
            # 302 redirect (not internal rewrite) so the browser's base URL
            # updates and relative asset paths in viewer-tiers.html resolve.
            target = "/reconcile_tiers/web/viewer-tiers.html"
            if self.path != parsed.path:  # preserve query + hash
                target += self.path[len(parsed.path) :]
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", target)
            self.end_headers()
            return
        if parsed.path == "/viewer.html":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/reconcile/viewer.html")
            self.end_headers()
            return
        if parsed.path == "/calm-viewer":
            target = "/reconcile_tiers/web/viewer-calm.html"
            if self.path != parsed.path:
                target += self.path[len(parsed.path) :]
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", target)
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/alignment-calibration":
            self._handle_calibration_post()
            return
        if parsed.path == "/v3-roof-proposal-label":
            self._handle_roof_proposal_label_post()
            return
        if parsed.path == "/v3-roof-proposal-split":
            self._handle_roof_proposal_split_post()
            return
        if parsed.path == "/flag-queue":
            self._handle_flag_queue_post()
            return
        if parsed.path == "/roof-rating":
            self._handle_roof_rating_post()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _handle_ortofoto(self, query: str):
        params = urllib.parse.parse_qs(query)
        z = (params.get("z") or [None])[0]
        x = (params.get("x") or [None])[0]
        y = (params.get("y") or [None])[0]
        if not z or not x or not y:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing z/x/y query params")
            return

        # Prefer server-side key from env; allow query fallback for local testing.
        api_key = (
            resolve_datafordeleren_api_key() or (params.get("apiKey") or [None])[0]
        )
        if not api_key:
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                "Missing Datafordeleren key "
                "(set DATAFORDELEREN_API_KEY or "
                "provide apiKey query param)",
            )
            return

        wmts_url = urllib.parse.urlencode(
            {
                "apikey": api_key,
                "SERVICE": "WMTS",
                "REQUEST": "GetTile",
                "VERSION": "1.0.0",
                "STYLE": "default",
                "FORMAT": "image/jpeg",
                "TILEMATRIXSET": "DFD_GoogleMapsCompatible",
                "TILEMATRIX": z,
                "TILEROW": y,
                "TILECOL": x,
                "Layer": "orto_foraar_webm",
            }
        )
        url = f"{WMTS_BASE}?{wmts_url}"

        request = urllib.request.Request(
            url, headers={"User-Agent": "tirana-viewer/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as resp:
                body = resp.read()
                content_type = resp.headers.get("Content-Type", "image/jpeg")
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_GATEWAY, f"Ortofoto upstream error: {exc}")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "public, max-age=86400")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_calibration(self):
        if not CALIBRATION_PATH.exists():
            return {}
        try:
            with open(CALIBRATION_PATH) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_calibration(self, data):
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CALIBRATION_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        tmp.replace(CALIBRATION_PATH)

    def _handle_calibration_get(self):
        payload = self._read_calibration()
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_calibration_post(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return

        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Body must be an object")
            return
        uuid = payload.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            self.send_error(HTTPStatus.BAD_REQUEST, "uuid is required")
            return

        try:
            record = {
                "rotation_deg": float(payload.get("rotation_deg", 0.0)),
                "offset_east_m": float(payload.get("offset_east_m", 0.0)),
                "offset_north_m": float(payload.get("offset_north_m", 0.0)),
            }
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid numeric values")
            return

        all_data = self._read_calibration()
        all_data[uuid] = record
        self._write_calibration(all_data)

        out = json.dumps({"ok": True, "uuid": uuid, "record": record}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _read_roof_ratings(self) -> dict:
        if not ROOF_RATINGS_PATH.exists():
            return {}
        try:
            with open(ROOF_RATINGS_PATH) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_roof_ratings(self, data: dict) -> None:
        ROOF_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = ROOF_RATINGS_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        tmp.replace(ROOF_RATINGS_PATH)

    def _handle_roof_rating_get(self) -> None:
        payload = self._read_roof_ratings()
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_roof_rating_post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Body must be an object")
            return

        uuid = payload.get("uuid")
        if not isinstance(uuid, str) or not _is_safe_uuid(uuid):
            self.send_error(HTTPStatus.BAD_REQUEST, "uuid is required")
            return

        rating_in = payload.get("rating")
        if isinstance(rating_in, bool):
            rating: int | str | None = None
            invalid = True
        elif isinstance(rating_in, int) and 1 <= rating_in <= 5:
            rating = rating_in
            invalid = False
        elif rating_in == "upstream_error":
            rating = "upstream_error"
            invalid = False
        elif rating_in is None:
            rating = None
            invalid = False
        else:
            rating = None
            invalid = True

        if invalid:
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                "rating must be 1..5, 'upstream_error', or null",
            )
            return

        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        all_data = self._read_roof_ratings()
        if rating is None:
            all_data.pop(uuid, None)
            record = None
        else:
            record = {"rating": rating, "updated_at": timestamp}
            all_data[uuid] = record
        self._write_roof_ratings(all_data)

        self._send_json(HTTPStatus.OK, {"ok": True, "uuid": uuid, "record": record})

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_flag_queue_post(self) -> None:
        """Persist a flag queue from the viewer.

        Body: {building_uuid, items[], screenshot_data_url?, source?}.
        Writes .context/flag-queues/<uuid>/<ts>.json (+ optional .png).
        """
        import base64
        import uuid as uuidlib

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Body must be an object")
            return

        building_uuid = payload.get("building_uuid")
        if not isinstance(building_uuid, str) or not _is_safe_uuid(building_uuid):
            self.send_error(HTTPStatus.BAD_REQUEST, "building_uuid is required")
            return

        items_in = payload.get("items")
        if not isinstance(items_in, list) or not items_in:
            self.send_error(HTTPStatus.BAD_REQUEST, "items must be a non-empty list")
            return

        cleaned: list[dict] = []
        for entry in items_in:
            if not isinstance(entry, dict):
                continue
            locator = entry.get("locator")
            if not isinstance(locator, str) or "::" not in locator:
                continue
            kind, parts = _split_locator(locator)
            cleaned.append(
                {
                    "id": entry.get("id") or uuidlib.uuid4().hex[:12],
                    "locator": locator,
                    "kind": entry.get("kind") or kind,
                    "parts": entry.get("parts")
                    if isinstance(entry.get("parts"), list)
                    else parts,
                    "rule": entry.get("rule"),
                    "note": entry.get("note"),
                    "severity": entry.get("severity"),
                    "evidence": entry.get("evidence"),
                    "dismissed": bool(entry.get("dismissed", False)),
                }
            )
        if not cleaned:
            self.send_error(HTTPStatus.BAD_REQUEST, "no valid items in body")
            return

        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        queue_dir = FLAG_QUEUES_ROOT / building_uuid
        queue_dir.mkdir(parents=True, exist_ok=True)
        queue_path = queue_dir / f"{timestamp}.json"

        screenshot_rel: str | None = None
        screenshot_data_url = payload.get("screenshot_data_url")
        if isinstance(screenshot_data_url, str) and screenshot_data_url.startswith(
            "data:image/"
        ):
            try:
                _, b64 = screenshot_data_url.split(",", 1)
                png_bytes = base64.b64decode(b64)
                shot_path = queue_dir / f"{timestamp}.png"
                shot_path.write_bytes(png_bytes)
                screenshot_rel = shot_path.name
            except Exception:
                screenshot_rel = None

        source = payload.get("source")
        if source not in ("viewer", "merged"):
            source = "viewer"

        queue = {
            "schema": "flag-queue/v1",
            "building_uuid": building_uuid,
            "created": timestamp,
            "source": source,
            "screenshot": screenshot_rel,
            "items": cleaned,
        }
        queue_path.write_text(json.dumps(queue, indent=2))

        # Fold dismissed auto-flags into the persistent calibration so future
        # cohort scans suppress them by default.
        from reconcile_tiers.audit import calibration as calib

        calibration = calib.load(building_uuid, FLAG_CALIBRATION_ROOT)
        prior_dismiss_count = len(calibration.get("dismissals") or [])
        calib.merge_dismissals(calibration, cleaned, timestamp=timestamp)
        new_dismiss_count = len(calibration.get("dismissals") or [])
        if new_dismiss_count != prior_dismiss_count or any(
            it.get("dismissed") and it.get("rule") for it in cleaned
        ):
            calib.save(building_uuid, FLAG_CALIBRATION_ROOT, calibration)

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "queue_path": str(queue_path.relative_to(WORKSPACE_ROOT)),
                "item_count": len(cleaned),
                "dismissals_recorded": new_dismiss_count - prior_dismiss_count,
            },
        )

    def _handle_flag_queue_get(self, query: str) -> None:
        """Return the most recent auto-scan-*.json for a building, if any."""
        params = urllib.parse.parse_qs(query)
        building_uuid = (params.get("uuid") or [""])[0]
        if not _is_safe_uuid(building_uuid):
            self.send_error(HTTPStatus.BAD_REQUEST, "uuid query param required")
            return
        latest_name = (
            "auto-scan-latest-scored.json"
            if (params.get("impact") or ["0"])[0] == "1"
            else "auto-scan-latest.json"
        )
        latest = FLAG_QUEUES_ROOT / building_uuid / latest_name
        if not latest.exists():
            self._send_json(HTTPStatus.OK, {"items": [], "source": None})
            return
        try:
            data = json.loads(latest.read_text())
        except Exception:
            self._send_json(HTTPStatus.OK, {"items": [], "source": None})
            return
        self._send_json(HTTPStatus.OK, data)

    def _handle_tier_payload_get(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        building_uuid = (params.get("uuid") or [""])[0]
        if not _is_safe_uuid(building_uuid):
            self.send_error(HTTPStatus.BAD_REQUEST, "uuid query param required")
            return
        path = PIPELINE_ROOT / building_uuid / "tier_payload.json"
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "tier_payload.json not found")
            return
        try:
            self._send_json(HTTPStatus.OK, json.loads(path.read_text()))
        except Exception as exc:
            self.send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to read tier payload: {exc}"
            )

    def _handle_energy_estimate_get(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        building_uuid = (params.get("uuid") or [""])[0]
        if not _is_safe_uuid(building_uuid):
            self.send_error(HTTPStatus.BAD_REQUEST, "uuid query param required")
            return
        path = PIPELINE_ROOT / building_uuid / "energy_estimate.json"
        if not path.exists():
            self._send_json(HTTPStatus.NOT_FOUND, {})
            return
        try:
            self._send_json(HTTPStatus.OK, json.loads(path.read_text()))
        except Exception:
            self._send_json(HTTPStatus.NOT_FOUND, {})

    def _read_roof_proposal_labels_for(self, building_uuid: str) -> dict:
        """Return {proposal_id: label} with last-write-wins for the given building."""
        if not ROOF_PROPOSAL_LABELS_PATH.exists():
            return {}
        latest: dict[str, str] = {}
        with open(ROOF_PROPOSAL_LABELS_PATH) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("building_uuid") != building_uuid:
                    continue
                pid = entry.get("proposal_id")
                lbl = entry.get("label")
                if isinstance(pid, str) and isinstance(lbl, str):
                    latest[pid] = lbl
        return latest

    def _read_all_roof_proposal_labels(self) -> dict:
        """Return {proposal_id: label} across all buildings with last-write-wins."""
        if not ROOF_PROPOSAL_LABELS_PATH.exists():
            return {}
        latest: dict[str, str] = {}
        with open(ROOF_PROPOSAL_LABELS_PATH) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = entry.get("proposal_id")
                lbl = entry.get("label")
                if isinstance(pid, str) and isinstance(lbl, str):
                    latest[pid] = lbl
        return latest

    def _append_roof_proposal_label(self, entry: dict) -> None:
        ROOF_PROPOSAL_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ROOF_PROPOSAL_LABELS_PATH, "a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _handle_roof_proposal_labels_get(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        building_uuid = (params.get("building_uuid") or [None])[0]
        if not building_uuid:
            self.send_error(
                HTTPStatus.BAD_REQUEST, "building_uuid query param required"
            )
            return
        labels = self._read_roof_proposal_labels_for(building_uuid)
        body = json.dumps({"building_uuid": building_uuid, "labels": labels}).encode(
            "utf-8"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_roof_proposal_queue_get(self) -> None:
        global V3_CACHE, V3_CACHE_MTIME
        if not V3_RESULTS_PATH.exists():
            body = json.dumps({"count": 0, "labeled": 0, "proposals": []}).encode(
                "utf-8"
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            mtime = V3_RESULTS_PATH.stat().st_mtime
            if mtime != V3_CACHE_MTIME or not V3_CACHE:
                with open(V3_RESULTS_PATH) as handle:
                    data = json.load(handle)
                V3_CACHE = {
                    entry.get("building_uuid"): entry
                    for entry in data
                    if entry.get("building_uuid")
                }
                V3_CACHE_MTIME = mtime
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_GATEWAY, f"v3 load failed: {exc}")
            return
        labels = self._read_all_roof_proposal_labels()
        _ensure_v3_scores()
        scored = bool(V3_SCORES)
        out = []
        labeled = 0
        auto_accept_n = 0
        auto_reject_n = 0
        for uuid, bldg in V3_CACHE.items():
            address = bldg.get("address", "")
            # Only merged segments are renderable/labelable in the current
            # viewer — raw `roof_proposals[]` are pre-merge and pre-room-split
            # inputs, so iterating them in "Next unlabeled" jumps to items
            # that aren't drawn on screen.
            for seg in bldg.get("merged_roof_segments", []) or []:
                sid = seg.get("id")
                if not isinstance(sid, str):
                    continue
                lbl = labels.get(sid, "unlabeled")
                if lbl != "unlabeled":
                    labeled += 1
                sc = V3_SCORES.get(sid) if scored else None
                auto_label = sc.get("autonomy_label") if sc else None
                if auto_label == "auto_accept":
                    auto_accept_n += 1
                elif auto_label == "auto_reject":
                    auto_reject_n += 1
                out.append(
                    {
                        "building_uuid": uuid,
                        "address": address,
                        "proposal_id": sid,
                        "kind": "v3-merged-roof-segment",
                        "heuristic_label": seg.get("heuristic_label"),
                        "label": lbl,
                        "score": sc.get("score") if sc else None,
                        "autonomy_label": auto_label,
                        "rule_fires": sc.get("rule_fires") if sc else None,
                    }
                )
        # When a scored file is present, prioritize uncertain "review" items
        # at the front of the queue so "Next unlabeled" surfaces the hardest
        # calls first; keep everything else in stable order after that.
        if scored:

            def _rank(p: dict) -> tuple[int, float]:
                al = p.get("autonomy_label")
                # 0: review (uncertain first), 1: auto_accept, 2: auto_reject, 3: no
                # score
                bucket = {"review": 0, "auto_accept": 1, "auto_reject": 2}.get(al, 3)
                s = p.get("score")
                unc = abs((s if s is not None else 0.5) - 0.5)
                return (bucket, unc)

            out.sort(key=_rank)
        body = json.dumps(
            {
                "count": len(out),
                "labeled": labeled,
                "scored": scored,
                "auto_accept": auto_accept_n,
                "auto_reject": auto_reject_n,
                "proposals": out,
            }
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_roof_proposal_label_post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Body must be an object")
            return

        building_uuid = payload.get("building_uuid")
        proposal_id = payload.get("proposal_id")
        label = payload.get("label")
        if (
            not isinstance(building_uuid, str)
            or not building_uuid
            or not isinstance(proposal_id, str)
            or not proposal_id
            or label not in ("accepted", "rejected", "skipped")
        ):
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                "building_uuid, proposal_id, label (accepted|rejected|skipped) "
                "required",
            )
            return

        entry = {
            "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "building_uuid": building_uuid,
            "proposal_id": proposal_id,
            "label": label,
            "reasons": payload.get("reasons") or [],
            "labeler": payload.get("labeler") or None,
            "features_snapshot": payload.get("features_snapshot") or {},
            "heuristic_label": payload.get("heuristic_label"),
            # Merge-mode enrichment — everything a reverse-engineering pass
            # needs to reconstruct the labeled entity (merged plane, all
            # contributing raw proposals, per-member post-clip coords, the
            # opposing planes that clipped the rain-hitting region, and the
            # merge thresholds).
            "merge_mode": bool(payload.get("merge_mode", False)),
            "cluster_canonical_id": payload.get("cluster_canonical_id"),
            "part_index": payload.get("part_index", 0),
            "part_count": payload.get("part_count", 1),
            "merged_plane": payload.get("merged_plane"),
            "cluster_members": payload.get("cluster_members") or [],
            "cluster_params": payload.get("cluster_params"),
            "opposing_cluster_canonicals": payload.get("opposing_cluster_canonicals")
            or [],
            "opposing_planes": payload.get("opposing_planes") or [],
            "side_pieces": payload.get("side_pieces") or [],
            # Merged-segment fields (v3-merged-roof-segment): the segment's
            # final lifted 3D ring, the room/part/gap piece it was clipped to,
            # the building-boundary ring used for the outer clip, and the raw
            # proposal ids that contributed to this merged plane.
            "kind": payload.get("kind"),
            "member_proposal_ids": payload.get("member_proposal_ids") or [],
            "room_boundary_refs": payload.get("room_boundary_refs") or [],
            "building_boundary_xz": payload.get("building_boundary_xz"),
            "segment_corners_xyz": payload.get("segment_corners_xyz"),
        }
        self._append_roof_proposal_label(entry)
        out = json.dumps({"ok": True, "entry": entry}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _read_all_roof_proposal_splits(self) -> list[dict]:
        if not ROOF_PROPOSAL_SPLITS_PATH.exists():
            return []
        out: list[dict] = []
        with open(ROOF_PROPOSAL_SPLITS_PATH) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    out.append(entry)
        return out

    def _read_roof_proposal_splits_for(self, building_uuid: str) -> list[dict]:
        return [
            s
            for s in self._read_all_roof_proposal_splits()
            if s.get("building_uuid") == building_uuid
        ]

    def _read_all_roof_proposal_children_map(self) -> dict[str, list[str]]:
        """parent_id -> [child_id, ...] across all buildings (last-write-wins)."""
        out: dict[str, list[str]] = {}
        for rec in self._read_all_roof_proposal_splits():
            pid = rec.get("parent_id")
            children = rec.get("children") or []
            if not isinstance(pid, str) or not isinstance(children, list):
                continue
            kids = [c.get("id") for c in children if isinstance(c, dict)]
            kids = [k for k in kids if isinstance(k, str)]
            if len(kids) >= 2:
                out[pid] = kids
        return out

    def _append_roof_proposal_split(self, entry: dict) -> None:
        ROOF_PROPOSAL_SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ROOF_PROPOSAL_SPLITS_PATH, "a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _handle_roof_proposal_splits_get(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        building_uuid = (params.get("building_uuid") or [None])[0]
        if not building_uuid:
            self.send_error(
                HTTPStatus.BAD_REQUEST, "building_uuid query param required"
            )
            return
        records = self._read_roof_proposal_splits_for(building_uuid)
        body = json.dumps(
            {
                "building_uuid": building_uuid,
                "splits": records,
            }
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve_proposal_geometry(
        self, building_uuid: str, proposal_id: str
    ) -> tuple[list[list[float]], tuple[float, float, float, float], dict]:
        """Return (corners, plane, original_proposal_dict) for a leaf proposal.

        Walks splits.jsonl to find corners when proposal_id is a synthesized
        child. The plane is always inherited from the original ancestor in
        V3_CACHE.
        """
        _ensure_v3_cache()
        bldg = V3_CACHE.get(building_uuid)
        if not bldg:
            raise LookupError(f"building {building_uuid} not in v3 cache")
        # Inner part of the id; strip anything after first '#' to locate original.
        parts = proposal_id.split("::", 2)
        if len(parts) != 3:
            raise ValueError(f"bad proposal id: {proposal_id}")
        kind = parts[1]
        inner = parts[2]
        original_inner = inner.split("#", 1)[0]
        original_id = f"{parts[0]}::{parts[1]}::{original_inner}"
        # Merged segments live in `merged_roof_segments[]` with plane under
        # `merged_plane`; raw proposals live in `roof_proposals[]` with plane
        # under `plane`. Dispatch by the kind token in the id.
        if kind == "v3-merged-roof-segment":
            source_list = bldg.get("merged_roof_segments") or []
            plane_key = "merged_plane"
        else:
            source_list = bldg.get("roof_proposals") or []
            plane_key = "plane"
        original = next(
            (p for p in source_list if p.get("id") == original_id),
            None,
        )
        if not original:
            raise LookupError(f"original proposal {original_id} not found")
        plane_raw = original.get(plane_key)
        if not (isinstance(plane_raw, (list, tuple)) and len(plane_raw) == 4):
            raise ValueError(f"original proposal has no 4-tuple {plane_key}")
        plane = (
            float(plane_raw[0]),
            float(plane_raw[1]),
            float(plane_raw[2]),
            float(plane_raw[3]),
        )
        # Corners: either the original or a child from a split record.
        if proposal_id == original_id:
            corners = original.get("corners") or []
        else:
            corners = None
            for rec in self._read_roof_proposal_splits_for(building_uuid):
                for child in rec.get("children") or []:
                    if child.get("id") == proposal_id:
                        corners = child.get("corners") or []
                        break
                if corners is not None:
                    break
            if corners is None:
                raise LookupError(f"corners for {proposal_id} not found in splits")
        return list(corners), plane, original

    def _handle_roof_proposal_split_post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Body must be an object")
            return
        building_uuid = payload.get("building_uuid")
        parent_id = payload.get("parent_proposal_id")
        split_line = payload.get("split_line")
        if (
            not isinstance(building_uuid, str)
            or not building_uuid
            or not isinstance(parent_id, str)
            or not parent_id
            or not isinstance(split_line, list)
            or len(split_line) != 2
        ):
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                "building_uuid, parent_proposal_id, split_line [[x,z],[x,z]] required",
            )
            return
        try:
            p1 = (float(split_line[0][0]), float(split_line[0][1]))
            p2 = (float(split_line[1][0]), float(split_line[1][1]))
        except Exception:
            self.send_error(
                HTTPStatus.BAD_REQUEST, "split_line must be [[x,z],[x,z]] numbers"
            )
            return

        try:
            parent_corners, plane, original = self._resolve_proposal_geometry(
                building_uuid, parent_id
            )
        except Exception as exc:
            self.send_error(HTTPStatus.NOT_FOUND, f"Proposal lookup failed: {exc}")
            return

        try:
            children = _split_proposal_polygon(parent_corners, plane, p1, p2)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        parts = parent_id.split("::", 2)
        base_inner = parts[2]
        child_entries = []
        for i, (corners, centroid_xz) in enumerate(children):
            child_id = f"{parts[0]}::{parts[1]}::{base_inner}#{i}"
            child_entries.append(
                {
                    "id": child_id,
                    "corners": corners,
                    "centroid_xz": list(centroid_xz),
                }
            )

        entry = {
            "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "building_uuid": building_uuid,
            "parent_id": parent_id,
            "original_id": original.get("id"),
            "split_line": [list(p1), list(p2)],
            "children": child_entries,
        }
        self._append_roof_proposal_split(entry)

        response = {
            "ok": True,
            "parent_id": parent_id,
            "building_uuid": building_uuid,
            "children": [
                {
                    "id": c["id"],
                    "corners": c["corners"],
                    "features": original.get("features") or {},
                    "heuristic_label": original.get("heuristic_label"),
                    "segment_index": original.get("segment_index"),
                    "source_room_id": original.get("source_room_id"),
                    "source_wall_id": original.get("source_wall_id"),
                    "slab_room_id": original.get("slab_room_id"),
                    "slab_id": original.get("slab_id"),
                }
                for c in child_entries
            ],
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_v3(self, query: str):
        global V3_CACHE, V3_CACHE_MTIME
        params = urllib.parse.parse_qs(query)
        uuid = (params.get("uuid") or [None])[0]
        if not uuid:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing uuid query param")
            return
        if not V3_RESULTS_PATH.exists():
            self.send_error(
                HTTPStatus.NOT_FOUND,
                "reconcile_v3_results.json not found; run "
                "`python -m reconcile_v3.cli --all` first.",
            )
            return
        try:
            mtime = V3_RESULTS_PATH.stat().st_mtime
            if mtime != V3_CACHE_MTIME or not V3_CACHE:
                with open(V3_RESULTS_PATH) as handle:
                    data = json.load(handle)
                V3_CACHE = {
                    entry.get("building_uuid"): entry
                    for entry in data
                    if entry.get("building_uuid")
                }
                V3_CACHE_MTIME = mtime
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_GATEWAY, f"v3 load failed: {exc}")
            return
        payload = V3_CACHE.get(uuid)
        if payload is None:
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"No v3 results for {uuid}; rerun the v3 CLI for this building.",
            )
            return
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_candidate_faces(self, query: str):
        global CANDIDATE_FACES_CACHE, CANDIDATE_FACES_CACHE_MTIME
        params = urllib.parse.parse_qs(query)
        uuid = (params.get("uuid") or [None])[0]
        if not uuid:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing uuid query param")
            return
        candidate_faces_path = _resolve_artifact_path(
            env_var="VIEWER_CANDIDATE_FACES_PATH",
            default_candidates=CANDIDATE_FACES_PATH_CANDIDATES,
        )
        if not candidate_faces_path.exists():
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"{candidate_faces_path.name} not found; run "
                "`python scripts/build_candidate_faces.py` first.",
            )
            return
        try:
            mtime = candidate_faces_path.stat().st_mtime
            if mtime != CANDIDATE_FACES_CACHE_MTIME or not CANDIDATE_FACES_CACHE:
                with open(candidate_faces_path) as handle:
                    data = json.load(handle)
                CANDIDATE_FACES_CACHE = {
                    entry.get("building_uuid"): entry
                    for entry in data
                    if entry.get("building_uuid")
                }
                CANDIDATE_FACES_CACHE_MTIME = mtime
        except Exception as exc:
            self.send_error(
                HTTPStatus.BAD_GATEWAY, f"candidate-faces load failed: {exc}"
            )
            return
        payload = CANDIDATE_FACES_CACHE.get(uuid)
        if payload is None:
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"No candidate faces for {uuid}; rerun build_candidate_faces.py.",
            )
            return
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ensure_roof_caches(self) -> tuple[bool, str]:
        """Load the three roof-viewer source files on demand with mtime checks.

        Returns (ok, message). On failure, the caller should emit the message.
        """
        global ROOF_RESULTS_CACHE, ROOF_RESULTS_CACHE_MTIME
        global BUILDINGS_3D_CACHE, BUILDINGS_3D_CACHE_MTIME
        global ROOF_AUDIT_CACHE, ROOF_AUDIT_CACHE_MTIME

        if not ROOF_RESULTS_PATH.exists():
            return False, f"{ROOF_RESULTS_PATH.name} not found"
        if not BUILDINGS_3D_PATH.exists():
            return False, f"{BUILDINGS_3D_PATH.name} not found"

        try:
            rm = ROOF_RESULTS_PATH.stat().st_mtime
            if rm != ROOF_RESULTS_CACHE_MTIME or not ROOF_RESULTS_CACHE:
                with open(ROOF_RESULTS_PATH) as handle:
                    data = json.load(handle)
                ROOF_RESULTS_CACHE = data if isinstance(data, dict) else {}
                ROOF_RESULTS_CACHE_MTIME = rm
        except Exception as exc:
            return False, f"roof results load failed: {exc}"

        try:
            bm = BUILDINGS_3D_PATH.stat().st_mtime
            if bm != BUILDINGS_3D_CACHE_MTIME or not BUILDINGS_3D_CACHE:
                with open(BUILDINGS_3D_PATH) as handle:
                    data = json.load(handle)
                by_uuid: dict[str, dict] = {}
                for b in data or []:
                    uid = b.get("uuid")
                    if uid:
                        by_uuid[uid] = b
                BUILDINGS_3D_CACHE = by_uuid
                BUILDINGS_3D_CACHE_MTIME = bm
        except Exception as exc:
            return False, f"buildings_3d load failed: {exc}"

        try:
            if ROOF_AUDIT_PATH.exists():
                am = ROOF_AUDIT_PATH.stat().st_mtime
                if am != ROOF_AUDIT_CACHE_MTIME or not ROOF_AUDIT_CACHE:
                    with open(ROOF_AUDIT_PATH) as handle:
                        audit = json.load(handle)
                    rows = audit.get("rows") or []
                    by_uuid: dict[str, list[dict]] = {}
                    for r in rows:
                        uid = r.get("uuid")
                        if uid:
                            by_uuid.setdefault(uid, []).append(r)
                    ROOF_AUDIT_CACHE = by_uuid
                    ROOF_AUDIT_CACHE_MTIME = am
            else:
                ROOF_AUDIT_CACHE = {}
                ROOF_AUDIT_CACHE_MTIME = 0.0
        except Exception:
            ROOF_AUDIT_CACHE = {}
            ROOF_AUDIT_CACHE_MTIME = 0.0

        return True, ""

    def _roof_audit_for(self, uuid: str) -> dict[str, dict]:
        """Return {element_id: audit_row} for a building."""
        rows = ROOF_AUDIT_CACHE.get(uuid) or []
        return {r.get("element_id"): r for r in rows if r.get("element_id")}

    def _handle_roof_index(self) -> None:
        """Return a compact list of buildings for the roof viewer sidebar.

        Each entry: {uuid, address, n_oblique, n_flat, n_weak_committed,
        n_weak_candidate, n_rooms, n_stories}. Sorted by weak committed desc.
        """
        global ROOF_INDEX_CACHE, ROOF_INDEX_CACHE_KEY
        ok, msg = self._ensure_roof_caches()
        if not ok:
            self.send_error(HTTPStatus.NOT_FOUND, msg)
            return

        key = (
            ROOF_RESULTS_CACHE_MTIME,
            BUILDINGS_3D_CACHE_MTIME,
            ROOF_AUDIT_CACHE_MTIME,
        )
        if ROOF_INDEX_CACHE is not None and key == ROOF_INDEX_CACHE_KEY:
            index = ROOF_INDEX_CACHE
        else:
            index = []
            for uuid, building in BUILDINGS_3D_CACHE.items():
                roof = ROOF_RESULTS_CACHE.get(uuid)
                if roof is None:
                    continue
                rs = roof.get("roof_surfaces") or {}
                n_oblique = len(rs.get("oblique") or [])
                n_flat = len(rs.get("flat") or [])
                audit = self._roof_audit_for(uuid)
                n_weak_committed = sum(
                    1
                    for r in audit.values()
                    if r.get("group") == "committed_oblique"
                    and r.get("classification") == "weak"
                )
                n_weak_candidate = sum(
                    1
                    for r in audit.values()
                    if r.get("group") == "candidate_oblique"
                    and r.get("classification") == "weak"
                )
                addr = building.get("address")
                if isinstance(addr, dict):
                    addr = " ".join(
                        str(addr.get(k, ""))
                        for k in ("street", "number", "postcode", "city")
                        if addr.get(k)
                    ).strip()
                rooms = building.get("rooms") or []
                stories = {int(r.get("story", 0)) for r in rooms}
                index.append(
                    {
                        "uuid": uuid,
                        "address": addr or "(no address)",
                        "n_oblique": n_oblique,
                        "n_flat": n_flat,
                        "n_weak_committed": n_weak_committed,
                        "n_weak_candidate": n_weak_candidate,
                        "n_rooms": len(rooms),
                        "n_stories": len(stories),
                    }
                )
            index.sort(
                key=lambda e: (
                    -e["n_weak_committed"],
                    -e["n_weak_candidate"],
                    -e["n_oblique"],
                )
            )
            ROOF_INDEX_CACHE = index
            ROOF_INDEX_CACHE_KEY = key

        body = json.dumps({"buildings": index}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_tier_index(self) -> None:
        """Group buildings by complexity tier for viewer-tiers.html.

        Response shape:
          {"tiers": [{"tier": 1, "label": "...", "count": N,
                      "buildings": [{"uuid", "address", "n_stories",
                                     "n_oblique", "n_flat",
                                     "has_half_height", "has_gable"}]}]}
        """
        try:
            from .complexity_tiers import TIER_LABELS, classify_building
        except ImportError:
            from complexity_tiers import TIER_LABELS, classify_building

        global TIER_INDEX_CACHE, TIER_INDEX_CACHE_KEY
        ok, msg = self._ensure_roof_caches()
        if not ok:
            self.send_error(HTTPStatus.NOT_FOUND, msg)
            return

        key = (ROOF_RESULTS_CACHE_MTIME, BUILDINGS_3D_CACHE_MTIME)
        if TIER_INDEX_CACHE is not None and key == TIER_INDEX_CACHE_KEY:
            tiers_list = TIER_INDEX_CACHE
        else:
            buckets: dict[int, list[dict]] = {t: [] for t in TIER_LABELS}
            for uuid, building in BUILDINGS_3D_CACHE.items():
                roof = ROOF_RESULTS_CACHE.get(uuid)
                result = classify_building(building, roof)
                addr = building.get("address")
                if isinstance(addr, dict):
                    addr = " ".join(
                        str(addr.get(k, ""))
                        for k in ("street", "number", "postcode", "city")
                        if addr.get(k)
                    ).strip()
                entry = {
                    "uuid": uuid,
                    "address": addr or "(no address)",
                    **result["signals"],
                }
                buckets[result["tier"]].append(entry)
            for bucket in buckets.values():
                bucket.sort(key=lambda e: e["address"].lower())
            tiers_list = [
                {
                    "tier": tier,
                    "label": TIER_LABELS[tier],
                    "count": len(buckets[tier]),
                    "buildings": buckets[tier],
                }
                for tier in sorted(TIER_LABELS)
            ]
            TIER_INDEX_CACHE = tiers_list
            TIER_INDEX_CACHE_KEY = key

        body = json.dumps({"tiers": tiers_list}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_building_merged(self, query: str) -> None:
        """Per-building merged-model payload for the tier preview.

        Returns the slice of `buildings_3d.json` needed to render the
        heuristic "Full model" look (walls, floors, gap closures, stitch
        walls, cross-storey gaps), plus knee walls / dormer cheeks &
        headers, raw-ceiling fallback scraps, and the V2 slanted ceiling
        pieces — all with dormer cutouts already subtracted in XZ so the
        tier preview does not have to recompute them.
        """
        params = urllib.parse.parse_qs(query)
        uuid = (params.get("uuid") or [None])[0]
        if not uuid:
            self.send_error(HTTPStatus.BAD_REQUEST, "missing uuid")
            return
        ok, msg = self._ensure_roof_caches()
        if not ok:
            self.send_error(HTTPStatus.NOT_FOUND, msg)
            return
        building = BUILDINGS_3D_CACHE.get(uuid)
        if building is None:
            self.send_error(HTTPStatus.NOT_FOUND, f"no building for uuid {uuid}")
            return
        # Rooms are heavy (raw_ceiling_planes, walls_merged w/ metadata); strip
        # to just the fields the tier preview renders.
        rooms_out = []
        for room in building.get("rooms") or []:
            rooms_out.append(
                {
                    "story": room.get("story", 0),
                    "floor_polygon": room.get("floor_polygon") or [],
                    "walls_computed": [
                        {
                            "corners": w.get("corners"),
                            "extension_strip": w.get("extension_strip") or [],
                        }
                        for w in room.get("walls_computed") or []
                    ],
                    # Raw scanned walls (Apple RoomPlan output). The Full
                    # model toggle in viewer.html doesn't render these —
                    # they're shown only under the "Merged (Apple)"
                    # toggle — but some scans have walls_merged entries
                    # that never made it into walls_computed (e.g. walls
                    # below `ytop_filter` or edge cases in the reconciler).
                    # Include them here so the tier preview doesn't miss
                    # visible walls the user sees in viewer.html.
                    "walls_merged": [
                        {"corners": w.get("corners")}
                        for w in room.get("walls_merged") or []
                        if w.get("corners")
                    ],
                    "doors": [
                        {"corners": d.get("corners")} for d in room.get("doors") or []
                    ],
                    "windows": [
                        {"corners": w.get("corners")} for w in room.get("windows") or []
                    ],
                }
            )
        roof = ROOF_RESULTS_CACHE.get(uuid) or {}
        # Thermal knee walls + dormer cheeks/headers from the roof pipeline —
        # small verticals that close the envelope below slanted surfaces.
        # Flat ceilings are no longer shipped here: roofs come from V2
        # split pieces + raw-ceiling fallback (see raw_ceiling_fallback below).
        thermal = (roof.get("ceiling") or {}).get("thermal") or []
        has_backend_arrangement = bool(
            (roof.get("roof_surfaces") or {}).get("oblique_split") or []
        )
        thermal_kinds = {"thermal-knee"}
        if not has_backend_arrangement:
            thermal_kinds.update({"thermal-dormer-cheek", "thermal-dormer-header"})
        knee_walls = [
            {"poly": t.get("poly"), "kind": t.get("kind")}
            for t in thermal
            if t.get("poly") and t.get("kind") in thermal_kinds
        ]
        raw_ceiling_fallback = _raw_ceiling_fallback_for_uuid(uuid, building)
        slanted_pieces = _slanted_pieces_for_uuid(uuid)
        slanted_roof_xz = _slanted_pieces_xz_union(slanted_pieces)
        gap_walls = [
            {"corners": g.get("corners"), "type": g.get("type")}
            for g in building.get("gap_walls") or []
            if g.get("corners")
        ]
        gap_closures = [
            {"corners": g.get("corners"), "type": g.get("type")}
            for g in building.get("gap_closures") or []
            if g.get("corners")
        ]
        cross_floor_gaps = [
            {
                "corners": g.get("corners"),
                "ceiling_corners": g.get("ceiling_corners"),
                "type": g.get("type"),
            }
            for g in building.get("cross_floor_gaps") or []
            if g.get("corners")
        ]
        gap_walls = _filter_gap_ceiling_caps_for_full_model(gap_walls, slanted_roof_xz)
        gap_closures = _filter_gap_ceiling_caps_for_full_model(
            gap_closures, slanted_roof_xz
        )
        cross_floor_gaps = _filter_cross_floor_gap_ceiling_lids_for_full_model(
            cross_floor_gaps, slanted_pieces
        )
        payload = {
            "uuid": uuid,
            "rooms": rooms_out,
            "gap_walls": gap_walls,
            "gap_closures": gap_closures,
            "stitch_walls": [
                {"corners": s.get("corners")}
                for s in building.get("stitch_walls") or []
                if s.get("corners")
            ],
            "cross_floor_gaps": cross_floor_gaps,
            "raw_ceiling_fallback": raw_ceiling_fallback,
            "knee_walls": knee_walls,
            "slanted_pieces": slanted_pieces,
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_roof_detail(self, query: str) -> None:
        """Return per-building roof payload for viewer-roof.html.

        Shape (compact):
            {
              uuid, address,
              segments: [{id, a, b, incl, azimuth, story, room_idx,
                          cluster_index, used}],
              clusters: [{index, avgIncl, avgAzimuth, seg_count}],
              candidates: [{index, corners, story, avg_incl, avg_azimuth,
                            cluster_index, audit}],
              committed: [{index, corners, story, avg_incl, avg_azimuth,
                           audit}],
              flat: [{index, corners, story, audit}],
              raw_ceilings: [{story, room_index, plane_index, corners,
                              exposed}],
              floors: [{story, room_index, corners}],
              audit_thresholds: {...},
            }

        ``audit`` is ``None`` for surfaces not scored (or when the audit file
        is missing). A candidate's ``cluster_index`` is inferred by matching
        its ``cl.refPt`` against ``valid_clusters[i].refPt``.
        """
        params = urllib.parse.parse_qs(query)
        uuid = (params.get("uuid") or [None])[0]
        if not uuid:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing uuid query param")
            return
        ok, msg = self._ensure_roof_caches()
        if not ok:
            self.send_error(HTTPStatus.NOT_FOUND, msg)
            return
        building = BUILDINGS_3D_CACHE.get(uuid)
        roof = ROOF_RESULTS_CACHE.get(uuid)
        if building is None or roof is None:
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"No roof data or buildings_3d entry for {uuid}",
            )
            return

        audit_by_id = self._roof_audit_for(uuid)

        def _slim_audit(row: dict | None) -> dict | None:
            if not row:
                return None
            return {
                "classification": row.get("classification"),
                "match_count": row.get("match_count"),
                "xz_coverage": row.get("xz_coverage"),
                "median_dy": row.get("median_dy"),
                "p95_abs_dy": row.get("p95_abs_dy"),
                "normal_dot_median": row.get("normal_dot_median"),
                "raw_y_min": row.get("raw_y_min"),
                "raw_y_max": row.get("raw_y_max"),
                "ceiling_raw_ids": row.get("ceiling_raw_ids"),
                "element_id": row.get("element_id"),
            }

        valid_clusters = roof.get("valid_clusters") or []
        cluster_key_to_index: dict[tuple, int] = {}
        clusters_out: list[dict] = []
        for idx, cl in enumerate(valid_clusters):
            rp = cl.get("refPt") or {}
            key = (
                round(float(rp.get("x", 0.0)), 4),
                round(float(rp.get("y", 0.0)), 4),
                round(float(rp.get("z", 0.0)), 4),
                round(float(cl.get("avgIncl", 0.0)), 3),
                round(float(cl.get("avgAzimuth", 0.0)), 3),
            )
            cluster_key_to_index[key] = idx
            clusters_out.append(
                {
                    "index": idx,
                    "avgIncl": cl.get("avgIncl"),
                    "avgAzimuth": cl.get("avgAzimuth"),
                    "seg_count": len(cl.get("segs") or []),
                    "refPt": cl.get("refPt"),
                }
            )

        # Tag wall-top segments by the cluster they belong to.
        seg_cluster_lookup: dict[tuple, int] = {}
        for idx, cl in enumerate(valid_clusters):
            for s in cl.get("segs") or []:
                a = s.get("a") or []
                b = s.get("b") or []
                if len(a) < 3 or len(b) < 3:
                    continue
                k = (
                    round(float(a[0]), 4),
                    round(float(a[1]), 4),
                    round(float(a[2]), 4),
                    round(float(b[0]), 4),
                    round(float(b[1]), 4),
                    round(float(b[2]), 4),
                )
                seg_cluster_lookup[k] = idx

        segments_out: list[dict] = []
        for sidx, s in enumerate(roof.get("segments") or []):
            a = s.get("a") or []
            b = s.get("b") or []
            if len(a) < 3 or len(b) < 3:
                continue
            k = (
                round(float(a[0]), 4),
                round(float(a[1]), 4),
                round(float(a[2]), 4),
                round(float(b[0]), 4),
                round(float(b[1]), 4),
                round(float(b[2]), 4),
            )
            ci = seg_cluster_lookup.get(k)
            segments_out.append(
                {
                    "id": f"seg:{sidx}",
                    "a": a,
                    "b": b,
                    "incl": s.get("incl"),
                    "azimuth": s.get("azimuth"),
                    "len": s.get("len"),
                    "story": s.get("story"),
                    "room_idx": s.get("room_idx"),
                    "cluster_index": ci,
                    "used": ci is not None,
                }
            )

        ceiling = roof.get("ceiling") or {}
        candidates_out: list[dict] = []
        for idx, plane in enumerate(ceiling.get("planes") or []):
            corners = self._candidate_corners_3d(plane)
            cl = plane.get("cl") or {}
            rp = cl.get("refPt") or {}
            key = (
                round(float(rp.get("x", 0.0)), 4),
                round(float(rp.get("y", 0.0)), 4),
                round(float(rp.get("z", 0.0)), 4),
                round(float(cl.get("avgIncl", 0.0)), 3),
                round(float(cl.get("avgAzimuth", 0.0)), 3),
            )
            cand_cluster = cluster_key_to_index.get(key)
            audit_id = f"{uuid}::ceiling-oblique::ceiling-oblique:{idx}"
            candidates_out.append(
                {
                    "index": idx,
                    "corners": corners,
                    "story": plane.get("dominantStory"),
                    "avg_incl": cl.get("avgIncl"),
                    "avg_azimuth": cl.get("avgAzimuth"),
                    "cluster_index": cand_cluster,
                    "room_indices": plane.get("room_indices"),
                    "audit_id": audit_id,
                    "audit": _slim_audit(audit_by_id.get(audit_id)),
                }
            )

        roof_surfaces = roof.get("roof_surfaces") or {}
        committed_out: list[dict] = []
        for idx, ob in enumerate(roof_surfaces.get("oblique") or []):
            corners = ob.get("corners") or []
            cl = ob.get("cluster") or {}
            audit_id = f"{uuid}::roof-oblique::oblique:{idx}"
            committed_out.append(
                {
                    "index": idx,
                    "corners": corners,
                    "story": ob.get("dominant_story"),
                    "avg_incl": cl.get("avgIncl"),
                    "avg_azimuth": cl.get("avgAzimuth"),
                    "surface_kind": ob.get("surface_kind"),
                    "audit_id": audit_id,
                    "audit": _slim_audit(audit_by_id.get(audit_id)),
                }
            )
        flat_out: list[dict] = []
        for idx, fl in enumerate(roof_surfaces.get("flat") or []):
            audit_id = f"{uuid}::roof-flat::flat:{idx}"
            flat_out.append(
                {
                    "index": idx,
                    "corners": fl.get("corners") or [],
                    "story": fl.get("dominant_story"),
                    "surface_kind": fl.get("surface_kind"),
                    "audit_id": audit_id,
                    "audit": _slim_audit(audit_by_id.get(audit_id)),
                }
            )

        exposed_keys: set[tuple[int, int]] = set()
        for entry in ceiling.get("exposed_rooms") or []:
            si = entry.get("story")
            ri = entry.get("room_index")
            if si is not None and ri is not None:
                exposed_keys.add((int(si), int(ri)))

        raw_ceilings_out: list[dict] = []
        floors_out: list[dict] = []
        ceilings_out: list[dict] = []
        for ri, room in enumerate(building.get("rooms") or []):
            story = int(room.get("story", 0))
            # raw_ceiling_planes was populated by an earlier pipeline
            # revision; the current rebuild no longer attaches it. Tolerate
            # either shape so the endpoint works across rebuilds.
            for pi, plane in enumerate(room.get("raw_ceiling_planes") or []):
                corners = plane.get("corners") or []
                if len(corners) < 3:
                    continue
                raw_ceilings_out.append(
                    {
                        "story": story,
                        "room_index": ri,
                        "plane_index": pi,
                        "corners": corners,
                        "exposed": (story, ri) in exposed_keys,
                        "element_id": f"{uuid}::ceiling-raw::{story}:{ri}:{pi}",
                    }
                )
            fp = room.get("floor_polygon") or []
            if len(fp) >= 3:
                floors_out.append(
                    {
                        "story": story,
                        "room_index": ri,
                        "corners": fp,
                    }
                )
            cp = room.get("ceiling_polygon") or []
            if len(cp) >= 3:
                ceilings_out.append(
                    {
                        "story": story,
                        "room_index": ri,
                        "corners": cp,
                        "ceiling_type": room.get("ceiling_type"),
                        "ridge_y": room.get("ceiling_ridge_height"),
                        "eave_y": room.get("ceiling_eave_height"),
                    }
                )

        addr = building.get("address")
        if isinstance(addr, dict):
            addr = " ".join(
                str(addr.get(k, ""))
                for k in ("street", "number", "postcode", "city")
                if addr.get(k)
            ).strip()

        payload = {
            "uuid": uuid,
            "address": addr or None,
            "segments": segments_out,
            "clusters": clusters_out,
            "candidates": candidates_out,
            "committed": committed_out,
            "flat": flat_out,
            "raw_ceilings": raw_ceilings_out,
            "floors": floors_out,
            "room_ceilings": ceilings_out,
            "audit_thresholds": {
                "weak_abs_dy_m": 0.5,
                "weak_normal_dot": 0.80,
                "min_xz_coverage_for_judgement": 0.10,
            },
        }

        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _candidate_corners_3d(plane: dict) -> list[list[float]]:
        """Expand a ceiling.plane bbox (ridge/slope coords) to 4 3D corners.

        Mirrors ``scripts/audit_scan_ceiling_support.py::_candidate_polygon``
        but produces y-values from the plane equation instead of dropping to
        the XZ projection.
        """
        try:
            rx = float(plane["ridgeX"])
            rz = float(plane["ridgeZ"])
            sx = float(plane["slopeX"])
            sz = float(plane["slopeZ"])
            ref = plane["ref"]
            rx0, ry0, rz0 = float(ref["x"]), float(ref["y"]), float(ref["z"])
            n = plane["n"]
            nx, ny, nz = float(n["x"]), float(n["y"]), float(n["z"])
        except (KeyError, TypeError, ValueError):
            return []

        def y_at(x: float, z: float) -> float:
            if abs(ny) < 1e-6:
                return ry0
            return ry0 - (nx * (x - rx0) + nz * (z - rz0)) / ny

        bounds = [
            (plane["minRidge"], plane["minSlope"]),
            (plane["maxRidge"], plane["minSlope"]),
            (plane["maxRidge"], plane["maxSlope"]),
            (plane["minRidge"], plane["maxSlope"]),
        ]
        corners: list[list[float]] = []
        for r_, s_ in bounds:
            x = rx0 + float(r_) * rx + float(s_) * sx
            z = rz0 + float(r_) * rz + float(s_) * sz
            corners.append([x, float(y_at(x, z)), z])
        return corners

    def _handle_ridge_eave_scores(self, query: str):
        """Return ridge/eave topology scores for one building.

        Produced by ``scripts/score_candidates_ridge_eave.py``. Response is
        the per-building scoring slice — candidate best-pair scores plus
        the pair geometry (ridge, medial axis, eaves, OBB) for rendering.
        """
        global RIDGE_EAVE_CACHE, RIDGE_EAVE_CACHE_MTIME
        params = urllib.parse.parse_qs(query)
        uuid = (params.get("uuid") or [None])[0]
        if not uuid:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing uuid query param")
            return
        ridge_eave_scores_path = _resolve_artifact_path(
            env_var="VIEWER_RIDGE_EAVE_SCORES_PATH",
            default_candidates=RIDGE_EAVE_SCORES_PATH_CANDIDATES,
        )
        if not ridge_eave_scores_path.exists():
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"{ridge_eave_scores_path.name} not found; run "
                "`python scripts/score_candidates_ridge_eave.py` first.",
            )
            return
        try:
            mtime = ridge_eave_scores_path.stat().st_mtime
            if mtime != RIDGE_EAVE_CACHE_MTIME or not RIDGE_EAVE_CACHE:
                with open(ridge_eave_scores_path) as handle:
                    data = json.load(handle)
                RIDGE_EAVE_CACHE = {
                    entry.get("building_uuid"): entry
                    for entry in (data.get("buildings") or [])
                    if entry.get("building_uuid")
                }
                RIDGE_EAVE_CACHE_MTIME = mtime
        except Exception as exc:
            self.send_error(
                HTTPStatus.BAD_GATEWAY, f"ridge-eave-scores load failed: {exc}"
            )
            return
        payload = RIDGE_EAVE_CACHE.get(uuid)
        if payload is None:
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"No ridge/eave scores for {uuid}; rerun scorer.",
            )
            return
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_raw_ceiling_prototype(self):
        """Return the Phase-2 raw-ceiling role + reconstruction sidecar.

        Combines ``reports/raw_ceiling_prototype/roles.json`` and
        ``reconstructions.json`` into a single payload the viewer fetches
        once. Either file may be absent (prototype not yet run) — the
        missing section returns as empty.
        """
        global RAW_CEILING_PROTOTYPE_CACHE, RAW_CEILING_PROTOTYPE_CACHE_MTIME
        roles_mtime = (
            RAW_CEILING_ROLES_PATH.stat().st_mtime
            if RAW_CEILING_ROLES_PATH.exists()
            else 0.0
        )
        recon_mtime = (
            RAW_CEILING_RECON_PATH.stat().st_mtime
            if RAW_CEILING_RECON_PATH.exists()
            else 0.0
        )
        if (
            not RAW_CEILING_PROTOTYPE_CACHE
            or (roles_mtime, recon_mtime) != RAW_CEILING_PROTOTYPE_CACHE_MTIME
        ):
            roles_data: dict = {}
            recon_data: dict = {}
            if RAW_CEILING_ROLES_PATH.exists():
                try:
                    roles_data = json.loads(RAW_CEILING_ROLES_PATH.read_text())
                except Exception:
                    roles_data = {}
            if RAW_CEILING_RECON_PATH.exists():
                try:
                    recon_data = json.loads(RAW_CEILING_RECON_PATH.read_text())
                except Exception:
                    recon_data = {}
            RAW_CEILING_PROTOTYPE_CACHE = {
                "thresholds": roles_data.get("thresholds") or {},
                "planes": roles_data.get("planes") or {},
                "rooms": roles_data.get("rooms") or {},
                "reconstructions": recon_data.get("buildings") or {},
                "available": bool(roles_data) or bool(recon_data),
            }
            RAW_CEILING_PROTOTYPE_CACHE_MTIME = (roles_mtime, recon_mtime)
        body = json.dumps(RAW_CEILING_PROTOTYPE_CACHE).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_computed_overextend(self):
        """Return per-surface overextend polygons keyed by building uuid.

        Source: ``reports/computed_extent_vs_raw/overextend_polygons.json``,
        produced by ``scripts/audit_computed_surface_extent_vs_raw.py``.
        Missing file returns an empty payload with ``available: false``.
        """
        global COMPUTED_OVEREXTEND_CACHE, COMPUTED_OVEREXTEND_CACHE_MTIME
        mtime = (
            COMPUTED_OVEREXTEND_PATH.stat().st_mtime
            if COMPUTED_OVEREXTEND_PATH.exists()
            else 0.0
        )
        if not COMPUTED_OVEREXTEND_CACHE or mtime != COMPUTED_OVEREXTEND_CACHE_MTIME:
            data: dict = {}
            if COMPUTED_OVEREXTEND_PATH.exists():
                try:
                    data = json.loads(COMPUTED_OVEREXTEND_PATH.read_text())
                except Exception:
                    data = {}
            COMPUTED_OVEREXTEND_CACHE = {
                "buildings": data.get("buildings") or {},
                "available": bool(data),
            }
            COMPUTED_OVEREXTEND_CACHE_MTIME = mtime
        body = json.dumps(COMPUTED_OVEREXTEND_CACHE).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_raw_disagreement(self):
        """Return raw-ceiling orientation-disagreement polygons keyed by
        building uuid.

        Source: ``reports/raw_orientation_disagreement/disagreement_polygons.json``,
        produced by ``scripts/audit_raw_orientation_disagreement.py``.
        Missing file returns an empty payload with ``available: false``.
        """
        global RAW_DISAGREEMENT_CACHE, RAW_DISAGREEMENT_CACHE_MTIME
        mtime = (
            RAW_DISAGREEMENT_PATH.stat().st_mtime
            if RAW_DISAGREEMENT_PATH.exists()
            else 0.0
        )
        if not RAW_DISAGREEMENT_CACHE or mtime != RAW_DISAGREEMENT_CACHE_MTIME:
            data: dict = {}
            if RAW_DISAGREEMENT_PATH.exists():
                try:
                    data = json.loads(RAW_DISAGREEMENT_PATH.read_text())
                except Exception:
                    data = {}
            RAW_DISAGREEMENT_CACHE = {
                "buildings": data.get("buildings") or {},
                "available": bool(data),
            }
            RAW_DISAGREEMENT_CACHE_MTIME = mtime
        body = json.dumps(RAW_DISAGREEMENT_CACHE).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_raw_ceiling_plane_splits(self, query: str):
        """Return raw-eave-supported split polygons keyed by building uuid.

        Query:
          - ``version=v1`` (default):
          ``reports/raw_ceiling_plane_scorer/plane_extent_splits.json``
          - ``version=v2``:
          ``.context/raw_ceiling_plane_scorer_v2_full/plane_extent_splits.json``
        Missing files return an empty payload with ``available: false``.
        """
        global \
            RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION, \
            RAW_CEILING_PLANE_SPLITS_CACHE_MTIME_BY_VERSION
        params = urllib.parse.parse_qs(query or "")
        requested_version = (
            str((params.get("version") or ["v1"])[0] or "v1").strip().lower()
        )
        version = (
            requested_version
            if requested_version in RAW_CEILING_PLANE_SPLITS_PATHS
            else "v1"
        )
        path = RAW_CEILING_PLANE_SPLITS_PATHS[version]
        mtime = path.stat().st_mtime if path.exists() else 0.0
        if not RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION.get(
            version
        ) or mtime != RAW_CEILING_PLANE_SPLITS_CACHE_MTIME_BY_VERSION.get(version, 0.0):
            data: dict = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                except Exception:
                    data = {}
            RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION[version] = {
                "buildings": data.get("buildings") or {},
                "available": bool(data),
                "version": version,
            }
            RAW_CEILING_PLANE_SPLITS_CACHE_MTIME_BY_VERSION[version] = mtime
        body = json.dumps(RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION[version]).encode(
            "utf-8"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_ceiling_replacement(self):
        """Return clean-ceiling replacement polygons keyed by building uuid.

        Source: ``reports/noisy_slanted_ceilings/replacement_polygons.json``,
        produced by ``scripts/audit_noisy_slanted_ceiling_replacement.py``.
        Missing file returns an empty payload with ``available: false``.
        """
        global CEILING_REPLACEMENT_CACHE, CEILING_REPLACEMENT_CACHE_MTIME
        mtime = (
            CEILING_REPLACEMENT_PATH.stat().st_mtime
            if CEILING_REPLACEMENT_PATH.exists()
            else 0.0
        )
        if not CEILING_REPLACEMENT_CACHE or mtime != CEILING_REPLACEMENT_CACHE_MTIME:
            data: dict = {}
            if CEILING_REPLACEMENT_PATH.exists():
                try:
                    data = json.loads(CEILING_REPLACEMENT_PATH.read_text())
                except Exception:
                    data = {}
            CEILING_REPLACEMENT_CACHE = {
                "buildings": data.get("buildings") or {},
                "available": bool(data),
            }
            CEILING_REPLACEMENT_CACHE_MTIME = mtime
        body = json.dumps(CEILING_REPLACEMENT_CACHE).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_reconstruction(self, query: str):
        """Return the BIP solver's selection for one building, joined with
        the matching candidate-face footprints so the viewer can render the
        selected envelope without a second round-trip.

        Response shape::

            {
              "building_uuid": "...",
              "status": "solved" | "ambiguous" | "infeasible" | "no_candidates",
              "decision": "auto_accept" | "review",
              "selected_face_ids": [...],
              "selected_faces": [ <candidate-face dict>, ... ],
              "objective_value": ..., "runner_up_objective": ...,
              "coverage_ratio": ..., "lp_gap": ..., "solve_time_ms": ...,
              "reason": ...
            }
        """
        global RECONSTRUCTION_CACHE, RECONSTRUCTION_CACHE_MTIME
        global CANDIDATE_FACES_CACHE, CANDIDATE_FACES_CACHE_MTIME
        params = urllib.parse.parse_qs(query)
        uuid = (params.get("uuid") or [None])[0]
        if not uuid:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing uuid query param")
            return
        reconstruction_path = _resolve_artifact_path(
            env_var="VIEWER_RECONSTRUCTION_PATH",
            default_candidates=RECONSTRUCTION_PATH_CANDIDATES,
        )
        candidate_faces_path = _resolve_artifact_path(
            env_var="VIEWER_CANDIDATE_FACES_PATH",
            default_candidates=CANDIDATE_FACES_PATH_CANDIDATES,
        )
        if not reconstruction_path.exists():
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"{reconstruction_path.name} not found; run "
                "`python scripts/run_reconstruction_solver.py` first.",
            )
            return
        try:
            mtime = reconstruction_path.stat().st_mtime
            if mtime != RECONSTRUCTION_CACHE_MTIME or not RECONSTRUCTION_CACHE:
                with open(reconstruction_path) as handle:
                    data = json.load(handle)
                RECONSTRUCTION_CACHE = {
                    entry.get("building_uuid"): entry
                    for entry in data
                    if entry.get("building_uuid")
                }
                RECONSTRUCTION_CACHE_MTIME = mtime
        except Exception as exc:
            self.send_error(
                HTTPStatus.BAD_GATEWAY, f"reconstruction load failed: {exc}"
            )
            return
        payload = RECONSTRUCTION_CACHE.get(uuid)
        if payload is None:
            self.send_error(
                HTTPStatus.NOT_FOUND,
                f"No reconstruction result for {uuid}; rerun the solver CLI.",
            )
            return

        # Join with the candidate-face cache so the viewer has polygon
        # footprints for the selected ids.
        try:
            cmtime = (
                candidate_faces_path.stat().st_mtime
                if candidate_faces_path.exists()
                else 0.0
            )
            if cmtime and (
                cmtime != CANDIDATE_FACES_CACHE_MTIME or not CANDIDATE_FACES_CACHE
            ):
                with open(candidate_faces_path) as handle:
                    cdata = json.load(handle)
                CANDIDATE_FACES_CACHE = {
                    entry.get("building_uuid"): entry
                    for entry in cdata
                    if entry.get("building_uuid")
                }
                CANDIDATE_FACES_CACHE_MTIME = cmtime
        except Exception:
            pass

        selected_ids = set(payload.get("selected_face_ids") or [])
        selected_faces: list[dict] = []
        bldg_candidates = (CANDIDATE_FACES_CACHE.get(uuid) or {}).get(
            "candidates"
        ) or []
        if selected_ids and bldg_candidates:
            selected_faces = [c for c in bldg_candidates if c.get("id") in selected_ids]

        response = dict(payload)
        response["selected_faces"] = selected_faces
        body = json.dumps(response).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_ontology_artifacts(self, query: str):
        params = urllib.parse.parse_qs(query)
        uuid = (params.get("uuid") or [None])[0]
        view = (params.get("view") or ["summary"])[0]
        part_id = (params.get("part_id") or [None])[0]
        if not uuid:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing uuid query param")
            return
        try:
            entry = ONTOLOGY_CACHE.get(uuid)
            if entry is None:
                entry = _build_ontology_cache_entry(uuid)
                ONTOLOGY_CACHE[uuid] = entry
            if view == "summary":
                payload = entry["summary"]
            elif view == "part":
                if not part_id:
                    self.send_error(
                        HTTPStatus.BAD_REQUEST, "Missing part_id for view=part"
                    )
                    return
                payload = entry["parts"].get(part_id)
                if payload is None:
                    self.send_error(
                        HTTPStatus.NOT_FOUND, f"No ontology part {part_id} for {uuid}"
                    )
                    return
            elif view == "full-model":
                payload = entry.get("full_model")
                if payload is None:
                    self.send_error(
                        HTTPStatus.NOT_FOUND, f"No full-model payload for {uuid}"
                    )
                    return
            else:
                self.send_error(
                    HTTPStatus.BAD_REQUEST, f"Unsupported ontology view: {view}"
                )
                return
        except FileNotFoundError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_GATEWAY, f"Ontology build failed: {exc}")
            return

        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Keep default static path normalization behavior explicit.
    def translate_path(self, path: str) -> str:
        path = urllib.parse.urlparse(path).path
        path = posixpath.normpath(urllib.parse.unquote(path))
        words = [w for w in path.split("/") if w]
        resolved = Path(self.directory)
        for word in words:
            resolved = resolved / word
        return str(resolved)


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main():
    server = ReusableHTTPServer((HOST, PORT), ViewerHandler)
    print(f"Serving viewer on http://{HOST}:{PORT}/viewer.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
