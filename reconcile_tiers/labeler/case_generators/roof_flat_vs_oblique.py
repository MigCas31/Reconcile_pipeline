"""Generate labeler cases for the roof flat-vs-oblique decision.

For every ceiling piece in every building's ``tier_payload.json`` whose plane
inclination falls in an ambiguous band, emit a case with two candidate
reconstructions ("flat" and "oblique") plus the ``heuristic_label`` of what
the pipeline currently chose.

Run:

    python -m reconcile_tiers.labeler.case_generators.roof_flat_vs_oblique \\
        --run-id roof-flat-vs-oblique-2026-05-03 \\
        --pipeline-dir pipeline-outputs

Cases land at ``.context/labeler/runs/<run_id>/{meta.json, cases.jsonl}``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from reconcile_tiers.labeler.schema import Case, CaseOption, RunMeta
from reconcile_tiers.labeler.storage import write_run

GENERATOR = "roof_flat_vs_oblique"
GENERATOR_VERSION = "7"
DECISION_TYPE = "roof-flat-vs-oblique"

# Plane-orientation alignment: two ceiling pieces are "aligned" when their
# unit-normal angular separation is below this threshold. 25° is generous
# enough to catch real-roof noise yet small enough to keep different roof
# faces (gables, dormers) out of the candidate set.
ALIGNED_NORMAL_MAX_DEG = 25.0
# Pieces with at least this much XZ overlap area count as "overlapping" --
# i.e. plausibly fragments of one larger face.
MIN_XZ_OVERLAP_M2 = 0.20

# Oblique alternatives: the closest few oblique neighbours, used to project
# the piece's XZ footprint onto each neighbour's plane. Tests the hypothesis
# "this fragment belongs to that adjacent slope".
OBLIQUE_NEIGHBOUR_MAX_XZ_DIST_M = 8.0
OBLIQUE_NEIGHBOUR_MIN_INCL_DEG = 5.0
MAX_NEIGHBOUR_OBLIQUE_OPTIONS = 3
PLANE_SAME_TOL = 1e-3

# A "flat" alternative for an oblique piece is most architecturally plausible
# at the height of a *neighbouring* flat ceiling (e.g. knee-wall-top or
# main-room-ceiling height), not at the median Y of the oblique itself.
NEIGHBOUR_MAX_XZ_DIST_M = 4.0
NEIGHBOUR_MAX_INCL_DEG = 5.0
FLAT_Y_DEDUPE_M = 0.10

# Ambiguous bands: 5-15° straddles the practical flat/oblique boundary used
# downstream; 70-85° is included because the user requested it (mostly empty
# in practice -- see audit summary).
AMBIGUOUS_BANDS = ((5.0, 15.0), (70.0, 85.0))
HEURISTIC_BOUNDARY_DEG = 8.0


def _inclination_deg(plane: dict[str, float]) -> float:
    b = float(plane.get("b", 0.0))
    return math.degrees(math.acos(min(1.0, max(0.0, abs(b)))))


def _in_band(
    value: float, bands: tuple[tuple[float, float], ...]
) -> tuple[float, float] | None:
    for lo, hi in bands:
        if lo < value < hi:
            return (lo, hi)
    return None


def _polygon_area_xz(corners: list[dict[str, float]]) -> float:
    """Signed XZ-projected polygon area; absolute value approximates the
    horizontal projection footprint, which is what matters for flat-vs-oblique."""
    if len(corners) < 3:
        return 0.0
    s = 0.0
    n = len(corners)
    for i in range(n):
        x1, z1 = corners[i]["x"], corners[i]["z"]
        x2, z2 = corners[(i + 1) % n]["x"], corners[(i + 1) % n]["z"]
        s += x1 * z2 - x2 * z1
    return abs(s) * 0.5


def _centroid(corners: list[dict[str, float]]) -> list[float]:
    n = len(corners)
    return [
        sum(c["x"] for c in corners) / n,
        sum(c["y"] for c in corners) / n,
        sum(c["z"] for c in corners) / n,
    ]


def _bounding_box(corners: list[dict[str, float]]) -> tuple[list[float], list[float]]:
    xs = [c["x"] for c in corners]
    ys = [c["y"] for c in corners]
    zs = [c["z"] for c in corners]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def _camera_offset_for(corners: list[dict[str, float]]) -> list[float]:
    lo, hi = _bounding_box(corners)
    span = max(hi[i] - lo[i] for i in range(3))
    d = max(6.0, span * 2.5)
    return [d * 0.7, d * 0.6, d * 0.7]


def _flat_polygon_at(corners: list[dict[str, float]], y: float) -> list[list[float]]:
    """Flatten the piece by keeping XZ corners and snapping Y to a chosen value."""
    if not corners:
        return []
    return [[c["x"], y, c["z"]] for c in corners]


def _piece_centroid_xz(corners: list[dict[str, float]]) -> tuple[float, float]:
    n = len(corners)
    return (sum(c["x"] for c in corners) / n, sum(c["z"] for c in corners) / n)


def _point_in_polygon_xz(
    point: tuple[float, float], poly: list[dict[str, float]]
) -> bool:
    """Ray-casting point-in-polygon on the XZ plane."""
    if len(poly) < 3:
        return False
    x, z = point
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, zi = poly[i]["x"], poly[i]["z"]
        xj, zj = poly[j]["x"], poly[j]["z"]
        intersect = ((zi > z) != (zj > z)) and (
            x < (xj - xi) * (z - zi) / (zj - zi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _find_containing_room(
    *,
    piece: dict[str, Any],
    rooms: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the room whose floor polygon contains the piece's XZ centroid."""
    cx, cz = _piece_centroid_xz(piece["corners"])
    for room in rooms:
        for floor in room.get("floor", []) or []:
            corners = floor.get("corners")
            if corners and _point_in_polygon_xz((cx, cz), corners):
                return room
    return None


def _same_room_other_pieces(
    *,
    piece: dict[str, Any],
    all_pieces: list[dict[str, Any]],
    rooms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Other ceiling pieces whose XZ centroid lands in the same room.

    Same-room context is the strongest architectural prior we have: pieces in
    the same room are parts of one ceiling system, so they're prime candidates
    for "should be coplanar with this". Returned without filtering by
    orientation -- the labeler visually judges whether the projection works.
    """
    container = _find_containing_room(piece=piece, rooms=rooms)
    if container is None:
        return []
    container_id = container.get("locator_id")
    if not container_id:
        return []
    out: list[dict[str, Any]] = []
    for q in all_pieces:
        if q.get("locator_id") == piece.get("locator_id"):
            continue
        if not q.get("corners"):
            continue
        q_room = _find_containing_room(piece=q, rooms=rooms)
        if q_room is not None and q_room.get("locator_id") == container_id:
            out.append(q)
    return out


def _piece_mean_y(corners: list[dict[str, float]]) -> float:
    return sum(c["y"] for c in corners) / len(corners)


def _neighbour_flat_y_values(
    *,
    piece: dict[str, Any],
    all_pieces: list[dict[str, Any]],
) -> list[float]:
    """Y heights of nearby (XZ-close) flat ceiling pieces in the same building.

    Neighbour heights are the architecturally meaningful candidates for a
    "flat" alternative -- they match the wall-top of an adjacent room, the
    knee-wall top, or the next bay's ceiling. Sorted ascending."""
    px, pz = _piece_centroid_xz(piece["corners"])
    p_locator = piece["locator_id"]
    ys: list[float] = []
    for q in all_pieces:
        if q.get("locator_id") == p_locator:
            continue
        q_corners = q.get("corners")
        q_plane = q.get("plane")
        if not q_corners or not q_plane:
            continue
        if _inclination_deg(q_plane) > NEIGHBOUR_MAX_INCL_DEG:
            continue
        qx, qz = _piece_centroid_xz(q_corners)
        if math.hypot(qx - px, qz - pz) > NEIGHBOUR_MAX_XZ_DIST_M:
            continue
        ys.append(_piece_mean_y(q_corners))
    ys.sort()
    return ys


def _dedupe_y(values: list[float], tol: float = FLAT_Y_DEDUPE_M) -> list[float]:
    out: list[float] = []
    for y in sorted(values):
        if not out or abs(y - out[-1]) > tol:
            out.append(y)
    return out


# Up to three flat candidates: low-neighbour, high-neighbour, piece-midline.
# Ordered low-to-high after dedupe so the strip reads bottom-to-top.
def _flat_candidates(
    *,
    piece: dict[str, Any],
    all_pieces: list[dict[str, Any]],
) -> list[tuple[str, str, float, str]]:
    """Returns list of (option_id, label, y, color) for flat candidates."""
    midline_y = _piece_mean_y(piece["corners"])
    neighbours = _neighbour_flat_y_values(piece=piece, all_pieces=all_pieces)

    candidates: list[tuple[str, float]] = []
    if neighbours:
        candidates.append(("low neighbour", neighbours[0]))
        if (
            len(neighbours) >= 2
            and abs(neighbours[-1] - neighbours[0]) > FLAT_Y_DEDUPE_M
        ):
            candidates.append(("high neighbour", neighbours[-1]))
    candidates.append(("midline", midline_y))

    seen: list[float] = []
    out: list[tuple[str, str, float, str]] = []
    palette = {
        "low neighbour": "#60a5fa",
        "high neighbour": "#3b82f6",
        "midline": "#a5b4fc",
    }
    for tag, y in candidates:
        if any(abs(y - s) <= FLAT_Y_DEDUPE_M for s in seen):
            continue
        seen.append(y)
        out.append(
            (
                f"flat-{tag.replace(' ', '-')}",
                f"Flat ({tag}, y={y:.2f})",
                y,
                palette[tag],
            )
        )
    return out


def _oblique_polygon(corners: list[dict[str, float]]) -> list[list[float]]:
    return [[c["x"], c["y"], c["z"]] for c in corners]


def _project_xz_to_plane(
    corners: list[dict[str, float]],
    plane: dict[str, float],
) -> list[list[float]] | None:
    """Project each XZ corner onto plane.

    Convention in tier_payload.json: ``a·x + b·y + c·z = d`` (verified by
    plugging a known piece's corners back into the equation). Solve for y.
    """
    a = float(plane.get("a", 0.0))
    b = float(plane.get("b", 0.0))
    c = float(plane.get("c", 0.0))
    d = float(plane.get("d", 0.0))
    if abs(b) < 1e-6:
        # Vertical plane has no y(x,z) -- can't render the candidate footprint.
        return None
    out: list[list[float]] = []
    for corner in corners:
        x = corner["x"]
        z = corner["z"]
        y = (d - a * x - c * z) / b
        out.append([x, y, z])
    return out


def _planes_close(p: dict[str, float], q: dict[str, float]) -> bool:
    return all(
        abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) < PLANE_SAME_TOL
        for k in ("a", "b", "c", "d")
    )


def _normal_angle_deg(p: dict[str, float], q: dict[str, float]) -> float:
    """Angle between plane unit normals, treating +/-n as equivalent."""
    pn = (float(p.get("a", 0.0)), float(p.get("b", 0.0)), float(p.get("c", 0.0)))
    qn = (float(q.get("a", 0.0)), float(q.get("b", 0.0)), float(q.get("c", 0.0)))
    pl = math.sqrt(sum(c * c for c in pn))
    ql = math.sqrt(sum(c * c for c in qn))
    if pl < 1e-9 or ql < 1e-9:
        return 180.0
    cos = (pn[0] * qn[0] + pn[1] * qn[1] + pn[2] * qn[2]) / (pl * ql)
    return math.degrees(math.acos(min(1.0, max(0.0, abs(cos)))))


def _xz_polygon(corners: list[dict[str, float]]):
    # Local import keeps the dependency obvious at call site and survives
    # formatter import-shuffling (this module is small enough that a
    # top-level shapely import gets stripped as "unused at module level").
    from shapely.geometry import Polygon

    pts = [(c["x"], c["z"]) for c in corners]
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.area <= 0:
        return None
    return poly


def _coplanar_oblique_candidates(
    *,
    piece: dict[str, Any],
    all_pieces: list[dict[str, Any]],
    rooms: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any], dict[str, float]]]:
    """Strong "should-be-coplanar" candidates, ranked by tier.

    Returns ``(kind_label, neighbour_piece, metrics)``. ``metrics`` carries
    the angular separation, XZ overlap area, and centroid distance.

    Tier 0 -- same-room oblique pieces. Strongest architectural prior: room
    context says these are parts of one ceiling system. No orientation
    filter -- the labeler visually decides whether the projection works.

    Tier 1 -- XZ-overlapping AND aligned plane normal (< 25°). The "fragment
    of one bigger face" hypothesis when no room mate exists.

    Tier 2 -- nearest aligned neighbour (no overlap required), used to fill
    the option strip if Tier 0 + Tier 1 leave room.
    """
    p_locator = piece["locator_id"]
    p_plane = piece.get("plane") or {}
    px, pz = _piece_centroid_xz(piece["corners"])
    p_xz = _xz_polygon(piece["corners"])

    def metrics_for(q: dict[str, Any]) -> dict[str, float]:
        q_corners = q["corners"]
        qx, qz = _piece_centroid_xz(q_corners)
        q_xz = _xz_polygon(q_corners)
        overlap = 0.0
        if p_xz is not None and q_xz is not None:
            try:
                overlap = float(p_xz.intersection(q_xz).area)
            except Exception:
                overlap = 0.0
        return {
            "normal_angle_deg": round(
                _normal_angle_deg(q.get("plane") or {}, p_plane), 2
            ),
            "xz_overlap_m2": round(overlap, 3),
            "xz_centroid_dist_m": round(math.hypot(qx - px, qz - pz), 3),
        }

    same_room = _same_room_other_pieces(piece=piece, all_pieces=all_pieces, rooms=rooms)
    same_room_ids = {q["locator_id"] for q in same_room if q.get("locator_id")}

    # Tier 0 -- same-room oblique pieces, no orientation filter.
    tier0: list[tuple[float, float, dict[str, Any], dict[str, float]]] = []
    for q in same_room:
        q_plane = q.get("plane") or {}
        if _inclination_deg(q_plane) < OBLIQUE_NEIGHBOUR_MIN_INCL_DEG:
            continue
        if _planes_close(q_plane, p_plane):
            continue
        m = metrics_for(q)
        # Within Tier 0: prefer the closest, then most-aligned.
        tier0.append((m["xz_centroid_dist_m"], m["normal_angle_deg"], q, m))

    # Tier 1 + Tier 2 -- same as before, but skip pieces already chosen as Tier 0.
    tier1: list[tuple[float, float, dict[str, Any], dict[str, float]]] = []
    tier2: list[tuple[float, float, dict[str, Any], dict[str, float]]] = []
    for q in all_pieces:
        if q.get("locator_id") == p_locator:
            continue
        if q.get("locator_id") in same_room_ids:
            continue
        q_corners = q.get("corners")
        q_plane = q.get("plane")
        if not q_corners or not q_plane:
            continue
        if _inclination_deg(q_plane) < OBLIQUE_NEIGHBOUR_MIN_INCL_DEG:
            continue
        if _planes_close(q_plane, p_plane):
            continue
        angle = _normal_angle_deg(q_plane, p_plane)
        if angle > ALIGNED_NORMAL_MAX_DEG:
            continue
        m = metrics_for(q)
        if m["xz_overlap_m2"] >= MIN_XZ_OVERLAP_M2:
            tier1.append((-m["xz_overlap_m2"], angle, q, m))
        elif m["xz_centroid_dist_m"] <= OBLIQUE_NEIGHBOUR_MAX_XZ_DIST_M:
            tier2.append((m["xz_centroid_dist_m"], angle, q, m))

    tier0.sort(key=lambda t: (t[0], t[1]))
    tier1.sort(key=lambda t: (t[0], t[1]))
    tier2.sort(key=lambda t: (t[0], t[1]))

    out: list[tuple[str, dict[str, Any], dict[str, float]]] = []
    for _, _, q, m in tier0:
        if len(out) >= MAX_NEIGHBOUR_OBLIQUE_OPTIONS:
            break
        out.append(
            (
                (
                    f"same-room piece (Δ{m['normal_angle_deg']}°, "
                    f"{m['xz_centroid_dist_m']} m)"
                ),
                q,
                m,
            )
        )
    for _, _, q, m in tier1:
        if len(out) >= MAX_NEIGHBOUR_OBLIQUE_OPTIONS:
            break
        out.append(
            (
                (
                    f"overlapping piece (Δ{m['normal_angle_deg']}°, "
                    f"ov {m['xz_overlap_m2']} m^2)"
                ),
                q,
                m,
            )
        )
    for _, _, q, m in tier2:
        if len(out) >= MAX_NEIGHBOUR_OBLIQUE_OPTIONS:
            break
        out.append(
            (
                f"aligned neighbour (Δ{m['normal_angle_deg']}°, "
                f"{m['xz_centroid_dist_m']} m)",
                q,
                m,
            )
        )
    return out


def _emit_case(
    *,
    building_uuid: str,
    piece: dict[str, Any],
    all_pieces: list[dict[str, Any]],
    rooms: list[dict[str, Any]],
    case_idx: int,
    band: tuple[float, float],
) -> Case:
    incl = _inclination_deg(piece["plane"])
    corners = piece["corners"]
    area = _polygon_area_xz(corners)
    centroid = _centroid(corners)

    containing_room = _find_containing_room(piece=piece, rooms=rooms)
    flat_cands = _flat_candidates(piece=piece, all_pieces=all_pieces)
    oblique_neighbours = _coplanar_oblique_candidates(
        piece=piece, all_pieces=all_pieces, rooms=rooms
    )

    options: list[CaseOption] = [
        CaseOption(
            id="oblique",
            label="Oblique (own plane)",
            polygons=[_oblique_polygon(corners)],
            color="#f97316",
        ),
    ]
    # Three palette steps so Tier 0 (same-room) reads as the strongest /
    # most saturated, fading through Tier 1, Tier 2.
    neighbour_palette = ("#ea580c", "#fb923c", "#fdba74")
    for i, (kind, neighbour, _metrics) in enumerate(oblique_neighbours):
        projected = _project_xz_to_plane(corners, neighbour["plane"])
        if not projected:
            continue
        # Coplanar hypothesis: render the candidate placed on the neighbour's
        # plane PLUS the neighbour's own polygon (which already lies on that
        # plane). The two polygons should merge into one continuous surface
        # if they really are coplanar.
        neighbour_poly = _oblique_polygon(neighbour["corners"])
        # Strip parens / commas / spaces out of `kind` to keep option ids
        # JSON-friendly; full descriptive label still goes on the card.
        opt_slug = kind.split(" ")[0].replace("(", "").replace(")", "")
        options.append(
            CaseOption(
                id=f"coplanar-{opt_slug}-{i}",
                label=f"Coplanar with {kind}",
                polygons=[projected, neighbour_poly],
                color=neighbour_palette[i] if i < len(neighbour_palette) else "#fed7aa",
            )
        )
    for opt_id, label, y, color in flat_cands:
        options.append(
            CaseOption(
                id=opt_id,
                label=label,
                polygons=[_flat_polygon_at(corners, y)],
                color=color,
            )
        )
    options.append(
        CaseOption(
            id="neither",
            label="Neither (not a real ceiling plane)",
            polygons=[],
            color="#71717a",
        )
    )

    # Heuristic: oblique above the boundary; below it, point at the first
    # available flat option (typically the low-neighbour candidate).
    if incl >= HEURISTIC_BOUNDARY_DEG:
        heuristic = "oblique"
    elif flat_cands:
        heuristic = flat_cands[0][0]
    else:
        heuristic = "oblique"

    return Case(
        case_id=f"{building_uuid}::ceiling-{case_idx}",
        building_uuid=building_uuid,
        decision_type=DECISION_TYPE,
        locator_id=piece["locator_id"],
        options=options,
        camera_target=centroid,
        camera_offset=_camera_offset_for(corners),
        features={
            "inclination_deg": round(incl, 3),
            "band_lo": band[0],
            "band_hi": band[1],
            "area_xz_m2": round(area, 3),
            "vertex_count": len(corners),
            "has_holes": bool(piece.get("holes")),
            "source": piece.get("source"),
            "role": piece.get("role"),
            "support_quality": piece.get("support_quality"),
            "merged_from_count": len(piece.get("merged_from") or []),
            "plane_normal": [
                piece["plane"].get("a", 0.0),
                piece["plane"].get("b", 0.0),
                piece["plane"].get("c", 0.0),
            ],
            "midline_y": round(_piece_mean_y(corners), 3),
            "neighbour_flat_y_count": len(
                _neighbour_flat_y_values(piece=piece, all_pieces=all_pieces)
            ),
            "flat_candidate_count": len(flat_cands),
            "oblique_neighbour_count": len(oblique_neighbours),
            "room_locator_id": containing_room.get("locator_id")
            if containing_room
            else None,
            "room_story": containing_room.get("story") if containing_room else None,
        },
        heuristic_label=heuristic,
        highlight_locators=[containing_room["locator_id"]]
        if containing_room and containing_room.get("locator_id")
        else [],
    )


def generate(pipeline_dir: Path) -> list[Case]:
    cases: list[Case] = []
    for child in sorted(pipeline_dir.iterdir()):
        if not child.is_dir():
            continue
        payload_path = child / "tier_payload.json"
        if not payload_path.exists():
            continue
        try:
            with payload_path.open() as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        uuid = payload.get("uuid") or child.name
        all_pieces = payload.get("ceiling", []) or []
        rooms = payload.get("rooms", []) or []
        per_building_idx = 0
        for piece in all_pieces:
            plane = piece.get("plane")
            corners = piece.get("corners")
            if not plane or not corners or len(corners) < 3:
                continue
            incl = _inclination_deg(plane)
            band = _in_band(incl, AMBIGUOUS_BANDS)
            if band is None:
                continue
            case = _emit_case(
                building_uuid=uuid,
                piece=piece,
                all_pieces=all_pieces,
                rooms=rooms,
                case_idx=per_building_idx,
                band=band,
            )
            cases.append(case)
            per_building_idx += 1
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="run identifier (slug)")
    parser.add_argument(
        "--pipeline-dir",
        default="pipeline-outputs",
        help="root holding <uuid>/tier_payload.json",
    )
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    pipeline_dir = Path(args.pipeline_dir).resolve()
    if not pipeline_dir.is_dir():
        raise SystemExit(f"pipeline-dir not found: {pipeline_dir}")

    cases = generate(pipeline_dir)
    meta = RunMeta(
        run_id=args.run_id,
        decision_type=DECISION_TYPE,
        generator=GENERATOR,
        generator_version=GENERATOR_VERSION,
        created_at=time.time(),
        total_cases=len(cases),
        description=args.description,
    )
    out_dir = write_run(meta, cases)
    print(f"wrote {len(cases)} cases to {out_dir}")


if __name__ == "__main__":
    main()
