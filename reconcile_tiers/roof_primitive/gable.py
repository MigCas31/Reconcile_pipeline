"""Gable primitive: detection + synthesis.

A gable roof has two sloped surfaces meeting at a ridge. The ridge runs
parallel to the wing's long axis; the eaves run along the wing's long
sides; the gable ends sit on the short sides of the wing footprint with
triangular wall tops peaking at the ridge.

Detection signals (from observation):
- The wing's room walls split into two height families:
    * eave walls — flat-top walls at the lower height (the long sides)
    * gable-end walls — walls whose top *slopes* up to a peak (the short
      sides). Their max top y is the ridge height.
- The wing's oblique wall segments cluster into two opposing azimuth
  families perpendicular to the ridge axis.
- The wing footprint is rectangular and axis-aligned. The long axis is
  the ridge axis.

Synthesis (from params):
- Two rectangles, each from one eave (eave_y) up to the ridge (ridge_y).
  Plane is constructed analytically from the two-point eave + the parallel
  ridge — corners are exactly on the plane by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import Polygon

from reconcile_tiers._core.plane import Plane
from reconcile_tiers.extract.building import BuildingModel
from reconcile_tiers.roof.roof import ObliqueSurface, RoofCluster
from reconcile_tiers.roof_primitive.types import GableParams

# Wall is "flat-top" if its top edge spans less than this many meters in y.
EAVE_FLAT_TOL_M = 0.10
# Wall is "gable-end" sloped-top if its top y range exceeds this.
GABLE_END_MIN_RISE_M = 0.30
# Min long-axis / short-axis ratio to call the wing rectangular-enough.
WING_ASPECT_MIN = 1.10
# Two oblique segment families count as opposite if their azimuths are
# within this many degrees of being 180° apart.
OPPOSITE_AZIMUTH_TOL_DEG = 25.0
# Minimum perpendicular distance between the two eave lines of a real
# gable. If the splitter produces two eave clusters separated by less than
# this, the wing isn't a true gable (degenerate / narrow extension).
MIN_EAVE_SEPARATION_M = 1.5


@dataclass(frozen=True, slots=True)
class _WallTopology:
    wall_id: str
    y_top_min: float
    y_top_max: float
    xz_endpoints: tuple[tuple[float, float], tuple[float, float]]  # bottom corner pair
    length_xz: float


def _wall_topology(wall) -> _WallTopology | None:
    if len(wall.corners) < 3:
        return None
    ys = [float(c[1]) for c in wall.corners]
    y_bottom = min(ys)
    bottom = [c for c in wall.corners if float(c[1]) <= y_bottom + 0.05]
    top = [c for c in wall.corners if float(c[1]) > y_bottom + 0.05]
    if len(bottom) < 2 or not top:
        return None
    # Pair of XZ-extreme bottom corners ≈ wall direction
    bx = sorted({(round(c[0], 4), round(c[2], 4)) for c in bottom})
    if len(bx) < 2:
        return None
    a, b = bx[0], bx[-1]
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length < 0.2:
        return None
    top_ys = [float(c[1]) for c in top]
    return _WallTopology(
        wall_id=wall.id,
        y_top_min=min(top_ys),
        y_top_max=max(top_ys),
        xz_endpoints=(a, b),
        length_xz=length,
    )


def _wall_axis_math_deg(
    endpoints: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    a, b = endpoints
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


def _angle_diff_mod180(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def _angle_diff_mod360(a: float, b: float) -> float:
    diff = abs((a - b) % 360.0)
    return min(diff, 360.0 - diff)


def classify_gable(
    wing,
    wing_index: int,
    model: BuildingModel,
    *,
    wall_axis_math: float,
) -> GableParams | None:
    """Return GableParams if `wing` looks like a gable, else None.

    Detection rules:
    1. Take all walls of all rooms whose floor-XZ centroid is inside the wing.
    2. Classify each wall:
        - "eave" if its top edge is flat (y range ≤ EAVE_FLAT_TOL_M).
        - "gable_end" if its top edge rises by ≥ GABLE_END_MIN_RISE_M.
    3. Need ≥ 2 eaves AND ≥ 2 gable-ends.
    4. Eaves must run along one of the building axes; gable-ends along the
       perpendicular. Picks the longer family as eaves.
    5. eave_y = median y_top of eaves. ridge_y = median y_top_max of
       gable-ends.
    6. Build the ridge: midline between the two eave-line clusters, along
       the eave axis, at y=ridge_y.
    """
    wing_poly = wing.polygon
    wing_rooms = []
    for room in model.rooms:
        if len(room.floor_polygon) < 3:
            continue
        room_xz = Polygon([(p[0], p[2]) for p in room.floor_polygon])
        if not wing_poly.intersects(room_xz):
            continue
        if wing_poly.intersection(room_xz).area / room_xz.area < 0.5:
            continue
        wing_rooms.append(room)
    if not wing_rooms:
        return None

    room_apex: dict[int, float] = {}
    for room in wing_rooms:
        apex = room.ceiling_ridge_height
        if apex is None and room.walls_computed:
            wall_tops = [
                max(float(c[1]) for c in w.corners)
                for w in room.walls_computed
                if len(w.corners) >= 3
            ]
            apex = max(wall_tops) if wall_tops else None
        if apex is not None:
            room_apex[room.index] = float(apex)
    if room_apex:
        max_apex = max(room_apex.values())
        active_rooms = [
            room
            for room in wing_rooms
            if room_apex.get(room.index, float("-inf")) >= max_apex - 0.35
        ]
    else:
        max_story = max(room.story for room in wing_rooms)
        active_rooms = [room for room in wing_rooms if room.story == max_story]

    candidate_walls: list[_WallTopology] = []
    story_votes: dict[int, int] = {}
    for room in active_rooms:
        story_votes[room.story] = story_votes.get(room.story, 0) + 1
        for wall in room.walls_computed:
            topo = _wall_topology(wall)
            if topo is None:
                continue
            candidate_walls.append(topo)
    if not candidate_walls:
        return None
    dominant_story = (
        max(story_votes, key=lambda s: (story_votes[s], s)) if story_votes else 0
    )

    eaves: list[_WallTopology] = []
    gable_ends: list[_WallTopology] = []
    for w in candidate_walls:
        rise = w.y_top_max - w.y_top_min
        if rise <= EAVE_FLAT_TOL_M:
            eaves.append(w)
        elif rise >= GABLE_END_MIN_RISE_M:
            gable_ends.append(w)
    if len(eaves) < 2 or len(gable_ends) < 2:
        return None

    # Group eaves by axis direction — should land on building wall_axis_math
    # or wall_axis_math + 90. Pick the family with the longest total length.
    families: dict[int, list[_WallTopology]] = {0: [], 1: []}
    for w in eaves:
        wall_axis = _wall_axis_math_deg(w.xz_endpoints)
        if _angle_diff_mod180(wall_axis, wall_axis_math) <= 25.0:
            families[0].append(w)
        elif _angle_diff_mod180(wall_axis, (wall_axis_math + 90.0) % 180.0) <= 25.0:
            families[1].append(w)
    eave_family = max(families.values(), key=lambda fs: sum(w.length_xz for w in fs))
    if len(eave_family) < 2:
        return None
    eave_axis_math = sum(
        _wall_axis_math_deg(w.xz_endpoints) for w in eave_family
    ) / len(eave_family)

    # Gable ends should run perpendicular to eaves — sanity check.
    perp_axis = (eave_axis_math + 90.0) % 180.0
    perp_ends = [
        w
        for w in gable_ends
        if _angle_diff_mod180(_wall_axis_math_deg(w.xz_endpoints), perp_axis) <= 25.0
    ]
    if len(perp_ends) < 2:
        return None

    eave_y = float(sorted(w.y_top_min for w in eave_family)[len(eave_family) // 2])
    ridge_y = float(sorted(w.y_top_max for w in perp_ends)[len(perp_ends) // 2])
    if ridge_y - eave_y < GABLE_END_MIN_RISE_M:
        return None

    # Pick dominant_story by matching the gable's eave height to per-room
    # ceiling heights — the gable belongs to the story whose top reaches it.
    # Rooms can fall under the wing on multiple stories (e.g., a multi-story
    # building with a gable on the top story); without this match the simple
    # vote-by-room-count picks an arbitrary story (often the ground floor)
    # and downstream painters reject the slopes via the `room.story !=
    # surface.dominant_story` check.
    matching_story_votes: dict[int, int] = {}
    for room in active_rooms:
        # Score: room contributes if its apex (or wall-top max) reaches the
        # gable ridge. Matching against eave_y pulls in lower storeys under
        # the same footprint; ridge_y localizes the primitive to the room
        # that actually contains the gable pair.
        apex = room.ceiling_ridge_height
        if apex is None and room.walls_computed:
            wall_tops = [
                max(float(c[1]) for c in w.corners)
                for w in room.walls_computed
                if len(w.corners) >= 3
            ]
            apex = max(wall_tops) if wall_tops else None
        if apex is None:
            continue
        if apex >= ridge_y - 0.3:
            matching_story_votes[room.story] = (
                matching_story_votes.get(room.story, 0) + 1
            )
    if matching_story_votes:
        dominant_story = max(
            matching_story_votes, key=lambda s: (matching_story_votes[s], s)
        )

    eave_a, eave_b = _split_eaves_into_two_lines(eave_family, eave_axis_math)
    if eave_a is None or eave_b is None:
        return None

    # Sanity gate: eave_a and eave_b must lie on opposite sides of the ridge —
    # i.e. their midpoints must be separated perpendicular to the ridge axis
    # by more than `MIN_EAVE_SEPARATION_M`. Without this check, a wing whose
    # eave-flagged walls all cluster at one short-axis projection (e.g. a
    # narrow extension where the splitter's midpoint cut produces two
    # near-coincident sub-clusters) yields a half_width near zero and synth
    # slopes inclined ~85-90° (effectively vertical, not a roof).
    a_mid = ((eave_a[0][0] + eave_a[1][0]) / 2.0, (eave_a[0][1] + eave_a[1][1]) / 2.0)
    b_mid = ((eave_b[0][0] + eave_b[1][0]) / 2.0, (eave_b[0][1] + eave_b[1][1]) / 2.0)
    perp_rad = math.radians(eave_axis_math + 90.0)
    nx, nz = math.cos(perp_rad), math.sin(perp_rad)
    eave_separation = abs((b_mid[0] - a_mid[0]) * nx + (b_mid[1] - a_mid[1]) * nz)
    if eave_separation < MIN_EAVE_SEPARATION_M:
        return None

    # The ridge runs parallel to the observed eave walls. Earlier versions
    # snapped this to the wing long axis, but real scanned rooms can have a
    # longer gable-end span than eave span; in that case wing-long snapping
    # collapses the two eaves onto the ridge and synthesises near-vertical
    # "roof" planes. The eave family is the direct physical signal here.
    ridge_axis_math = eave_axis_math % 180.0
    ridge = _ridge_along_axis(eave_a, eave_b, ridge_axis_math)
    # `_ridge_along_axis` may degenerate if eaves don't span perpendicular
    # to `ridge_axis_math`; fall back to the eave-derived ridge.
    if ridge is None:
        ridge = _ridge_from_eaves(eave_a, eave_b)
        ridge_axis_math = (
            math.degrees(
                math.atan2(ridge[1][1] - ridge[0][1], ridge[1][0] - ridge[0][0])
            )
            % 180.0
        )

    return GableParams(
        wing_index=wing_index,
        eave_y=eave_y,
        ridge_y=ridge_y,
        ridge_axis_math_deg=ridge_axis_math,
        eave_a_xz=eave_a,
        eave_b_xz=eave_b,
        ridge_xz=ridge,
        dominant_story=dominant_story,
    )


def _split_eaves_into_two_lines(
    eaves: list[_WallTopology], axis_math: float
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]] | None,
    tuple[tuple[float, float], tuple[float, float]] | None,
]:
    """Eaves run along `axis_math`. Project each eave's centroid onto the
    perpendicular axis; cluster into two groups (the two parallel eave
    lines on opposite sides of the ridge).
    """
    perp_rad = math.radians(axis_math + 90.0)
    nx, nz = math.cos(perp_rad), math.sin(perp_rad)
    projections = []
    for w in eaves:
        cx = (w.xz_endpoints[0][0] + w.xz_endpoints[1][0]) / 2.0
        cz = (w.xz_endpoints[0][1] + w.xz_endpoints[1][1]) / 2.0
        projections.append((cx * nx + cz * nz, w))
    projections.sort()
    if len(projections) < 2:
        return None, None
    # Two clusters: split at midpoint of projection range.
    p_min, p_max = projections[0][0], projections[-1][0]
    if p_max - p_min < 1.0:
        return None, None
    midpoint = (p_min + p_max) / 2.0
    side_a = [w for p, w in projections if p < midpoint]
    side_b = [w for p, w in projections if p >= midpoint]
    if not side_a or not side_b:
        return None, None
    line_a = _merge_eave_segments(side_a)
    line_b = _merge_eave_segments(side_b)
    return line_a, line_b


def _merge_eave_segments(
    walls: list[_WallTopology],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the convex extent of all wall endpoints projected on the
    eave axis."""
    pts = [pt for w in walls for pt in w.xz_endpoints]
    # Pick the two pts that are farthest apart.
    best = (pts[0], pts[0], 0.0)
    for i, p in enumerate(pts):
        for q in pts[i + 1 :]:
            d = math.hypot(q[0] - p[0], q[1] - p[1])
            if d > best[2]:
                best = (p, q, d)
    return (best[0], best[1])


def _ridge_along_axis(
    eave_a: tuple[tuple[float, float], tuple[float, float]],
    eave_b: tuple[tuple[float, float], tuple[float, float]],
    axis_math_deg: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Build a ridge line that runs along `axis_math_deg`, centred between
    the two eaves, with extent matching the eaves' projection onto that
    axis. Returns None if the eaves' projections onto the axis collapse.
    """
    angle = math.radians(axis_math_deg)
    dx, dz = math.cos(angle), math.sin(angle)
    a_mid = ((eave_a[0][0] + eave_a[1][0]) / 2.0, (eave_a[0][1] + eave_a[1][1]) / 2.0)
    b_mid = ((eave_b[0][0] + eave_b[1][0]) / 2.0, (eave_b[0][1] + eave_b[1][1]) / 2.0)
    ridge_mid = ((a_mid[0] + b_mid[0]) / 2.0, (a_mid[1] + b_mid[1]) / 2.0)
    # Project all eave endpoints onto the axis and take the extent.
    projections = [pt[0] * dx + pt[1] * dz for eave in (eave_a, eave_b) for pt in eave]
    extent = max(projections) - min(projections)
    if extent < 1.0:
        return None
    half = extent / 2.0
    mid_proj = ridge_mid[0] * dx + ridge_mid[1] * dz
    p1 = (
        ridge_mid[0]
        + dx * (-mid_proj - 0.0 - half + (mid_proj + half - mid_proj))
        - dx * half,
        ridge_mid[1] + dz * 0.0 - dz * half,
    )
    # Simpler: just place ridge endpoints at ridge_mid ± axis_dir * half.
    p1 = (ridge_mid[0] - dx * half, ridge_mid[1] - dz * half)
    p2 = (ridge_mid[0] + dx * half, ridge_mid[1] + dz * half)
    return (p1, p2)


def _ridge_from_eaves(
    eave_a: tuple[tuple[float, float], tuple[float, float]],
    eave_b: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Ridge = midline between eave_a and eave_b, projected to share their
    direction. Endpoints chosen so the ridge spans the same long-axis
    extent as the eaves.
    """
    # Midpoint of each eave's XZ.
    a_mid = ((eave_a[0][0] + eave_a[1][0]) / 2.0, (eave_a[0][1] + eave_a[1][1]) / 2.0)
    b_mid = ((eave_b[0][0] + eave_b[1][0]) / 2.0, (eave_b[0][1] + eave_b[1][1]) / 2.0)
    ridge_mid = ((a_mid[0] + b_mid[0]) / 2.0, (a_mid[1] + b_mid[1]) / 2.0)
    # Direction along the eave (use the longer eave for orientation).
    if math.hypot(
        eave_a[1][0] - eave_a[0][0], eave_a[1][1] - eave_a[0][1]
    ) >= math.hypot(eave_b[1][0] - eave_b[0][0], eave_b[1][1] - eave_b[0][1]):
        dx, dz = eave_a[1][0] - eave_a[0][0], eave_a[1][1] - eave_a[0][1]
    else:
        dx, dz = eave_b[1][0] - eave_b[0][0], eave_b[1][1] - eave_b[0][1]
    length = math.hypot(dx, dz) / 2.0
    nx, nz = dx / (2.0 * length), dz / (2.0 * length)
    p1 = (ridge_mid[0] - nx * length, ridge_mid[1] - nz * length)
    p2 = (ridge_mid[0] + nx * length, ridge_mid[1] + nz * length)
    return (p1, p2)


def synthesise_gable(params: GableParams) -> list[ObliqueSurface]:
    """Construct the two slopes parametrically as planar rectangles.

    Both slopes are built from *canonical* eaves derived from the ridge
    line — eave_a' = ridge offset by perpendicular*half_width, eave_b' =
    ridge offset by -perpendicular*half_width. This guarantees the two
    slopes are perfect mirror images about the ridge: their plane normals
    have a_a + a_b = 0 and c_a + c_b = 0 exactly. Downstream classifiers
    (notably `classify_oblique_roof`'s ridge-from-plane-pair formula) rely
    on this symmetry; using the raw observed eave_a / eave_b corners
    introduced ~0.05-0.10 of asymmetry from wall-snap noise that pushed
    the formula outside its plausibility tolerance and caused gable
    buildings to misclassify as `complex`.

    The original observed eave footprint stays in `params.eave_a_xz` /
    `eave_b_xz` for downstream consumers that need the actual wall trace.
    """
    ridge = params.ridge_xz
    rdx = ridge[1][0] - ridge[0][0]
    rdz = ridge[1][1] - ridge[0][1]
    ridge_len = math.hypot(rdx, rdz)
    if ridge_len < 1e-9:
        return []
    rdx, rdz = rdx / ridge_len, rdz / ridge_len  # unit ridge direction

    # Compute half_width from the observed eaves: distance from ridge midpoint
    # to each eave's centroid, projected onto the perpendicular.
    perp_x, perp_z = -rdz, rdx
    ridge_mid = ((ridge[0][0] + ridge[1][0]) / 2.0, (ridge[0][1] + ridge[1][1]) / 2.0)
    eave_a_mid = (
        (params.eave_a_xz[0][0] + params.eave_a_xz[1][0]) / 2.0,
        (params.eave_a_xz[0][1] + params.eave_a_xz[1][1]) / 2.0,
    )
    eave_b_mid = (
        (params.eave_b_xz[0][0] + params.eave_b_xz[1][0]) / 2.0,
        (params.eave_b_xz[0][1] + params.eave_b_xz[1][1]) / 2.0,
    )
    half_a = abs(
        (eave_a_mid[0] - ridge_mid[0]) * perp_x
        + (eave_a_mid[1] - ridge_mid[1]) * perp_z
    )
    half_b = abs(
        (eave_b_mid[0] - ridge_mid[0]) * perp_x
        + (eave_b_mid[1] - ridge_mid[1]) * perp_z
    )
    half_width = (half_a + half_b) / 2.0
    if half_width < 1e-9:
        return []

    # Canonical eave endpoints: ridge endpoints offset by ±perpendicular*half_width.
    # eave_a sits on the +perp side, eave_b on the -perp side.
    eave_a_canon = (
        (ridge[0][0] + perp_x * half_width, ridge[0][1] + perp_z * half_width),
        (ridge[1][0] + perp_x * half_width, ridge[1][1] + perp_z * half_width),
    )
    eave_b_canon = (
        (ridge[0][0] - perp_x * half_width, ridge[0][1] - perp_z * half_width),
        (ridge[1][0] - perp_x * half_width, ridge[1][1] - perp_z * half_width),
    )

    surfaces: list[ObliqueSurface] = []
    for eave_canon, ridge_xz_ordered in (
        (eave_a_canon, (ridge[0], ridge[1])),
        (eave_b_canon, (ridge[0], ridge[1])),
    ):
        corners = [
            [float(eave_canon[0][0]), float(params.eave_y), float(eave_canon[0][1])],
            [float(eave_canon[1][0]), float(params.eave_y), float(eave_canon[1][1])],
            [
                float(ridge_xz_ordered[1][0]),
                float(params.ridge_y),
                float(ridge_xz_ordered[1][1]),
            ],
            [
                float(ridge_xz_ordered[0][0]),
                float(params.ridge_y),
                float(ridge_xz_ordered[0][1]),
            ],
        ]
        plane = _plane_through_three(corners[0], corners[1], corners[2])
        slope_az = math.degrees(math.atan2(-plane.a, -plane.c)) % 360.0
        incl = math.degrees(math.atan2(math.hypot(plane.a, plane.c), abs(plane.b)))
        ref_pt = [
            sum(c[0] for c in corners) / 4.0,
            sum(c[1] for c in corners) / 4.0,
            sum(c[2] for c in corners) / 4.0,
        ]
        surfaces.append(
            ObliqueSurface(
                corners=corners,
                plane=plane,
                cluster=RoofCluster(
                    segments=[],
                    avg_incl=incl,
                    avg_azimuth=slope_az,
                    ref_pt=ref_pt,
                ),
                dominant_story=params.dominant_story,
                ridge=_ridge_dict_for(params),
            )
        )
    _validate_synthesised_surfaces(surfaces, params)
    return surfaces


# Hard tolerances for synthesis-time validation. Tighter than downstream
# audit thresholds — synthesis is supposed to produce *exact* geometry.
_SYNTH_PLANARITY_TOL_M = 1e-3
_SYNTH_Y_SPAN_TOL_M = 1e-3


class PrimitiveSynthesisError(AssertionError):
    """Raised when a primitive synthesiser produces broken geometry.

    These should never happen in production — they indicate a bug in the
    constructor (non-planar quad, wrong y span, wrong corner ordering, ...).
    """


def _validate_synthesised_surfaces(
    surfaces: list[ObliqueSurface], params: GableParams
) -> None:
    """Hard invariants for gable synthesis. Run inside `synthesise_gable`
    before returning so any broken construction fails loud at the source.

    Checks:
    - Each surface's corners lie on its declared plane within
      `_SYNTH_PLANARITY_TOL_M` (catches non-planar quads — e.g. eave/ridge
      length mismatch).
    - min(corner.y) == params.eave_y within `_SYNTH_Y_SPAN_TOL_M`.
    - max(corner.y) == params.ridge_y within `_SYNTH_Y_SPAN_TOL_M`
      (catches synthesisers that accidentally build a partial slope).
    """
    for idx, s in enumerate(surfaces):
        a, b, c, d = s.plane.a, s.plane.b, s.plane.c, s.plane.d
        for ci, p in enumerate(s.corners):
            res = abs(a * p[0] + b * p[1] + c * p[2] - d)
            if res > _SYNTH_PLANARITY_TOL_M:
                raise PrimitiveSynthesisError(
                    f"gable surface {idx} corner {ci} not on plane: "
                    f"residual={res:.6f} m > tol={_SYNTH_PLANARITY_TOL_M}. "
                    f"params.wing_index={params.wing_index}"
                )
        ys = [p[1] for p in s.corners]
        if abs(min(ys) - params.eave_y) > _SYNTH_Y_SPAN_TOL_M:
            raise PrimitiveSynthesisError(
                f"gable surface {idx} min_y={min(ys):.4f} != eave_y="
                f"{params.eave_y:.4f}. params.wing_index={params.wing_index}"
            )
        if abs(max(ys) - params.ridge_y) > _SYNTH_Y_SPAN_TOL_M:
            raise PrimitiveSynthesisError(
                f"gable surface {idx} max_y={max(ys):.4f} != ridge_y="
                f"{params.ridge_y:.4f}. params.wing_index={params.wing_index}"
            )


def _plane_through_three(p0: list[float], p1: list[float], p2: list[float]) -> Plane:
    """Plane through three (non-collinear) points. Normal points up
    (b > 0) by convention so downstream code sees a roof-like normal."""
    u = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    v = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    a = u[1] * v[2] - u[2] * v[1]
    b = u[2] * v[0] - u[0] * v[2]
    c = u[0] * v[1] - u[1] * v[0]
    norm = math.sqrt(a * a + b * b + c * c)
    if norm < 1e-9:
        return Plane(a=0.0, b=1.0, c=0.0, d=float(p0[1]))
    a, b, c = a / norm, b / norm, c / norm
    if b < 0:
        a, b, c = -a, -b, -c
    d = a * p0[0] + b * p0[1] + c * p0[2]
    return Plane(a=float(a), b=float(b), c=float(c), d=float(d))


def _ridge_dict_for(params: GableParams) -> dict[str, float]:
    """Build the ridge dict in the same shape as `obliques._ridge_for`,
    so downstream code that reads `surface.ridge` keeps working.
    """
    angle = math.radians(params.ridge_axis_math_deg + 90.0)
    rx = math.cos(angle)
    rz = math.sin(angle)
    proj_a = params.ridge_xz[0][0] * rx + params.ridge_xz[0][1] * rz
    proj_b = params.ridge_xz[1][0] * rx + params.ridge_xz[1][1] * rz
    return {
        "x": float(rx),
        "z": float(rz),
        "min": float(min(proj_a, proj_b)),
        "max": float(max(proj_a, proj_b)),
    }
