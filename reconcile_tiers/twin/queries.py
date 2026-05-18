"""Per-(x,z) plane classifier — the canonical answer to "which plane
is the ceiling/roof at this point?" given a constructed `Twin`.

The unified rule (derived from the corpus pattern catalogue, see
`.context/digital_twin_walkthrough/per_pattern_priors.md`):

  ceiling_y_at(x, z) = MIN over candidate planes whose XZ polygon
                       contains (x, z), with Y > floor_y.
  roof_y_at(x, z)    = MAX over the same set without the floor floor.

Polygon-containment gates the candidate set; the min/max picks among
those gated. No scan-precision tolerance, no magnitude threshold —
only `FLOAT_EPS` to enforce "above floor".

Returns `None` when no candidate covers (x, z): a real gap that
should surface on the residual stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reconcile_tiers.payload.schema import Plane
from reconcile_tiers.twin._geometry import FLOAT_EPS
from reconcile_tiers.twin.types import (
    Ceiling,
    CeilingKind,
    RoofSurface,
    Room,
    Twin,
    Wing,
)

QueryKind = Literal["ceiling", "roof"]


@dataclass(frozen=True, slots=True)
class Contender:
    """One candidate plane considered at a query point."""

    primitive_id: str
    y: float
    plane: Plane
    polygon_area_xz: float
    evidence_count: int
    source: Literal["scan", "computed", "mixed"]


@dataclass(frozen=True, slots=True)
class ResolvedY:
    """Result of `ceiling_y_at` / `roof_y_at`."""

    y: float
    chosen_id: str
    contender_ids: tuple[str, ...]
    reason: str


def ceiling_y_at(twin: Twin, x: float, z: float) -> ResolvedY | None:
    """Pick the canonical ceiling Y at (x, z).

    Candidates are pooled at the WING level — every room's Ceiling
    primitive (or its parts) plus the wing's RoofSurfaces all
    compete. A plane belongs to a building part, not to a single
    room; a gable's slope spans many rooms simultaneously, and the
    classifier honours that by treating planes as global within their
    wing. Polygon-containment gates the candidate set; min-Y above
    the host room's floor picks the winner.
    """
    wing = _wing_at(twin, x, z)
    if wing is None:
        return None
    floor_y = _floor_y_at(wing, x, z)
    if floor_y is None:
        return None
    pool = _wing_candidate_pool(wing)
    contenders = _filter_contenders(pool, x, z)
    if not contenders:
        return None
    valid: list[tuple[float, dict]] = []
    for cont in contenders:
        y = _plane_y_at(cont["plane"], x, z)
        if y is None or y <= floor_y + FLOAT_EPS:
            continue
        valid.append((y, cont))
    if not valid:
        return None
    valid.sort(key=lambda yv: yv[0])
    chosen_y, chosen_cont = valid[0]
    reason = "single_candidate" if len(valid) == 1 else "min_y_above_floor"
    return ResolvedY(
        y=chosen_y,
        chosen_id=chosen_cont["id"],
        contender_ids=tuple(cont["id"] for _, cont in valid),
        reason=reason,
    )


def roof_y_at(twin: Twin, x: float, z: float) -> ResolvedY | None:
    """Pick the canonical roof envelope Y at (x, z).

    Same wing-level candidate pool as `ceiling_y_at`; max-Y picks the
    topmost upward-facing plane covering (x, z). For flat-roof
    buildings with no oblique RoofSurface, the topmost flat ceiling
    IS the roof; max-Y handles that uniformly.
    """
    wing = _wing_at(twin, x, z)
    if wing is None:
        return None
    pool = _wing_candidate_pool(wing)
    contenders = _filter_contenders(pool, x, z)
    if not contenders:
        return None
    valid: list[tuple[float, dict]] = []
    for cont in contenders:
        y = _plane_y_at(cont["plane"], x, z)
        if y is None:
            continue
        valid.append((y, cont))
    if not valid:
        return None
    valid.sort(key=lambda yv: -yv[0])
    chosen_y, chosen_cont = valid[0]
    reason = "single_candidate" if len(valid) == 1 else "max_y"
    return ResolvedY(
        y=chosen_y,
        chosen_id=chosen_cont["id"],
        contender_ids=tuple(cont["id"] for _, cont in valid),
        reason=reason,
    )


def _wing_candidate_pool(wing: Wing) -> list[dict]:
    """The set of architectural planes available within the wing.

    Architectural rule: planes belong to building parts, not to
    rooms. **Slope planes are wing-level**: a gable's south slope
    is one plane that spans every room beneath it. The wing's
    `RoofSurface` (the cluster of co-facing oblique evidence) IS
    that canonical plane. The per-room oblique sub-Ceilings are
    *evidence* that contributed to the cluster, not independent
    planes — including them in the candidate pool would create
    phantom duplicates.

    **Flat lids are room-level**: a kitchen at 2.5 m and a
    bedroom at 2.4 m are different planes, one per room. They
    stay as candidates.

    So the pool is: every flat ceiling primitive (full or composite
    flat-part) + every wing RoofSurface. No per-room obliques.
    """
    pool: list[dict] = []
    for story in wing.stories:
        for room in story.rooms:
            ceiling = room.ceiling
            if ceiling.kind is CeilingKind.FLAT and ceiling.plane is not None:
                pool.append(_ceiling_to_dict(ceiling))
            elif ceiling.kind is CeilingKind.COMPOSITE:
                for part in ceiling.parts:
                    if part.kind is CeilingKind.FLAT and part.plane is not None:
                        pool.append(_ceiling_to_dict(part))
            # CeilingKind.OBLIQUE (single-plane sloped rooms) and
            # composite OBLIQUE parts are NOT pooled here — the wing's
            # RoofSurface for that architectural face represents the
            # canonical plane.
    if wing.roof is not None:
        for surface in wing.roof.surfaces:
            polygon = [(c.x, c.z) for c in surface.polygon]
            pool.append(
                {
                    "id": surface.id,
                    "plane": surface.plane,
                    "plane_kind": CeilingKind.OBLIQUE,
                    "polygon": polygon,
                    "polygon_area_xz": _polygon_area_xz(polygon),
                    "evidence_count": len(surface.evidence),
                    "source": _evidence_source_summary(surface.evidence),
                }
            )
    return pool


def _filter_contenders(pool: list[dict], x: float, z: float) -> list[dict]:
    return [c for c in pool if _contains_xz(c["polygon"], x, z)]


def _floor_y_at(wing: Wing, x: float, z: float) -> float | None:
    """Find the floor Y of the room containing (x, z). Falls back to
    the lowest room floor in the wing if no room contains the
    point — useful when (x, z) is just outside any room polygon but
    inside the wing footprint."""
    matches: list[tuple[float, float]] = []
    floor_ys: list[float] = []
    for story in wing.stories:
        for room in story.rooms:
            poly = [(c.x, c.z) for c in room.floor.polygon]
            if _contains_xz(poly, x, z):
                cx = sum(c.x for c in room.floor.polygon) / len(room.floor.polygon)
                cz = sum(c.z for c in room.floor.polygon) / len(room.floor.polygon)
                d2 = (cx - x) ** 2 + (cz - z) ** 2
                matches.append((d2, float(room.floor.polygon[0].y)))
            floor_ys.append(float(room.floor.polygon[0].y))
    if matches:
        matches.sort(key=lambda dr: dr[0])
        return matches[0][1]
    if floor_ys:
        return min(floor_ys)
    return None


def contenders_at(
    twin: Twin, x: float, z: float, kind: QueryKind
) -> tuple[Contender, ...]:
    """Wing-level candidate pool, filtered by polygon containment.
    Same pool for ceiling and roof; only the sort order differs."""
    wing = _wing_at(twin, x, z)
    if wing is None:
        return ()
    pool = _wing_candidate_pool(wing)
    contenders = _filter_contenders(pool, x, z)
    out: list[Contender] = []
    if kind == "ceiling":
        floor_y = _floor_y_at(wing, x, z)
        if floor_y is None:
            return ()
        for cand in contenders:
            y = _plane_y_at(cand["plane"], x, z)
            if y is None or y <= floor_y + FLOAT_EPS:
                continue
            out.append(_to_contender(cand, y))
        out.sort(key=lambda c: c.y)
        return tuple(out)
    elif kind == "roof":
        for cand in contenders:
            y = _plane_y_at(cand["plane"], x, z)
            if y is None:
                continue
            out.append(_to_contender(cand, y))
        out.sort(key=lambda c: -c.y)
        return tuple(out)
    raise ValueError(f"unknown kind: {kind}")


# ---- internals ----------------------------------------------------


def _room_at(twin: Twin, x: float, z: float) -> Room | None:
    """Find the Room whose Floor polygon contains (x, z). Tie-break by
    nearest centroid when multiple rooms claim the point (rare, due to
    extraction noise on shared walls)."""
    matches: list[tuple[float, Room]] = []
    for wing in twin.building.wings:
        for story in wing.stories:
            for room in story.rooms:
                poly = [(c.x, c.z) for c in room.floor.polygon]
                if not _contains_xz(poly, x, z):
                    continue
                cx = sum(c.x for c in room.floor.polygon) / len(room.floor.polygon)
                cz = sum(c.z for c in room.floor.polygon) / len(room.floor.polygon)
                d2 = (cx - x) ** 2 + (cz - z) ** 2
                matches.append((d2, room))
    if not matches:
        return None
    matches.sort(key=lambda dr: dr[0])
    return matches[0][1]


def _wing_at(twin: Twin, x: float, z: float) -> Wing | None:
    matches: list[tuple[float, Wing]] = []
    for wing in twin.building.wings:
        poly = [(c.x, c.z) for c in wing.footprint]
        if not _contains_xz(poly, x, z):
            continue
        cx = sum(c.x for c in wing.footprint) / len(wing.footprint)
        cz = sum(c.z for c in wing.footprint) / len(wing.footprint)
        d2 = (cx - x) ** 2 + (cz - z) ** 2
        matches.append((d2, wing))
    if not matches:
        return None
    matches.sort(key=lambda dw: dw[0])
    return matches[0][1]


def _top_story_room_at(wing: Wing, x: float, z: float) -> Room | None:
    if not wing.stories:
        return None
    top_idx = max(s.rooms[0].story_index for s in wing.stories if s.rooms)
    for story in wing.stories:
        for room in story.rooms:
            if room.story_index != top_idx:
                continue
            poly = [(c.x, c.z) for c in room.floor.polygon]
            if _contains_xz(poly, x, z):
                return room
    return None


def _ceiling_candidates(room: Room) -> list[dict]:
    """Walk Room.ceiling and return its candidate sub-pieces as dicts
    with `id`, `plane`, `polygon` (XZ tuples), `plane_kind`,
    `evidence_count`, `source`, `polygon_area_xz`."""
    out: list[dict] = []
    ceiling = room.ceiling
    if ceiling.kind in (CeilingKind.FLAT, CeilingKind.OBLIQUE):
        if ceiling.plane is not None:
            out.append(_ceiling_to_dict(ceiling))
    elif ceiling.kind is CeilingKind.COMPOSITE:
        for part in ceiling.parts:
            if part.plane is not None:
                out.append(_ceiling_to_dict(part))
    return out


def _ceiling_to_dict(ceiling: Ceiling) -> dict:
    polygon = [(c.x, c.z) for c in ceiling.polygon]
    return {
        "id": ceiling.id,
        "plane": ceiling.plane,
        "plane_kind": ceiling.kind,
        "polygon": polygon,
        "polygon_area_xz": _polygon_area_xz(polygon),
        "evidence_count": len(ceiling.evidence),
        "source": _evidence_source_summary(ceiling.evidence),
    }


def _to_contender(cand: dict, y: float) -> Contender:
    return Contender(
        primitive_id=cand["id"],
        y=float(y),
        plane=cand["plane"],
        polygon_area_xz=float(cand["polygon_area_xz"]),
        evidence_count=int(cand["evidence_count"]),
        source=cand["source"],
    )


def _surface_to_contender(surface: RoofSurface, y: float) -> Contender:
    polygon = [(c.x, c.z) for c in surface.polygon]
    return Contender(
        primitive_id=surface.id,
        y=float(y),
        plane=surface.plane,
        polygon_area_xz=float(_polygon_area_xz(polygon)),
        evidence_count=len(surface.evidence),
        source=_evidence_source_summary(surface.evidence),
    )


def _evidence_source_summary(
    evidence: tuple,
) -> Literal["scan", "computed", "mixed"]:
    has_scan = any(ev.provenance.kind == "scan" for ev in evidence)
    has_computed = any(ev.provenance.kind == "computed" for ev in evidence)
    if has_scan and has_computed:
        return "mixed"
    if has_scan:
        return "scan"
    return "computed"


def _polygon_area_xz(corners_xz: list[tuple[float, float]]) -> float:
    n = len(corners_xz)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        ax, az = corners_xz[i]
        bx, bz = corners_xz[(i + 1) % n]
        total += ax * bz - bx * az
    return abs(total) * 0.5


def _contains_xz(corners_xz: list[tuple[float, float]], px: float, pz: float) -> bool:
    n = len(corners_xz)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        ax, az = corners_xz[i]
        bx, bz = corners_xz[j]
        if (az > pz) != (bz > pz):
            x_at = ax + (pz - az) * (bx - ax) / (bz - az + 1e-30)
            if px < x_at:
                inside = not inside
        j = i
    return inside


def _plane_y_at(plane: Plane, x: float, z: float) -> float | None:
    """Solve plane.a*x + plane.b*y + plane.c*z + plane.d = 0 for y."""
    if abs(plane.b) < FLOAT_EPS:
        return None  # vertical plane, y is undetermined at (x, z)
    return -(plane.a * x + plane.c * z + plane.d) / plane.b
