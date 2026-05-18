"""Phase A simulation for the notch-aware stitch snap fix.

Scans every building in ``reconcile/buildings_3d.json`` looking for
``stitch_walls`` that run near-parallel to, and offset from, a non-owner
``walls_computed`` in the same story. For each such event, evaluates the
"notch predicate" using the non-owner room's ``floor_polygon``:

- If the non-owner room's floor covers the area between the stitch and the wall
  **outside** the stitch's own along-range but **not inside** it, that is a
  scan-boundary notch -> ``repaired`` (snap would close the gap).
- If the non-owner room's floor covers neither side, it's probably a real void
  (chimney, chase, stair) -> ``unchanged`` (leave alone).
- If a door or window centroid lies in the strip between stitch and the wall,
  flag as ``regressed`` (snapping would potentially collapse a real portal).

Writes ``.context/stitch_snap_simulation.json`` with totals and a sample of
20 snap candidates for viewer sanity-checking.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

_ = Point  # keep formatter from stripping the used-but-at-runtime import

REPO = Path(__file__).resolve().parents[1]
BUILDINGS = REPO / "reconcile" / "buildings_3d.json"
OUT = REPO / ".context" / "stitch_snap_simulation.json"

PARALLEL_DOT_MIN = 0.98
PERP_MIN = 0.4
PERP_MAX = 2.0
ALONG_OVERLAP_MIN = 0.2
NOTCH_SAMPLES = 21
NOTCH_COVERED_FRAC = 0.60  # covered-outside fraction required
NOTCH_INSIDE_MAX = 0.20  # inside-stitch fraction must stay below this
NOTCH_EXT_MARGIN = 0.2  # along offset past stitch ends to sample "outside"
NOTCH_EXT_SPAN = 1.0  # sample window on each side past stitch ends


def _xz(pt):
    return (float(pt[0]), float(pt[2]))


def _safe_polygon(xz_points):
    if len(xz_points) < 3:
        return None
    poly = Polygon(xz_points)
    if not poly.is_valid:
        fixed = make_valid(poly)
        if hasattr(fixed, "geoms"):
            polys = [g for g in fixed.geoms if isinstance(g, Polygon)]
            poly = max(polys, key=lambda p: p.area) if polys else poly
        elif isinstance(fixed, Polygon):
            poly = fixed
    return poly if poly.is_valid and not poly.is_empty else None


def _wall_seg(corners):
    if not corners or len(corners) < 2:
        return None
    a = _xz(corners[0])
    b = _xz(corners[1])
    dx, dz = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dz)
    if L < 0.1:
        return None
    return a, b, (dx / L, dz / L), L


def _project_along_perp(origin, direction, point):
    """Return (along, perp) coordinates of point in the wall's local frame."""
    dx, dz = point[0] - origin[0], point[1] - origin[1]
    along = dx * direction[0] + dz * direction[1]
    perp = dx * (-direction[1]) + dz * direction[0]
    return along, perp


def _world(origin, direction, along, perp):
    normal = (-direction[1], direction[0])
    return (
        origin[0] + along * direction[0] + perp * normal[0],
        origin[1] + along * direction[1] + perp * normal[1],
    )


def _notch_predicate(
    wall_origin, wall_dir, wall_len, stitch_along, perp_offset, room_poly
):
    """Sample the strip between the wall and the stitch.

    Return a dict describing fractions covered by ``room_poly`` in three bands:
      - inside the stitch's along-range
      - just past each end of the stitch, within the wall's span

    Decision: notch == True iff the INSIDE band is largely uncovered while at
    least one of the OUTSIDE bands is largely covered. That matches the signal
    we saw on the target building (room 5 notches inward only near the stitch
    and reaches the wall on either side).
    """
    if room_poly is None:
        return False, {"reason": "no_polygon"}

    a_lo, a_hi = stitch_along
    strip_mid_perp = perp_offset / 2.0

    # Sample points inside the stitch's along-range at mid-perp.
    inside_hits = 0
    inside_total = 0
    for i in range(NOTCH_SAMPLES):
        t = i / (NOTCH_SAMPLES - 1)
        a = a_lo + t * (a_hi - a_lo)
        p = _world(wall_origin, wall_dir, a, strip_mid_perp)
        inside_total += 1
        if room_poly.covers(Point(p[0], p[1])):
            inside_hits += 1

    # Sample windows just past each stitch end, clamped to the wall's span.
    outside_hits = 0
    outside_total = 0
    windows = [
        (
            max(0.0, a_lo - NOTCH_EXT_MARGIN - NOTCH_EXT_SPAN),
            max(0.0, a_lo - NOTCH_EXT_MARGIN),
        ),
        (
            min(wall_len, a_hi + NOTCH_EXT_MARGIN),
            min(wall_len, a_hi + NOTCH_EXT_MARGIN + NOTCH_EXT_SPAN),
        ),
    ]
    for w_lo, w_hi in windows:
        if w_hi - w_lo < 0.1:
            continue
        for i in range(NOTCH_SAMPLES):
            t = i / (NOTCH_SAMPLES - 1)
            a = w_lo + t * (w_hi - w_lo)
            p = _world(wall_origin, wall_dir, a, strip_mid_perp)
            outside_total += 1
            if room_poly.covers(Point(p[0], p[1])):
                outside_hits += 1

    inside_frac = inside_hits / inside_total if inside_total else 0.0
    outside_frac = outside_hits / outside_total if outside_total else 0.0

    is_notch = outside_frac >= NOTCH_COVERED_FRAC and inside_frac <= NOTCH_INSIDE_MAX

    return is_notch, {
        "inside_frac": round(inside_frac, 3),
        "outside_frac": round(outside_frac, 3),
        "inside_samples": inside_total,
        "outside_samples": outside_total,
    }


def _portal_in_strip(room, wall_origin, wall_dir, stitch_along, perp_offset):
    """Return True if any door/window/opening centroid falls inside the strip
    between the stitch and the wall, within the stitch's along-range.

    A portal in the strip means a real wall with a door exists somewhere in
    the supposed notch - strongly argues against snapping.
    """
    a_lo, a_hi = stitch_along
    for key in ("doors", "windows", "openings"):
        for item in room.get(key, []):
            corners = item.get("corners", [])
            if not corners:
                continue
            cx = sum(float(p[0]) for p in corners) / len(corners)
            cz = sum(float(p[2]) for p in corners) / len(corners)
            along, perp = _project_along_perp(wall_origin, wall_dir, (cx, cz))
            if not (a_lo - 0.3 <= along <= a_hi + 0.3):
                continue
            if 0.1 < abs(perp) < perp_offset + 0.3 and (
                (perp_offset > 0 and 0 < perp < perp_offset)
                or (perp_offset < 0 and perp_offset < perp < 0)
            ):
                return True, {"kind": key, "id": item.get("id"), "centroid": [cx, cz]}
    return False, None


def simulate():
    buildings = json.load(open(BUILDINGS))

    totals = defaultdict(int)
    per_building = {}
    sample_snaps = []

    for b in buildings:
        uuid = b["uuid"]
        rooms = b.get("rooms", [])
        sws = b.get("stitch_walls", [])
        if not sws:
            continue

        room_polys = {}
        for ri, room in enumerate(rooms):
            fp = room.get("floor_polygon", [])
            room_polys[ri] = _safe_polygon([(float(p[0]), float(p[2])) for p in fp])

        wall_index = []  # (ri, story, wi, seg, wall_dict)
        for ri, room in enumerate(rooms):
            story = int(room.get("story", 0))
            for wi, w in enumerate(room.get("walls_computed", [])):
                seg = _wall_seg(w.get("corners", []))
                if not seg:
                    continue
                wall_index.append((ri, story, wi, seg, w))

        b_events = {"repaired": 0, "unchanged": 0, "regressed": 0, "events": []}

        for si, sw in enumerate(sws):
            s_seg = _wall_seg(sw.get("corners", []))
            if not s_seg:
                continue
            sa, sb, sd, _sL = s_seg
            s_story = int(sw.get("story", 0))
            sr = sw.get("room_indices") or (
                [sw.get("room_index")] if sw.get("room_index") is not None else []
            )
            sr = set(r for r in sr if r is not None)

            for ri, w_story, _wi, (wa, _wb, wd, wL), wdict in wall_index:
                if w_story != s_story:
                    continue
                if ri in sr:
                    continue
                dot = sd[0] * wd[0] + sd[1] * wd[1]
                # allow both parallel and anti-parallel
                if abs(dot) < PARALLEL_DOT_MIN:
                    continue
                along_a, perp_a = _project_along_perp(wa, wd, sa)
                along_b, perp_b = _project_along_perp(wa, wd, sb)
                if abs(perp_a - perp_b) > 0.15:
                    continue
                perp_avg = (perp_a + perp_b) / 2.0
                if not (PERP_MIN <= abs(perp_avg) <= PERP_MAX):
                    continue
                a_lo, a_hi = sorted((along_a, along_b))
                clipped_lo = max(a_lo, 0.0)
                clipped_hi = min(a_hi, wL)
                overlap = max(0.0, clipped_hi - clipped_lo)
                if overlap < ALONG_OVERLAP_MIN:
                    continue

                # Notch predicate: does any stitch-owner room's floor polygon
                # reach the non-owner wall OUTSIDE the stitch's along-range but
                # notch away from it INSIDE? That is the signal that the stitch
                # zone is a scan-boundary gap, not a real void.
                notch_ok = False
                notch_meta = None
                notch_room = None
                for owner_ri in sr:
                    owner_poly = room_polys.get(owner_ri)
                    if owner_poly is None:
                        continue
                    ok, meta = _notch_predicate(
                        wa, wd, wL, (a_lo, a_hi), perp_avg, owner_poly
                    )
                    if ok:
                        notch_ok = True
                        notch_meta = meta
                        notch_room = owner_ri
                        break
                    if notch_meta is None:
                        notch_meta = meta

                # Portal-in-strip check against the non-owner room (wall's owner)
                # AND against the stitch-owner room that notched. A door centroid
                # inside the strip argues the gap is a real portal - don't snap.
                portal_hit, portal_info = _portal_in_strip(
                    rooms[ri], wa, wd, (a_lo, a_hi), perp_avg
                )
                if not portal_hit and notch_room is not None:
                    portal_hit, portal_info = _portal_in_strip(
                        rooms[notch_room], wa, wd, (a_lo, a_hi), perp_avg
                    )

                category = "unchanged"
                if notch_ok and portal_hit:
                    category = "regressed"
                elif notch_ok:
                    category = "repaired"

                b_events[category] += 1
                b_events["events"].append(
                    {
                        "sw_index": si,
                        "wall_room": ri,
                        "wall_id": wdict.get("id"),
                        "notch_room": notch_room,
                        "perp_offset_m": round(perp_avg, 3),
                        "overlap_m": round(overlap, 3),
                        "category": category,
                        "notch": notch_meta,
                        "portal": portal_info,
                    }
                )

                if category == "repaired" and len(sample_snaps) < 20:
                    sample_snaps.append(
                        {
                            "uuid": uuid,
                            "sw_index": si,
                            "wall_room": ri,
                            "wall_id": wdict.get("id"),
                            "notch_room": notch_room,
                            "perp_offset_m": round(perp_avg, 3),
                            "overlap_m": round(overlap, 3),
                            "notch": notch_meta,
                        }
                    )

                # Only snap to one wall per stitch event.
                break

        per_building[uuid] = {
            "repaired": b_events["repaired"],
            "unchanged": b_events["unchanged"],
            "regressed": b_events["regressed"],
        }
        for k in ("repaired", "unchanged", "regressed"):
            totals[k] += b_events[k]

    totals["total_events"] = (
        totals["repaired"] + totals["unchanged"] + totals["regressed"]
    )
    totals["buildings_with_repairs"] = sum(
        1 for v in per_building.values() if v["repaired"] > 0
    )
    totals["buildings_with_regressions"] = sum(
        1 for v in per_building.values() if v["regressed"] > 0
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "totals": dict(totals),
                "per_building": per_building,
                "sample_snaps": sample_snaps,
                "predicate": {
                    "PARALLEL_DOT_MIN": PARALLEL_DOT_MIN,
                    "PERP_MIN": PERP_MIN,
                    "PERP_MAX": PERP_MAX,
                    "ALONG_OVERLAP_MIN": ALONG_OVERLAP_MIN,
                    "NOTCH_COVERED_FRAC": NOTCH_COVERED_FRAC,
                    "NOTCH_INSIDE_MAX": NOTCH_INSIDE_MAX,
                    "NOTCH_EXT_MARGIN": NOTCH_EXT_MARGIN,
                    "NOTCH_EXT_SPAN": NOTCH_EXT_SPAN,
                },
            },
            indent=2,
        )
    )

    print(json.dumps(dict(totals), indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    simulate()
