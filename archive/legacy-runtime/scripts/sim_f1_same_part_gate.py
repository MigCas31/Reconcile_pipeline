"""Simulate F1 (+F1b) — hard-gate cross-part sloped coverage.

F1  changes ``reconcile/roof_algorithms_py/roof_coverage_graph.py`` so that in
    the envelope loop (≈ lines 172-197), when
    ``envelope["envelope_kind"] == "oblique"`` and ``not same_part`` we
    ``continue`` instead of recording the sloped state / hypothesis.

F1b changes ``reconcile/roof_algorithms_py/roof_cell_complex.py`` so that in
    the upper-void surface loop (≈ line 577) we also skip oblique
    ``surface_record`` entries where ``part_id`` is not in
    ``surface_record['part_ids']``. Without F1b, F1 flips
    ``target_sloped_hypothesis_id`` to ``None``, which *short-circuits* the
    existing hypothesis filter (``and target_sloped_hypothesis_id`` is
    falsy), so every oblique surface in the building — including the exact
    cross-part one that F1 suppressed — is back in the overlap pool.

This script replays both scenarios on the emitted
``roof_algorithms_py_results.json`` and compares them:

  scenario_F1 only
  ------------------
  Cells whose base atom was in the cohort AND whose base atom would NOT
  remain an upper-void candidate (flat_role ∉ {ceiling_cap,
  ambiguous_flat_over_sloped_part} AND the room has no oblique atom).

  scenario_F1+F1b
  ------------------
  All cells where ``cell.part_id`` does not own the cell's
  ``roof_hypothesis_id`` (i.e. every cross-part attic cell), regardless of
  the atom's other qualifiers.

A cell is **regressed** only if F1b removes an attic cap the room genuinely
needs. A room "still has a roof" if it has non-cross-part attic cells, or
has flat ceiling partitions that are not tagged ``ambiguous_flat_over_sloped_part``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reconcile" / "roof_algorithms_py_results.json"

ATTIC_KINDS = {"attic", "gable_attic", "ambiguous_attic"}
UPPER_VOID_FLAT_ROLES = {"ceiling_cap", "ambiguous_flat_over_sloped_part"}


def _part_oblique_index(building: dict) -> dict[str, set[str]]:
    return {
        n["id"]: set(n.get("oblique_hypothesis_ids") or [])
        for n in (building.get("building_part_graph") or {}).get("nodes") or []
    }


def _partitions_by_atom_id(building: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    cp = building.get("ceiling_partitions") or {}
    for atom in cp.get("flat") or []:
        out[atom["id"]] = atom
    for atom in cp.get("oblique") or []:
        out[atom["id"]] = atom
    return out


def _room_has_oblique_atom(building: dict) -> dict[tuple, bool]:
    """Map (room_id-or-index, story) -> whether the room has any oblique partition."""
    out: dict[tuple, bool] = defaultdict(bool)
    cp = building.get("ceiling_partitions") or {}
    for atom in cp.get("oblique") or []:
        key = (atom.get("room_id") or atom.get("room_index"), atom.get("story"))
        out[key] = True
    return out


def simulate_building(building: dict) -> dict:
    coverage_graph = building.get("roof_coverage_graph") or {}
    atom_coverage: dict[str, dict] = coverage_graph.get("atom_coverage") or {}
    coverage_graph.get("atom_regions") or {}
    cells = (building.get("roof_cell_complex") or {}).get("cells") or []
    part_oblique = _part_oblique_index(building)
    partitions_by_id = _partitions_by_atom_id(building)
    room_oblique = _room_has_oblique_atom(building)

    def hypo_owner_parts(hypo_id: str | None) -> set[str]:
        if not hypo_id:
            return set()
        return {pid for pid, ob in part_oblique.items() if hypo_id in ob}

    cohort_atoms: dict[str, dict] = {
        atom_id: cov
        for atom_id, cov in atom_coverage.items()
        if cov.get("sloped_hypothesis_id") and not cov.get("sloped_part_match")
    }

    f1_removed: list[dict] = []
    f1b_removed: list[dict] = []

    for cell in cells:
        if cell.get("cell_kind") not in ATTIC_KINDS:
            continue
        cell_part = cell.get("part_id")
        cell_hypo = cell.get("roof_hypothesis_id")
        owners = hypo_owner_parts(cell_hypo)
        if not (cell_part and owners and cell_part not in owners):
            continue

        base_atom_id = str(cell.get("base_atom_id") or "")
        partition = partitions_by_id.get(base_atom_id) or {}
        flat_role = partition.get("flat_role")
        room_key = (
            partition.get("room_id") or partition.get("room_index"),
            partition.get("story"),
        )
        atom_stays_upper_void = flat_role in UPPER_VOID_FLAT_ROLES or room_oblique.get(
            room_key, False
        )

        detail = {
            "cell_id": cell["id"],
            "base_atom_id": base_atom_id,
            "room_id": cell.get("room_id"),
            "cell_part": cell_part,
            "roof_hypothesis_id": cell_hypo,
            "flat_role": flat_role,
            "atom_stays_upper_void": atom_stays_upper_void,
        }
        # F1+F1b: every cross-part attic cell is removed.
        f1b_removed.append(detail)
        # F1 alone: only removed if the atom is no longer an upper-void candidate.
        if not atom_stays_upper_void:
            f1_removed.append(detail)

    f1_ids = {r["cell_id"] for r in f1_removed}
    f1b_ids = {r["cell_id"] for r in f1b_removed}

    # Room-level outcome: for each scenario, rooms that lose their last attic cell.
    room_before: dict = defaultdict(list)
    room_after_f1: dict = defaultdict(list)
    room_after_f1b: dict = defaultdict(list)
    for c in cells:
        if c.get("cell_kind") not in ATTIC_KINDS:
            continue
        rid = c.get("room_id")
        if rid is None:
            continue
        room_before[rid].append(c["id"])
        if c["id"] not in f1_ids:
            room_after_f1[rid].append(c["id"])
        if c["id"] not in f1b_ids:
            room_after_f1b[rid].append(c["id"])
    rooms_lost_f1 = sorted(rid for rid in room_before if not room_after_f1.get(rid))
    rooms_lost_f1b = sorted(rid for rid in room_before if not room_after_f1b.get(rid))

    # For each room that loses all attic cells, check whether it still has a
    # non-ambiguous flat ceiling partition — this confirms the room is a
    # genuinely-flat case that doesn't need an attic cell.
    flat_by_room: dict = defaultdict(list)
    for atom in (building.get("ceiling_partitions") or {}).get("flat") or []:
        rid = atom.get("room_id") or f"room:{atom.get('room_index')}"
        flat_by_room[rid].append(atom.get("flat_role"))

    def room_has_clean_flat_cover(rid: str) -> bool:
        roles = flat_by_room.get(rid) or []
        return any(r and r not in UPPER_VOID_FLAT_ROLES for r in roles)

    ambiguous_rooms_f1b = [
        rid for rid in rooms_lost_f1b if not room_has_clean_flat_cover(rid)
    ]

    return {
        "cohort_atoms": len(cohort_atoms),
        "total_attic_cells_before": len(room_before)
        and sum(len(v) for v in room_before.values()),
        "f1_removed": len(f1_removed),
        "f1b_removed": len(f1b_removed),
        "rooms_lost_f1": rooms_lost_f1,
        "rooms_lost_f1b": rooms_lost_f1b,
        "rooms_lost_f1b_without_clean_flat_cover": ambiguous_rooms_f1b,
        "f1_removed_details": f1_removed[:5],
        "f1b_removed_details": f1b_removed[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", action="append", default=None)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--verbose-uuids", nargs="*", default=[])
    args = parser.parse_args()

    results = json.loads(RESULTS_PATH.read_text())

    per_building: dict[str, dict] = {}
    agg_cohort = 0
    agg_f1 = 0
    agg_f1b = 0
    agg_rooms_lost_f1 = 0
    agg_rooms_lost_f1b = 0
    agg_rooms_ambiguous = 0
    total = 0
    hits = 0

    for uuid, building in results.items():
        if args.uuid and uuid not in args.uuid:
            continue
        total += 1
        out = simulate_building(building)
        if out["cohort_atoms"] == 0 and out["f1b_removed"] == 0:
            continue
        per_building[uuid] = out
        hits += 1
        agg_cohort += out["cohort_atoms"]
        agg_f1 += out["f1_removed"]
        agg_f1b += out["f1b_removed"]
        agg_rooms_lost_f1 += len(out["rooms_lost_f1"])
        agg_rooms_lost_f1b += len(out["rooms_lost_f1b"])
        agg_rooms_ambiguous += len(out["rooms_lost_f1b_without_clean_flat_cover"])

    print(f"buildings scanned:                       {total}")
    print(f"buildings with cross-part leaks:         {hits}")
    print(f"cohort atoms (sum):                      {agg_cohort}")
    print(f"cells removed by F1 alone:               {agg_f1}")
    print(f"cells removed by F1+F1b:                 {agg_f1b}")
    print(f"rooms losing all attic cells (F1):       {agg_rooms_lost_f1}")
    print(f"rooms losing all attic cells (F1+F1b):   {agg_rooms_lost_f1b}")
    print(f"  ...of which have NO clean flat cover:  {agg_rooms_ambiguous}")
    print()
    print("Those 'no clean flat cover' rooms are the only candidate regressions —")
    print("rooms where, under F1+F1b, neither an attic cell nor a flat-ceiling")
    print("partition (other than ambiguous_flat_over_sloped_part) would cap them.")

    top = sorted(
        per_building.items(),
        key=lambda kv: kv[1]["f1b_removed"],
        reverse=True,
    )[: args.top]
    print("\ntop buildings (by F1+F1b cell removal):")
    hdr = (
        f"{'uuid':<38} {'cohort':>6} {'F1':>5} {'F1+F1b':>7} {'lost(F1)':>9} "
        f"{'lost(F1+F1b)':>13} {'ambig':>6}"
    )
    print(hdr)
    for uuid, out in top:
        print(
            f"{uuid:<38} {out['cohort_atoms']:>6} {out['f1_removed']:>5} "
            f"{out['f1b_removed']:>7} {len(out['rooms_lost_f1']):>9} "
            f"{len(out['rooms_lost_f1b']):>13} "
            f"{len(out['rooms_lost_f1b_without_clean_flat_cover']):>6}"
        )

    for uuid in args.verbose_uuids:
        out = per_building.get(uuid)
        if not out:
            print(f"\n{uuid}: no cross-part coverage detected")
            continue
        print(f"\n--- {uuid} ---")
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
