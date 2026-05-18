#!/usr/bin/env python3
"""Scan corpus to measure blast radius of candidate-atom demotion (P1a/P1b).

For each building:
  - counts of top_boundary_atom roles from /ontology-artifacts?view=summary
  - counts of semantic_atom surfaces by category in /ontology-artifacts?view=full-model
  - total unresolved_region surfaces rendered from atoms

Usage: run viewer on :8090, then execute this script.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "pipeline-outputs"
VIEWER = "http://localhost:8090"


def fetch(view: str, uuid: str) -> dict | None:
    url = f"{VIEWER}/ontology-artifacts?uuid={urllib.parse.quote(uuid)}&view={view}"
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  !! {view} error for {uuid}: {exc}", file=sys.stderr)
        return None


def iter_atoms(summary: dict):
    for atom in summary.get("semantic_atoms") or []:
        if isinstance(atom, dict) and atom.get("role"):
            yield atom


def main() -> int:
    uuids = sorted(p.name for p in CORPUS.iterdir() if p.is_dir())
    total = len(uuids)
    print(f"Scanning {total} buildings against {VIEWER}\n")

    per_building: list[dict] = []
    start = time.time()
    for idx, uuid in enumerate(uuids, 1):
        t0 = time.time()
        summary = fetch("summary", uuid)
        full = fetch("full-model", uuid)
        if summary is None or full is None:
            per_building.append({"uuid": uuid, "error": True})
            print(f"[{idx}/{total}] ERR {uuid}")
            continue

        role_counts: dict[str, int] = {}
        for atom in iter_atoms(summary):
            role = str(atom.get("role") or "")
            role_counts[role] = role_counts.get(role, 0) + 1

        atom_cat_counts: dict[str, int] = {}
        for s in full.get("renderable_surfaces", []):
            if str(s.get("source_kind") or "") != "semantic_atom":
                continue
            cat = str(s.get("category") or "")
            atom_cat_counts[cat] = atom_cat_counts.get(cat, 0) + 1

        candidates = role_counts.get("attic_floor_candidate", 0) + role_counts.get(
            "flat_transition_cap_candidate", 0
        )
        atom_unresolved = atom_cat_counts.get("unresolved_region", 0)

        per_building.append(
            {
                "uuid": uuid,
                "role_counts": role_counts,
                "atom_surface_counts": atom_cat_counts,
                "candidate_atoms": candidates,
                "atom_unresolved_surfaces": atom_unresolved,
            }
        )
        elapsed = time.time() - t0
        tag = f"cand={candidates} unresolved={atom_unresolved}"
        print(f"[{idx}/{total}] ok {uuid} {tag} {elapsed:.2f}s")

    total_elapsed = time.time() - start

    ok = [b for b in per_building if not b.get("error")]
    errored = [b for b in per_building if b.get("error")]
    with_cand = [b for b in ok if b["candidate_atoms"] > 0]
    with_unres = [b for b in ok if b["atom_unresolved_surfaces"] > 0]

    def sum_role(role: str) -> int:
        return sum(b["role_counts"].get(role, 0) for b in ok)

    def sum_cat(cat: str) -> int:
        return sum(b["atom_surface_counts"].get(cat, 0) for b in ok)

    print("\n" + "=" * 66)
    print(f"SCAN COMPLETE — {total_elapsed:.1f}s")
    print("=" * 66)
    print(f"Buildings total:            {total}")
    print(f"  ok:                       {len(ok)}")
    print(f"  errored:                  {len(errored)}")
    print()
    print(
        f"Buildings with candidate atoms:                  {len(with_cand)} / {len(ok)}"
    )
    print(
        f"Buildings emitting atom-level unresolved surfs:  {len(with_unres)} / "
        f"{len(ok)}"
    )
    print()
    print("Atom roles across corpus:")
    for role in [
        "sloped_ceiling",
        "flat_ceiling",
        "attic_floor",
        "attic_floor_inferred",
        "attic_floor_candidate",
        "flat_transition_cap",
        "flat_transition_cap_inferred",
        "flat_transition_cap_candidate",
    ]:
        print(f"  {role:35s} {sum_role(role):6d}")
    print()
    print("Atom-sourced surface categories in full-model:")
    for cat in [
        "room_ceiling_sloped",
        "room_ceiling_flat",
        "attic_floor",
        "unresolved_region",
    ]:
        print(f"  {cat:35s} {sum_cat(cat):6d}")
    print()
    suppressed = (
        sum_role("attic_floor_candidate")
        + sum_role("flat_transition_cap_candidate")
        - sum_cat("unresolved_region")
    )
    print(f"Candidate atoms suppressed by per-room unresolved fallback: {suppressed}")
    print()
    print("Top 15 buildings by candidate count:")
    for b in sorted(ok, key=lambda x: -x["candidate_atoms"])[:15]:
        if b["candidate_atoms"] == 0:
            break
        print(
            f"  {b['uuid']}  cand={b['candidate_atoms']:3d}  "
            f"(attic_c={b['role_counts'].get('attic_floor_candidate', 0)} "
            f"cap_c={b['role_counts'].get('flat_transition_cap_candidate', 0)})  "
            f"unresolved_surfs={b['atom_unresolved_surfaces']}"
        )

    if errored:
        print("\nErrored buildings:")
        for b in errored:
            print(f"  {b['uuid']}")

    out = Path("/tmp/candidate_atom_blast_radius.json")
    out.write_text(json.dumps({"per_building": per_building}, indent=2))
    print(f"\nDetails saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
