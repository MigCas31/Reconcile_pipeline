"""Deterministic IDs for gap polygons and derived gap-wall elements."""

from __future__ import annotations

from hashlib import sha256
from math import isfinite


def _rounded_xz_ring(corners: list[list[float]]) -> list[tuple[float, float]]:
    ring = [
        (round(float(corner[0]), 4), round(float(corner[2]), 4))
        for corner in (corners or [])
        if len(corner) >= 3
    ]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring


def _canonicalize_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(ring) < 3:
        return ring

    variants: list[list[tuple[float, float]]] = []
    for candidate in (ring, list(reversed(ring))):
        best_rotation = min(
            (candidate[index:] + candidate[:index] for index in range(len(candidate))),
            key=lambda rotated: tuple(rotated),
        )
        variants.append(best_rotation)
    return min(variants, key=lambda rotated: tuple(rotated))


def stable_gap_anchor_id(gap: dict) -> str:
    """Return a deterministic gap anchor id from topology or geometry."""
    ontology_gap_id = gap.get("_ontology_gap_id")
    if isinstance(ontology_gap_id, str) and ontology_gap_id:
        return ontology_gap_id

    existing_id = gap.get("id")
    if isinstance(existing_id, str) and existing_id:
        return existing_id

    ring = _canonicalize_ring(_rounded_xz_ring(gap.get("corners") or []))
    story = gap.get("story")
    story_token = (
        str(int(story)) if isinstance(story, (int, float)) and isfinite(story) else "na"
    )
    ring_token = "|".join(f"{x:.4f},{z:.4f}" for x, z in ring) or "empty"
    digest = sha256(
        f"{gap.get('type', 'gap')}|{story_token}|{ring_token}".encode()
    ).hexdigest()[:16]
    return f"gap:{gap.get('type', 'gap')}:{story_token}:{digest}"


def stable_gap_wall_id(
    gap: dict,
    wall_type: str,
    role: str,
    index: int | None = None,
) -> str:
    """Return a deterministic ID for a derived gap-wall element."""
    parts = ["gw", stable_gap_anchor_id(gap), wall_type, role]
    if index is not None:
        parts.append(str(int(index)))
    return ":".join(parts)
