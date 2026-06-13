"""Tests for minimum cycle basis."""

from __future__ import annotations

from reconcile_tiers.room_postprocessing.minimum_cycle_basis import (
    minimum_cycle_basis,
)


def test_triangle_yields_one_cycle() -> None:
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c"), ("c", "a")]
    cycles = minimum_cycle_basis(nodes, edges)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b", "c"}


def test_two_adjacent_rectangles_yield_two_cycles_not_exterior() -> None:
    """Two rooms sharing a wall: MCB = 2 room cycles, not the outer perimeter."""

    nodes = ["sw", "se", "mid_s", "mid_n", "ne", "nw"]
    edges = [
        ("sw", "mid_s"),
        ("mid_s", "se"),
        ("se", "ne"),
        ("ne", "mid_n"),
        ("mid_n", "nw"),
        ("nw", "sw"),
        ("mid_s", "mid_n"),
    ]
    cycles = minimum_cycle_basis(nodes, edges)
    assert len(cycles) == 2
    cycle_sets = [frozenset(c) for c in cycles]
    assert frozenset({"sw", "mid_s", "mid_n", "nw"}) in cycle_sets
    assert frozenset({"mid_s", "se", "ne", "mid_n"}) in cycle_sets
