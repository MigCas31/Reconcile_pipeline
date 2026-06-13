"""Minimum cycle basis for undirected graphs (de Pina / Kavitha)."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Callable
from itertools import pairwise


def minimum_cycle_basis(
    nodes: list[str],
    edges: list[tuple[str, str]],
    *,
    weight_fn: Callable[[str, str], float] | None = None,
) -> list[list[str]]:
    """Return a minimum-weight cycle basis as unordered node lists per component."""

    if not nodes or not edges:
        return []

    adj, weights = _build_adjacency(nodes, edges, weight_fn)
    all_cycles: list[list[str]] = []
    for component in _connected_components(nodes, adj):
        comp_adj = {n: [m for m in adj[n] if m in component] for n in component}
        comp_weights = {
            _edge_key(a, b): weights[_edge_key(a, b)]
            for a in component
            for b in comp_adj[a]
            if _edge_key(a, b) in weights
        }
        all_cycles.extend(_min_cycle_basis_component(component, comp_adj, comp_weights))
    return all_cycles


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _build_adjacency(
    nodes: list[str],
    edges: list[tuple[str, str]],
    weight_fn: Callable[[str, str], float] | None,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float]]:
    adj: dict[str, list[str]] = defaultdict(list)
    weights: dict[tuple[str, str], float] = {}
    node_set = set(nodes)
    for a, b in edges:
        if a not in node_set or b not in node_set or a == b:
            continue
        key = _edge_key(a, b)
        if key in weights:
            continue
        w = 1.0 if weight_fn is None else float(weight_fn(a, b))
        weights[key] = w
        adj[a].append(b)
        adj[b].append(a)
    for n in nodes:
        adj.setdefault(n, [])
    return adj, weights


def _connected_components(
    nodes: list[str],
    adj: dict[str, list[str]],
) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in nodes:
        if start in seen:
            continue
        comp: set[str] = set()
        queue = [start]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            for nxt in adj.get(cur, []):
                if nxt not in seen:
                    queue.append(nxt)
        if comp:
            components.append(comp)
    return components


def _spanning_tree_edges(
    component: set[str],
    adj: dict[str, list[str]],
) -> set[tuple[str, str]]:
    """Any spanning forest edge set (BFS)."""

    tree: set[tuple[str, str]] = set()
    start = next(iter(component))
    seen = {start}
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nxt in adj.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            tree.add(_edge_key(cur, nxt))
            queue.append(nxt)
    return tree


def _min_cycle_basis_component(
    component: set[str],
    adj: dict[str, list[str]],
    weights: dict[tuple[str, str], float],
) -> list[list[str]]:
    tree = _spanning_tree_edges(component, adj)
    all_edges = set(weights)
    chords = [e for e in all_edges if e not in tree]

    set_orth: list[set[tuple[str, str]]] = [{e} for e in chords]
    cycles: list[list[str]] = []

    while set_orth:
        base = set_orth.pop()
        cycle_edges = _min_cycle(component, adj, weights, base)
        cycles.append(_nodes_from_cycle_edges(cycle_edges))
        set_orth = [
            (
                {e for e in orth if e not in base and e[::-1] not in base}
                | {e for e in base if e not in orth and e[::-1] not in orth}
            )
            if sum((e in orth or e[::-1] in orth) for e in cycle_edges) % 2
            else orth
            for orth in set_orth
        ]

    return cycles


def _lift_id(node: str) -> str:
    return f"{node}::lift"


def _base_id(node: str) -> str:
    return node.split("::lift", 1)[0] if node.endswith("::lift") else node


def _dijkstra(
    start: str,
    targets: set[str],
    adj_lift: dict[str, list[tuple[str, float]]],
) -> tuple[float, list[str]]:
    dist: dict[str, float] = {start: 0.0}
    prev: dict[str, str | None] = {start: None}
    heap: list[tuple[float, str]] = [(0.0, start)]

    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, math.inf):
            continue
        if cur in targets:
            path: list[str] = []
            node: str | None = cur
            while node is not None:
                path.append(node)
                node = prev[node]
            path.reverse()
            return d, path

        for nxt, w in adj_lift.get(cur, []):
            nd = d + w
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(heap, (nd, nxt))

    return math.inf, []


def _min_cycle(
    component: set[str],
    adj: dict[str, list[str]],
    weights: dict[tuple[str, str], float],
    orth: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Minimum-weight cycle orthogonal to ``orth`` via lifted-graph shortest path."""

    lift_adj: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for a in component:
        a_lift = _lift_id(a)
        for b in adj.get(a, []):
            if b not in component:
                continue
            b_lift = _lift_id(b)
            key = _edge_key(a, b)
            wt = weights.get(key, 1.0)
            if key in orth:
                lift_adj[a].append((b_lift, wt))
                lift_adj[a_lift].append((b, wt))
            else:
                lift_adj[a].append((b, wt))
                lift_adj[a_lift].append((b_lift, wt))

    lift_dist = {
        n: _dijkstra(n, {_lift_id(n)}, lift_adj)[0] for n in component
    }
    start = min(lift_dist, key=lambda k: lift_dist[k])
    _, path = _dijkstra(start, {_lift_id(start)}, lift_adj)
    if not path:
        return []

    mapped = [_base_id(n) for n in path]
    edgelist = [_edge_key(a, b) for a, b in pairwise(mapped)]

    edgeset: set[tuple[str, str]] = set()
    for e in edgelist:
        rev = (e[1], e[0])
        if e in edgeset:
            edgeset.remove(e)
        elif rev in edgeset:
            edgeset.remove(rev)
        else:
            edgeset.add(e)

    min_edgelist: list[tuple[str, str]] = []
    for e in edgelist:
        rev = (e[1], e[0])
        if e in edgeset:
            min_edgelist.append(e)
            edgeset.remove(e)
        elif rev in edgeset:
            min_edgelist.append(rev)
            edgeset.remove(rev)
    return min_edgelist


def _nodes_from_cycle_edges(edges: list[tuple[str, str]]) -> list[str]:
    if not edges:
        return []

    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    start = edges[0][0]
    ordered = [start]
    prev: str | None = None
    cur = start
    for _ in range(len(edges) + 1):
        neighbors = adj[cur]
        nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
        if nxt == start:
            break
        ordered.append(nxt)
        prev, cur = cur, nxt
    return ordered
