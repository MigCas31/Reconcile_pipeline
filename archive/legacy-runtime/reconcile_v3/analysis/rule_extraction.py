"""Phase 5: extract shippable auto-reject rules from a depth-4 decision tree.

Strategy: fit a single depth-4 tree on the full labeled set (CV in Phase 4
validated that shallow trees generalize). Walk leaves; keep those where
the majority class is *reject*, support is ≥ ``min_support`` samples, and
purity is ≥ ``min_purity``. Per rule, emit conjunction text + example
proposal IDs for viewer verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .modelling import prepare_data


@dataclass
class LeafRule:
    leaf_id: int
    conditions: list[
        str
    ]  # e.g. ["drainage_to_building_center_cos <= -0.412", "covered_side_count > 2"]
    support: int
    reject_count: int
    accept_count: int
    purity: float  # fraction of reject in the leaf
    reject_examples: list[str] = field(default_factory=list)
    accept_examples: list[str] = field(default_factory=list)

    @property
    def rule_text(self) -> str:
        return " AND ".join(self.conditions)


def _walk_leaves(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    leaf_of_row: np.ndarray,
    y: np.ndarray,
    proposal_ids: np.ndarray,
    min_support: int,
    min_purity: float,
    n_reject_examples: int = 5,
    n_accept_examples: int = 3,
    rng: np.random.Generator | None = None,
) -> list[LeafRule]:
    """Collect reject leaves meeting support/purity gates."""
    rng = rng or np.random.default_rng(0)
    t = tree.tree_
    is_leaf = t.children_left == -1

    # Build the path (root→leaf) conditions for every leaf.
    def path_for(leaf: int) -> list[str]:
        # Reverse-engineer the path by walking the whole tree with a stack.
        stack = [(0, [])]
        while stack:
            node, conds = stack.pop()
            if node == leaf:
                return conds
            if is_leaf[node]:
                continue
            f = feature_names[t.feature[node]]
            thr = t.threshold[node]
            stack.append((t.children_left[node], [*conds, f"{f} <= {thr:.4g}"]))
            stack.append((t.children_right[node], [*conds, f"{f} > {thr:.4g}"]))
        return []

    rules: list[LeafRule] = []
    for leaf_id in np.where(is_leaf)[0]:
        mask = leaf_of_row == leaf_id
        n = int(mask.sum())
        if n < min_support:
            continue
        reject_n = int((y[mask] == 0).sum())
        accept_n = n - reject_n
        # Reject leaf: majority class reject AND purity meets gate.
        if reject_n <= accept_n:
            continue
        purity = reject_n / n
        if purity < min_purity:
            continue
        conds = path_for(int(leaf_id))
        # Example IDs
        reject_idx = np.where(mask & (y == 0))[0]
        accept_idx = np.where(mask & (y == 1))[0]
        rj_pick = rng.choice(
            reject_idx, size=min(n_reject_examples, len(reject_idx)), replace=False
        )
        ac_pick = (
            rng.choice(
                accept_idx, size=min(n_accept_examples, len(accept_idx)), replace=False
            )
            if len(accept_idx)
            else np.array([], dtype=int)
        )
        rules.append(
            LeafRule(
                leaf_id=int(leaf_id),
                conditions=conds,
                support=n,
                reject_count=reject_n,
                accept_count=accept_n,
                purity=float(purity),
                reject_examples=[str(proposal_ids[i]) for i in rj_pick],
                accept_examples=[str(proposal_ids[i]) for i in ac_pick],
            )
        )
    rules.sort(key=lambda r: r.support * r.purity, reverse=True)
    return rules


def extract_rules(
    df: pd.DataFrame,
    min_support: int = 50,
    min_purity: float = 0.95,
    max_depth: int = 4,
    random_state: int = 0,
) -> tuple[list[LeafRule], DecisionTreeClassifier, list[str]]:
    """Fit a depth-``max_depth`` tree on all labeled rows; return reject leaves."""
    data = prepare_data(df)
    proposal_ids = (
        df[df["label"].isin(("accepted", "rejected"))]["proposal_id"]
        .reset_index(drop=True)
        .to_numpy()
    )
    # Median-impute so the tree is deterministic at inference time.
    medians = data.X.median(numeric_only=True)
    X_imp = data.X.fillna(medians)
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    tree.fit(X_imp, data.y, sample_weight=data.weights)
    leaf_of_row = tree.apply(X_imp)
    rules = _walk_leaves(
        tree,
        data.feature_names,
        leaf_of_row,
        data.y,
        proposal_ids,
        min_support=min_support,
        min_purity=min_purity,
    )
    return rules, tree, data.feature_names


def write_markdown(
    rules: list[LeafRule],
    out_path: Path,
    min_support: int,
    min_purity: float,
    tree_n_leaves: int,
    tree_depth: int,
    total_rows: int,
    total_reject: int,
) -> None:
    lines: list[str] = []
    lines.append("# Auto-reject rules — depth-4 tree\n\n")
    lines.append(
        f"Rules extracted from a single "
        f"`DecisionTreeClassifier(max_depth={tree_depth}, "
        f"class_weight='balanced')` trained on {total_rows} labeled merged-roof "
        f"segments. CV F1 gate passed in Phase 4; this tree is for rule extraction "
        f"only, not inference.\n\n"
    )
    lines.append(
        f"- gates: **support ≥ {min_support}**, **purity ≥ {min_purity:.2f}** (reject "
        f"fraction)\n"
    )
    lines.append(
        f"- tree leaves: {tree_n_leaves}; shippable reject leaves: {len(rules)}\n"
    )
    covered = sum(r.reject_count for r in rules)
    lines.append(
        f"- coverage: {covered} / {total_reject} rejects "
        f"({100 * covered / max(total_reject, 1):.1f}%)\n\n"
    )
    lines.append("Example IDs round-trip via the viewer search bar.\n\n")
    lines.append("---\n\n")

    for i, r in enumerate(rules, start=1):
        lines.append(f"## Rule {i} — support {r.support}, purity {r.purity:.3f}\n\n")
        lines.append("**Reject** when _all_ of:\n\n")
        for c in r.conditions:
            lines.append(f"- `{c}`\n")
        lines.append(
            f"\nCounts: {r.reject_count} reject / {r.accept_count} accept "
            f"(leaf_id={r.leaf_id}).\n\n"
        )
        lines.append("Reject examples (expected to fail in the viewer):\n")
        for e in r.reject_examples:
            lines.append(f"- `{e}`\n")
        if r.accept_examples:
            lines.append("\nAccept examples (false positives — verify in viewer):\n")
            for e in r.accept_examples:
                lines.append(f"- `{e}`\n")
        lines.append("\n---\n\n")

    if not rules:
        lines.append(
            "_No leaves met both gates. Loosen `min_support` or `min_purity`, "
            "or widen `max_depth`, if a rule surface is needed._\n"
        )
    out_path.write_text("".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features",
        default="artifacts/features_expanded.parquet",
    )
    ap.add_argument("--out", default="artifacts/auto_reject_rules.md")
    ap.add_argument("--min-support", type=int, default=50)
    ap.add_argument("--min-purity", type=float, default=0.95)
    ap.add_argument("--max-depth", type=int, default=4)
    args = ap.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found.", file=sys.stderr)
        return 2
    df = pd.read_parquet(feat_path)
    rules, tree, _feat_names = extract_rules(
        df,
        min_support=args.min_support,
        min_purity=args.min_purity,
        max_depth=args.max_depth,
    )
    total_reject = int((df["label"] == "rejected").sum())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(
        rules,
        out_path,
        min_support=args.min_support,
        min_purity=args.min_purity,
        tree_n_leaves=int((tree.tree_.children_left == -1).sum()),
        tree_depth=args.max_depth,
        total_rows=int(df["label"].isin(("accepted", "rejected")).sum()),
        total_reject=total_reject,
    )
    # Machine-readable sidecar (for score_results to load at inference).
    rules_json = [
        {
            "leaf_id": r.leaf_id,
            "conditions": r.conditions,
            "support": r.support,
            "purity": r.purity,
            "reject_count": r.reject_count,
            "accept_count": r.accept_count,
            "reject_examples": r.reject_examples,
            "accept_examples": r.accept_examples,
        }
        for r in rules
    ]
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(rules_json, indent=2))
    # Persist the tree too so inference can evaluate it deterministically.
    import pickle

    tree_path = out_path.parent / "rule_tree.pkl"
    # Persist the training-time medians alongside the tree; inference must
    # impute identically or leaf assignments will drift.
    data = prepare_data(df)
    medians = data.X.median(numeric_only=True).to_dict()
    approved_leaf_ids = [r.leaf_id for r in rules]
    with tree_path.open("wb") as f:
        pickle.dump(
            {
                "tree": tree,
                "feature_names": _feat_names,
                "medians": medians,
                "approved_reject_leaves": approved_leaf_ids,
            },
            f,
        )

    print(
        f"{len(rules)} reject rule(s) → {out_path} "
        f"(covers {sum(r.reject_count for r in rules)} / {total_reject} rejects, "
        f"{100 * sum(r.reject_count for r in rules) / max(total_reject, 1):.1f}%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
