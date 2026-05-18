"""Argparse facade for energy sensitivity tooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reconcile_tiers.energy.calibrate import calibrate
from reconcile_tiers.energy.estimator import estimate, estimator_summary_dict
from reconcile_tiers.energy.score_flags import score_flag_queue
from reconcile_tiers.energy.silent_fix import dry_run
from reconcile_tiers.payload.schema import payload_from_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "score", "silent-fix-dry-run"):
        p = sub.add_parser(name)
        p.add_argument("--uuid", required=True)
        p.add_argument("--root", type=Path, default=Path("pipeline-outputs"))
        if name != "estimate":
            p.add_argument(
                "--queues-root", type=Path, default=Path(".context/flag-queues")
            )
    p = sub.add_parser("calibrate")
    p.add_argument("--root", type=Path, default=Path("pipeline-outputs"))
    p.add_argument("--queues-root", type=Path, default=Path(".context/flag-queues"))
    p.add_argument("--out", type=Path, default=Path(".context/energy-calibration"))
    args = parser.parse_args(argv)

    if args.command == "estimate":
        payload = payload_from_dict(
            json.loads((args.root / args.uuid / "tier_payload.json").read_text())
        )
        print(json.dumps(estimator_summary_dict(estimate(payload)), indent=2))
    elif args.command == "score":
        print(
            json.dumps(
                score_flag_queue(
                    args.uuid, root=args.root, queues_root=args.queues_root
                ),
                indent=2,
            )
        )
    elif args.command == "silent-fix-dry-run":
        print(
            json.dumps(
                dry_run(args.uuid, root=args.root, queues_root=args.queues_root),
                indent=2,
            )
        )
    elif args.command == "calibrate":
        print(
            json.dumps(
                calibrate(
                    root=args.root, queues_root=args.queues_root, out_root=args.out
                ),
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
