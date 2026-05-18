#!/usr/bin/env python3
"""Phase B.4 — prepare a screenshot queue for visually checking BIP
solver output against the V3 reference envelope.

This script does NOT drive the browser. It writes a
``screenshots_todo.json`` file that a separate agent picks up and
processes via the chrome-devtools MCP tools (the MCP lane is not
accessible from a plain Python CLI). The companion agent workflow is
in ``.agents/skills/screenshot-solver-output/`` (TBD).

Workflow
--------
1. This script: pick N pilot buildings, emit a todo queue of
   ``(uuid, layers_preset, cam_preset)`` tuples + an empty index.html
   shell that the agent fills in.
2. Agent loop (manual or via ``/screenshot-solver-output`` skill):

       for entry in screenshots_todo.json:
           mcp.new_page(f"http://localhost:8080/viewer.html#b={uuid}"
                        f"&layers={layers}&cam={cam}")
           mcp.wait_for("#render-complete")
           mcp.take_screenshot(path=f"{uuid}_{layers}_{cam}.png")

3. Agent writes ``DONE`` marker so the script (re-run) can stitch the
   screenshots into index.html for scroll-review.

Presets
-------
``v3_reference``   V3 full-model reference (slabs + walls + doors +
                   windows + flat_ceilings + slanted_roofs).
``bip_output``     V3 reference base + BIP envelope overlay.
``bip_only``       Walls + floors + BIP envelope (envelope isolated).

Cameras: ``iso`` (default), ``south``, ``east``, ``overhead``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Map from preset name → list of viewer layer keys (LAYER_CONTROL_IDS
# keys in reconcile/viewer-modules/constants.js). Keep in sync with the
# HTML toggle ids.
LAYER_PRESETS: dict[str, list[str]] = {
    "v3_reference": [
        "fullModel",  # triggers the unified V3 render
        "v3Model",
        "gableExtension",
    ],
    "bip_output": [
        "v3Model",
        "gableExtension",
        "reconstruction",
    ],
    "bip_only": [
        "computed",  # walls + floors skeleton
        "floors",
        "reconstruction",
    ],
}

# Per-building screenshot triple: which pairs we want for a pilot.
PILOT_TRIPLES: list[tuple[str, str]] = [
    ("v3_reference", "iso"),
    ("bip_output", "iso"),
    ("bip_only", "overhead"),
]


def _sample_pilot_uuids(
    selections_path: Path,
    n: int,
    seed: int,
) -> list[str]:
    with selections_path.open() as handle:
        selections = json.load(handle)
    rng = random.Random(seed)
    uuids = [s["building_uuid"] for s in selections]
    rng.shuffle(uuids)
    return uuids[:n]


def _index_html_shell(entries: list[dict]) -> str:
    """HTML shell with empty <img> placeholders keyed on the filenames
    the agent will produce. Re-running this script after the agent
    finishes picks up the actual images via the filesystem."""
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>BIP reconstruction screenshots</title>",
        "<style>",
        (
            "body { font-family: system-ui, sans-serif; background: #111; "
            "color: #eee; margin: 20px; }"
        ),
        (
            "section { margin-bottom: 40px; border-bottom: 1px solid #333; "
            "padding-bottom: 20px; }"
        ),
        "h2 { font-size: 14px; color: #aaa; }",
        "figure { display: inline-block; margin: 8px; }",
        "figcaption { font-size: 11px; color: #888; text-align: center; }",
        "img { width: 420px; border: 1px solid #333; background: #000; }",
        (
            ".missing { width: 420px; height: 300px; border: 1px dashed #555; "
            "display:inline-block; text-align:center; line-height:300px; "
            "color:#666; }"
        ),
        "</style></head><body>",
        f"<h1>BIP reconstruction screenshots — {len(entries)} buildings</h1>",
    ]
    for entry in entries:
        uuid = entry["building_uuid"]
        parts.append(f"<section><h2>{uuid}</h2>")
        for shot in entry["shots"]:
            fname = shot["filename"]
            label = f"{shot['layers']} / {shot['cam']}"
            parts.append(
                f"<figure><img src='{fname}' alt='{label}' "
                f"onerror=\"this.outerHTML='<div class=missing>missing: "
                f"{fname}</div>'\">"
                f"<figcaption>{label}</figcaption></figure>"
            )
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selections",
        default="reports/reconstruction_20260419/selections.json",
        type=Path,
    )
    ap.add_argument(
        "--out-dir",
        default="reports/reconstruction_screenshots_20260419",
        type=Path,
    )
    ap.add_argument(
        "--uuids",
        nargs="*",
        default=None,
        help="Explicit list of UUIDs. Overrides --n-pilot.",
    )
    ap.add_argument("--n-pilot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--viewer-url",
        default="http://localhost:8080/viewer.html",
        help="Base URL the agent will navigate to per entry.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.uuids:
        uuids = args.uuids
    else:
        if not args.selections.exists():
            print(
                f"Selections file not found: {args.selections}. "
                "Run scripts/run_reconstruction_solver.py first, or pass --uuids.",
                file=sys.stderr,
            )
            return 2
        uuids = _sample_pilot_uuids(args.selections, args.n_pilot, args.seed)

    todo_entries: list[dict] = []
    for uuid in uuids:
        shots = []
        for layers_preset, cam in PILOT_TRIPLES:
            layers = LAYER_PRESETS.get(layers_preset)
            if layers is None:
                print(f"Unknown layers preset: {layers_preset}", file=sys.stderr)
                continue
            fname = f"{uuid}_{layers_preset}_{cam}.png"
            hash_frag = f"#b={uuid}&layers={','.join(layers)}&cam={cam}"
            shots.append(
                {
                    "url": f"{args.viewer_url}{hash_frag}",
                    "filename": fname,
                    "layers": layers_preset,
                    "cam": cam,
                }
            )
        todo_entries.append({"building_uuid": uuid, "shots": shots})

    todo_path = args.out_dir / "screenshots_todo.json"
    with todo_path.open("w") as handle:
        json.dump(
            {
                "viewer_base": args.viewer_url,
                "entries": todo_entries,
                "instructions": (
                    "For each entry.shots item: mcp.navigate(url), "
                    "mcp.wait_for('#render-complete'), "
                    "mcp.take_screenshot(path=<out-dir>/<filename>)."
                ),
            },
            handle,
            indent=2,
        )

    index_path = args.out_dir / "index.html"
    index_path.write_text(_index_html_shell(todo_entries))

    print(
        f"Queued {sum(len(e['shots']) for e in todo_entries)} screenshots "
        f"across {len(todo_entries)} buildings."
    )
    print(f"Wrote {todo_path}")
    print(f"Wrote {index_path}")
    print(
        "\nNext: run the chrome-devtools MCP workflow to fill "
        "the queue (screenshots land next to index.html)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
