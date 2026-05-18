"""Phase B.5 smoke test — the hyperparam search itself isn't checked for
correctness (it's a random search), just that the plumbing runs end-to-
end. We build a minimal in-memory pilot and verify a 3-trial run writes
the expected files with sensible shape.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

if str(Path(__file__).resolve().parents[2] / "scripts") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import optimize_reconstruction as opt_mod

from reconcile_v3.reconstruction.solver import SolverConfig

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _fake_pilot_building(uuid: str) -> dict:
    # Minimal gable: two candidate faces covering opposite halves of a
    # 4x4 footprint, with a matching V3 reference envelope.
    south = [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]
    north = [[0.0, 2.0], [4.0, 2.0], [4.0, 4.0], [0.0, 4.0]]
    sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    candidates = [
        {
            "id": f"{uuid}::candidate::south",
            "building_uuid": uuid,
            "parent_segment_id": f"{uuid}::v3-merged-roof-segment::south",
            "plane": [0.0, 1.0, -1.0, 2.0],
            "footprint_xz": south,
            "area_m2": 8.0,
            "azimuth_deg": 180.0,
            "inclination_deg": 45.0,
            "neighbors": [f"{uuid}::candidate::north"],
            "support_m2": 8.0,
            "extended": False,
            "gbm_prior": None,
        },
        {
            "id": f"{uuid}::candidate::north",
            "building_uuid": uuid,
            "parent_segment_id": f"{uuid}::v3-merged-roof-segment::north",
            "plane": [0.0, 1.0, 1.0, -2.0],
            "footprint_xz": north,
            "area_m2": 8.0,
            "azimuth_deg": 0.0,
            "inclination_deg": 45.0,
            "neighbors": [f"{uuid}::candidate::south"],
            "support_m2": 8.0,
            "extended": False,
            "gbm_prior": None,
        },
    ]
    return {
        "uuid": uuid,
        "candidates": candidates,
        "scan_fp": [(x, z) for x, z in sq],
        "v3": {
            "slanted_roofs": [
                {"corners": [[x, 0.0, z] for (x, z) in south]},
                {"corners": [[x, 0.0, z] for (x, z) in north]},
            ],
            "flat_ceilings": [],
        },
    }


def test_run_trial_runs_end_to_end() -> None:
    pilot = [_fake_pilot_building("test-a"), _fake_pilot_building("test-b")]
    cfg = SolverConfig(w_prior=0.0)
    score, aggregate, per_bldg = opt_mod._run_trial(cfg, pilot)
    assert isinstance(score, float)
    assert aggregate["n_buildings"] == 2
    assert aggregate["n_with_reference"] == 2
    # With a perfect fixture the solver should achieve high IoU on both.
    assert aggregate["mean_iou"] > 0.9
    assert all("uuid" in row for row in per_bldg)


def test_sample_config_produces_values_in_range() -> None:
    rng = random.Random(0)
    for _ in range(20):
        cfg = opt_mod._sample_config(rng)
        assert 0.5 <= cfg.w_fit <= 2.0
        assert 0.0 <= cfg.w_prior <= 1.0
        assert 0.01 <= cfg.w_complexity <= 0.5
        assert 0.80 <= cfg.theta_cov <= 0.95
        assert 0.05 <= cfg.theta_overlap <= 0.30
        assert 20.0 <= cfg.theta_az_deg <= 90.0
        assert 2 <= cfg.k_azimuth_bins <= 8


def test_top_trial_written_to_disk(tmp_path: Path) -> None:
    """Run the whole pipeline (3 trials) against an in-memory pilot and
    confirm trials.json / top_trials.md land on disk with the expected
    shape. We bypass --candidates/--v3-results loading by
    monkey-patching ``_load_pilot``.
    """
    pilot = [_fake_pilot_building(f"test-{i}") for i in range(2)]
    orig_loader = opt_mod._load_pilot
    opt_mod._load_pilot = lambda *args, **kw: pilot
    try:
        argv = [
            "optimize_reconstruction.py",
            "--out-dir",
            str(tmp_path),
            "--n-trials",
            "3",
            "--pilot-n",
            "2",
            "--seed",
            "0",
        ]
        orig_argv = sys.argv
        sys.argv = argv
        try:
            rc = opt_mod.main()
        finally:
            sys.argv = orig_argv
    finally:
        opt_mod._load_pilot = orig_loader
    assert rc == 0

    trials_path = tmp_path / "trials.json"
    top_path = tmp_path / "top_trials.md"
    assert trials_path.exists()
    assert top_path.exists()

    trials = json.loads(trials_path.read_text())
    assert len(trials) == 3
    # Sorted by score descending.
    scores = [t["score"] for t in trials]
    assert scores == sorted(scores, reverse=True)
    # Each trial carries config + aggregate + per-building rows.
    for t in trials:
        assert set(t.keys()) >= {
            "trial",
            "config",
            "score",
            "aggregate",
            "per_building",
        }
        assert isinstance(t["config"], dict)
        # Config shape matches SolverConfig.
        sample = SolverConfig(**t["config"])
        assert asdict(sample) == t["config"]
