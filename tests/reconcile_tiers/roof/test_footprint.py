from pathlib import Path

from reconcile_tiers.roof.footprint import build_building_footprint
from tests.reconcile_tiers.roof.helpers import make_gable_model


def test_building_footprint_preserves_scan_room_outline():
    footprint = build_building_footprint(make_gable_model())

    assert footprint is not None
    assert footprint.top_story == 0
    assert len(footprint.polygon_xz) >= 4
    assert footprint.area > 20.0


def test_roof_package_uses_make_valid_instead_of_buffer_zero_for_repairs():
    roof_files = Path("reconcile_tiers/roof").glob("*.py")

    offenders = [str(path) for path in roof_files if "buffer(0)" in path.read_text()]

    assert offenders == []
