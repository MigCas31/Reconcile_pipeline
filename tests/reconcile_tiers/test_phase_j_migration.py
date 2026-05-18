from pathlib import Path


def test_phase_j_legacy_tier_surface_still_exists_until_migration_review():
    """Phase J is blocked, not executed: these files are the reviewed surface."""
    expected_legacy_files = [
        Path("reconcile/complexity_tiers.py"),
        Path("reconcile/viewer-tiers.html"),
        Path("reconcile/viewer-modules/tier-preview.js"),
        Path("archive/legacy-runtime/tests/test_complexity_tiers.py"),
    ]

    missing = [str(path) for path in expected_legacy_files if not path.exists()]

    assert missing == []


def test_phase_j_deletion_blockers_are_documented():
    audit = Path("reconcile_tiers/archive/MIGRATION_AUDIT.md").read_text()

    assert "Phase J status: blocked" in audit
    assert "new static viewer path is separate from this legacy surface" in audit
    assert "generated-payload tier distribution drift" in audit
    assert "No legacy files were deleted" in audit
    assert "Tier 2 deletion remains out of scope" in audit


def test_deferred_migration_items_are_archived():
    archive = Path("reconcile_tiers/archive")
    deferred = (archive / "DEFERRED_ITEMS.md").read_text()

    assert (archive / "MIGRATION_AUDIT.md").exists()
    assert "Legacy deletion" in deferred
    assert "Thermal cap emission" in deferred


def test_legacy_runtime_directories_are_archived_under_root():
    archive = Path("archive/legacy-runtime")
    legacy_names = ["reconcile", "reconcile_ext", "reconcile_v2", "reconcile_v3"]

    assert (Path("archive") / "README.md").exists()
    for name in legacy_names:
        root_path = Path(name)
        archived_path = archive / name

        assert archived_path.is_dir()
        assert root_path.is_symlink()
        assert root_path.resolve() == archived_path.resolve()


def test_legacy_support_material_is_archived_under_root():
    archive = Path("archive/legacy-runtime")
    symlinked_dirs = ["scripts", "reports", "artifacts"]

    for name in symlinked_dirs:
        root_path = Path(name)
        archived_path = archive / name

        assert archived_path.is_dir()
        assert root_path.is_symlink()
        assert root_path.resolve() == archived_path.resolve()

    assert (archive / "docs" / "raw_ceiling_plane_scorer_refactor.md").exists()
    assert (archive / "tests" / "test_complexity_tiers.py").exists()
    assert not any(Path("tests").glob("test_*.py"))
