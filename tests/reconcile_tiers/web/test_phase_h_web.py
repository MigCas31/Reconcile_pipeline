import subprocess
from pathlib import Path


def test_phase_h_node_unit_tests_pass():
    result = subprocess.run(
        [
            "node",
            "--test",
            *sorted(
                str(path) for path in Path("tests/reconcile_tiers/web/js").glob("*.mjs")
            ),
        ],
        cwd=Path(__file__).parents[3],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_tier_web_does_not_reintroduce_renderer_fixups():
    web_dir = Path("reconcile_tiers/web")
    banned = [
        "MATERIALS.ceiling",
        "walls_merged",
        "flattenToMeanY",
        "orientHorizontalLidUp",
        "polygonPlaneBasis",
        "computeBuildingCenter",
    ]

    contents = "\n".join(path.read_text() for path in web_dir.glob("*.js"))

    for token in banned:
        assert token not in contents


def test_static_tier_viewer_runtime_is_decoupled_from_legacy_reconcile_paths():
    web_dir = Path("reconcile_tiers/web")
    web_contents = "\n".join(path.read_text() for path in web_dir.glob("*.*"))
    web_banned = [
        "/tier-index",
        "/building-merged",
        "viewer_server",
        "viewer-main",
        "viewer-modules",
        "ontology",
        "reconcile_v2",
        "reconcile_v3",
    ]
    for token in web_banned:
        assert token not in web_contents

    runtime_files = [
        path
        for path in Path("reconcile_tiers").rglob("*.py")
        if "archive" not in path.parts
        and "scripts" not in path.parts
        and "__pycache__" not in path.parts
    ]
    py_banned = [
        "from reconcile ",
        "from reconcile.",
        "import reconcile ",
        "import reconcile.",
        "reconcile_v2",
        "reconcile_v3",
    ]
    offenders = [
        (str(path), token)
        for path in runtime_files
        for token in py_banned
        if token in path.read_text()
    ]

    assert offenders == []


def test_tier_viewer_exposes_clickable_locator_selection():
    html = Path("reconcile_tiers/web/viewer-tiers.html").read_text()
    main_js = Path("reconcile_tiers/web/viewer-tiers-main.js").read_text()

    assert 'src="./viewer-tiers-main.js"' in html
    assert "selectedLocator: null" in main_js
    assert 'canvas.addEventListener("click"' in main_js
    assert 'canvas.addEventListener("contextmenu"' in main_js
    assert "new THREE.BoxHelper" in main_js
    assert "navigator.clipboard" in main_js


def test_tier_viewer_does_not_write_selected_locator_into_search():
    main_js = Path("reconcile_tiers/web/viewer-tiers-main.js").read_text()

    assert "search.value = uid" not in main_js
    assert 'getElementById("roof-rating-panel")' in main_js
    assert "event.stopPropagation()" in main_js


def test_tier_preview_renders_wall_like_faces_and_windows_double_sided_only():
    preview_js = Path("reconcile_tiers/web/tier-preview.js").read_text()

    assert (
        'const renderBothSides = def.name === "window" || def.name === "structure" || '
        'def.name === "structureFill"' in preview_js
    )
    assert "side: renderBothSides ? THREE.DoubleSide : THREE.FrontSide" in preview_js
    assert "options.side = THREE.DoubleSide" not in preview_js


def test_tier_preview_renders_windows_as_pascal_style_cutout_children():
    preview_js = Path("reconcile_tiers/web/tier-preview.js").read_text()

    assert "room.windows?.forEach((window, index) => {" in preview_js
    assert (
        "addWindowModel(scene, state, window, `${room.locator_id}:window:${index}`, "
        "story);" in preview_js
    )
    assert "function addWindowModel(scene, state, quad, uid, story)" in preview_js
    assert "WINDOW_FRAME_DEPTH_M" in preview_js
    assert "addOpeningPartBox(parent, frame" in preview_js
    assert "new THREE.MeshBasicMaterial(options)" in preview_js
    assert "material.forceSinglePass = true" in preview_js
    assert "wall.cutouts?.map((quad) => quad.corners) ?? []" in preview_js


def test_tier_preview_renders_doors_ceilings_and_roofs_like_pascal():
    preview_js = Path("reconcile_tiers/web/tier-preview.js").read_text()
    palette_js = Path("reconcile_tiers/web/material-palette.js").read_text()
    calm_palette_js = Path(
        "reconcile_tiers/web/calm/material-palette-calm.js"
    ).read_text()

    assert (
        "addDoorModel(scene, state, door, `${room.locator_id}:door:${index}`, story);"
        in preview_js
    )
    assert "function addDoorModel(scene, state, quad, uid, story)" in preview_js
    assert "DOOR_FRAME_DEPTH_M" in preview_js
    assert "DOOR_LEAF_DEPTH_M" in preview_js
    assert "function addCeilingGrid(parent, corners, holes, story)" in preview_js
    assert "const CEILING_GRID_STEP_M = 0.2;" in preview_js
    assert 'return "ceilingTop";' in preview_js
    assert 'roof: { name: "roof", fill: 0xe5e5e5' in palette_js
    assert 'roof: { name: "roof", fill: 0xe5e5e5' in calm_palette_js


def test_tier_preview_passes_dormer_face_cutouts_to_polygon_renderer():
    preview_js = Path("reconcile_tiers/web/tier-preview.js").read_text()

    assert "face.cutouts?.map((quad) => quad.corners) ?? []" in preview_js


def test_tier_preview_maps_dormer_faces_to_physical_materials():
    preview_js = Path("reconcile_tiers/web/tier-preview.js").read_text()

    assert 'face?.kind === "dormer_header"' in preview_js
    assert 'return { material: MATERIALS.roof, materialName: "roof" }' in preview_js
    assert (
        'return { material: MATERIALS.structure, materialName: "structure" }'
        in preview_js
    )


def test_tier_viewer_does_not_use_webgl_gtao_pass():
    main_js = Path("reconcile_tiers/web/viewer-tiers-main.js").read_text()

    assert "GTAOPass" not in main_js


def test_tier_viewer_uses_webgpu_ssgi_with_explicit_fallbacks():
    html = Path("reconcile_tiers/web/viewer-tiers.html").read_text()
    main_js = Path("reconcile_tiers/web/viewer-tiers-main.js").read_text()

    assert "three@0.184.0/build/three.webgpu.js" in html
    assert "three@0.184.0/build/three.tsl.js" in html
    assert 'import("three/webgpu")' in main_js
    assert 'import("three/tsl")' in main_js
    assert 'import("three/addons/tsl/display/SSGINode.js")' in main_js
    assert 'import("three/addons/tsl/display/DenoiseNode.js")' in main_js
    assert "navigator.gpu.requestAdapter" in main_js
    assert "falling back to WebGL" in main_js
    assert "falling back to direct rendering" in main_js
