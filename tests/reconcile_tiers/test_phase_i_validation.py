import functools
import json
import os
import shutil
import struct
import threading
import zlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reconcile_tiers.build import build_tier_payload
from reconcile_tiers.build_internals.io_index import payload_json
from reconcile_tiers.payload.schema import (
    CeilingSource,
    DormerFaceKind,
    payload_to_dict,
)

# Goldens are the canonical priors-OFF baseline. Skip when the architectural
# priors toggle is on — geometry intentionally shifts (Y on snapped raw
# ceilings, cluster avg_azimuth) so direct snapshot equality doesn't apply.
_PRIORS_ON_SKIP = pytest.mark.skipif(
    os.environ.get("ARCHITECTURAL_PRIORS") == "1",
    reason="goldens are the priors-OFF baseline; priors-ON shifts geometry",
)


@pytest.fixture(autouse=True)
def _force_legacy_priors_off(monkeypatch):
    monkeypatch.setenv("ARCHITECTURAL_PRIORS", "0")


COHORT_UUIDS = [
    "f40dcc9f-b97b-4bef-8b40-ba011aabf0bd",
    "05cecad4-119e-4dd2-beaf-d4af36973644",
    "19459014-f39b-4bf6-945e-c5b50930dea3",
    "16784bad-2cd9-4f4c-bb26-60355981cfe2",
    "3297bc28-5d39-4358-bd2e-0f1183cae489",
    "0d3f2993-8386-4130-8f1c-b2938c410828",
]
GOLDEN_ROOT = Path("tests/golden")
METRICS_PATH = GOLDEN_ROOT / "cohort_metrics.json"
SCREENSHOT_MAX_DIFF_FRACTION = 0.05


def _payload(uuid: str):
    return build_tier_payload(uuid, Path("pipeline-outputs"), Path(".scan-cache"))


def _xz_area(corners: list[dict[str, float]]) -> float:
    if len(corners) < 3:
        return 0.0
    points = [(float(point["x"]), float(point["z"])) for point in corners]
    area = 0.0
    for idx, (x0, z0) in enumerate(points):
        x1, z1 = points[(idx + 1) % len(points)]
        area += x0 * z1 - x1 * z0
    return abs(area) * 0.5


def _metrics(payload_dict: dict) -> dict:
    ceiling = payload_dict["ceiling"]
    knee_walls = payload_dict["knee_walls"]
    dormer_faces = payload_dict["dormer_faces"]
    return {
        "tier": payload_dict["classification"]["tier"],
        "roof_type": payload_dict["classification"]["roof_type"],
        "n_oblique": payload_dict["classification"]["n_oblique"],
        "n_flat": payload_dict["classification"]["n_flat"],
        "n_dormers": sum(
            1
            for face in dormer_faces
            if face["kind"] == DormerFaceKind.DORMER_HEADER.value
        ),
        "n_ceiling_pieces": len(ceiling),
        "n_knee_walls": len(knee_walls),
        "n_thermal_caps": sum(
            1 for piece in ceiling if piece["source"] == CeilingSource.THERMAL_CAP.value
        ),
        "total_ceiling_xz_area": round(
            sum(_xz_area(piece["corners"]) for piece in ceiling), 4
        ),
    }


def _load_golden_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


@_PRIORS_ON_SKIP
@pytest.mark.parametrize("uuid", COHORT_UUIDS)
def test_tier_payload_matches_golden_snapshot(uuid: str):
    payload = _payload(uuid)
    current = json.loads(payload_json(payload))
    golden = json.loads((GOLDEN_ROOT / "tier_payload" / f"{uuid}.json").read_text())

    assert current == golden


@_PRIORS_ON_SKIP
@pytest.mark.parametrize("uuid", COHORT_UUIDS)
def test_cohort_metrics_match_committed_tolerances(uuid: str):
    payload = _payload(uuid)
    current = _metrics(payload_to_dict(payload))
    golden = _load_golden_metrics()
    expected = golden["metrics"][uuid]
    tolerances = golden["tolerances"]

    for key in (
        "tier",
        "roof_type",
        "n_oblique",
        "n_flat",
        "n_dormers",
        "n_ceiling_pieces",
        "n_knee_walls",
        "n_thermal_caps",
    ):
        assert current[key] == expected[key]
    assert (
        abs(current["total_ceiling_xz_area"] - expected["total_ceiling_xz_area"])
        <= tolerances["total_ceiling_xz_area_abs_m2"]
    )


def _decode_png_rgba(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break

    assert width is not None and height is not None and bit_depth == 8
    channels = {2: 3, 6: 4}[color_type]
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    rows: list[bytearray] = []
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        row = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        prev = rows[-1] if rows else bytearray(stride)
        for idx in range(stride):
            left = row[idx - channels] if idx >= channels else 0
            up = prev[idx]
            up_left = prev[idx - channels] if idx >= channels else 0
            if filter_type == 1:
                row[idx] = (row[idx] + left) & 0xFF
            elif filter_type == 2:
                row[idx] = (row[idx] + up) & 0xFF
            elif filter_type == 3:
                row[idx] = (row[idx] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = (
                    left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                )
                row[idx] = (row[idx] + predictor) & 0xFF
            elif filter_type != 0:
                raise AssertionError(f"unsupported PNG filter {filter_type}")
        rows.append(row)

    rgba = bytearray()
    for row in rows:
        for idx in range(0, len(row), channels):
            rgba.extend(row[idx : idx + 3])
            rgba.append(row[idx + 3] if channels == 4 else 255)
    return width, height, bytes(rgba)


def _pixel_diff_fraction(actual: Path, expected: Path) -> float:
    width_a, height_a, rgba_a = _decode_png_rgba(actual)
    width_b, height_b, rgba_b = _decode_png_rgba(expected)
    assert (width_a, height_a) == (width_b, height_b) == (1280, 720)
    pixels = width_a * height_a
    mismatches = 0
    for idx in range(0, len(rgba_a), 4):
        if (
            max(
                abs(rgba_a[idx + channel] - rgba_b[idx + channel])
                for channel in range(4)
            )
            > 16
        ):
            mismatches += 1
    return mismatches / pixels


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass


@pytest.fixture(scope="module")
def phase_i_static_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase_i_static")
    shutil.copytree(Path("reconcile_tiers/web"), root / "reconcile_tiers" / "web")
    buildings = []
    for uuid in COHORT_UUIDS:
        payload = _payload(uuid)
        payload_dir = root / "pipeline-outputs" / uuid
        payload_dir.mkdir(parents=True)
        payload_dir.joinpath("tier_payload.json").write_text(payload_json(payload))
        buildings.append({"uuid": uuid, "address": payload.address})
    (root / "pipeline-outputs" / "tier_index.json").write_text(
        json.dumps({"buildings": buildings}, indent=2, sort_keys=True) + "\n"
    )
    return root


@pytest.fixture(scope="module")
def phase_i_server(phase_i_static_root):
    handler = functools.partial(_QuietHandler, directory=str(phase_i_static_root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


@_PRIORS_ON_SKIP
def test_screenshot_pixel_diff_against_goldens(phase_i_server, tmp_path):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        try:
            for idx, uuid in enumerate(COHORT_UUIDS):
                page.goto(
                    f"{phase_i_server}/reconcile_tiers/web/viewer-tiers.html?v=phase-i-{idx}#b={uuid}",
                    wait_until="networkidle",
                    timeout=60000,
                )
                page.wait_for_function(
                    "window.__tierState && window.__tierState.locatorCount > 0",
                    timeout=20000,
                )
                page.wait_for_timeout(500)
                actual = tmp_path / f"{uuid}.png"
                page.screenshot(path=str(actual))
                expected = GOLDEN_ROOT / "screenshots" / f"{uuid}.png"
                assert (
                    _pixel_diff_fraction(actual, expected)
                    <= SCREENSHOT_MAX_DIFF_FRACTION
                )
        finally:
            browser.close()
