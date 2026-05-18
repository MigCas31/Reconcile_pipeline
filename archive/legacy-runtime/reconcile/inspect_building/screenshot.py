"""Drive viewer-tiers.html via Playwright and capture multi-angle screenshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1600, "height": 1000}

# Camera presets relative to scene centroid + bounding-sphere radius.
# (offset_x, offset_y, offset_z) multipliers on the radius.
PRESETS: dict[str, tuple[float, float, float]] = {
    "iso": (1.2, 0.9, 1.2),
    "overhead": (0.01, 2.0, 0.01),
    "south": (0.0, 0.4, 1.6),
    "east": (1.6, 0.4, 0.0),
}

CAMERA_SCRIPT = """
([dx, dy, dz]) => {
  const v = window.__tierViewer;
  if (!v) throw new Error('__tierViewer missing');
  const { scene, camera, controls, requestRender } = v;
  const THREE_BOX = new
  (camera.constructor.prototype.constructor.toString().includes('Perspective')
    ? window.THREE?.Box3 || (function(){ throw new Error('THREE.Box3 not on '
                                                         'window'); })()
    : null);
}
"""

# We compute the bounding box ourselves in JS to avoid relying on THREE on window.
SET_CAMERA_JS = """
([dx, dy, dz]) => {
  const v = window.__tierViewer;
  if (!v) throw new Error('__tierViewer missing');
  const { scene, camera, controls, requestRender } = v;

  // Compute scene AABB by walking meshes flagged as tierPreview.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  scene.traverse((obj) => {
    if (!(obj.isMesh && obj.userData?.tierPreview && !obj.userData.pickOnly &&
    !obj.userData.framingIgnore)) return;
    obj.updateWorldMatrix(true, false);
    const geom = obj.geometry;
    if (!geom) return;
    if (!geom.boundingBox) geom.computeBoundingBox();
    const box = geom.boundingBox.clone().applyMatrix4(obj.matrixWorld);
    minX = Math.min(
        minX,
        box.min.x,
    ); minY = Math.min(minY, box.min.y); minZ = Math.min(minZ, box.min.z);
    maxX = Math.max(
        maxX,
        box.max.x,
    ); maxY = Math.max(maxY, box.max.y); maxZ = Math.max(maxZ, box.max.z);
  });
  if (!isFinite(minX)) return false;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const sx = maxX - minX, sy = maxY - minY, sz = maxZ - minZ;
  const radius = Math.max(sx, sy, sz, 4);

  controls.target.set(cx, cy, cz);
  camera.position.set(cx + radius * dx, cy + radius * dy, cz + radius * dz);
  camera.near = Math.max(0.05, radius / 200);
  camera.far = radius * 20;
  camera.updateProjectionMatrix();
  controls.update();
  if (requestRender) requestRender();
  return true;
}
"""

SELECT_LOCATOR_JS = """
(uid) => {
  const search = document.querySelector('#search');
  if (!search) throw new Error('search input missing');
  search.value = uid;
  const event = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true });
  search.dispatchEvent(event);
}
"""


def capture(
    uuid: str, eid: str | None, out_dir: Path, base_url: str, timeout_ms: int = 30000
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    captured: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        url = f"{base_url}/reconcile_tiers/web/viewer-tiers.html#b={uuid}"
        page.goto(url, wait_until="networkidle")

        # Wait for the tiers viewer to finish loading the requested building.
        page.wait_for_function(
            "(uuid) => window.__tierState && window.__tierState.activeUuid === uuid",
            arg=uuid,
        )
        # Let framing + render passes settle.
        page.wait_for_timeout(600)

        for name, deltas in PRESETS.items():
            ok = page.evaluate(SET_CAMERA_JS, list(deltas))
            if not ok:
                print(
                    f"[screenshot] preset {name}: no meshes to frame", file=sys.stderr
                )
                continue
            page.wait_for_timeout(250)
            shot_path = out_dir / f"{name}.png"
            page.screenshot(path=str(shot_path), full_page=False)
            captured.append(shot_path)

        if eid:
            page.evaluate(SELECT_LOCATOR_JS, eid)
            # Wait for either selectedLocator to update or for a "no mesh" status.
            try:
                page.wait_for_function(
                    "(uid) => window.__tierState && window.__tierState.selectedLocator "
                    "=== uid",
                    arg=eid,
                    timeout=5000,
                )
                page.wait_for_timeout(400)
                shot_path = out_dir / "element.png"
                page.screenshot(path=str(shot_path), full_page=False)
                captured.append(shot_path)
            except Exception as exc:
                print(f"[screenshot] element {eid}: not found ({exc})", file=sys.stderr)

        browser.close()

    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--eid", default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args()

    shots = capture(args.uuid, args.eid, args.out, args.base_url, args.timeout_ms)
    for path in shots:
        print(path)
    return 0 if shots else 1


if __name__ == "__main__":
    raise SystemExit(main())
