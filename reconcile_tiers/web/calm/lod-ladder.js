export function lodFromCamera(camera, center, diagonal) {
  const distance = camera.position.distanceTo(center);
  const ratio = diagonal > 0 ? distance / diagonal : 2;
  if (ratio > 2.5) return 1;
  if (ratio < 1.0) return 3;
  return 2;
}

export function applyLod(scene, level) {
  scene.traverse((obj) => {
    if (!obj.userData?.tierPreview || obj.userData?.pickOnly) return;
    const name = obj.material?.name || "";
    if (level <= 1 && (name === "window" || name === "doorLeaf")) {
      obj.visible = false;
    } else {
      obj.visible = true;
    }
    obj.material?.opacity != null && (obj.material.opacity = obj.material.opacity);
  });
}

