import * as THREE from "three";

export function frameScene(scene, camera, controls, fallbackCenter) {
  const box = new THREE.Box3();
  scene.traverse((obj) => {
    if (obj.userData?.tierPreview && obj.isMesh && !obj.userData.pickOnly && !obj.userData.framingIgnore) box.expandByObject(obj);
  });
  const center = box.isEmpty() ? fallbackCenter : box.getCenter(new THREE.Vector3());
  const size = box.isEmpty() ? new THREE.Vector3(8, 4, 8) : box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 4);
  controls.target.copy(center);
  camera.position.set(center.x + radius * 1.2, center.y + radius * 0.8, center.z + radius * 1.2);
  camera.near = Math.max(0.05, radius / 200);
  camera.far = radius * 20;
  camera.updateProjectionMatrix();
  controls.update();
  return { center, diagonal: size.length() || radius };
}

export function flyToLocator(scene, camera, controls, locator) {
  const mesh = scene.userData.tierLocatorMap?.get(locator);
  if (!mesh) return false;
  const box = new THREE.Box3().setFromObject(mesh);
  const center = box.getCenter(new THREE.Vector3());
  const size = Math.max(box.getSize(new THREE.Vector3()).length(), 1);
  controls.target.copy(center);
  camera.position.set(center.x + size, center.y + size * 0.7, center.z + size);
  controls.update();
  return true;
}

