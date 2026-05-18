import * as THREE from "three";

/**
 * Sketchy edges (report §7.1).
 *
 * Replaces the calm outline material on existing LineSegments with a
 * ShaderMaterial that perturbs each vertex in NDC space by a small
 * pixel-scale offset derived from a deterministic hash of the world
 * position. Shared corners hash to the same offset, so segments stay
 * visually connected. This produces a hand-drawn wobble at near-zero
 * compute cost and leaves geometry untouched.
 *
 * The jitter amplitude is in pixels; we pass `uResolution` so the same
 * amplitude reads consistently regardless of viewport size.
 */

const VERTEX_SHADER = /* glsl */ `
  uniform vec2 uResolution;
  uniform float uJitterPx;

  // Two cheap hashes from the position; produce ~uniform values in [-1, 1].
  float hash1(vec3 p) {
    return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453) * 2.0 - 1.0;
  }
  float hash2(vec3 p) {
    return fract(sin(dot(p, vec3(63.7264, 10.873, 91.123))) * 17283.7919) * 2.0 - 1.0;
  }

  void main() {
    vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    // Pixel-space jitter, perspective-correct (multiply by clip.w so the
    // offset survives the perspective divide as a constant pixel amount).
    vec2 jitterPx = vec2(hash1(position), hash2(position)) * uJitterPx;
    clip.xy += jitterPx / max(uResolution, vec2(1.0)) * 2.0 * clip.w;
    gl_Position = clip;
  }
`;

const FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;
  void main() {
    gl_FragColor = vec4(uColor, uOpacity);
  }
`;

export function createSketchyEdgeMaterial({ color = 0x3a3a3a, opacity = 0.75, jitterPx = 2.5 } = {}) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uResolution: { value: new THREE.Vector2(1, 1) },
      uJitterPx: { value: jitterPx },
      uColor: { value: new THREE.Color(color) },
      uOpacity: { value: opacity },
    },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: true,
  });
  material.userData.isSketchyEdge = true;
  return material;
}

const SHARED_MATERIAL = createSketchyEdgeMaterial();

export function applySketchyEdges(scene) {
  scene.traverse((obj) => {
    if (!obj.isLineSegments) return;
    if (!obj.userData?.tierPreview) return;
    if (obj.material?.userData?.isSketchyEdge) return;
    obj.material = SHARED_MATERIAL;
  });
}

export function setSketchyEdgeResolution(width, height) {
  SHARED_MATERIAL.uniforms.uResolution.value.set(width, height);
}
