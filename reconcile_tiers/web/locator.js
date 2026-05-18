export function makeElementUid(uuid, scope, ...parts) {
  return `${uuid}::tier-${scope}::${parts.map(String).join(":")}`;
}

export function parseElementUid(uid) {
  const match = String(uid || "").match(/^(.+)::tier-([^:]+)::(.+)$/);
  if (!match) return null;
  return {
    uuid: match[1],
    scope: match[2],
    parts: match[3] ? match[3].split(":") : [],
  };
}

export function attachLocator(mesh, uid) {
  if (!mesh.userData) mesh.userData = {};
  mesh.userData.elementLocator = uid;
  mesh.userData.elementUid = uid;
  return mesh;
}
