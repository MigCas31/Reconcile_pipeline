// Dual-pane labeler.
//
// Left pane = baseline (the heuristic option, kept as a fixed reference).
// Right pane = the user's pick — what they're actively choosing. Clicking a
// card or pressing 1/2/3 puts that option into the right pane. Tab swaps
// left↔right (rare; mainly for re-comparing). Enter commits whatever is in
// the RIGHT pane.
//
// Both panes share a single PerspectiveCamera so the views stay aligned.
// Auto-orbit rotates the camera around the case target; pauses while the user
// is dragging, resumes 3 s after release, toggle with `o`.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const params = new URLSearchParams(location.search);
const RUN_ID = params.get("run") || "";
let cursor = parseInt(params.get("index") || "0", 10);
if (Number.isNaN(cursor) || cursor < 0) cursor = 0;

const $ = (s) => document.querySelector(s);
const setStatus = (msg) => { $("#status").textContent = msg; };

if (!RUN_ID) {
  setStatus("missing ?run=<run_id>");
  throw new Error("missing run param");
}

// ---- shared camera & dual scenes ------------------------------------------

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
camera.position.set(15, 12, 15);

function makePane(canvasId, sideKey) {
  const canvas = document.getElementById(canvasId);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(sideKey === "left" ? 0x0c0e12 : 0x0a0d12, 1);
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const sun = new THREE.DirectionalLight(0xffffff, 0.85);
  sun.position.set(20, 30, 10);
  scene.add(sun);
  const buildingGroup = new THREE.Group();
  const optionGroup = new THREE.Group();
  scene.add(buildingGroup);
  scene.add(optionGroup);
  return { canvas, renderer, scene, buildingGroup, optionGroup };
}

const left = makePane("canvas-left", "left");
const right = makePane("canvas-right", "right");

// OrbitControls is attached to BOTH canvases so the user can drag in either.
// Both controllers point at the same camera + the same target; we sync them.
const controlsLeft = new OrbitControls(camera, left.canvas);
const controlsRight = new OrbitControls(camera, right.canvas);
[controlsLeft, controlsRight].forEach((c) => {
  c.enableDamping = true;
  c.dampingFactor = 0.12;
});

// `lastInteraction` is when the user last *finished* a drag. While they're
// actively dragging we set `userDragging=true` so auto-orbit can't fight them
// frame-by-frame. We deliberately do NOT listen to OrbitControls' `change`
// event — applyAutoOrbit moves the camera every frame, which would fire change
// and self-cancel the orbit timer.
let lastInteraction = 0;
let userDragging = false;
[controlsLeft, controlsRight].forEach((c) => {
  c.addEventListener("start", () => { userDragging = true; });
  c.addEventListener("end", () => { userDragging = false; lastInteraction = performance.now(); });
});

// ---- auto-orbit ------------------------------------------------------------

const ORBIT_RPS = 0.06;            // revolutions per second
const ORBIT_RESUME_AFTER_MS = 3000; // idle delay before auto-orbit resumes
let orbitEnabled = true;
function isAutoOrbiting() {
  if (!orbitEnabled || userDragging) return false;
  return performance.now() - lastInteraction >= ORBIT_RESUME_AFTER_MS;
}
function applyAutoOrbit(dtMs) {
  if (!isAutoOrbiting()) return;
  const target = controlsLeft.target;
  const offset = camera.position.clone().sub(target);
  const radius = Math.hypot(offset.x, offset.z);
  if (radius < 1e-3) return;
  let theta = Math.atan2(offset.z, offset.x);
  theta += ORBIT_RPS * Math.PI * 2 * (dtMs / 1000);
  offset.x = radius * Math.cos(theta);
  offset.z = radius * Math.sin(theta);
  camera.position.copy(target).add(offset);
}

const orbitToggleBtn = $("#orbit-toggle");
function paintOrbitToggle() {
  orbitToggleBtn.classList.toggle("on", orbitEnabled);
  orbitToggleBtn.textContent = orbitEnabled ? "● orbit" : "○ paused";
}
orbitToggleBtn.addEventListener("click", () => {
  orbitEnabled = !orbitEnabled;
  paintOrbitToggle();
  // When turning on, kick the timer so orbit starts immediately.
  if (orbitEnabled) lastInteraction = 0;
});
paintOrbitToggle();

// ---- resize / render loop --------------------------------------------------

function resizePane(pane) {
  const w = pane.canvas.clientWidth;
  const h = pane.canvas.clientHeight;
  if (w === 0 || h === 0) return;
  pane.renderer.setSize(w, h, false);
}
function resizeAll() {
  resizePane(left);
  resizePane(right);
  // Aspect uses left canvas; right matches because grid splits 50/50.
  const w = left.canvas.clientWidth;
  const h = left.canvas.clientHeight;
  if (w > 0 && h > 0) {
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}
window.addEventListener("resize", resizeAll);

let prevFrame = performance.now();
function loop() {
  const now = performance.now();
  const dt = now - prevFrame;
  prevFrame = now;
  resizeAll();
  applyAutoOrbit(dt);
  controlsLeft.update();
  controlsRight.update();
  left.renderer.render(left.scene, camera);
  right.renderer.render(right.scene, camera);
  requestAnimationFrame(loop);
}
loop();

// ---- helpers --------------------------------------------------------------

function clearGroup(g) {
  while (g.children.length) {
    const c = g.children.pop();
    if (c.geometry) c.geometry.dispose();
    if (c.material) {
      if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose());
      else c.material.dispose();
    }
  }
}

function vec3List(corners) {
  return corners.map((c) =>
    Array.isArray(c) ? [c[0], c[1], c[2]] : [c.x, c.y, c.z]
  );
}

function polygonMesh(corners, color, opacity) {
  if (!corners || corners.length < 3) return null;
  const positions = [];
  for (let i = 1; i < corners.length - 1; i++) {
    positions.push(...corners[0], ...corners[i], ...corners[i + 1]);
  }
  if (positions.length === 0) return null;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geom.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color, opacity, transparent: opacity < 1.0,
    side: THREE.DoubleSide, flatShading: true,
    metalness: 0.0, roughness: 0.85,
  });
  return new THREE.Mesh(geom, mat);
}

function polygonOutline(corners, color) {
  if (!corners || corners.length < 2) return null;
  const pts = corners.map((c) => new THREE.Vector3(c[0], c[1], c[2]));
  pts.push(pts[0]);
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
}

function renderBuilding(group, payload, highlightSet) {
  clearGroup(group);
  const isHighlight = (loc) => highlightSet && loc && highlightSet.has(loc);
  for (const room of payload.rooms || []) {
    const roomHi = isHighlight(room.locator_id);
    const wallColor = roomHi ? 0x86efac : 0x37414d;
    const wallOpacity = roomHi ? 0.55 : 0.32;
    const floorColor = roomHi ? 0x14532d : 0x1f2937;
    const floorOpacity = roomHi ? 0.7 : 0.5;
    for (const wall of room.walls || []) {
      const m = polygonMesh(vec3List(wall.corners), wallColor, wallOpacity);
      if (m) group.add(m);
    }
    for (const f of room.floor || []) {
      const m = polygonMesh(vec3List(f.corners), floorColor, floorOpacity);
      if (m) group.add(m);
    }
  }
  for (const piece of payload.ceiling || []) {
    const m = polygonMesh(vec3List(piece.corners), 0x4b5563, 0.28);
    if (m) group.add(m);
  }
  for (const piece of payload.visual_shells || []) {
    const m = polygonMesh(vec3List(piece.corners), 0x44403c, 0.18);
    if (m) group.add(m);
  }
}

function renderOption(group, option) {
  clearGroup(group);
  if (!option) return;
  const color = new THREE.Color(option.color || "#fbbf24");
  for (const poly of option.polygons || []) {
    const m = polygonMesh(poly, color, 0.92);
    if (m) group.add(m);
    const ol = polygonOutline(poly, color);
    if (ol) group.add(ol);
  }
}

// ---- API ------------------------------------------------------------------

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} on ${url}`);
  return res.json();
}

async function loadRunMeta() {
  const data = await fetchJSON(`/labeler/runs/${encodeURIComponent(RUN_ID)}`);
  $("#run-name").textContent = data.meta.run_id;
  $("#counts").textContent = `${data.labelled}/${data.case_count} labelled`;
  return data;
}

let currentCase = null;
let currentLabel = null;
let leftIdx = 0;
let rightIdx = 1;

function optionAt(idx) {
  if (!currentCase) return null;
  if (idx < 0 || idx >= currentCase.options.length) return null;
  return currentCase.options[idx];
}

function paintPanes() {
  if (!currentCase) return;
  const lo = optionAt(leftIdx);
  const ro = optionAt(rightIdx);
  renderOption(left.optionGroup, lo);
  renderOption(right.optionGroup, ro);
  $("#left-name").textContent = lo ? lo.label : "—";
  $("#right-name").textContent = ro ? ro.label : "—";
  paintCards();
}

function paintCards() {
  const root = $("#strip-cards");
  root.innerHTML = "";
  if (!currentCase) return;
  currentCase.options.forEach((opt, idx) => {
    const card = document.createElement("div");
    card.className = "card";
    if (idx === leftIdx && idx === rightIdx) card.classList.add("both");
    else if (idx === leftIdx) card.classList.add("left");
    else if (idx === rightIdx) card.classList.add("right");
    if (currentLabel?.label_kind === "select" && currentLabel.selected_option_id === opt.id) {
      card.classList.add("committed");
    }
    card.innerHTML = `
      <span class="card-key">${idx + 1}</span>
      <span class="swatch" style="background:${opt.color};color:${opt.color}"></span>
      <span class="card-label">${opt.label}</span>`;
    card.addEventListener("click", () => {
      // Click sends the option to the RIGHT pane (always). Clicking the same
      // option as left is allowed — it means "my pick equals the baseline".
      rightIdx = idx;
      paintPanes();
    });
    root.appendChild(card);
  });
}

function paintCaseInfo() {
  if (!currentCase) return;
  // Short slug: last 8 chars of building uuid + tail of locator (kind + id)
  const uuidShort = currentCase.building_uuid.slice(-8);
  const locTail = currentCase.locator_id.split("::").slice(-2).join("::");
  const story = currentCase.features?.room_story;
  const roomBit = story != null ? ` · story ${story}` : "";
  $("#case-slug").textContent = `${uuidShort} · ${locTail}${roomBit}`;
  $("#locator").textContent = currentCase.locator_id;
  $("#progress").textContent = `${cursor + 1} / ${RUN_TOTAL}`;
  const incl = currentCase.features?.inclination_deg;
  const inclText = incl != null ? `${incl.toFixed(1)}°` : "";
  $("#incl-pill").textContent = incl != null ? `incl ${incl.toFixed(1)}°` : "";
  $("#case-incl").textContent = inclText;
  const heur = $("#heuristic-pill");
  heur.textContent = currentCase.heuristic_label
    ? `heuristic: ${currentCase.heuristic_label}` : "";
  heur.dataset.label = currentCase.heuristic_label || "";

  const tbl = $("#features");
  tbl.innerHTML = "";
  for (const [k, v] of Object.entries(currentCase.features || {})) {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td"); td1.textContent = k;
    const td2 = document.createElement("td");
    td2.textContent = typeof v === "number" ? Number(v.toFixed(4)) : JSON.stringify(v);
    tr.append(td1, td2); tbl.appendChild(tr);
  }
  const prior = $("#prior-label");
  if (currentLabel) {
    prior.hidden = false;
    prior.querySelector("pre").textContent = JSON.stringify(currentLabel, null, 2);
  } else {
    prior.hidden = true;
  }
}

function frameTo(caseObj) {
  const target = caseObj.camera_target || [0, 0, 0];
  const offset = caseObj.camera_offset || [12, 8, 12];
  controlsLeft.target.set(target[0], target[1], target[2]);
  controlsRight.target.set(target[0], target[1], target[2]);
  camera.position.set(
    target[0] + offset[0],
    target[1] + offset[1],
    target[2] + offset[2],
  );
  camera.updateProjectionMatrix();
  controlsLeft.update();
  controlsRight.update();
  // Auto-orbit should start as soon as a new case loads.
  lastInteraction = 0;
  userDragging = false;
}

function pickInitialIndices(caseObj) {
  // Left = heuristic option (fixed baseline). Right starts on the same
  // option, meaning "default pick = baseline". The user diverges the right
  // pane by clicking a card or pressing 1/2/3, then commits with Enter.
  const heur = caseObj.heuristic_label;
  let li = 0;
  if (heur) {
    const idx = caseObj.options.findIndex((o) => o.id === heur);
    if (idx !== -1) li = idx;
  }
  return [li, li];
}

async function loadCase(index) {
  setStatus(`loading case ${index}…`);
  const data = await fetchJSON(
    `/labeler/runs/${encodeURIComponent(RUN_ID)}/case?index=${index}`,
  );
  currentCase = data.case;
  currentLabel = data.label || null;
  cursor = data.index;
  history.replaceState(null, "", `?run=${encodeURIComponent(RUN_ID)}&index=${cursor}`);

  const payload = await fetchJSON(
    `/pipeline-outputs/${encodeURIComponent(currentCase.building_uuid)}/tier_payload.json`,
  );
  const highlightSet = new Set(currentCase.highlight_locators || []);
  renderBuilding(left.buildingGroup, payload, highlightSet);
  renderBuilding(right.buildingGroup, payload, highlightSet);

  [leftIdx, rightIdx] = pickInitialIndices(currentCase);
  paintCaseInfo();
  paintPanes();
  frameTo(currentCase);
  setStatus("");
}

let RUN_TOTAL = 0;

// ---- commit ---------------------------------------------------------------

async function postLabel(body) {
  const res = await fetch(
    `/labeler/runs/${encodeURIComponent(RUN_ID)}/labels`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST failed: ${res.status} ${text}`);
  }
  return res.json();
}

async function commitPick() {
  if (!currentCase) return;
  const opt = optionAt(rightIdx);
  if (!opt) return;
  setStatus(`committing ${opt.id}…`);
  // Visual confirmation: snap left pane to match right so both show the
  // committed option for a brief moment before advancing.
  leftIdx = rightIdx;
  paintPanes();
  await postLabel({
    case_id: currentCase.case_id,
    decision_type: currentCase.decision_type,
    label_kind: "select",
    selected_option_id: opt.id,
    reasons: [],
  });
  setStatus(`saved · ${opt.id}`);
  await loadRunMeta();
  await new Promise((r) => setTimeout(r, 320));
  goNext();
}

async function commitKind(kind) {
  if (!currentCase) return;
  setStatus(`committing ${kind}…`);
  await postLabel({
    case_id: currentCase.case_id,
    decision_type: currentCase.decision_type,
    label_kind: kind,
    reasons: [],
  });
  setStatus(`saved · ${kind}`);
  await loadRunMeta();
  goNext();
}

async function commitNeither() {
  // "Neither" = explicit reject of all options. Persist as select="neither" if
  // such an option exists, otherwise fall back to label_kind=skip with a
  // distinguishing reason. The case generators include a "neither" option for
  // this run, so we expect the first branch.
  if (!currentCase) return;
  const idx = currentCase.options.findIndex((o) => o.id === "neither");
  if (idx !== -1) {
    setStatus("committing neither…");
    await postLabel({
      case_id: currentCase.case_id,
      decision_type: currentCase.decision_type,
      label_kind: "select",
      selected_option_id: "neither",
      reasons: [],
    });
  } else {
    await postLabel({
      case_id: currentCase.case_id,
      decision_type: currentCase.decision_type,
      label_kind: "skip",
      reasons: ["neither"],
    });
  }
  setStatus("saved · neither");
  await loadRunMeta();
  goNext();
}

async function goNext() {
  if (cursor + 1 >= RUN_TOTAL) {
    setStatus("end of run");
    return;
  }
  await loadCase(cursor + 1).catch((e) => setStatus(e.message));
}
async function goPrev() {
  if (cursor === 0) return;
  await loadCase(cursor - 1).catch((e) => setStatus(e.message));
}

function swapLeftRight() {
  const tmp = leftIdx;
  leftIdx = rightIdx;
  rightIdx = tmp;
  paintPanes();
}

function setRightToIdx(idx) {
  if (!currentCase) return;
  if (idx < 0 || idx >= currentCase.options.length) return;
  rightIdx = idx;
  paintPanes();
}

function cycleRight(direction) {
  if (!currentCase) return;
  const n = currentCase.options.length;
  if (n === 0) return;
  rightIdx = (rightIdx + direction + n) % n;
  paintPanes();
}

// ---- bindings -------------------------------------------------------------

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "Tab") { e.preventDefault(); swapLeftRight(); return; }
  if (e.key === "Enter") { commitPick().catch((err) => setStatus(err.message)); return; }
  if (e.key === "n" || e.key === "N") { commitNeither().catch((err) => setStatus(err.message)); return; }
  if (e.key === "s" || e.key === "S") { commitKind("skip").catch((err) => setStatus(err.message)); return; }
  if (e.key === "u" || e.key === "U") { commitKind("unsure").catch((err) => setStatus(err.message)); return; }
  if (e.key === "ArrowLeft") { cycleRight(-1); return; }
  if (e.key === "ArrowRight") { cycleRight(+1); return; }
  if (e.key === "ArrowUp") { goPrev(); return; }
  if (e.key === "ArrowDown") { goNext(); return; }
  if (e.key === "o" || e.key === "O") {
    orbitEnabled = !orbitEnabled;
    paintOrbitToggle();
    if (orbitEnabled) lastInteraction = 0;
    return;
  }
  if (e.key === "r" || e.key === "R") { if (currentCase) frameTo(currentCase); return; }
  if (/^[1-9]$/.test(e.key)) {
    const idx = parseInt(e.key, 10) - 1;
    setRightToIdx(idx);
    return;
  }
});

document.querySelectorAll(".action").forEach((btn) => {
  btn.addEventListener("click", () => {
    const act = btn.dataset.act;
    if (act === "commit") commitPick().catch((e) => setStatus(e.message));
    else if (act === "neither") commitNeither().catch((e) => setStatus(e.message));
    else if (act === "skip") commitKind("skip").catch((e) => setStatus(e.message));
    else if (act === "unsure") commitKind("unsure").catch((e) => setStatus(e.message));
  });
});

$("#nav-prev").addEventListener("click", () => goPrev());
$("#nav-next").addEventListener("click", () => goNext());
$("#swap-btn").addEventListener("click", swapLeftRight);

// ---- bootstrap ------------------------------------------------------------

(async function init() {
  try {
    const meta = await loadRunMeta();
    RUN_TOTAL = meta.case_count;
    if (RUN_TOTAL === 0) { setStatus("run has no cases"); return; }
    if (cursor >= RUN_TOTAL) cursor = 0;
    await loadCase(cursor);
  } catch (e) {
    setStatus(e.message);
    console.error(e);
  }
})();
