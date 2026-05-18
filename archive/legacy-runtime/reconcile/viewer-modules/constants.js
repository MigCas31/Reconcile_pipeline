export const STORY_COLORS = [
  0xff6b6b, 0x4ecdc4, 0xf7dc6f, 0xbb8fce, 0x82e0aa, 0x45b7d1,
];

export const STORY_WALL_COLORS = [
  0xcc4444, 0x339999, 0xccaa33, 0x8855aa, 0x559966, 0x338899,
];

export const ROOM_COLORS = [
  0xff6b6b, 0x4ecdc4, 0x45b7d1, 0xf7dc6f, 0xbb8fce,
  0x82e0aa, 0xf0b27a, 0xaed6f1, 0xd7bde2, 0xa3e4d7,
  0xf5b7b1, 0x85c1e9, 0xfad7a0, 0xd5f5e3, 0xe8daef,
];

export const DOOR_COLOR = 0xcc7733;
export const DOOR_EDGE = 0xff9944;
export const WINDOW_COLOR = 0x33aadd;
export const WINDOW_EDGE = 0x55ccff;
export const MERGED_COLOR = 0x4488ff;
export const MERGED_EDGE = 0x6699ff;

export const RAW_CEILING_COLOR = 0xd4a574;
export const RAW_CEILING_EDGE = 0xe6b87d;

// Phase-2 raw-ceiling role palette (v2: computed localizes, raw identifies).
// shell_passthrough = green (pipeline owns this room; raw is decorative).
// shell_override = blue (fragmented room; raw confirms pipeline + adds specificity).
// reconstruction_input = magenta (fragmented room + raw disagrees; drives new geometry).
// interior_soffit = amber (inside the envelope, reject from outer shell).
// diagnostic = slate (confident room + raw disagrees; flag for review, never discarded).
export const RAW_CEILING_ROLE_COLORS = {
  shell_passthrough: 0x16a34a,
  shell_override: 0x3b82f6,
  reconstruction_input: 0xd946ef,
  interior_soffit: 0xf59e0b,
  diagnostic: 0x64748b,
  unclassified: 0xd4a574,
};

// Reconstruction piece palette — dormer pieces and wing pieces share the
// dormer roles where meaningful (slope = main front, cheeks = sides, header
// = flat top). Wings get a distinct slope hue so they read apart from
// dormers in a mixed-archetype building.
export const CEILING_RECON_DORMER_COLORS = {
  slope: 0x3b82f6,
  cheek_left: 0x0ea5e9,
  cheek_right: 0x06b6d4,
  header: 0x8b5cf6,
};
export const CEILING_RECON_WING_COLORS = {
  slope: 0xef4444,
  flat_top: 0xf97316,
};

export const ROOF_CLUSTER_COLORS = [
  0xff4444, 0x44bbff, 0x44dd44, 0xffaa22, 0xcc44ff,
  0xff66aa, 0x22ddcc, 0xdddd22, 0x8888ff, 0xff8844,
];

export const SOURCE_COLORS = {
  'scan-cache':       { fill: 0x33cc66, edge: 0x44ee77 },
  'hybrid':           { fill: 0xccaa33, edge: 0xeebb44 },
  'merged-room':      { fill: 0x4488ff, edge: 0x6699ff },
  'scan-cache-dedup': { fill: 0x44cccc, edge: 0x55eedd },
};

export const SOURCE_LABELS = {
  'scan-cache':       'Scan (SVD)',
  'hybrid':           'Hybrid',
  'merged-room':      'Merged fallback',
  'scan-cache-dedup': 'Scan (dedup)',
};

export const LAYER_CONTROL_IDS = {
  merged: 'show-merged',
  computed: 'show-computed',
  doors: 'show-doors',
  windows: 'show-windows',
  floors: 'show-floors',
  gaps: 'show-gaps',
  crossStory: 'show-cross-story',
  extensions: 'show-extensions',
  overlaps: 'show-overlaps',
  wallClips: 'show-wall-clips',
  extGaps: 'show-ext-gaps',
  ceilings: 'show-ceilings',
  rawCeilings: 'show-raw-ceilings',
  rawCeilingsRoles: 'show-raw-ceilings-roles',
  rawCeilingsReconstructions: 'show-raw-ceilings-reconstructions',
  computedOverextend: 'show-computed-overextend',
  rawDisagreement: 'show-raw-disagreement',
  rawCeilingPlaneSplits: 'show-raw-ceiling-plane-splits',
  rawCeilingPlaneSplitCandidates: 'show-raw-ceiling-plane-split-candidates',
  ceilingReplacement: 'show-ceiling-replacement',
  thermalCeilings: 'show-thermal-ceilings',
  roofClusters: 'show-roof-clusters',
  ontologySemantics: 'show-ontology-semantics',
  ontologyContinuation: 'show-ontology-continuation',
  ontologyCells: 'show-ontology-cells',
  fullModel: 'show-full-model',
  v3Model: 'show-v3-model',
  v3Proposals: 'show-v3-roof-proposals',
  gableExtension: 'show-v3-gable-extension',
  candidateFaces: 'show-candidate-faces',
  reconstruction: 'show-reconstruction',
  ridgeEave: 'show-ridge-eave',
};

// Candidate-face colors (Phase A of the reconstruction track). Original faces
// are the untouched segment footprint; extended faces are the ridge-extrapolated
// region reclaimed by clipping against opposing planes.
export const CANDIDATE_FACE_COLOR_ORIGINAL = 0x38bdf8;
export const CANDIDATE_FACE_COLOR_EXTENDED = 0xfb923c;

// BIP envelope color (Phase B.1). Faces selected by the solver — green
// for auto-accept, amber for review.
export const RECONSTRUCTION_COLOR_ACCEPT = 0x22c55e;
export const RECONSTRUCTION_COLOR_REVIEW = 0xf59e0b;

// Ridge/eave scoring overlay colors.
export const RIDGE_EAVE_RIDGE_COLOR = 0xffd700;      // yellow ridge
export const RIDGE_EAVE_EAVE_COLOR = 0xfb923c;       // orange eaves
export const RIDGE_EAVE_MEDIAL_COLOR = 0xffffff;     // white medial axis (dashed)
export const RIDGE_EAVE_OBB_COLOR = 0x6366f1;        // indigo part OBB
// Score gradient: red (0) → yellow (0.5) → green (1)
export const RIDGE_EAVE_SCORE_LOW = 0xef4444;
export const RIDGE_EAVE_SCORE_MID = 0xfacc15;
export const RIDGE_EAVE_SCORE_HIGH = 0x22c55e;

export const GABLE_STATUS_COLORS = {
  gable_along_extend: 0x22c55e,
  gable_cross_review: 0xfacc15,
  gable_complete: 0x3b82f6,
  gable_ambiguous: 0xa855f7,
  not_gable: 0x94a3b8,
};

export const GABLE_RIDGE_COLOR = 0xef4444;
export const GABLE_EXTENSION_TARGET_COLOR = 0xf97316;

// Computed-extent-vs-raw overextend overlay (from
// scripts/audit_computed_surface_extent_vs_raw.py). The polygon is the XZ
// region of a computed roof surface that has no overlapping raw ceiling
// evidence beneath it, lifted to the surface's Y plane. Colour ramps by
// overextend_fraction_xz so a glance tells you "edge creep" vs "mostly
// unsupported".
export const COMPUTED_OVEREXTEND_COLORS = {
  low: 0xfde68a,   // soft amber (<=10% overextend)
  mid: 0xf97316,   // orange (10-40%)
  high: 0xdc2626,  // red (>40%)
};

// Raw-ceiling orientation-disagreement overlay (from
// scripts/audit_raw_orientation_disagreement.py). Each polygon is the XZ
// intersection of two raw ceiling planes with substantially different
// fitted normals — a slope-split (dormer, hip, ridge) the pipeline may
// have flattened into a single computed surface. Colour ramps by angle
// between normals.
export const RAW_DISAGREEMENT_COLORS = {
  low: 0x93c5fd,   // soft blue (20-30°)
  mid: 0x6366f1,   // indigo (30-60°)
  high: 0xa855f7,  // violet (>60°)
};

// Raw-eave-supported split overlay (from
// scripts/prototype_raw_ceiling_plane_scorer.py). Supported stripes are
// coloured by chain-support confidence; residual pieces are muted so the
// unsupported remainder is visible without dominating the scene.
export const RAW_CEILING_SPLIT_COLORS = {
  supported_low: 0xf59e0b,
  supported_mid: 0x84cc16,
  supported_high: 0x16a34a,
  not_final_low: 0x60a5fa,
  not_final_mid: 0x3b82f6,
  not_final_high: 0x1d4ed8,
  suspect_interior_slice: 0xec4899,
  ownership_competitor_loss: 0xdc2626,
  ownership_unpaired_through: 0x2563eb,
  residual: 0x94a3b8,
  intersection_seam: 0x9333ea,
};

// Clean-ceiling replacement overlay (from
// scripts/audit_noisy_slanted_ceiling_replacement.py). Each polygon is
// either (a) the computed oblique roof surface clipped to a noisy-slanted
// room, or (b) a two-piece oblique + flat-cap reconstruction when the raw
// scan shows a horizontal cap above the slope. Colour ramps by planes/m²
// density so a glance reads "how chaotic was the scan?" for the cleaned
// region.
export const CEILING_REPLACEMENT_COLORS = {
  low: 0xbbf7d0,   // mint (<=0.7 planes/m²)
  mid: 0x22c55e,   // green (0.7-1.5)
  high: 0x15803d,  // dark green (>1.5)
};

export const LAYER_KEYS = Object.keys(LAYER_CONTROL_IDS);

export const PIPELINE_STEPS = [
  { id: 'walls-raw-ceilings', label: 'Computed walls + raw ceilings', adds: ['computed', 'extensions', 'gaps', 'crossStory', 'extGaps', 'rawCeilings'] },
  { id: 'raw-ceiling-prototype', label: '+ Raw-ceiling roles + reconstructions', adds: ['rawCeilings', 'rawCeilingsRoles', 'rawCeilingsReconstructions'] },
  { id: 'computed-overextend', label: '+ Computed-extent overextend', adds: ['computed', 'ceilings', 'rawCeilings', 'computedOverextend'] },
  { id: 'raw-disagreement', label: '+ Raw orientation disagreement', adds: ['rawCeilings', 'rawDisagreement'] },
  { id: 'raw-eave-splits', label: '+ Raw eave-supported splits', adds: ['rawCeilings', 'rawCeilingPlaneSplits'] },
  { id: 'ceiling-replacement', label: '+ Clean ceilings (replace noisy slanted)', adds: ['rawCeilings', 'ceilingReplacement'] },
  { id: 'ridge-eave', label: '+ Ridge/eave scoring', adds: ['ridgeEave'] },
  { id: 'full-model', label: 'Final full model', exclusive: ['fullModel', 'ridgeEave'] },
];

export const EMPTY_MAP_STYLE = { version: 8, sources: {}, layers: [] };
