#!/usr/bin/env bash
# Build a "full understanding" report for a building UUID or element ID.
#
#   reconcile/inspect_building.sh <uuid>
#   reconcile/inspect_building.sh '<uuid>::<kind>::<id>'
#
# Output: .context/building-reports/<uuid>/<timestamp>/report.md (+ shots/, metrics.json, val3dity.json)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <uuid|element-id>" >&2
  exit 64
fi

INPUT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

UUID_RE='^[0-9a-fA-F-]{36}$'
if [[ "$INPUT" =~ $UUID_RE ]]; then
  UUID="$INPUT"
  ELEMENT_ID=""
else
  UUID="${INPUT%%::*}"
  ELEMENT_ID="$INPUT"
  if ! [[ "$UUID" =~ $UUID_RE ]]; then
    echo "error: could not parse UUID from '$INPUT'" >&2
    exit 64
  fi
fi

BUILDING_DIR="pipeline-outputs/$UUID"
if [[ ! -d "$BUILDING_DIR" ]]; then
  echo "error: $BUILDING_DIR not found. Run extraction first." >&2
  exit 66
fi

TS="$(date +%Y%m%dT%H%M%S)"
REPORT_DIR="$REPO_ROOT/.context/building-reports/$UUID/$TS"
mkdir -p "$REPORT_DIR/shots" "$REPORT_DIR/artifacts"

echo "[inspect-building] uuid=$UUID element=${ELEMENT_ID:-<none>}"
echo "[inspect-building] report dir: $REPORT_DIR"

PYTHON=(uv run python)
if ! command -v uv >/dev/null 2>&1; then
  PYTHON=(python)
fi

# 1. Element trace (if element id given)
if [[ -n "$ELEMENT_ID" ]]; then
  echo "[inspect-building] resolving element..."
  if ! "${PYTHON[@]}" -m reconcile.element_locator --element-id "$ELEMENT_ID" \
      > "$REPORT_DIR/element.json" 2> "$REPORT_DIR/element.err"; then
    echo "[inspect-building] element_locator failed; see element.err" >&2
    rm -f "$REPORT_DIR/element.json"
  else
    rm -f "$REPORT_DIR/element.err"
  fi
fi

# 2. Symlink the per-building artifacts so the report links work locally.
for name in merged.glb tier_payload.json datapack.json roofdiffusion.json reconciled.json; do
  if [[ -f "$BUILDING_DIR/$name" ]]; then
    ln -sf "$REPO_ROOT/$BUILDING_DIR/$name" "$REPORT_DIR/artifacts/$name"
  fi
done
if [[ -f "$BUILDING_DIR/lod2/reconstruction.obj" ]]; then
  ln -sf "$REPO_ROOT/$BUILDING_DIR/lod2/reconstruction.obj" "$REPORT_DIR/artifacts/reconstruction.obj"
fi

# 3. Realism check on the LoD2 mesh (trimesh) + cross-check vs tier_payload
OBJ_PATH="$BUILDING_DIR/lod2/reconstruction.obj"
TIER_PAYLOAD_PATH="$BUILDING_DIR/tier_payload.json"
if [[ -f "$OBJ_PATH" ]]; then
  echo "[inspect-building] running LoD2 realism checks..."
  REALISM_ARGS=(--obj "$OBJ_PATH" --out "$REPORT_DIR/metrics.json")
  if [[ -f "$TIER_PAYLOAD_PATH" ]]; then
    REALISM_ARGS+=(--tier-payload "$TIER_PAYLOAD_PATH")
  fi
  "${PYTHON[@]}" -m reconcile.inspect_building.metrics "${REALISM_ARGS[@]}" >/dev/null
else
  echo "[inspect-building] no lod2/reconstruction.obj — skipping realism check"
fi

# 3b. Tier-payload defect audit (what the user actually sees in the viewer)
if [[ -f "$TIER_PAYLOAD_PATH" ]]; then
  echo "[inspect-building] auditing tier_payload defects..."
  "${PYTHON[@]}" -m reconcile.inspect_building.audit \
    --tier-payload "$TIER_PAYLOAD_PATH" \
    --out "$REPORT_DIR/audit.json" >/dev/null
else
  echo "[inspect-building] no tier_payload.json — skipping defect audit"
fi

# 4. val3dity probe (optional)
if command -v val3dity >/dev/null 2>&1 && [[ -f "$OBJ_PATH" ]]; then
  echo "[inspect-building] running val3dity..."
  if ! val3dity --primitive MultiSurface -r "$REPORT_DIR/val3dity.json" "$OBJ_PATH" \
      > "$REPORT_DIR/val3dity.log" 2>&1; then
    echo "[inspect-building] val3dity exited non-zero; see val3dity.log"
  fi
else
  if ! command -v val3dity >/dev/null 2>&1; then
    touch "$REPORT_DIR/.val3dity-missing"
    echo "[inspect-building] val3dity not installed (brew tap tudelft3d/software && brew install val3dity)"
  fi
fi

# 5. Start a static file server for the tiers viewer (it loads everything from
#    pipeline-outputs/* as static files and does not need viewer_server.py).
VIEWER_PORT="${INSPECT_VIEWER_PORT:-8765}"
VIEWER_URL="http://localhost:$VIEWER_PORT"
HEALTH_PATH="/pipeline-outputs/tier_index.json"
STARTED_VIEWER=0
if ! curl -sf "$VIEWER_URL$HEALTH_PATH" >/dev/null 2>&1; then
  echo "[inspect-building] starting static server on :$VIEWER_PORT..."
  "${PYTHON[@]}" -m http.server "$VIEWER_PORT" --bind 127.0.0.1 \
    > "$REPORT_DIR/viewer.log" 2>&1 &
  VIEWER_PID=$!
  STARTED_VIEWER=1
  trap '[[ $STARTED_VIEWER -eq 1 ]] && kill $VIEWER_PID 2>/dev/null || true' EXIT
  for _ in $(seq 1 30); do
    sleep 0.5
    if curl -sf "$VIEWER_URL$HEALTH_PATH" >/dev/null 2>&1; then break; fi
  done
  if ! curl -sf "$VIEWER_URL$HEALTH_PATH" >/dev/null 2>&1; then
    echo "[inspect-building] static server failed to start; see viewer.log" >&2
  fi
fi

# 6. Screenshots
echo "[inspect-building] capturing screenshots..."
SHOT_ARGS=(--uuid "$UUID" --out "$REPORT_DIR/shots" --base-url "$VIEWER_URL")
if [[ -n "$ELEMENT_ID" ]]; then
  SHOT_ARGS+=(--eid "$ELEMENT_ID")
fi
if ! "${PYTHON[@]}" -m reconcile.inspect_building.screenshot "${SHOT_ARGS[@]}"; then
  echo "[inspect-building] screenshot capture failed" >&2
fi

# 7. Final summary
echo "[inspect-building] writing report..."
SUM_ARGS=(--uuid "$UUID" --out "$REPORT_DIR")
if [[ -n "$ELEMENT_ID" ]]; then
  SUM_ARGS+=(--element-id "$ELEMENT_ID")
fi
"${PYTHON[@]}" -m reconcile.inspect_building.summarize "${SUM_ARGS[@]}"

echo
echo "Done. Open: $REPORT_DIR/report.md"
