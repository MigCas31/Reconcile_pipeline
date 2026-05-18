#!/usr/bin/env zsh
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PORT=${CONDUCTOR_PORT:-${PORT:-8080}}
export VIEWER_PORT="$PORT"

if [[ -f "$SCRIPT_DIR/.context/roof_ratings.json" ]]; then
  python -m reconcile_tiers.quality.fit_calibration \
    --ratings "$SCRIPT_DIR/.context/roof_ratings.json" \
    --pipeline-dir "$SCRIPT_DIR/pipeline-outputs" \
    || echo "warning: roof rating calibration failed; continuing with existing calibration"
fi

python -m reconcile_tiers.cli --all --force

(sleep 2 && open "http://localhost:${PORT}/reconcile_tiers/web/viewer-tiers.html") &

exec python -m reconcile_tiers.server
