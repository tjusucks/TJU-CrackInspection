#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_PATH="${1:-${INPUT_PATH:-}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-$HOME/tests/som_wall_block_results}}"

BACKEND_DIR="${BACKEND_DIR:-$HOME/CrackDetection_Backend}"
CONDA_ENV="${CONDA_ENV:-crack}"
CONFIDENCE="${CONFIDENCE:-0.7}"
MIN_CRACK_PIXELS="${MIN_CRACK_PIXELS:-25}"
MIN_CRACK_AREA_RATIO="${MIN_CRACK_AREA_RATIO:-0.0002}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-}"
CLASSIFIER_THRESHOLD="${CLASSIFIER_THRESHOLD:-}"

YOLO_MODEL="${YOLO_MODEL:-$BACKEND_DIR/model/stonewall-YOLOv8-obb.pt}"
SAM_MODEL="${SAM_MODEL:-$BACKEND_DIR/model/mobile-sam.pt}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: $0 <input-image-or-dir> [output-dir]"
  echo
  echo "Optional environment variables:"
  echo "  BACKEND_DIR=$BACKEND_DIR"
  echo "  CONDA_ENV=$CONDA_ENV"
  echo "  YOLO_MODEL=$YOLO_MODEL"
  echo "  SAM_MODEL=$SAM_MODEL"
  echo "  CONFIDENCE=$CONFIDENCE"
  echo "  MIN_CRACK_PIXELS=$MIN_CRACK_PIXELS"
  echo "  MIN_CRACK_AREA_RATIO=$MIN_CRACK_AREA_RATIO"
  echo "  CLASSIFIER_MODEL=$CLASSIFIER_MODEL"
  echo "  CLASSIFIER_THRESHOLD=$CLASSIFIER_THRESHOLD"
  exit 1
fi

ARGS=(
  --input "$INPUT_PATH"
  --output-dir "$OUTPUT_DIR"
  --backend-dir "$BACKEND_DIR"
  --yolo-model "$YOLO_MODEL"
  --sam-model "$SAM_MODEL"
  --confidence "$CONFIDENCE"
  --min-crack-pixels "$MIN_CRACK_PIXELS"
  --min-crack-area-ratio "$MIN_CRACK_AREA_RATIO"
)

if [[ -n "$CLASSIFIER_MODEL" ]]; then
  ARGS+=(--classifier-model "$CLASSIFIER_MODEL")
fi

if [[ -n "$CLASSIFIER_THRESHOLD" ]]; then
  ARGS+=(--classifier-threshold "$CLASSIFIER_THRESHOLD")
fi

conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/som_wall_block_pipeline.py" "${ARGS[@]}"
