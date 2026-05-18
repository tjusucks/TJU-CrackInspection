#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_PATH="${1:-${INPUT_PATH:-}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-$HOME/tests/crack_size_results}}"
CONDA_ENV="${CONDA_ENV:-crack}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-$HOME/som_component_annotation/som_crack_classifier.joblib}"
CLASSIFIER_THRESHOLD="${CLASSIFIER_THRESHOLD:-}"
MM_PER_PIXEL="${MM_PER_PIXEL:-}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: $0 <input-image> [output-dir]"
  echo
  echo "Optional environment variables:"
  echo "  CONDA_ENV=$CONDA_ENV"
  echo "  CLASSIFIER_MODEL=$CLASSIFIER_MODEL"
  echo "  CLASSIFIER_THRESHOLD=$CLASSIFIER_THRESHOLD"
  echo "  MM_PER_PIXEL=$MM_PER_PIXEL"
  exit 1
fi

ARGS=(
  --input "$INPUT_PATH"
  --output-dir "$OUTPUT_DIR"
  --classifier-model "$CLASSIFIER_MODEL"
)

if [[ -n "$CLASSIFIER_THRESHOLD" ]]; then
  ARGS+=(--classifier-threshold "$CLASSIFIER_THRESHOLD")
fi

if [[ -n "$MM_PER_PIXEL" ]]; then
  ARGS+=(--mm-per-pixel "$MM_PER_PIXEL")
fi

conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/detect_crack_size.py" "${ARGS[@]}"
