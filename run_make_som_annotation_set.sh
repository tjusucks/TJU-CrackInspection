#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-crack}"
SOURCE_DIR="${SOURCE_DIR:-$HOME/crack_datasets/images_denoise}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/som_component_annotation}"
TARGET_COUNT="${TARGET_COUNT:-1500}"
MAX_COMPONENTS_PER_IMAGE="${MAX_COMPONENTS_PER_IMAGE:-8}"
MIN_AREA="${MIN_AREA:-20}"
PADDING="${PADDING:-48}"
SEED="${SEED:-42}"

conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/make_som_annotation_set.py" \
  --source-dir "$SOURCE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --target-count "$TARGET_COUNT" \
  --max-components-per-image "$MAX_COMPONENTS_PER_IMAGE" \
  --min-area "$MIN_AREA" \
  --padding "$PADDING" \
  --seed "$SEED"
