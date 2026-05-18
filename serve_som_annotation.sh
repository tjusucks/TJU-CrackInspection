#!/usr/bin/env bash
set -euo pipefail

ANNOTATION_DIR="${1:-$HOME/som_component_annotation}"
PORT="${PORT:-8765}"

if [[ ! -d "$ANNOTATION_DIR" ]]; then
  echo "Annotation directory not found: $ANNOTATION_DIR"
  exit 1
fi

echo "Serving annotation files from: $ANNOTATION_DIR"
echo "Open this URL in your Windows browser:"
echo "  http://localhost:$PORT/annotate.html"
echo

cd "$ANNOTATION_DIR"
python3 -m http.server "$PORT"
