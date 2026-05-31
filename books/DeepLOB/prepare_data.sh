#!/usr/bin/env bash
set -euo pipefail

DATA_URL="https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip"
OUTPUT_DIR="${1:-$PWD/data}"
ZIP_PATH="$OUTPUT_DIR/data.zip"

if [[ -d "$OUTPUT_DIR" && "${FORCE:-0}" != "1" ]]; then
  echo "data/ already exists, skip download."
  exit 0
fi

command -v wget >/dev/null 2>&1 || {
  echo "wget is required but was not found." >&2
  exit 1
}

command -v unzip >/dev/null 2>&1 || {
  echo "unzip is required but was not found." >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"

echo "Downloading data to: $ZIP_PATH"
wget --progress=bar:force:noscroll -O "$ZIP_PATH" "$DATA_URL"

unzip -q -o "$ZIP_PATH" -d "$OUTPUT_DIR"

if [[ -d "$OUTPUT_DIR/data" ]]; then
  for item in "$OUTPUT_DIR/data"/*; do
    [[ -e "$item" ]] || continue
    mv -f "$item" "$OUTPUT_DIR/"
  done
  rmdir "$OUTPUT_DIR/data"
fi

echo "Data prepared in data/."
