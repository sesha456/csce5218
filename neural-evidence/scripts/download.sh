#!/usr/bin/env bash
# Download FEVER 1.0 and SciFact into $1 (default: data/raw).
set -euo pipefail
RAW=${1:-data/raw}
mkdir -p "$RAW/fever" "$RAW/scifact"
F=https://fever.ai/download/fever
for f in train.jsonl shared_task_dev.jsonl wiki-pages.zip; do
  [ -s "$RAW/fever/$f" ] || wget -q --show-progress -O "$RAW/fever/$f" "$F/$f"
done
if [ ! -s "$RAW/scifact/claims_train.jsonl" ]; then
  wget -q -O "$RAW/scifact/data.tar.gz" https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
  tar xzf "$RAW/scifact/data.tar.gz" -C "$RAW/scifact" --strip-components=1
fi
ls -lh "$RAW/fever" "$RAW/scifact"
