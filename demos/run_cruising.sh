#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p demos/recordings demos/telemetry
TS="$(date +%Y%m%d_%H%M%S)"
OUT="demos/recordings/Cruising_${TS}.mp4"
echo "Fly-By-Wire Cruising → ${OUT}"
python src/carla_pilot.py \
  --launch \
  --opendrive-preset straight \
  --use-neuron-throttle \
  --chase-dynamic \
  --record-camera chase_low \
  --record-video "${OUT}" \
  --record-max-seconds 22 \
  --cam-w 640 \
  --cam-h 360 \
  --record-w 1920 \
  --record-h 1080 \
  --print-every 30
