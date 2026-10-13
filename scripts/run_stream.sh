#!/usr/bin/env bash
# Run the streaming pipeline once: fresh topics -> detector consumer -> producer.
#   replay: scripts/run_stream.sh replay MANIFEST IMAGES_DIR LOG WEIGHTS
#   live:   scripts/run_stream.sh live   UNTIL(YYYYmmddHHMM) - LOG WEIGHTS   (needs LTA_KEY)
# Redpanda runs from docker-compose.yml inside WSL2 here. Windows-side clients could not use the
# localhost relay reliably, so the broker advertises the WSL VM address:
#   TCAM_ADVERTISE_HOST=<wsl ip> docker compose up -d ;  TCAM_BOOTSTRAP=<wsl ip>:39092
set -euo pipefail
MODE=$1; ARG=$2; IMAGES=$3; LOG=$4; WEIGHTS=$5
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
RPK=${RPK:-"wsl.exe -d Ubuntu-24.04 -- docker exec tcam-redpanda rpk"}
THRESH=${THRESH:-$ROOT/data/thresholds.json}

$RPK topic delete tcam.frames tcam.camera-state >/dev/null 2>&1 || true
$RPK topic create tcam.frames tcam.camera-state -c max.message.bytes=4000000 >/dev/null
rm -f "$LOG"
TH_ARG=""; [ -f "$THRESH" ] && TH_ARG="--thresholds $THRESH"
export PYTHONPATH="$ROOT/src"
export TCAM_BOOTSTRAP=${TCAM_BOOTSTRAP:-127.0.0.1:39092}
$PY -u -m tcam.stream.detector --weights "$WEIGHTS" --imgsz 1280 --batch 4 --log "$LOG" \
    --group "det-$(date +%s)" --idle-exit "${IDLE_EXIT:-60}" $TH_ARG &
DET=$!
sleep 25   # model load + warm-up
if [ "$MODE" = replay ]; then
  $PY -u -m tcam.stream.producer replay --manifest "$ARG" --images "$IMAGES"
else
  $PY -u -m tcam.stream.producer live --every 30 --until "$ARG"
fi
wait $DET
