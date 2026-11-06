#!/usr/bin/env bash
# Run the streaming pipeline once: fresh topics -> detector consumer -> producer.
#   replay: scripts/run_stream.sh replay MANIFEST IMAGES_DIR LOG WEIGHTS
#   live:   scripts/run_stream.sh live   UNTIL(YYYYmmddHHMM) - LOG WEIGHTS   (needs LTA_KEY)
# Redpanda runs from docker-compose.yml (`docker compose up -d`); RPK is the command that reaches
# its rpk CLI. The default suits Docker on the same host. When the broker runs in WSL2 Docker and
# the clients on Windows (how the README numbers were measured), the localhost relay was not
# reliable, so the broker advertises the WSL VM address instead:
#   TCAM_BIND=0.0.0.0 TCAM_ADVERTISE_HOST=<wsl ip> docker compose up -d ;  TCAM_BOOTSTRAP=<wsl ip>:39092
#   RPK="wsl.exe -d <distro> -- docker exec tcam-redpanda rpk"
set -euo pipefail
MODE=$1; ARG=$2; IMAGES=$3; LOG=$4; WEIGHTS=$5
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
RPK=${RPK:-"docker exec tcam-redpanda rpk"}
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
