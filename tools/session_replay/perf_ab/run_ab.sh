#!/usr/bin/env bash
# Recorder on/off A/B over the ui_crawler story suite (spec phase 5).
#
# Runs the full 36-story suite N times per arm, alternating O-N-O-N so
# cache/thermal drift hits both arms equally. Each run gets a fresh isolated
# storage dir and its own server on $PORT; arms differ ONLY in GENTLY_REPLAY.
#
# Usage: bash tools/session_replay/perf_ab/run_ab.sh [reps]
# Run from the repo root. Compare with compare.py afterwards.

set -u
REPS="${1:-3}"
PORT="${PORT:-8090}"
PY="${PY:-/home/dna/lab/projects/gently/.venv/bin/python}"
OUT_ROOT="tools/ui_crawler/out/ab"
WORK="${TMPDIR:-/tmp}/gently-replay-ab"

mkdir -p "$OUT_ROOT" "$WORK"

wait_port_up() {
  for _ in $(seq 1 120); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/" && return 0
    sleep 1
  done
  return 1
}

wait_port_down() {
  for _ in $(seq 1 30); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/" || return 0
    sleep 1
  done
  return 1
}

run_one() {
  local arm="$1" rep="$2" replay_flag="$3"
  local tag="${arm}${rep}"
  local storage="$WORK/storage-$tag"
  local out="$OUT_ROOT/$tag"
  rm -rf "$storage" "$out"
  echo "=== [$tag] server up (GENTLY_REPLAY=$replay_flag) ==="
  GENTLY_REPLAY="$replay_flag" GENTLY_VIZ_PORT="$PORT" GENTLY_STORAGE_PATH="$storage" \
    "$PY" launch_gently.py --no-api --no-auth --no-browser --offline \
    > "$WORK/server-$tag.log" 2>&1 &
  local server_pid=$!
  if ! wait_port_up; then
    echo "!!! [$tag] server never came up (log: $WORK/server-$tag.log)"
    kill "$server_pid" 2>/dev/null
    return 1
  fi
  echo "=== [$tag] stories ==="
  local t0=$SECONDS
  "$PY" tools/ui_crawler/run_stories.py --url "http://127.0.0.1:$PORT" --out "$out" \
    > "$WORK/stories-$tag.log" 2>&1
  echo "[$tag] stories exit=$? wall=$((SECONDS - t0))s"
  kill "$server_pid" 2>/dev/null
  wait "$server_pid" 2>/dev/null
  wait_port_down || echo "!!! [$tag] port did not free"
}

for rep in $(seq 1 "$REPS"); do
  run_one off "$rep" 0
  run_one on "$rep" 1
done
echo "A/B complete → $OUT_ROOT"
