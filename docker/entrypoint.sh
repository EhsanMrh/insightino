#!/usr/bin/env bash
set -euo pipefail

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT__STORAGE__STORAGE_PATH="${QDRANT__STORAGE__STORAGE_PATH:-/qdrant/storage}"
mkdir -p "$QDRANT__STORAGE__STORAGE_PATH"

# Start Qdrant in the background
echo "[entrypoint] Starting Qdrant..."
/usr/local/bin/qdrant > /var/log/qdrant.log 2>&1 &
QDRANT_PID=$!

# Wait for Qdrant readiness
echo "[entrypoint] Waiting for Qdrant at ${QDRANT_URL} ..."
for i in {1..120}; do
  if curl -sSf "${QDRANT_URL}/readyz" >/dev/null 2>&1; then
    echo "[entrypoint] Qdrant is ready."
    break
  fi
  sleep 1
done

cleanup() {
  echo "[entrypoint] Shutting down..."
  kill -TERM "$QDRANT_PID" 2>/dev/null || true
  wait "$QDRANT_PID" 2>/dev/null || true
}
trap cleanup TERM INT

echo "[entrypoint] Launching Gradio UI..."
python -m src.ui.gradio_app
cleanup


