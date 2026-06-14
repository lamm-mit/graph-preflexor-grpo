#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$ROOT/frontend"
API_PORT="8765"
VITE_PORT="5177"
DEV="0"
SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      DEV="1"
      shift
      ;;
    --vite-port)
      VITE_PORT="${2:-5177}"
      shift 2
      ;;
    --port)
      API_PORT="${2:-8765}"
      SERVER_ARGS+=("$1" "$API_PORT")
      shift 2
      ;;
    *)
      SERVER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "$FRONTEND/node_modules" ]]; then
  npm --prefix "$FRONTEND" install
fi

if [[ "$DEV" == "1" ]]; then
  PYTHONUNBUFFERED=1 GRAPH_EXPLORER_API_PORT="$API_PORT" python "$ROOT/server.py" "${SERVER_ARGS[@]}" &
  API_PID="$!"
  trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM
  sleep 0.8
  if ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID"
    exit $?
  fi
  GRAPH_EXPLORER_API_PORT="$API_PORT" GRAPH_EXPLORER_VITE_PORT="$VITE_PORT" npm --prefix "$FRONTEND" run dev
else
  npm --prefix "$FRONTEND" run build
  python "$ROOT/server.py" "${SERVER_ARGS[@]}"
fi
