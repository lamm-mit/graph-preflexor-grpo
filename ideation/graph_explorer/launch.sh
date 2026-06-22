#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$ROOT/frontend"
API_PORT="8765"
VITE_PORT="5177"
DEV="0"
KILL_STALE="0"
SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      DEV="1"
      shift
      ;;
    --kill-stale)
      KILL_STALE="1"
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

port_pids() {
  lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
}

describe_port() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
}

stop_port() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(port_pids "$port" | sort -u | tr '\n' ' ')"
  if [[ -z "${pids// }" ]]; then
    return 0
  fi
  echo "[launch] stopping stale $label listener(s) on port $port: $pids"
  kill $pids 2>/dev/null || true
  sleep 0.5
}

require_port_free() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(port_pids "$port" | sort -u | tr '\n' ' ')"
  if [[ -z "${pids// }" ]]; then
    return 0
  fi
  if [[ "$KILL_STALE" == "1" ]]; then
    stop_port "$port" "$label"
    pids="$(port_pids "$port" | sort -u | tr '\n' ' ')"
    if [[ -z "${pids// }" ]]; then
      return 0
    fi
  fi
  echo "[launch] port $port is already in use by $label listener(s):" >&2
  describe_port "$port" >&2
  echo "[launch] stop that process or rerun with --kill-stale." >&2
  exit 1
}

if [[ ! -d "$FRONTEND/node_modules" ]]; then
  npm --prefix "$FRONTEND" install
fi

if [[ "$DEV" == "1" ]]; then
  require_port_free "$API_PORT" "backend"
  require_port_free "$VITE_PORT" "Vite"
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
  require_port_free "$API_PORT" "backend"
  npm --prefix "$FRONTEND" run build
  python "$ROOT/server.py" "${SERVER_ARGS[@]}"
fi
