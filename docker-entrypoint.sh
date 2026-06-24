#!/bin/sh
# Copyright 2026 Query Farm LLC - https://query.farm
#
# Dispatch the single vgi-xgboost image into one of its two transports:
#   http   (default) the HTTP server on $PORT (Fly.io / local HTTP)
#   stdio            a worker DuckDB spawns over stdio (on-host execution)
# Any other first argument is exec'd verbatim (escape hatch for debugging).
#
# Both modes share one /data volume (see the farm.query.vgi.volumes image label):
#   /data/models  -> XGBOOST_MODELS_DIR  (model registry)
set -e

# A freshly-mounted (empty) volume has no subdirs; create the registry dir so the
# non-root user can write models. Harmless when /data is unmounted.
mkdir -p "${XGBOOST_MODELS_DIR:-/data/models}"

case "${1:-http}" in
  http)
    shift 2>/dev/null || true
    exec python /app/serve.py --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  stdio)
    shift 2>/dev/null || true
    exec python /app/xgboost_worker.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
