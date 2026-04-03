#!/usr/bin/env sh
set -eu

PORT_TO_USE="${PORT:-7860}"

exec uvicorn daniela_bino_backend:app --host 0.0.0.0 --port "${PORT_TO_USE}"
