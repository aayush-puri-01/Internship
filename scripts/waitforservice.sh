#!/bin/sh
# wait-for-fastapi.sh

HOST="fastapi" # Or whatever your FastAPI service's hostname is
PORT=8000

echo "Waiting for FastAPI at $HOST:$PORT..."

until nc -z "$HOST" "$PORT" 2>/dev/null; do
  echo "Waiting for FastAPI..."
  sleep 2
done

echo "FASTAPI is up - starting GRADIO app"
exec "$@"