#!/bin/sh
# Full-stack startup script untuk Azure Container Apps

set -e

echo "Starting QLoRA Full-Stack Application..."

# Set default values
export WORKERS=${WORKERS:-4}
export PORT=${PORT:-8000}
export MAX_REQUESTS=${MAX_REQUESTS:-1000}
export MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}

# Create necessary directories
mkdir -p logs models checkpoints data

# Start nginx di background
echo "Starting Nginx..."
nginx -g "daemon off;" &
NGINX_PID=$!

# Wait sebentar untuk nginx start
sleep 2

# Start backend
echo "Starting Backend..."
exec gunicorn backend.main:app \
    --bind 0.0.0.0:$PORT \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests $MAX_REQUESTS \
    --max-requests-jitter $MAX_REQUESTS_JITTER \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --timeout 120 \
    --keep-alive 2 \
    --preload &

BACKEND_PID=$!

# Function untuk graceful shutdown
graceful_shutdown() {
    echo "Received shutdown signal, gracefully shutting down..."
    
    # Stop backend
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Stopping backend (PID: $BACKEND_PID)..."
        kill -TERM $BACKEND_PID
        wait $BACKEND_PID
    fi
    
    # Stop nginx
    if [ ! -z "$NGINX_PID" ]; then
        echo "Stopping nginx (PID: $NGINX_PID)..."
        kill -TERM $NGINX_PID
        wait $NGINX_PID
    fi
    
    echo "All services stopped gracefully."
    exit 0
}

# Trap signals
trap graceful_shutdown SIGTERM SIGINT

# Wait untuk semua background processes
wait
