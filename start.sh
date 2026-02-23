#!/bin/bash
# Start script untuk QLoRA full-stack deployment

set -e

echo "Starting QLoRA Application..."

# Start nginx in background
echo "Starting nginx..."
nginx -g "daemon off;" &
NGINX_PID=$!

# Start backend server
echo "Starting backend server..."
cd /app
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4 &
BACKEND_PID=$!

# Function to cleanup processes
cleanup() {
    echo "Stopping application..."
    kill -TERM $BACKEND_PID $NGINX_PID 2>/dev/null || true
    wait $BACKEND_PID $NGINX_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Wait for processes
echo "Application started successfully"
echo "Frontend: http://localhost"
echo "Backend API: http://localhost:8000"
echo "Health checks: http://localhost/health and http://localhost:8000/health"

# Keep script running
wait