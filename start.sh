#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Shutting down servers..."
    if [ ! -z "$SIGNAL_PID" ]; then
        kill $SIGNAL_PID
    fi
    if [ ! -z "$HTTP_PID" ]; then
        kill $HTTP_PID
    fi
    exit
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

echo "=== Reality Engine Startup Script ==="

# 1. Build the Engine (WASM)
echo "[1/3] Building Reality Engine (WASM)..."
cd reality-engine
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed. Please run setup_dev.sh first."
    exit 1
fi
wasm-pack build --target web
if [ $? -ne 0 ]; then
    echo "Error: Build failed."
    exit 1
fi
cd ..

# 2. Start Signaling Server
echo "[2/3] Starting Signaling Server..."
cd reality-signal-server
cargo run -q &
SIGNAL_PID=$!
cd ..

# Wait a moment for the server to initialize
sleep 2

# 3. Start HTTP Server
echo "[3/3] Starting HTTP Server..."
# Python 3 is preferred
if command -v python3 &> /dev/null; then
    python3 -m http.server 8000 &
    HTTP_PID=$!
elif command -v python &> /dev/null; then
    python -m http.server 8000 &
    HTTP_PID=$!
else
    echo "Error: Python is not installed. Cannot start HTTP server."
    cleanup
    exit 1
fi

echo "========================================="
echo "All systems operational!"
echo "Open your browser to: http://localhost:8000/reality-engine/index.html"
echo "Press Ctrl+C to stop."
echo "========================================="

# Keep script running to maintain background processes
wait
