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
echo "[1/2] Building Reality Engine (WASM)..."
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

# 2. Start Signaling & Web Server
echo "[2/2] Starting Signaling Server..."
cd reality-signal-server
cargo run -q &
SIGNAL_PID=$!
cd ..

# Wait a moment for the server to initialize
sleep 2

echo "========================================="
echo "All systems operational!"
echo "Open your browser to: http://localhost:9000/"
echo "Press Ctrl+C to stop."
echo "========================================="

# Keep script running to maintain background processes
wait
