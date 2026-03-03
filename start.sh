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

# Setup Python virtual environment
setup_venv() {
    local python_cmd=$1
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        $python_cmd -m venv venv
    fi

    # Activate venv depending on OS
    if [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
}

# Python 3 is preferred
if python3 -c "pass" &> /dev/null; then
    setup_venv python3
    python3 -m http.server 8000 &
    HTTP_PID=$!
elif python -c "pass" &> /dev/null; then
    setup_venv python
    python -m http.server 8000 &
    HTTP_PID=$!
else
    echo "Error: Python is not installed or not working correctly. Cannot start HTTP server."
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
