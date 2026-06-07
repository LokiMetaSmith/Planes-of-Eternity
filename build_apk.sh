#!/bin/bash
set -e

# Build the WebAssembly engine
cd reality-engine
trunk build --release
cd ..

# Initialize and build Android APK wrapper
cd reality-app
cargo tauri build --target aarch64-linux-android
