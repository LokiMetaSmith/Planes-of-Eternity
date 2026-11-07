#!/bin/bash
set -e

# Build the WebAssembly engine
cd reality-engine
trunk build --release
cd ..

# Initialize and build Android APK wrapper
cd reality-app
ANDROID_HOME=$HOME/android-sdk NDK_HOME=$HOME/android-sdk/ndk/25.2.9519653 npx tauri android build
