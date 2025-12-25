#!/bin/bash

# Setup Script for RealityGame on Linux/macOS

echo "Setting up RealityGame..."

# 1. Locate Unreal Engine
UE_PATH=""

if [ -n "$UE5_ROOT" ]; then
    UE_PATH="$UE5_ROOT"
    echo "Found UE5_ROOT environment variable: $UE_PATH"
elif [ -d "/Users/Shared/Epic Games/UE_5.3" ]; then
    UE_PATH="/Users/Shared/Epic Games/UE_5.3"
    echo "Found standard macOS path: $UE_PATH"
elif [ -d "/Users/Shared/Epic Games/UE_5.2" ]; then
    UE_PATH="/Users/Shared/Epic Games/UE_5.2"
    echo "Found standard macOS path: $UE_PATH"
elif [ -d "/opt/unreal-engine" ]; then
    UE_PATH="/opt/unreal-engine"
    echo "Found standard Linux path: $UE_PATH"
else
    echo "Error: Could not locate Unreal Engine installation."
    echo "Please set the UE5_ROOT environment variable to your Unreal Engine directory."
    exit 1
fi

# Validate path
if [ ! -f "$UE_PATH/Engine/Build/BatchFiles/Mac/Build.sh" ] && [ ! -f "$UE_PATH/Engine/Build/BatchFiles/Linux/Build.sh" ]; then
     echo "Error: Build scripts not found in $UE_PATH. Please check the path."
     exit 1
fi

# Detect OS
OS_NAME=$(uname)
BUILD_SCRIPT=""
GEN_PROJECT_SCRIPT=""
EDITOR_BIN=""
PLATFORM_ARG=""

if [ "$OS_NAME" == "Darwin" ]; then
    echo "Detected macOS."
    BUILD_SCRIPT="$UE_PATH/Engine/Build/BatchFiles/Mac/Build.sh"
    GEN_PROJECT_SCRIPT="$UE_PATH/Engine/Build/BatchFiles/Mac/GenerateProjectFiles.sh"
    EDITOR_BIN="$UE_PATH/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
    PLATFORM_ARG="Mac"
else
    echo "Detected Linux."
    BUILD_SCRIPT="$UE_PATH/Engine/Build/BatchFiles/Linux/Build.sh"
    GEN_PROJECT_SCRIPT="$UE_PATH/Engine/Build/BatchFiles/Linux/GenerateProjectFiles.sh"
    EDITOR_BIN="$UE_PATH/Engine/Binaries/Linux/UnrealEditor"
    PLATFORM_ARG="Linux"
fi

# 2. Generate Project Files
echo "Generating Project Files..."
"$GEN_PROJECT_SCRIPT" -project="$(pwd)/RealityGame.uproject" -game

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate project files."
    exit 1
fi

# 3. Build Project
echo "Building RealityGameEditor..."
"$BUILD_SCRIPT" RealityGameEditor "$PLATFORM_ARG" Development -project="$(pwd)/RealityGame.uproject" -waitmutex

if [ $? -ne 0 ]; then
    echo "Error: Build failed."
    exit 1
fi

echo "Build Successful!"

# 4. Launch Editor (Optional)
read -p "Do you want to launch the Unreal Editor? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Launching Editor..."
    "$EDITOR_BIN" "$(pwd)/RealityGame.uproject" &
fi
