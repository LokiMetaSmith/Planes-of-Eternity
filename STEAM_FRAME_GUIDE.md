# Steam Frame Integration Guide

This guide explains how to compile the reality-engine to run natively on the Steam Frame.

Steam Frame runs SteamOS on a Snapdragon 8 Gen 3 (ARM64) processor. The best way to run highly optimized native code is to compile a Linux ARM64 binary using Tauri.

## Building for Steam Frame (Native Linux ARM64)

Since compiling a Tauri application requires specific C-libraries (GTK+, WebKit) that are tricky to cross-compile from an x86 host, we've provided a Docker-based build script that ensures a clean, isolated compilation environment.

1. Ensure you have Docker installed on your host machine.
2. Run the provided build script from the root of the repository:
   ```bash
   ./build_linux_arm64.sh
   ```
3. This script will:
   - Set up an isolated Ubuntu Docker container with an `aarch64` multiarch configuration.
   - Install all required `arm64` GTK+ and WebKit development libraries.
   - Build the `reality-engine` WebAssembly frontend using Trunk.
   - Cross-compile the `reality-app` Tauri wrapper targeting `aarch64-unknown-linux-gnu`.
   - Extract the built binary to the host directory.
4. Once completed, the binary will be output to:
   `./reality-app-steam-frame`

You can then package this binary and upload it as a Linux build to Steamworks, following the standard Steam hardware guidelines for Steam Frame.
