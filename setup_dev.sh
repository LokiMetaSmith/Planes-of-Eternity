#!/bin/bash
set -e

echo "Checking environment for Reality Engine development..."

if ! command -v cargo &> /dev/null; then
    echo "Error: Rust (cargo) is not installed."
    echo "Please install Rust via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "Then restart your terminal."
    exit 1
fi

echo "Rust is installed."

if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack is missing. Installing..."
    cargo install wasm-pack
else
    echo "wasm-pack is already installed."
fi

echo "Environment setup complete!"
