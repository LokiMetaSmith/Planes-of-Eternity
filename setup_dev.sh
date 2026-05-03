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

if ! command -v trunk &> /dev/null; then
    echo "trunk is missing. Downloading pre-compiled release..."
    TRUNK_VERSION="v0.21.14"
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    if [ "$OS" = "linux" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            TARGET="x86_64-unknown-linux-gnu"
        elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
            TARGET="aarch64-unknown-linux-gnu"
        else
            echo "Unsupported architecture: $ARCH"
            exit 1
        fi
    elif [ "$OS" = "darwin" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            TARGET="x86_64-apple-darwin"
        elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
            TARGET="aarch64-apple-darwin"
        else
            echo "Unsupported architecture: $ARCH"
            exit 1
        fi
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi

    TRUNK_URL="https://github.com/trunk-rs/trunk/releases/download/${TRUNK_VERSION}/trunk-${TARGET}.tar.gz"
    echo "Downloading trunk from ${TRUNK_URL}"
    mkdir -p "$HOME/.cargo/bin"
    curl -sL "${TRUNK_URL}" | tar -xzf - -C "$HOME/.cargo/bin" trunk

    echo "trunk installed successfully!"
else
    echo "trunk is already installed."
fi

echo "Environment setup complete!"
