#!/bin/bash
set -e

echo "Setting up Docker cross-compilation for Tauri ARM64..."
cat << 'DOCKERFILE' > Dockerfile.cross
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    pkg-config \
    libssl-dev \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    dpkg-dev

RUN dpkg --add-architecture arm64
RUN sed -i 's/deb http/deb [arch=amd64] http/g' /etc/apt/sources.list
RUN echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy main restricted universe multiverse" > /etc/apt/sources.list.d/arm64.list
RUN echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list.d/arm64.list
RUN echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-security main restricted universe multiverse" >> /etc/apt/sources.list.d/arm64.list

RUN apt-get update && apt-get install -y \
    libwebkit2gtk-4.1-dev:arm64 \
    libgtk-3-dev:arm64 \
    libsoup-3.0-dev:arm64 \
    libayatana-appindicator3-dev:arm64 \
    librsvg2-dev:arm64 \
    libglib2.0-dev:arm64

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup target add aarch64-unknown-linux-gnu
RUN rustup target add wasm32-unknown-unknown
RUN cargo install trunk

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y nodejs

# Environment variables for cross compilation
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
ENV PKG_CONFIG_ALLOW_CROSS=1
ENV PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig
ENV PKG_CONFIG_SYSROOT_DIR=/

WORKDIR /app
COPY . .

WORKDIR /app/reality-engine
RUN trunk build --release

WORKDIR /app/reality-app
RUN npm install
# Run build targeting aarch64, skipping AppImage to avoid FUSE issues inside Docker
RUN npm run tauri build -- --target aarch64-unknown-linux-gnu --bundles deb
DOCKERFILE

echo "Building Docker image and compiling reality-app..."
docker build -t reality-app-cross -f Dockerfile.cross .

echo "Extracting the compiled binary..."
CONTAINER_ID=$(docker create reality-app-cross)
# The binary is named 'reality-engine' because of productName in tauri.conf.json
docker cp ${CONTAINER_ID}:/app/reality-app/src-tauri/target/aarch64-unknown-linux-gnu/release/reality-engine ./reality-app-steam-frame
docker rm ${CONTAINER_ID}

echo "Binary available at: ./reality-app-steam-frame"
