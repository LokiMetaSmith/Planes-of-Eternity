# Android Build Environment Setup for CI/CD

To build the Tauri Android APK in the CI/CD sandbox environment, we must configure the Android Command Line Tools, NDK, and Tauri.
This is because a full Android Studio graphical installation is unavailable in the headless sandbox environment.

## 1. Download and Install Android Command Line Tools
```bash
# Setup paths
export ANDROID_HOME=$HOME/android-sdk
export NDK_HOME=$ANDROID_HOME/ndk/25.2.9519653
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin

# Create directories
mkdir -p $ANDROID_HOME/cmdline-tools
cd $ANDROID_HOME/cmdline-tools

# Download latest command line tools
wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip -q commandlinetools-linux-11076708_latest.zip
mv cmdline-tools latest
rm commandlinetools-linux-11076708_latest.zip
```

## 2. Accept SDK Licenses and Install Packages
```bash
yes | sdkmanager --licenses
yes | sdkmanager --install "platforms;android-33" "build-tools;33.0.2" "ndk;25.2.9519653"
```

## 3. Configure Rust Target
```bash
rustup target add aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android
```

## 4. Install Tauri
```bash
# Assuming node and npm are installed
cd reality-app
npm install @tauri-apps/cli@2 @tauri-apps/api@2 @tauri-apps/plugin-log@2
```

## 5. Build the Android APK
```bash
cd reality-app
npx tauri android build
```
