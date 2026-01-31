# Build Instructions - Windows SDK Required

## Issue
The build is failing because the Windows SDK is not installed. This is required for Rust to compile build scripts when using the MSVC toolchain.

## Solution

### Option 1: Install via Script (Recommended)
Run the provided PowerShell script:
```powershell
.\install_windows_sdk.ps1
```

**Note:** This script may require administrator privileges. If it fails, use Option 2.

### Option 2: Manual Installation
1. Open **Visual Studio Installer** (search for it in Start Menu)
2. Click **"Modify"** on your Visual Studio 2022 installation
3. Go to the **"Individual components"** tab
4. Search for **"Windows 10 SDK"** or **"Windows 11 SDK"**
5. Select the **latest version** (e.g., 10.0.22621.0 or newer)
6. Also ensure **"Windows 10 SDK"** or **"Windows 11 SDK"** is checked in the main workloads
7. Click **"Modify"** to install

### Option 3: Install Standalone Windows SDK
If Visual Studio Installer doesn't work, you can download the Windows SDK directly:
- Download from: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
- Install the latest Windows 10/11 SDK

## Verify Installation
After installation, verify the SDK is installed:
```powershell
Test-Path "C:\Program Files (x86)\Windows Kits\10\Lib"
```

This should return `True`.

## Build After Installation
Once the Windows SDK is installed, build the project:
```powershell
cd reality-engine
.\build.ps1
```

Or use the batch file:
```cmd
cd reality-engine
build.bat
```

## Alternative: Use GNU Toolchain (Not Recommended)
If you cannot install the Windows SDK, you can switch to the GNU toolchain:
```powershell
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```

However, this may cause compatibility issues with some dependencies.

