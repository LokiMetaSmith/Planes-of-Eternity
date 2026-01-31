Write-Host "Checking environment for Reality Engine development..."

# Check for conflicting link.exe (GNU vs MSVC)
$linkCmd = Get-Command link -ErrorAction SilentlyContinue
if ($linkCmd) {
    # Heuristic: If path contains "Git", "MinGW", or "usr\bin", it's likely the GNU linker which conflicts with Rust/MSVC
    if ($linkCmd.Source -match "Git" -or $linkCmd.Source -match "MinGW" -or $linkCmd.Source -match "usr\\bin") {
        Write-Warning "⚠️  POTENTIAL CONFLICT DETECTED ⚠️"
        Write-Warning "The 'link' command in your PATH appears to be from Git/MinGW:"
        Write-Warning "   $($linkCmd.Source)"
        Write-Warning "This will likely cause 'link: extra operand' errors during build."
        Write-Warning ""
        Write-Warning "SOLUTION:"
        Write-Warning "1. Run this script from the 'Developer Command Prompt for VS'."
        Write-Warning "2. Or ensure Visual Studio C++ Build Tools are installed and their paths precede Git in your PATH."
        Write-Warning "----------------------------------------------------------------"
    }
}

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Rust (cargo) is not installed or not in your PATH."
    Write-Host "Please install Rust from https://win.rustup.rs/"
    Write-Host "After installation, restart your terminal and run this script again."
    exit 1
}

Write-Host "Rust is installed."

if (-not (Get-Command wasm-pack -ErrorAction SilentlyContinue)) {
    Write-Host "wasm-pack is missing. Installing pre-built binary (v0.13.0)..."

    $version = "v0.13.0"
    $downloadUrl = "https://github.com/rustwasm/wasm-pack/releases/download/$version/wasm-pack-$version-x86_64-pc-windows-msvc.tar.gz"
    $tempDir = [System.IO.Path]::GetTempPath()
    $archivePath = Join-Path $tempDir "wasm-pack.tar.gz"

    # Download
    Write-Host "Downloading from $downloadUrl..."
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath -ErrorAction Stop
    } catch {
        Write-Error "Failed to download wasm-pack: $_"
        exit 1
    }

    # Extract
    Write-Host "Extracting..."
    try {
        # Using tar which is available in Windows 10/11
        tar -xf $archivePath -C $tempDir
    } catch {
        Write-Error "Failed to extract archive. Ensure tar is available or extract manually."
        exit 1
    }

    # Move binary
    $extractedDir = Join-Path $tempDir "wasm-pack-$version-x86_64-pc-windows-msvc"
    $sourceExe = Join-Path $extractedDir "wasm-pack.exe"
    $targetDir = "$env:USERPROFILE\.cargo\bin"

    if (-not (Test-Path $targetDir)) {
         Write-Host "Creating $targetDir..."
         New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    }

    Write-Host "Installing to $targetDir..."
    if (Test-Path (Join-Path $targetDir "wasm-pack.exe")) {
        Remove-Item -Force (Join-Path $targetDir "wasm-pack.exe")
    }
    Move-Item -Force -Path $sourceExe -Destination $targetDir

    # Cleanup
    Remove-Item $archivePath -Force -ErrorAction SilentlyContinue
    Remove-Item $extractedDir -Recurse -Force -ErrorAction SilentlyContinue

    Write-Host "wasm-pack installed successfully!"
} else {
    Write-Host "wasm-pack is already installed."
}

Write-Host "Checking for Windows SDK..."
$sdkPath = "${env:ProgramFiles(x86)}\Windows Kits\10\Lib"
if (-not (Test-Path $sdkPath)) {
    Write-Warning "⚠️  WINDOWS SDK NOT FOUND ⚠️"
    Write-Warning "The Windows SDK appears to be missing. This is required for building with MSVC."
    Write-Warning "Without it, you may encounter linker errors (e.g., missing kernel32.lib)."
    Write-Warning ""
    Write-Warning "To install it automatically, run:"
    Write-Warning "    .\install_windows_sdk.ps1"
    Write-Warning ""
} else {
    Write-Host "Windows SDK found."
}

Write-Host "Environment setup complete!"
