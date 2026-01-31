Write-Host "Checking environment for Reality Engine development..."

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

Write-Host "Environment setup complete!"
