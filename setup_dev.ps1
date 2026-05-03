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

if (-not (Get-Command trunk -ErrorAction SilentlyContinue)) {
    Write-Host "trunk is missing. Downloading pre-compiled release..."

    $trunkVersion = "v0.21.14"
    $trunkUrl = "https://github.com/trunk-rs/trunk/releases/download/$trunkVersion/trunk-x86_64-pc-windows-msvc.zip"
    $zipPath = "$env:TEMP\trunk.zip"
    $cargoBinPath = "$env:USERPROFILE\.cargo\bin"

    Invoke-WebRequest -Uri $trunkUrl -OutFile $zipPath

    if (-not (Test-Path $cargoBinPath)) {
        New-Item -ItemType Directory -Force -Path $cargoBinPath | Out-Null
    }

    Expand-Archive -Path $zipPath -DestinationPath $cargoBinPath -Force
    Remove-Item $zipPath

    Write-Host "trunk installed successfully!"
} else {
    Write-Host "trunk is already installed."
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
