Write-Host "Checking environment for Reality Engine development..."

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Rust (cargo) is not installed or not in your PATH."
    Write-Host "Please install Rust from https://win.rustup.rs/"
    Write-Host "After installation, restart your terminal and run this script again."
    exit 1
}

Write-Host "Rust is installed."

if (-not (Get-Command wasm-pack -ErrorAction SilentlyContinue)) {
    Write-Host "wasm-pack is missing. Installing..."
    cargo install wasm-pack
} else {
    Write-Host "wasm-pack is already installed."
}

Write-Host "Environment setup complete!"
