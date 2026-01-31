# Build script that fixes PATH to use MSVC linker instead of Cygwin

Write-Host "Setting up build environment..."

# Find Visual Studio installation
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"
)

$vsPath = $null
foreach ($path in $vsPaths) {
    if (Test-Path $path) {
        $vsPath = $path
        break
    }
}

if (-not $vsPath) {
    Write-Error "Visual Studio not found. Please install Visual Studio with C++ Build Tools."
    exit 1
}

Write-Host "Found Visual Studio at: $vsPath"

# Remove Cygwin and other conflicting paths from PATH
$newPath = ($env:PATH -split ';' | Where-Object { 
    $_ -notmatch 'cygwin' -and 
    $_ -notmatch 'Git\\usr\\bin' -and
    $_ -notmatch 'MinGW'
}) -join ';'

$env:PATH = $newPath

# Use vcvarsall.bat to set up the environment
$vcvarsPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
if (Test-Path $vcvarsPath) {
    Write-Host "Setting up Visual Studio environment..."
    # Call vcvarsall.bat and capture the environment
    $tempFile = [System.IO.Path]::GetTempFileName()
    cmd /c "`"$vcvarsPath`" x64 > `"$tempFile`" 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
    Remove-Item $tempFile -ErrorAction SilentlyContinue
} else {
    Write-Warning "vcvarsall.bat not found. Attempting manual PATH setup..."
    # Fallback: try to find and add common paths
    $linkerPath = Get-ChildItem $vsPath -Recurse -Filter "link.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($linkerPath) {
        $linkerDir = $linkerPath.DirectoryName
        $env:PATH = "$linkerDir;$env:PATH"
    }
}

Write-Host "Building with wasm-pack..."
wasm-pack build --target web

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!"
} else {
    Write-Error "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

