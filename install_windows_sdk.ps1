# Script to install Windows SDK for Visual Studio 2022
# This is required for building Rust projects that use the MSVC toolchain

Write-Host "Installing Windows SDK for Visual Studio 2022..."
Write-Host ""

$vsInstallerPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vs_installer.exe"

if (-not (Test-Path $vsInstallerPath)) {
    Write-Error "Visual Studio Installer not found at: $vsInstallerPath"
    Write-Host ""
    Write-Host "Please install Visual Studio 2022 first from:"
    Write-Host "https://visualstudio.microsoft.com/downloads/"
    exit 1
}

Write-Host "Found Visual Studio Installer at: $vsInstallerPath"
Write-Host ""
Write-Host "Installing Windows 10/11 SDK..."
Write-Host "This may take several minutes and may require administrator privileges."
Write-Host ""

# Try to install the Windows SDK
# Note: This may require admin rights
try {
    $process = Start-Process -FilePath $vsInstallerPath -ArgumentList @(
        "modify",
        "--installPath", "`"C:\Program Files\Microsoft Visual Studio\2022\Community`"",
        "--add", "Microsoft.VisualStudio.Component.Windows10SDK",
        "--quiet",
        "--norestart"
    ) -Wait -PassThru -Verb RunAs

    if ($process.ExitCode -eq 0) {
        Write-Host "Windows SDK installation completed successfully!"
    } else {
        Write-Warning "Installation may have failed. Exit code: $($process.ExitCode)"
        Write-Host ""
        Write-Host "You may need to install the Windows SDK manually:"
        Write-Host "1. Open Visual Studio Installer"
        Write-Host "2. Click 'Modify' on Visual Studio 2022"
        Write-Host "3. Go to 'Individual components' tab"
        Write-Host "4. Search for 'Windows 10 SDK' or 'Windows 11 SDK'"
        Write-Host "5. Select the latest version and click 'Modify'"
    }
} catch {
    Write-Error "Failed to install Windows SDK: $_"
    Write-Host ""
    Write-Host "Please install the Windows SDK manually using Visual Studio Installer."
}

Write-Host ""
Write-Host "After installation, verify the SDK is installed:"
Write-Host "Test-Path `"C:\Program Files (x86)\Windows Kits\10\Lib`""

