@echo off
setlocal

echo Setting up RealityGame...

:: 1. Locate Unreal Engine
set "UE_PATH="

if defined UE5_ROOT (
    set "UE_PATH=%UE5_ROOT%"
    echo Found UE5_ROOT environment variable: %UE_PATH%
) else if exist "C:\Program Files\Epic Games\UE_5.3" (
    set "UE_PATH=C:\Program Files\Epic Games\UE_5.3"
    echo Found standard path: %UE_PATH%
) else if exist "C:\Program Files\Epic Games\UE_5.2" (
    set "UE_PATH=C:\Program Files\Epic Games\UE_5.2"
    echo Found standard path: %UE_PATH%
) else (
    echo Error: Could not locate Unreal Engine installation.
    echo Please set the UE5_ROOT environment variable to your Unreal Engine directory.
    goto :error
)

set "UBT=%UE_PATH%\Engine\Binaries\DotNET\UnrealBuildTool\UnrealBuildTool.exe"
set "EDITOR=%UE_PATH%\Engine\Binaries\Win64\UnrealEditor.exe"

if not exist "%UBT%" (
    echo Error: UnrealBuildTool not found at %UBT%
    goto :error
)

:: 2. Generate Project Files
echo Generating Project Files...
"%UBT%" -projectfiles -project="%~dp0RealityGame.uproject" -game -rocket -progress

if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to generate project files.
    goto :error
)

:: 3. Build Project
echo Building RealityGameEditor...
"%UBT%" RealityGameEditor Win64 Development -project="%~dp0RealityGame.uproject" -waitmutex

if %ERRORLEVEL% NEQ 0 (
    echo Error: Build failed.
    goto :error
)

echo Build Successful!

:: 4. Launch Editor (Optional)
set /P "LAUNCH=Do you want to launch the Unreal Editor? (y/n) "
if /I "%LAUNCH%"=="y" (
    echo Launching Editor...
    start "" "%EDITOR%" "%~dp0RealityGame.uproject"
)

goto :eof

:error
pause
exit /b 1
