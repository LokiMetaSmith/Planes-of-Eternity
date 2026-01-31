@echo off
setlocal enabledelayedexpansion
REM Build script that sets up Visual Studio environment and builds with wasm-pack

echo Setting up Visual Studio environment...

REM Find Visual Studio
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
if not exist "%VS_PATH%" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional"
)
if not exist "%VS_PATH%" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
)

if not exist "%VS_PATH%" (
    echo Visual Studio not found. Please install Visual Studio with C++ Build Tools.
    exit /b 1
)

echo Found Visual Studio at: %VS_PATH%

REM Set up Visual Studio environment
call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo Failed to set up Visual Studio environment
    exit /b 1
)

REM Remove Cygwin from PATH after vcvarsall sets it up
set "CLEAN_PATH="
for %%p in ("%PATH:;=" "%") do (
    echo %%p | findstr /i /c:"cygwin" >nul
    if errorlevel 1 (
        if defined CLEAN_PATH (
            set "CLEAN_PATH=!CLEAN_PATH!;%%p"
        ) else (
            set "CLEAN_PATH=%%p"
        )
    )
)
set "PATH=!CLEAN_PATH!"

REM Ensure LIB includes Windows SDK paths (vcvarsall should set this, but ensure it's there)
if not defined LIB (
    set "LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64"
)
REM Add Windows SDK to LIB if it exists
set "SDK_FOUND=0"

REM First check if LIB already contains Windows Kits (set by vcvarsall)
echo %LIB% | findstr /i /c:"Windows Kits" >nul
if not errorlevel 1 (
    set "SDK_FOUND=1"
    echo Windows SDK detected in LIB environment variable.
)

REM If not found, try to find it manually
if "!SDK_FOUND!"=="0" (
    if exist "C:\Program Files (x86)\Windows Kits\10\Lib" (
        set "SDK_VERSION="
        REM Loop through all to get the last one (highest version)
        for /f "delims=" %%d in ('dir /b /ad "C:\Program Files (x86)\Windows Kits\10\Lib" 2^>nul') do (
            set "SDK_VERSION=%%d"
        )

        if defined SDK_VERSION (
            echo Found Windows SDK version: !SDK_VERSION! - Adding to LIB...
            set "LIB=!LIB!;C:\Program Files (x86)\Windows Kits\10\Lib\!SDK_VERSION!\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\!SDK_VERSION!\ucrt\x64"
            set "SDK_FOUND=1"
        )
    )
)

if "!SDK_FOUND!"=="0" (
    echo.
    echo **********************************************************************
    echo * ERROR: Windows SDK not found!                                      *
    echo *                                                                    *
    echo * The build requires 'kernel32.lib' which is part of the Windows SDK.*
    echo *                                                                    *
    echo * Please run the installation script in PowerShell:                  *
    echo *     ..\install_windows_sdk.ps1                                     *
    echo *                                                                    *
    echo * Or install "Windows 10/11 SDK" via Visual Studio Installer.        *
    echo **********************************************************************
    echo.
    exit /b 1
)

echo Building with wasm-pack...
wasm-pack build --target web

if errorlevel 1 (
    echo Build failed
    exit /b 1
)

echo Build successful!

