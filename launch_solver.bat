@echo off
setlocal enabledelayedexpansion
title Generic Optical Reader ^& Math Solver
cd /d "%~dp0"

echo Starting Generic Optical Reader ^& Math Solver...
echo.

set "PYEXE="

REM Prefer 3.10-3.13 via the py launcher: opencv-python and easyocr/torch
REM currently have the most reliable prebuilt wheels on these versions.
REM (Very new releases like 3.14 sometimes lack wheels and fall back to
REM  a source build, which usually fails without a C compiler installed.)
for %%V in (3.12 3.13 3.11 3.10) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PYEXE=py -%%V"
    )
)

REM Fall back to whatever "python" resolves to on PATH.
if not defined PYEXE (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYEXE=python"
)

if not defined PYEXE (
    echo No usable Python 3.10-3.13 installation was found on this system.
    echo Download one from https://www.python.org/downloads/ and re-run this script.
    echo ^(Tick "Add python.exe to PATH" during install.^)
    echo.
    pause
    exit /b 1
)

echo Using interpreter: %PYEXE%
echo.

REM Make sure required packages are present; install anything missing.
%PYEXE% -c "import PIL, mss, numpy, cv2, sympy, easyocr, pyautogui, pynput" >nul 2>&1
if errorlevel 1 (
    echo Some dependencies are missing - installing now.
    echo This can take several minutes the first time ^(EasyOCR pulls in torch^).
    echo.
    %PYEXE% -m pip install --upgrade pip
    %PYEXE% -m pip install easyocr opencv-python numpy sympy pyautogui mss pillow pynput
    if errorlevel 1 (
        echo.
        echo Dependency installation failed - check the error above and your
        echo internet connection, then re-run this script.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Dependencies installed successfully.
    echo.
)

REM Run the GUI
%PYEXE% gui.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Error occurred! Press any key to exit...
    pause > nul
)
