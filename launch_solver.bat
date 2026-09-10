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

    REM Upgrading pip itself is a nice-to-have, not a requirement - on a
    REM locked-down account this step can fail on its own for the same
    REM permissions reason as the real install below, so don't treat a
    REM failure here as fatal; just try and move on.
    %PYEXE% -m pip install --upgrade pip >nul 2>&1

    %PYEXE% -m pip install easyocr opencv-python numpy sympy pyautogui mss pillow pynput certifi
    if errorlevel 1 (
        echo.
        echo System-wide install failed. This is almost always a permissions
        echo issue, not a real error - it usually means this Windows account
        echo doesn't have write access to Python's shared install folder,
        echo which is common on school/lab computers without admin rights.
        echo Retrying with --user, which installs into your OWN profile
        echo instead and never needs admin access...
        echo.
        %PYEXE% -m pip install --user easyocr opencv-python numpy sympy pyautogui mss pillow pynput certifi
        if errorlevel 1 (
            echo.
            echo Dependency installation failed even with --user - check the
            echo error above and your internet connection, then re-run this
            echo script.
            echo.
            pause
            exit /b 1
        )
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
