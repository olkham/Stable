@echo off
rem Stable installer (Windows / cmd.exe).
rem
rem Run it from an OPEN command prompt so the virtual environment is still
rem active when it finishes:
rem
rem     install.bat
rem
rem Double-clicking works too, but the window closes at the end and takes the
rem activated environment with it.
rem
rem No SETLOCAL anywhere in this script: environment changes must survive back
rem into the caller's session, which is what leaves the .venv active.

pushd "%~dp0"
echo.
echo === Stable installer ===
echo.

rem ---------------------------------------------------------------- find python
set "CAND1="
set "CAND2="
set "PATH1="
set "PATH2="
set "VER1="
set "VER2="

where python >nul 2>nul
if %errorlevel%==0 set "CAND1=python"
where python3 >nul 2>nul
if %errorlevel%==0 set "CAND2=python3"

if defined CAND1 for /f "delims=" %%p in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PATH1=%%p"
if defined CAND2 for /f "delims=" %%p in ('python3 -c "import sys;print(sys.executable)" 2^>nul') do set "PATH2=%%p"

rem A bare `python3` on Windows is often the Microsoft Store stub, which is on
rem PATH but prints nothing and launches the Store instead of an interpreter.
if defined CAND2 if not defined PATH2 set "CAND2="
if defined CAND1 if not defined PATH1 set "CAND1="

if not defined CAND1 if not defined CAND2 goto :no_python

rem Both names very often point at the same interpreter; don't ask a question
rem whose two answers are identical.
if defined CAND1 if defined CAND2 if /i "%PATH1%"=="%PATH2%" set "CAND2="

if defined CAND1 for /f "delims=" %%v in ('python -c "import platform;print(platform.python_version())" 2^>nul') do set "VER1=%%v"
if defined CAND2 for /f "delims=" %%v in ('python3 -c "import platform;print(platform.python_version())" 2^>nul') do set "VER2=%%v"

if not defined CAND2 (
    set "PYEXE=%CAND1%"
    echo Using the only Python found: %CAND1% ^(%VER1%^)
    echo   %PATH1%
    goto :check_version
)
if not defined CAND1 (
    set "PYEXE=%CAND2%"
    echo Using the only Python found: %CAND2% ^(%VER2%^)
    echo   %PATH2%
    goto :check_version
)

echo Two Python interpreters are available:
echo.
echo   [1] python   %VER1%
echo       %PATH1%
echo   [2] python3  %VER2%
echo       %PATH2%
echo.
set "CHOICE="
set /p "CHOICE=Create the virtual environment with which one? [1/2] (default 1): "
if "%CHOICE%"=="2" (set "PYEXE=%CAND2%") else (set "PYEXE=%CAND1%")
echo.
echo Using %PYEXE%.

:check_version
rem Stable needs 3.11+ (pyproject: requires-python = ">=3.11").
"%PYEXE%" -c "import sys;sys.exit(0 if sys.version_info>=(3,11) else 1)"
if not %errorlevel%==0 goto :old_python

rem ------------------------------------------------------------- create the venv
if exist ".venv\Scripts\activate.bat" (
    echo.
    echo Reusing the existing .venv ^(delete it yourself if you want a clean one^).
) else (
    echo.
    echo Creating .venv ...
    "%PYEXE%" -m venv .venv
    if not %errorlevel%==0 goto :venv_failed
)

set "VENVPY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENVPY%" goto :venv_failed

rem ------------------------------------------------------------------ pip + install
echo.
echo Upgrading pip if needed ...
"%VENVPY%" -m pip install --upgrade pip
if not %errorlevel%==0 goto :pip_failed

rem Windows will not let pip replace an .exe that is running, and it discovers
rem this *after* uninstalling the old version - which leaves no working install
rem at all. Cheaper to refuse up front than to explain the wreckage.
tasklist /FI "IMAGENAME eq stable.exe" 2>nul | find /I "stable.exe" >nul
if %errorlevel%==0 goto :stable_running

echo.
echo Installing Stable and its dependencies ...
"%VENVPY%" -m pip install -e ".[all]"
if not %errorlevel%==0 goto :install_failed

rem ------------------------------------------------------------------ gpu support
rem PyPI's `torch` wheel is CPU-only on Windows, so the CUDA build has to come
rem from PyTorch's own index and which index depends on the driver. That choice
rem (and verifying the result actually sees the card) lives in Python, where it
rem can be tested -- see scripts\install_gpu.py. It asks first, skips silently
rem on a machine with no NVIDIA GPU, and never fails the install.
"%VENVPY%" scripts\install_gpu.py --python "%VENVPY%"

rem ------------------------------------------------------------ deep tracker code
rem Optional and never fails the install; empty input (e.g. piped) means skip.
echo.
echo CoTracker3 and TAPIR need their model code from GitHub (requires git).
set "DEEPCHOICE="
set /p "DEEPCHOICE=Install it now? [y/N]: "
if /I not "%DEEPCHOICE:~0,1%"=="y" goto :deep_done
where git >nul 2>nul
if not %errorlevel%==0 (
    echo git not found on PATH - skipping. Install git and see the README.
    goto :deep_done
)
"%VENVPY%" -m pip install "tapnet @ git+https://github.com/google-deepmind/tapnet.git"
if not %errorlevel%==0 echo WARNING: installing TAPIR ^(tapnet^) failed - see the pip output above.
"%VENVPY%" -m pip install "cotracker @ git+https://github.com/facebookresearch/co-tracker.git"
if not %errorlevel%==0 echo WARNING: installing CoTracker3 failed - see the pip output above.
:deep_done

rem ---------------------------------------------------------------------- activate
echo.
call ".venv\Scripts\activate.bat"

echo ============================================================
echo  Stable is installed and this environment is now ACTIVE.
echo.
echo  Start the app with:
echo.
echo      stable serve
echo.
echo  That serves the web UI and opens your browser. Other things
echo  worth knowing:
echo.
echo      stable sample      run the bundled demo clip end to end
echo      stable trackers    list the tracking algorithms available
echo      stable devices     show whether your GPU is being used
echo      stable --help      every command
echo.
echo  The deep trackers (CoTracker3 / TAPIR) also need their model
echo  code, which is not ours to ship. If you skipped that step
echo  above, see docs\windows-gpu-setup.md step 4. RAFT needs
echo  nothing further.
echo.
echo  Next time, reactivate with:  .venv\Scripts\activate.bat
echo ============================================================
popd
goto :eof

rem ------------------------------------------------------------------------ errors
:no_python
echo ERROR: Neither "python" nor "python3" was found on your PATH.
echo Install Python 3.11 or newer from https://www.python.org/downloads/
echo and make sure you tick "Add python.exe to PATH" in the installer.
popd
goto :eof

:old_python
echo.
echo ERROR: %PYEXE% is too old. Stable needs Python 3.11 or newer.
"%PYEXE%" -c "import platform;print('  you have: '+platform.python_version())"
popd
goto :eof

:venv_failed
echo.
echo ERROR: could not create the virtual environment in .venv
echo On some systems the "venv" module ships separately from Python.
popd
goto :eof

:pip_failed
echo.
echo ERROR: upgrading pip failed. Check your network or proxy settings.
popd
goto :eof

:install_failed
echo.
echo ERROR: installing Stable failed - see the pip output above.
popd
goto :eof

:stable_running
echo.
echo ERROR: Stable is currently running, and Windows will not let the
echo installer replace files that are in use.
echo.
echo Close any "stable serve" window (Ctrl+C) and run this again. If you
echo cannot find it:  taskkill /IM stable.exe /F
popd
goto :eof
