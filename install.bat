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

echo.
echo Installing Stable and its dependencies ...
"%VENVPY%" -m pip install -e ".[all]"
if not %errorlevel%==0 goto :install_failed

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
echo      stable --help      every command
echo.
echo  GPU tracking (CoTracker3 / TAPIR) needs PyTorch, which is
echo  installed separately - see docs\windows-gpu-setup.md.
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
