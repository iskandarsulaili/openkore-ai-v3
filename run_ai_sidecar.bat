@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "REPO_ROOT=%SCRIPT_DIR%"
set "SIDECAR_DIR=%REPO_ROOT%\AI_sidecar"
set "VENV_DIR=%SIDECAR_DIR%\.venv"

call :log Validating repository layout...
if not exist "%SIDECAR_DIR%\" (
  call :error_exit Expected directory not found: AI_sidecar\. Place this script in openkore-ai-v3\ and run it again.
  exit /b 1
)
if not exist "%SIDECAR_DIR%\pyproject.toml" (
  call :error_exit Expected file not found: AI_sidecar\pyproject.toml. Ensure script is run from the openkore-ai-v3 root.
  exit /b 1
)

cd /d "%REPO_ROOT%" || (
  call :error_exit Failed to switch to repository root: %REPO_ROOT%
  exit /b 1
)

call :log Locating Python 3.11+...
set "PYTHON_EXE="
set "PYTHON_ARGS="

where py >nul 2>&1 && (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
  )
)

if not defined PYTHON_EXE (
  where python >nul 2>&1 && (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && (
      set "PYTHON_EXE=python"
      set "PYTHON_ARGS="
    )
  )
)

if not defined PYTHON_EXE (
  call :error_exit Python 3.11+ was not found. Install Python 3.11 or newer, then rerun this script.
  exit /b 1
)

for /f "delims=" %%v in ('"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set "PYTHON_VERSION=%%v"
call :log Using Python: %PYTHON_EXE% %PYTHON_ARGS% (%PYTHON_VERSION%)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  call :log Creating virtual environment at .venv\...
  "%PYTHON_EXE%" %PYTHON_ARGS% -m venv "%VENV_DIR%" || (
    call :error_exit Failed to create virtual environment at %VENV_DIR%.
    exit /b 1
  )
) else (
  call :log Virtual environment already exists at .venv\.
)

call :log Installing/updating dependencies from AI_sidecar\pyproject.toml...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip || (
  call :error_exit Failed to upgrade pip in .venv.
  exit /b 1
)
"%VENV_DIR%\Scripts\python.exe" -m pip install -e "%SIDECAR_DIR%" || (
  call :error_exit Failed to install AI Sidecar dependencies.
  exit /b 1
)

if not exist "%SIDECAR_DIR%\.env" (
  if exist "%SIDECAR_DIR%\.env.example" (
    call :log Creating AI_sidecar\.env from AI_sidecar\.env.example...
    copy /Y "%SIDECAR_DIR%\.env.example" "%SIDECAR_DIR%\.env" >nul || (
      call :error_exit Failed to create AI_sidecar\.env from template.
      exit /b 1
    )
  ) else (
    call :error_exit AI_sidecar\.env.example not found, cannot create AI_sidecar\.env.
    exit /b 1
  )
) else (
  call :log Environment file already exists at AI_sidecar\.env.
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
  call :error_exit Virtual environment activation script not found at %VENV_DIR%\Scripts\activate.bat.
  exit /b 1
)

call :log Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat" || (
  call :error_exit Failed to activate virtual environment.
  exit /b 1
)

cd /d "%SIDECAR_DIR%" || (
  call :error_exit Failed to switch to AI Sidecar directory: %SIDECAR_DIR%
  exit /b 1
)

call :log Starting AI Sidecar in foreground. Press Ctrl+C to stop.
where openkore-ai-sidecar >nul 2>&1
if %errorlevel%==0 (
  openkore-ai-sidecar
  set "APP_EXIT=!ERRORLEVEL!"
) else (
  call :log CLI entrypoint not found; falling back to python -m ai_sidecar.app
  python -m ai_sidecar.app
  set "APP_EXIT=!ERRORLEVEL!"
)

if not "!APP_EXIT!"=="0" (
  call :error_exit AI Sidecar exited with non-zero code !APP_EXIT!.
  exit /b 1
)

exit /b 0

:log
echo [AI-SIDECAR] %*
exit /b 0

:error_exit
echo [AI-SIDECAR][ERROR] %* 1>&2
exit /b 1
