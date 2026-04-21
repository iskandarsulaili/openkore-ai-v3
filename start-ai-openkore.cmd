@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-ai-openkore.ps1" %*
exit /b %errorlevel%

