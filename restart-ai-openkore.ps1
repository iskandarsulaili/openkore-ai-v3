[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RootDir = $PSScriptRoot

& (Join-Path $RootDir 'stop-ai-openkore.ps1')
Start-Sleep -Seconds 1
& (Join-Path $RootDir 'start-ai-openkore.ps1')

