[CmdletBinding()]
param(
    [int]$GraceSeconds = 10
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RootDir = $PSScriptRoot
$PidDir = Join-Path $RootDir '.ai-sidecar-runtime\pids'
$OpenkorePidFile = Join-Path $PidDir 'openkore.pid'
$SidecarPidFile = Join-Path $PidDir 'sidecar.pid'

function Write-Info([string]$Message) { Write-Host "[INFO ] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[ OK  ] $Message" -ForegroundColor Green }
function Write-WarnMsg([string]$Message) { Write-Host "[WARN ] $Message" -ForegroundColor Yellow }
function Write-Err([string]$Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red }

function Get-PidFromFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $raw = (Get-Content -LiteralPath $Path -Raw).Trim()
    if ($raw -match '^\d+$') { return [int]$raw }
    return $null
}

function Test-PidRunning {
    param([Parameter(Mandatory = $true)][int]$Pid)
    return ($null -ne (Get-Process -Id $Pid -ErrorAction SilentlyContinue))
}

function Stop-ByPidFile {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$PidFile,
        [Parameter(Mandatory = $true)][int]$Grace
    )

    $procId = Get-PidFromFile -Path $PidFile
    if ($null -eq $procId) {
        if (Test-Path -LiteralPath $PidFile -PathType Leaf) {
            Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        }
        Write-WarnMsg "$Label: no pid file"
        return
    }

    if (-not (Test-PidRunning -Pid $procId)) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        Write-WarnMsg "$Label: stale pid $procId; cleaned pid file"
        return
    }

    Write-Info "$Label: sending stop signal to pid $procId"
    Stop-Process -Id $procId -ErrorAction SilentlyContinue

    $deadline = (Get-Date).AddSeconds($Grace)
    while ((Get-Date) -lt $deadline) {
        if (-not (Test-PidRunning -Pid $procId)) {
            Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
            Write-Ok "$Label: stopped gracefully"
            return
        }
        Start-Sleep -Seconds 1
    }

    if (Test-PidRunning -Pid $procId) {
        Write-WarnMsg "$Label: still running after ${Grace}s; forcing stop"
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }

    if (Test-PidRunning -Pid $procId) {
        throw "$Label: unable to stop process $procId"
    }

    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Write-Ok "$Label: stopped forcefully"
}

try {
    New-Item -ItemType Directory -Path $PidDir -Force | Out-Null
    Stop-ByPidFile -Label 'OpenKore' -PidFile $OpenkorePidFile -Grace $GraceSeconds
    Stop-ByPidFile -Label 'AI sidecar' -PidFile $SidecarPidFile -Grace $GraceSeconds
    Write-Ok 'Stop workflow completed'
}
catch {
    Write-Err $_.Exception.Message
    exit 1
}
