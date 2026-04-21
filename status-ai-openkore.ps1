[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RootDir = $PSScriptRoot
$PidDir = Join-Path $RootDir '.ai-sidecar-runtime\pids'
$LauncherEnvFile = Join-Path $RootDir 'ai-sidecar-launcher.env'
$LauncherEnvExample = Join-Path $RootDir 'ai-sidecar-launcher.env.example'

function Write-Info([string]$Message) { Write-Host "[INFO ] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[ OK  ] $Message" -ForegroundColor Green }
function Write-WarnMsg([string]$Message) { Write-Host "[WARN ] $Message" -ForegroundColor Yellow }

function Import-LauncherEnv {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if ($trimmed.Length -eq 0 -or $trimmed.StartsWith('#')) { continue }
        if ($trimmed -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
            Set-Variable -Scope Script -Name $Matches[1] -Value $Matches[2]
        }
    }
}

function Get-LauncherValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][object]$DefaultValue
    )

    $var = Get-Variable -Name $Name -Scope Script -ErrorAction SilentlyContinue
    if ($null -eq $var -or $null -eq $var.Value -or ("$($var.Value)" -eq '')) {
        return $DefaultValue
    }
    return $var.Value
}

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

function Show-ComponentStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$PidFile
    )

    $procId = Get-PidFromFile -Path $PidFile
    if ($null -eq $procId) {
        Write-WarnMsg "$Label: not running"
        return
    }

    if (Test-PidRunning -Pid $procId) {
        Write-Ok "$Label: running (pid $procId)"
    }
    else {
        Write-WarnMsg "$Label: stale pid file (pid $procId)"
    }
}

if (-not (Test-Path -LiteralPath $LauncherEnvFile -PathType Leaf) -and (Test-Path -LiteralPath $LauncherEnvExample -PathType Leaf)) {
    Import-LauncherEnv -Path $LauncherEnvExample
}
else {
    Import-LauncherEnv -Path $LauncherEnvFile
}

$SidecarHost = [string](Get-LauncherValue -Name 'SIDECAR_HOST' -DefaultValue '127.0.0.1')
$SidecarPort = [int](Get-LauncherValue -Name 'SIDECAR_PORT' -DefaultValue 18081)
$SidecarHealthPath = [string](Get-LauncherValue -Name 'SIDECAR_HEALTH_PATH' -DefaultValue '/v1/health/live')

New-Item -ItemType Directory -Path $PidDir -Force | Out-Null

Show-ComponentStatus -Label 'AI sidecar' -PidFile (Join-Path $PidDir 'sidecar.pid')
Show-ComponentStatus -Label 'OpenKore' -PidFile (Join-Path $PidDir 'openkore.pid')

$healthUrl = "http://$SidecarHost`:$SidecarPort$SidecarHealthPath"
try {
    $null = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
    Write-Ok "Sidecar health endpoint reachable: $healthUrl"
}
catch {
    Write-WarnMsg "Sidecar health endpoint not reachable: $healthUrl"
}
