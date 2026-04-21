[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RootDir = $PSScriptRoot
$AiSidecarDir = Join-Path $RootDir 'AI_sidecar'
$RuntimeDir = Join-Path $RootDir '.ai-sidecar-runtime'
$PidDir = Join-Path $RuntimeDir 'pids'
$LogDir = Join-Path $RuntimeDir 'logs'

$LauncherEnvFile = Join-Path $RootDir 'ai-sidecar-launcher.env'
$LauncherEnvExample = Join-Path $RootDir 'ai-sidecar-launcher.env.example'

function Write-Info([string]$Message) { Write-Host "[INFO ] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[ OK  ] $Message" -ForegroundColor Green }
function Write-WarnMsg([string]$Message) { Write-Host "[WARN ] $Message" -ForegroundColor Yellow }
function Write-Err([string]$Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red }

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

function Resolve-PathFromRoot {
    param([Parameter(Mandatory = $true)][string]$PathSpec)
    if ([System.IO.Path]::IsPathRooted($PathSpec)) { return $PathSpec }
    return (Join-Path $RootDir $PathSpec)
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

function Wait-SidecarHealth {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            return $true
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

try {
    if (-not (Test-Path -LiteralPath $LauncherEnvFile -PathType Leaf) -and (Test-Path -LiteralPath $LauncherEnvExample -PathType Leaf)) {
        Copy-Item -LiteralPath $LauncherEnvExample -Destination $LauncherEnvFile -Force
        Write-WarnMsg 'Created ai-sidecar-launcher.env from template.'
    }
    Import-LauncherEnv -Path $LauncherEnvFile

    $SidecarHost = [string](Get-LauncherValue -Name 'SIDECAR_HOST' -DefaultValue '127.0.0.1')
    $SidecarPort = [int](Get-LauncherValue -Name 'SIDECAR_PORT' -DefaultValue 18081)
    $SidecarHealthPath = [string](Get-LauncherValue -Name 'SIDECAR_HEALTH_PATH' -DefaultValue '/v1/health/live')
    $SidecarStartTimeout = [int](Get-LauncherValue -Name 'SIDECAR_START_TIMEOUT_SECONDS' -DefaultValue 45)

    $SidecarLog = Resolve-PathFromRoot ([string](Get-LauncherValue -Name 'SIDECAR_LOG_FILE' -DefaultValue '.ai-sidecar-runtime/logs/sidecar.log'))
    $OpenkoreLog = Resolve-PathFromRoot ([string](Get-LauncherValue -Name 'OPENKORE_LOG_FILE' -DefaultValue '.ai-sidecar-runtime/logs/openkore.log'))
    $OpenkoreErrLog = "$OpenkoreLog.err"

    $SidecarPidFile = Join-Path $PidDir 'sidecar.pid'
    $OpenkorePidFile = Join-Path $PidDir 'openkore.pid'

    if (-not (Test-Path -LiteralPath $AiSidecarDir -PathType Container)) {
        throw "Missing AI_sidecar directory: $AiSidecarDir"
    }

    $venvPython = Join-Path $AiSidecarDir '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw 'Virtual environment missing at AI_sidecar/.venv. Run .\setup-ai-sidecar.ps1 first.'
    }

    New-Item -ItemType Directory -Path $PidDir -Force | Out-Null
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    New-Item -ItemType Directory -Path (Split-Path -Parent $SidecarLog) -Force | Out-Null
    New-Item -ItemType Directory -Path (Split-Path -Parent $OpenkoreLog) -Force | Out-Null

    $sidecarPid = Get-PidFromFile -Path $SidecarPidFile
    if ($null -ne $sidecarPid -and (Test-PidRunning -Pid $sidecarPid)) {
        Write-Info "Sidecar already running (pid: $sidecarPid)"
    }
    else {
        if ($null -ne $sidecarPid) { Remove-Item -LiteralPath $SidecarPidFile -Force -ErrorAction SilentlyContinue }
        Write-Info 'Starting AI sidecar process'
        $sidecarProc = Start-Process -FilePath $venvPython `
            -ArgumentList @('-m', 'ai_sidecar.app') `
            -WorkingDirectory $AiSidecarDir `
            -RedirectStandardOutput $SidecarLog `
            -RedirectStandardError "$SidecarLog.err" `
            -PassThru
        Set-Content -LiteralPath $SidecarPidFile -Value $sidecarProc.Id -Encoding ascii
        Write-Ok "Sidecar launched (pid: $($sidecarProc.Id))"
    }

    $healthUrl = "http://$SidecarHost`:$SidecarPort$SidecarHealthPath"
    Write-Info "Waiting for sidecar health endpoint: $healthUrl"
    if (-not (Wait-SidecarHealth -Url $healthUrl -TimeoutSeconds $SidecarStartTimeout)) {
        throw "Sidecar did not become healthy within ${SidecarStartTimeout}s"
    }
    Write-Ok 'Sidecar is healthy'

    $openkorePid = Get-PidFromFile -Path $OpenkorePidFile
    if ($null -ne $openkorePid -and (Test-PidRunning -Pid $openkorePid)) {
        Write-WarnMsg "OpenKore already running (pid: $openkorePid). Leaving it unchanged."
        exit 0
    }

    if ($null -ne $openkorePid) {
        Remove-Item -LiteralPath $OpenkorePidFile -Force -ErrorAction SilentlyContinue
    }

    $args = @((Join-Path $RootDir 'openkore.pl'))
    $openkoreArgsRaw = [string](Get-LauncherValue -Name 'OPENKORE_ARGS' -DefaultValue '')
    if ($openkoreArgsRaw) {
        $extraArgs = $openkoreArgsRaw.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
        $args += $extraArgs
    }

    Write-Info 'Starting OpenKore process'
    $openkoreProc = Start-Process -FilePath 'perl' `
        -ArgumentList $args `
        -WorkingDirectory $RootDir `
        -RedirectStandardOutput $OpenkoreLog `
        -RedirectStandardError $OpenkoreErrLog `
        -PassThru

    Set-Content -LiteralPath $OpenkorePidFile -Value $openkoreProc.Id -Encoding ascii
    Start-Sleep -Seconds 1

    if (Test-PidRunning -Pid $openkoreProc.Id) {
        Write-Ok "OpenKore launched (pid: $($openkoreProc.Id))"
        Write-Info "Sidecar log:  $SidecarLog"
        Write-Info "OpenKore log: $OpenkoreLog"
    }
    else {
        Remove-Item -LiteralPath $OpenkorePidFile -Force -ErrorAction SilentlyContinue
        throw "OpenKore exited immediately. Check log: $OpenkoreLog"
    }
}
catch {
    Write-Err $_.Exception.Message
    exit 1
}
