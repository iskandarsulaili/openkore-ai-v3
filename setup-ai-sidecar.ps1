[CmdletBinding()]
param(
    [string]$SidecarHost = "127.0.0.1",
    [int]$SidecarPort = 18081
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RootDir = $PSScriptRoot
$AiSidecarDir = Join-Path $RootDir 'AI_sidecar'
$ControlDir = Join-Path $RootDir 'control'
$RuntimeDir = Join-Path $RootDir '.ai-sidecar-runtime'
$BackupDir = Join-Path $RuntimeDir (Join-Path 'backups' (Get-Date -Format 'yyyyMMdd-HHmmss'))

function Write-Info([string]$Message) { Write-Host "[INFO ] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[ OK  ] $Message" -ForegroundColor Green }
function Write-WarnMsg([string]$Message) { Write-Host "[WARN ] $Message" -ForegroundColor Yellow }
function Write-Err([string]$Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red }

function Assert-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Backup-File {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    $target = Join-Path $BackupDir ((Split-Path -Path $Path -Leaf) + '.bak')
    Copy-Item -LiteralPath $Path -Destination $target -Force
}

function Update-KeyValueFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][string]$Value
    )
    $lines = @()
    if (Test-Path -LiteralPath $Path) {
        $lines = Get-Content -LiteralPath $Path
    }

    $updated = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\Q$Key\E\s+") {
            $lines[$i] = "$Key $Value"
            $updated = $true
            break
        }
    }

    if (-not $updated) {
        $lines += "$Key $Value"
    }

    Set-Content -LiteralPath $Path -Value $lines -Encoding ascii
}

function Update-EnvKey {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][string]$Value
    )

    $lines = @()
    if (Test-Path -LiteralPath $Path) {
        $lines = Get-Content -LiteralPath $Path
    }

    $updated = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\Q$Key\E=") {
            $lines[$i] = "$Key=$Value"
            $updated = $true
            break
        }
    }

    if (-not $updated) {
        $lines += "$Key=$Value"
    }

    Set-Content -LiteralPath $Path -Value $lines -Encoding ascii
}

function Ensure-PluginInSys {
    param([Parameter(Mandatory = $true)][string]$Path)

    $lines = Get-Content -LiteralPath $Path

    $loadPluginsIndex = -1
    $loadPluginsListIndex = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^loadPlugins\s+') { $loadPluginsIndex = $i }
        if ($lines[$i] -match '^loadPlugins_list\s+') { $loadPluginsListIndex = $i }
    }

    if ($loadPluginsIndex -lt 0) {
        $lines += 'loadPlugins 2'
    }
    elseif ($lines[$loadPluginsIndex] -match '^loadPlugins\s+0(\s|$)') {
        $lines[$loadPluginsIndex] = 'loadPlugins 2'
        Write-WarnMsg 'loadPlugins was 0; switched to 2 so aiSidecarBridge can load'
    }

    if ($loadPluginsListIndex -lt 0) {
        $lines += 'loadPlugins_list aiSidecarBridge'
        Write-Ok 'Added new loadPlugins_list with aiSidecarBridge'
    }
    else {
        $raw = $lines[$loadPluginsListIndex] -replace '^loadPlugins_list\s+', ''
        $items = @()
        if ($raw.Trim().Length -gt 0) {
            $items = $raw.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
        }

        if ($items -notcontains 'aiSidecarBridge') {
            $items += 'aiSidecarBridge'
            $lines[$loadPluginsListIndex] = 'loadPlugins_list ' + ($items -join ',')
            Write-Ok 'Added aiSidecarBridge into loadPlugins_list'
        }
        else {
            Write-Ok 'aiSidecarBridge already present in loadPlugins_list'
        }
    }

    Set-Content -LiteralPath $Path -Value $lines -Encoding ascii
}

function Test-PythonCandidate {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter()][string[]]$PrefixArgs = @()
    )

    try {
        $versionText = & $Executable @PrefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if (-not $versionText) { return $null }
        $version = [version]($versionText.Trim() + '.0')
        if ($version -lt [version]'3.11.0') { return $null }
        return [PSCustomObject]@{
            Exe = $Executable
            PrefixArgs = $PrefixArgs
            Version = $versionText.Trim()
        }
    }
    catch {
        return $null
    }
}

function Resolve-Python {
    $candidates = @()
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += [PSCustomObject]@{ Exe = 'python'; PrefixArgs = @() }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += [PSCustomObject]@{ Exe = 'py'; PrefixArgs = @('-3.11') }
        $candidates += [PSCustomObject]@{ Exe = 'py'; PrefixArgs = @('-3') }
    }

    foreach ($candidate in $candidates) {
        $result = Test-PythonCandidate -Executable $candidate.Exe -PrefixArgs $candidate.PrefixArgs
        if ($null -ne $result) { return $result }
    }

    throw 'Python 3.11+ is required but no compatible python launcher was found.'
}

try {
    Write-Info "Starting first-time AI sidecar + OpenKore setup"
    Write-Info "Repository root: $RootDir"

    Assert-CommandExists -Name perl
    Assert-CommandExists -Name Copy-Item

    $python = Resolve-Python
    Write-Ok "Detected Python $($python.Version) via '$($python.Exe) $($python.PrefixArgs -join ' ')'."

    if (-not (Test-Path -LiteralPath $AiSidecarDir -PathType Container)) {
        throw "AI_sidecar directory not found: $AiSidecarDir"
    }
    if (-not (Test-Path -LiteralPath $ControlDir -PathType Container)) {
        throw "control directory not found: $ControlDir"
    }

    New-Item -ItemType Directory -Path (Join-Path $RuntimeDir 'logs') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $RuntimeDir 'pids') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $RuntimeDir 'backups') -Force | Out-Null
    Write-Ok 'Ensured runtime directories under .ai-sidecar-runtime'

    $envExample = Join-Path $AiSidecarDir '.env.example'
    $envFile = Join-Path $AiSidecarDir '.env'
    $controlFile = Join-Path $ControlDir 'ai_sidecar.txt'
    $policyFile = Join-Path $ControlDir 'ai_sidecar_policy.txt'
    $sysFile = Join-Path $ControlDir 'sys.txt'
    $launcherEnvExample = Join-Path $RootDir 'ai-sidecar-launcher.env.example'
    $launcherEnvFile = Join-Path $RootDir 'ai-sidecar-launcher.env'
    $controlTemplate = Join-Path $RootDir 'ai-sidecar-control.template.txt'
    $policyTemplate = Join-Path $RootDir 'ai-sidecar-policy.template.txt'

    Backup-File -Path $envFile
    Backup-File -Path $controlFile
    Backup-File -Path $policyFile
    Backup-File -Path $sysFile

    if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
        Copy-Item -LiteralPath $envExample -Destination $envFile -Force
        Write-Ok 'Created AI_sidecar/.env from .env.example'
    }
    else {
        Write-Info 'Using existing AI_sidecar/.env'
    }

    Update-EnvKey -Path $envFile -Key 'OPENKORE_AI_HOST' -Value $SidecarHost
    Update-EnvKey -Path $envFile -Key 'OPENKORE_AI_PORT' -Value "$SidecarPort"
    Update-EnvKey -Path $envFile -Key 'OPENKORE_AI_SQLITE_PATH' -Value 'AI_sidecar/data/sidecar.sqlite'
    Update-EnvKey -Path $envFile -Key 'OPENKORE_AI_MEMORY_OPENMEMORY_PATH' -Value 'AI_sidecar/data/openmemory.sqlite'

    if (-not (Test-Path -LiteralPath $controlFile -PathType Leaf)) {
        Copy-Item -LiteralPath $controlTemplate -Destination $controlFile -Force
        Write-Ok 'Created control/ai_sidecar.txt from template'
    }
    if (-not (Test-Path -LiteralPath $policyFile -PathType Leaf)) {
        Copy-Item -LiteralPath $policyTemplate -Destination $policyFile -Force
        Write-Ok 'Created control/ai_sidecar_policy.txt from template'
    }

    Update-KeyValueFile -Path $controlFile -Key 'aiSidecar_enable' -Value '1'
    Update-KeyValueFile -Path $controlFile -Key 'aiSidecar_baseUrl' -Value "http://$SidecarHost`:$SidecarPort"
    Update-KeyValueFile -Path $controlFile -Key 'aiSidecar_verbose' -Value '1'

    Ensure-PluginInSys -Path $sysFile

    if (-not (Test-Path -LiteralPath $launcherEnvFile -PathType Leaf)) {
        Copy-Item -LiteralPath $launcherEnvExample -Destination $launcherEnvFile -Force
        Write-Ok 'Created ai-sidecar-launcher.env from template'
    }

    $venvDir = Join-Path $AiSidecarDir '.venv'
    $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        Write-Info 'Creating Python virtual environment in AI_sidecar/.venv'
        & $python.Exe @($python.PrefixArgs + @('-m', 'venv', $venvDir))
        Write-Ok 'Created AI_sidecar/.venv'
    }
    else {
        Write-Info 'Reusing existing AI_sidecar/.venv'
    }

    Write-Info 'Installing sidecar package dependencies'
    & $venvPython -m pip install --upgrade pip setuptools wheel
    & $venvPython -m pip install -e $AiSidecarDir
    & $venvPython -c 'import ai_sidecar; print("ai_sidecar import check: OK")'

    Write-Ok 'Setup completed successfully'
    Write-Info "Backups saved in: $BackupDir"
    Write-Info 'Next step: .\start-ai-openkore.ps1'
}
catch {
    Write-Err $_.Exception.Message
    exit 1
}
