# AI Sidecar & OpenKore Automation Suite - Comprehensive Installation Guide

## Overview

This guide provides step-by-step instructions for installing and configuring the AI Sidecar and OpenKore automation suite on both Windows and Linux operating systems. The system consists of two main components:

1. **OpenKore Core** - Perl-based Ragnarok Online automation client
2. **AI Sidecar** - Python FastAPI service providing AI-driven decision making

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation & Configuration](#installation--configuration)
   - [AI Sidecar Installation](#ai-sidecar-installation)
   - [OpenKore Installation](#openkore-installation)
4. [Integration](#integration)
5. [Execution & Verification](#execution--verification)
6. [Troubleshooting Appendix](#troubleshooting-appendix)

---

## 1. Prerequisites

### Windows Requirements

| Software | Version | Download Link | Verification Command |
|----------|---------|---------------|----------------------|
| **Python** | ≥ 3.11 | [python.org/downloads](https://www.python.org/downloads/) | `python --version` |
| **Perl** | ≥ 5.32 (Strawberry Perl recommended) | [strawberryperl.com](https://strawberryperl.com/) | `perl -v` |
| **Git** | Latest | [git-scm.com/download/win](https://git-scm.com/download/win) | `git --version` |

### Linux Requirements (Ubuntu/Debian)

| Software | Version | Installation Command | Verification Command |
|----------|---------|----------------------|----------------------|
| **Python** | ≥ 3.11 | `sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip` | `python3 --version` |
| **Perl** | ≥ 5.32 | `sudo apt install perl` | `perl -v` |
| **Git** | Latest | `sudo apt install git` | `git --version` |
| **SQLite3** | ≥ 3.35 | `sudo apt install sqlite3 libsqlite3-dev` | `sqlite3 --version` |

### Cross-Platform Additional Requirements

- **JSON::PP Perl Module** (Required for OpenKore bridge) - Verify with: `perl -e "use JSON::PP; print 'JSON::PP module is available\n';"`
- **Python Virtual Environment** (venv or conda)
- **Administrative/Sudo Privileges** for system package installation
- **Network Connectivity** for dependency downloads

**Important:** The OpenKore bridge plugin requires JSON::PP to function. If this module is missing, the bridge will disable itself but OpenKore will continue running.

---

## 2. Environment Setup

### Windows Environment Variables

**Administrator Note:** Some operations require Windows Command Prompt to be opened as Administrator:
- Modifying system PATH environment variables
- Installing Perl modules via CPAN (if installing to system Perl)
- Changing Windows Firewall rules
- Installing system-wide Python packages

If you encounter permission errors, right-click Command Prompt and select "Run as administrator".

#### Temporary Session Configuration (Command Prompt):
```cmd
set PATH=%PATH%;C:\Python311\Scripts;C:\Python311
set PATH=%PATH%;C:\StrawberryPerl\perl\bin
set OPENKORE_AI_ENV=development
```

#### Permanent Configuration (System Properties):
1. Open **System Properties** → **Advanced** → **Environment Variables**
2. Add to **System Variables**:
   - `PYTHONPATH`: `C:\Python311\Lib;C:\Python311\DLLs`
   - Add to `PATH`: `C:\Python311\Scripts;C:\Python311;C:\StrawberryPerl\perl\bin`

### Linux Environment Variables

#### Temporary Session Configuration (Bash):
```bash
export PATH="$PATH:/usr/local/bin"
export OPENKORE_AI_ENV=development
export PYTHONPATH="/usr/local/lib/python3.11/site-packages"
```

#### Permanent Configuration (~/.bashrc or ~/.profile):
```bash
echo 'export PATH="$PATH:/usr/local/bin"' >> ~/.bashrc
echo 'export OPENKORE_AI_ENV="development"' >> ~/.bashrc
echo 'export PYTHONPATH="/usr/local/lib/python3.11/site-packages:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

### Python Virtual Environment Setup

#### Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Linux (Bash):
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Installation & Configuration

### AI Sidecar Installation

#### Step 1: Clone the Repository

**Windows (Command Prompt):**
```cmd
git clone https://github.com/OpenKore/openkore-ai-v3.git
cd openkore-ai-v3\AI_sidecar
```

**Linux (Bash):**
```bash
git clone https://github.com/OpenKore/openkore-ai-v3.git
cd openkore-ai-v3/AI_sidecar
```

#### Step 2: Create Virtual Environment

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Linux (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3: Install Python Dependencies

**Windows (Command Prompt with activated virtual environment):**
```cmd
python -m pip install --upgrade pip
python -m pip install -e .
```

**Linux (Bash with activated virtual environment):**
```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

**Required Dependencies (from `pyproject.toml`):**
- `fastapi>=0.115.0`
- `uvicorn[standard]>=0.30.0`
- `pydantic>=2.8.0`
- `pydantic-settings>=2.4.0`
- `scikit-learn>=1.7.1`
- `crewai==1.14.2`
- `openmemory-py>=1.3.2`

#### Step 4: Configure Environment

**Windows (Command Prompt):**
```cmd
:: Copy environment template
copy .env.example .env

:: Edit .env file with your configuration
:: Important variables to configure:
:: - OPENKORE_AI_PORT=18081 (default)
:: - OPENKORE_AI_PROVIDER_DEEPSEEK_API_KEY (if using DeepSeek)
:: - OPENKORE_AI_PROVIDER_OLLAMA_BASE_URL (if using Ollama)
```

**Linux (Bash):**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
# Important variables to configure:
# - OPENKORE_AI_PORT=18081 (default)
# - OPENKORE_AI_PROVIDER_DEEPSEEK_API_KEY (if using DeepSeek)
# - OPENKORE_AI_PROVIDER_OLLAMA_BASE_URL (if using Ollama)
```

#### Step 5: Verify Installation
```bash
# Check Python version compatibility
python --version

# Verify package installation
python -c "import fastapi; import crewai; print('Dependencies installed successfully')"
```

### OpenKore Installation

**Important:** This repository already contains the complete OpenKore runtime files (`openkore.pl`, `start.pl`, launcher executables, and all required plugins). You do NOT need to clone a separate OpenKore repository.

#### Step 1: Verify OpenKore Files
Check that the OpenKore runtime files are present in the current directory:

```bash
# Linux/macOS
ls -la openkore.pl start.pl control/ plugins/

# Windows Command Prompt
dir openkore.pl start.pl control/ plugins/
```

If you're already in the `openkore-ai-v3` directory, you're in the correct location.

#### Step 2: Install Perl Dependencies

**Windows (Strawberry Perl):**
```cmd
# Using CPAN client (Strawberry Perl includes CPAN by default)
cpan install JSON::PP
cpan install Time::HiRes
cpan install Carp::Assert
```

**Linux (Ubuntu/Debian):**
```bash
# Install system packages
sudo apt install libjson-pp-perl libtime-hires-perl

# Alternative using CPAN
sudo cpan install JSON::PP
sudo cpan install Time::HiRes
```

#### Step 3: Verify Perl Module Installation
```bash
perl -e "use JSON::PP; use Time::HiRes; print 'Perl modules installed successfully\n';"
```

#### Step 4: Configure OpenKore Control Files

The following control files are pre-configured in the repository:

1. **`control/sys.txt`** - Plugin configuration
   - Already includes `aiSidecarBridge` in `loadPlugins_list`
   - `loadPlugins 2` enables selective plugin loading

2. **`control/ai_sidecar.txt`** - Bridge configuration
   - Base URL: `http://127.0.0.1:18081`
   - Timeouts: Connect=250ms, IO=1500ms
   - Snapshot interval: 500ms
   - Action polling: Enabled (250ms interval)

3. **`control/ai_sidecar_policy.txt`** - Command safety policy
   - Defines allowed/disallowed commands
   - Configurable safety thresholds

4. **`control/config.txt`** - Primary runtime configuration
   - Server connection settings
   - Character configuration
   - Bot behavior parameters

#### Step 5: Build XSTools (Windows Only)
```cmd
# Run from openkore-ai-v3 directory
python src/scons-local-3.1.2/scons.py
```

---

## 4. Integration

### Component Communication Architecture

```
OpenKore Client (Perl) ↔ aiSidecarBridge Plugin ↔ AI Sidecar (FastAPI)
      │                            │                         │
  control/                    HTTP/JSON                  Port 18081
config.txt               (127.0.0.1:18081)            Python Service
```

### Network Configuration

1. **Default Port**: `18081` (configurable in `.env` and `ai_sidecar.txt`)
2. **Bind Address**: `127.0.0.1` (localhost only for security)
3. **Firewall Rules**:
   - Windows: Allow inbound connections on port 18081 (requires Administrator privileges)
     ```cmd
     netsh advfirewall firewall add rule name="OpenKore AI Sidecar" dir=in action=allow protocol=TCP localport=18081
     ```
   - Linux: `sudo ufw allow 18081/tcp`

### API Endpoints (Implemented Routes)

The following endpoints are implemented and actively used by the bridge plugin:

| Endpoint | Method | Purpose | Used By Bridge |
|----------|--------|---------|----------------|
| `/v1/health/live` | GET | Service liveness check | Manual verification |
| `/v1/health/ready` | GET | Service readiness check | Manual verification |
| `/v1/ingest/register` | POST | Register bot with sidecar | Bridge at startup |
| `/v1/ingest/snapshot` | POST | Receive bot state snapshots | Bridge every 500ms |
| `/v1/actions/next` | POST | Poll for next AI-generated action | Bridge every 250ms |
| `/v1/acknowledgements/action` | POST | Acknowledge action execution | Bridge after action execution |
| `/v1/telemetry/ingest` | POST | Receive telemetry data | Bridge every 1000ms |
| `/v1/macros` | POST | Publish macro updates | External tools/API clients |
| `/v1/fleet` | GET | Multi-bot fleet status | Monitoring dashboards |

**Note:** The bridge plugin uses specific endpoint paths (e.g., `/v1/actions/next` not `/v1/actions`). Ensure your configuration matches these exact paths.

### Configuration Synchronization

1. **AI Sidecar → OpenKore**:
   - Macros published to `control/ai_sidecar_generated_macros.txt` (regular macros)
   - Event macros published to `control/ai_sidecar_generated_eventmacros.txt` (event macros)
   - Hot-reload triggered via bridge plugin using safe built-in flow

2. **OpenKore → AI Sidecar**:
   - State snapshots sent every 500ms via POST `/v1/ingest/snapshot`
   - Telemetry data sent every 1000ms via POST `/v1/telemetry/ingest`
   - Action acknowledgements via POST `/v1/acknowledgements/action`

---

## 5. Execution & Verification

### Startup Sequence

#### Step 1: Start AI Sidecar Service

**Windows (Command Prompt):**
```cmd
cd openkore-ai-v3\AI_sidecar
.venv\Scripts\activate
openkore-ai-sidecar
```

**Linux (Bash):**
```bash
cd openkore-ai-v3/AI_sidecar
source .venv/bin/activate
openkore-ai-sidecar
```

**Alternative startup method:**
```bash
python -m ai_sidecar.app
```

#### Step 2: Verify Sidecar Service

**Windows (Command Prompt with curl installed) or Linux (Bash):**
```bash
# Check if service is running
curl http://127.0.0.1:18081/v1/health/live
curl http://127.0.0.1:18081/v1/health/ready
```

**Expected responses:**
- `/v1/health/live`: `{"ok": true, "status": "live", "started_at": "...", "now": "..."}`
- `/v1/health/ready`: `{"ok": true, "status": "ready", "bots_registered": 0, "bots_persisted": 0, "snapshots_cached": 0, ...}` (richer operational status)
```

#### Step 3: Start OpenKore Client

**Windows (Using Launchers):**
- `start.exe` - Console interface
- `wxstart.exe` - wxWidgets GUI
- `tkstart.exe` - Tkinter GUI

**Linux/Windows (Command Line):**
```bash
cd openkore-ai-v3
perl openkore.pl
```

#### Step 4: Verify Integration

1. **Check Bridge Plugin Loading**:
   ```
   [OpenKore Console] Plugin 'aiSidecarBridge' loaded successfully
   ```

2. **Verify Connection**:
   ```
   [aiSidecarBridge] Connected to sidecar at http://127.0.0.1:18081
   ```

3. **Monitor Communication**:
   - Sidecar logs: `[INFO] Received snapshot from bot: [bot_id]`
   - OpenKore logs: `[aiSidecarBridge] Action received: [command]`

### Health Checks

#### Service Health Endpoints:
```bash
# Basic health check
curl http://127.0.0.1:18081/v1/health/live

# Detailed readiness check
curl http://127.0.0.1:18081/v1/health/ready

# API documentation (if OPENKORE_AI_ENABLE_DOCS=1)
# http://127.0.0.1:18081/docs
# http://127.0.0.1:18081/redoc
```

#### Database Health:
```bash
# Check SQLite database
sqlite3 AI_sidecar/data/sidecar.sqlite "SELECT COUNT(*) FROM bot_registry;"
```

#### Process Monitoring:
```bash
# Windows
tasklist | findstr "python perl"

# Linux
ps aux | grep -E "(python|perl|openkore)"
```

### Troubleshooting Startup Issues

1. **Port Already in Use**:
   ```bash
   # Windows
   netstat -ano | findstr :18081
   
   # Linux
   sudo lsof -i :18081
   sudo kill -9 [PID]
   ```

2. **Service Not Starting**:
   ```bash
   # Check logs (created at runtime in AI_sidecar/logs/ directory)
   tail -f AI_sidecar/logs/sidecar.log 2>/dev/null || echo "Log file not found yet - service may not have started"
   
   # Increase log level
   export OPENKORE_AI_LOG_LEVEL=DEBUG
   ```

3. **Bridge Connection Failed**:
   - Verify `control/ai_sidecar.txt` base URL
   - Check firewall settings
   - Confirm sidecar service is running

---

## 6. Troubleshooting Appendix

### Common Errors and Solutions

#### Python-Related Issues

**Error: `ModuleNotFoundError: No module named 'fastapi'`**
```bash
# Solution: Reinstall in virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate
pip install -e .
```

**Error: `Python version must be >=3.11`**
```bash
# Solution: Install Python 3.11+
# Windows: Download from python.org
# Linux: sudo apt install python3.11
```

**Error: `scikit-learn installation failed`**
```bash
# Solution: Install build dependencies
# Windows: Install Visual C++ Build Tools
# Linux: sudo apt install python3-dev build-essential
pip install --no-cache-dir scikit-learn
```

#### Perl-Related Issues

**Error: `Can't locate JSON/PP.pm in @INC`**
```bash
# Solution: Install JSON::PP module
# Windows (Strawberry Perl):
cpan install JSON::PP

# Linux:
sudo apt install libjson-pp-perl
# or
sudo cpan install JSON::PP
```

**Error: `XSTools.dll not found` (Windows)**
```cmd
# Solution: Build XSTools
python src/scons-local-3.1.2/scons.py
```

**Error: `Permission denied` on plugin files**
```bash
# Solution: Adjust file permissions
chmod +x plugins/*.pl
chmod +x openkore.pl
```

#### Network and Connection Issues

**Error: `Connection refused` on port 18081**
```bash
# Solution 1: Check if service is running
curl http://127.0.0.1:18081/v1/health/live

# Solution 2: Check firewall
# Windows: netsh advfirewall firewall show rule name="OpenKore AI"
# Linux: sudo ufw status

# Solution 3: Verify bind address in .env
# OPENKORE_AI_HOST=127.0.0.1 (not 0.0.0.0)
```

**Error: `Timeout waiting for sidecar response`**
```bash
# Solution: Increase timeouts in control/ai_sidecar.txt
aiSidecar_connectTimeoutMs 1000
aiSidecar_ioTimeoutMs 5000
```

#### Platform-Specific Issues

**Windows: `'python' is not recognized as an internal or external command`**
```cmd
# Solution: Add Python to PATH or use full path
C:\Python311\python.exe --version
# Or reinstall Python with "Add to PATH" option checked
```

**Linux: `perl: warning: Setting locale failed`**
```bash
# Solution: Set locale environment variables
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
```

**macOS: `Library not loaded` errors**
```bash
# Solution: Install Homebrew and required libraries
brew install perl
brew install python@3.11
```

### Diagnostic Commands

#### System Verification:
```bash
# Python environment
python --version
pip list | grep -E "(fastapi|uvicorn|crewai)"

# Perl environment
perl -v
perl -e "print join('\n', @INC);"

# Network connectivity
curl -v http://127.0.0.1:18081/v1/health/live
telnet 127.0.0.1 18081

# Process status
ps aux | grep -E "(sidecar|openkore)"
netstat -tulpn | grep :18081
```

#### Log Analysis:
```bash
# AI Sidecar logs (created at runtime)
tail -f AI_sidecar/logs/sidecar.log 2>/dev/null || echo "AI Sidecar log not found - check if service is running"

# OpenKore logs (created at runtime in logs/ directory)
tail -f logs/openkore.log 2>/dev/null || echo "OpenKore log not found - check if OpenKore is running"

# Bridge plugin logs (enable verbose mode in control/ai_sidecar.txt)
# Set: aiSidecar_verbose 1
# Bridge logs appear in OpenKore console, not separate files
```

#### Database Diagnostics:
```bash
# Check SQLite database integrity
sqlite3 AI_sidecar/data/sidecar.sqlite "PRAGMA integrity_check;"

# Check table sizes
sqlite3 AI_sidecar/data/sidecar.sqlite "SELECT name FROM sqlite_master WHERE type='table';"
sqlite3 AI_sidecar/data/sidecar.sqlite "SELECT COUNT(*) FROM bot_registry;"
sqlite3 AI_sidecar/data/sidecar.sqlite "SELECT COUNT(*) FROM action_queue;"
```

### Performance Optimization Tips

#### Windows Optimization:
1. **Disable Windows Defender Real-time Scanning** for OpenKore directories
2. **Set process priority**: `start /high perl openkore.pl`
3. **Adjust power settings** to High Performance mode

#### Linux Optimization:
1. **Increase file descriptor limits**:
   ```bash
   ulimit -n 65536
   echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
   echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
   ```

2. **Optimize kernel parameters**:
   ```bash
   echo "net.core.somaxconn = 1024" | sudo tee -a /etc/sysctl.conf
   echo "vm.swappiness = 10" | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```

#### Python Performance:
1. **Use PyPy for better performance** (optional):
   ```bash
   pypy3 -m venv .venv-pypy
   source .venv-pypy/bin/activate
   pip install -e .
   ```

2. **Enable JIT compilation** for CPU-intensive operations

### Maintenance and Updates

#### Regular Maintenance Tasks:
1. **Clear old logs** (if log directories exist):
   ```bash
   # Clean OpenKore logs (logs/ directory created at runtime)
   find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
   
   # Clean AI Sidecar logs (AI_sidecar/logs/ directory created at runtime)
   find AI_sidecar/logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
   ```

2. **Vacuum SQLite databases**:
   ```bash
   sqlite3 AI_sidecar/data/sidecar.sqlite "VACUUM;"
   sqlite3 AI_sidecar/data/openmemory.sqlite "VACUUM;"
   ```

3. **Update dependencies**:
   ```bash
   pip list --outdated
   pip install --upgrade -e .
   ```

#### Backup Procedures:
1. **Configuration backup**:
   ```bash
   tar -czf openkore-backup-$(date +%Y%m%d).tar.gz control/ AI_sidecar/.env AI_sidecar/data/
   ```

2. **Database backup**:
   ```bash
   sqlite3 AI_sidecar/data/sidecar.sqlite ".backup sidecar-backup.sqlite"
   ```

### Support and Resources

#### Official Documentation:
- [OpenKore Wiki](https://openkore.com/wiki/)
- [AI Sidecar Documentation](AI_sidecar/docs/)
- [API Contracts](AI_sidecar/docs/api-contracts.md)

#### Community Support:
- [OpenKore Forum](https://forums.openkore.com/)
- [Discord Community](https://discord.com/invite/hdAhPM6)
- [GitHub Issues](https://github.com/OpenKore/openkore-ai-v3/issues)

#### Monitoring Tools:
1. **Sidecar Dashboard**: `http://127.0.0.1:18081/docs` (when enabled)
2. **OpenKore Console**: Built-in monitoring commands
3. **System Monitoring**: Use `htop`, `nmon`, or Windows Task Manager

### Conclusion

This comprehensive guide covers the installation, configuration, and operation of the AI Sidecar and OpenKore automation suite on both Windows and Linux platforms. The system is designed for production use with fail-open architecture, bounded latency, and comprehensive monitoring.

**Key Success Indicators:**
1. AI Sidecar service running on port 18081
2. OpenKore client connected to the sidecar bridge
3. Two-way communication established (snapshots and actions)
4. Macro publication and hot-reload functioning
5. Database persistence operational

For advanced configuration, performance tuning, or custom integration, refer to the detailed documentation in the `AI_sidecar/docs/` directory and the OpenKore wiki.

---
*Last Updated: 2026-04-21*
*Documentation Version: 1.0*
*Compatible with: openkore-ai-v3 repository*