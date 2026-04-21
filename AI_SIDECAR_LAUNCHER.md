# AI sidecar + OpenKore launcher scripts

This repository now includes first-time setup and lifecycle scripts for running the local AI sidecar together with OpenKore.

## Files

### Setup wizards

- Linux: [`setup-ai-sidecar.sh`](./setup-ai-sidecar.sh)
- Windows: [`setup-ai-sidecar.ps1`](./setup-ai-sidecar.ps1)

### Start scripts

- Linux: [`start-ai-openkore.sh`](./start-ai-openkore.sh)
- Windows PowerShell: [`start-ai-openkore.ps1`](./start-ai-openkore.ps1)
- Windows batch wrapper: [`start-ai-openkore.cmd`](./start-ai-openkore.cmd)

### Supporting lifecycle scripts

- Linux stop/status/restart:
  - [`stop-ai-openkore.sh`](./stop-ai-openkore.sh)
  - [`status-ai-openkore.sh`](./status-ai-openkore.sh)
  - [`restart-ai-openkore.sh`](./restart-ai-openkore.sh)
- Windows stop/status/restart:
  - [`stop-ai-openkore.ps1`](./stop-ai-openkore.ps1)
  - [`status-ai-openkore.ps1`](./status-ai-openkore.ps1)
  - [`restart-ai-openkore.ps1`](./restart-ai-openkore.ps1)

### Configuration templates

- Launcher env template: [`ai-sidecar-launcher.env.example`](./ai-sidecar-launcher.env.example)
- Bridge control template: [`ai-sidecar-control.template.txt`](./ai-sidecar-control.template.txt)
- Bridge policy template: [`ai-sidecar-policy.template.txt`](./ai-sidecar-policy.template.txt)

## Runtime directories

Scripts use:

- PID files: `.ai-sidecar-runtime/pids/`
- Logs: `.ai-sidecar-runtime/logs/`
- Setup backups: `.ai-sidecar-runtime/backups/<timestamp>/`

These runtime artifacts are ignored by git via [`.gitignore`](./.gitignore).

## Linux quick flow

1. Run setup:

   ```bash
   ./setup-ai-sidecar.sh
   ```

2. Start both services:

   ```bash
   ./start-ai-openkore.sh
   ```

3. Check status:

   ```bash
   ./status-ai-openkore.sh
   ```

4. Stop:

   ```bash
   ./stop-ai-openkore.sh
   ```

## Windows quick flow

1. Run setup:

   ```powershell
   .\setup-ai-sidecar.ps1
   ```

2. Start both services:

   ```powershell
   .\start-ai-openkore.ps1
   ```

   or batch wrapper:

   ```bat
   start-ai-openkore.cmd
   ```

3. Check status:

   ```powershell
   .\status-ai-openkore.ps1
   ```

4. Stop:

   ```powershell
   .\stop-ai-openkore.ps1
   ```

