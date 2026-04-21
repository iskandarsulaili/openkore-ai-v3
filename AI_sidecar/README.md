# OpenKore AI Sidecar (local)

Local FastAPI sidecar for the implemented OpenKore bridge in this repository.

This directory documents the **current local sidecar as implemented now**. It is the source of truth for:

- the Perl bridge plugin lifecycle and fail-open behavior
- the local REST API contract used by the bridge
- action queueing, polling, acknowledgement, and latency-budget fallback
- macro publication and hot reload through the existing `macro` and `eventMacro` plugins
- SQLite persistence, local memory capture, and optional OpenMemory-backed retrieval
- multi-bot local fleet status, audit, and operational runbooks

## Implemented now

- Local HTTP bridge plugin at `plugins/aiSidecarBridge/aiSidecarBridge.pl`
- FastAPI sidecar with `/v1/health`, `/v1/ingest`, `/v1/actions`, `/v1/acknowledgements`, `/v1/macros`, `/v1/telemetry`, and `/v1/fleet`
- In-memory bot registry, snapshot cache, action queue, and latency router
- Durable SQLite repositories for bot registry, snapshots, actions, telemetry, macro publications, audit events, and local memory
- Macro artifact publication into `control/` plus queued reload orchestration back through OpenKore command pathways
- Optional OpenMemory provider with SQLite or in-memory fallback

## Not implemented in this local sidecar

- No direct replacement of OpenKore AI internals; OpenKore remains authoritative for execution
- No HTTPS bridge transport; the Perl bridge accepts `http://` only
- No automatic rollback of an already executed OpenKore command if acknowledgement later fails
- No automatic offline-state transition based on heartbeat expiry
- No broad repository-wide signal ingestion beyond the bridge snapshot, telemetry, command, and macro surfaces documented here

## Repository surfaces covered by these docs

- Bridge plugin: [`plugins/aiSidecarBridge/aiSidecarBridge.pl`](../plugins/aiSidecarBridge/aiSidecarBridge.pl)
- Local sidecar package: [`AI_sidecar/ai_sidecar`](./ai_sidecar)
- Bridge-loaded control files and sidecar publication artifacts:
  - [`control/ai_sidecar.txt`](../control/ai_sidecar.txt)
  - [`control/ai_sidecar_policy.txt`](../control/ai_sidecar_policy.txt)
  - [`control/ai_sidecar_macros.txt`](../control/ai_sidecar_macros.txt) (published catalog artifact, not bridge-loaded at startup)
  - [`control/ai_sidecar_macro_manifest.json`](../control/ai_sidecar_macro_manifest.json)
- Macro/plugin surfaces:
  - [`plugins/macro/macro.pl`](../plugins/macro/macro.pl)
  - [`plugins/eventMacro/eventMacro.pl`](../plugins/eventMacro/eventMacro.pl)
  - [`plugins/profiles/profiles.pl`](../plugins/profiles/profiles.pl)
- Core lifecycle touchpoints:
  - [`openkore.pl`](../openkore.pl)
  - [`src/functions.pl`](../src/functions.pl)

## Quick start

### 1. Prepare the Python environment

From [`AI_sidecar`](.):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. Create the environment file

```bash
cp .env.example .env
```

Important implementation detail:

- code defaults `memory_backend` to `openmemory`
- [`.env.example`](./.env.example) overrides that to `sqlite`

If you copy [`.env.example`](./.env.example) without changes, the sidecar runs with SQLite-backed memory by default.

### 3. Start the sidecar

```bash
openkore-ai-sidecar
```

Default bind:

- host: `127.0.0.1`
- port: `18081`

Optional alternative:

```bash
python -m ai_sidecar.app
```

### 4. Verify health

```bash
curl http://127.0.0.1:18081/v1/health/live
curl http://127.0.0.1:18081/v1/health/ready
```

If `OPENKORE_AI_ENABLE_DOCS=1`, FastAPI docs are also exposed at:

- `http://127.0.0.1:18081/docs`
- `http://127.0.0.1:18081/redoc`

### 5. Verify OpenKore plugin loading

[`control/sys.txt`](../control/sys.txt) already includes `aiSidecarBridge` in `loadPlugins_list`. The bridge loads through the normal OpenKore plugin startup path.

The bridge then loads its own control files:

- [`control/ai_sidecar.txt`](../control/ai_sidecar.txt)
- [`control/ai_sidecar_policy.txt`](../control/ai_sidecar_policy.txt)

## Documentation set

- [Architecture and runtime flow](./docs/architecture.md)
- [REST API contracts and endpoint catalog](./docs/api-contracts.md)
- [Operations, setup, health, artifacts, and rollback](./docs/operations.md)
- [Persistence and memory subsystem](./docs/persistence-memory.md)
- [Macro publication and hot-reload workflow](./docs/macro-workflow.md)
- [Observable-state inventory and next repository-grounded integrations](./docs/observable-state-inventory.md)

## Recommended reading order

1. Start with [architecture](./docs/architecture.md)
2. Then read [API contracts](./docs/api-contracts.md)
3. Use [operations](./docs/operations.md) for deployment and incident handling
4. Use [persistence and memory](./docs/persistence-memory.md) for storage and retrieval details
5. Use [macro workflow](./docs/macro-workflow.md) for publication and reload behavior
6. Use [observable-state inventory](./docs/observable-state-inventory.md) to understand what is and is not currently ingested
