# Operations, setup, health, artifacts, and rollback

## Local deployment model

The implemented sidecar is a **local companion service** for one or more OpenKore processes on the same machine or trusted local network segment.

The bridge configuration in [`control/ai_sidecar.txt`](../../control/ai_sidecar.txt) points to `http://127.0.0.1:18081` by default.

## Setup

### Python prerequisites

- Python 3.11 or newer
- ability to install the package in [`AI_sidecar`](..)

### Installation

```bash
cd AI_sidecar
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

### Start

```bash
openkore-ai-sidecar
```

### Smoke test

```bash
curl http://127.0.0.1:18081/v1/health/live
curl http://127.0.0.1:18081/v1/health/ready
```

## Sidecar configuration

Environment variables are defined in [`.env.example`](../.env.example).

Operationally important settings:

| Variable | Purpose |
| --- | --- |
| `OPENKORE_AI_HOST` / `OPENKORE_AI_PORT` | Bind address |
| `OPENKORE_AI_LOG_LEVEL` / `OPENKORE_AI_LOG_JSON` | Logging behavior |
| `OPENKORE_AI_ENABLE_DOCS` | Enable or disable FastAPI docs pages |
| `OPENKORE_AI_ACTION_DEFAULT_TTL_SECONDS` | Default TTL when a queued action has invalid expiry ordering |
| `OPENKORE_AI_ACTION_MAX_QUEUE_PER_BOT` | Maximum queue depth per bot |
| `OPENKORE_AI_LATENCY_BUDGET_MS` | Budget for `/v1/actions/next` before noop fallback |
| `OPENKORE_AI_SQLITE_PATH` | Primary SQLite database |
| `OPENKORE_AI_MEMORY_BACKEND` | `sqlite`, `openmemory`, or `auto` |
| `OPENKORE_AI_MEMORY_OPENMEMORY_PATH` | Local OpenMemory path |

## OpenKore bridge control plane

### Control files

| File | Current role |
| --- | --- |
| [`control/ai_sidecar.txt`](../../control/ai_sidecar.txt) | Bridge enablement, timeouts, polling intervals, payload bounds, macro reload settings |
| [`control/ai_sidecar_policy.txt`](../../control/ai_sidecar_policy.txt) | Command allow/deny policy for bridge-executed console commands |
| [`control/ai_sidecar_macros.txt`](../../control/ai_sidecar_macros.txt) | Published macro catalog pointer file |

### Plugin loading

[`control/sys.txt`](../../control/sys.txt) currently uses `loadPlugins 2` and includes `aiSidecarBridge` in `loadPlugins_list`.

That means the bridge is already part of the configured startup plugin set in this repository.

### Bridge timing defaults

These default values come from [`control/ai_sidecar.txt`](../../control/ai_sidecar.txt):

| Setting | Default |
| --- | --- |
| connect timeout | 40 ms |
| IO timeout | 90 ms |
| snapshot interval | 500 ms |
| action poll interval | 250 ms |
| acknowledgement retry | 500 ms |
| acknowledgement max age | 5000 ms |
| registration retry | 5000 ms |
| telemetry flush interval | 1000 ms |

## Artifact locations

### Sidecar-owned runtime artifacts

| Path | Current contents |
| --- | --- |
| `AI_sidecar/data/sidecar.sqlite` | Primary SQLite state store |
| `AI_sidecar/data/openmemory.sqlite` | OpenMemory local store path when used |

### Macro publication artifacts

| Path | Current contents |
| --- | --- |
| `control/ai_sidecar_generated_macros.txt` | Generated `macro` plugin file |
| `control/ai_sidecar_generated_eventmacros.txt` | Generated `eventMacro` file |
| `control/ai_sidecar_macros.txt` | Catalog pointer file |
| `control/ai_sidecar_macro_manifest.json` | Latest publication manifest |

### Logs

The sidecar logs to stdout. JSON logging is enabled by default through [`.env.example`](../.env.example).

The bridge logs through normal OpenKore logging and warning pathways.

## Health and observability

### Health endpoints

- `/v1/health/live` confirms the FastAPI process is up
- `/v1/health/ready` shows operational readiness and degradation flags

### Telemetry surfaces

- bridge sends batched telemetry to `/v1/telemetry/ingest`
- sidecar exposes `/v1/telemetry/summary` and `/v1/telemetry/incidents`
- warning and error telemetry are treated as incidents in SQLite

### Audit trail

Audit records are written for:

- runtime initialization
- bot registration
- action queue decisions
- action acknowledgements
- telemetry ingest batches
- macro publication
- bot assignment updates
- dispatch rollback caused by latency fallback

Audit queries are exposed at `/v1/fleet/audit` and through per-bot fleet detail responses.

## Degradation runbook

| Condition | Implemented effect | Operational meaning |
| --- | --- | --- |
| `JSON::PP` unavailable in OpenKore | Bridge disables itself and logs warning | OpenKore continues with no sidecar traffic |
| sidecar unreachable | registration/snapshot/poll/telemetry failures warn and remain fail-open | OpenKore keeps running without remote decisions |
| `/v1/actions/next` exceeds latency budget | sidecar rolls back dispatch and returns noop | OpenKore receives no action for that poll cycle |
| acknowledgement upload fails | bridge retries until ack age limit, then drops stale ack | executed command is not rolled back |
| persistence init fails | sidecar marks `persistence_degraded` and falls back | health shows degraded state |
| OpenMemory init fails | sidecar falls back to SQLite or in-memory memory | memory remains available but provider changed |

## Rollback behavior

### What is rolled back now

1. **Action dispatch before response return**  
   If `/v1/actions/next` exceeds the latency budget after selecting an action, the sidecar rolls the action back from `dispatched` to `queued` and returns a noop.

2. **Macro file publication failure during disk writes**  
   The macro publisher snapshots previous file contents and restores them if any artifact write fails.

### What is not rolled back now

1. **Already executed OpenKore console commands**  
   Once `Commands::run` has executed, later acknowledgement or telemetry failure does not reverse the command.

2. **Failed bridge-side macro reload after successful file publication**  
   If generated files were published successfully but the bridge later fails while running `conf` or `plugin reload`, the published files remain on disk.

3. **Dropped stale acknowledgements**  
   When ack age exceeds `aiSidecar_ackMaxAgeMs`, the bridge drops the queued ack; it does not reconstruct execution state.

## Local fleet operations

The local sidecar can manage multiple bots concurrently.

Operational surfaces:

- `/v1/fleet/bots`
- `/v1/fleet/bots/{bot_id}`
- `/v1/fleet/status`
- `/v1/fleet/bots/{bot_id}/assignment`
- `/v1/fleet/memory/{bot_id}`

Current implementation notes:

- queueing and snapshots are isolated per `bot_id`
- telemetry and memory are partitioned per `bot_id`
- fleet aggregation is local to this sidecar process only
- liveness is currently optimistic; bots are not automatically marked offline by timeout expiry

## Implemented now vs extension points

### Implemented now

- reproducible local setup from this repository
- explicit readiness, telemetry, audit, and fleet endpoints
- documented artifact paths and degradation states

### Future extension points, not implemented now

- automatic offline detection and expiry-based liveness
- richer operator tooling around replay, repair, and queue administration
- automatic rollback choreography for post-dispatch OpenKore failures
