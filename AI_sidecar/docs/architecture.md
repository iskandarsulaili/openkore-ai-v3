# Sidecar architecture and runtime flow

## Scope

This document describes the **implemented local architecture** in [`AI_sidecar`](..), not a conceptual future platform.

## Runtime components

### OpenKore side

1. [`openkore.pl`](../../openkore.pl) starts OpenKore and hands control to the normal interface loop.
2. [`src/functions.pl`](../../src/functions.pl) loads plugins and invokes plugin hooks in `mainLoop()`.
3. [`plugins/aiSidecarBridge/aiSidecarBridge.pl`](../../plugins/aiSidecarBridge/aiSidecarBridge.pl) registers three hooks:
   - `start3`
   - `mainLoop_pre`
   - `mainLoop_post`
4. The bridge reads:
   - OpenKore globals such as `%config`, `$char`, `$field`, `@ai_seq`, and `$net`
   - bridge control from [`control/ai_sidecar.txt`](../../control/ai_sidecar.txt)
   - command policy from [`control/ai_sidecar_policy.txt`](../../control/ai_sidecar_policy.txt)
5. The bridge performs local HTTP POST requests to the Python sidecar.

### Python side

[`AI_sidecar/ai_sidecar`](../ai_sidecar) contains these implemented responsibilities:

| Package area | Current responsibility |
| --- | --- |
| `app.py` | FastAPI application assembly and process entry point |
| `config.py` | Environment-based configuration via `OPENKORE_AI_*` |
| `lifecycle.py` | Runtime construction plus orchestration for registration, snapshots, actions, telemetry, macros, audit, and memory |
| `api/routers` | HTTP routes and contract enforcement |
| `contracts` | Pydantic request/response models |
| `runtime` | In-memory bot registry, action queue, latency budget tracking, snapshot cache |
| `persistence` | SQLite schema and repositories |
| `observability` | Durable telemetry ingestion and audit wrapper |
| `memory` | Local embedding, episodic store, semantic store, OpenMemory/SQLite/in-memory providers |
| `domain/macro_compiler.py` | Macro compilation, artifact publication, and disk rollback during file-write failure |

## End-to-end control loop

The current implementation is easiest to reason about as a PDCA loop.

| Phase | Implemented now |
| --- | --- |
| Plan | Clients enqueue actions or publish macro bundles through the sidecar API. The sidecar decides queue admission using priority tier, idempotency key, conflict key, TTL, and queue capacity. |
| Do | The bridge polls `/v1/actions/next` and dispatches either a console command or an internal macro reload flow through `Commands::run`. |
| Check | The bridge pushes snapshots and telemetry, and returns action acknowledgements. Operators inspect `/v1/health`, `/v1/telemetry`, `/v1/fleet`, audit history, SQLite records, and macro artifacts. |
| Act | New actions can supersede queued conflicting work, operators can update bot assignments, and new macro publications can replace previous published artifacts and queue another reload. |

This is an operational loop, not an autonomous planner built into the repository.

## Bridge lifecycle integration points

### Startup

At `start3`, the bridge:

- checks whether `JSON::PP` is available
- registers control files through OpenKore `Settings`
- loads bridge config and bridge policy
- initializes next-run timestamps for registration, snapshot, poll, acknowledgement, and telemetry
- attempts initial bot registration with `/v1/ingest/register`

If `JSON::PP` is unavailable, the bridge logs a warning and disables itself without stopping OpenKore.

### Main loop pre-hook

At `mainLoop_pre`, the bridge only performs snapshot pushing. This keeps outbound state capture before the normal OpenKore post-loop action polling step.

### Main loop post-hook

At `mainLoop_post`, the bridge may:

- retry registration if needed
- poll for the next action
- flush one pending acknowledgement
- flush a telemetry batch

This sequence makes action execution and acknowledgement part of the ordinary OpenKore loop instead of a separate thread.

## Data flow

```text
OpenKore mainLoop
  -> aiSidecarBridge start3/mainLoop_pre/mainLoop_post
    -> POST /v1/ingest/register
    -> POST /v1/ingest/snapshot
    -> POST /v1/actions/next
      -> ActionQueue + LatencyRouter
      -> return action or noop fallback
    -> OpenKore Commands::run(...) or macro reload safe flow
    -> POST /v1/acknowledgements/action
    -> POST /v1/telemetry/ingest

Python sidecar
  -> RuntimeState
  -> SQLite repositories / in-memory runtime / memory providers
  -> fleet, telemetry, audit, memory, macro artifact visibility
```

## Snapshot responsibilities

The bridge currently sends a compact bot-state snapshot containing:

- `position.map`, `position.x`, `position.y`
- `vitals.hp`, `hp_max`, `sp`, `sp_max`, `weight`, `weight_max`
- `combat.ai_sequence`, `combat.is_in_combat`
- `inventory.zeny`, `inventory.item_count`
- raw trimmed fields for `char_name`, `master`, `ai_sequence`, and a short `ai_queue`

The snapshot is then:

- stored in the in-memory snapshot cache
- written to SQLite snapshot history when persistence is enabled
- converted into episodic and semantic memory entries

## Action execution model

### Queueing

Actions are stored per bot in [`runtime/action_queue.py`](../ai_sidecar/runtime/action_queue.py). Ordering is determined by:

1. priority tier
2. `created_at`
3. enqueue sequence
4. `action_id`

Implemented priority order is:

1. `reflex`
2. `tactical`
3. `strategic`
4. `macro_management`

Lower numeric order wins.

### Dispatch

The bridge polls `/v1/actions/next`. The sidecar marks the selected action as dispatched before returning it. If the request exceeds the latency budget, the sidecar rolls the dispatch back and returns a noop payload instead.

### Acknowledgement

The bridge later POSTs `/v1/acknowledgements/action`. A successful acknowledgement moves the action to `acknowledged`; a failed execution moves it to `dropped`.

## Macro publication architecture

Macro publication is a sidecar-owned build-and-publish flow:

1. Accept `MacroPublishRequest`
2. Compile macro and eventMacro text
3. Atomically write generated artifacts under [`control`](../../control)
4. Persist publication metadata in SQLite
5. Queue a `macro_reload` action for the target bot
6. Let the bridge perform safe OpenKore reload commands

The bridge does **not** compile macro content. It only performs the reload steps against already published files.

## Multi-bot local fleet model

The sidecar is already multi-bot at the local service level.

- every API payload is keyed by `meta.bot_id`
- bot registry, queueing, snapshots, telemetry, macro publications, audit, and memory are all partitioned by `bot_id`
- fleet endpoints aggregate across all bots known to the process

Current bot identity generated by the bridge is:

- `master:character_name` when a character is available
- otherwise `master:username`

There is no separate central coordination plane in this local sidecar.

## Fail-open and degradation behavior

### OpenKore-side fail-open

The bridge is intentionally conservative:

- registration failure does not stop OpenKore
- snapshot failure does not stop OpenKore
- action poll failure does not stop OpenKore
- telemetry failure does not stop OpenKore
- acknowledgement failure only causes retries within the configured age budget

### Python-side degradation

The sidecar can degrade independently:

- if primary SQLite initialization fails, runtime falls back to in-memory telemetry and may still initialize a memory-only SQLite fallback
- if OpenMemory initialization fails, the provider falls back to SQLite or in-memory memory
- if a route exceeds the latency budget, `/v1/actions/next` returns a noop payload

## Implemented now vs extension points

### Implemented now

- local bridge hooks into the existing OpenKore plugin lifecycle
- one local FastAPI process can serve multiple bots
- queueing, persistence, memory, macro publication, telemetry, audit, and fleet status are operational

### Future extension points, not implemented now

- richer ingestion of actor lists, chat streams, targets, routing state, and plugin-specific events
- explicit liveness expiry / offline transitions
- remote transport, distributed workers, or central orchestration in this local package
- automatic rollback of already executed OpenKore commands
