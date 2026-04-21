# Persistence and memory subsystem

## Persistence overview

The sidecar uses SQLite for durable local state through [`ai_sidecar/persistence/db.py`](../ai_sidecar/persistence/db.py) and [`ai_sidecar/persistence/repositories.py`](../ai_sidecar/persistence/repositories.py).

SQLite initialization enables:

- `journal_mode=WAL`
- `synchronous=NORMAL`
- `foreign_keys=ON`
- configured `busy_timeout`

## Primary database path

Default configured path:

- `AI_sidecar/data/sidecar.sqlite`

This value comes from `OPENKORE_AI_SQLITE_PATH`.

## Schema inventory

| Table | Purpose |
| --- | --- |
| `bot_registry` | Durable bot identity, role, assignment, capabilities, attributes, liveness summary |
| `snapshots` | Snapshot history per bot |
| `actions` | Action queue lifecycle history including ack result fields |
| `telemetry_events` | Raw telemetry event history |
| `telemetry_counters` | Runtime counters aggregated over time |
| `macro_publications` | Latest and historical macro publication metadata |
| `audit_events` | Operator/audit history |
| `memory_episodes` | Episodic memory records |
| `memory_semantic_records` | Semantic memory records |
| `memory_embeddings` | Embedding vectors for semantic records |

## Repository responsibilities

### Bot repository

Persists:

- `bot_name`
- `role`
- `assignment`
- `capabilities`
- `attributes`
- `first_seen_at`, `last_seen_at`, `last_tick_id`
- `liveness_state`

### Snapshot repository

Persists full snapshot payload JSON and trims history per bot to `OPENKORE_AI_PERSISTENCE_SNAPSHOT_HISTORY_PER_BOT`.

### Action repository

Persists:

- original proposal JSON
- priority tier and conflict/idempotency keys
- queued, dispatched, and acknowledged timestamps
- ack result code and message
- `status_reason`

### Telemetry repository

Persists raw telemetry plus:

- per-bot trimming to `OPENKORE_AI_TELEMETRY_MAX_PER_BOT`
- incident flagging for warning and error levels
- operational-window summaries
- cumulative counters

### Macro repository

Persists:

- publication id
- version
- content SHA-256
- manifest JSON
- artifact paths JSON
- macro/eventMacro/automacro counts

### Audit repository

Persists bounded audit history, trimming to `OPENKORE_AI_PERSISTENCE_AUDIT_HISTORY`.

## Runtime persistence degradation

### Normal path

When primary SQLite initialization succeeds:

- `runtime.repositories` is available
- health exposes `persistence_enabled = true`
- fleet, snapshots, actions, telemetry, audit, and macro publication history are fully queryable

### Degraded path

When primary repository initialization fails:

- the runtime logs initialization failure
- `runtime.repositories` becomes `None`
- `runtime.persistence_degraded = true`
- telemetry falls back to in-memory storage
- audit storage is unavailable
- fleet history/detail endpoints lose persisted history sections

Important nuance: the runtime still attempts a **memory-only SQLite fallback** for the memory repository. That means semantic and episodic memory may still remain SQLite-backed even while primary persistence is reported as disabled.

## Memory subsystem overview

Memory is split into:

- **episodic memory** for recent event records
- **semantic memory** for similarity search

The current memory service is constructed in [`ai_sidecar/lifecycle.py`](../ai_sidecar/lifecycle.py).

## Implemented memory providers

| Provider | Current behavior |
| --- | --- |
| `SQLiteMemoryProvider` | Stores episodes and semantic vectors in the sidecar SQLite schema |
| `OpenMemoryProvider` | Uses `openmemory-py` if initialization succeeds, otherwise falls back |
| `InMemoryMemoryProvider` | Final fallback when durable providers are unavailable |

### Configuration sources

- code default in [`ai_sidecar/config.py`](../ai_sidecar/config.py): `memory_backend = openmemory`
- example environment in [`.env.example`](../.env.example): `OPENKORE_AI_MEMORY_BACKEND=sqlite`

The effective backend depends on environment resolution at runtime.

## Local embedding implementation

The SQLite and in-memory semantic paths use a deterministic local embedder from [`ai_sidecar/memory/embeddings.py`](../ai_sidecar/memory/embeddings.py).

Current properties:

- lexical tokenization with a simple regex
- hashed sparse vector generation
- no external model download
- cosine similarity over stored vectors

This is a local deterministic retrieval mechanism, not an external embedding-model integration.

## Captured memory sources implemented now

The sidecar currently captures memory from these events:

| Source | Episodic | Semantic |
| --- | --- | --- |
| Snapshot ingest | Yes | Yes |
| Action queue decision | Yes | Yes |
| Action dispatch | Yes | Yes |
| Action acknowledgement | Yes | Yes |
| Macro publication | Yes, through action-style capture | Yes |

More concretely:

- snapshot summaries include map, coordinates, HP summary, and AI state
- action memory entries include queue/dispatch/ack phases and relevant metadata
- macro publication is stored as a `macro_publish` action-style memory item

## Memory retrieval surfaces

`GET /v1/fleet/memory/{bot_id}` returns:

- semantic matches for the query
- recent episodes
- provider stats

## OpenMemory-specific caveats

These are implementation details that matter operationally.

### Search and history

When OpenMemory is enabled successfully:

- semantic search comes from `openmemory.client.Memory.search(...)`
- recent episodes come from `openmemory.client.Memory.history(...)`

### Stats mismatch

`OpenMemoryProvider.stats()` currently delegates to the SQLite fallback provider.

Operational consequence:

- when OpenMemory is active and writes are handled by OpenMemory, `/v1/fleet/memory/{bot_id}` may return OpenMemory matches/history while `stats` under-report or show zero SQLite-backed records

### Normalized episode shape

OpenMemory history is currently mapped into sidecar episode rows with:

- synthetic `event_type = history`
- `created_at` set to the current read time rather than a preserved provider timestamp

This is implemented behavior, not just a documentation caveat.

## In-memory fallback limits

When the in-memory provider is used:

- episodes are truncated to the last 5000 per bot
- semantic records are truncated to the last 10000 per bot
- all memory is process-local and lost on restart

## Implemented now vs extension points

### Implemented now

- durable SQLite schema for full local sidecar state
- deterministic local semantic retrieval without external model dependencies
- optional OpenMemory integration with fallback behavior

### Future extension points, not implemented now

- provider-accurate stats for OpenMemory-backed data
- richer provider-preserved timestamps and metadata normalization
- external embedding services or higher-fidelity vector search engines
