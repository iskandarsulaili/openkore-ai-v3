# REST API contracts and endpoint catalog

## Contract conventions

### Transport

- JSON request and response bodies
- local HTTP only for the Perl bridge path
- current bridge implementation rejects non-`http://` base URLs

### Shared metadata

Most write endpoints carry `meta` with these fields:

| Field | Meaning |
| --- | --- |
| `contract_version` | Contract version string, currently `v1` |
| `emitted_at` | Timestamp provided by the caller |
| `trace_id` | Caller-generated trace identifier |
| `source` | Caller source, such as `openkore-bridge` |
| `bot_id` | Per-bot partition key used across the sidecar |

### Bridge-emitted defaults

The Perl bridge currently emits:

- `contract_version` from [`control/ai_sidecar.txt`](../../control/ai_sidecar.txt)
- `source` defaulting to `openkore-bridge`
- `bot_id` generated from OpenKore runtime identity

## Endpoint catalog

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/health/live` | Process liveness |
| `GET` | `/v1/health/ready` | Runtime readiness and degradation summary |
| `POST` | `/v1/ingest/register` | Register or refresh a bot identity |
| `POST` | `/v1/ingest/snapshot` | Push a bot state snapshot |
| `POST` | `/v1/actions/queue` | Queue an action for a bot |
| `POST` | `/v1/actions/next` | Poll the next action for a bot |
| `POST` | `/v1/acknowledgements/action` | Acknowledge action execution result |
| `POST` | `/v1/macros/publish` | Compile and publish macro artifacts |
| `POST` | `/v1/telemetry/ingest` | Ingest telemetry events |
| `GET` | `/v1/telemetry/summary` | Aggregated telemetry window summary |
| `GET` | `/v1/telemetry/incidents` | Recent warning/error telemetry |
| `GET` | `/v1/fleet/bots` | List bots |
| `GET` | `/v1/fleet/bots/{bot_id}` | Per-bot detailed status |
| `PUT` | `/v1/fleet/bots/{bot_id}/assignment` | Update role, assignment, and attributes |
| `GET` | `/v1/fleet/status` | Fleet-wide totals |
| `GET` | `/v1/fleet/audit` | Audit history query |
| `GET` | `/v1/fleet/memory/{bot_id}` | Semantic context lookup plus recent episodes |
| `POST` | `/v2/reflex/rules` | Add or update a reflex rule for a bot |
| `GET` | `/v2/reflex/rules/{bot_id}` | List active reflex rules for a bot |
| `POST` | `/v2/reflex/rules/{rule_id}/enable` | Enable or disable one reflex rule |
| `GET` | `/v2/reflex/triggers/{bot_id}` | Recent reflex trigger decisions and outcomes |
| `GET` | `/v2/reflex/breakers/{bot_id}` | Reflex circuit breaker state by family |

## Health routes

### `GET /v1/health/live`

Returns process liveness only.

Response fields:

- `ok`
- `status = live`
- `started_at`
- `now`

### `GET /v1/health/ready`

Returns operational state for the current runtime.

Response fields include:

- registered bot count from the in-memory registry
- persisted bot count when repositories are available
- snapshot cache size and persisted snapshot count
- telemetry event count and backlog size
- `persistence_enabled`
- `persistence_degraded`
- `sqlite_path`
- rolling average route latency
- runtime counters

## Ingest routes

### `POST /v1/ingest/register`

Request model: `BotRegistrationRequest`

Important current bridge behavior:

- bridge sends `bot_name`
- bridge sends `capabilities`
- bridge sends `attributes.reason` and `attributes.master`
- bridge does **not** currently send `role` or `assignment`

Current bridge capabilities list:

- `bridge_snapshot_push`
- `bridge_action_poll`
- `bridge_action_ack`
- `bridge_telemetry_push`
- `bridge_macro_reload_orchestration`

Response model: `BotRegistrationResponse`

### `POST /v1/ingest/snapshot`

Request model: `BotStateSnapshot`

Current snapshot fields actually populated by the bridge:

| Section | Current fields |
| --- | --- |
| `position` | `map`, `x`, `y` |
| `vitals` | `hp`, `hp_max`, `sp`, `sp_max`, `weight`, `weight_max` |
| `combat` | `ai_sequence`, `is_in_combat` |
| `inventory` | `zeny`, `item_count` |
| `raw` | `char_name`, `master`, `ai_sequence`, `ai_queue` |

Current contract field present but not bridge-populated now:

- `combat.target_id`

Response model: `SnapshotIngestResponse`

## Actions

### Action statuses

| Status | Meaning |
| --- | --- |
| `queued` | Accepted and waiting for dispatch |
| `dispatched` | Returned by `/v1/actions/next` and awaiting acknowledgement |
| `acknowledged` | Executed successfully and acknowledged |
| `expired` | TTL elapsed before completion |
| `dropped` | Rejected, failed, or negatively acknowledged |
| `superseded` | Replaced by a better conflicting queued action |

### Priority tiers

Actual queue precedence is:

1. `reflex`
2. `tactical`
3. `strategic`
4. `macro_management`

`macro_management` is therefore the lowest-priority implemented tier.

### `POST /v1/actions/queue`

Request model: `QueueActionRequest`

Admission behavior implemented now:

- if `expires_at <= created_at`, the API rewrites `expires_at` to `created_at + default_ttl`
- if the action is already expired by the time it reaches the queue, it is rejected as `expired`
- duplicate `idempotency_key` for the same bot returns the existing action while it remains active
- same-bot queued actions sharing a `conflict_key` compete by queue ordering
- when the queue is full, lower-priority work may be dropped

Response model: `QueueActionResponse`

### `POST /v1/actions/next`

Request model: `NextActionRequest`

Response model: `NextActionResponse`

Important runtime behavior:

- on success, `has_action = true` and `action` is an `ActionProposal`
- when no action is available, `has_action = false` and `action` is a noop payload
- when route latency exceeds the configured budget, the sidecar rolls the dispatch back and still returns a noop payload with `reason = latency_budget_exceeded`

### `POST /v1/acknowledgements/action`

Request model: `ActionAckRequest`

Response model: `ActionAckResponse`

Current behavior:

- `success = true` moves the action to `acknowledged`
- `success = false` moves the action to `dropped`
- the runtime accepts acknowledgements for actions still in `queued` or `dispatched`

## Bridge-executable action subset

The sidecar accepts generic action payloads, but the current bridge only executes this subset:

| `kind` | Supported by bridge | Notes |
| --- | --- | --- |
| `command` | Yes | Executed via `Commands::run` if policy allows it |
| `macro_reload` | Yes | Executed through the bridge's internal safe reload flow |
| anything else | No | Acknowledged as `unsupported_kind` |

Additional end-to-end constraint:

- `ActionProposal.command` contract allows up to 256 characters
- bridge policy enforces a stricter runtime maximum from `aiSidecar_maxCommandLength`, default `160`

For bridge-driven command actions, the effective operational limit is therefore 160 characters unless the control file is changed.

## Macros

### `POST /v1/macros/publish`

Request model: `MacroPublishRequest`

Inputs:

- `macros`
- `event_macros`
- `automacros`
- optional `target_bot_id`
- `enqueue_reload`
- `reload_conflict_key`
- `macro_plugin`
- `event_macro_plugin`

Response model: `MacroPublishResponse`

If publication succeeds, the response includes:

- publication id
- version
- content SHA-256
- published timestamp
- artifact paths
- reload queue result fields

Important implemented behavior:

- macro publication writes files first, then queues reload
- reload is not performed inline inside the publish request
- queued reload uses `kind = macro_reload` and priority `macro_management`

## Telemetry

### `POST /v1/telemetry/ingest`

Request model: `TelemetryIngestRequest`

Response model: `TelemetryIngestResponse`

Telemetry repository behavior:

- persists warning and error levels as incidents
- trims per-bot history to the configured maximum
- uses a durable backlog if repository ingestion fails during runtime

### `GET /v1/telemetry/summary`

Query parameter:

- optional `bot_id`

Response model: `TelemetrySummaryResponse`

Summary fields:

- current window size in minutes
- window start timestamp
- total events
- per-level counts
- top events
- recent incidents

### `GET /v1/telemetry/incidents`

Query parameters:

- optional `bot_id`
- `limit`

Returns recent warning/error events only.

## Fleet and memory routes

### `GET /v1/fleet/bots`

Lists current bot views with:

- identity, role, assignment, capabilities, attributes
- first/last seen timestamps
- last tick id
- pending action count
- latest snapshot time
- telemetry event count

### `GET /v1/fleet/bots/{bot_id}`

Adds:

- recent actions
- recent snapshots
- latest macro publication
- recent audit events

### `PUT /v1/fleet/bots/{bot_id}/assignment`

Allows external operators to set or update:

- `role`
- `assignment`
- arbitrary string `attributes`

### `GET /v1/fleet/status`

Fleet-wide summary fields:

- total bots
- online bots
- total pending actions
- action totals by status
- telemetry window summary
- counters

### `GET /v1/fleet/audit`

Query parameters:

- optional `bot_id`
- optional `event_type`
- `limit`

### `GET /v1/fleet/memory/{bot_id}`

Query parameters:

- `query`
- `limit`

Returns:

- semantic matches
- recent episodes
- provider stats

## Implemented now vs extension points

### Implemented now

- all routes listed above are present in the local FastAPI application
- contracts are enforced through Pydantic models
- bridge traffic uses `/v1/ingest`, `/v1/actions`, `/v1/acknowledgements`, and `/v1/telemetry`

### Future extension points, not implemented now

- additional bridge-executable action kinds beyond `command` and `macro_reload`
- stronger transport/authentication layers for non-local deployments
- richer typed fleet/history contracts instead of some current `dict[str, object]` payload sections

## Reflex layer routes (`/v2/reflex`)

The reflex layer is deterministic and non-LLM. It evaluates rule DSL predicates against incoming normalized events plus enriched world state, then executes in strict target order:

1. direct queue action
2. published micro-macro
3. eventMacro-bound trigger command

Every trigger writes an outcome record and uses ack feedback to update breaker health.

### `POST /v2/reflex/rules`

Request model: `ReflexRuleUpsertRequest`

Payload fields include:

- `rule_id`, `enabled`, `priority`
- trigger clause with `all` and/or `any` predicates
- `guards`
- `action_template`
- `fallback_macro`
- `cooldown_ms`
- `circuit_breaker_key`
- optional `event_macro_conditions`

Response model: `ReflexRuleUpsertResponse`

### `GET /v2/reflex/rules/{bot_id}`

Response model: `ReflexRuleListResponse`

Returns all active persisted-in-runtime rule definitions for the bot (defaults plus overrides).

### `POST /v2/reflex/rules/{rule_id}/enable`

Request model: `ReflexRuleEnableRequest`

Response model: `ReflexRuleEnableResponse`

Uses `meta.bot_id` to scope which bot's rule set is updated.

### `GET /v2/reflex/triggers/{bot_id}`

Query parameters:

- optional `limit`

Response model: `ReflexTriggerListResponse`

Includes per-trigger details such as:

- match latency
- suppression reason
- execution target
- emitted action id
- ack-updated outcome

### `GET /v2/reflex/breakers/{bot_id}`

Response model: `ReflexBreakerListResponse`

Implemented breaker families:

- provider
- macro
- combat
- social
- fleet
- queue
