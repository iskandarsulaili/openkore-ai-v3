# Observable-state inventory

## Purpose

This document inventories the OpenKore-side signals and configuration surfaces that the local sidecar can consume **now**, and separates them from nearby repository-grounded sources that are visible in this codebase but **not yet wired into the sidecar**.

This is intentionally conservative. It does not claim ingestion of any signal that is not connected in the current implementation.

## Consumed now

### 1. OpenKore lifecycle hooks

| Surface | Evidence | Current use |
| --- | --- | --- |
| `start3` | `plugins/aiSidecarBridge/aiSidecarBridge.pl` | initial control-file load, state initialization, registration attempt |
| `mainLoop_pre` | same file | periodic snapshot push |
| `mainLoop_post` | same file | registration retry, action poll, ack flush, telemetry flush |

### 2. OpenKore identity and basic config

| Surface | Current ingestion |
| --- | --- |
| `%config{master}` | used in bot id generation and snapshot raw payload |
| `%config{username}` | fallback identity when character name is unavailable |
| `$char->{name}` | bot name and identity |

### 3. Network state

| Surface | Current ingestion |
| --- | --- |
| `$net->getState()` | used to suppress snapshot polling and action polling when not in game unless configured otherwise |

### 4. Spatial state

| Surface | Current ingestion |
| --- | --- |
| `Misc::calcPosition($char)` | `position.x`, `position.y` |
| `$field->baseName()` | `position.map` |

### 5. Vitals and inventory digest

| Surface | Current ingestion |
| --- | --- |
| `$char->{hp}`, `$char->{hp_max}` | snapshot vitals |
| `$char->{sp}`, `$char->{sp_max}` | snapshot vitals |
| `$char->{weight}`, `$char->{weight_max}` | snapshot vitals |
| `$char->{zeny}` | inventory digest |
| `$char->{inventory}` array length | inventory item count only |

### 6. AI state digest

| Surface | Current ingestion |
| --- | --- |
| `@ai_seq[0]` | top AI sequence name |
| first few entries of `@ai_seq` | raw `ai_queue` summary |
| top sequence name matching `attack|skill_use|route|follow` | derived `is_in_combat` flag |

### 7. Raw trimmed strings

The bridge currently includes a bounded `raw` section containing:

- character name
- master name
- current AI sequence
- short AI queue summary

These are truncated by `aiSidecar_maxRawChars`.

### 8. Action execution results

| Surface | Current ingestion |
| --- | --- |
| bridge poll id | included in action polling and ack payloads |
| action success/failure | ack payload |
| result code/message | ack payload and telemetry |
| observed latency | ack payload and telemetry metrics |

### 9. Bridge-local telemetry

The bridge emits telemetry events for:

- registration failures
- snapshot failures
- poll failures
- action execution outcomes
- macro reload outcomes

Telemetry ingestion is implemented now. Telemetry itself is not a direct OpenKore core signal; it is bridge-generated operational reporting.

### 10. Bridge control surfaces actually consumed now

| Surface | Current ingestion |
| --- | --- |
| [`control/ai_sidecar.txt`](../../control/ai_sidecar.txt) | bridge behavior and safety limits |
| [`control/ai_sidecar_policy.txt`](../../control/ai_sidecar_policy.txt) | command allow/deny enforcement |

### 11. Sidecar-published artifact surfaces

These files are present and operationally useful, but they are **not currently read by the bridge as configuration inputs**:

- [`control/ai_sidecar_macros.txt`](../../control/ai_sidecar_macros.txt)
- [`control/ai_sidecar_macro_manifest.json`](../../control/ai_sidecar_macro_manifest.json)

### 12. Existing plugin reload surfaces

The sidecar can affect existing plugin state through the bridge's safe reload flow:

- `macro_file` config plus `plugin reload macro`
- `eventMacro_file` config plus `plugin reload eventMacro`

This is an implemented control surface, not a read-side ingestion surface.

### 13. Sidecar-derived local memory

These are not native OpenKore signals, but they are derived from currently ingested state and therefore observable through the local sidecar:

- snapshot summaries
- action queue/dispatch/ack summaries
- macro publication summaries

## Implemented-now limits

These fields exist in contracts or nearby code but are **not currently populated by the bridge**:

| Surface | Status now |
| --- | --- |
| `combat.target_id` in snapshot contract | not populated |
| detailed inventory contents | not populated |
| party, actor, monster, NPC, pet, portal lists | not populated |
| chat contents and message streams | not populated |
| route path details / AI sequence arguments | not populated |
| profile selection details | not sent to sidecar |
| macro/eventMacro runtime state | not ingested |
| explicit offline transitions | not computed |

## Adjacent repository-grounded sources to integrate next

These sources are visible in the repository and are realistic next integration candidates, but they are **not wired to the sidecar today**.

### Priority 1: macro and event automation runtime state

| Source | Repository evidence | Why it matters | Status now |
| --- | --- | --- | --- |
| `macro` plugin automacro hooks and log hooks | [`plugins/macro/macro.pl`](../../plugins/macro/macro.pl) | exposes trigger surfaces such as packet, AI, and console-driven macro activation | not ingested |
| `eventMacro` trigger and condition engine | [`plugins/eventMacro/eventMacro.pl`](../../plugins/eventMacro/eventMacro.pl) | exposes triggered automacro state, condition checks, and file includes | not ingested |

Concrete next signals available there include:

- automacro trigger state
- macro currently running / paused / finished
- eventMacro variable state
- eventMacro include enablement and parse outcomes

### Priority 2: chat and packet-driven event streams

Repository-grounded evidence from the macro plugin shows hooks for:

- `packet_privMsg`
- `packet_pubMsg`
- `packet_sysMsg`
- `packet_partyMsg`
- `packet_guildMsg`
- `packet_mapChange`
- `packet_skilluse`
- `packet_areaSpell`

These are valuable next ingestion targets because they carry real-time behavior signals already used elsewhere in the repository.

Status now: not ingested by the sidecar bridge.

### Priority 3: richer AI and routing state

Repository evidence:

- [`src/functions.pl`](../../src/functions.pl)
- OpenKore AI modules loaded in [`openkore.pl`](../../openkore.pl)

Useful next signals:

- target identifiers and attack context
- route destination and route failure state
- AI sequence arguments, not only names
- disconnect / reconnect / transition events

Status now: only top-level `@ai_seq` name and a short queue digest are used.

### Priority 4: control-file and table-driven behavior inputs

Repository evidence:

- broad control-file loading in [`src/functions.pl`](../../src/functions.pl)

Useful next configuration/state ingestion targets:

- `config.txt`
- `mon_control.txt`
- `items_control.txt`
- `shop.txt`
- `buyer_shop.txt`
- `timeouts.txt`
- `priority.txt`
- route weight and avoid files

Status now: the sidecar bridge only reads its own dedicated `ai_sidecar*.txt` files.

### Priority 5: profile-aware fleet context

Repository evidence:

- [`plugins/profiles/profiles.pl`](../../plugins/profiles/profiles.pl)

Why it matters:

- local fleet operators often separate bots by profile folder
- per-profile plugin loading and control-folder resolution can materially change observable behavior

Status now:

- the repository loads the bridge through `loadPlugins_list`
- profile state itself is not sent to the sidecar

## Recommended next integration order

1. Macro/eventMacro runtime state and trigger outcomes
2. Chat and packet event streams already proven useful by macro automation
3. Richer AI target/routing context
4. Profile and control-folder context
5. Inventory/entity detail expansion

## Implemented now vs extension points

### Implemented now

- bridge consumes a compact, bounded, low-latency subset of OpenKore runtime state
- that subset is sufficient for registration, snapshots, queue polling, acknowledgement, telemetry, memory capture, and macro reload orchestration

### Future extension points, not implemented now

- broad event ingestion from existing automation plugins
- richer actor/chat/control/table awareness
- stronger semantic understanding of OpenKore state beyond the current compact snapshot digest
