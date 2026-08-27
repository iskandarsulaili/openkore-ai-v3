# CHARSTATUS.JSON — REAL-TIME CHAR STATE ARCHITECTURE
## Implementation Checklist & Tracker

**Goal:** One authoritative, real-time, complete char+world state contract extracted from OpenKore core by the bridge, written to a durable per-bot JSON file, and consumed by ALL three brains (Conscious/LLM, Subconscious/ML, Reflex) via the sidecar. Read-only for brains.

**Design doc:** see the "Answer only" charstatus design (identity/vitals/position/inventory/stats/combat/environment/party/economy/AI/telemetry + adversarial additions).

---

## STATUS: IN PROGRESS (Batch 1 — bridge extraction + sidecar reader + endpoint)

### ✅ DONE
- [x] **Bridge: charstatus.json atomic writer** (`_write_charstatus_file`) — temp+rename atomic write, monotonic `seq` per bot, per-bot file `data/charstatus/charstatus_<bot>.json`, gated by `aiSidecar_charstatusEnabled` (default on), `aiSidecar_charstatusDir` config.
- [x] **Bridge: full charstatus contract builder** (`_build_charstatus_payload`) — all 11 sections: schema_version/seq/snapshot_ts/server_time/freshness/in_game/last_seen_ts/bot_id + identity + vitals (hp/sp/weight/ratios/dead/sitting/status_effects) + position (map/x/y/direction/move_dest/route_failure/stuck) + inventory (zeny/items/equipment) + stats (str/agi/vit/int/dex/luk + bonuses) + skills (list/cooldowns/points) + combat (ai_sequence/is_in_combat/target_id/name/hp_pct/nearby_monsters) + environment (map/is_town/is_field/time_of_day) + party + economy + ai (state/queue/death_count/respawn/reconnect/npc_dialog) + telemetry.
- [x] **Bridge: status effects (SC) + cooldowns extraction** — from `$char->{statuses}`; `_DELAY` suffix → cooldowns, else status_effects; remaining-seconds computed.
- [x] **Bridge: stats extraction** — str/agi/vit/int/dex/luk + bonuses from OpenKore core.
- [x] **Bridge: current attack target** — from `AI::args(0)->{attackID}` + `$monsters{$id}` (name + hp_pct).
- [x] **Bridge: environment** — is_town/is_field + time_of_day (server time).
- [x] **Bridge: raw snapshot enrichment** — status_effects/cooldowns/stats/target added to `$raw` so the POSTed snapshot carries them too.
- [x] **Bridge: UTF-8 sanitizer** — `_sanitize_utf8_deep`/`_sanitize_utf8_scalar` (CP949/CP1252 → UTF-8) applied before JSON encode (was corrupting the file).
- [x] **Bridge: char_id/account_id unpack** — via `_actor_id_from_any` (was raw packed binary).
- [x] **Sidecar: CharStatusReader** (`runtime/charstatus.py`) — reads durable file, mtime-cached, stale-guard (max_age_s), per-bot; path = `<root>/data/charstatus/`.
- [x] **Sidecar: RuntimeState.charstatus_reader** wired in `create_runtime`.
- [x] **Sidecar: GET /v1/fleet/charstatus/{bot_id}** — prefers durable charstatus.json, falls back to snapshot_cache.
- [v] **Verify live** — charstatus.json written atomically (seq monotonic), valid UTF-8, full contract (char_id 2000011, vitals/stats/combat/env/inventory), endpoint returns `source: charstatus.json`.

### ⏳ PENDING
- [ ] **Wire brains** — Conscious (LLM prompt injection), Subconscious (ML state), Reflex (rules) consume charstatus fields (status_effects/cooldowns/stats/target).
- [ ] **Config** — add `aiSidecar_charstatusEnabled`/`aiSidecar_charstatusDir` defaults to bridge config.
- [ ] **Tests** — add unit tests for CharStatusReader + bridge contract builder.
- [ ] **Commit + push** — after live verify.

---

## Adversarial Additions (from design)
1. Atomic write race → temp+rename + monotonic seq ✅
2. Multi-bot collision → per-bot file ✅
3. Stale-data blindspot → freshness/in_game/last_seen_ts + reader max_age_s ✅
4. Missing status effects (SC) → included ✅
5. Cooldowns → included ✅
6. Target + nearby mobs → included ✅
7. Route/stuck → route_failure_count/stuck_detected ✅
8. Server time vs local → server_time + time_of_day ✅
9. Schema contract + validation → schema_version ✅
10. Read-only for brains → documented, endpoint is GET ✅
11. Unwired data → every field wired to a consumer (pending brain wiring)
12. Data freshness budget → snapshot tick already bounded; reader max_age_s
13. Derived vs authoritative → freshness/in_game authoritative
14. PII/security → loopback-only (bridge→sidecar 127.0.0.1)
15. Backpressure → bridge writes async, sidecar reads latest
