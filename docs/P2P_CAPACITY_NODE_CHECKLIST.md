# openkore-ai-v3 — RAW P2P Capacity-Node Integration Checklist (b6)

Goal: make openkore-ai-v3 a FULL RAW P2P capacity node — PLAYER/farm bot AND
peer-host map-server AND relay AND in-game mesh member — server-agnostic, zero
hardcoded server facts, zero human intervention, live-production ready.

Scope mirrors RAW's own transport (docs/RAW_P2P_INTEGRATION_PLAN.md §1):
- LOGIN/CHAR/MAP: TCP via playit (already works)
- Peer-host: run a map-server.exe that registers with char, claims empty maps
  (EVE model), serves them
- Relay: be a TURN/relay member of the mesh
- In-game mesh: WebRTC data-channel for position/state (0x035F/0x0361)

## BATCH B6-a — FOUNDATION (verify-before-modify)
- [ ] 1. Read RAW peer-host architecture: hosted_creds, host-creds mediator,
      E5 attestation, map takeover/claim, host_assignable_maps, standalone host
      reward, relay protocol, mesh handshake (rathena-AI-world docs + code).
- [ ] 2. Map RAW transport contract → what a Perl capacity node must speak:
      (a) map-server.exe (we already cross-build one — reuse it), (b) relay =
      coordinator WS/REST, (c) mesh = WebRTC.
- [ ] 3. Decide packaging: bot embeds the cross-compiled map-server.exe + DLL
      (like RAW client) vs pure-Perl. Likely: reuse RAW's single-file map-server.exe
      + spawn it, since RAW already solved attestation/host-creds.
- [x] 4. char-select retry (2a blocker): ALREADY IMPLEMENTED (Receive.pm:12888 retry
      loop + DirectConnection.pm:597 re-enter). Root cause of churn was a SERVER-side
      char-server binary going missing (status=203/EXEC crash-loop) -> login refused
      all clients ('no char-server online'). Rebuilt char-server (src/char make, 93MB,
      digest lockstep) + restarted rathena-char -> login+char+map reconnect.
- [x] 5. Pathfinding reopen-skip fix: the 2.25 CLOSED-reopen (cc7576bac) caused
      unbounded openList growth on dense maps -> pathStep block -> keepalive
      starvation -> 0x0081 disconnect churn. Consistent-heuristic A* never reopens;
      fixed to skip CLOSED (7cb9715c6) + dynamic openList realloc growth (d3b5dbfda).
      VERIFIED LIVE: bot in-game izlude, RSS 206MB flat 13+ min, 0 disconnects.

## BATCH B6-aw — SELF-AWARENESS (SOUL.md + MEMORY.md, Hermes memory pattern)
The conscious tier now has the Hermes-style curated in-context memory so it can
self-learn / self-heal / self-improve coherently across every reasoning call.
Reverse-engineered hermes-agent source first (tools/memory_tool.py MemoryStore,
system_prompt.py:827 volatile-parts injection) to match the real pattern.
- [x] SA-1. SOUL.md (a53c8c646): curated identity + decision doctrine injected
      verbatim into every conscious-tier LLM call (Hermes has no identity doc;
      this is OUR addition).
- [x] SA-2. MEMORY.md (a53c8c646): LLM-curated durable lessons (Hermes memory-tool
      contract — the agent decides what to remember, NOT a DB dump). Char-bounded
      (100k), injected verbatim each call. Gitignored as runtime-learned state.
- [x] SA-3. Injection wired into BOTH LLM paths: LLMManager.complete/complete_json
      + model_router.generate_with_fallback (every reasoning call). Proven via
      /tmp/test_sa_direct.py PASS.
- [x] SA-4. API: /v1/self/{status,lesson,soul,hub} — conscious LLM writes lessons,
      fleet observes pool. LIVE: status soul=2953, hub round-trip verified.
- [x] SA-5. LessonsHub (4e2229248): SQLite central sink shared by all fleet bots —
      the "central sink now" (push/pull round-trip proven, boot-time cross-bot
      merge). Remote RAW-style HTTP sink (memory_sink_endpoint) optional.
- [x] SA-6. P2P knowledge mesh (ad39cfe09): was DORMANT (buried in a fragile
      pdca strategic-init block wrapped in except:pass; an earlier unguarded
      service threw and aborted before P2P init, so no node ever started).
      Now GUARANTEED-init in create_runtime: P2PKnowledgeNode + P2PNetworkManager,
      wired experience_db/npc_discovery/server_adaptation, HTTP server started,
      peers connected. VERIFIED LIVE: listener on 18428, health
      {"status":"ok","bot_id":"sidecar:default"}. Config p2p_bot_id/p2p_listen_port.
      This is the BOT-TO-BOT LEARNING channel — NOT the launcher's transport mesh.

## DESIGN RULE — COMPLEMENT, NEVER CONFLICT (founder directive 2026-08-26)
openkore-ai-v3 is delivered as part of the PlayRAW-launcher-downloaded client
bundle. Its "bot serves maps" capacity must COMPLEMENT the launcher's existing
P2P stack (peer-host map-server.exe, P2P relay, in-game WebRTC mesh) — it must
NOT conflict with or duplicate them:
- The launcher ALREADY spawns the peer-host map-server.exe (single-file, E5
  attestation, host-creds, seed-stats reward). The bot must REUSE that same
  binary as its capacity process — never ship a second/competing host.
- Single-writer per map is sacred (EVE model): central owns all, a host claims
  ONLY empty maps via the char JIT assigner. The bot host must observe the same
  assigner rule so it never fights the launcher/another host for a map.
- Relay membership + in-game mesh (0x035F/0x0361) are the LAUNCHER's DLL's job.
  The bot's P2P knowledge mesh (p2p_knowledge.py gossip) is a SEPARATE
  bot-to-bot learning channel (experiences/hunting-zones/prices/failures), NOT
  the transport mesh — it must not try to also carry 0x035F position/state.
- Distribution: the capacity host + id.conf/token are bundled into the
  launcher-downloaded client tree (client/peer-host/ or client/<bot>/); the
  manifest pins + self-update already cover the host binary.

## BATCH STATUS-b — PEER-HOST (bot serves maps)
- [ ] 6. Wire bot → host-creds (GET /ads/host-creds) + E5 attestation (maplogin
      with host session creds) → char register → claim empty maps (host_assignable).
- [ ] 7. Spawn + supervise the map-server.exe capacity process (lifecycle: boot,
      heartbeat, host-secs, kill on quit).
- [ ] 8. Host reward telemetry (seed-stats / game_host_seconds).

## BATCH C — RELAY
- [ ] 9. Register bot as relay member (coordinator), honest capacity, TTL keepalive.

## BATCH D — IN-GAME MESH
- [ ] 10. WebRTC data channel in Perl (libdatachannel bindings) for 0x035F/0x0361
      position/state to peers. (LARGE — evaluate feasibility before build.)

## BATCH E — VERIFY
- [ ] 11. Two-node E2E (bot + RAW client peer), EXP growth, host serves a map,
      relay relays, mesh passes 0x035F. Update checklist + commit per batch.
