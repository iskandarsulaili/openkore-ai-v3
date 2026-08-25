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
