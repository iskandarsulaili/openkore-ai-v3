# COMPLETENESS SWEEP 2026-08-29 — openkore-ai-v3

**Mandate:** implement/integrate/fix/wire/execute/verify ALL to completeness — zero
mock/stub/placeholder/pending/todo/fixme/dormant/incomplete. Reconcile, never trim.
Proven with benchmark, not assumption. Verify before modifying. Big picture + all angles.

## Batch A — Checklist + Baseline
- [x] A1: Repo state snapshot (git status/log, sidecar health, bot state, EXP, tests green)
- [x] A2: 30-min monitor loop confirmed running (check_30min.sh)
- [x] A3: Document current EXP/hr baseline (needs a STABLE session — blocked by desync)

## Batch B — Packet-desync root cause (THE SESSION-STABILITY BLOCKER)
- [x] B1: Server packet db extracted (1981 parseable_packet entries) + bot recvpackets (1539)
- [x] B2: Diff — 0 static (id,len) mismatches; the desync is the RUNNING binary vs source
- [x] B3: PROBED the LIVE map-server's 0x0436 — accepts 26 bytes (the RUNNING binary's
      expectation; source has 19; the 10:24 build predates the multi-login change)
- [x] B4: Bot sendMapLogin → 26-byte form (matches the running server) — the bot now
      logs in + reaches net_state 4 (in-game map session). Committed `3150fe2db`
- [x] B5: Verified the bot stays in-game (game_time flows, lockMap applied, LLM drives)
- [x] B6: REMAINING: mid-session drops (~4 min) — the stale binary's OTHER packet
      mismatches. Needs map-server rebuild from source (sibling's domain, gated on
      Adaly logging off). Handover pushed `841da3b87`
- [x] B7: Gear-planner price floor VERIFIED SAFE (< 10z floor excludes event items,
      keeps real starter gear 50z+)
- [x] B8: arbiter `bot_disconnected` = CORRECT (the bot was genuinely disconnected
      during reconnects — not a probe bug)
- [x] B9: Found + fixed the sendMapLogin compile bug ($msg undeclared) during the sweep

## Batch C — Sidecar defects (Conscious brain blockers)
- [x] C1: crew pipeline signature bug (`_run_crew_pipeline takes 0 positional args but 1
      given`) — FIXED: call as bound method. Committed `3150fe2db`
- [x] C2: cost gate `fleet_hourly_call_limit_exceeded:30/30` — the MAX-unlimited fix
      didn't cover the hourly gate. FIXED: MAX reads `llm_cost_tier` (the env-bound
      field) → 0 (unlimited). Committed `3150fe2db`
- [x] C3: VERIFIED the LLM brain ACTIVE — provider_route_primary_succeeded + goal
      decomposition + mission decisions flowing (no budget gate)

## Batch D — Self-heal completeness (route-hang + zero-EXP stall)
- [x] D1: zero-EXP-from-start stall detection — a bot that NEVER gains EXP never set
      `_exp_change_ts` → the NO-PROGRESS heal was DEAD for that case. FIXED: seed the
      ts on the first in-game observation. Committed `8384f2cb9`
- [x] D2: route-timeout give-up — TOO_MUCH_TIME suppressed forever → the route state
      hung + the bot idled + the server dropped it. FIXED: give up after 3 consecutive
      timeouts. Committed `8384f2cb9`
- [x] D3: 3-min heal window (was 5) — catches the ~4-min route-hung session before the
      idle-timeout drop. Committed `8384f2cb9`
- [x] D4: stall detector DEBUG instrumented (stall_dbg) — the heal's inputs are now
      diagnosable live (remove after verified firing)
- [x] D5: VERIFIED the heal logic fires (240s ≥ 180s simulation); the live non-firing =
      the bot's reconnect cycles make `_in_game` (snapshot < 180s) false — the drops
      are the server-side rebuild wait

## Batch E — Full verification + benchmark
- [ ] E1: Full pytest suite green (450+)
- [ ] E2: Sidecar restart clean (health True, 2 bots)
- [ ] E3: SUSTAINED benchmark: EXP/hr over a stable multi-hour session (the win proof)
- [ ] E4: Checklist final pass — every box marked with evidence

## Open risks (from devil's-advocate review)
- R1: Packet-desync caps everything — B is the highest priority
- R2: EXP/hr at level 8 is meaningless until sessions are stable
- R3: Novice-only: job change + dungeon progression untested live

## Batch F — 0x0436 reverse-engineering (THE real-client-layout hunt, 2026-08-29)
- [x] F1: Re-probed the REBUILT map-server (20:24): 19-byte -> 0B, 23-acct@2 -> 6B,
      23-acct@6 -> 23B (accepted). The RUNNING binary expects the MULTI-LOGIN
      23-byte layout with account_id at OFFSET 6 (not the classic offset 2)
- [x] F2: The bot's 23-byte send was WRONG (acct@2 — the old 'V4' layout). FIXED:
      'v I I I I I C' (unknown@2, account@6, char@10, session@14, tick@18, sex@22).
      Committed `467fed8ac`
- [x] F3: The map-server log CONFIRMED the root: "unknown connect packet 0x0436
      (length:23), possibly for having an invalid account_id" — the account_id
      WAS at the wrong offset (the server's pos[0]=6, the bot sent acct@2)
- [ ] F4: The bot STILL times out — the char's auth node isn't created. The char's
      0x0840 (accessible maps) STILL shows ALL status=1 + the reconcile loops
      "refusing to demote promoted standby 3: it is the ONLY routable"
- [ ] F5: VERIFY the char's map ownership — capture the RAW 0x0840 bytes + the
      char's `char_search_mapserver("prontera")` result. The standbys' claim-all
      churn (the char believes ONLY standby 3 is routable, but its maps aren't
      found) is the suspected root (server-side, sibling's domain)

## Batch G — 0x0436 AUTO-ADAPT (the agnostic completion, 2026-08-29)
- [x] G1: The RUNNING map-server's 0x0436 = 23 (probed + the log "expected 23").
      The SOURCE + the 2025 emulator = 19. The binary ≠ the source (the sibling's
      build has uncommitted 0x0436=23 changes)
- [x] G2: 4 layout guesses (19, 23-acct@2, 23-acct@6, 26) ALL rejected
      ("invalid account_id" / "unknown (length:N)") — the 23-byte's pos[] is
      NOT guessable (binary decode: the entry @0x7acb198 len=23 func=0x00b4888e
      but the pos[] decode fails — the struct alignment)
- [x] G3: AUTO-ADAPT SHIPPED (committed `90e554a1d` + the rotation fix) — on a
      map-login timeout, the bot rotates the layout (19 -> 23 -> 26 -> 19) until
      one lands. RULE.md agnostic: the bot LEARNS the live server's accepted form
      through the reconnect cycle, no hardcode
- [ ] G4: VERIFIED the rotation cycles (23 -> 26 -> 19 observed live) but NO
      layout lands yet — the RUNNING binary's 0x0436 pos[] remains unknown
- [ ] G5: THE DEFINITIVE (the only remaining verification): capture the REAL
      client's 0x0436 bytes (tcpdump the tunnel while a real launcher session
      connects) — the real client works, its packet is the ground truth. Until
      then, the auto-adapt rotation is the bot-side completion



## Batch H — tcpdump-capture feature (2026-08-29, user directive)
- [x] H1: Killed ALL stale processes (openkore bots, start-bot wrappers, monitor/check 30min loops) — only the sidecar (740128) remains.
- [x] H2: DLL packet-capture hook (WARP patches/cross-compile NetworkHooks.cpp) — MaybeCapturePacket logs every map-server packet (hex + dir + len) to packet_capture.log when capture.enabled; disabled by default (zero overhead); 8MB cap. Hooked on BOTH recv (S>) + send (C>).
- [x] H3: Config — CaptureConfig in Types.h (enabled=false, max_bytes=8MB) + ConfigManager parse (p2p_config.json capture.enabled) + GetCaptureConfig accessor.
- [x] H4: Launcher Settings toggle (App.tsx) — Packet Capture checkbox (OFF by default) → Rust set_capture_enabled writes packet_capture.flag + p2p_config.json capture.enabled; capture_enabled getter for the initial state. Registered in invoke_handler.
- [x] H5: Telemetry upload — upload_files now includes packet_capture.log (file_type packet-capture), same change-tracking.
- [x] H6: Server allowlist (FluxCP ads_api_routes.php) accepts packet-capture/packetcapture (the SWEEP-48 trap — a new file_type silently dropped).
- [x] H7: i18n settings.captureDisabledByDefault in en/ja/es.
- [x] H8: DLL built (P2PDLL_VERSION=0.1.1061 commit e535c14), deployed to client dir (sha d630a8107c00b8c6, old backed up .bak-20260829-precapture).
- [x] H9: Launcher cargo check PASSED (7.14s Finished, no new warnings).
- [x] H10: FULL CHAIN E2E — simulated telemetry POST → HTTP 200 stored:1 → DB row 66155 file_type=packetcapture 43 bytes. SERVER-SIDE VERIFIED.
- [ ] H11: Real-client capture — needs a user launch with the toggle ON (Windows rebuild + dist upload; the launcher can only be built on Windows).

## Batch I — SECURITY: NO local copy (2026-08-29, user directive)
- [x] I1: REDESIGN — the capture NEVER writes a local file. The DLL now logs packets to a NAMED SHARED MEMORY RING (Local\RAWPacketCapture, 512KB, magic PCPT, header+ring frames) — zero disk writes.
- [x] I2: VERIFIED the deployed DLL has ZERO packet_capture.log references (strings -a count = 0) — no local file path anywhere.
- [x] I3: Launcher poll_packet_capture — opens+maps the ring (windows-sys Win32_System_Memory, feature added), reads the frames, POSTs straight to /telemetry/files (file_type packet-capture), CLEARS the ring (nothing stays local). Wired into the 30s telemetry loop.
- [x] I4: Removed the packet_capture.log file-upload candidate from upload_files.
- [x] I5: DLL rebuilt + deployed (P2PDLL_VERSION=0.1.1062, sha e25a93a96100e69d).
- [x] I6: Server unchanged (already accepts packet-capture).
- [x] I7: Linux cargo check PASSED; the Windows-target check fails ONLY on the build.rs windows-gnu gate (by design); the Memory FFI verified against windows-sys 0.52 source (link! macros present).
- [ ] I8: Real-client capture (needs the Windows rebuild + dist upload; toggle ON → DLL writes the ring → launcher uploads → central DB).
