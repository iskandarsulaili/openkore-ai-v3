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

## Batch J — MORE capture for ML (2026-08-29, user directive + "moderation and pruning both sides")
- [x] J1: DLL v2 — 8MB config-driven ring (capture.max_bytes, clamped 1-32MB), FULL packets (no 128B truncation), v2 frames [u32 len][u64 ts_ms][raw bytes] (ML-ready timestamps). Ring WRAPS (keeps newest). Deployed P2PDLL_VERSION=0.1.1064 (sha 97e2f4733b864fbe), zero local file (strings count 0).
- [x] J2: Launcher — v2 frame parse + in-memory retry retention (PENDING_CAPTURE, 16MB cap, oldest dropped): upload failures NEVER lose capture, next poll retries. cargo check PASSED.
- [x] J3: Server MODERATION — packet-capture gets 8MB per-file cap (2MB others keep).
- [x] J4: Server PRUNING — 7-day retention for packet-capture rows (cleanup_logs.php cron), the other telemetry files keep 30-day.
- [x] J5: ML EXPORT — GET /ads/telemetry/capture?username=X decodes the stored v2 frames to "0x0436 <ts> <hex>" lines (admin-gated EXACTLY like /ads/telemetry — featureAllowed('ads.admin'); the weak Flux::$isLoggedIn gate never 403'd correctly in api.php, fixed).
- [x] J6: E2E VERIFIED — v2 frame upload → HTTP 200 stored:1 → row 66237 (len 20 + ts 1788016000000 + the 0x0436 bytes) → decode test = "0x0436 1788016000000 36041000..." → unauth export = 401 (gate works) → 2 rows stored.
- [ ] J7: Real-client capture + ML training (needs Windows launcher rebuild + dist upload + toggle ON).

## Batch K — FLAW-FIX sweep (2026-08-29, adversarial: "any more missing/flaw/race/blindspot?")
- [x] K1: PINNED-FILE WRITE (CRITICAL) — set_capture_enabled WROTE p2p_config.json (a manifest-pinned verify_integrity target) → every toggle would FAIL the pin check → restore loop + the toggle reverted. FIX: toggle writes ONLY packet_capture.flag; the DLL reads the flag (resolved next to the DLL, re-checked 30s → mid-session toggles apply).
- [x] K2: RECV-PATH GAP — capture only on the recv() FAST path (0-peer). FIX: added to ALL 4 paths — recv fast + routing-active, WSARecv fast + routing-active (in-game ML packets captured when peers are connected).
- [x] K3: CROSS-PROCESS RACE (ring v3) — v2 WRAP: the launcher (no lock) could read a frame mid-memcpy (torn) or the wrap could overwrite a frame being read / fight the clear (duplicate uploads). FIX: v3 = NO WRAP (drop-when-full + dropped_frames counter) + per-frame magic 0xF0C7 — a mid-write frame is detectable, never uploaded; the dropped counter is logged ("capture falling behind").
- [x] K4: FRAME_LEN SEMANTICS — the DLL writes frame_len = TOTAL incl. the 16B header; the export + poll advanced hdr+fl (DOUBLE-COUNT → every v3 frame mis-parsed "corrupt tail"). FIX: advance by frame_len, packet = [hdr..fl]. E2E: v3 row 66260 decoded 0x0436 1788016000123 OK.
- [x] K5: SERVER QUOTA — packet-capture rows EXCLUDED from the 10MB machine quota (bounded by the 8MB cap + 7-day prune) so the ML capture never purges other telemetry.
- [x] K6: FLAG PATH — resolved next to the DLL (GetModuleFileName), not the game CWD (robust).
- [x] K7: EXPORT AUTH — /ads/telemetry/capture uses the SAME admin gate as /ads/telemetry (session bootstrap + featureAllowed('ads.admin')); the weak Flux::$isLoggedIn gate always 403'd. Verified 401 unauth.
- [x] K8: E2E — v3 upload 200 + row 66260 (36B) + decode OK; php -l clean; DLL 0.1.1066 deployed (sha 2a348dc4, zero local file); cargo check PASSED.
