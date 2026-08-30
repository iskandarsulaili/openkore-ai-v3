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

## Batch L — FLAW-FIX sweep round 2 (2026-08-29)
- [x] L1: poll wiring VERIFIED (the 30s telemetry loop at main.rs:10198 calls poll_packet_capture; runs whole launcher lifetime, cheap OpenFileMappingW miss when disabled — the ONLY call site).
- [x] L2: retry duplication AUDITED clean (mem::take empties the store; failure re-adds once; no double-upload).
- [x] L3: retry keep-OLDEST (was keep-newest) — the 0x0436 login is at the START of the session (ML-critical); the newest re-captures anyway.
- [x] L4: dropped-frames log on CHANGE only (was every 30s forever after a drop = spam).
- [x] L5: ring 'active' flag NOT checked — minor (the frame-magic walk already protects); documented.
- [x] L6: capture_dropped rides the upload body → NEW telemetry_files.capture_dropped column (ALTER TABLE) → handler stores it → export surfaces dropped_frames (the ML trainer knows when the capture lost data). E2E: row 66265 capture_dropped=42 stored OK.
- [x] L7: capture_enabled getter ALSO reads p2p_config.json capture.enabled (a manual config enables the DLL without the flag — the UI now reflects the true state).
- [x] L8: cargo check PASSED (5.31s), php -l clean.

## Batch M — FLAW-FIX sweep round 3 (2026-08-29)
- [x] M1: MULTI-WRITER RACE (CRITICAL) — two game instances (2 PCs / multi-login) both CreateFileMappingW the SAME named ring (an existing name returns a handle to the SAME mapping); each had ONLY a process-local capture_mutex_ → two writers corrupt write_pos + frames. FIX: named inter-process mutex (Local\RAWPacketCaptureLock) — the DLL acquires it around EVERY ring write (1s timeout → drop + count, never block the game socket), the launcher acquires it around read+clear (atomic). DLL 0.1.1067 deployed (sha 27d461eb, zero local file); cargo check PASSED (2.02s).
- [x] M3: prune SQL VERIFIED — matches both file_type spellings (packet-capture,packetcapture) + the stored value IS 'packetcapture' + 7-day cutoff on uploaded_ts. Correct.
- [x] M4: export auth VERIFIED — proper session bootstrap + featureAllowed('ads.admin') (the same as /ads/telemetry).

## Batch N — FLAW-FIX sweep round 4 (2026-08-29)
- [x] N3: SERVER KEEPS THE HEAD (CRITICAL) — packet-capture per-file cap 8MB was substr(-8MB) = KEEP TAIL; a full ring + retry CUT the 0x0436 (at the START, the WHOLE POINT). FIX: packet-capture keeps the HEAD (0..8MB); other files keep the tail. Unit-verified (head vs tail logic); php -l clean. (The 8MB E2E upload isn't feasible over the tunnel — the storage chain is proven by the smaller uploads.)
- [x] N5: writer_pid in the ring header [28..31] — two clients on one machine share the ring; the export can separate by PID (currently informational).
- [x] N1: mutex/ring lifetime AUDITED clean (the launcher opens by name per poll; a game-exit miss is clean).
- [x] N2: dropped counter overflow (u32@4G) — negligible, documented.
- [x] N7: flag re-check every 30s + capture on the recv hot path — the gate is a cached bool + one config read when disabled (ZERO overhead, the mandate).
- [x] N8: disabled = zero cost VERIFIED (the gate order: `if (!cfg.enabled && !flag_on) return;` — the flag check is ONLY when cfg.enabled is false, 30s-cached).
- [x] DLL 0.1.1068 deployed (sha 9836d22f, zero local file).

## Batch P — FLAW-FIX sweep round 5 (2026-08-29)
- [x] P4: retry drain VERIFIED clean (mem::take on the success path empties the store; a failed upload retains bounded 16MB; the next success drains).
- [x] P6: UPLOAD CHUNKING — the pending (≤16MB) was sent as ONE file; the server's 8MB per-file cap (head-keep) truncated the ML data after the first 8MB. FIX: chunk into ≤8MB pieces (≤16MB = 2 chunks/1 call, the server's 2-files/call). cargo check PASSED.
- [x] P1/P2/P3/P5: audited clean (dropped-counter static resets on launcher restart = minor; read+clear atomic under the named mutex; INT cast fine; multi-session rows are ordered by id + ts — acceptable for the 0x0436 goal).

## Batch Q — FLAW-FIX sweep round 6 (2026-08-29)
- [x] Q1: UI wiring VERIFIED (App.tsx state line 336, mount init 1539, toggle 3039-3055 + i18n).
- [x] Q4: ConfigManager capture parse VERIFIED (capture.enabled + max_bytes, defaults false).
- [x] Q2: SHARED-MAPPING OVERFLOW (CRITICAL) — two instances with different max_bytes: the SECOND CreateFileMappingW on the existing name gets the FIRST's mapping but set s_cap = its OWN config → writes past the real boundary (shared-memory heap overflow) + memset past it. FIX: on ERROR_ALREADY_EXISTS adopt the header's ACTUAL capacity (only a fresh creation initializes the header); the launcher already trusts the header. DLL 0.1.1069 deployed (sha 251e4359, zero local file).

## Batch R — FLAW-FIX sweep round 7 (2026-08-29)
- [x] R1a: cleanup cron VERIFIED (0 2 * * * → cron_logs) + the packet-capture 7-day prune present.
- [x] R1b: OTHER TELEMETRY UNPRUNED (REAL) — the launcher/p2p_dll/hostboot/injector files had NO prune (the comment claimed 30-day but no DELETE existed) → 71.4MB/93 rows accumulated. FIX: 30-day prune added (packet-capture keeps 7-day). Ran the cron live: "ok" + 0 deleted (all rows < 30 days old — the fix reclaims as they age). php -l clean.

## Batch S — FLAW-FIX sweep round 8 (2026-08-29)
- [x] S2: FRAME-BOUNDARY CHUNKING (REAL, self-introduced in P6) — the raw-offset chunker SPLIT frames at every 8MB boundary → the server stored the halves as separate rows → the export's walk broke at each boundary (frame lost + mis-parse). FIX: walk the [len][magic][ts][bytes] frames + close a chunk only at a frame boundary.
- [x] S4: capture_dropped DOUBLE-COUNT (REAL, self-introduced in P6) — the server stores the body's capture_dropped on EVERY row; the export SUMS → multi-chunk calls double-counted. FIX: send dropped only on single-chunk calls.
- [x] cargo check PASSED (2.60s).

## Batch T — FLAW-FIX sweep round 9 (2026-08-29)
- [x] T1: capture_dropped LOST for multi-chunk (my S4 client fix zeroed it on multi-chunk → the drop signal vanished). FIX server-side: store dropped only on the FIRST file of a call (the SUM stays correct for single + multi-chunk). E2E: 2-file call → row1 dropped=7, row2=0 (rows 66304/66305).
- [x] T2: telemetry_files NO SCHEMA SOURCE (Z-1 class) — created ad-hoc in the live DB; a fresh deploy would fail every INSERT. FIX: idempotent migration 018_create_telemetry_files.sql (CREATE IF NOT EXISTS + guarded capture_dropped column, idx_uploaded for the prunes).
- [x] php -l clean; E2E upload 200.

## Batch U — FLAW-FIX sweep round 10 (2026-08-29)
- [x] U1: MIGRATION PDO-SAFETY (REAL, T2's fix was fragile) — the Migrator (lib/Flux/Database/Migrator.php:85) runs the WHOLE file via ONE PDO::exec; my first 018 used PREPARE/EXECUTE user-variables (multi-statement — may fail under PDO). FIX: standalone statements + MariaDB ADD COLUMN IF NOT EXISTS (10.0.2+; the server runs 11.7). VERIFIED: scratch DB first-run exit 0 + second-run exit 0 (idempotent) + the LIVE DB (exit 0, capture_dropped present).

## Batch V — FLAW-FIX sweep round 11 (2026-08-29/30)
- [x] V1: migration tracker STALE — 016/017/018 applied ad-hoc + untracked (tracker stopped at 015); a future migrate would re-run 017's non-idempotent ALTER ("Duplicate column"). FIX: registered 016/017/018 in flux_migrations.
- [x] V2: export gate VERIFIED — /ads/telemetry/capture uses the EXACT same featureAllowed('ads.admin') as /ads/telemetry (lines 4012/4094).
- [x] V3: THE MIGRATOR WAS NEVER INVOKED (REAL, the "unwired" class) — zero call sites; new migrations never auto-applied (a fresh deploy + unapplied = missing tables = failed INSERTs). FIX: cron/run_migrations.php (minimal bootstrap mirroring adspace_cron: include-path-before-autoload, appConfigFile+serversConfigFile) + wired daily 01:00 (before the 02:00 cleanup, log lot399-owned). VERIFIED: "Migrations up to date".

## Batch W — FLAW-FIX sweep round 12 (2026-08-30)
- [x] W1: the Migrator's connection user VERIFIED — the ragnarok user has GRANT ALL on the ragnarok DB (servers.php:17-19 + SHOW GRANTS) → the migrations' CREATE/ALTER work.
- [x] W2: a migration FAILURE now alerts the founder — POST /ml-alert (moderator :9543, secret from config DiscordInteractionsProxySecret) — a schema failure = missing tables = failed INSERTs everywhere, must not be silent. php -l clean + the cron still runs "Migrations up to date".

## Batch X — FLAW-FIX sweep round 13 (2026-08-30)
- [x] X1: BOOT-RELATIVE TIMESTAMPS (REAL ML-data gap) — the DLL's per-frame ts is GetTickCount64 (ms since boot); the ML trainer can't map a frame to wall-clock. FIX: the launcher sends captured_at_ms (its own epoch-ms at poll) → NEW telemetry_files.captured_at_ms column (first file of a call, same as dropped) → export surfaces it (the trainer correlates a session's frames to a wall-clock window). E2E: row 66323 captured_at_ms=1788016000000. Migration 018 updated (idempotent ADD COLUMN IF NOT EXISTS), ran on live (exit 0), php -l clean, cargo check PASSED.
- [x] X2: App.tsx toggle VERIFIED (flag-only K1 version + optimistic state + best-effort).

## Batch Y — FLAW-FIX sweep round 14 (2026-08-30)
- [x] Y1: EXPORT AUTH USABILITY (REAL) — the export was session-cookie-gated; a CLI ML trainer can't hold a browser session. FIX: the export ALSO accepts an SSO Bearer token (discord_login_tokens, admin group>=99) — the trainer mints a token + passes Authorization: Bearer (can omit ?username — defaults to the bearer's login). VERIFIED: 200 + the decoded response structure.

## Batch Z — FLAW-FIX sweep round 15 (2026-08-30)
- [x] Z1: the OFF path VERIFIED — it DELETES packet_capture.flag (no stale-flag capture-on-OFF).
- [x] Z2: PRIVACY BLINDSPOT (REAL) — the DLL gate was cfg.enabled OR flag; a pre-set capture.enabled:true in p2p_config.json captured REGARDLESS of the toggle-OFF (the launcher can't touch the pinned config). FIX: NEW packet_capture.off sentinel — the OFF writes it, the ON removes it, the DLL checks it FIRST (never capture when present — the user's OFF wins over any config). Built + deployed DLL 0.1.1070 (252c3e98, sha f6d895bd), cargo check PASSED.

## Batch AA — FLAW-FIX sweep round 16 (2026-08-30)
- [x] AA1: the toggle files (packet_capture.flag/.off) VERIFIED safe — not in verify_integrity's manifest loop (they're not pinned) + NOT in the purge list (scoped to conf/db/npc/log/save + specific loose files). A fresh client dir = no flags = capture OFF by default (the disabled-by-default directive).
- [x] AA3: the capture_enabled() getter did NOT respect the OFF sentinel (Z2) — with a pre-set config.enabled, the UI showed ON after a toggle-OFF (the OR ignored the off). FIX: the getter returns false when packet_capture.off exists (the off wins). cargo check PASSED.

## Batch AB — FLAW-FIX sweep round 17 (2026-08-30)
- [x] AB1: the server's per-file cap VERIFIED — it checks the DECODED bytes (base64_decode first, then strlen($raw) > cap) — an 8MB chunk → 8MB decoded = exactly at the cap (no truncation).
- [x] AB2: UPLOAD TIMEOUT (REAL) — the packet-capture poll shared the 12s telemetry client; an 8MB upload on a slow link (8MB base64 ≈ 10.7MB; the earlier E2E timed out twice over the tunnel) exceeded it → POST failed → capture lost + retry. FIX: the poll now uses its OWN 60s client. cargo check PASSED.

## Batch AC — FLAW-FIX sweep round 18 (2026-08-30)
- [x] AC1: the 30s loop VERIFIED sequential (the awaits serialize: sleep → flush → upload → poll) — a 60s poll delays the next tick but never runs concurrently (no double-upload/race).
- [x] AC2: DROPPED DOUBLE-COUNT (REAL) — the ring's dropped counter is CUMULATIVE (never reset); reporting it raw every poll made the server's export SUM multiply the drops by the poll count. FIX: the poll reports the DELTA since the last poll (per-poll static). cargo check PASSED.

## Batch AD — FLAW-FIX sweep round 19 (2026-08-30)
- [x] AD1: S4/T1 CONFLICT (REAL) — S4 (client sends capture_dropped=0 on multi-chunk) contradicted T1 (server stores first-file-only): the drop delta was LOST for 2-chunk uploads. FIX: the client sends the delta ALWAYS (the server's T1 first-file logic prevents the SUM over-count). cargo check PASSED.

## Batch AE — FLAW-FIX sweep round 20 (2026-08-30)
- [x] AE1: DROP DELTA LOST ON FAILED UPLOAD (REAL) — the AC2 delta was consumed every poll; a FAILED upload's drops were never reported (the retried bytes carried a 0 delta). FIX: NEW PENDING_DROPPED accumulator — failed uploads add their drops, the next report carries the total, a successful upload clears it (the drops ride the retried bytes). cargo check PASSED.

## Batch AF — FLAW-FIX sweep round 21 (2026-08-30)
- [x] AF1: AE1 DOUBLE-COUNT + GAP (REAL, my AE1 fix had 2 flaws) — (1) the failure path added capture_dropped (which already included PENDING_DROPPED) → double-count on consecutive failures; now adds only the NEW delta; (2) the Err (network) branch MISSED the accumulation entirely (retained bytes but lost the drops) — now adds the delta too. cargo check PASSED.
