# FOUNDER STEERS — openkore-ai-v3 (recorded 2026-08-24, authoritative)

This doc records every steer/directive the founder gave during the openkore-ai-v3
work. It is the single source of truth to prevent drifting or loss. If any later
instruction conflicts, the LATEST dated steer wins.

## A. RAW boundary (evolved)
- A1. (initial) Do NOT touch RAW — fix openkore-ai-v3 client-side only, agnostic with RAW.
- A2. (relaxed) Founder authorized touching the map-server: "Do whatever it takes to
  make everything work perfectly. Make sure NOTHING break." → map-server is in-scope
  IF changes are surgical + verified + nothing breaks. Everything else in RAW stays
  unmodified.
- A3. openkore-ai-v3 MUST fix/wire itself agnostic with RAW — never fix RAW to match the bot.

## B. Network / connectivity (hard)
- B1. ALWAYS use ipv6-raw-server.openkore-ai.com + respective playit.gg ports
  (login 46019, char 21544, map 53378). NEVER use any local port (verbatim, repeated).
- B2. Client-side only for bot fixes unless A2 grants map-server access.

## C. Multi-login / accounts (hard)
- C1. A player must be able to multi-login the SAME account with BOTH a real client
  (RAW client) AND openkore-ai-v3, on different chars. RAW already supports per-char
  multi-login (GID=char_id). openkore-ai-v3 must select a different char slot
  (sendCharLogin(config{char})) and coexist with the player's client session.
- C2. Do NOT use kicapmasin888/kicapmasin000 accounts for testing. Use one single
  test account (testbot99; chars TestBotA/B/C). Report any RAW-side issue.

## D. Scope of openkore-ai-v3 (completeness mandate)
- D1. Integrated peer host, p2p relay, AND in-game p2p into openkore-ai-v3 —
  as capable as the RAW map-server / RAW client.
- D2. IPv6 and UDP support — just as map-server can do / RAW client is capable of.
- D3. Plan the architecture carefully and implement/wire/verify them.
- D4. Player bot AND capacity node (peer host / relay / in-game mesh) — "Both and
  all of them without exception."
- D5. Zero mock/stub/placeholder/pending/todo/fixme/dormant/incomplete. Ready for
  live production release. Reconcile, NEVER trim/remove features.
- D6. Checklist-first: create/update checklist docs before/during/after each batch.

## E. Windows launcher integration (hard)
- E1. openkore-ai-v3 ready for Windows, delivered via: dist/, manifest entries,
  and a launcher option to configure + run openkore-ai-v3.
- E2. HASH/VERIFY EXEMPTION (verbatim): "For the windows openkore-ai-v3, do not
  strict on the hash/verify files/folder, because openkore-ai-v3 naturally will
  change/modify/update quite a lot of times. but yet, they will get the sidecar
  update." → Do NOT enforce strict integrity/hash on the openkore-ai-v3 folder;
  still deliver sidecar updates normally.
- E3. openkore-ai-v3 is a separate package/folder from the game client (it changes
  often; don't let it trip the client's integrity/restore logic).

## F. LLM key model for openkore-ai-v3 (verbatim steer)
- F1. In the launcher, a user may use the SAME LLM key they key in in Settings for
  ONLY their own openkore-ai-v3 AI bot.
- F2. Their LLM options are: (a) LLM key only for their own in-game AI NPC / AI
  agent usage, (b) shared LLM pool, and 3 new combos:
    1. self + openkore-ai-v3
    2. LLM pool + openkore-ai-v3
    3. openkore-ai-v3 only
  (or any combination the founder might have missed).
- F3. NO REWARDS for LLM used for openkore-ai-v3 (verbatim). LLM usage by
  openkore-ai-v3 must NOT earn reward/credits (unlike the reward economy's
  llm_pool source).

## G. Working directives (recurring founder mandates)
- G1. "Implement/integrate/fix/wire/execute/verify all as in doc to fully and max
  utilize each regardless severity or pre-existing." Completeness over everything.
- G2. "Verify first before making any modification, to prevent unwanted issue."
- G3. "When verifying, always see from bigger picture and all angles to prevent
  unexpected issue."
- G4. Solve issues with the best/latest pragmatic solution, from knowledge AND online.
- G5. Commit after each batch; push at reasonable stages (except sensitive/secrets).
- G6. Checklist-first: docs/ZERO_INTERVENTION_COMPLETENESS.md is the master tracker.

## H. Anti-loop / style
- H1. Founder catches tool-loops ("you are in loop"). After 2 identical tool calls
  with identical output, CHANGE STRATEGY — don't re-grep the same file.
- H2. Answer only when asked; plain language, honest limits. Verify against actual
  code/live endpoints, not memory/checklist claims.

## I. Log / telemetry / ML crowdsource (founder directive 2026-08-24)
- I1. openkore-ai-v3 MUST log/telemetry its runtime — designed to be easy to debug
  AND to eventually train our own ML model from crowdsourced data.
- I2. This especially (but not limited to) applies to AI bots running on user PCs
  on WINDOWS later.
- I3. Design implication: capture bot actions, decisions, outcomes, rewards,
  positions, packet/state snapshots into a structured, anonymous, uploadable log
  stream that can be aggregated server-side into a training dataset — mirroring
  the RAW mobs-ML collector pattern (ai_responses/reward-confidence, watermark,
  per-session). Zero PII; session-keyed; rate-limited; founder-reviewable.
- I4. Do NOT let telemetry hurt the bot (fire-and-forget async, non-blocking,
  bounded queue, offline-tolerant). This is a first-class feature, not a stub.

## J. OS agnostic (founder directive 2026-08-24)
- J1. ALL openkore-ai-v3 code must be OS agnostic (Linux dev host + Windows user
  PCs). No Windows-only / Linux-only assumptions in bot, sidecar, bridge, launcher
  integration, pathfinding, or telemetry.
- J2. Any Windows API shim (e.g. GetTickCount in the C A* pathfinder) must have a
  correct cross-platform equivalent — the Linux shim already returns real ms, but
  every such call must be verified cross-platform before relying on it.
- J3. Paths, process spawning, signals, socket/keepalive, file I/O, and telemetry
  upload must all use portable constructs.

## K. P2P crowdsource: self-healing/self-learning/self-improving across peers (founder 2026-08-24)
- K1. openkore-ai-v3 bots must crowdsource from EACH OTHER (other AI bot peers) —
  P2P self-healing, self-learning, self-improving. Not just telemetry-to-server.
- K2. Mechanism: bots exchange learned solutions / decision outcomes / route fixes /
  economy+combat lessons over the same P2P mesh, aggregate, and improve each other
  (a "bot swarm learning" layer over the RAW in-game P2P mesh).
- K3. Keep it anonymous + bounded + rate-limited + fail-safe: a peer's bad/untrusted
  lesson must never degrade the bot (weighted trust, majority/confidence aggregation,
  rollback to known-good). Mirrors RAW mobs-ML champion-gate pattern.
- K4. Do NOT let P2P crowdsourcing hurt the bot or the session (async, non-blocking,
  offline-tolerant, never a hard dependency).

## L. 4th-job end-game build (founder 2026-08-24)
- L1. openkore-ai-v3 bots must be specialized, job-agnostic, and target 4th-job in
  their end-game build — i.e. the AI can play ANY job (no hardcoded job literals)
  and its long-term progression/goal is a 4th-job end-game build.
- L2. Job choice/gear/skill/stat decisions come from the AI/LLM sidecar, never
  hardcoded job IDs or per-job item lists (mirrors RAW's agnostic directive).
- L3. Progression planner must know the 4th-job end-game target and route the char
  through 1st/2nd/3rd/4th (via real quests, per the RAW job-quest directive) toward
  a specialized 4th-job build.

## M. Launcher config model (founder 2026-08-24)
- M1. ALL core openkore-ai-v3 config is set AUTOMATICALLY based on the user's
  login in the launcher (account/char/server/ports/credentials — no manual
  config.txt editing).
- M2. Only a MINIMAL set of common/customizable options (most common + core
  only) is exposed as launcher Settings. Keep it minimal — do not surface the
  full config surface.
- M3. Update docs (this file + ZERO_INTERVENTION docs) to reflect the
  launcher-driven config model.


