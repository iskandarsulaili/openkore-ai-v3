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
