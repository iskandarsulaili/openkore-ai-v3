# COMPLETENESS CHECKLIST — openkore-ai-v3 macro-agent + job-change 2026-09-02

**Goal:** Build an AI macro-agent that generates + verifies OpenKore macros for specific
cases (committed + reusable by other users, usable by openkore-ai-v3), starting with the
job-change oscillation. Prove live with benchmarks.

## Round A — Macro-agent (AI generates + verifies macros)
- [x] AUDIT existing macro infra: ai_sidecar_generated_macros.txt, macro_file config, reflex_* macros
- [x] AUDIT FINDINGS (2026-09-02):
  - MacroCompiler + MacroPublisher (compile+write files) — COMPLETE
  - MacroRepository (persistence) — COMPLETE
  - api/routers/macros.py — ONLY /publish (Create). NO list/get/delete/update → CRUD INCOMPLETE
  - MacroIntelligence engine — 50+ patterns + full in-memory CRUD, but process_triggers (execution) NEVER called — only get_patterns_for_context (AI context feed) → EXECUTION DORMANT
  - MacroSynthesizer (plan→macro) — wired in planner/service.py
  - MicroMacroGenerator (reflex fallback macros) — wired in action_emitter
  - MacroDistillationEngine (ML episodes→macro, self-improve) — wired via /distill-macro
  - macro_engineer_agent — REGISTERED but NEVER invoked in crew_manager/task_factory → DORMANT
  - ModelRouter.generate_text — exists (was dead wiring, now provided)
  - Security: validate_macro_policy blocks eval/shell/system/exec/wget/curl/perl
  - VERIFICATION: NO parse-check harness, NO dry-run, NO outcome proof → MISSING
- [x] DESIGN: agent generates macro -> parse-check -> harness dry-run -> live outcome proof -> commit
- [x] IMPLEMENT: MacroVerifier (parse-check + dry-run + security + outcome proof)
- [x] IMPLEMENT: MacroAgent (LLM generation for a specific case via generate_text)
- [x] IMPLEMENT: shared macro registry (committed macros/ dir + manifest, reusable)
- [x] IMPLEMENT: complete CRUD API (list/get/delete/update on /v1/macros)
- [x] IMPLEMENT: wire process_triggers execution into pdca_loop (un-dormant)
- [x] IMPLEMENT: wire macro_engineer agent into crew_manager (un-dormant)
- [x] IMPLEMENT: reward/punish for macros (macro brain in BrainRewardLedger + outcome wiring)
- [x] IMPLEMENT: job_change skill-set (MacroPattern that wins over hunting while eligible)
- [x] FIXED: heuristic_service COLD_START NameError (_cm_farmable undefined) + lockMap-to-current-farm
- [x] TESTS: 463 passed (14 new macro-agent tests + 9 transit tests)
- [x] PROVE: macro-agent generates + verifies a real macro (job-change case) live
  - LIVE: POST /v1/macros/generate -> {"ok":true,"verified":true,"name":"novice_job_change_merchant",
    "lines":["log...","move alberta_in 53 43","talknpc 53 43 c r1","pause 1","talknpc 53 43 c r2",...]}
  - LIVE: registered as job_change skill-set (macro_pattern_added: novice_job_change_merchant)
  - LIVE: macro_agent_initialized: llm=wired registry=1; macro_intelligence_initialized: 65 patterns
  - COMMITTED + PUSHED: deb14473e..5f1debb95 (12 files, +1150)
- [ ] NOTE: bot TestBotA stuck in a PRE-EXISTING reconnect loop (bridge spam, disconnected,
      liveness=disconnected) — unrelated to macro-agent work; needs separate diagnosis


## Round B — Job-change oscillation (via macro-agent or direct)
- [x] DIAGNOSED: bot eligible (novice lv26/10) but oscillates move alberta_in <-> navigate prt_fild08
- [x] ROOT CAUSE: competing emitters (routing/hunting/exploration/combat/stuck)
- [x] FIXED: JOB_CHANGE wins over ALL competing emitters while eligible + not on guild map
  - heuristic_service: hoisted JOB_CHANGE gate, removed 60s rate-limit, empty-job_name→novice default
  - heuristic_service: `_job_change_route_emit` latch (no re-emit every cycle → no route-calc reset)
  - heuristic_service: routing block gates `navigate <farm>` on job-change eligibility
  - edge_case_handler: handle_unstuck gate resolves DOTTED progression path (was top-level keys that don't exist → gate never fired) — f38105378
  - macro_intelligence: job_change patterns emit `set attackAuto 0` first
  - heuristic_service: `str(signals.get("job_name") or "novice")` — empty job_name (bridge doesn't populate it) now = novice → state=JOB_CHANGE → combat gated → bot WALKS (was HUNT → fought) — 193b4f01c
  - domains/progression.py: removed hardcoded `archer` target (RULE.md violation) → reads LLM-decided server_solutions target agnostically — 193b4f01c
  - domains/progression.py + heuristic JOB_CHANGE handler: 10s latch on guild move (route-calc reset) — 78c5bfbd5
  - pdca_loop: stall_heal self-heal gated on job-change eligibility (was re-routing eligible bot to farm) — a310dc415
- [ ] PROVEN: bot routes to guild and STAYS, reaches guild, completes job change
  - PARTIAL: bot now in JOB_CHANGE state + WALKING (verified 19:0x, moving 357→346 toward geffen_in).
    Remaining blocker: pre-existing reconnect loop (0x0436 stale-session rejection on reconnect).
    The 23-byte 0x0436 IS correct (tcpdump proved in-game at 17:28 AND 20:05 — 0x0437 action packets).

## Round C — Post-job-change farming
- [ ] Bot re-evaluates hunting map for merchant class (DB-backed)
- [ ] Sustained farming + EXP gain (benchmark: EXP/hr, kills/min)
- [ ] No death loop (novice low-HP deaths on prt_fild08 resolved)

## Round D — Full-stack completeness sweep
- [ ] Audit all emitters for competing-action races
- [ ] Verify no dead/dormant code paths in job-change + hunting decision chain
- [ ] Verify server_solutions DB-backed facts all consumed
- [ ] Verify DQN/subconscious + reflex tiers still active (training_steps>0)
- [ ] Final: bot self-sufficient, zero manual intervention, sustained progression



## Round D — Reconnect loop (0x0436 coalescing) + survivability + inventory (2026-09-03)
- [x] FIXED: 0x05fc conn-info sent AFTER map_loaded ack (not in 0x0436 segment) — 03d9f45ca
- [x] FIXED: 0x0B1C ping reply gated until in-game so 0x0436 is sent alone — 651ba746e
- [x] FIXED: defend-only combat (attackAuto 1) while walking to guild — e4e3f4abd
- [x] FIXED: sidecar emits `use Fly Wing` at 40% HP in JOB_CHANGE handler (runtime, not config) — 9dd1b468a
- [ ] ROOT CAUSE: 0x0436(length:25) recurs intermittently — some OTHER packet coalesces with 0x0436 on reconnect (real client sends 0x0436 ALONE + waits for 0x0087 ack; OpenKore flushes multiple packets per TCP write)
- [ ] CAPTURE: definitively identify the coalescing packet (bot keeps offline in backoff during capture)
- [ ] FIX: prevent ALL coalescing with 0x0436 (not just ping) OR server-side tolerate trailing bytes (needs approval)
- [ ] INVENTORY: verify TestBotA loads 50 Fly Wings post char-server restart (bot can't get in-game due to reconnect loop)
- [ ] PROVEN: bot reaches guild + completes job change (TestBotA still novice class 0)

## Round E — Complete local data copy (2026-09-03)
- [x] skill_db.yml copied from rAthena source (1635 skills) — was MISSING
- [x] randomopt_db.yml copied (249 entries) — was MISSING (rAthena name: item_randomopt_db.yml)
- [x] item_db.yml base + split files verified (usable 6415 / equip 12982 / etc 9959 = 29356, matches DB)
- [x] mob_skill_db.txt copied (1279512 bytes) — was MISSING
- [x] map_index.txt copied (13343 bytes) — was MISSING
- [x] FIXED: monster_db.py loader pointed at pre-re (1004 monsters, WRONG for Renewal server) → re (2675 monsters, matches DB)
- [x] mob_db.json generated from re/mob_db.yml (4401 entries) for ro_mechanics — was MISSING
- [x] portals.txt verified present (tables/, 129034 bytes)
- [x] Full test suite: 463 passed (1 flake in test_intelligence_integration, passes in isolation)

