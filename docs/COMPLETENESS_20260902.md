# COMPLETENESS CHECKLIST — openkore-ai-v3 map-login + stat-allocation 2026-09-02

**Goal:** Fix the map-login reconnect loop (bot stuck, 0x006A reject), correct stat
allocation, verify 2-1 job change completes, prove live.

## Round A — Map-login reconnect loop (0x006A) — DONE
- [x] DIAGNOSED: bot in reconnect loop — map-server reject 0x006A (error 1/2, alternates)
- [x] ROOT CAUSE: auto-adapt fallback branches (19/23/26) used `pack('V',$accountID)` on a
      RAW 4-byte binary string -> coerced to 0 -> account/char/session=0 -> guaranteed
      0x006A rejection -> infinite reconnect. Only the layout branch did the correct
      `unpack('V')` roundtrip.
- [x] FIXED: numeric-coerced (unpack('V')) in ALL branches. Verified hex: acct=2000011
      char=90071254 sess=4086626567, no zeros.
- [x] PROVEN: bot logs in + stays (1 reconnect, 0 rejections, 12+ min uptime, farming).

## Round B — Stat allocation — DONE
- [x] TRACED: `stat_add dex` on a Mage was NOT a hardcoded bug — the bot is actually
      **Novice (class=0)**, so DEX-first is CORRECT for Novice.
- [x] ROOT CAUSE of "156 points unspent": bridge rewrites `stat_add <stat>` -> `st add <stat>`
      but `_command_allowed` checks the REWRITTEN root `st` against the allowlist, which had
      `stat_add` but NOT `st` -> every stat command `policy_rejected` -> points never spent.
- [x] FIXED: added `aiSidecarPolicy_allow_37 st` to the policy allowlist.
- [x] PROVEN: points 156->118->83, DEX 1->17->20, STR 1->10 (Novice DEX-first, correct).
- [x] ALSO: StatBreakpointPlanner returned None for 1st-class jobs (mage->wizard alias) —
      fixed so cold_start uses the INT-first Wizard build for a Mage.

## Round C — 2-1 job change (Mage job_lv 10 -> Wizard)
- [x] Trigger fires (first-class at job_lv>=10 routes to 2-1 class) — verified jc2_diag fired=True
- [x] NPC coords resolved DB-backed (gef_tower 106,35) — verified jc2_emit target='wizard'
- [x] FIXED: store job_change_target is the NOVICE dict, not the 2-1 target — now derives
      mage->wizard from JOB_2_1_CLASSES (was routing to stale merchant/mage-guild)
- [ ] LLM dialog responder completes the menu (bot was routing to gef_tower; needs to
      complete the Wizard guild dialog)
- [ ] Verified: job becomes Wizard (bot is Novice class=0, not yet Mage — the charstatus
      "Mage" was stale; actual class=0 Novice)

## Round D — Cleanup
- [x] Removed debugMapLogin diagnostic + jc_reached/jc2_diag/jc2_emit temp diagnostics
- [x] Committed: e5aa63f76 (maplogin+stat), policy allowlist st
- [ ] Final verify: bot stable + stats spending + job-change completes
