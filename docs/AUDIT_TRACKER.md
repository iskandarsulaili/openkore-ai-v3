# AUDIT_TRACKER.md — Completeness & Live-Readiness Audit

> Tracks every finding, fix, and verification batch. The goal is FULL completeness —
> zero stub/placeholder/pending/todo/fixme/dormant/incomplete — for live production.
> Do NOT trim features; implement what's incomplete. Update after each batch.

## Rules
- A batch is "done" only when: implemented, wired, verified (test + live where possible),
  and the tracker is updated in the same commit.
- "Dead" code is investigated before dismissing: it may be an incomplete-but-needed impl.
- Every fix must remain server-agnostic (discovered knowledge + reasoning, not per-server literals).

## Batch tracker

| Batch | Finding | Status | Verification |
|---|---|---|---|
| B1 | mon_control.txt duplicate-append dedup gap | ✅ done | bridge `_append_mon_control_dedup` helper, both writers use it; live mon_control files deduped (each profile 559→8 lines). `make test` 1168/1168, bridge syntax OK |
| B2 | supervisor sidecar-down gap (60s tick) | ✅ done | `fleet_supervisor.sh` self-heal tick 60s→15s to shrink the sidecar-down window; bash -n OK, service restarted |
| B3 | bot_id_canonicalized noise / actions latency throttle (deep-incomplete?) | ✅ done | Investigated at source: BOTH are by-design diagnostics, not incomplete impls. `actions_next_latency_budget_exceeded` logs poll latency but does NOT block real actions (delivery proceeds when first is not None). `bot_id_canonicalized` is correct id reconciliation on register (stable canonical id, no fragmentation). Verified non-issues. |
| B4 | ReinforcementLearner select_action not driving decisions | ✅ done | Added gated `behavior_override(state, min_experiences)` to the learner (returns action only when >=100 exp AND greedy, else None) + wired it into pdca_loop to enqueue a mapped command (`farm`→attackAuto 3, `rest`→sit, etc.) as a strategic suggestion. No-op until trained (fresh learner = 1 exp → None). Verified: compile OK, 24 pdca/reinforcement/subconscious tests + 11 cold-start/save-point pass. |
| B5 | (deep) any dormant/incomplete module reachable from the live path | ⬜ pending | — |

## Completed fixes (this session, before this tracker)

- **Snapshot-dropping 422** (`40295e8b2`): bridge weight float (370.6) vs int schema →
  `int_from_float` → dropped every snapshot. Fixed `Vitals`+`InventoryDigest` weight→float.
  Verified: 0 validation failures after restart.
- **Validation-logging visibility** (`40295e8b2`): 422 path+error+body now in log message.
- **RL-state gitignore** (`40295e8b2`): subconscious learned state not committed.
