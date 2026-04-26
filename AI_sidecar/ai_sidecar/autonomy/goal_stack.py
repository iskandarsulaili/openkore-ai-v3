from __future__ import annotations

from dataclasses import dataclass

from ai_sidecar.contracts.autonomy import GoalCategory, GoalDirective, GoalStackState, SituationalAssessment


@dataclass(slots=True)
class GoalStackComputation:
    goal_stack: list[GoalDirective]
    selected_goal: GoalDirective


def compute_goal_stack(*, assessment: SituationalAssessment, horizon: str) -> GoalStackComputation:
    map_name = str(assessment.map_name or "unknown")
    progression = assessment.progression_recommendation if isinstance(assessment.progression_recommendation, dict) else {}
    job_advancement = assessment.job_advancement if isinstance(assessment.job_advancement, dict) else {}
    opportunistic = assessment.opportunistic_upgrades if isinstance(assessment.opportunistic_upgrades, dict) else {}

    points_pending = bool(assessment.skill_points > 0 or assessment.stat_points > 0)
    playbook_supported = bool(job_advancement.get("supported"))
    playbook_ready = bool(job_advancement.get("ready"))
    route_id = str(job_advancement.get("route_id") or "")
    target_job = str(job_advancement.get("target_job") or "")
    missing_requirements = [str(item) for item in job_advancement.get("missing_requirements", []) if str(item).strip()]
    unsupported_notes = [str(item) for item in job_advancement.get("notes", []) if str(item).strip()]

    leveling_objective_template = str(
        progression.get("objective_template") or "continue deterministic leveling progression safely"
    )
    leveling_target_maps = [str(item) for item in progression.get("target_maps", []) if str(item).strip()]

    survival_active = bool(
        assessment.is_dead
        or assessment.is_disconnected
        or assessment.hp_ratio <= 0.35
        or assessment.danger_score >= 0.75
        or assessment.death_risk_score >= 0.75
        or (assessment.reconnect_age_s is not None and assessment.reconnect_age_s >= 20.0)
    )
    job_advancement_active = bool(points_pending or (playbook_supported and playbook_ready))
    opportunistic_active = bool(opportunistic.get("actionable"))
    leveling_active = True

    top_opportunity = {}
    opportunities = opportunistic.get("opportunities") if isinstance(opportunistic.get("opportunities"), list) else []
    if opportunities and isinstance(opportunities[0], dict):
        top_opportunity = opportunities[0]
    non_actionable_reasons = [
        str(item)
        for item in opportunistic.get("non_actionable_reasons", [])
        if str(item).strip()
    ]
    execution_hints = [
        dict(item)
        for item in opportunistic.get("execution_hints", [])
        if isinstance(item, dict)
    ]

    if playbook_supported and playbook_ready and route_id and target_job:
        job_advancement_objective = (
            f"execute curated job-change playbook {route_id} from {map_name} toward {target_job}"
        )
        job_advancement_rationale = "deterministic_priority:job_advancement;knowledge_backed:playbook_ready"
        job_advancement_blockers: list[str] = []
    elif points_pending:
        job_advancement_objective = f"allocate pending progression points safely on {map_name}"
        job_advancement_rationale = "deterministic_priority:job_advancement;knowledge_backed:points_pending"
        job_advancement_blockers = []
    elif playbook_supported and route_id and target_job:
        job_advancement_objective = (
            f"prepare requirements for curated route {route_id} toward {target_job} from {map_name}"
        )
        job_advancement_rationale = "deterministic_priority:job_advancement;knowledge_backed:requirements_pending"
        job_advancement_blockers = list(missing_requirements)
    else:
        job_advancement_objective = f"job advancement route unsupported for current class on {map_name}"
        job_advancement_rationale = "deterministic_priority:job_advancement;knowledge_backed:unsupported_route"
        job_advancement_blockers = list(unsupported_notes)

    if leveling_target_maps:
        leveling_objective = f"{leveling_objective_template} near {','.join(leveling_target_maps[:2])}"
    else:
        leveling_objective = f"{leveling_objective_template} on {map_name}"

    if opportunistic_active and top_opportunity:
        candidate_name = str(top_opportunity.get("candidate_item_name") or top_opportunity.get("candidate_item_id") or "upgrade")
        slot_name = str(top_opportunity.get("slot") or "equipment")
        domain_name = str(top_opportunity.get("domain") or "opportunistic_upgrades")
        score_delta = int(top_opportunity.get("score_delta") or 0)
        buy_price = int(top_opportunity.get("buy_price") or 0)
        opportunistic_objective = (
            f"execute curated opportunistic {domain_name} {slot_name} upgrade to {candidate_name} "
            f"(score_delta={score_delta}, buy_price={buy_price}) from {map_name}"
        )
        opportunistic_rationale = "deterministic_priority:opportunistic_upgrades;knowledge_backed:stage4_actionable"
        opportunistic_blockers: list[str] = []
    else:
        opportunistic_objective = f"hold opportunistic upgrade posture on {map_name} until deterministic evidence is complete"
        opportunistic_rationale = "deterministic_priority:opportunistic_upgrades;knowledge_backed:stage4_non_actionable"
        opportunistic_blockers = non_actionable_reasons[:8]

    goals: list[GoalDirective] = [
        GoalDirective(
            goal_key=GoalCategory.survival,
            priority_rank=1,
            active=survival_active,
            objective=f"stabilize survival posture safely on {map_name}",
            rationale="deterministic_priority:survival",
            blockers=[],
            metadata={
                "horizon": horizon,
                "hp_ratio": assessment.hp_ratio,
                "danger_score": assessment.danger_score,
                "death_risk_score": assessment.death_risk_score,
            },
        ),
        GoalDirective(
            goal_key=GoalCategory.job_advancement,
            priority_rank=2,
            active=job_advancement_active,
            objective=job_advancement_objective,
            rationale=job_advancement_rationale,
            blockers=job_advancement_blockers,
            metadata={
                "horizon": horizon,
                "skill_points": assessment.skill_points,
                "stat_points": assessment.stat_points,
                "job_exp_ratio": assessment.job_exp_ratio,
                "active_quest_count": assessment.active_quest_count,
                "playbook_supported": playbook_supported,
                "playbook_ready": playbook_ready,
                "route_id": route_id,
                "target_job": target_job,
                "missing_requirements": missing_requirements,
                "job_advancement": dict(job_advancement),
            },
        ),
        GoalDirective(
            goal_key=GoalCategory.opportunistic_upgrades,
            priority_rank=3,
            active=opportunistic_active,
            objective=opportunistic_objective,
            rationale=opportunistic_rationale,
            blockers=opportunistic_blockers,
            metadata={
                "horizon": horizon,
                "knowledge_loaded": bool(opportunistic.get("knowledge_loaded")),
                "supported": bool(opportunistic.get("supported")),
                "status": str(opportunistic.get("status") or "unknown"),
                "actionable": bool(opportunistic.get("actionable")),
                "known_rule_ids": [
                    str(item) for item in opportunistic.get("known_rule_ids", []) if str(item).strip()
                ],
                "overweight_ratio": assessment.overweight_ratio,
                "vendor_exposure": assessment.vendor_exposure,
                "recommended_domain": str(top_opportunity.get("domain") or ""),
                "recommended_opportunity": dict(top_opportunity) if isinstance(top_opportunity, dict) else {},
                "execution_hints": execution_hints,
                "non_actionable_reasons": non_actionable_reasons,
            },
        ),
        GoalDirective(
            goal_key=GoalCategory.leveling,
            priority_rank=4,
            active=leveling_active,
            objective=leveling_objective,
            rationale="deterministic_priority:leveling;knowledge_backed:progression_profiles",
            blockers=[],
            metadata={
                "horizon": horizon,
                "base_level": assessment.base_level,
                "job_level": assessment.job_level,
                "base_exp_ratio": assessment.base_exp_ratio,
                "job_exp_ratio": assessment.job_exp_ratio,
                "progression_recommendation": dict(progression),
            },
        ),
    ]

    selected = next((item for item in goals if item.active), goals[0])
    return GoalStackComputation(goal_stack=goals, selected_goal=selected)


def summarize_goal_stack(*, state: GoalStackState) -> dict[str, object]:
    return {
        "decision_version": state.decision_version,
        "horizon": state.horizon,
        "selected_goal": state.selected_goal.goal_key.value,
        "selected_objective": state.selected_goal.objective,
        "stack": [
            {
                "goal_key": item.goal_key.value,
                "priority_rank": item.priority_rank,
                "active": item.active,
                "objective": item.objective,
            }
            for item in state.goal_stack
        ],
    }
