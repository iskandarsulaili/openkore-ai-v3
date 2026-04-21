from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FleetConflictResolver:
    def resolve_constraints(self, *, constraints: dict[str, object]) -> dict[str, object]:
        avoid_items = constraints.get("avoid") if isinstance(constraints.get("avoid"), list) else []
        required_items = constraints.get("required") if isinstance(constraints.get("required"), list) else []
        source_items = constraints.get("sources") if isinstance(constraints.get("sources"), list) else []

        dedup_avoid = self._dedup_dict_list(avoid_items)
        dedup_required = self._dedup_dict_list(required_items)
        sources = sorted({str(item) for item in source_items if str(item)})

        return {
            "avoid": dedup_avoid,
            "required": dedup_required,
            "sources": sources,
            "policy": {
                "step_1_detect_conflict": True,
                "step_2_compare_priority_and_lease": True,
                "step_3_apply_doctrine": True,
                "step_4_emit_constraints": True,
                "step_5_rearbitrate_pending_strategic": True,
            },
        }

    def rearbitrate_action_metadata(self, *, action_metadata: dict[str, object], constraints: dict[str, object]) -> dict[str, object]:
        out = dict(action_metadata)
        avoid = constraints.get("avoid") if isinstance(constraints.get("avoid"), list) else []
        blocked_conflicts = {
            str(item.get("conflict_key") or item.get("keep") or "")
            for item in avoid
            if isinstance(item, dict)
        }
        current_conflict_key = str(out.get("conflict_key") or "")
        if current_conflict_key and current_conflict_key in blocked_conflicts:
            out["fleet_blocked"] = True
            out["fleet_block_reason"] = "conflict_constraint"
        else:
            out["fleet_blocked"] = False
            out["fleet_block_reason"] = ""
        return out

    def _dedup_dict_list(self, rows: list[object]) -> list[dict[str, object]]:
        seen: set[tuple[tuple[str, str], ...]] = set()
        result: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = tuple(sorted((str(k), str(v)) for k, v in row.items()))
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(dict(row))
        return result

