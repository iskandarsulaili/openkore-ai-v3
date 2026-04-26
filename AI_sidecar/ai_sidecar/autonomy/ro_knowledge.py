from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any

logger = logging.getLogger(__name__)

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_AUTONOMY_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _AUTONOMY_DIR / "data"
_DEFAULT_TABLES_DIR = _AUTONOMY_DIR.parents[2] / "tables"


def _normalize_job_name(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = _NORMALIZE_RE.sub("_", text).strip("_")
    return text


def _parse_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_coordinates(value: str) -> tuple[int, int] | None:
    parts = [item for item in str(value or "").strip().split() if item]
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


@dataclass(slots=True)
class ProgressionBand:
    label: str
    min_base_level: int
    max_base_level: int
    recommended_focus: str
    target_maps: list[str]
    objective_template: str


@dataclass(slots=True)
class ProgressionProfile:
    key: str
    job_names: list[str]
    bands: list[ProgressionBand]


@dataclass(slots=True)
class JobBuildEntry:
    job_class: str
    build_name: str
    stat_priority: list[str]


@dataclass(slots=True)
class JobChangePlaybook:
    route_id: str
    route_priority: int
    current_job: str
    target_job: str
    required_base_level: int
    required_job_level: int
    steps: list[str]
    notes: list[str]


@dataclass(slots=True)
class OpportunisticUpgradeRule:
    rule_id: str
    domain: str
    evidence_mode: str
    job_names: list[str]
    slot: str
    current_item_ids: list[str]
    candidate_item_id: str
    candidate_item_name: str
    required_base_level: int
    score_current: int
    score_candidate: int
    max_price_zeny: int
    execution_mode: str
    direct_command: str
    direct_conflict_key: str
    direct_preconditions: list[str]
    signal_requirements: dict[str, object]
    control_plan: dict[str, object]
    macro_bundle: dict[str, object]
    notes: list[str]


@dataclass(slots=True)
class ROKnowledgeBundle:
    version: str
    profile_lookup: dict[str, ProgressionProfile]
    default_profile: ProgressionProfile | None
    playbooks_by_job: dict[str, list[JobChangePlaybook]]
    job_change_locations: dict[str, dict[str, object]]
    job_builds: dict[str, list[JobBuildEntry]]
    market_source_allowlist: list[str]
    upgrade_rules_by_job: dict[str, list[OpportunisticUpgradeRule]]
    default_upgrade_rules: list[OpportunisticUpgradeRule]

    def recommend_leveling(self, *, job_name: str | None, base_level: int) -> dict[str, object]:
        profile = self._resolve_profile(job_name)
        if profile is None:
            return {
                "knowledge_loaded": False,
                "job_name": str(job_name or ""),
                "base_level": int(base_level),
                "profile_key": "unknown",
                "band_label": "unknown",
                "recommended_focus": "safe_grind",
                "target_maps": [],
                "objective_template": "continue deterministic leveling progression safely",
            }

        ordered_bands = sorted(profile.bands, key=lambda item: (item.min_base_level, item.max_base_level))
        selected = next(
            (
                item
                for item in ordered_bands
                if int(base_level) >= int(item.min_base_level) and int(base_level) <= int(item.max_base_level)
            ),
            ordered_bands[-1] if ordered_bands else None,
        )
        if selected is None:
            return {
                "knowledge_loaded": True,
                "job_name": str(job_name or ""),
                "base_level": int(base_level),
                "profile_key": profile.key,
                "band_label": "unknown",
                "recommended_focus": "safe_grind",
                "target_maps": [],
                "objective_template": "continue deterministic leveling progression safely",
            }

        return {
            "knowledge_loaded": True,
            "job_name": str(job_name or ""),
            "base_level": int(base_level),
            "profile_key": profile.key,
            "band_label": selected.label,
            "recommended_focus": selected.recommended_focus,
            "target_maps": list(selected.target_maps),
            "objective_template": selected.objective_template,
        }

    def assess_job_advancement(self, *, job_name: str | None, base_level: int, job_level: int) -> dict[str, object]:
        normalized_job = _normalize_job_name(job_name)
        candidates = sorted(
            self.playbooks_by_job.get(normalized_job, []),
            key=lambda item: (item.route_priority, item.route_id),
        )
        if not candidates:
            return {
                "supported": False,
                "ready": False,
                "status": "unsupported_job_route",
                "current_job": str(job_name or "unknown"),
                "route_id": "",
                "target_job": "",
                "requirements": {},
                "missing_requirements": [],
                "known_routes": [],
                "playbook_steps": [],
                "location": {},
                "build_recommendation": {},
                "notes": ["no curated stage-3 playbook configured for this class route"],
            }

        selected = candidates[0]
        missing: list[str] = []
        if int(base_level) < int(selected.required_base_level):
            missing.append(f"base_level<{selected.required_base_level}")
        if int(job_level) < int(selected.required_job_level):
            missing.append(f"job_level<{selected.required_job_level}")

        target_key = _normalize_job_name(selected.target_job)
        location = self.job_change_locations.get(target_key, {})
        build_entries = self.job_builds.get(target_key, [])
        default_build = build_entries[0] if build_entries else None

        return {
            "supported": True,
            "ready": len(missing) == 0,
            "status": "ready" if len(missing) == 0 else "requirements_unmet",
            "current_job": selected.current_job,
            "route_id": selected.route_id,
            "target_job": selected.target_job,
            "requirements": {
                "base_level": int(selected.required_base_level),
                "job_level": int(selected.required_job_level),
            },
            "missing_requirements": missing,
            "known_routes": [item.route_id for item in candidates],
            "playbook_steps": list(selected.steps),
            "location": dict(location),
            "build_recommendation": {
                "build_name": default_build.build_name if default_build is not None else "",
                "stat_priority": list(default_build.stat_priority) if default_build is not None else [],
            },
            "notes": list(selected.notes),
        }

    def assess_opportunistic_upgrades(
        self,
        *,
        job_name: str | None,
        base_level: int,
        zeny: int,
        inventory_items: list[dict[str, object]],
        market_listings: list[dict[str, object]],
        signals: dict[str, object] | None = None,
    ) -> dict[str, object]:
        normalized_job = _normalize_job_name(job_name)
        selected_rules = self._resolve_upgrade_rules(job_name)
        signal_map = dict(signals or {})
        if not selected_rules:
            return {
                "knowledge_loaded": True,
                "supported": False,
                "actionable": False,
                "status": "unsupported",
                "job_name": str(job_name or "unknown"),
                "base_level": int(base_level),
                "zeny": int(max(0, int(zeny))),
                "signals": signal_map,
                "known_rule_ids": [],
                "opportunities": [],
                "non_actionable_reasons": ["no_curated_upgrade_rules_for_job"],
            }

        equipped_by_slot: dict[str, dict[str, object]] = {}
        for item in inventory_items:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("equipped")):
                continue
            slot = self._inventory_slot(item)
            if not slot:
                continue
            equipped_by_slot.setdefault(slot, item)

        market_quotes = self._index_market_quotes(market_listings)
        available_zeny = int(max(0, int(zeny)))

        opportunities: list[dict[str, object]] = []
        non_actionable_reasons: list[str] = []

        for rule in selected_rules:
            signal_reasons = self._evaluate_signal_requirements(
                rule_id=rule.rule_id,
                requirements=rule.signal_requirements,
                signals=signal_map,
            )
            if signal_reasons:
                non_actionable_reasons.extend(signal_reasons)
                continue

            if int(base_level) < int(rule.required_base_level):
                non_actionable_reasons.append(f"{rule.rule_id}:base_level<{rule.required_base_level}")
                continue

            if str(rule.evidence_mode or "").strip().lower() == "signals":
                execution_payload, execution_error = self._normalize_execution_payload(rule=rule)
                if execution_error:
                    non_actionable_reasons.append(f"{rule.rule_id}:{execution_error}")
                    continue

                opportunities.append(
                    {
                        "rule_id": rule.rule_id,
                        "domain": rule.domain,
                        "evidence_mode": "signals",
                        "job_name": str(job_name or "unknown"),
                        "job_key": normalized_job,
                        "slot": rule.slot,
                        "current_item_id": "",
                        "current_item_name": "",
                        "candidate_item_id": rule.candidate_item_id,
                        "candidate_item_name": rule.candidate_item_name,
                        "buy_price": 0,
                        "max_price_zeny": 0,
                        "score_current": int(rule.score_current),
                        "score_candidate": int(rule.score_candidate),
                        "score_delta": int(rule.score_candidate) - int(rule.score_current),
                        "zeny_after": available_zeny,
                        "quote_source": "signals",
                        "execution_mode": rule.execution_mode,
                        "execution_payload": execution_payload,
                        "signal_requirements": dict(rule.signal_requirements),
                        "notes": list(rule.notes),
                    }
                )
                continue

            slot = str(rule.slot).strip().lower()
            current = equipped_by_slot.get(slot)
            if current is None:
                non_actionable_reasons.append(f"{rule.rule_id}:missing_current_slot_context")
                continue

            current_item_id = str(current.get("item_id") or "").strip().lower()
            allowed_current = {str(item).strip().lower() for item in rule.current_item_ids}
            if current_item_id not in allowed_current:
                non_actionable_reasons.append(f"{rule.rule_id}:current_item_mismatch")
                continue

            quote = market_quotes.get(str(rule.candidate_item_id).strip().lower())
            if quote is None:
                non_actionable_reasons.append(f"{rule.rule_id}:market_quote_missing")
                continue

            buy_price = quote.get("buy_price")
            source = str(quote.get("source") or "").strip().lower()
            if not isinstance(buy_price, int) or buy_price <= 0:
                non_actionable_reasons.append(f"{rule.rule_id}:market_buy_price_missing")
                continue
            if not source:
                non_actionable_reasons.append(f"{rule.rule_id}:market_source_missing")
                continue

            allowlist = {str(item).strip().lower() for item in self.market_source_allowlist if str(item).strip()}
            if allowlist and source not in allowlist:
                non_actionable_reasons.append(f"{rule.rule_id}:market_source_unsupported:{source}")
                continue
            if buy_price > int(rule.max_price_zeny):
                non_actionable_reasons.append(f"{rule.rule_id}:price_above_cap")
                continue
            if available_zeny < buy_price:
                non_actionable_reasons.append(f"{rule.rule_id}:insufficient_zeny")
                continue

            score_delta = int(rule.score_candidate) - int(rule.score_current)
            if score_delta <= 0:
                non_actionable_reasons.append(f"{rule.rule_id}:non_positive_score_delta")
                continue

            execution_payload, execution_error = self._normalize_execution_payload(rule=rule)
            if execution_error:
                non_actionable_reasons.append(f"{rule.rule_id}:{execution_error}")
                continue

            opportunities.append(
                {
                    "rule_id": rule.rule_id,
                    "domain": rule.domain,
                    "evidence_mode": "gear_market",
                    "job_name": str(job_name or "unknown"),
                    "job_key": normalized_job,
                    "slot": rule.slot,
                    "current_item_id": str(current.get("item_id") or ""),
                    "current_item_name": str(current.get("name") or current.get("item_id") or "").strip(),
                    "candidate_item_id": rule.candidate_item_id,
                    "candidate_item_name": rule.candidate_item_name,
                    "buy_price": int(buy_price),
                    "max_price_zeny": int(rule.max_price_zeny),
                    "score_current": int(rule.score_current),
                    "score_candidate": int(rule.score_candidate),
                    "score_delta": int(score_delta),
                    "zeny_after": max(0, available_zeny - int(buy_price)),
                    "quote_source": source,
                    "execution_mode": rule.execution_mode,
                    "execution_payload": execution_payload,
                    "signal_requirements": dict(rule.signal_requirements),
                    "notes": list(rule.notes),
                }
            )

        opportunities = sorted(
            opportunities,
            key=lambda item: (-int(item.get("score_delta") or 0), int(item.get("buy_price") or 10**9), str(item.get("rule_id") or "")),
        )
        deduped_reasons: list[str] = []
        seen: set[str] = set()
        for item in non_actionable_reasons:
            if item in seen:
                continue
            seen.add(item)
            deduped_reasons.append(item)

        return {
            "knowledge_loaded": True,
            "supported": True,
            "actionable": bool(opportunities),
            "status": "actionable" if opportunities else "non_actionable",
            "job_name": str(job_name or "unknown"),
            "base_level": int(base_level),
            "zeny": available_zeny,
            "signals": signal_map,
            "known_rule_ids": [item.rule_id for item in selected_rules],
            "opportunities": opportunities,
            "non_actionable_reasons": deduped_reasons,
        }

    def _evaluate_signal_requirements(
        self,
        *,
        rule_id: str,
        requirements: dict[str, object],
        signals: dict[str, object],
    ) -> list[str]:
        if not requirements:
            return []
        reasons: list[str] = []
        for key, expected in requirements.items():
            raw_key = str(key or "").strip()
            if not raw_key:
                continue
            if raw_key.endswith("_gte"):
                signal_key = raw_key[: -len("_gte")]
                observed = _parse_float(signals.get(signal_key), default=float("-inf"))
                threshold = _parse_float(expected, default=0.0)
                if observed < threshold:
                    reasons.append(f"{rule_id}:signal<{signal_key}>_lt_{threshold}")
                continue
            if raw_key.endswith("_lte"):
                signal_key = raw_key[: -len("_lte")]
                observed = _parse_float(signals.get(signal_key), default=float("inf"))
                threshold = _parse_float(expected, default=0.0)
                if observed > threshold:
                    reasons.append(f"{rule_id}:signal<{signal_key}>_gt_{threshold}")
                continue

            observed = signals.get(raw_key)
            if isinstance(expected, bool):
                if isinstance(observed, bool):
                    observed_bool = observed
                elif isinstance(observed, (int, float)):
                    observed_bool = observed != 0
                else:
                    observed_bool = str(observed or "").strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "active"}
                if observed_bool is not expected:
                    reasons.append(f"{rule_id}:signal<{raw_key}>_mismatch")
                continue
            if isinstance(expected, (int, float)):
                if _parse_float(observed, default=float("nan")) != _parse_float(expected, default=0.0):
                    reasons.append(f"{rule_id}:signal<{raw_key}>_mismatch")
                continue
            if str(observed or "").strip().lower() != str(expected or "").strip().lower():
                reasons.append(f"{rule_id}:signal<{raw_key}>_mismatch")
        return reasons

    def _resolve_upgrade_rules(self, job_name: str | None) -> list[OpportunisticUpgradeRule]:
        normalized_job = _normalize_job_name(job_name)
        selected: list[OpportunisticUpgradeRule] = []
        if normalized_job and normalized_job in self.upgrade_rules_by_job:
            selected.extend(self.upgrade_rules_by_job.get(normalized_job, []))
        selected.extend(self.default_upgrade_rules)

        deduped: list[OpportunisticUpgradeRule] = []
        seen: set[str] = set()
        for rule in selected:
            rule_id = str(rule.rule_id or "").strip()
            if not rule_id or rule_id in seen:
                continue
            seen.add(rule_id)
            deduped.append(rule)
        return deduped

    def _inventory_slot(self, item: dict[str, object]) -> str:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        for key in ("slot", "equip_slot", "equipment_slot", "location"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        category = str(item.get("category") or "").strip().lower()
        name = str(item.get("name") or "").strip().lower()
        if category == "weapon" or "sword" in name or "knife" in name:
            return "weapon"
        if category in {"shield"}:
            return "shield"
        if category in {"armor", "garment", "robe", "gear"}:
            return "armor"
        return ""

    def _index_market_quotes(self, market_listings: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        by_item: dict[str, dict[str, object]] = {}
        for quote in market_listings:
            if not isinstance(quote, dict):
                continue
            item_id = str(quote.get("item_id") or "").strip().lower()
            if not item_id:
                continue
            buy_price = _parse_int(quote.get("buy_price"), default=-1)
            source = str(quote.get("source") or "").strip().lower()
            current = by_item.get(item_id)
            if current is None:
                by_item[item_id] = {
                    "item_id": item_id,
                    "buy_price": buy_price if buy_price > 0 else None,
                    "source": source,
                }
                continue
            previous_price = current.get("buy_price")
            if isinstance(previous_price, int) and previous_price > 0 and buy_price > 0 and buy_price >= previous_price:
                continue
            if buy_price > 0:
                by_item[item_id] = {
                    "item_id": item_id,
                    "buy_price": buy_price,
                    "source": source,
                }
        return by_item

    def _normalize_execution_payload(self, *, rule: OpportunisticUpgradeRule) -> tuple[dict[str, object], str]:
        mode = str(rule.execution_mode or "").strip().lower()
        if mode == "direct":
            command = str(rule.direct_command or "").strip()
            if not command:
                return {}, "direct_command_missing"
            return {
                "direct_command": command,
                "conflict_key": str(rule.direct_conflict_key or "").strip(),
                "preconditions": [str(item) for item in rule.direct_preconditions if str(item).strip()],
            }, ""
        if mode == "config":
            desired_raw = rule.control_plan.get("desired") if isinstance(rule.control_plan.get("desired"), dict) else {}
            desired = {str(k): str(v) for k, v in desired_raw.items() if str(k).strip()}
            target_path = str(rule.control_plan.get("target_path") or "").strip()
            name = str(rule.control_plan.get("name") or "").strip() or "config.txt"
            if not desired or not target_path:
                return {}, "config_plan_incomplete"
            return {
                "profile": str(rule.control_plan.get("profile") or "").strip() or None,
                "artifact_type": str(rule.control_plan.get("artifact_type") or "config").strip() or "config",
                "name": name,
                "target_path": target_path,
                "desired": desired,
                "source": "crewai",
            }, ""
        if mode == "macro":
            bundle = dict(rule.macro_bundle or {})
            macros = bundle.get("macros")
            event_macros = bundle.get("event_macros")
            automacros = bundle.get("automacros")
            has_any = bool((isinstance(macros, list) and macros) or (isinstance(event_macros, list) and event_macros) or (isinstance(automacros, list) and automacros))
            if not has_any:
                return {}, "macro_bundle_empty"
            return bundle, ""
        return {}, "execution_mode_unsupported"

    def _resolve_profile(self, job_name: str | None) -> ProgressionProfile | None:
        normalized_job = _normalize_job_name(job_name)
        if normalized_job and normalized_job in self.profile_lookup:
            return self.profile_lookup[normalized_job]
        return self.default_profile


def load_ro_knowledge(
    *,
    data_dir: Path | None = None,
    tables_dir: Path | None = None,
) -> ROKnowledgeBundle:
    data_root = data_dir or _DEFAULT_DATA_DIR
    tables_root = tables_dir or _DEFAULT_TABLES_DIR

    progression_payload = _read_json(data_root / "progression_profiles.json")
    playbook_payload = _read_json(data_root / "job_change_playbooks.json")
    opportunistic_payload = _read_json(data_root / "opportunistic_upgrades.json")

    profiles, default_profile = _parse_progression_profiles(progression_payload)
    playbooks_by_job = _parse_playbooks(playbook_payload)
    job_change_locations = _parse_job_change_locations(tables_root / "job_change_locations.txt")
    job_builds = _parse_job_builds(tables_root / "job_builds.txt")
    market_source_allowlist, upgrade_rules_by_job, default_upgrade_rules = _parse_opportunistic_upgrades(opportunistic_payload)

    version = str(progression_payload.get("version") or "stage3-ro-progression-v1")
    return ROKnowledgeBundle(
        version=version,
        profile_lookup=profiles,
        default_profile=default_profile,
        playbooks_by_job=playbooks_by_job,
        job_change_locations=job_change_locations,
        job_builds=job_builds,
        market_source_allowlist=market_source_allowlist,
        upgrade_rules_by_job=upgrade_rules_by_job,
        default_upgrade_rules=default_upgrade_rules,
    )


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        logger.warning(
            "autonomy_ro_knowledge_json_missing",
            extra={"event": "autonomy_ro_knowledge_json_missing", "path": str(path)},
        )
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception(
            "autonomy_ro_knowledge_json_invalid",
            extra={"event": "autonomy_ro_knowledge_json_invalid", "path": str(path)},
        )
        return {}


def _parse_progression_profiles(payload: dict[str, object]) -> tuple[dict[str, ProgressionProfile], ProgressionProfile | None]:
    profiles_raw = payload.get("profiles")
    if not isinstance(profiles_raw, list):
        return {}, None

    lookup: dict[str, ProgressionProfile] = {}
    default_profile: ProgressionProfile | None = None
    for item in profiles_raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue

        job_names = [str(name).strip() for name in item.get("job_names", []) if str(name).strip()]
        bands_raw = item.get("bands") if isinstance(item.get("bands"), list) else []
        bands: list[ProgressionBand] = []
        for band in bands_raw:
            if not isinstance(band, dict):
                continue
            bands.append(
                ProgressionBand(
                    label=str(band.get("label") or "default"),
                    min_base_level=_parse_int(band.get("min_base_level"), default=1),
                    max_base_level=_parse_int(band.get("max_base_level"), default=255),
                    recommended_focus=str(band.get("recommended_focus") or "safe_grind"),
                    target_maps=[str(value) for value in band.get("target_maps", []) if str(value).strip()],
                    objective_template=str(
                        band.get("objective_template") or "continue deterministic leveling progression safely"
                    ),
                )
            )

        profile = ProgressionProfile(key=key, job_names=job_names, bands=bands)
        raw_default = item.get("default")
        is_default = bool(raw_default is True or str(raw_default or "").strip().lower() == "true")
        if is_default:
            default_profile = profile
        for job_name in job_names:
            lookup[_normalize_job_name(job_name)] = profile

    return lookup, default_profile


def _parse_playbooks(payload: dict[str, object]) -> dict[str, list[JobChangePlaybook]]:
    routes_raw = payload.get("routes")
    if not isinstance(routes_raw, list):
        return {}

    by_current_job: dict[str, list[JobChangePlaybook]] = {}
    for item in routes_raw:
        if not isinstance(item, dict):
            continue
        if item.get("enabled") is False:
            continue

        route_id = str(item.get("route_id") or "").strip()
        current_job = str(item.get("current_job") or "").strip()
        target_job = str(item.get("target_job") or "").strip()
        if not route_id or not current_job or not target_job:
            continue

        record = JobChangePlaybook(
            route_id=route_id,
            route_priority=max(1, _parse_int(item.get("route_priority"), default=999)),
            current_job=current_job,
            target_job=target_job,
            required_base_level=max(0, _parse_int(item.get("required_base_level"), default=0)),
            required_job_level=max(0, _parse_int(item.get("required_job_level"), default=0)),
            steps=[str(step) for step in item.get("steps", []) if str(step).strip()],
            notes=[str(note) for note in item.get("notes", []) if str(note).strip()],
        )
        key = _normalize_job_name(current_job)
        by_current_job.setdefault(key, []).append(record)

    return by_current_job


def _parse_opportunistic_upgrades(
    payload: dict[str, object],
) -> tuple[list[str], dict[str, list[OpportunisticUpgradeRule]], list[OpportunisticUpgradeRule]]:
    allowlist = [str(item).strip().lower() for item in payload.get("market_source_allowlist", []) if str(item).strip()]
    rules_raw = payload.get("rules") if isinstance(payload.get("rules"), list) else []
    by_job: dict[str, list[OpportunisticUpgradeRule]] = {}
    default_rules: list[OpportunisticUpgradeRule] = []

    for item in rules_raw:
        if not isinstance(item, dict):
            continue
        if item.get("enabled") is False:
            continue

        rule_id = str(item.get("rule_id") or "").strip()
        evidence_mode = str(item.get("evidence_mode") or "gear_market").strip().lower()
        if evidence_mode not in {"gear_market", "signals"}:
            continue

        slot = str(item.get("slot") or "").strip().lower()
        candidate_item_id = str(item.get("candidate_item_id") or "").strip().lower()
        if not rule_id:
            continue
        if evidence_mode == "gear_market" and (not slot or not candidate_item_id):
            continue
        if evidence_mode == "signals":
            slot = slot or "activity"
            candidate_item_id = candidate_item_id or rule_id

        mode = str(item.get("execution_mode") or "").strip().lower()
        if mode not in {"direct", "config", "macro"}:
            continue

        rule = OpportunisticUpgradeRule(
            rule_id=rule_id,
            domain=str(item.get("domain") or "opportunistic_upgrades").strip().lower() or "opportunistic_upgrades",
            evidence_mode=evidence_mode,
            job_names=[str(value).strip() for value in item.get("job_names", []) if str(value).strip()],
            slot=slot,
            current_item_ids=[str(value).strip().lower() for value in item.get("current_item_ids", []) if str(value).strip()],
            candidate_item_id=candidate_item_id,
            candidate_item_name=str(item.get("candidate_item_name") or item.get("candidate_item_id") or rule_id).strip(),
            required_base_level=max(1, _parse_int(item.get("required_base_level"), default=1)),
            score_current=max(0, _parse_int(item.get("score_current"), default=0)),
            score_candidate=max(0, _parse_int(item.get("score_candidate"), default=0)),
            max_price_zeny=max(0, _parse_int(item.get("max_price_zeny"), default=0)),
            execution_mode=mode,
            direct_command=str(item.get("direct_command") or "").strip(),
            direct_conflict_key=str(item.get("direct_conflict_key") or "").strip(),
            direct_preconditions=[str(value).strip() for value in item.get("direct_preconditions", []) if str(value).strip()],
            signal_requirements=dict(item.get("signal_requirements") or {}),
            control_plan=dict(item.get("control_plan") or {}),
            macro_bundle=dict(item.get("macro_bundle") or {}),
            notes=[str(value).strip() for value in item.get("notes", []) if str(value).strip()],
        )
        if not rule.job_names:
            default_rules.append(rule)
            continue

        for job_name in rule.job_names:
            by_job.setdefault(_normalize_job_name(job_name), []).append(rule)

    return allowlist, by_job, default_rules


def _parse_job_change_locations(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        logger.warning(
            "autonomy_ro_knowledge_locations_missing",
            extra={"event": "autonomy_ro_knowledge_locations_missing", "path": str(path)},
        )
        return {}

    locations: dict[str, dict[str, object]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [item.strip() for item in line.split("|")]
        if len(parts) < 5:
            continue
        target_job, map_name, coordinates, description, requirements = parts[:5]
        parsed_coords = _parse_coordinates(coordinates)
        locations[_normalize_job_name(target_job)] = {
            "target_job": target_job,
            "map": map_name,
            "coordinates": parsed_coords,
            "description": description,
            "requirements_text": requirements,
        }
    return locations


def _parse_job_builds(path: Path) -> dict[str, list[JobBuildEntry]]:
    if not path.exists():
        logger.warning(
            "autonomy_ro_knowledge_job_builds_missing",
            extra={"event": "autonomy_ro_knowledge_job_builds_missing", "path": str(path)},
        )
        return {}

    builds: dict[str, list[JobBuildEntry]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue

        job_class = parts[0]
        build_name = parts[1]
        stat_priority = parts[2:8]
        entry = JobBuildEntry(job_class=job_class, build_name=build_name, stat_priority=stat_priority)
        builds.setdefault(_normalize_job_name(job_class), []).append(entry)
    return builds
