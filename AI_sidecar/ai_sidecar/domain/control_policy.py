from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ai_sidecar.contracts.control_domain import ControlOwnerScope, ControlPolicyRule, ControlPolicySnapshot


@dataclass(slots=True)
class ControlPolicy:
    policy: ControlPolicySnapshot

    def is_protected(self, key: str) -> bool:
        normalized = (key or "").strip()
        if not normalized:
            return False
        for exact in self.policy.protected_exact:
            if normalized == exact:
                return True
        for prefix in self.policy.protected_prefixes:
            if normalized.startswith(prefix):
                return True
        return False

    def owner_for(self, key: str) -> ControlOwnerScope:
        normalized = (key or "").strip()
        for rule in self.policy.rules:
            if normalized == "":
                continue
            if _match_pattern(rule.key_pattern, normalized):
                return rule.owner
        return ControlOwnerScope.sidecar

    def allow_write(self, key: str, owner: ControlOwnerScope) -> tuple[bool, str]:
        normalized = (key or "").strip()
        if not normalized:
            return False, "empty_key"
        if self.is_protected(normalized):
            return False, "protected_key"
        for rule in self.policy.rules:
            if _match_pattern(rule.key_pattern, normalized):
                if not rule.allow_write:
                    return False, rule.reason or "write_blocked"
                if rule.owner != owner:
                    return False, "owner_mismatch"
                return True, "ok"
        if owner != ControlOwnerScope.sidecar:
            return False, "owner_mismatch"
        return True, "ok"


def _match_pattern(pattern: str, key: str) -> bool:
    if not pattern:
        return False
    if pattern.endswith("*"):
        return key.startswith(pattern[:-1])
    return key == pattern


def default_control_policy() -> ControlPolicy:
    protected_exact = [
        "master",
        "server",
        "username",
        "password",
        "char",
        "charName",
    ]
    protected_prefixes = [
        "master_",
        "server_",
    ]
    rules = [
        ControlPolicyRule(key_pattern="aiSidecar_*", owner=ControlOwnerScope.sidecar, allow_write=True),
        ControlPolicyRule(key_pattern="macro_*", owner=ControlOwnerScope.sidecar, allow_write=True),
        ControlPolicyRule(key_pattern="eventMacro_*", owner=ControlOwnerScope.sidecar, allow_write=True),
        ControlPolicyRule(key_pattern="loadPlugins_list", owner=ControlOwnerScope.operator, allow_write=False, reason="managed_by_operator"),
    ]
    snapshot = ControlPolicySnapshot(
        version="local-default",
        protected_prefixes=protected_prefixes,
        protected_exact=protected_exact,
        rules=rules,
    )
    return ControlPolicy(policy=snapshot)


def ensure_policy_snapshot(policy: ControlPolicySnapshot | None) -> ControlPolicy:
    if policy is None:
        return default_control_policy()
    return ControlPolicy(policy=policy)


def filter_allowed_keys(
    *,
    keys: Iterable[str],
    policy: ControlPolicy,
    owner: ControlOwnerScope,
) -> tuple[list[str], dict[str, str]]:
    allowed: list[str] = []
    denied: dict[str, str] = {}
    for key in keys:
        ok, reason = policy.allow_write(key, owner)
        if ok:
            allowed.append(key)
        else:
            denied[str(key)] = reason
    return allowed, denied

