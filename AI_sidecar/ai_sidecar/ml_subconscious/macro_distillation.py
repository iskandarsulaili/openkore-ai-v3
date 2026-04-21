from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.macros import EventAutomacro, MacroPublishRequest, MacroRoutine
from ai_sidecar.contracts.ml_subconscious import MLTrainingEpisode

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MacroDistillationEngine:
    _lock: RLock = field(default_factory=RLock)
    _stats: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._stats = {
            "distill_attempts": 0,
            "distill_candidates": 0,
            "distill_published": 0,
        }

    def _extract_sequence(self, episode: MLTrainingEpisode, *, max_steps: int) -> list[str]:
        payload = dict(episode.decision_payload or {})
        action = dict(episode.executed_action or {})

        sequence = payload.get("sequence")
        if isinstance(sequence, list):
            rows = [str(item).strip() for item in sequence if str(item).strip()]
            return rows[:max_steps]

        action_sequence = payload.get("action_sequence")
        if isinstance(action_sequence, list):
            rows = [str(item).strip() for item in action_sequence if str(item).strip()]
            return rows[:max_steps]

        command = action.get("command") or payload.get("command")
        if isinstance(command, str) and command.strip():
            return [command.strip()[:256]]
        return []

    def _is_deterministic(self, signatures: list[tuple[str, ...]], *, min_support: int) -> tuple[bool, tuple[str, ...], int]:
        if len(signatures) < min_support:
            return False, tuple(), 0
        freq: dict[tuple[str, ...], int] = {}
        for sig in signatures:
            freq[sig] = freq.get(sig, 0) + 1
        best_sig = tuple()
        best_count = 0
        for sig, count in freq.items():
            if count > best_count:
                best_sig = sig
                best_count = count
        deterministic = bool(best_count >= min_support and best_sig)
        return deterministic, best_sig, best_count

    def distill(
        self,
        *,
        meta: ContractMeta,
        episodes: list[MLTrainingEpisode],
        min_support: int,
        max_steps: int,
        enqueue_reload: bool,
        publish_macro,
    ) -> dict[str, object]:
        with self._lock:
            self._stats["distill_attempts"] += 1

        bot_id = meta.bot_id
        if not episodes:
            return {
                "ok": False,
                "message": "no_episodes",
                "bot_id": bot_id,
                "proposal_id": "",
                "support": 0,
                "success_rate": 0.0,
                "macro": {},
                "automacro": {},
                "publication": None,
            }

        signatures: list[tuple[str, ...]] = []
        successful = 0
        for episode in episodes:
            if episode.outcome.success:
                successful += 1
            seq = self._extract_sequence(episode, max_steps=max_steps)
            if seq:
                signatures.append(tuple(seq))

        deterministic, best_seq, support = self._is_deterministic(signatures, min_support=min_support)
        success_rate = float(successful) / float(len(episodes))
        if not deterministic:
            return {
                "ok": False,
                "message": "no_deterministic_sequence",
                "bot_id": bot_id,
                "proposal_id": "",
                "support": support,
                "success_rate": success_rate,
                "macro": {},
                "automacro": {},
                "publication": None,
            }

        with self._lock:
            self._stats["distill_candidates"] += 1

        proposal_id = f"distill-{uuid4().hex[:12]}"
        macro_name = f"ml_distilled_{uuid4().hex[:10]}"
        macro = MacroRoutine(name=macro_name, lines=list(best_seq))
        automacro = EventAutomacro(
            name=f"on_{macro_name}",
            conditions=["OnCharLogIn"],
            call=macro_name,
            parameters={"bot_id": bot_id},
        )

        publication = None
        if enqueue_reload:
            try:
                ok, info, message = publish_macro(
                    MacroPublishRequest(
                        meta=meta,
                        target_bot_id=bot_id,
                        macros=[macro],
                        event_macros=[],
                        automacros=[automacro],
                        enqueue_reload=True,
                    )
                )
                publication = {
                    "ok": ok,
                    "message": message,
                    "details": info,
                }
                if ok:
                    with self._lock:
                        self._stats["distill_published"] += 1
            except Exception as exc:
                logger.exception(
                    "ml_macro_distill_publish_failed",
                    extra={"event": "ml_macro_distill_publish_failed", "bot_id": bot_id, "proposal_id": proposal_id},
                )
                publication = {
                    "ok": False,
                    "message": str(exc),
                    "details": None,
                }

        return {
            "ok": True,
            "message": "distilled",
            "bot_id": bot_id,
            "proposal_id": proposal_id,
            "support": support,
            "success_rate": success_rate,
            "macro": macro.model_dump(mode="json"),
            "automacro": automacro.model_dump(mode="json"),
            "publication": publication,
        }

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

