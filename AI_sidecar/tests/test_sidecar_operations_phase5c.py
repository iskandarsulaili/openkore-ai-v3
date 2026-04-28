from __future__ import annotations

import asyncio
import json

from ai_sidecar.config import settings
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.macros import MacroPublishRequest, MacroRoutine
from ai_sidecar.domain.macro_compiler import MacroPublisher
from ai_sidecar.lifecycle import create_runtime


def _configure_isolated_runtime(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "sidecar.sqlite"))
    monkeypatch.setattr(settings, "memory_openmemory_path", str(tmp_path / "openmemory.sqlite"))
    monkeypatch.setattr(settings, "provider_ollama_enabled", False)
    monkeypatch.setattr(settings, "provider_openai_enabled", False)
    monkeypatch.setattr(settings, "provider_deepseek_enabled", False)
    monkeypatch.setattr(settings, "provider_policy_json", "")
    monkeypatch.setattr(settings, "fleet_central_enabled", False)
    monkeypatch.setattr(settings, "crewai_enabled", False)
    monkeypatch.setattr(settings, "crewai_memory_enabled", False)


def test_macro_publish_links_reload_action_and_suppresses_duplicate(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _configure_isolated_runtime(monkeypatch, tmp_path)
    runtime = create_runtime()
    runtime.macro_publisher = MacroPublisher(workspace_root=tmp_path)

    try:
        request = MacroPublishRequest(
            meta=ContractMeta(contract_version="v1", source="pytest", bot_id="bot:macro5c", trace_id="trace-macro5c"),
            target_bot_id="bot:macro5c",
            macros=[MacroRoutine(name="phase5c_macro", lines=["do ai auto"])],
            event_macros=[],
            automacros=[],
            enqueue_reload=True,
        )

        ok, publication, message = runtime.publish_macros(request)
        assert ok is True
        assert publication is not None
        assert message == "macro artifacts published"
        assert publication["reload_queued"] is True
        assert publication["reload_action_id"]

        operation_id = str(publication.get("operation_id") or "")
        assert operation_id
        assert runtime.repositories is not None

        operation = runtime.repositories.operations.get(operation_id=operation_id)
        assert operation is not None
        assert operation.status == "reload_pending"
        assert operation.linked_action_id == publication["reload_action_id"]

        action = runtime.repositories.actions.get(action_id=str(publication["reload_action_id"]))
        assert action is not None
        metadata = action.proposal.get("metadata") if isinstance(action.proposal, dict) else {}
        assert isinstance(metadata, dict)
        assert metadata.get("sidecar_operation_id") == operation_id
        assert metadata.get("operation_kind") == "macro_publish"

        ok_dupe, publication_dupe, message_dupe = runtime.publish_macros(request)
        assert ok_dupe is True
        assert publication_dupe is not None
        assert message_dupe == "macro artifacts already current"
        assert publication_dupe["reload_queued"] is False
        assert publication_dupe["reload_reason"] == "already_current"
        assert publication_dupe["operation_id"] == operation_id
        assert runtime.repositories.actions.count(bot_id="bot:macro5c") == 1

        operation_after = runtime.repositories.operations.get(operation_id=operation_id)
        assert operation_after is not None
        assert operation_after.status == "completed"
        assert operation_after.reconciled_at is not None
    finally:
        asyncio.run(runtime.shutdown())


def test_restart_reconciles_pending_operation_with_current_artifact(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _configure_isolated_runtime(monkeypatch, tmp_path)

    runtime1 = create_runtime()
    artifact_relpath = "openkore-ai-v3/AI_sidecar/tests/.phase5c_reconcile_artifact.txt"
    artifact_abspath = runtime1.workspace_root / artifact_relpath
    artifact_abspath.parent.mkdir(parents=True, exist_ok=True)
    desired_text = "aiSidecar_enable 1\n"
    artifact_abspath.write_text(desired_text, encoding="utf-8")

    try:
        desired_checksum = runtime1._sha256_text(desired_text)
        operation_id = runtime1.begin_sidecar_operation(
            bot_id="bot:reconcile5c",
            operation_kind="control_apply",
            artifact_kind="config",
            artifact_path=artifact_relpath,
            idempotency_key=f"control-apply:bot:reconcile5c:{artifact_relpath}:{desired_checksum}",
            payload={"test_case": "restart_reconcile"},
            base_checksum=desired_checksum,
            desired_checksum=desired_checksum,
            status="applied",
            status_reason="seed_pending_operation",
        )
        assert operation_id is not None
    finally:
        asyncio.run(runtime1.shutdown())

    runtime2 = create_runtime()
    try:
        assert runtime2.repositories is not None
        reconciled = runtime2.repositories.operations.get(operation_id=str(operation_id))
        assert reconciled is not None
        assert reconciled.status == "completed"
        assert reconciled.status_reason == "reconcile_artifact_current"
        assert reconciled.reconciled_at is not None
    finally:
        asyncio.run(runtime2.shutdown())
        if artifact_abspath.exists():
            artifact_abspath.unlink()


def test_restart_reconciles_pending_macro_publish_with_current_manifest(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    _configure_isolated_runtime(monkeypatch, tmp_path)

    runtime1 = create_runtime()
    manifest_relpath = "openkore-ai-v3/AI_sidecar/tests/.phase5e_macro_manifest.json"
    manifest_abspath = runtime1.workspace_root / manifest_relpath
    manifest_abspath.parent.mkdir(parents=True, exist_ok=True)
    desired_checksum = runtime1._sha256_text("phase5e-macro-manifest-content")
    manifest_abspath.write_text(
        json.dumps(
            {
                "publication_id": "pub-phase5e",
                "version": "v-phase5e",
                "content_sha256": desired_checksum,
                "published_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    try:
        operation_id = runtime1.begin_sidecar_operation(
            bot_id="bot:macro-reconcile5e",
            operation_kind="macro_publish",
            artifact_kind="macro_manifest",
            artifact_path=manifest_relpath,
            idempotency_key=f"macro-publish:bot:macro-reconcile5e:{desired_checksum}",
            payload={"test_case": "restart_reconcile_macro_manifest"},
            base_checksum=desired_checksum,
            desired_checksum=desired_checksum,
            status="applied",
            status_reason="seed_pending_macro_operation",
        )
        assert operation_id is not None
    finally:
        asyncio.run(runtime1.shutdown())

    runtime2 = create_runtime()
    try:
        assert runtime2.repositories is not None
        reconciled = runtime2.repositories.operations.get(operation_id=str(operation_id))
        assert reconciled is not None
        assert reconciled.status == "completed"
        assert reconciled.status_reason == "reconcile_artifact_current"
        assert reconciled.reconciled_at is not None
    finally:
        asyncio.run(runtime2.shutdown())
        if manifest_abspath.exists():
            manifest_abspath.unlink()
