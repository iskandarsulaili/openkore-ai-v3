from __future__ import annotations

from ai_sidecar.contracts.macros import EventAutomacro
from ai_sidecar.domain.macro_compiler import MacroCompiler


def test_macro_compiler_normalizes_bare_simple_event_condition_to_pair_form():
    compiler = MacroCompiler()

    compiled = compiler.compile(
        macros=[],
        event_macros=[],
        automacros=[
            EventAutomacro(
                name="on_login_test",
                conditions=["OnCharLogIn"],
                call="login_handler",
                parameters={"priority": "0"},
            )
        ],
    )

    assert "    OnCharLogIn 1\n" in compiled.event_macro_text
    assert "    call login_handler\n" in compiled.event_macro_text


def test_macro_compiler_keeps_existing_pair_and_expression_conditions_unchanged():
    compiler = MacroCompiler()

    compiled = compiler.compile(
        macros=[],
        event_macros=[],
        automacros=[
            EventAutomacro(
                name="compat_conditions",
                conditions=["OnCharLogIn 1", "BaseLevel >= 1"],
                call="compat_handler",
                parameters={"priority": "0"},
            )
        ],
    )

    assert "    OnCharLogIn 1\n" in compiled.event_macro_text
    assert "    BaseLevel >= 1\n" in compiled.event_macro_text
