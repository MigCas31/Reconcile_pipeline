"""Tests for the Gemini agent shim.

We do NOT call the real model. Instead we exercise the failure paths
(missing key, missing dep) and the chat() function with a stub `google`
package injected into sys.modules so the SDK adapter receives a fake
response shape.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


def _purge_agent_module() -> None:
    sys.modules.pop("reconcile_tiers.gemini_agent", None)


def test_missing_key_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _purge_agent_module()
    from reconcile_tiers import gemini_agent

    with pytest.raises(gemini_agent.GeminiUnavailable, match="GEMINI_API_KEY"):
        gemini_agent.chat("test")


def test_missing_sdk_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    # Force the import to fail.
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.genai", None)
    _purge_agent_module()
    from reconcile_tiers import gemini_agent

    with pytest.raises(gemini_agent.GeminiUnavailable, match="google-genai"):
        gemini_agent.chat("test")


def _install_fake_genai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_text: str = "All good — nothing to do.",
    fake_calls: list[dict[str, Any]] | None = None,
) -> None:
    """Inject a stub google.genai package shaped to satisfy the agent."""
    fake_calls = fake_calls or []

    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")

    class _AutomaticConfig:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    class _GenerateContentConfig:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    class _Part:
        def __init__(self, name: str, args: dict[str, Any]) -> None:
            self.function_call = types.SimpleNamespace(name=name, args=args)

    class _Content:
        def __init__(self, parts: list[Any]) -> None:
            self.parts = parts

    class _Response:
        def __init__(self) -> None:
            self.text = final_text
            self.automatic_function_calling_history = [
                _Content(
                    parts=[_Part(call["name"], call["args"]) for call in fake_calls]
                )
            ]

    class _Models:
        def generate_content(
            self, *, model: str, contents: str, config: Any
        ) -> _Response:
            assert isinstance(model, str)
            assert "Selected element context" in contents or contents
            return _Response()

    class _Client:
        def __init__(self, *, api_key: str | None = None) -> None:
            assert api_key
            self.models = _Models()

    genai_mod.Client = _Client  # type: ignore[attr-defined]
    types_mod.GenerateContentConfig = _GenerateContentConfig  # type: ignore[attr-defined]
    types_mod.AutomaticFunctionCallingConfig = _AutomaticConfig  # type: ignore[attr-defined]
    genai_mod.types = types_mod  # type: ignore[attr-defined]
    google_mod.genai = genai_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)


def test_chat_returns_final_text_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    _install_fake_genai(
        monkeypatch,
        final_text="Ran inspect_building; suggest validate_building next.",
        fake_calls=[
            {"name": "inspect_building", "args": {"uuid": "abc"}},
        ],
    )
    _purge_agent_module()
    from reconcile_tiers import gemini_agent

    out = gemini_agent.chat("Look at this building", targets=[])
    assert out["final_text"] == "Ran inspect_building; suggest validate_building next."
    assert out["tool_calls"] == [{"name": "inspect_building", "args": {"uuid": "abc"}}]
    assert "model" in out


def test_chat_handles_missing_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    _install_fake_genai(monkeypatch, final_text="")
    _purge_agent_module()
    from reconcile_tiers import gemini_agent

    out = gemini_agent.chat("hello", targets=[])
    assert out["final_text"] == ""


def test_chat_includes_target_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model should receive a JSON block describing the selected elements."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    captured: dict[str, Any] = {}

    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")

    class _Config:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    class _AutoConfig:
        def __init__(self, **kw: Any) -> None:
            pass

    class _Models:
        def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
            captured["contents"] = contents
            return types.SimpleNamespace(
                text="ok",
                automatic_function_calling_history=[],
            )

    class _Client:
        def __init__(self, *, api_key: str | None = None) -> None:
            self.models = _Models()

    genai_mod.Client = _Client  # type: ignore[attr-defined]
    types_mod.GenerateContentConfig = _Config  # type: ignore[attr-defined]
    types_mod.AutomaticFunctionCallingConfig = _AutoConfig  # type: ignore[attr-defined]
    genai_mod.types = types_mod  # type: ignore[attr-defined]
    google_mod.genai = genai_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    _purge_agent_module()
    from reconcile_tiers import gemini_agent

    # An invalid locator just lands as an "error" entry in the JSON block —
    # the test cares about KNOWN_PIVOTS being inserted into the prompt.
    gemini_agent.chat("inspect this", targets=["bogus::tier-ceiling-flat::0"])
    assert "Known abandoned approaches" in captured["contents"]
    assert "AZIMUTH_FILTER_THRESHOLD" in captured["contents"]
