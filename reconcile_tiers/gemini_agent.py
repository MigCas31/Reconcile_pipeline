"""Gemini chat → dev_tools.REGISTRY orchestration.

Wraps the ``google-genai`` SDK in automatic-function-calling mode. The model
is given the entire ``dev_tools.REGISTRY`` as tools and a system prompt that
includes ``KNOWN_PIVOTS`` so it doesn't re-suggest abandoned approaches.

Auth: ``GEMINI_API_KEY`` env var (Gemini API direct, no Vertex / ADC).
Model: ``gemini-flash-latest``.

Failure modes are explicit:

- ``google-genai`` not installed → ``GeminiUnavailable("install google-genai")``
- ``GEMINI_API_KEY`` missing → ``GeminiUnavailable("set GEMINI_API_KEY")``
- API error → propagated as ``RuntimeError`` with the reason

The HTTP layer maps these to 503 / 500 + a clear message.
"""

from __future__ import annotations

import json
import os
from typing import Any

from reconcile_tiers import dev_tools, quick_actions

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
MAX_TOOL_ITERATIONS = 6


class GeminiUnavailable(RuntimeError):
    """The Gemini chat surface cannot be reached (missing dep / key)."""


def _ensure_ready() -> tuple[Any, Any]:
    if not os.environ.get("GEMINI_API_KEY"):
        raise GeminiUnavailable(
            "GEMINI_API_KEY not set. Generate a key at "
            "https://aistudio.google.com/apikey and `export GEMINI_API_KEY=...`."
        )
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError as exc:
        raise GeminiUnavailable(
            "google-genai not installed. `pip install '.[gemini]'` to enable."
        ) from exc
    return genai, gtypes


SYSTEM_INSTRUCTION = """\
You are a backend co-pilot for the tirana reconcile_tiers pipeline. You operate
through a small registry of typed tools that wrap CLIs the user already runs by
hand. Be concise; assume the user is the senior engineer maintaining the
pipeline.

Hard rules:
1. Mutating tools (rebuild_*, apply_threshold_tweak, audit_corpus) require
   `confirmed=True`. Always call the tool once with `confirmed=False` so the
   frontend can show a confirmation dialog. Only re-call with `confirmed=True`
   if the user confirms in their next message.
2. tier_payload.json is *derived*; never propose editing it directly. Real
   fixes go through reconcile_tiers/extract, /classify, or /build.
3. Heed the KNOWN_PIVOTS list passed in the user context — those approaches
   were tried and abandoned; do not re-suggest them.
4. Prefer single-building rebuilds over corpus rebuilds when the user is
   debugging one address.
5. If you have insufficient information, prefer reading (inspect_building,
   debug_element, read_tracking_progress) before mutating.

When you've finished, return a short final message summarizing what ran and
the next concrete action the user should take.
"""


def _tool_callables() -> list:
    """Expose dev_tools functions to the SDK's automatic function calling.

    Returns a list of callables; the SDK introspects type hints to build the
    function declarations. We use the underlying functions, not DevTool dataclass
    instances.
    """
    return [tool.fn for tool in dev_tools.REGISTRY.values()]


def _build_user_message(prompt: str, targets: list[str]) -> str:
    """Pack the user's selection context into the first user turn."""
    parts: list[str] = [prompt.strip()]

    target_summaries: list[dict[str, Any]] = []
    for locator in targets:
        try:
            info = quick_actions.element_info(locator)
            target_summaries.append(info)
        except Exception as exc:  # pragma: no cover - best-effort context
            target_summaries.append({"locator": locator, "error": str(exc)})

    if target_summaries:
        parts.append(
            "Selected element context:\n```json\n"
            + json.dumps(target_summaries, indent=2)
            + "\n```"
        )

    parts.append(
        "Known abandoned approaches (do not re-suggest):\n"
        + "\n".join(f"- {p}" for p in dev_tools.KNOWN_PIVOTS)
    )
    return "\n\n".join(parts)


def chat(prompt: str, targets: list[str] | None = None) -> dict[str, Any]:
    """Run one chat turn with automatic function calling. Returns the
    final text plus a record of every tool call made along the way.
    """
    genai, gtypes = _ensure_ready()
    targets = targets or []

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = gtypes.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=_tool_callables(),
        automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(
            maximum_remote_calls=MAX_TOOL_ITERATIONS,
        ),
    )

    user_msg = _build_user_message(prompt, targets)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_msg,
            config=config,
        )
    except Exception as exc:
        raise RuntimeError(f"gemini call failed: {exc}") from exc

    tool_calls = _extract_tool_calls(response)

    final_text = ""
    try:
        final_text = response.text or ""
    except Exception:
        # response.text accessor can raise when there are only function calls
        pass

    return {
        "final_text": final_text.strip(),
        "tool_calls": tool_calls,
        "model": MODEL_NAME,
    }


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Pull function-call records from the SDK's afc_history when available."""
    out: list[dict[str, Any]] = []
    history = getattr(response, "automatic_function_calling_history", None)
    if not history:
        return out
    for content in history:
        for part in getattr(content, "parts", None) or []:
            fc = getattr(part, "function_call", None)
            if fc and getattr(fc, "name", None):
                out.append(
                    {
                        "name": fc.name,
                        "args": dict(getattr(fc, "args", None) or {}),
                    }
                )
    return out


__all__ = ["MODEL_NAME", "GeminiUnavailable", "chat"]
