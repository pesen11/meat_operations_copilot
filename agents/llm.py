"""
The single place this project talks to Claude.

Design notes that matter:

1. **The LLM never computes numbers.** Nothing in this module does
   arithmetic, and nothing downstream is allowed to ask Claude for a
   figure. Claude is used for exactly two jobs: turning a natural-language
   question into a structured plan (which tool to call with which
   arguments), and narrating numbers that deterministic Python already
   computed. evals/number_guard.py enforces the second half.

2. **Offline mode is first-class, not a test hack.** With no credentials
   configured, `is_live()` is False and callers fall back to deterministic
   Python paths (keyword intent parsing, template narration). The whole
   agent graph therefore runs, and is tested, on a machine with no API key
   - the narration is plainer, but every number in it is identical, because
   the numbers never came from the model in the first place.

3. **Prompt caching.** System prompts are sent as a cached block
   (`cache_control: ephemeral`). System text must therefore stay byte-stable
   across calls - never interpolate a timestamp, a question, or a row of
   data into a system prompt. Volatile content goes in the user turn, after
   the cache breakpoint. Check `last_usage()` if hit rates look wrong.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Model choice, per Anthropic's current lineup:
#   - Opus 5 for planning and narration, where a wrong reading of the
#     question is expensive and the call volume is low.
#   - Haiku 4.5 for the RAG relevance filter, which runs once per retrieved
#     chunk and is a cheap binary judgement (see rag/reranker.py).
OPS_MODEL = "claude-opus-5"
FAST_MODEL = "claude-haiku-4-5"

DEFAULT_MAX_TOKENS = 16000

_client_lock = threading.Lock()
_client: Any = None


def _credentials_present() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def is_live() -> bool:
    """True when real API calls should be made.

    OPS_COPILOT_LLM=stub forces offline mode even when a key is present -
    used by the test suite so a developer with a working key still gets
    deterministic, free test runs.
    """
    if os.getenv("OPS_COPILOT_LLM", "").lower() == "stub":
        return False
    return _credentials_present()


def get_client():
    """Lazily construct the Anthropic client (resolves credentials from the
    environment or an `ant auth login` profile)."""
    global _client
    with _client_lock:
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic()
        return _client


@dataclass
class LLMResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Optional[dict] = None
    live: bool = True


_last_usage: dict = {}


def last_usage() -> dict:
    """Usage from the most recent live call, including cache counters. If
    `cache_read_input_tokens` stays at 0 across repeated calls, something is
    invalidating the cached system prefix."""
    return dict(_last_usage)


def _record_usage(response) -> dict:
    global _last_usage
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    _last_usage = {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
    }
    return dict(_last_usage)


def _system_blocks(system: str) -> list[dict]:
    """System prompt as a single cached block. Keep `system` byte-stable."""
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def complete(system: str, user: str, *, model: str = OPS_MODEL,
             max_tokens: int = DEFAULT_MAX_TOKENS,
             effort: str = "medium",
             prefill_offline: str = "") -> LLMResult:
    """
    One non-streaming completion. Returns `prefill_offline` verbatim when
    offline, so callers always get an LLMResult and never have to branch on
    credentials themselves.
    """
    if not is_live():
        return LLMResult(text=prefill_offline, live=False)

    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_system_blocks(system),
        output_config={"effort": effort},
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        raise RuntimeError(
            f"Claude declined this request (category={getattr(detail, 'category', None)}). "
            "Nothing was generated."
        )
    text = "".join(b.text for b in response.content if b.type == "text")
    return LLMResult(text=text, stop_reason=response.stop_reason,
                     usage=_record_usage(response), live=True)


def complete_json(system: str, user: str, schema: dict, *,
                  model: str = OPS_MODEL, max_tokens: int = 4096,
                  effort: str = "medium",
                  offline_value: Optional[dict] = None) -> dict:
    """
    Structured output: constrains the response to `schema` via
    `output_config.format`, so the caller gets a dict it can rely on rather
    than a JSON-shaped string it has to defend against.
    """
    if not is_live():
        return dict(offline_value or {})

    response = get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_system_blocks(system),
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": schema},
        },
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request; no structured output produced.")
    _record_usage(response)
    text = "".join(b.text for b in response.content if b.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Structured output was not valid JSON: {text[:400]}") from exc


# ---------------------------------------------------------------------------
# LangChain tool -> Anthropic tool schema
# ---------------------------------------------------------------------------

def to_anthropic_tools(tools: list) -> list[dict]:
    """Convert LangChain @tool objects into Anthropic tool definitions.

    `strict: true` is set so tool arguments are guaranteed to validate
    against the schema - the tool functions do catalog lookups by SKU and a
    malformed argument would surface as a confusing KeyError rather than a
    clean validation failure.
    """
    out = []
    for t in tools:
        schema = t.args_schema.model_json_schema() if t.args_schema else {
            "type": "object", "properties": {}}
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("required", [])
        schema["additionalProperties"] = False
        out.append({
            "name": t.name,
            "description": t.description,
            "input_schema": schema,
            "strict": True,
        })
    return out


def run_tool_loop(system: str, user: str, tools: list, *,
                  model: str = OPS_MODEL, max_iterations: int = 6,
                  max_tokens: int = DEFAULT_MAX_TOKENS,
                  on_tool_call: Optional[Callable[[str, dict, Any], None]] = None
                  ) -> LLMResult:
    """
    Manual agentic loop: Claude picks tools, we execute them, repeat until
    it stops asking. The loop is written out rather than delegated to the
    SDK's tool runner because every tool result is recorded verbatim for the
    number guard in evals/number_guard.py - the recommendation narration is
    later checked against exactly these values.

    Returns an LLMResult whose `tool_calls` lists every (name, args, result)
    triple in order. Offline, this returns immediately with no tool calls;
    callers that need tool results without an LLM should call the tools
    directly, which is what the agent nodes do.
    """
    if not is_live():
        return LLMResult(text="", live=False)

    by_name = {t.name: t for t in tools}
    anthropic_tools = to_anthropic_tools(tools)
    messages: list[dict] = [{"role": "user", "content": user}]
    calls: list[dict] = []
    text_out = ""

    for _ in range(max_iterations):
        response = get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_system_blocks(system),
            tools=anthropic_tools,
            messages=messages,
        )
        _record_usage(response)
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined this request.")

        text_out = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "tool_use":
            return LLMResult(text=text_out, tool_calls=calls,
                             stop_reason=response.stop_reason, usage=last_usage())

        messages.append({"role": "assistant", "content": response.content})

        # Every tool_result for one assistant turn goes back in a SINGLE user
        # message - splitting them trains the model out of parallel calls.
        results: list[dict] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool = by_name.get(block.name)
            if tool is None:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Unknown tool '{block.name}'", "is_error": True})
                continue
            try:
                value = tool.invoke(dict(block.input))
                calls.append({"name": block.name, "args": dict(block.input), "result": value})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(value, default=str)})
                if on_tool_call:
                    on_tool_call(block.name, dict(block.input), value)
            except Exception as exc:  # surfaced to the model so it can correct itself
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"{type(exc).__name__}: {exc}", "is_error": True})
        messages.append({"role": "user", "content": results})

    return LLMResult(text=text_out, tool_calls=calls,
                     stop_reason="max_iterations", usage=last_usage())
