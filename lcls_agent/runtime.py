"""Execute one SDK query and persist a compact, inspectable run record."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from lcls_agent.config import HostConfig


@dataclass(frozen=True)
class AgentRun:
    run_dir: Path
    result: dict[str, Any]

    @property
    def succeeded(self) -> bool:
        return self.result.get("status") == "success"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _create_run_dir(root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        run_dir = output_dir.expanduser().resolve()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = root / "outputs" / "agent_runs" / f"{stamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _display_message(message: Any, stream: TextIO) -> None:
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock

    if not isinstance(message, AssistantMessage):
        return
    for block in message.content:
        if isinstance(block, ToolUseBlock):
            print(f"[tool] {block.name}", file=stream, flush=True)
        elif isinstance(block, TextBlock):
            print(block.text, file=stream, flush=True)


async def run_agent(
    prompt: str,
    config: HostConfig,
    *,
    output_dir: Path | None = None,
    stream: TextIO = sys.stdout,
    query_fn: Callable[..., AsyncIterator[Any]] | None = None,
) -> AgentRun:
    """Run one bounded agent task and record its request, events, and result."""
    from claude_agent_sdk import ResultMessage, query

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    run_dir = _create_run_dir(config.root, output_dir)
    options = config.sdk_options()
    request = {
        "created_at": datetime.now(UTC).isoformat(),
        "prompt": prompt,
        "cwd": config.root,
        "model": config.resolved_model,
        "permission_mode": config.permission_mode,
        "tools": config.tools,
        "auto_approved_tools": list(options.allowed_tools),
        "max_turns": config.max_turns,
        "max_budget_usd": config.max_budget_usd,
    }
    _write_json(run_dir / "request.json", request)

    final_message = None
    failure = None
    selected_query = query_fn or query
    with (run_dir / "events.jsonl").open("x") as events:
        try:
            async for message in selected_query(prompt=prompt, options=options):
                event = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "message_type": type(message).__name__,
                    "message": _jsonable(message),
                }
                events.write(json.dumps(event, sort_keys=True) + "\n")
                events.flush()
                _display_message(message, stream)
                if isinstance(message, ResultMessage):
                    final_message = message
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}

    if final_message is None:
        result = {
            "status": "error",
            "error": failure or {"message": "no result message"},
        }
    else:
        result = _jsonable(final_message)
        result["status"] = (
            "success"
            if final_message.subtype == "success"
            and not final_message.is_error
            and failure is None
            else "error"
        )
        if failure is not None:
            result["exception"] = failure
    _write_json(run_dir / "result.json", result)
    return AgentRun(run_dir=run_dir, result=result)
