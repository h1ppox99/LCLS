"""Run one agent task: wire the automask tools, stream the answer, report cost.

The Claude Agent SDK already persists the session transcript, and the terminal
already shows the streamed answer, so nothing here re-records either. The only
thing written is whatever the automask tools deliberately produce (masks,
profiles, previews) under the run's working directory.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from lcls_agent.config import HostConfig


@dataclass(frozen=True)
class AgentResult:
    """Where the run's artifacts landed, and the SDK's final result message."""

    workdir: Path
    result: Any | None  # claude_agent_sdk.ResultMessage, or None if none arrived

    @property
    def succeeded(self) -> bool:
        message = self.result
        return (
            message is not None
            and message.subtype == "success"
            and not message.is_error
        )


def _default_workdir(root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / "outputs" / "agent_runs" / stamp


def _display(message: Any, stream: TextIO) -> None:
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
    query_fn=None,
) -> AgentResult:
    """Run one bounded agent task, streaming its answer to ``stream``."""
    from claude_agent_sdk import ResultMessage, query

    from lcls_agent.tools import SERVER_NAME, tool_names

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    workdir = (output_dir or _default_workdir(config.root)).expanduser().resolve()
    tools, read_only = tool_names()
    server = {
        "type": "stdio",
        "command": sys.executable,
        "args": [
            "-m",
            "lcls_agent.mcp_server",
            "--workdir",
            str(workdir / SERVER_NAME),
            "--backend",
            config.backend,
        ],
    }
    options = config.sdk_options(
        automask_server=server,
        automask_tools=tools,
        automask_read_only=read_only,
    )

    final = None
    async for message in (query_fn or query)(prompt=prompt, options=options):
        _display(message, stream)
        if isinstance(message, ResultMessage):
            final = message
    return AgentResult(workdir=workdir, result=final)
