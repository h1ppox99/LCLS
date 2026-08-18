"""Runtime configuration for the minimal LCLS agent host."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PermissionMode = Literal["auto", "dontAsk"]

REPO_ROOT = Path(__file__).resolve().parent.parent
READ_TOOLS = ("Read", "Glob", "Grep")
WORKSPACE_TOOLS = READ_TOOLS + ("Edit", "Write", "Bash")

SYSTEM_PROMPT = """You are the general LCLS workspace agent for this repository.
Inspect evidence before acting. Treat raw experiment data and calibration data as
read-only. Never expose credentials. Work only on the user's requested task and
report the files changed, commands run, tests performed, and any unresolved risk.
"""


@dataclass(frozen=True)
class HostConfig:
    """Configuration shared by one SDK query and its recorded artifacts."""

    root: Path = REPO_ROOT
    model: str | None = None
    permission_mode: PermissionMode = "auto"
    max_turns: int = 20
    max_budget_usd: float = 2.0

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        object.__setattr__(self, "root", root)
        if not root.is_dir():
            raise ValueError(f"agent root does not exist: {root}")
        if self.permission_mode not in ("auto", "dontAsk"):
            raise ValueError(f"unsupported permission mode: {self.permission_mode}")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")

    @property
    def resolved_model(self) -> str | None:
        return (
            self.model
            or os.environ.get("LCLS_AGENT_MODEL")
            or os.environ.get("ANTHROPIC_MODEL")
        )

    @property
    def tools(self) -> tuple[str, ...]:
        if self.permission_mode == "dontAsk":
            return READ_TOOLS
        return WORKSPACE_TOOLS

    def sdk_options(self):
        """Build options lazily so `doctor` can report a missing SDK cleanly."""
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            tools=list(self.tools),
            allowed_tools=list(READ_TOOLS),
            system_prompt=SYSTEM_PROMPT,
            permission_mode=self.permission_mode,
            cwd=self.root,
            model=self.resolved_model,
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            setting_sources=[],
            skills=[],
        )
