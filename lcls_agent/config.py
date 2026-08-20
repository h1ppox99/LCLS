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
SKILL_NAMES = ("automask",)
SKILL_TOOL = "Skill"

SYSTEM_PROMPT = """You are the general LCLS workspace agent for this repository.
Inspect evidence before acting. Treat raw experiment data and calibration data as
read-only. Never expose credentials. Work only on the requested task.

For automasking work, use the `automask` MCP tools rather than ad hoc Python
or a shell CLI. They keep run profiles, selections, and pipelines alive as objects
you name by short handle (`prof-1`, `sel-2`, `pipe-1`); pass those handles between
tools, and pass parameters inline as objects. Typical flow: `inspect_run` (or
`load_profile` to reuse a cache) -> `define_selection` -> `describe_selection`/
`preview_selection` -> `define_pipeline` -> `build_mask` -> `validate_mask`. Call
`automask_catalog` first when you need the exact statistic, regularizer, or
parameter shapes; skip it for inspection- or selection-only work.

Answer concisely and report handles produced, tools run, artifacts written, and
unresolved risks.
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
        base = READ_TOOLS if self.permission_mode == "dontAsk" else WORKSPACE_TOOLS
        return base + (SKILL_TOOL,)

    def sdk_options(
        self,
        *,
        automask_server=None,
        automask_tools: tuple[str, ...] = (),
        automask_read_only: tuple[str, ...] = (),
    ):
        """Build options lazily so `doctor` can report a missing SDK cleanly.

        The automask MCP tools only ever write inside the run's own
        working directory (guarded against protected data trees), so they are
        auto-approved: the full set in ``auto`` mode, the read-only subset in
        ``dontAsk`` mode, which also hides the mutating ones entirely.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        from lcls_agent.tools import SERVER_NAME

        if self.permission_mode == "dontAsk":
            exposed = tuple(automask_read_only)
            auto_approved = READ_TOOLS + tuple(automask_read_only)
        else:
            exposed = tuple(automask_tools)
            auto_approved = READ_TOOLS + tuple(automask_tools)
        return ClaudeAgentOptions(
            tools=list(self.tools) + list(exposed),
            allowed_tools=list(auto_approved),
            mcp_servers=(
                {SERVER_NAME: automask_server} if automask_server is not None else {}
            ),
            system_prompt=SYSTEM_PROMPT,
            permission_mode=self.permission_mode,
            cwd=self.root,
            model=self.resolved_model,
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            setting_sources=["project"],
            skills=list(SKILL_NAMES),
        )
