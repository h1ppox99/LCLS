from __future__ import annotations

import asyncio
import io

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from lcls_agent.cli import build_parser, main
from lcls_agent.config import (
    REPO_ROOT,
    HostConfig,
    READ_TOOLS,
    SKILL_NAMES,
    WORKSPACE_TOOLS,
)
from lcls_agent.runtime import run_agent


def test_auto_mode_exposes_workspace_tools_but_only_preapproves_reads(tmp_path):
    options = HostConfig(root=tmp_path, permission_mode="auto").sdk_options()

    assert options.tools == [*WORKSPACE_TOOLS, "Skill"]
    assert options.allowed_tools == list(READ_TOOLS)
    assert options.permission_mode == "auto"
    assert options.setting_sources == ["project"]
    assert options.skills == list(SKILL_NAMES)


def test_dont_ask_mode_is_read_only(tmp_path):
    options = HostConfig(root=tmp_path, permission_mode="dontAsk").sdk_options()

    assert options.tools == [*READ_TOOLS, "Skill"]
    assert options.allowed_tools == list(READ_TOOLS)
    assert options.permission_mode == "dontAsk"


def test_allowlisted_project_skills_exist():
    for name in SKILL_NAMES:
        path = REPO_ROOT / ".claude" / "skills" / name / "SKILL.md"

        assert path.is_file()
        assert path.read_text().startswith(f"---\nname: {name}\n")


def test_invalid_limits_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="max_turns"):
        HostConfig(root=tmp_path, max_turns=0)
    with pytest.raises(ValueError, match="max_budget_usd"):
        HostConfig(root=tmp_path, max_budget_usd=0)


def test_cli_defaults_to_auto_mode():
    args = build_parser().parse_args(["run", "inspect the repository"])

    assert args.permission_mode == "auto"
    assert args.max_turns == 20
    assert args.max_budget_usd == 2.0


def test_cli_explains_soft_budget_stop(monkeypatch, tmp_path, capsys):
    async def fake_run_agent(prompt, config, output_dir=None):
        from types import SimpleNamespace

        from lcls_agent.runtime import AgentResult

        return AgentResult(
            workdir=tmp_path,
            result=SimpleNamespace(
                subtype="error_max_budget_usd",
                is_error=True,
                total_cost_usd=0.31,
                result="",
            ),
        )

    monkeypatch.setattr("lcls_agent.cli.run_agent", fake_run_agent)

    status = main(["run", "brief task", "--max-budget-usd", "0.20"])

    assert status == 1
    captured = capsys.readouterr()
    assert "[cost] $0.3100" in captured.out
    assert "stopping threshold, not an exact cap" in captured.err


def test_runtime_streams_answer_and_returns_result(tmp_path):
    async def fake_query(*, prompt, options):
        assert prompt == "summarize"
        assert options.permission_mode == "auto"
        yield AssistantMessage(content=[TextBlock("done")], model="test-model")
        yield ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=5,
            is_error=False,
            num_turns=1,
            session_id="session-test",
            total_cost_usd=0.01,
            result="done",
        )

    output_dir = tmp_path / "run"
    stream = io.StringIO()
    completed = asyncio.run(
        run_agent(
            "summarize",
            HostConfig(root=tmp_path),
            output_dir=output_dir,
            stream=stream,
            query_fn=fake_query,
        )
    )

    assert completed.succeeded
    assert stream.getvalue() == "done\n"
    assert completed.result.session_id == "session-test"
    assert completed.result.total_cost_usd == 0.01
    assert completed.workdir == output_dir.resolve()
    # A run that invokes no automask tool leaves no working directory behind.
    assert not output_dir.exists()
