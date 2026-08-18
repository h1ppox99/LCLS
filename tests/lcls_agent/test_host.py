from __future__ import annotations

import asyncio
import io
import json

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from lcls_agent.cli import build_parser
from lcls_agent.config import HostConfig, READ_TOOLS, WORKSPACE_TOOLS
from lcls_agent.runtime import run_agent


def test_auto_mode_exposes_workspace_tools_but_only_preapproves_reads(tmp_path):
    options = HostConfig(root=tmp_path, permission_mode="auto").sdk_options()

    assert options.tools == list(WORKSPACE_TOOLS)
    assert options.allowed_tools == list(READ_TOOLS)
    assert options.permission_mode == "auto"
    assert options.setting_sources == []
    assert options.skills == []


def test_dont_ask_mode_is_read_only(tmp_path):
    options = HostConfig(root=tmp_path, permission_mode="dontAsk").sdk_options()

    assert options.tools == list(READ_TOOLS)
    assert options.allowed_tools == list(READ_TOOLS)
    assert options.permission_mode == "dontAsk"


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


def test_runtime_records_success_without_a_model_call(tmp_path):
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

    output_dir = tmp_path / "recorded-run"
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
    request = json.loads((output_dir / "request.json").read_text())
    result = json.loads((output_dir / "result.json").read_text())
    events = (output_dir / "events.jsonl").read_text().splitlines()
    assert request["permission_mode"] == "auto"
    assert result["session_id"] == "session-test"
    assert result["total_cost_usd"] == 0.01
    assert len(events) == 2
