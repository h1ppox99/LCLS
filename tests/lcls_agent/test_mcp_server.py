from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lcls_agent.config import REPO_ROOT
from lcls_agent.mcp_server import build_parser, default_workdir
from lcls_agent.tools import TOOL_NAMES


def test_direct_launch_gets_a_unique_project_workdir(tmp_path, monkeypatch):
    monkeypatch.setattr("lcls_agent.mcp_server.os.getpid", lambda: 1234)

    workdir = default_workdir(tmp_path)

    sessions_root = tmp_path / "outputs" / "claude_sessions"
    assert workdir.name == "automask"
    assert workdir.parent.name.endswith("-1234")
    day = workdir.parent.parent
    assert day.parent == sessions_root
    assert day.name.replace("-", "") == workdir.parent.name[:8]
    assert build_parser().parse_args([]).workdir is None


def test_project_mcp_config_launches_the_server_module():
    config = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))

    assert config["mcpServers"]["automask"] == {
        "type": "stdio",
        "command": "python",
        "args": ["-m", "lcls_agent.mcp_server"],
    }

    async def exercise():
        server = config["mcpServers"]["automask"]
        params = StdioServerParameters(
            command=server["command"],
            args=server["args"],
        )
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                listed = await client.list_tools()
                assert tuple(tool.name for tool in listed.tools) == TOOL_NAMES

    asyncio.run(exercise())


def test_stdio_server_preserves_tools_and_session_handles(tmp_path):
    async def exercise():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "lcls_agent.mcp_server",
                "--workdir",
                str(tmp_path / "automask"),
            ],
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                listed = await client.list_tools()
                assert tuple(tool.name for tool in listed.tools) == TOOL_NAMES
                selection = next(
                    tool for tool in listed.tools if tool.name == "define_selection"
                )
                assert selection.inputSchema["properties"]["spec"] == {
                    "type": "object",
                    "additionalProperties": True,
                }

                first = await client.call_tool("define_selection", {"spec": {}})
                second = await client.call_tool("define_selection", {"spec": {}})
                assert json.loads(first.content[0].text)["selection"] == "sel-1"
                assert json.loads(second.content[0].text)["selection"] == "sel-2"

                failure = await client.call_tool(
                    "describe_selection",
                    {"profile": "prof-999", "selection": "sel-1"},
                )
                assert failure.isError
                assert "unknown profile handle" in failure.content[0].text

    asyncio.run(exercise())
    assert not (tmp_path / "automask").exists()
