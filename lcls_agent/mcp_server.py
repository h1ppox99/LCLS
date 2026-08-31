"""Standalone stdio transport for one in-memory automask Session."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from lcls_agent.config import REPO_ROOT
from lcls_agent.session import Session
from lcls_agent.tools import build_automask_server


def default_workdir(root: str | Path = REPO_ROOT) -> Path:
    now = datetime.now(UTC)
    day = now.strftime("%Y-%m-%d")
    session = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    return Path(root) / "outputs" / "claude_sessions" / day / session / "automask"


def build_server(workdir: str | Path):
    """Create the existing MCP tool implementation with a fresh Session."""
    session = Session(workdir)
    server_config, _, _ = build_automask_server(session)
    return server_config["instance"]


async def serve(workdir: str | Path) -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(workdir)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automask MCP stdio server")
    parser.add_argument("--workdir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workdir = default_workdir() if args.workdir is None else args.workdir
    asyncio.run(serve(workdir.expanduser().resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
