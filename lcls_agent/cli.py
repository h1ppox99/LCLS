"""Command-line interface for the minimal LCLS agent host."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from lcls_agent.config import HostConfig, REPO_ROOT, SKILL_NAMES
from lcls_agent.runtime import run_agent


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def doctor() -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python", sys.version_info >= (3, 11), sys.version.split()[0]))
    for module_name, distribution in (
        ("automask", "automask"),
        ("claude_agent_sdk", "claude-agent-sdk"),
        ("mcp", "mcp"),
        ("pytest", "pytest"),
    ):
        try:
            importlib.import_module(module_name)
            checks.append((module_name, True, _version(distribution)))
        except Exception as exc:
            checks.append((module_name, False, f"{type(exc).__name__}: {exc}"))

    cli_path = None
    try:
        sdk = importlib.import_module("claude_agent_sdk")
        cli_path = Path(sdk.__file__).resolve().parent / "_bundled" / "claude"
        checks.append(("bundled Claude CLI", cli_path.is_file(), str(cli_path)))
    except Exception as exc:
        checks.append(("bundled Claude CLI", False, str(exc)))

    for skill_name in SKILL_NAMES:
        skill_path = REPO_ROOT / ".claude" / "skills" / skill_name / "SKILL.md"
        checks.append((f"skill {skill_name}", skill_path.is_file(), str(skill_path)))

    try:
        importlib.import_module("psana")
        checks.append(("psana", True, str(Path(sys.executable).resolve())))
    except Exception as exc:
        checks.append(("psana", False, f"{type(exc).__name__}: {exc}"))

    auth_names = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    )
    detected = [name for name in auth_names if os.environ.get(name)]
    auth_ok = bool(detected)
    auth_detail = ", ".join(detected)
    if not auth_ok and cli_path is not None and cli_path.is_file():
        try:
            completed = subprocess.run(
                [str(cli_path), "auth", "status", "--json"],
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            status = json.loads(completed.stdout) if completed.returncode == 0 else {}
            auth_ok = bool(status.get("loggedIn"))
            if auth_ok:
                method = status.get("authMethod", "Claude login")
                subscription = status.get("subscriptionType")
                auth_detail = (
                    f"{method} ({subscription})" if subscription else str(method)
                )
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass
    if not auth_detail:
        auth_detail = "no environment credential or Claude login detected"
    checks.append(("authentication", auth_ok, auth_detail))

    required = {
        "python",
        "automask",
        "claude_agent_sdk",
        "mcp",
        "pytest",
        "bundled Claude CLI",
    }
    required.update(f"skill {name}" for name in SKILL_NAMES)
    failed = False
    for name, ok, detail in checks:
        label = "ok" if ok else "warn"
        print(f"[{label}] {name}: {detail}")
        if name in required and not ok:
            failed = True
    print(f"[ok] repository: {REPO_ROOT}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcls-agent",
        description="Minimal Claude Agent SDK host for the LCLS workspace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "doctor", help="check the local runtime without calling a model"
    )

    run_parser = subparsers.add_parser("run", help="run one bounded agent task")
    run_parser.add_argument("prompt", help="task for the workspace agent")
    run_parser.add_argument(
        "--model", default=None, help="model ID; otherwise use environment/default"
    )
    run_parser.add_argument(
        "--permission-mode",
        choices=("auto", "dontAsk"),
        default=os.environ.get("LCLS_AGENT_PERMISSION_MODE", "auto"),
        help="auto classifies mutations; dontAsk exposes read tools only",
    )
    run_parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("LCLS_AGENT_MAX_TURNS", "20")),
    )
    run_parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=float(os.environ.get("LCLS_AGENT_MAX_BUDGET_USD", "2.0")),
    )
    run_parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()

    try:
        config = HostConfig(
            model=args.model,
            permission_mode=args.permission_mode,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
        )
        completed = asyncio.run(
            run_agent(args.prompt, config, output_dir=args.output_dir)
        )
    except Exception as exc:
        print(f"lcls-agent: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"[artifacts] {completed.workdir}")
    result = completed.result
    cost = getattr(result, "total_cost_usd", None)
    if cost is not None:
        print(f"[cost] ${cost:.4f}")
    if not completed.succeeded:
        subtype = getattr(result, "subtype", None)
        failure_text = str(getattr(result, "result", "") or "").lower()
        if subtype == "error_max_budget_usd":
            print(
                "[hint] --max-budget-usd is a stopping threshold, not an exact "
                "cap; one model call can carry the final cost past it. Use a "
                "lower threshold, a cheaper model, or a narrower task.",
                file=sys.stderr,
            )
        elif "not logged in" in failure_text:
            print(
                "[hint] Authenticate the bundled Claude runtime or export the "
                "gateway credentials before retrying.",
                file=sys.stderr,
            )
        elif config.permission_mode == "auto" and "permission" in failure_text:
            print(
                "[hint] Retry with --permission-mode dontAsk if auto mode is "
                "unavailable for this account.",
                file=sys.stderr,
            )
    return 0 if completed.succeeded else 1
