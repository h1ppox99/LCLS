"""Automask MCP tools backed by one live :class:`lcls_agent.session.Session`.

These are the agent's door into automask. Each tool is a thin async wrapper that
calls a ``Session`` method and returns its JSON summary; the ``Session`` holds the
live ``RunProfile``/``ShotSelection``/``Pipeline`` objects, so the agent only ever
passes short handles (``prof-1``, ``sel-2``) and inline parameter objects. The SDK
validates each call against the ``input_schema`` declared here, which is what
replaces the old hand-written recipe-file validation layer.

Selection and pipeline parameter shapes are documented by the ``automask_catalog``
tool; both accept the same field-native dicts the library dataclasses take.
"""

from __future__ import annotations

import json
from typing import Any

from lcls_agent.session import Session

SERVER_NAME = "automask"
TOOL_NAMES = (
    "automask_catalog",
    "inspect_run",
    "load_profile",
    "define_selection",
    "describe_selection",
    "define_pipeline",
    "preview_selection",
    "build_mask",
    "validate_mask",
)
READ_ONLY_TOOL_NAMES = (
    "automask_catalog",
    "load_profile",
    "describe_selection",
)


def tool_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    def qualify(name: str) -> str:
        return f"mcp__{SERVER_NAME}__{name}"

    return tuple(map(qualify, TOOL_NAMES)), tuple(map(qualify, READ_ONLY_TOOL_NAMES))


def _text(payload: Any, *, is_error: bool = False) -> dict:
    result: dict[str, Any] = {
        "content": [{"type": "text", "text": json.dumps(payload, default=str)}]
    }
    if is_error:
        result["is_error"] = True
    return result


def _guard(call):
    """Turn a Session call into a tool result, surfacing errors to the model."""
    try:
        return _text(call())
    except Exception as error:  # noqa: BLE001 - report, don't crash the session
        return _text({"error": f"{type(error).__name__}: {error}"}, is_error=True)


def build_automask_server(session: Session):
    """Build the MCP implementation used by the standalone stdio process."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    _object = {"type": "object", "additionalProperties": True}

    @tool(
        "automask_catalog",
        "List registered statistics, regularizers, selection operators, and the "
        "exact parameter shapes accepted by define_selection/define_pipeline.",
        {"type": "object", "properties": {}},
    )
    async def catalog(args):
        return _guard(session.catalog)

    @tool(
        "inspect_run",
        "Profile one experiment run (reads psana), cache it, and return a profile "
        "handle plus a summary. Use load_profile to reuse an earlier cache.",
        {
            "type": "object",
            "properties": {
                "run": {"type": "integer"},
                "experiment": {"type": "string"},
                "detector": {"type": "string"},
                "detector_source": {"type": "string"},
                "detector_calib_type": {"type": "string"},
                "max_events": {"type": "integer"},
            },
            "required": ["run"],
        },
    )
    async def inspect_run(args):
        return _guard(
            lambda: session.inspect(
                int(args["run"]),
                experiment=args.get("experiment"),
                detector=args.get("detector"),
                detector_source=args.get("detector_source"),
                detector_calib_type=args.get("detector_calib_type"),
                max_events=args.get("max_events"),
            )
        )

    @tool(
        "load_profile",
        "Reuse a run profile cached by an earlier session (by run number) into a "
        "fresh profile handle, skipping the psana read. Errors if not cached.",
        {
            "type": "object",
            "properties": {"run": {"type": "integer"}},
            "required": ["run"],
        },
    )
    async def load_profile(args):
        return _guard(lambda: session.load_profile(int(args["run"])))

    @tool(
        "define_selection",
        "Register a shot selection from inline params (where/trim/n_shots/"
        "normalization); returns a selection handle. Empty selects every shot.",
        {"type": "object", "properties": {"spec": _object}},
    )
    async def define_selection(args):
        return _guard(lambda: session.define_selection(args.get("spec")))

    @tool(
        "describe_selection",
        "Stage-by-stage shot counts for a selection handle against a profile "
        "handle, without decoding frames.",
        {
            "type": "object",
            "properties": {
                "profile": {"type": "string"},
                "selection": {"type": "string"},
            },
            "required": ["profile", "selection"],
        },
    )
    async def describe_selection(args):
        return _guard(
            lambda: session.describe_selection(args["profile"], args["selection"])
        )

    @tool(
        "define_pipeline",
        "Register a mask pipeline from inline channel params; returns a pipeline "
        "handle. Omit spec for the production recipe.",
        {"type": "object", "properties": {"spec": _object}},
    )
    async def define_pipeline(args):
        return _guard(lambda: session.define_pipeline(args.get("spec")))

    @tool(
        "preview_selection",
        "Materialize and render one selected-shot reduction (mean/std/median/mad); "
        "persists the array and a PNG, returns image statistics.",
        {
            "type": "object",
            "properties": {
                "profile": {"type": "string"},
                "selection": {"type": "string"},
                "reduction": {"type": "string"},
            },
            "required": ["profile", "selection"],
        },
    )
    async def preview_selection(args):
        return _guard(
            lambda: session.preview_selection(
                args["profile"],
                args["selection"],
                reduction=args.get("reduction", "mean"),
            )
        )

    @tool(
        "build_mask",
        "Run a profile + selection + pipeline into a final boolean mask "
        "deliverable (mask.npy + overview.png) and return its statistics.",
        {
            "type": "object",
            "properties": {
                "profile": {"type": "string"},
                "selection": {"type": "string"},
                "pipeline": {"type": "string"},
                "gain": {"type": "integer"},
            },
            "required": ["profile", "selection", "pipeline"],
        },
    )
    async def build_mask(args):
        return _guard(
            lambda: session.build_mask(
                args["profile"],
                args["selection"],
                args["pipeline"],
                gain=int(args.get("gain", 0)),
            )
        )

    @tool(
        "validate_mask",
        "Run declared data/model perturbations on a pipeline and persist the "
        "validation report; returns a handle for the recommended pipeline.",
        {
            "type": "object",
            "properties": {
                "profile": {"type": "string"},
                "selection": {"type": "string"},
                "pipeline": {"type": "string"},
                "design": _object,
            },
            "required": ["profile", "selection", "pipeline"],
        },
    )
    async def validate_mask(args):
        return _guard(
            lambda: session.validate_mask(
                args["profile"],
                args["selection"],
                args["pipeline"],
                design=args.get("design"),
            )
        )

    tools = [
        catalog,
        inspect_run,
        load_profile,
        define_selection,
        describe_selection,
        define_pipeline,
        preview_selection,
        build_mask,
        validate_mask,
    ]
    server = create_sdk_mcp_server(SERVER_NAME, tools=tools)
    names, read_only = tool_names()
    return server, names, read_only
