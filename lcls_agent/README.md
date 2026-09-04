# Minimal LCLS agent host

`lcls-agent` runs one bounded Claude Agent SDK task from the LCLS repository
root. It records the request, streamed assistant response, complete SDK event
stream, terminal result, cost, and session ID under `outputs/agent_runs/`.

Activate the shared environment before using it:

```bash
source psana_env.sh
lcls-agent doctor
```

The data backend is inherited from `psana_env.local`, or can be selected for a
bounded run with `--backend local` or `--backend slac`. The MCP subprocess gets
the same explicit backend.

## Direct Claude Code use

The checked-in `.mcp.json` lets Claude Code act as the interactive LCLS agent
host and connect directly to the `automask` MCP server. Launch it from the
activated psana environment so the configured `python` command resolves to the
correct interpreter:

```bash
source psana_env.sh

CLAUDE_BIN="$(python -c 'from pathlib import Path; import claude_agent_sdk; print(Path(claude_agent_sdk.__file__).parent / "_bundled/claude")')"
"$CLAUDE_BIN" mcp list
"$CLAUDE_BIN"
```

Approve the project-scoped `automask` server when Claude Code first prompts. If
needed, use `/mcp` inside Claude Code to inspect or approve it. Each Claude Code
server process gets a unique work directory under
`outputs/claude_sessions/<YYYY-MM-DD>/<timestamp>-<pid>/automask`; its handles last until that
server exits. `CLAUDE.md` imports the repository guidance from `AGENTS.md`.

The `lcls-agent` command remains available for bounded, non-interactive SDK
runs; Claude Code does not invoke it.

Authentication is inherited from the shell or an existing Claude login. The
host deliberately does not read a repository `.env` file or personal Claude
settings.

The default `auto` permission mode exposes Read, Glob, Grep, Edit, Write, Bash,
and the repository's allowlisted `automask` skill. Read/search and that exact
skill are pre-approved; mutations and Bash operations are evaluated by the SDK
permission classifier:

```bash
lcls-agent run "Inspect this repository and summarize its masking architecture" \
  --max-turns 5 --max-budget-usd 0.25
```

If auto mode is unavailable, `dontAsk` is the simple restrictive fallback. It
exposes Read, Glob, Grep, and the `automask` skill, but no mutation or Bash
tools:

```bash
lcls-agent run "Review automask/masking.py" --permission-mode dontAsk
```

The host loads only project settings and the repository-local skill at
`.claude/skills/automask/`; it does not load user or local settings. The skill
provides scientific decision guidance while `docs/AUTOMASK.md` and the
`automask_catalog` tool remain the authoritative library and capability contracts.

Each agent run launches a standalone `automask` MCP server over stdio. That
process owns one in-memory `Session`, so profile, selection, and pipeline handles
remain valid for the run and disappear when it ends. Run profiles retain their
existing disk cache; selections and pipelines are not serialized.

`--max-budget-usd` is a stopping threshold rather than an exact charge cap.
The SDK checks it between model calls, so the final call can exceed the supplied
value. Use a lower threshold, a cheaper `--model`, and a narrowly scoped prompt
when cost predictability matters. A budget-stopped run exits nonzero but still
retains any streamed answer in `response.md`.
