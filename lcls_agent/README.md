# Minimal LCLS agent host

`lcls-agent` runs one bounded Claude Agent SDK task from the LCLS repository
root. It records the request, streamed assistant response, complete SDK event
stream, terminal result, cost, and session ID under `outputs/agent_runs/`.

Activate the shared environment before using it:

```bash
source psana_env.sh
lcls-agent doctor
```

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

`--max-budget-usd` is a stopping threshold rather than an exact charge cap.
The SDK checks it between model calls, so the final call can exceed the supplied
value. Use a lower threshold, a cheaper `--model`, and a narrowly scoped prompt
when cost predictability matters. A budget-stopped run exits nonzero but still
retains any streamed answer in `response.md`.
