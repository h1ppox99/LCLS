# Minimal LCLS agent host

`lcls-agent` runs one bounded Claude Agent SDK task from the LCLS repository
root. It records the request, complete SDK event stream, terminal result, cost,
and session ID under `outputs/agent_runs/`.

Activate the shared environment before using it:

```bash
source psana_env.sh
lcls-agent doctor
```

Authentication is inherited from the shell or an existing Claude login. The
host deliberately does not read a repository `.env` file or personal Claude
settings.

The default `auto` permission mode exposes Read, Glob, Grep, Edit, Write, and
Bash. Read/search operations are pre-approved; other operations are evaluated
by the SDK permission classifier:

```bash
lcls-agent run "Inspect this repository and summarize its masking architecture" \
  --max-turns 5 --max-budget-usd 0.25
```

If auto mode is unavailable, `dontAsk` is the simple restrictive fallback. It
only exposes and pre-approves Read, Glob, and Grep:

```bash
lcls-agent run "Review automask/masking.py" --permission-mode dontAsk
```

The initial host loads no skills or filesystem settings. Project instructions,
typed automask tools, and masking skills will be added after the authenticated
SDK and Bash smoke tests succeed.
