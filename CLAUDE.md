@AGENTS.md

# Claude Code integration

For automasking tasks, use the project `automask` MCP tools instead of ad hoc
Python or a shell interface. The MCP server keeps profiles, selections, and
pipelines alive behind short handles for the lifetime of this Claude Code
session. Run profiles retain their existing disk cache; selection and pipeline
handles do not survive an MCP server restart.

Call `automask_catalog` when exact selection, statistic, regularizer, or pipeline
parameter shapes are needed. Keep raw experiment and calibration data read-only.
