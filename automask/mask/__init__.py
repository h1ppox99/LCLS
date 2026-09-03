"""mask -- the masking model.

The `Pipeline`/`Channel` framework plus the `stats`, `regularization`, and
`combine` registries it composes into one run-level boolean mask.
"""

from automask.mask.pipeline import (
    Channel,
    Pipeline,
    floor_channels,
    production_pipeline,
)

__all__ = ["Channel", "Pipeline", "floor_channels", "production_pipeline"]
