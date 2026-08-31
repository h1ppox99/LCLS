"""regularization package -- importing it registers every regularizer."""

from automask.regularization.base import REGULARIZERS, RegSpec, register_reg  # noqa: F401
from automask.regularization import tv, pad  # noqa: F401  (registration side effect)
from automask.regularization import blob_scale, fill_holes, area_gate  # noqa: F401
