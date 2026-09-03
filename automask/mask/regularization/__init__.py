"""regularization package -- importing it registers every regularizer."""

from automask.mask.regularization.base import REGULARIZERS, RegSpec, register_reg  # noqa: F401
from automask.mask.regularization import tv, pad  # noqa: F401  (registration side effect)
from automask.mask.regularization import blob_scale, fill_holes, area_gate  # noqa: F401
