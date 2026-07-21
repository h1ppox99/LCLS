"""regularization package -- importing it registers every regularizer."""
from regularization.base import REGULARIZERS, RegSpec, register_reg  # noqa: F401
from regularization import tv, pad, close_open  # noqa: F401  (registration side effect)
