"""stats package -- importing it registers every statistic."""

from automask.stats.base import STATS, StatSpec, register_stat  # noqa: F401

# registration side effects (order-independent):
from automask.stats import variance, mad_variance, blackhat, hough_lines  # noqa: F401
from automask.stats import geometry, sigma_clipping, asic_polish  # noqa: F401
from automask.stats import status_as_mask  # noqa: F401
