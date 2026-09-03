"""stats package -- importing it registers every statistic."""

from automask.mask.stats.base import STATS, StatSpec, register_stat  # noqa: F401

# registration side effects (order-independent):
from automask.mask.stats import variance, mad_variance, blackhat, hough_lines  # noqa: F401
from automask.mask.stats import geometry, sigma_clipping, asic_polish  # noqa: F401
from automask.mask.stats import status_as_mask  # noqa: F401
