"""stats package -- importing it registers every statistic."""
from automask.stats.base import STATS, StatSpec, register_stat  # noqa: F401
# registration side effects (order-independent):
from automask.stats import variance, blackhat  # noqa: F401
from automask.stats import geometry, sigma_clipping, calib  # noqa: F401
