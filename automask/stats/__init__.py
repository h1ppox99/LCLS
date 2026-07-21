"""stats package -- importing it registers every statistic."""
from automask.stats.base import STATS, StatSpec, register_stat  # noqa: F401
# registration side effects (order-independent):
from automask.stats import variance, window_median, blackhat  # noqa: F401
from automask.stats import geometry, radial_median  # noqa: F401
