"""stats package -- importing it registers every statistic."""
from stats.base import STATS, StatSpec, register_stat  # noqa: F401
# registration side effects (order-independent):
from stats import variance, window_median, blackhat, geometry, calib  # noqa: F401
from stats import radial_median, azimuthal_sigma  # noqa: F401
