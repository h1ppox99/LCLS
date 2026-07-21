"""combine package -- importing it registers every combiner."""
from combine.base import COMBINERS, CombineSpec, register_combine  # noqa: F401
from combine import union, weighted_sum, mahalanobis  # noqa: F401  (registration side effect)
