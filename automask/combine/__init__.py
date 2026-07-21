"""combine package -- importing it registers every combiner."""
from automask.combine.base import COMBINERS, CombineSpec, register_combine  # noqa: F401
from automask.combine import union, weighted_sum, mahalanobis  # noqa: F401  (registration side effect)
