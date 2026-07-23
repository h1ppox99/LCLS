"""
automask.features -- the feature layer between ShotSelection and the masking stats.

    from automask.features import FEATURES, FeatureSpec, FeatureStore, get_spec

Importing this package populates ``FEATURES`` with the default catalogue.
"""
from automask.features.base import FEATURES, FeatureSpec, Reduction, get_spec, register
from automask.features.store import FeatureStore
from automask.features import catalog  # noqa: F401  (registers default features)

__all__ = [
    "FEATURES", "FeatureSpec", "Reduction", "get_spec", "register", "FeatureStore",
]
