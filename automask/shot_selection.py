"""Declarative selection of event indices from a profiled run."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import numpy as np

from automask.run_profile import RunProfile

Operator = Literal[
    "==", "!=", "<", "<=", ">", ">=", "between", "in", "not in",
    "finite", "nonzero",
]


@dataclass(frozen=True)
class Condition:
    """One comparison against a canonical field in ``RunProfile.values``."""

    field: str
    operator: Operator
    value: Any = None

    def __post_init__(self) -> None:
        operators = {
            "==", "!=", "<", "<=", ">", ">=", "between", "in", "not in",
            "finite", "nonzero",
        }
        if not self.field:
            raise ValueError("a condition needs a field name")
        if self.operator not in operators:
            raise ValueError(f"unknown condition operator {self.operator!r}")
        if self.operator == "between":
            if not isinstance(self.value, (tuple, list)) or len(self.value) != 2:
                raise ValueError("'between' needs a (minimum, maximum) pair")
            low, high = self.value
            if low > high:
                raise ValueError("'between' minimum must not exceed maximum")
            object.__setattr__(self, "value", (low, high))
        if self.operator in ("in", "not in"):
            if isinstance(self.value, (str, bytes)):
                raise ValueError(f"{self.operator!r} needs a sequence of values")
            try:
                values = tuple(self.value)
            except TypeError as error:
                raise ValueError(
                    f"{self.operator!r} needs a sequence of values"
                ) from error
            object.__setattr__(self, "value", values)
        if self.operator in ("finite", "nonzero") and self.value is not None:
            raise ValueError(f"{self.operator!r} does not take a value")

    def resolve(self, profile: RunProfile) -> np.ndarray:
        values = profile.column(self.field)
        try:
            if self.operator == "==":
                return values == self.value
            if self.operator == "!=":
                return values != self.value
            if self.operator == "<":
                return values < self.value
            if self.operator == "<=":
                return values <= self.value
            if self.operator == ">":
                return values > self.value
            if self.operator == ">=":
                return values >= self.value
            if self.operator == "between":
                low, high = self.value
                return (values >= low) & (values <= high)
            if self.operator == "in":
                return np.isin(values, self.value)
            if self.operator == "not in":
                return ~np.isin(values, self.value)
            if self.operator == "finite":
                return np.isfinite(values)
            return np.isfinite(values) & (values != 0)
        except TypeError as error:
            raise TypeError(
                f"run {profile.run}: cannot apply {self.operator!r} to field "
                f"{self.field!r} with dtype {values.dtype}"
            ) from error

    def label(self) -> str:
        if self.operator in ("finite", "nonzero"):
            return f"{self.field} {self.operator}"
        return f"{self.field} {self.operator} {self.value!r}"


@dataclass(frozen=True)
class PercentileTrim:
    """Drop low and high tails of one numeric field after conditions apply."""

    field: str
    low: float = 0.0
    high: float = 0.0

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("a percentile trim needs a field name")
        if not (0.0 <= self.low < 1.0 and 0.0 <= self.high < 1.0):
            raise ValueError("percentile trim fractions must be in [0, 1)")
        if self.low + self.high >= 1.0:
            raise ValueError("percentile trim must leave some shots")


@dataclass(frozen=True)
class ShotSelection:
    """A field-native recipe for selecting shots from a ``RunProfile``.

    All conditions are ANDed. The optional percentile trim is evaluated only
    among shots that pass those conditions. ``n_shots`` then takes evenly
    spaced survivors in stream order. ``normalization`` names the field used by
    ImageStore and automatically excludes non-finite and zero values.
    """

    where: Tuple[Condition, ...] = ()
    trim: Optional[PercentileTrim] = None
    n_shots: Optional[int] = None
    normalization: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", tuple(self.where))
        if not all(isinstance(condition, Condition) for condition in self.where):
            raise TypeError("where must contain Condition objects")
        if self.trim is not None and not isinstance(self.trim, PercentileTrim):
            raise TypeError("trim must be a PercentileTrim or None")
        if self.n_shots is not None and self.n_shots < 1:
            raise ValueError("n_shots must be positive or None")
        if self.normalization is not None and not self.normalization:
            raise ValueError("normalization must be a field name or None")

    def _where_mask(self, profile: RunProfile) -> np.ndarray:
        keep = np.ones(profile.events, dtype=bool)
        for condition in self.where:
            keep &= np.asarray(condition.resolve(profile), dtype=bool)
        return keep

    def _eligible_mask(self, profile: RunProfile) -> np.ndarray:
        keep = self._where_mask(profile)
        if self.normalization is not None:
            values = np.asarray(profile.column(self.normalization))
            try:
                keep &= np.isfinite(values) & (values != 0)
            except TypeError as error:
                raise TypeError(
                    f"run {profile.run}: normalization field "
                    f"{self.normalization!r} must be numeric"
                ) from error
        return keep

    def _trimmed_indices(
        self, profile: RunProfile, require_nonempty: bool = True
    ) -> np.ndarray:
        keep = self._eligible_mask(profile)
        if not keep.any():
            if require_nonempty:
                raise RuntimeError(
                    f"run {profile.run}: no shots match "
                    f"{[condition.label() for condition in self.where]}"
                )
            return np.empty(0, dtype=np.int64)
        if self.trim is not None:
            values = np.asarray(profile.column(self.trim.field))
            try:
                finite = np.isfinite(values)
            except TypeError as error:
                raise TypeError(
                    f"run {profile.run}: trim field {self.trim.field!r} "
                    "must be numeric"
                ) from error
            eligible = keep & finite
            if not eligible.any():
                raise RuntimeError(
                    f"run {profile.run}: trim field {self.trim.field!r} has no "
                    "finite values among matching shots"
                )
            low = np.quantile(values[eligible], self.trim.low)
            high = np.quantile(values[eligible], 1.0 - self.trim.high)
            keep = eligible & (values >= low) & (values <= high)
        return np.flatnonzero(keep)

    def _cap(self, survivors: np.ndarray) -> np.ndarray:
        if self.n_shots is None or self.n_shots >= survivors.size:
            return survivors
        positions = np.linspace(0, survivors.size - 1, self.n_shots)
        positions = np.unique(positions.round().astype(np.int64))
        return survivors[positions]

    def resolve(self, profile: RunProfile) -> np.ndarray:
        """Return selected event indices in stream order."""
        return self._cap(self._trimmed_indices(profile))

    @staticmethod
    def create_k_folds(indices: np.ndarray, k: int = 1) -> list:
        """Deal selected event indices round-robin into ``k`` folds."""
        indices = np.asarray(indices)
        if k < 1 or k > indices.size:
            raise ValueError(f"k must be in [1, {indices.size}], got {k}")
        return [indices[fold::k] for fold in range(k)]

    def describe(self, profile: RunProfile) -> dict:
        """Return counts for each selection stage without decoding frames."""
        condition_counts = [
            {
                "condition": condition.label(),
                "n_matching": int(condition.resolve(profile).sum()),
            }
            for condition in self.where
        ]
        after_conditions = int(self._where_mask(profile).sum())
        eligible = int(self._eligible_mask(profile).sum())
        trimmed = self._trimmed_indices(profile, require_nonempty=False)
        selected = int(self._cap(trimmed).size)
        return {
            "n_events": int(profile.events),
            "conditions": condition_counts,
            "n_after_conditions": after_conditions,
            "n_eligible": eligible,
            "n_after_trim": int(trimmed.size),
            "n_selected": selected,
        }

    def normalization_reference(
        self, profile: RunProfile, indices: np.ndarray
    ) -> float:
        """Median normalization value over selected shots."""
        if self.normalization is None:
            raise ValueError("selection does not request normalization")
        return float(np.median(profile.column(self.normalization)[indices]))
