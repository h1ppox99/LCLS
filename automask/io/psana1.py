"""Psana1 run sources for local development and SLAC execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class Psana1RunSource:
    """Description of one psana1 run and how psana should resolve it."""

    experiment: str
    run: int
    files: Tuple[Path, ...] = ()
    calib_dir: Optional[Path] = None
    smd: bool = False
    mpi: bool = False

    @classmethod
    def from_files(
        cls,
        experiment: str,
        run: int,
        files: Iterable[str | Path],
        calib_dir: str | Path | None = None,
    ) -> "Psana1RunSource":
        paths = tuple(Path(path).expanduser().resolve() for path in files)
        if not paths:
            raise ValueError(
                "an explicit-file psana source needs at least one XTC file"
            )
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"XTC file does not exist: {missing[0]}")
        calibration = (
            Path(calib_dir).expanduser().resolve() if calib_dir is not None else None
        )
        return cls(
            experiment=experiment,
            run=int(run),
            files=paths,
            calib_dir=calibration,
        )

    @classmethod
    def from_experiment(
        cls,
        experiment: str,
        run: int,
        *,
        smd: bool = True,
        mpi: bool = False,
    ) -> "Psana1RunSource":
        return cls(
            experiment=experiment,
            run=int(run),
            smd=smd,
            mpi=mpi,
        )

    @property
    def dataset(self) -> str:
        suffix = ":smd" if self.smd else ""
        return f"exp={self.experiment}:run={self.run}{suffix}"

    def open(self):
        """Create the psana datasource described by this object."""
        import psana

        if self.calib_dir is not None:
            if not self.calib_dir.is_dir():
                raise FileNotFoundError(
                    f"psana calibration directory does not exist: {self.calib_dir}"
                )
            psana.setOption("psana.calib-dir", str(self.calib_dir))

        if self.files:
            if self.mpi:
                raise ValueError("MPIDataSource is not supported with explicit files")
            return psana.DataSource(*(str(path) for path in self.files))

        factory = psana.MPIDataSource if self.mpi else psana.DataSource
        return factory(self.dataset)
