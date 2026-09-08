from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.trajectory import Trajectory


class DataSourceInterface(ABC):
    """
    A source of raw trajectories.

    Implementations may be backed by:
      - freshly collected environments
      - HDF5 / Zarr / parquet files
      - ARC replay files
      - robotics logs
      - remote datasets

    Everything above this interface may be source-specific.
    Everything below it should work with canonical Trajectory objects.
    """

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def get_trajectory(self, index: int) -> "Trajectory":
        ...

    def __getitem__(self, index: int) -> "Trajectory":
        return self.get_trajectory(index)
