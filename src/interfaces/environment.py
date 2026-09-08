from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass
class StepResult(Generic[ObservationT]):
    """Result of one environment step in the common source interface."""

    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class EnvironmentInterface(ABC, Generic[ObservationT, ActionT]):
    """
    Minimal contract for interactive data sources.

    Gymnasium, ARC, robotics, or another interactive environment only need to
    implement this contract. Everything after collection works on Trajectory
    objects and is independent of the environment implementation.
    """

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[ObservationT, dict[str, Any]]:
        ...

    @abstractmethod
    def step(self, action: ActionT) -> StepResult[ObservationT]:
        ...

    @abstractmethod
    def sample_action(self) -> ActionT:
        ...

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        ...

    @property
    @abstractmethod
    def action_space(self) -> Any:
        ...

    def close(self) -> None:
        pass
