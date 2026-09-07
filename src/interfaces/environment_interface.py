#TODO: change file formatting with prettier

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass
class StepResult(Generic[ObservationT]): 
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
    Common interface for all environments used in this project.

    Later implementations:
        - Gymnasium
        - Atari
        - ARC-AGI-3
    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> tuple[ObservationT, dict[str, Any]]:
        pass

    @abstractmethod
    def step(self, action: ActionT) -> StepResult[ObservationT]:
        pass

    @abstractmethod
    def sample_action(self) -> ActionT:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Any:
        pass

    def close(self) -> None:
        pass