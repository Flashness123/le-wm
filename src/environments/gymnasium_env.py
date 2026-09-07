from typing import Any, Optional

import gymnasium as gym

from src.environment_interface import (
    EnvironmentInterface,
    StepResult,
)


class GymnasiumEnvironment(EnvironmentInterface[Any, Any]):

    def __init__(
        self,
        env_id: str,
        observation_mode: str = "state",
        **env_kwargs: Any,
    ):
        """
        observation_mode:
            "state"
                -> normal Gymnasium observation

            "pixels"
                -> rendered RGB image only

            "pixels_with_state"
                -> {
                       "state": original observation,
                       "pixels": RGB image
                   }

        For JEPA experiments I recommend "pixels_with_state":
        JEPA sees only pixels, while state remains available for evaluation.
        """

        self.env_id = env_id
        self.observation_mode = observation_mode

        if observation_mode == "state":

            self.env = gym.make(
                env_id,
                **env_kwargs,
            )

        elif observation_mode in {"pixels", "pixels_with_state"}:

            # Gymnasium requires rgb_array to be selected when
            # the environment is created.
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                **env_kwargs,
            )

            self.env = gym.wrappers.AddRenderObservation(
                env,
                render_only=(observation_mode == "pixels"),
            )

        else:
            raise ValueError(
                "observation_mode must be one of: "
                "'state', 'pixels', 'pixels_with_state'"
            )

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> tuple[Any, dict[str, Any]]:

        observation, info = self.env.reset(seed=seed)

        return observation, info

    def step(
        self,
        action: Any,
    ) -> StepResult[Any]:

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def sample_action(self) -> Any:
        return self.env.action_space.sample()

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def close(self) -> None:
        self.env.close()