from typing import Any

import gymnasium as gym

from src.interfaces.environment import EnvironmentInterface, StepResult


class GymnasiumEnvironment(EnvironmentInterface[Any, Any]):
    """Adapt a Gymnasium environment to the common EnvironmentInterface."""

    def __init__(self, env_id: str, observation_mode: str = "state", **env_kwargs: Any):
        self.env_id = env_id
        self.observation_mode = observation_mode

        # MiniGrid registers its environment IDs on import.
        if env_id.startswith("MiniGrid-"):
            import minigrid  # noqa: F401

        if observation_mode == "state":
            self.env = gym.make(env_id, **env_kwargs)
        elif observation_mode in {"pixels", "pixels_with_state"}:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            self.env = gym.wrappers.AddRenderObservation(
                env,
                render_only=(observation_mode == "pixels"),
            )
        else:
            raise ValueError(
                "observation_mode must be 'state', 'pixels', or 'pixels_with_state'"
            )

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self.env.action_space.seed(seed)
        return self.env.reset(seed=seed)

    def step(self, action: Any) -> StepResult[Any]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
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
