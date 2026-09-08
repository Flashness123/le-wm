from omegaconf import DictConfig

from src.environments.gymnasium_env import GymnasiumEnvironment
from src.interfaces.environment_interface import EnvironmentInterface


def build_environment(cfg: DictConfig) -> EnvironmentInterface:
    backend = cfg.backend

    if backend == "gymnasium":
        # MiniGrid registers its environments when imported.
        if str(cfg.env_id).startswith("MiniGrid-"):
            import minigrid  # noqa: F401

        return GymnasiumEnvironment(
            env_id=cfg.env_id,
            observation_mode=cfg.observation_mode,
        )

    raise NotImplementedError(
        f"Environment backend '{backend}' is not implemented yet. "
        "Add an adapter that implements EnvironmentInterface."
    )
