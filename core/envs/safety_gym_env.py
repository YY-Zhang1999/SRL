import gymnasium
from gymnasium import spaces
from gymnasium.core import ActType
from gymnasium import logger

import safety_gymnasium as sg
import numpy as np

from typing import Any

from gymnasium import Env, error, logger
from gymnasium.envs.registration import namespace  # noqa: F401 # pylint: disable=unused-import
from gymnasium.envs.registration import spec  # noqa: F401 # pylint: disable=unused-import
from gymnasium.envs.registration import EnvSpec, _check_metadata, _find_spec, load_env_creator


class Float32Wrapper(gymnasium.Wrapper):
    """
    将Safety-Gymnasium环境的观察空间和动作空间的数据类型转换为float32的包装器
    """

    def __init__(self, env):
        super().__init__(env)

        # 转换观察空间
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Box):
            self.observation_space = spaces.Box(
                low=obs_space.low,
                high=obs_space.high,
                shape=obs_space.shape,
                dtype=np.float32
            )

        # 转换动作空间
        act_space = env.action_space
        if isinstance(act_space, spaces.Box):
            self.action_space = spaces.Box(
                low=act_space.low,
                high=act_space.high,
                shape=act_space.shape,
                dtype=np.float32
            )

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        if 'cost' in info:
            logger.warn(
                'The info dict already contains a cost. '
                'Overwriting it may cause unexpected behavior.',
            )
        info['cost'] = cost
        return obs, reward, terminated, truncated, info

def make_safety_env(
        id: str | EnvSpec,  # pylint: disable=invalid-name,redefined-builtin
        max_episode_steps: int | None = None,
        autoreset: bool | None = None,
        apply_api_compatibility: bool | None = None,
        disable_env_checker: bool | None = None,
        **kwargs: Any,
) -> Env:

    env = sg.make(id, max_episode_steps, autoreset, apply_api_compatibility, disable_env_checker, **kwargs)
    env = Float32Wrapper(env)

    return env
