import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Union, NamedTuple, Tuple, List, Any
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import (
    RolloutBufferSamples,
    ReplayBufferSamples,
    DictReplayBufferSamples,
)

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class SafeRolloutBufferSamples(NamedTuple):
    """
    Enhanced rollout buffer samples with safety-related data.
    """
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    barrier_values: th.Tensor  # Barrier certificate values
    safety_advantages: th.Tensor  # Safety-adjusted advantages
    episode_starts: th.Tensor


class SafeReplayBufferSamples(NamedTuple):
    """
    Enhanced replay buffer samples with safety-related data.
    """
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    feasible_mask: th.Tensor  # Feasible state mask
    infeasible_mask: th.Tensor  # Infeasible state mask


class SafeReplayBuffer(BaseBuffer):
    """
    Replay buffer for off-policy algorithms with safety considerations.
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

        # Safety-related buffers
        #self.barrier_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        #self.next_barrier_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.feasible_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.infeasible_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        #self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def size(self) -> int:
        """Return current buffer size."""
        return self.buffer_size if self.full else self.pos

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            feasible: np.ndarray,
            infeasible: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add experience to buffer.
        Notes: infos must contains a key: "cost"
        """
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        # Safety data
        self.feasible_mask[self.pos] = np.array(feasible).copy()
        self.infeasible_mask[self.pos] = np.array(infeasible).copy()
        #self.episode_starts[self.pos] = np.array(episode_start).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> SafeReplayBufferSamples:
        """
        Sample experiences from buffer.
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)

        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None
    ) -> SafeReplayBufferSamples:
        """
        Get samples from array with normalization if needed.
        """
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.feasible_mask[batch_inds, env_indices].reshape(-1, 1),
            self.infeasible_mask[batch_inds, env_indices].reshape(-1, 1),
        )
        return SafeReplayBufferSamples(*tuple(map(self.to_torch, data)))

