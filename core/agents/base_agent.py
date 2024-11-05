from typing import Any, Dict, List, Optional, Tuple, Type, Union
import os
import numpy as np
import torch as th
import gym
from gym import spaces

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update


class BaseAgent(OffPolicyAlgorithm):
    """
    Base agent class extending SB3's OffPolicyAlgorithm.
    Implements common functionality for safe RL agents.
    """

    def __init__(
            self,
            policy: Union[str, Type["BasePolicy"]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-3,
            buffer_size: int = 1_000_000,
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = (1, "step"),
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            optimize_memory_usage: bool = False,
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        """
        Initialize the base agent.

        Args:
            policy: Policy class or string
            env: Training environment
            learning_rate: Learning rate
            buffer_size: Size of replay buffer
            learning_starts: Number of steps before training starts
            batch_size: Batch size for training
            tau: Soft update coefficient for target networks
            gamma: Discount factor
            train_freq: Update the model every `train_freq` steps
            gradient_steps: Number of gradient steps per update
            action_noise: Action noise for exploration
            optimize_memory_usage: Optimize memory usage of replay buffer
            policy_delay: Delay between policy updates
            target_policy_noise: Noise added to target policy
            target_noise_clip: Maximum value of target policy noise
            tensorboard_log: Tensorboard log directory
            policy_kwargs: Additional policy arguments
            verbose: Verbosity level
            seed: Random seed
            device: Device to run on
            _init_setup_model: Whether to build the network at initialization
            supported_action_spaces: Supported action spaces
        """
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=supported_action_spaces or (spaces.Box),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """
        Setup networks and optimizers.
        """
        super()._setup_model()
        self._create_aliases()

        # Get running statistics for batch normalization
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        """
        Create aliases for network components.
        """
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Train the agent.

        Args:
            gradient_steps: Number of gradient steps
            batch_size: Size of each batch
        """
        raise NotImplementedError("train() method must be implemented by child class")

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        """
        Train the agent.

        Args:
            total_timesteps: Total number of timesteps
            callback: Callback function
            log_interval: Logging interval
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Whether to reset timestep counter
            progress_bar: Whether to display progress bar
        """
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _update_target_networks(self) -> None:
        """
        Update target networks using Polyak averaging.
        """
        # Update critic target network
        polyak_update(
            params=self.critic.parameters(),
            target_params=self.critic_target.parameters(),
            tau=self.tau
        )

        # Update actor target network
        polyak_update(
            params=self.actor.parameters(),
            target_params=self.actor_target.parameters(),
            tau=self.tau
        )

        # Update batch normalization statistics
        polyak_update(
            params=self.critic_batch_norm_stats,
            target_params=self.critic_batch_norm_stats_target,
            tau=1.0
        )
        polyak_update(
            params=self.actor_batch_norm_stats,
            target_params=self.actor_batch_norm_stats_target,
            tau=1.0
        )

    def _sample_action(
            self,
            learning_starts: int,
            action_noise: Optional[ActionNoise] = None,
            n_envs: int = 1,
    ) -> np.ndarray:
        """
        Sample actions with noise for exploration.

        Args:
            learning_starts: Number of steps before learning starts
            action_noise: Action noise object
            n_envs: Number of environments

        Returns:
            actions: Sampled actions
        """
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase: sample random actions
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Sample actions from policy
            unscaled_action = self.predict(self._last_obs, deterministic=False)[0]

        # Add noise if specified
        if action_noise is not None:
            unscaled_action = np.clip(
                unscaled_action + action_noise(),
                -1,
                1
            )

        return unscaled_action

    def _excluded_save_params(self) -> List[str]:
        """Parameters to exclude when saving."""
        return super()._excluded_save_params() + [
            "actor",
            "critic",
            "actor_target",
            "critic_target"
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """Parameters to save in state_dict."""
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

    def save(
            self,
            path: str,
            exclude: Optional[List[str]] = None,
            include: Optional[List[str]] = None,
    ) -> None:
        """
        Save the model.

        Args:
            path: Path to save to
            exclude: List of parameters to exclude
            include: List of parameters to include
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        super().save(
            path=path,
            exclude=exclude,
            include=include
        )

    def _get_safe_action(
            self,
            observations: th.Tensor,
            actions: th.Tensor
    ) -> th.Tensor:
        """
        Get safe actions based on safety criteria.
        To be implemented by child classes.

        Args:
            observations: Current observations
            actions: Proposed actions

        Returns:
            safe_actions: Safe actions
        """
        raise NotImplementedError("_get_safe_action() must be implemented by child class")

    def _update_safety_critic(
            self,
            observations: th.Tensor,
            actions: th.Tensor,
            next_observations: th.Tensor,
            rewards: th.Tensor,
            dones: th.Tensor
    ) -> Dict[str, float]:
        """
        Update safety critic network.
        To be implemented by child classes.

        Args:
            observations: Current observations
            actions: Current actions
            next_observations: Next observations
            rewards: Rewards
            dones: Done flags

        Returns:
            metrics: Dictionary of update metrics
        """
        raise NotImplementedError("_update_safety_critic() must be implemented by child class")