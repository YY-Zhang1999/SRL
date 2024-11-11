from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import os
import numpy as np
import torch as th
import gym

from gym import spaces

from SRL.core.utils.buffers import SafeReplayBufferSamples, SafeReplayBuffer

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, TrainFreq, \
    TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv



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
            replay_buffer_class: Optional[Type[SafeReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
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
            sde_support=False,
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
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=sde_support,
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

    def _store_safe_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        feasible_mask: np.ndarray,
        infeasible_mask: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            feasible_mask,
            infeasible_mask,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Extract safety information
            total_cost = np.array([info.get("cost", 0) for info in infos])

            if total_cost.shape[0] != buffer_actions.shape[0]:
                raise ValueError(
                    f"Cost size mismatch: "                    
                    f"\nCost shape: {total_cost.shape}"
                    f"\nAction shape: {actions.shape}"
                )

            infeasible_mask = total_cost > 0
            feasible_mask = ~infeasible_mask

            # Reshape masks once at the end if needed
            if len(infeasible_mask.shape) == 1:
                infeasible_mask = infeasible_mask.reshape(1, -1)
                feasible_mask = feasible_mask.reshape(1, -1)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_safe_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, feasible_mask, infeasible_mask, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


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