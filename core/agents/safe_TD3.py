import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from stable_baselines3.common.callbacks import BaseCallback

from .base_agent import BaseAgent
from ..models.barrier import BarrierNetwork
from ..losses.losses import SRLNBCLoss
from ..utils.buffers import SafeReplayBuffer
from ..models.TD3_barrier_policy import SafeTD3Policy

class Safe_TD3(BaseAgent):
    """
    Twin Delayed DDPG (TD3) agent with safety constraints through barrier certificates.
    """

    def __init__(
            self,
            policy: Union[str, Type["SafeTD3Policy"]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 5e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 100,
            batch_size: int = 64,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
            gradient_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[SafeReplayBuffer]] = SafeReplayBuffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            barrier_lambda: float = 0.1,
            n_barrier_steps: int = 10,
            gamma_barrier: float = 0.99,
            safety_margin: float = 0.1,
            lambda_lr: float = 3e-2,
            _init_setup_model: bool = True,
    ):
        """
        Initialize TD3 agent with safety features.

        Additional Args (beyond BaseAgent):
            barrier_lambda: Weight of barrier loss
            n_barrier_steps: Number of steps for multi-step barrier
            gamma_barrier: Discount factor for multi-step barrier
            safety_margin: Minimum safety margin
        """
        if policy == "SafeTD3Policy":
            policy = SafeTD3Policy

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
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            n_barrier_steps=n_barrier_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            _init_setup_model=False,
        )

        # Safety parameters
        self.barrier_lambda = barrier_lambda
        self.gamma_barrier = gamma_barrier
        self.safety_margin = safety_margin
        self.lambda_lr = lambda_lr

        # TD3 specific parameters
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Setup networks and optimizers."""
        super()._setup_model()

        # Initialize barrier network if not provided in policy
        if not hasattr(self.policy, "barrier_net"):
            self.policy.barrier_net = BarrierNetwork(
                observation_space=self.observation_space,
                hidden_sizes=self.policy_kwargs.get("barrier_hidden_sizes", [400, 300]),
                device=self.device
            ).to(self.device)

        # Initialize unified loss function
        self.loss_fn = SRLNBCLoss({
            "lambda_barrier": self.barrier_lambda,
            "n_barrier_steps": self.n_barrier_steps,
            "gamma_barrier": self.gamma_barrier,
            "policy_delay": self.policy_delay,
            "target_policy_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
            "lambda_lr": self.lambda_lr,
            "device": self.device
        }).to(self.device)

        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def learn_(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps // 4:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps, mode=0)

        self.replay_buffer.reset()

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > total_timesteps // 2 + self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps, mode=1)

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int = 256, mode: int = 1) -> None:
        """
        Training loop for TD3 with safety considerations.

        Args:
            gradient_steps: Number of gradient steps
            batch_size: Size of each batch
            mode: Training mode.
                  0: Pre-training phase,
                  1: Turning phase
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update learning rate
        self._update_learning_rate([
            self.actor.optimizer,
            self.critic.optimizer,
            self.policy.barrier_net.optimizer
        ])

        actor_losses = []
        critic_losses = []
        barrier_losses = []
        safety_violations = []

        for _ in range(gradient_steps):
            self._n_updates += 1

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env
            )

            # Update critics and barrier certificate
            with th.no_grad():
                # Get noisy target actions
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                        self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)

                # Compute target Q-values
                target_q_values = th.cat(
                    self.critic_target(
                        replay_data.next_observations,
                        next_actions
                    ),
                    dim=1
                )
                target_q_values, _ = th.min(target_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q_values

            # Get current Q-values
            current_q_values = self.critic(
                replay_data.observations,
                replay_data.actions
            )

            # Compute critic loss
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            # Optimize critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_actions = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations,
                    actor_actions
                ).mean()

                # Get barrier values and compute barrier loss
                barrier_values, next_barrier_values = self.get_barrier_values(replay_data.next_n_observations, self.policy.barrier_net)

                total_loss, barrier_loss, invariant_loss, info = self.loss_fn(
                    barrier_values=barrier_values,
                    next_barrier_values=next_barrier_values,
                    feasible_mask=replay_data.feasible_mask,
                    infeasible_mask=replay_data.infeasible_mask,
                    episode_mask=replay_data.n_dones,
                    current_actions=actor_actions,
                    old_actions=replay_data.actions,
                    critic_loss=critic_loss,
                    actor_loss=actor_loss,
                )

                # Optimize actor and barrier
                if mode == 0:
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()

                else:
                    self.actor.optimizer.zero_grad()
                    total_loss.backward()
                    self.actor.optimizer.step()

                    self.loss_fn.lagrange_multiplier.update(actor_loss.detach(), invariant_loss.detach())

                self.policy.barrier_net.optimizer.zero_grad()
                barrier_loss.backward()
                self.policy.barrier_net.optimizer.step()

                # Update target networks
                self._update_target_networks()

                # Store losses
                actor_losses.append(actor_loss.item())
                barrier_losses.append(barrier_loss.item())

                # Compute safety violations
                safety_violation = replay_data.infeasible_mask.float().mean().item()
                safety_violations.append(safety_violation)

                if self._n_updates % 1000 == 0:
                    print(safety_violation, (barrier_values > 0).float().mean().item())
                    print(info)

        # Log training info
        self._log_training_info(
            actor_losses=actor_losses,
            critic_losses=critic_losses,
            barrier_losses=barrier_losses,
            safety_violations=safety_violations
        )

    def get_barrier_values(
            self,
            next_n_observations: th.Tensor,
            barrier_net: th.nn.Module
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute barrier values for current and n-step future observations.

        Args:
            next_n_observations: Tensor of shape (n_steps, batch_size, *obs_shape)
                               containing the next n observations
            barrier_net: Neural network that computes barrier values

        Returns:
            barrier_values: Tensor of shape (batch_size, 1) containing current barrier values
            next_barrier_values: Tensor of shape (n_steps, batch_size, 1)
                               containing current and future barrier values
        """
        # Ensure the input is on the correct device
        if not isinstance(next_n_observations, th.Tensor):
            next_n_observations = th.tensor(next_n_observations)

        next_n_observations = next_n_observations.to(self.device)

        # Compute barrier values for all n-step observations at once
        # Reshape to (n_steps * batch_size, *obs_shape)
        n_steps, batch_size = next_n_observations.shape[:2]
        reshaped_obs = next_n_observations.view(-1, *next_n_observations.shape[2:])

        # Get all barrier values in one forward pass
        all_barrier_values = barrier_net(reshaped_obs)

        # Reshape back to (n_steps, batch_size, 1)
        next_barrier_values = all_barrier_values.view(n_steps, batch_size, 1)

        # Get current barrier values (first step)
        barrier_values = next_barrier_values[0]  # Shape: (batch_size, 1)

        return barrier_values, next_barrier_values

    def _log_training_info(
            self,
            actor_losses: List[float],
            critic_losses: List[float],
            barrier_losses: List[float],
            safety_violations: List[float]
    ) -> None:
        """Log training metrics."""
        self.logger.record("train/n_updates", self._n_updates)

        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/barrier_loss", np.mean(barrier_losses))
            self.logger.record("train/safety_violations", np.mean(safety_violations))

        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def _get_safe_action(
            self,
            observations: th.Tensor,
            actions: th.Tensor,
            safe_mode: bool = False,
    ) -> th.Tensor:
        """
        Get safe actions by checking barrier certificate.

        Args:
            observations: Current observations
            actions: Proposed actions

        Returns:
            Safe actions
        """
        if not safe_mode:
            return actions

        with th.no_grad():
            barrier_values = self.policy.barrier_net(observations)
            unsafe_mask = barrier_values > -self.safety_margin

            if unsafe_mask.any():
                # Sample alternative actions
                n_samples = 10
                batch_size = observations.shape[0]
                alt_actions = th.randn(
                    (batch_size, n_samples, *self.action_space.shape),
                    device=self.device
                )

                # Evaluate safety of alternative actions
                safety_values = []
                for i in range(n_samples):
                    safety_value = self.critic.safety_critic(
                        observations,
                        alt_actions[:, i]
                    )
                    safety_values.append(safety_value)

                safety_values = th.stack(safety_values, dim=1)

                # Select safest actions
                safest_idx = safety_values.argmax(dim=1)
                safe_actions = alt_actions[
                    th.arange(batch_size),
                    safest_idx
                ]

                # Replace unsafe actions
                actions[unsafe_mask] = safe_actions[unsafe_mask]

        return actions

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            safe_mode: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get action with safety check.

        Args:
            observation: Current observation
            state: Current state (for recurrent policies)
            episode_start: Whether episode has started
            deterministic: Whether to use deterministic action

        Returns:
            action: Selected action
            state: Updated state
        """
        self.policy.set_training_mode(False)

        actions = self.policy.predict(observation, state, episode_start, deterministic)
        safe_actions, state = self._get_safe_action(observation, actions, safe_mode=safe_mode)

        return safe_actions, state

class SafeCritic_TD3(Safe_TD3):
    """
    Twin Delayed DDPG (TD3) agent with safety constraints through barrier certificates.
    """

    def __init__(
            self,
            policy: Union[str, Type["SafeTD3Policy"]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 5e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 100,
            batch_size: int = 64,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
            gradient_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[SafeReplayBuffer]] = SafeReplayBuffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            barrier_lambda: float = 0.1,
            n_barrier_steps: int = 10,
            gamma_barrier: float = 0.99,
            safety_margin: float = 0.1,
            lambda_lr: float = 3e-2,
            _init_setup_model: bool = True,
    ):
        """
        Initialize TD3 agent with safety features.

        Additional Args (beyond BaseAgent):
            barrier_lambda: Weight of barrier loss
            n_barrier_steps: Number of steps for multi-step barrier
            gamma_barrier: Discount factor for multi-step barrier
            safety_margin: Minimum safety margin
        """
        if policy == "SafeTD3Policy":
            policy = SafeTD3Policy

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
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            n_barrier_steps=n_barrier_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            _init_setup_model=False,
        )

        # Safety parameters
        self.barrier_lambda = barrier_lambda
        self.gamma_barrier = gamma_barrier
        self.safety_margin = safety_margin
        self.lambda_lr = lambda_lr

        # TD3 specific parameters
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Setup networks and optimizers."""
        super()._setup_model()

        # Initialize barrier network if not provided in policy
        if not hasattr(self.policy, "barrier_net"):
            self.policy.barrier_net = BarrierNetwork(
                observation_space=self.observation_space,
                hidden_sizes=self.policy_kwargs.get("barrier_hidden_sizes", [400, 300]),
                device=self.device
            ).to(self.device)

        # Initialize unified loss function
        self.loss_fn = SRLNBCLoss({
            "lambda_barrier": self.barrier_lambda,
            "n_barrier_steps": self.n_barrier_steps,
            "gamma_barrier": self.gamma_barrier,
            "policy_delay": self.policy_delay,
            "target_policy_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
            "lambda_lr": self.lambda_lr,
            "device": self.device
        }).to(self.device)

        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def learn_(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps // 2:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps, mode=0)

        self.replay_buffer.reset()

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > total_timesteps // 2 + self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps, mode=1)

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int = 256, mode: int = 1) -> None:
        """
        Training loop for TD3 with safety considerations.

        Args:
            gradient_steps: Number of gradient steps
            batch_size: Size of each batch
            mode: Training mode.
                  0: Pre-training phase,
                  1: Turning phase
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update learning rate
        self._update_learning_rate([
            self.actor.optimizer,
            self.critic.optimizer,
            self.policy.barrier_net.optimizer
        ])

        actor_losses = []
        critic_losses = []
        barrier_losses = []
        safety_violations = []

        for _ in range(gradient_steps):
            self._n_updates += 1

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env
            )

            # Update critics and barrier certificate
            with th.no_grad():
                # Get noisy target actions
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                        self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)

                # Compute target Q-values
                target_q_values = th.cat(
                    self.critic_target(
                        replay_data.next_observations,
                        next_actions
                    ),
                    dim=1
                )
                target_q_values, _ = th.min(target_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q_values

            # Get current Q-values
            current_q_values = self.critic(
                replay_data.observations,
                replay_data.actions
            )

            # Compute critic loss
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            # Optimize critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_actions = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations,
                    actor_actions
                ).mean()

                # Get barrier values and compute barrier loss
                barrier_values, next_barrier_values = self.get_barrier_values(replay_data.next_n_observations, self.policy.barrier_net)

                total_loss, barrier_loss, invariant_loss, info = self.loss_fn(
                    barrier_values=barrier_values,
                    next_barrier_values=next_barrier_values,
                    feasible_mask=replay_data.feasible_mask,
                    infeasible_mask=replay_data.infeasible_mask,
                    episode_mask=replay_data.n_dones,
                    current_actions=actor_actions,
                    old_actions=replay_data.actions,
                    critic_loss=critic_loss,
                    actor_loss=actor_loss,
                )

                # Optimize actor and barrier
                if mode == 0:
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()

                else:
                    self.actor.optimizer.zero_grad()
                    total_loss.backward()
                    self.actor.optimizer.step()

                    self.loss_fn.lagrange_multiplier.update(actor_loss.detach(), invariant_loss.detach())

                self.policy.barrier_net.optimizer.zero_grad()
                barrier_loss.backward()
                self.policy.barrier_net.optimizer.step()

                # Update target networks
                self._update_target_networks()

                # Store losses
                actor_losses.append(actor_loss.item())
                barrier_losses.append(barrier_loss.item())

                # Compute safety violations
                safety_violation = replay_data.infeasible_mask.float().mean().item()
                safety_violations.append(safety_violation)

                if self._n_updates % 1000 == 0:
                    print(safety_violation, (barrier_values > 0).float().mean().item())
                    print(info)

        # Log training info
        self._log_training_info(
            actor_losses=actor_losses,
            critic_losses=critic_losses,
            barrier_losses=barrier_losses,
            safety_violations=safety_violations
        )

    def get_barrier_values(
            self,
            next_n_observations: th.Tensor,
            barrier_net: th.nn.Module
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute barrier values for current and n-step future observations.

        Args:
            next_n_observations: Tensor of shape (n_steps, batch_size, *obs_shape)
                               containing the next n observations
            barrier_net: Neural network that computes barrier values

        Returns:
            barrier_values: Tensor of shape (batch_size, 1) containing current barrier values
            next_barrier_values: Tensor of shape (n_steps, batch_size, 1)
                               containing current and future barrier values
        """
        # Ensure the input is on the correct device
        if not isinstance(next_n_observations, th.Tensor):
            next_n_observations = th.tensor(next_n_observations)

        next_n_observations = next_n_observations.to(self.device)

        # Compute barrier values for all n-step observations at once
        # Reshape to (n_steps * batch_size, *obs_shape)
        n_steps, batch_size = next_n_observations.shape[:2]
        reshaped_obs = next_n_observations.view(-1, *next_n_observations.shape[2:])

        # Get all barrier values in one forward pass
        all_barrier_values = barrier_net(reshaped_obs)

        # Reshape back to (n_steps, batch_size, 1)
        next_barrier_values = all_barrier_values.view(n_steps, batch_size, 1)

        # Get current barrier values (first step)
        barrier_values = next_barrier_values[0]  # Shape: (batch_size, 1)

        return barrier_values, next_barrier_values

    def _log_training_info(
            self,
            actor_losses: List[float],
            critic_losses: List[float],
            barrier_losses: List[float],
            safety_violations: List[float]
    ) -> None:
        """Log training metrics."""
        self.logger.record("train/n_updates", self._n_updates)

        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/barrier_loss", np.mean(barrier_losses))
            self.logger.record("train/safety_violations", np.mean(safety_violations))

        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def _get_safe_action(
            self,
            observations: th.Tensor,
            actions: th.Tensor,
            safe_mode: bool = False,
    ) -> th.Tensor:
        """
        Get safe actions by checking barrier certificate.

        Args:
            observations: Current observations
            actions: Proposed actions

        Returns:
            Safe actions
        """
        if not safe_mode:
            return actions

        with th.no_grad():
            barrier_values = self.policy.barrier_net(observations)
            unsafe_mask = barrier_values > -self.safety_margin

            if unsafe_mask.any():
                # Sample alternative actions
                n_samples = 10
                batch_size = observations.shape[0]
                alt_actions = th.randn(
                    (batch_size, n_samples, *self.action_space.shape),
                    device=self.device
                )

                # Evaluate safety of alternative actions
                safety_values = []
                for i in range(n_samples):
                    safety_value = self.critic.safety_critic(
                        observations,
                        alt_actions[:, i]
                    )
                    safety_values.append(safety_value)

                safety_values = th.stack(safety_values, dim=1)

                # Select safest actions
                safest_idx = safety_values.argmax(dim=1)
                safe_actions = alt_actions[
                    th.arange(batch_size),
                    safest_idx
                ]

                # Replace unsafe actions
                actions[unsafe_mask] = safe_actions[unsafe_mask]

        return actions

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            safe_mode: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get action with safety check.

        Args:
            observation: Current observation
            state: Current state (for recurrent policies)
            episode_start: Whether episode has started
            deterministic: Whether to use deterministic action

        Returns:
            action: Selected action
            state: Updated state
        """
        self.policy.set_training_mode(False)

        actions = self.policy.predict(observation, state, episode_start, deterministic)
        safe_actions, state = self._get_safe_action(observation, actions, safe_mode=safe_mode)

        return safe_actions, state