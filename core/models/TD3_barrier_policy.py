import numpy as np
import torch as th
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Type, Union, Any
from gym import spaces
from functools import partial

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp
)
from stable_baselines3.common.type_aliases import Schedule


class SafeActor(BasePolicy):
    """
    Actor network for Safe TD3 policy.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            net_arch: List[int],
            activation_fn: Type[nn.Module],
            features_extractor: Optional[BaseFeaturesExtractor] = None,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            device: Union[th.device, str] = "auto",
            squash_output: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=squash_output,
            device=device,
        )

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        action_dim = get_flattened_obs_dim(self.action_space)

        self.mu = nn.Sequential(
            *create_mlp(
                self.features_dim,
                action_dim,
                net_arch,
                activation_fn,
                squash_output=squash_output,
            )
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Forward pass in actor network."""
        features = self.extract_features(obs, self.features_extractor)
        return self.mu(features)

    def scale_action(self, action: th.Tensor) -> th.Tensor:
        """Scale actions to action space."""
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (action + 1.0) * (high - low))


class SafeCritic(BasePolicy):
    """
    Critic network (Q-value) for Safe TD3 policy with safety value estimation.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            net_arch: List[int],
            activation_fn: Type[nn.Module],
            n_critics: int = 2,
            features_extractor: Optional[BaseFeaturesExtractor] = None,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            device: Union[th.device, str] = "auto",
            share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            device=device,
        )

        action_dim = get_flattened_obs_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []

        for idx in range(n_critics):
            q_net = nn.Sequential(
                *create_mlp(
                    self.features_dim + action_dim,
                    1,
                    net_arch,
                    activation_fn,
                )
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

        # Safety critic (estimates safety value)
        self.safety_critic = nn.Sequential(
            *create_mlp(
                self.features_dim + action_dim,
                1,
                net_arch,
                activation_fn,
            )
        )

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        """Forward pass in critic network."""
        features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)

        # Return all Q-values and safety value
        return tuple(q_net(qvalue_input) for q_net in self.q_networks) + \
               (self.safety_critic(qvalue_input),)


class SafeTD3Policy(BasePolicy):
    """
    Safe TD3 policy class with actor-critic architecture and safety constraints.
    Incorporates barrier certificates for safe action selection.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            barrier_net: Optional[nn.Module] = None,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            safety_margin: float = 0.1,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [400, 300]

        actor_arch, critic_arch = self._get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update({
            "n_critics": n_critics,
            "net_arch": critic_arch,
            "share_features_extractor": share_features_extractor,
        })

        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.barrier_net = barrier_net
        self.safety_margin = safety_margin

        self._build(lr_schedule)

    def _get_actor_critic_arch(
            self,
            net_arch: Union[List[int], Dict[str, List[int]]]
    ) -> Tuple[List[int], List[int]]:
        """Get separate architecture for actor and critic."""
        if isinstance(net_arch, list):
            actor_arch, critic_arch = net_arch, net_arch
        else:
            actor_arch = net_arch.get("pi", [64, 64])
            critic_arch = net_arch.get("qf", [400, 300])
        return actor_arch, critic_arch

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create actor, critic and their target networks.
        """
        self.actor = SafeActor(**self.actor_kwargs).to(self.device)
        self.actor_target = SafeActor(**self.actor_kwargs).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = SafeCritic(**self.critic_kwargs).to(self.device)
        self.critic_target = SafeCritic(**self.critic_kwargs).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Set optimizers
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

        # Set target networks to eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _predict(
            self,
            observation: th.Tensor,
            deterministic: bool = False,
    ) -> th.Tensor:
        """
        Get safe action prediction.
        """
        with th.no_grad():
            features = self.actor.extract_features(observation)
            actions = self.actor(observation)

            # If barrier network is available, ensure safety
            if self.barrier_net is not None:
                safety_values = self.barrier_net(observation)
                unsafe_mask = safety_values > -self.safety_margin

                if unsafe_mask.any():
                    # Get safety critic values for alternative actions
                    n_samples = 10
                    batch_size = observation.shape[0]
                    alternative_actions = th.randn(
                        (batch_size, n_samples, *self.action_space.shape),
                        device=self.device
                    )

                    safety_values = []
                    for i in range(n_samples):
                        safety_value = self.critic.safety_critic(
                            observation,
                            alternative_actions[:, i]
                        )
                        safety_values.append(safety_value)

                    safety_values = th.stack(safety_values, dim=1)

                    # Select safest actions for unsafe states
                    safest_action_idx = safety_values.argmax(dim=1)
                    safe_actions = alternative_actions[
                        th.arange(batch_size),
                        safest_action_idx
                    ]
                    actions[unsafe_mask] = safe_actions[unsafe_mask]

            return self.actor.scale_action(actions)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Forward pass."""
        return self._predict(observation, deterministic=deterministic)

    def evaluate_actions(
            self,
            obs: th.Tensor,
            actions: th.Tensor
    ) -> Tuple[th.Tensor, ...]:
        """
        Evaluate actions for given observations.
        """
        q_values = self.critic(obs, actions)
        return q_values

    def get_safety_critic_value(
            self,
            obs: th.Tensor,
            actions: th.Tensor
    ) -> th.Tensor:
        """
        Get safety critic value for state-action pairs.
        """
        _, _, safety_value = self.critic(obs, actions)
        return safety_value

    def set_training_mode(self, mode: bool) -> None:
        """Set training mode."""
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode