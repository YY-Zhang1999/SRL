import warnings
from typing import Dict, List, Tuple, Type, Union, Optional, Any
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import gym
from gym import spaces
import torch.nn.functional as F

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    create_mlp
)


class SafePolicy(BasePolicy):
    """
    Policy network for safe RL that incorporates barrier certificate constraints.
    Extends SB3's BasePolicy with safety-aware action distributions.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_dim: int = None,
            hidden_sizes: List[int] = [400, 300],
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            device: Union[torch.device, str] = "auto",
            **kwargs
    ):
        """
        Initialize safe policy network.

        Args:
            observation_space: Observation space
            action_space: Action space
            features_extractor: Feature extractor class
            features_dim: Dimension of extracted features
            hidden_sizes: Sizes of hidden layers
            normalize_images: Whether to normalize images
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer kwargs
            device: Device for computation
        """
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            **kwargs
        )

        self.features_dim = features_dim or get_flattened_obs_dim(observation_space)

        # Action distribution parameters
        if isinstance(action_space, spaces.Box):
            self.action_dim = np.prod(action_space.shape)
            self.action_dist = self.make_action_dist()
        else:
            raise NotImplementedError(f"Unsupported action space: {type(action_space)}")

        # Create policy network
        self.policy_net = self.create_policy_network(
            input_dim=self.features_dim,
            hidden_sizes=hidden_sizes,
            output_dim=2 * self.action_dim  # Mean and log_std
        )

        self.to(self.device)

    def create_policy_network(
            self,
            input_dim: int,
            hidden_sizes: List[int],
            output_dim: int
    ) -> nn.Module:
        """Create policy network."""
        policy_net = []
        current_dim = input_dim

        for hidden_size in hidden_sizes:
            policy_net.extend([
                nn.Linear(current_dim, hidden_size),
                nn.ReLU(),
            ])
            current_dim = hidden_size

        policy_net.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*policy_net)

    def make_action_dist(self) -> Distribution:
        """Create action distribution."""
        from stable_baselines3.common.distributions import (
            SquashedDiagGaussianDistribution,
            StateDependentNoiseDistribution
        )
        return SquashedDiagGaussianDistribution(self.action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of policy network.

        Args:
            obs: Observation tensor

        Returns:
            mean: Action mean
            log_std: Log standard deviation
        """
        features = self.extract_features(obs)
        policy_latent = self.policy_net(features)
        mean, log_std = torch.chunk(policy_latent, 2, dim=-1)
        return mean, log_std

    def evaluate_actions(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations.

        Args:
            obs: Observation tensor
            actions: Action tensor

        Returns:
            log_prob: Log probability of actions
            entropy: Policy entropy
            mean: Action mean
        """
        mean, log_std = self.forward(obs)
        distribution = self.action_dist.proba_distribution(mean, log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy, mean

    def get_actions(
            self,
            obs: torch.Tensor,
            deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get actions for given observations.

        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions

        Returns:
            actions: Selected actions
            log_prob: Log probability of actions
            mean: Action mean
        """
        mean, log_std = self.forward(obs)
        distribution = self.action_dist.proba_distribution(mean, log_std)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)

        return actions, log_prob, {"mean": mean, "log_std": log_std}


