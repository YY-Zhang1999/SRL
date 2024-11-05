import warnings
from typing import Dict, List, Tuple, Type, Union, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import gym

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp
)


class ValueNetwork(nn.Module):
    """
    Value network that estimates state values for safe RL.
    Supports both standard value estimation and safety-aware value estimation.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            features_extractor: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_dim: int = None,
            hidden_sizes: List[int] = [400, 300],
            normalize_images: bool = True,
            device: Union[torch.device, str] = "auto",
            **kwargs
    ):
        """
        Initialize value network.

        Args:
            observation_space: Observation space
            features_extractor: Feature extractor class
            features_dim: Dimension of extracted features
            hidden_sizes: Sizes of hidden layers
            normalize_images: Whether to normalize images
            device: Device for computation
        """
        super().__init__()

        self.normalize_images = normalize_images
        self.features_extractor = features_extractor(observation_space, **kwargs)
        self.features_dim = features_dim or get_flattened_obs_dim(observation_space)

        # Create value network
        self.value_net = self.create_value_network(
            input_dim=self.features_dim,
            hidden_sizes=hidden_sizes
        )

        # Initialize device
        self.device = torch.device(device)
        self.to(self.device)

    def create_value_network(
            self,
            input_dim: int,
            hidden_sizes: List[int]
    ) -> nn.Module:
        """Create value network."""
        value_net = []
        current_dim = input_dim

        for hidden_size in hidden_sizes:
            value_net.extend([
                nn.Linear(current_dim, hidden_size),
                nn.ReLU(),
            ])
            current_dim = hidden_size

        # Output layer (scalar value)
        value_net.append(nn.Linear(current_dim, 1))

        return nn.Sequential(*value_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of value network.

        Args:
            obs: Observation tensor

        Returns:
            values: State values
        """
        features = self.features_extractor(obs)
        values = self.value_net(features)
        return values.squeeze(-1)

    def predict_values(
            self,
            obs: torch.Tensor,
            actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict state values and additional value metrics.

        Args:
            obs: Observation tensor
            actions: Optional action tensor for Q-value estimation

        Returns:
            Dictionary containing:
                - values: State values
                - advantages: Advantages if actions provided
        """
        with torch.no_grad():
            values = self(obs)
            result = {"values": values}

            # If actions provided, compute advantages
            if actions is not None:
                next_values = values[1:]
                current_values = values[:-1]
                rewards = torch.zeros_like(current_values)  # Placeholder for actual rewards
                advantages = rewards + 0.99 * next_values - current_values
                result["advantages"] = advantages

            return result

    def evaluate_values(
            self,
            obs: torch.Tensor,
            target_values: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate value predictions against targets.

        Args:
            obs: Observation tensor
            target_values: Target value tensor

        Returns:
            value_loss: MSE loss between predictions and targets
            metrics: Dictionary of evaluation metrics
        """
        predicted_values = self(obs)
        value_loss = F.mse_loss(predicted_values, target_values)

        with torch.no_grad():
            metrics = {
                "value_loss": value_loss.item(),
                "value_mean": predicted_values.mean().item(),
                "value_std": predicted_values.std().item(),
            }

        return value_loss, metrics