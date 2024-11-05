import warnings
from typing import Dict, List, Tuple, Type, Union, Optional
from functools import partial

import gym
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp
)


class BarrierNetwork(nn.Module):
    """
    Neural Barrier Certificate network that learns a state-based safety measure.
    Compatible with stable-baselines3 architecture and feature extractors.

    The network outputs a scalar value for each state:
    - Negative values indicate safe states
    - Positive values indicate unsafe states
    - Zero level set represents the safety boundary
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            features_extractor: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_dim: int = None,
            hidden_sizes: List[int] = [64, 64],
            normalize_images: bool = True,
            device: Union[torch.device, str] = "auto",
            **kwargs
    ):
        """
        Initialize barrier network.

        Args:
            observation_space: Observation space
            features_extractor: Feature extractor for preprocessing observations
            features_dim: Dimension of extracted features (if None, automatically determined)
            hidden_sizes: Sizes of hidden layers in barrier MLP
            normalize_images: Whether to normalize image inputs
            device: Device to use for computation
            **kwargs: Additional arguments for feature extractor
        """
        super().__init__()

        self.normalize_images = normalize_images
        self.device = get_device(device)

        # Initialize feature extractor
        self.features_extractor = features_extractor(observation_space, **kwargs)
        self.features_dim = features_dim or get_flattened_obs_dim(observation_space)

        # Create barrier MLP
        self.barrier_net = self.create_barrier_network(
            input_dim=self.features_dim,
            hidden_sizes=hidden_sizes
        )

        self.to(self.device)

    def create_barrier_network(
            self,
            input_dim: int,
            hidden_sizes: List[int]
    ) -> nn.Module:
        """
        Create the barrier MLP network.

        Args:
            input_dim: Input dimension
            hidden_sizes: List of hidden layer sizes

        Returns:
            Barrier MLP network
        """
        # Create layers list with activation functions
        layers = []
        current_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.Tanh(),  # Use Tanh for better gradient properties
            ])
            current_dim = hidden_size

        # Output layer (scalar value)
        layers.append(nn.Linear(current_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass of barrier network.

        Args:
            observations: Environment observations

        Returns:
            Barrier values for given states
        """
        # Extract features
        features = self.features_extractor(observations)

        # Compute barrier values
        barrier_values = self.barrier_net(features)

        return barrier_values.squeeze(-1)  # Remove last dimension for scalar output

    def get_safety_measures(
            self,
            observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute safety-related measures for given observations.

        Args:
            observations: Environment observations

        Returns:
            Dictionary containing:
                - barrier_values: Barrier values for states
                - safety_margins: Distance to safety boundary
                - safety_indicators: Binary indicators of safety (1 for safe, 0 for unsafe)
        """
        with torch.no_grad():
            # Compute barrier values
            barrier_values = self(observations)

            # Compute safety margin (distance to zero level set)
            safety_margins = -barrier_values  # Negative barrier value means safe

            # Binary safety indicators
            safety_indicators = (barrier_values <= 0).float()

            return {
                "barrier_values": barrier_values,
                "safety_margins": safety_margins,
                "safety_indicators": safety_indicators
            }

    def compute_penalty_margin(
            self,
            observations: torch.Tensor,
            next_observations: torch.Tensor,
            epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute penalty margin for policy constraint based on barrier values.

        Args:
            observations: Current observations
            next_observations: Next observations
            epsilon: Small constant for numerical stability

        Returns:
            Penalty margins for state transitions
        """
        with torch.no_grad():
            # Get barrier values for current and next states
            barrier_values = self(observations)
            next_barrier_values = self(next_observations)

            # Compute penalty margin based on barrier value change
            penalty_margin = next_barrier_values - (1 - epsilon) * barrier_values

            return penalty_margin

    def predict_safety(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            deterministic: bool = True
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Predict safety of a single observation.

        Args:
            observation: Environment observation
            deterministic: Whether to return deterministic prediction

        Returns:
            is_safe: Boolean indicating if state is predicted as safe
            info: Dictionary with detailed safety measures
        """
        # Convert observation to tensor
        obs_tensor = torch.as_tensor(observation).float().to(self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            # Get safety measures
            safety_measures = self.get_safety_measures(obs_tensor)

            # Convert to numpy and get first element
            is_safe = bool(safety_measures["safety_indicators"][0].cpu().numpy())
            info = {
                k: float(v[0].cpu().numpy())
                for k, v in safety_measures.items()
            }

        return is_safe, info


class ContinuousBarrierNetwork(BarrierNetwork):
    """
    Barrier network for continuous observation spaces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DiscreteBarrierNetwork(BarrierNetwork):
    """
    Barrier network for discrete observation spaces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn("Discrete observation spaces are not fully tested with barrier certificates.")