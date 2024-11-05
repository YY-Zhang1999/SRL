from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import torch
import numpy as np


@dataclass
class DefaultConfig:
    """
    Default configuration for SRLNBC (Safe RL with Neural Barrier Certificate).
    Inherits basic parameters from TD3 and adds barrier certificate specific parameters.
    """

    # Basic RL parameters (from TD3)
    learning_rate: float = 3e-4  # Learning rate for all networks
    buffer_size: int = 1_000_000  # Replay buffer size
    learning_starts: int = 100  # Number of steps before training starts
    batch_size: int = 256  # Batch size for training
    tau: float = 0.005  # Soft update coefficient for target networks
    gamma: float = 0.99  # Discount factor
    train_freq: int = 1  # Update policy every train_freq steps
    gradient_steps: int = 1  # Number of gradient steps per update
    action_noise_std: float = 0.1  # Std of Gaussian action noise
    policy_delay: int = 2  # Number of steps between policy network updates
    target_policy_noise: float = 0.2  # Noise added to target policy
    target_noise_clip: float = 0.5  # Max value of target policy noise

    # Neural network architecture
    net_arch: Dict[str, list] = None  # Network architectures
    activation_fn: str = "ReLU"  # Activation function

    # Barrier certificate parameters
    barrier_lr: float = 1e-4  # Learning rate for barrier network
    barrier_hidden_sizes: list = None  # Hidden layer sizes for barrier network
    barrier_update_freq: int = 1  # Update barrier every N steps
    barrier_batch_size: int = 512  # Batch size for barrier updates
    lambda_barrier: float = 0.1  # Weight for barrier loss

    # Multi-step barrier parameters
    n_barrier_steps: int = 20  # Number of steps for multi-step barrier
    gamma_barrier: float = 0.99  # Discount factor for multi-step barrier

    # Lagrangian parameters
    lambda_init: float = 1.0  # Initial Lagrange multiplier
    lambda_learning_rate: float = 1e-3  # Learning rate for Lagrange multiplier

    # Training parameters
    total_timesteps: int = 1_000_000  # Total number of training steps
    eval_freq: int = 10000  # Evaluate policy every N steps
    n_eval_episodes: int = 10  # Number of episodes for evaluation
    max_episode_steps: int = 1000  # Maximum steps per episode
    save_freq: int = 100000  # Save model every N steps

    # Device and logging
    device: str = "auto"  # Device to use (auto, cpu, cuda)
    tensorboard_log: Optional[str] = None  # TensorBoard log directory
    verbose: int = 1  # Verbosity level
    seed: Optional[int] = None  # Random seed

    def __post_init__(self):
        """Initialize default values that depend on other parameters."""
        # Set default network architectures if not specified
        if self.net_arch is None:
            self.net_arch = {
                "policy": [400, 300],  # Actor network
                "qf": [400, 300],  # Critic network
                "barrier": [400, 300]  # Barrier network
            }

        if self.barrier_hidden_sizes is None:
            self.barrier_hidden_sizes = [400, 300]

        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def policy_kwargs(self) -> Dict[str, Any]:
        """Get policy network keyword arguments."""
        return {
            "net_arch": self.net_arch,
            "activation_fn": getattr(torch.nn, self.activation_fn),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DefaultConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        import json
        config_dict = self.to_dict()
        # Convert non-serializable objects to strings
        for k, v in config_dict.items():
            if isinstance(v, type):
                config_dict[k] = v.__name__
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> 'DefaultConfig':
        """Load configuration from file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        # Convert activation function string back to class
        if 'activation_fn' in config_dict:
            config_dict['activation_fn'] = getattr(torch.nn, config_dict['activation_fn'])
        return cls.from_dict(config_dict)


# Environment specific configurations
@dataclass
class SafetyGymConfig(DefaultConfig):
    """Configuration for Safety Gym environments."""

    def __post_init__(self):
        super().__post_init__()
        self.net_arch = {
            "policy": [256, 256],
            "qf": [256, 256],
            "barrier": [256, 256]
        }
        self.barrier_hidden_sizes = [256, 256]
        self.max_episode_steps = 1000
        self.learning_rate = 1e-3
        self.barrier_lr = 1e-4
        self.lambda_barrier = 0.1
        self.n_barrier_steps = 20


@dataclass
class MetaDriveConfig(DefaultConfig):
    """Configuration for MetaDrive environment."""

    def __post_init__(self):
        super().__post_init__()
        self.net_arch = {
            "policy": [512, 256],
            "qf": [512, 256],
            "barrier": [512, 256]
        }
        self.barrier_hidden_sizes = [512, 256]
        self.max_episode_steps = 1000
        self.learning_rate = 5e-4
        self.barrier_lr = 1e-4
        self.lambda_barrier = 0.2
        self.n_barrier_steps = 20


# Create default configurations
#default_config = DefaultConfig()
#safety_gym_config = SafetyGymConfig()
#metadrive_config = MetaDriveConfig()

if __name__ == '__main__':

    config = DefaultConfig()


    safety_config = SafetyGymConfig()


    safety_config.learning_rate = 1e-3
    safety_config.barrier_lr = 1e-4

    safety_config.save("safety_gym_config.json")

    loaded_config = DefaultConfig.load("safety_gym_config.json")
