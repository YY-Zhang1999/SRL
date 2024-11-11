import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
from .barrier_loss import BarrierLoss, PolicyLoss, ValueLoss

@dataclass
class LossInfo:
    """
    Container for storing loss values and related metrics.
    """
    # Total losses
    total_loss: float
    policy_loss: float
    value_loss: float
    barrier_loss: float

    # Policy loss components
    surrogate_loss: float = 0.0
    barrier_penalty: float = 0.0
    entropy_bonus: float = 0.0
    kl_loss: float = 0.0

    # Barrier loss components
    feasible_loss: float = 0.0
    infeasible_loss: float = 0.0
    invariant_loss: float = 0.0

    # Additional metrics
    policy_gradient_norm: float = 0.0
    value_gradient_norm: float = 0.0
    barrier_gradient_norm: float = 0.0
    lagrange_multiplier: float = 0.0


class PPONBCLoss(nn.Module):
    """
    Unified interface for SRLNBC (Safe RL with Neural Barrier Certificate) losses.
    Combines policy, value, and barrier losses with proper weighting and logging.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize loss components and weights.

        Args:
            config: Configuration dictionary containing loss parameters
        """
        super().__init__()

        # Initialize individual loss functions
        self.barrier_loss = BarrierLoss(config)
        self.policy_loss = PolicyLoss(config)
        self.value_loss = ValueLoss(config)

        # Loss coefficients
        self.vf_loss_coeff = config.get("vf_loss_coeff", 0.5)
        self.barrier_loss_coeff = config.get("lambda_barrier", 0.1)

        # Initialize Lagrange multiplier
        self.lagrange_multiplier = torch.nn.Parameter(
            torch.tensor(config.get("lambda_init", 1.0))
        )
        self.lambda_lr = config.get("lambda_learning_rate", 1e-3)

    def compute_losses(
            self,
            batch: Dict[str, torch.Tensor],
            policy_outputs: Dict[str, torch.Tensor],
            barrier_outputs: Dict[str, torch.Tensor],
            value_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, LossInfo]:
        """
        Compute all losses and combine them with proper weighting.

        Args:
            batch: Dictionary containing training batch data
            policy_outputs: Dictionary containing policy network outputs
            barrier_outputs: Dictionary containing barrier network outputs
            value_outputs: Dictionary containing value network outputs

        Returns:
            total_loss: Combined loss for optimization
            loss_info: Detailed loss information and metrics
        """
        # Compute barrier loss
        barrier_loss, barrier_info = self.barrier_loss(
            barrier_values=barrier_outputs["barrier_values"],
            next_barrier_values=barrier_outputs["next_barrier_values"],
            feasible_mask=batch["feasible_mask"],
            infeasible_mask=batch["infeasible_mask"],
            episode_mask=batch["episode_mask"]
        )

        # Compute policy loss
        policy_loss, policy_info = self.policy_loss.compute_policy_loss(
            pi_logp_ratio=policy_outputs["log_ratio"],
            advantages=batch["advantages"],
            penalty_margin=barrier_outputs["penalty_margin"],
            lagrange_multiplier=self.lagrange_multiplier,
            entropy=policy_outputs["entropy"],
            kl_div=policy_outputs.get("kl_divergence", None)
        )

        # Compute value loss
        value_loss, value_info = self.value_loss(
            values=value_outputs["values"],
            old_values=batch["old_values"],
            value_targets=batch["value_targets"]
        )

        # Combine losses
        total_loss = (
                policy_loss +
                self.vf_loss_coeff * value_loss +
                self.barrier_loss_coeff * barrier_loss
        )

        # Create loss info container
        loss_info = LossInfo(
            total_loss=total_loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            barrier_loss=barrier_loss.item(),
            surrogate_loss=policy_info["surrogate_loss"],
            barrier_penalty=policy_info["barrier_penalty"],
            entropy_bonus=policy_info["entropy_bonus"],
            kl_loss=policy_info["kl_loss"],
            feasible_loss=barrier_info["feasible_loss"],
            infeasible_loss=barrier_info["infeasible_loss"],
            invariant_loss=barrier_info["invariant_loss"],
            lagrange_multiplier=self.lagrange_multiplier.item()
        )

        return total_loss, loss_info

    def update_lagrange_multiplier(self, constraint_value: torch.Tensor):
        """
        Update Lagrange multiplier using gradient ascent.

        Args:
            constraint_value: Current value of constraint violation
        """
        with torch.no_grad():
            self.lagrange_multiplier.add_(self.lambda_lr * constraint_value)
            self.lagrange_multiplier.clamp_(min=0.0)  # Ensure non-negativity

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics for logging.

        Returns:
            Dictionary of metrics
        """
        return {
            "lagrange_multiplier": self.lagrange_multiplier.item()
        }


class SRLNBCLoss(nn.Module):
    """
    SRLNBC Loss implementation for TD3 with importance sampling and Lagrangian relaxation.
    Implements Equations (13) and (14) from the paper.
    """

    def __init__(
            self,
            config: Dict[str, Any],
    ):
        """
        Initialize loss components.

        Args:
            config: Configuration dictionary containing:
                - lambda_barrier: Weight for barrier loss
                - n_barrier_steps: Number of steps for multi-step barrier
                - gamma_barrier: Discount factor for barrier steps
                - target_policy_noise: Target policy noise std
                - target_noise_clip: Target policy noise clip
                - policy_delay: Policy update delay steps
                - device: Computation device
        """
        super().__init__()
        self.lambda_barrier = config["lambda_barrier"]
        #self.n_barrier_steps = config["n_barrier_steps"]
        self.n_barrier_steps = 1
        self.gamma_barrier = config["gamma_barrier"]
        self.target_policy_noise = config["target_policy_noise"]
        self.target_noise_clip = config["target_noise_clip"]
        self.policy_delay = config["policy_delay"]
        self.device = config["device"]

        # Initialize Lagrange multiplier
        self.lagrange_multiplier = nn.Parameter(torch.tensor(1.0))
        self.lambda_lr = 1e-3

        # Initialize individual loss functions
        self.barrier_loss = BarrierLoss(config)

    def compute_importance_weights(
            self,
            current_policy: nn.Module,
            old_policy: nn.Module,
            observations: torch.Tensor,
            actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance sampling weights.

        Args:
            current_policy: Current policy network
            old_policy: Old policy network that collected the data
            observations: Current observations
            actions: Current actions

        Returns:
            importance_weights: Importance sampling weights
        """

        with torch.no_grad():
            current_log_prob = self._get_log_prob(current_policy, observations, actions)
            old_log_prob = self._get_log_prob(old_policy, observations, actions)
            importance_weights = torch.exp(current_log_prob - old_log_prob)

            # Clip importance weights for stability
            importance_weights = torch.clamp(importance_weights, 0.1, 10.0)

        return importance_weights.mean()

    def _get_log_prob(self, policy: nn.Module, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log probability computation."""
        mean = policy.forward(obs)
        return -0.5 * ((actions - mean) ** 2).sum(dim=-1)

    def compute_barrier_penalty(
            self,
            barrier_values: torch.Tensor,
            next_barrier_values: torch.Tensor,
            episode_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute barrier penalty with multi-step invariant loss.

        Args:
            barrier_values: Current barrier values of current observations
            next_barrier_values: Next barrier values of next observations
            observations: Current observations
            next_observations: Next observations
            importance_weights: Importance sampling weights
            episode_mask: Episode boundary mask

        Returns:
            barrier_penalty: Weighted barrier penalty
        """

        barrier_penalty = self.barrier_loss.invariant_loss(
            barrier_values=barrier_values,
            next_barrier_values=next_barrier_values,
            episode_mask=episode_mask
        )
        return barrier_penalty

    def compute_barrier_loss(self,
                             barrier_values: torch.Tensor,
                             next_barrier_values: torch.Tensor,
                             feasible_mask: torch.Tensor,
                             infeasible_mask: torch.Tensor,
                             episode_mask: torch.Tensor
                             ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total barrier loss and individual components.

        Args:
            barrier_values: Current state barrier values
            next_barrier_values: Next state barrier values
            feasible_mask: Binary mask indicating feasible states
            infeasible_mask: Binary mask indicating infeasible states
            episode_mask: Binary mask indicating episode boundaries

        Returns:
            total_loss: Combined barrier loss
            loss_dict: Dictionary containing individual loss components
        """
        return self.barrier_loss(
            barrier_values=barrier_values,
            next_barrier_values=next_barrier_values,
            feasible_mask=feasible_mask,
            infeasible_mask=infeasible_mask,
            episode_mask=episode_mask
        )


    def forward(
            self,
            barrier_values: torch.Tensor,
            next_barrier_values: torch.Tensor,
            feasible_mask: torch.Tensor,
            infeasible_mask: torch.Tensor,
            observations: torch.Tensor,
            episode_mask: torch.Tensor,
            actions: torch.Tensor,
            current_policy: nn.Module,
            old_policy: nn.Module,
            critic_loss: torch.Tensor,
            actor_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, LossInfo]:
        """
        Compute combined loss using Lagrangian relaxation.

        Args:
            barrier_values: Current state barrier values
            next_barrier_values: Next state barrier values
            feasible_mask: Binary mask indicating feasible states
            infeasible_mask: Binary mask indicating infeasible states
            current_policy: Current policy network
            old_policy: Old policy network

        Returns:
            total_loss: Combined loss for optimization
            barrier_loss: total barrier loss
            loss_info: Detailed loss information
        """

        # Compute importance weights
        importance_weights = self.compute_importance_weights(
            current_policy=current_policy,
            old_policy=old_policy,
            observations=observations,
            actions=actions
        )

        # Compute barrier penalty
        barrier_loss, barrier_info = self.compute_barrier_loss(
            barrier_values=barrier_values,
            next_barrier_values=next_barrier_values,
            feasible_mask=feasible_mask,
            infeasible_mask=infeasible_mask,
            episode_mask=episode_mask
        )

        barrier_penalty = importance_weights * barrier_info["invariant_loss"]

        # Compute Lagrangian dual loss
        dual_loss = self.lagrange_multiplier * barrier_penalty

        # Combine losses using Lagrangian relaxation
        actor_loss = torch.as_tensor(actor_loss, device=self.device)
        total_loss = actor_loss + dual_loss

        # Update Lagrange multiplier
        with torch.no_grad():
            self.lagrange_multiplier.data.add_(
                self.lambda_lr * total_loss.detach()
            )
            self.lagrange_multiplier.data.clamp_(min=0.0)

        # Create loss info
        loss_info = LossInfo(
            total_loss=total_loss.item(),
            policy_loss=actor_loss.item(),
            value_loss=critic_loss.item(),
            barrier_loss=barrier_loss.item(),
            barrier_penalty=barrier_penalty.item(),
            feasible_loss=barrier_info["feasible_loss"],
            infeasible_loss=barrier_info["infeasible_loss"],
            invariant_loss=barrier_info["invariant_loss"],
            lagrange_multiplier=self.lagrange_multiplier.item(),
        )

        return total_loss, barrier_loss, loss_info

class SRLNBCLossWrapper:
    """
    A wrapper class that provides a simplified interface for loss computation
    and handles device management and gradient computation.
    """

    def __init__(
            self,
            config: Dict[str, Any],
            device: torch.device
    ):
        """
        Initialize the loss wrapper.

        Args:
            config: Configuration dictionary
            device: Device to use for computation
        """
        self.loss_fn = SRLNBCLoss(config).to(device)
        self.device = device

    def __call__(
            self,
            batch: Dict[str, torch.Tensor],
            policy_net: nn.Module,
            barrier_net: nn.Module,
            value_net: nn.Module
    ) -> Tuple[torch.Tensor, LossInfo]:
        """
        Compute losses and gradients for all networks.

        Args:
            batch: Training batch data
            policy_net: Policy network
            barrier_net: Barrier network
            value_net: Value network

        Returns:
            total_loss: Combined loss
            loss_info: Detailed loss information
        """
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # Compute network outputs
        policy_outputs = policy_net(batch)
        barrier_outputs = barrier_net(batch)
        value_outputs = value_net(batch)

        # Compute losses
        total_loss, loss_info = self.loss_fn.compute_losses(
            batch, policy_outputs, barrier_outputs, value_outputs
        )

        # Update Lagrange multiplier
        self.loss_fn.update_lagrange_multiplier(barrier_outputs["constraint_value"])

        return total_loss, loss_info

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        return self.loss_fn.get_metrics()

