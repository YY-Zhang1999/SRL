import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass


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


class SRLNBCLoss(nn.Module):
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

        from .barrier_loss import BarrierLoss, PolicyLoss, ValueLoss

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

