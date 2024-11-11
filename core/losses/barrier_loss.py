import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np


class BarrierLoss:
    """
    Implementation of Neural Barrier Certificate losses including:
    - Feasible loss: penalizes positive barrier values in feasible regions
    - Infeasible loss: penalizes negative barrier values in infeasible regions
    - Invariant loss: ensures barrier property across state transitions
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize barrier loss components.

        Args:
            config: Configuration dictionary containing barrier parameters
        """
        self.epsilon = 1e-8  # Small constant for numerical stability
        self.lambda_barrier = config["lambda_barrier"]
        self.n_barrier_steps = config["n_barrier_steps"]
        self.gamma_barrier = config["gamma_barrier"]

    def feasible_loss(self, barrier_values: torch.Tensor, feasible_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for feasible states (barrier value should be <= 0).

        Args:
            barrier_values: Barrier function values
            feasible_mask: Binary mask indicating feasible states

        Returns:
            Feasible region loss
        """
        loss = torch.maximum(barrier_values, torch.zeros_like(barrier_values))
        weighted_loss = feasible_mask * loss
        # Normalize by number of feasible states
        num_feasible = torch.maximum(torch.sum(feasible_mask), torch.ones_like(torch.sum(feasible_mask)))
        return torch.sum(weighted_loss) / num_feasible

    def infeasible_loss(self, barrier_values: torch.Tensor, infeasible_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for infeasible states (barrier value should be > 0).

        Args:
            barrier_values: Barrier function values
            infeasible_mask: Binary mask indicating infeasible states

        Returns:
            Infeasible region loss
        """
        loss = torch.maximum(-barrier_values, torch.zeros_like(barrier_values))
        weighted_loss = infeasible_mask * loss
        # Normalize by number of infeasible states
        num_infeasible = torch.maximum(torch.sum(infeasible_mask), torch.ones_like(torch.sum(infeasible_mask)))
        return torch.sum(weighted_loss) / num_infeasible

    def invariant_loss(
            self,
            barrier_values: torch.Tensor,
            next_barrier_values: torch.Tensor,
            episode_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-step invariant loss ensuring barrier property across transitions.

        Args:
            barrier_values: Current state barrier values
            next_barrier_values: Next state barrier values
            episode_mask: Binary mask indicating episode boundaries

        Returns:
            Multi-step invariant loss
        """
        total_invariant_loss = 0
        batch_size = barrier_values.shape[0]

        # Convert episode mask to boolean and handle shifting
        mask = episode_mask.bool()

        for i in range(self.n_barrier_steps):
            # Compute target coefficient based on barrier lambda
            target_coeff = (1 - self.lambda_barrier) ** (i + 1)

            # Compute epsilon term
            epsilon_term = self.epsilon * (1 - target_coeff) / self.lambda_barrier

            # Get relevant slices of barrier values
            future_barriers = next_barrier_values
            current_barriers = barrier_values

            # Compute invariant constraint violation
            inv_loss = torch.maximum(
                epsilon_term + future_barriers - target_coeff * current_barriers,
                torch.zeros_like(current_barriers)
            )

            # Apply episode boundary mask
            inv_loss = (1 - mask.float()) * inv_loss

            # Update mask for next step
            #if i < self.n_barrier_steps - 1:
            #    mask = mask[:-1] | mask[1:]

            # Compute mean loss for this step
            step_loss = torch.mean(inv_loss)

            # Add to total loss with gamma decay
            total_invariant_loss += (self.gamma_barrier ** i) * step_loss

        # Normalize multi-step loss
        normalization = (1 - self.gamma_barrier) / (1 - self.gamma_barrier ** self.n_barrier_steps)
        total_invariant_loss *= normalization

        return total_invariant_loss

    def __call__(
            self,
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
        # Compute individual losses
        feas_loss = self.feasible_loss(barrier_values, feasible_mask)
        infeas_loss = self.infeasible_loss(barrier_values, infeasible_mask)
        inv_loss = self.invariant_loss(barrier_values, next_barrier_values, episode_mask)

        # Combine losses
        total_loss = feas_loss + infeas_loss + inv_loss

        # Create loss dictionary
        loss_dict = {
            "feasible_loss": feas_loss.item(),
            "infeasible_loss": infeas_loss.item(),
            "invariant_loss": inv_loss.item(),
            "barrier_loss": total_loss.item()
        }

        return total_loss, loss_dict


class PolicyLoss:
    """
    Implementation of policy loss including:
    - PPO surrogate loss
    - Barrier constraint through Lagrangian relaxation
    - KL divergence penalty
    - Entropy bonus
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize policy loss components.

        Args:
            config: Configuration dictionary containing policy parameters
        """
        self.clip_param = config.get("clip_param", 0.2)
        self.kl_coeff = config.get("kl_coeff", 0.0)
        self.entropy_coeff = config.get("entropy_coeff", 0.01)
        self.vf_loss_coeff = config.get("vf_loss_coeff", 0.5)

    def compute_ppo_surrogate(
            self,
            advantages: torch.Tensor,
            log_ratio: torch.Tensor,
            ratio: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PPO surrogate objective.

        Args:
            advantages: Advantage values
            log_ratio: Log of action probabilities ratio (new/old)
            ratio: Action probabilities ratio (new/old)

        Returns:
            PPO surrogate loss
        """
        surrogate1 = advantages * ratio
        surrogate2 = advantages * torch.clamp(
            ratio,
            1.0 - self.clip_param,
            1.0 + self.clip_param
        )
        return -torch.mean(torch.min(surrogate1, surrogate2))

    def compute_barrier_penalty(
            self,
            ratio: torch.Tensor,
            penalty_margin: torch.Tensor,
            lagrange_multiplier: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute barrier constraint penalty using Lagrangian relaxation.

        Args:
            ratio: Action probabilities ratio (new/old)
            penalty_margin: Barrier penalty margin
            lagrange_multiplier: Lagrange multiplier for constraint

        Returns:
            Barrier penalty loss
        """
        # Normalize penalty margin
        penalty_margin = (penalty_margin - torch.mean(penalty_margin)) / \
                         (torch.std(penalty_margin) + 1e-8)

        surrogate_cost = -torch.mean(ratio * penalty_margin)
        penalty = lagrange_multiplier.detach()

        return surrogate_cost * penalty

    def compute_policy_loss(
            self,
            pi_logp_ratio: torch.Tensor,
            advantages: torch.Tensor,
            penalty_margin: torch.Tensor,
            lagrange_multiplier: torch.Tensor,
            entropy: torch.Tensor,
            kl_div: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total policy loss and components.

        Args:
            pi_logp_ratio: Log of action probabilities ratio (new/old)
            advantages: Advantage values
            penalty_margin: Barrier penalty margin
            lagrange_multiplier: Lagrange multiplier
            entropy: Policy entropy
            kl_div: KL divergence between old and new policy

        Returns:
            total_loss: Combined policy loss
            loss_dict: Dictionary containing individual loss components
        """
        ratio = torch.exp(pi_logp_ratio)

        # Compute loss components
        surrogate_loss = self.compute_ppo_surrogate(advantages, pi_logp_ratio, ratio)
        barrier_penalty = self.compute_barrier_penalty(ratio, penalty_margin, lagrange_multiplier)
        entropy_bonus = -self.entropy_coeff * torch.mean(entropy)

        # Initialize policy loss
        policy_loss = surrogate_loss + barrier_penalty + entropy_bonus

        # Add KL penalty if needed
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if self.kl_coeff > 0.0 and kl_div is not None:
            kl_loss = self.kl_coeff * torch.mean(kl_div)
            policy_loss += kl_loss

        # Create loss dictionary
        loss_dict = {
            "surrogate_loss": surrogate_loss.item(),
            "barrier_penalty": barrier_penalty.item(),
            "entropy_bonus": entropy_bonus.item(),
            "kl_loss": kl_loss.item(),
            "total_policy_loss": policy_loss.item()
        }

        return policy_loss, loss_dict


class ValueLoss:
    """
    Implementation of value function loss using clipped MSE loss as in PPO.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize value loss parameters.

        Args:
            config: Configuration dictionary containing value function parameters
        """
        self.vf_clip_param = config.get("vf_clip_param", 10.0)

    def __call__(
            self,
            values: torch.Tensor,
            old_values: torch.Tensor,
            value_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute clipped value function loss.

        Args:
            values: Current value function predictions
            old_values: Old value function predictions
            value_targets: Target values

        Returns:
            total_loss: Value function loss
            loss_dict: Dictionary containing loss information
        """
        # Compute unclipped loss
        value_loss1 = F.mse_loss(values, value_targets, reduction='none')

        # Compute clipped loss
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.vf_clip_param,
            self.vf_clip_param
        )
        value_loss2 = F.mse_loss(value_pred_clipped, value_targets, reduction='none')

        # Take maximum of clipped and unclipped loss
        value_loss = torch.max(value_loss1, value_loss2)
        value_loss = torch.mean(value_loss)

        loss_dict = {"value_loss": value_loss.item()}

        return value_loss, loss_dict


