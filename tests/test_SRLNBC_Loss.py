import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from typing import Tuple

from SRL.core.losses.losses import SRLNBCLoss, LossInfo
from SRL.core.models.barrier import BarrierNetwork
from SRL.core.models.TD3_barrier_policy import SafeTD3Policy


class DummyActor(nn.Module):
    """Dummy actor network for testing."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Dummy log probability computation."""
        mean = self.forward(obs)
        return -0.5 * ((actions - mean) ** 2).sum(dim=-1)


class DummyCritic(nn.Module):
    """Dummy critic network for testing."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, actions], dim=-1)
        return self.q1(x), self.q2(x)


class TestSRLNBCLoss(unittest.TestCase):
    """Test suite for SRLNBCLoss implementation."""

    def setUp(self):
        """Setup test environment and networks."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create environment
        self.env = gym.make("Pendulum-v1")
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Configuration
        self.config = {
            "lambda_barrier": 0.1,
            "n_barrier_steps": 5,
            "gamma_barrier": 0.99,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "policy_delay": 2,
            "device": self.device
        }

        # Initialize networks
        self.current_policy = DummyActor(self.obs_dim, self.action_dim).to(self.device)
        self.old_policy = DummyActor(self.obs_dim, self.action_dim).to(self.device)
        self.barrier_net = BarrierNetwork(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            hidden_sizes=[64, 64]
        ).to(self.device)
        self.critic = DummyCritic(self.obs_dim, self.action_dim).to(self.device)

        # Initialize loss function
        self.loss_fn = SRLNBCLoss(self.config).to(self.device)

        # Create sample batch
        self.batch_size = 32
        self.create_sample_batch()

    def create_sample_batch(self):
        """Create sample batch data for testing."""
        self.batch = {
            "observations": torch.randn(self.batch_size, self.obs_dim).to(self.device),
            "actions": torch.randn(self.batch_size, self.action_dim).to(self.device),
            "next_observations": torch.randn(self.batch_size, self.obs_dim).to(self.device),
            "rewards": torch.randn(self.batch_size, 1).to(self.device),
            "dones": torch.zeros(self.batch_size, 1).to(self.device),
            "episode_mask": torch.zeros(self.batch_size).to(self.device),
            "current_q_values": (
                torch.randn(self.batch_size, 1).to(self.device),
                torch.randn(self.batch_size, 1).to(self.device)
            ),
            "target_q_values": torch.randn(self.batch_size, 1).to(self.device),
            "feasible_mask": torch.zeros(self.batch_size).to(self.device),
            "infeasible_mask": torch.zeros(self.batch_size).to(self.device),
        }

        self.batch["barrier_values"] = self.barrier_net(self.batch["observations"])
        self.batch["next_barrier_values"] = self.barrier_net(self.batch["next_observations"])

        self.batch["feasible_mask"][:self.batch_size // 2] = 1
        self.batch["infeasible_mask"][self.batch_size // 2:] = 1


    def test_importance_weights(self):
        """Test importance weight computation."""
        importance_weights = self.loss_fn.compute_importance_weights(
            self.current_policy,
            self.old_policy,
            self.batch["observations"],
            self.batch["actions"]
        )
        print(importance_weights)

        # Check shape and values
        self.assertTrue(importance_weights >= 0.1)
        self.assertTrue(importance_weights <= 10.0)

    def test_barrier_penalty(self):
        """Test barrier penalty computation."""
        # Compute barrier penalty
        barrier_penalty = self.loss_fn.compute_barrier_penalty(
            self.batch["barrier_values"],
            self.batch["next_barrier_values"],
            self.batch["episode_mask"]
        )

        # Check output
        self.assertTrue(isinstance(barrier_penalty, torch.Tensor))
        self.assertEqual(barrier_penalty.shape, torch.Size([]))
        self.assertTrue(barrier_penalty.item() >= 0)

    def compute_td3_losses(
            self,
            current_q_values: Tuple[torch.Tensor, ...],
            target_q_values: torch.Tensor,
            actor: nn.Module,
            observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute TD3 critic and actor losses.

        Args:
            current_q_values: Current Q-values from both critics
            target_q_values: Target Q-values
            actor: Actor network
            observations: Current observations

        Returns:
            critic_loss: Combined critic loss
            actor_loss: Actor loss
        """
        # Compute critic loss
        critic_loss = sum(F.mse_loss(current_q, target_q_values)
                          for current_q in current_q_values)

        # Compute actor loss (only if policy_delay steps have passed)
        actor_actions = actor(observations)
        actor_loss = -self.critic.forward(
            observations,
            actor_actions
        )[0].mean()

        return critic_loss, actor_loss

    def test_full_loss_computation(self):
        """Test complete loss computation."""
        critic_loss, actor_loss = self.compute_td3_losses(
            self.batch["current_q_values"],
            self.batch["target_q_values"],
            self.current_policy,
            self.batch["observations"],
        )

        total_loss, barrier_loss, loss_info = self.loss_fn(
            self.batch["barrier_values"],
            self.batch["next_barrier_values"],
            self.batch["feasible_mask"],
            self.batch["infeasible_mask"],
            self.batch["observations"],
            self.batch["episode_mask"],
            self.batch["actions"],
            self.current_policy,
            self.old_policy,
            critic_loss,
            actor_loss
        )
        print(total_loss)
        print(loss_info)

        # Check loss output
        self.assertTrue(isinstance(total_loss, torch.Tensor))
        self.assertTrue(isinstance(loss_info, LossInfo))
        #self.assertTrue(total_loss.item() >= 0)

        # Check loss components
        self.assertTrue(loss_info.value_loss >= 0)
        self.assertTrue(loss_info.barrier_loss >= 0)
        self.assertTrue(loss_info.lagrange_multiplier >= 0)

    def test_gradient_flow(self):
        """Test gradient flow through networks."""
        # Initialize optimizers
        policy_optimizer = torch.optim.Adam(self.current_policy.parameters())
        barrier_optimizer = torch.optim.Adam(self.barrier_net.parameters())
        critic_optimizer = torch.optim.Adam(self.critic.parameters())

        # Compute loss
        critic_loss, actor_loss = self.compute_td3_losses(
            self.batch["current_q_values"],
            self.batch["target_q_values"],
            self.current_policy,
            self.batch["observations"],
        )
        print(critic_loss)

        total_loss, barrier_loss, loss_info = self.loss_fn(
            self.batch["barrier_values"],
            self.batch["next_barrier_values"],
            self.batch["feasible_mask"],
            self.batch["infeasible_mask"],
            self.batch["observations"],
            self.batch["episode_mask"],
            self.batch["actions"],
            self.current_policy,
            self.old_policy,
            critic_loss,
            actor_loss
        )

        # Check gradient flow
        critic_optimizer.zero_grad()
        critic_loss.requires_grad = True
        critic_loss.backward()
        critic_optimizer.step()

        policy_optimizer.zero_grad()
        total_loss.backward()
        policy_optimizer.step()

        barrier_optimizer.zero_grad()
        #barrier_loss.requires_grad = True
        barrier_loss.backward()
        barrier_optimizer.step()


        # Check gradients
        for net in [self.current_policy, self.barrier_net]:
            for param in net.parameters():
                self.assertTrue(param.grad is not None)
                self.assertFalse(torch.allclose(param.grad, torch.tensor(0.0).to(self.device)))


    def test_lagrange_multiplier_update(self):
        """Test Lagrange multiplier update."""
        initial_multiplier = self.loss_fn.lagrange_multiplier.item()

        # Compute loss multiple times
        for _ in range(5):
            critic_loss, actor_loss = self.compute_td3_losses(
                self.batch["current_q_values"],
                self.batch["target_q_values"],
                self.current_policy,
                self.batch["observations"],
            )

            self.loss_fn(
                self.batch["barrier_values"],
                self.batch["next_barrier_values"],
                self.batch["feasible_mask"],
                self.batch["infeasible_mask"],
                self.batch["observations"],
                self.batch["episode_mask"],
                self.batch["actions"],
                self.current_policy,
                self.old_policy,
                critic_loss,
                actor_loss
            )

            final_multiplier = self.loss_fn.lagrange_multiplier.item()

            # Check multiplier update
            self.assertNotEqual(initial_multiplier, final_multiplier)
            self.assertTrue(final_multiplier >= 0)



    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Create batch with extreme values
        extreme_batch = {
            **self.batch,
            "observations": 1e6 * torch.ones_like(self.batch["observations"]),
            "actions": 1e6 * torch.ones_like(self.batch["actions"]),
        }

        # Compute loss
        critic_loss, actor_loss = self.compute_td3_losses(
            self.batch["current_q_values"],
            self.batch["target_q_values"],
            self.current_policy,
            self.batch["observations"],
        )

        total_loss, barrier_loss, loss_info = self.loss_fn(
            self.batch["barrier_values"],
            self.batch["next_barrier_values"],
            self.batch["feasible_mask"],
            self.batch["infeasible_mask"],
            self.batch["observations"],
            self.batch["episode_mask"],
            self.batch["actions"],
            self.current_policy,
            self.old_policy,
            critic_loss,
            actor_loss
        )

        # Check for numerical stability
        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))
        self.assertFalse(torch.isnan(barrier_loss))
        self.assertFalse(torch.isinf(barrier_loss))
        self.assertTrue(np.isfinite(loss_info.total_loss))


if __name__ == '__main__':
    unittest.main()