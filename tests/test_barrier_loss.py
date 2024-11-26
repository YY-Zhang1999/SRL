import unittest
import torch
import numpy as np
from SRL.core.losses.barrier_loss import BarrierLoss
import torch.nn.functional as F

class TestBarrierLoss(unittest.TestCase):
    """
    Test suite for BarrierLoss implementation.
    Tests individual components and full loss computation.
    """

    def setUp(self):
        """Setup test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            "lambda_barrier": 0.1,
            "n_barrier_steps": 4,
            "gamma_barrier": 0.99,
        }
        self.loss_fn = BarrierLoss(self.config)

        # Create sample test data
        self.batch_size = 4
        self.create_test_data()

    def create_test_data(self):
        """Create test data for loss computation."""
        # Create barrier values for feasible and infeasible states
        self.barrier_values = torch.randn(self.batch_size).to(self.device)
        self.next_barrier_values = torch.randn(self.config["n_barrier_steps"] + 1, self.batch_size).to(self.device)
        self.next_barrier_values[0] = self.barrier_values

        # Create masks
        self.feasible_mask = torch.zeros(self.batch_size).to(self.device)
        self.feasible_mask[:self.batch_size // 2] = 1  # Half states are feasible

        self.infeasible_mask = torch.zeros(self.batch_size).to(self.device)
        self.infeasible_mask[self.batch_size // 2:] = 1  # Half states are infeasible

        self.episode_mask = torch.zeros(self.config["n_barrier_steps"] + 1, self.batch_size).to(self.device)
        for i in range(self.config["n_barrier_steps"] + 1):
            self.episode_mask[i][self.batch_size // 4 + i::self.batch_size // 4] = 1  # Episode boundaries


    def test_feasible_loss(self):
        return
        """Test feasible loss computation."""
        # Test case 1: All feasible states have negative barrier values
        barrier_values = -torch.ones(self.batch_size).to(self.device)
        feasible_mask = torch.ones(self.batch_size).to(self.device)
        loss = self.loss_fn.feasible_loss(barrier_values, feasible_mask)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0).to(self.device)))

        # Test case 2: All feasible states have positive barrier values
        barrier_values = torch.ones(self.batch_size).to(self.device)
        loss = self.loss_fn.feasible_loss(barrier_values, feasible_mask)
        self.assertTrue(loss.item() > 0)

        # Test case 3: Mixed barrier values
        barrier_values = torch.randn(self.batch_size).to(self.device)
        loss = self.loss_fn.feasible_loss(barrier_values, feasible_mask)
        self.assertTrue(loss.item() >= 0)

    def test_infeasible_loss(self):
        return
        """Test infeasible loss computation."""
        # Test case 1: All infeasible states have positive barrier values
        barrier_values = torch.ones(self.batch_size).to(self.device)
        infeasible_mask = torch.ones(self.batch_size).to(self.device)
        loss = self.loss_fn.infeasible_loss(barrier_values, infeasible_mask)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0).to(self.device)))

        # Test case 2: All infeasible states have negative barrier values
        barrier_values = -torch.ones(self.batch_size).to(self.device)
        loss = self.loss_fn.infeasible_loss(barrier_values, infeasible_mask)
        self.assertTrue(loss.item() > 0)

        # Test case 3: Mixed barrier values
        barrier_values = torch.randn(self.batch_size).to(self.device)
        loss = self.loss_fn.infeasible_loss(barrier_values, infeasible_mask)
        self.assertTrue(loss.item() >= 0)

    def test_invariant_loss(self):
        """Test invariant loss computation."""
        # Test case 1: Perfect invariance
        barrier_values = torch.ones(self.batch_size).to(self.device)

        next_n_barrier_values = torch.ones(self.config["n_barrier_steps"] + 1, self.batch_size).to(self.device)
        for i in range(self.config["n_barrier_steps"]):
            next_n_barrier_values[i + 1] = (1 - self.config["lambda_barrier"]) ** (i + 1) * barrier_values

        episode_mask = torch.zeros(self.config["n_barrier_steps"] + 1, self.batch_size).to(self.device)

        loss = self.loss_fn.invariant_loss(
            barrier_values,
            next_n_barrier_values,
            episode_mask
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0).to(self.device), 1e-3, 1e-5))

        # Test case 2: Invariance violation
        next_n_barrier_values[1] = barrier_values # No decrease
        loss = self.loss_fn.invariant_loss(
            barrier_values,
            next_n_barrier_values,
            episode_mask
        )
        self.assertTrue(loss.item() > 0)

        # Test case 3: Mixed barrier values
        next_n_barrier_values = torch.randn(self.config["n_barrier_steps"] + 1, self.batch_size).to(self.device)
        episode_mask = torch.randint(0, 2, (self.config["n_barrier_steps"] + 1, self.batch_size)).to(self.device)
        loss = self.loss_fn.invariant_loss(
            barrier_values,
            next_n_barrier_values,
            episode_mask
        )
        self.assertTrue(loss.item() >= 0)

        # Test case 3: Episode boundaries
        episode_mask = torch.ones((self.config["n_barrier_steps"] + 1, self.batch_size)).to(self.device)
        loss = self.loss_fn.invariant_loss(
            barrier_values,
            next_n_barrier_values,
            episode_mask
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0).to(self.device)))



    def test_complete_loss(self):
        return
        """Test complete loss computation."""
        total_loss, loss_dict = self.loss_fn(
            barrier_values=self.barrier_values,
            next_barrier_values=self.next_barrier_values,
            feasible_mask=self.feasible_mask,
            infeasible_mask=self.infeasible_mask,
            episode_mask=self.episode_mask
        )

        # Check loss components
        self.assertTrue(isinstance(total_loss, torch.Tensor))
        self.assertTrue(total_loss.item() >= 0)

        # Check loss dictionary
        expected_keys = {
            "feasible_loss",
            "infeasible_loss",
            "invariant_loss",
            "barrier_loss"
        }
        self.assertEqual(set(loss_dict.keys()), expected_keys)

        # Check individual loss values
        for value in loss_dict.values():
            self.assertTrue(isinstance(value, float))
            self.assertTrue(value >= 0)

    def test_gradient_flow(self):
        return
        """Test gradient flow through the loss function."""

        # Create dummy network and optimizer
        class DummyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1)

            def forward(self, x):
                x = F.relu(self.layer(x))
                return x

        net = DummyNet().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        # Forward pass
        input_tensor = torch.randn(self.batch_size, 1).to(self.device)
        barrier_values = net(input_tensor).squeeze()
        next_barrier_values = net(torch.randn(self.config["n_barrier_steps"] + 1, self.batch_size, 1).to(self.device)).squeeze()

        # Compute loss
        loss, _ = self.loss_fn(
            barrier_values=barrier_values,
            next_barrier_values=next_barrier_values,
            feasible_mask=self.feasible_mask,
            infeasible_mask=self.infeasible_mask,
            episode_mask=self.episode_mask
        )

        # Check gradient flow
        optimizer.zero_grad()
        loss.backward()

        for param in net.parameters():
            self.assertTrue(param.grad is not None)
            self.assertFalse(torch.allclose(param.grad, torch.tensor(0.0).to(self.device)))

    def test_numerical_stability(self):
        return 
        """Test numerical stability with extreme values."""
        # Test with very large values
        large_values = 1e6 * torch.ones(self.config["n_barrier_steps"] + 1, self.batch_size).to(self.device)
        total_loss, _ = self.loss_fn(
            barrier_values=large_values[0],
            next_barrier_values=large_values,
            feasible_mask=self.feasible_mask,
            infeasible_mask=self.infeasible_mask,
            episode_mask=self.episode_mask
        )
        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))

        # Test with very small values
        small_values = 1e-6 * torch.ones(self.config["n_barrier_steps"] + 1, self.batch_size).to(self.device)
        total_loss, _ = self.loss_fn(
            barrier_values=small_values[0],
            next_barrier_values=small_values,
            feasible_mask=self.feasible_mask,
            infeasible_mask=self.infeasible_mask,
            episode_mask=self.episode_mask
        )
        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))


if __name__ == '__main__':
    unittest.main()