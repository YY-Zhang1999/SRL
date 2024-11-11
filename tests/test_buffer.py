import unittest
import numpy as np
import torch as th
import gym
from typing import Dict, Tuple
from stable_baselines3.common.vec_env import DummyVecEnv

from SRL.core.utils.buffers import SafeReplayBuffer
from SRL.core.models.TD3_barrier_policy import SafeTD3Policy


class TestSafeBuffers(unittest.TestCase):
    """Test suite for safe replay and rollout buffers."""

    def setUp(self):
        """Setup test environment and buffers."""
        # Create environment
        env = gym.make("Pendulum-v1")
        self.env = DummyVecEnv([lambda: env])

        # Buffer parameters
        self.buffer_size = 1000
        self.n_envs = 3
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # Initialize buffers
        self.replay_buffer = SafeReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=self.device,
            n_envs=self.n_envs
        )

        # Sample data dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

    def generate_sample_data(self) -> Dict[str, np.ndarray]:
        """Generate sample transition data."""
        return {
            "obs": np.random.randn(self.n_envs, self.obs_dim).astype(np.float32),
            "next_obs": np.random.randn(self.n_envs, self.obs_dim).astype(np.float32),
            "action": np.random.randn(self.n_envs, self.action_dim).astype(np.float32),
            "reward": np.random.randn(self.n_envs).astype(np.float32),
            "done": np.zeros(self.n_envs).astype(np.float32),
            "feasible": np.random.randint(0, 2, self.n_envs).astype(np.float32),
            "infeasible": np.random.randint(0, 2, self.n_envs).astype(np.float32),
            'infos': [{} for _ in range(self.n_envs)]
        }

    def test_replay_buffer_add(self):
        """Test adding transitions to replay buffer."""
        # Generate and add sample transitions
        for _ in range(100):
            data = self.generate_sample_data()
            print(data["infeasible"])
            self.replay_buffer.add(
                obs=data["obs"],
                next_obs=data["next_obs"],
                action=data["action"],
                reward=data["reward"],
                done=data["done"],
                feasible=data["feasible"],
                infeasible=data["infeasible"],
                infos=data["infos"],
            )

        # Check buffer state
        self.assertEqual(self.replay_buffer.pos, 100)
        self.assertFalse(self.replay_buffer.full)

    def test_replay_buffer_sample(self):
        """Test sampling from replay buffer."""
        # Fill buffer with sample data
        for _ in range(200):
            data = self.generate_sample_data()
            self.replay_buffer.add(
                obs=data["obs"],
                next_obs=data["next_obs"],
                action=data["action"],
                reward=data["reward"],
                done=data["done"],
                feasible=data["feasible"],
                infeasible=data["infeasible"],
                infos=data["infos"],
            )


        # Sample from buffer
        batch_size = 32
        samples = self.replay_buffer.sample(batch_size)

        # Check sample properties
        self.assertEqual(samples.observations.shape, (batch_size, self.obs_dim))
        self.assertEqual(samples.actions.shape, (batch_size, self.action_dim))
        self.assertEqual(samples.next_observations.shape, (batch_size, self.obs_dim))
        self.assertEqual(samples.rewards.shape, (batch_size, 1))
        self.assertEqual(samples.dones.shape, (batch_size, 1))
        self.assertEqual(samples.feasible_mask.shape, (batch_size, 1))
        self.assertEqual(samples.infeasible_mask.shape, (batch_size, 1))

        # Check data types
        self.assertTrue(th.is_tensor(samples.observations))
        self.assertTrue(th.is_tensor(samples.actions))
        self.assertTrue(th.is_tensor(samples.feasible_mask))

    def test_rollout_buffer_add(self):
        return
        """Test adding transitions to rollout buffer."""
        # Generate and add sample transitions
        for _ in range(100):
            data = self.generate_sample_data()
            self.rollout_buffer.add(
                obs=data["obs"],
                action=data["action"],
                reward=data["reward"],
                episode_start=data["episode_start"],
                value=np.random.randn(self.n_envs),
                log_prob=np.random.randn(self.n_envs),
                barrier_value=data["barrier_value"]
            )

        # Check buffer state
        self.assertEqual(self.rollout_buffer.pos, 100)
        self.assertFalse(self.rollout_buffer.full)

    def test_advantage_computation(self):
        return
        """Test advantage computation in rollout buffer."""
        # Fill buffer
        for _ in range(self.buffer_size):
            data = self.generate_sample_data()
            self.rollout_buffer.add(
                obs=data["obs"],
                action=data["action"],
                reward=data["reward"],
                episode_start=data["episode_start"],
                value=np.random.randn(self.n_envs),
                log_prob=np.random.randn(self.n_envs),
                barrier_value=data["barrier_value"]
            )

        # Compute advantages
        last_value = np.random.randn(self.n_envs)
        last_barrier_value = np.random.randn(self.n_envs)
        dones = np.zeros(self.n_envs)

        self.rollout_buffer.compute_returns_and_advantage(
            last_value,
            last_barrier_value,
            dones
        )

        # Check computed values
        self.assertTrue(np.all(np.isfinite(self.rollout_buffer.advantages)))
        self.assertTrue(np.all(np.isfinite(self.rollout_buffer.returns)))
        self.assertTrue(np.all(np.isfinite(self.rollout_buffer.safety_advantages)))





if __name__ == '__main__':
    unittest.main()