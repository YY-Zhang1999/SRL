import unittest
import numpy as np
import torch as th
import gym
from typing import Dict, Tuple
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback, EvalCallback

from SRL.core.agents.safe_TD3 import Safe_TD3
from SRL.core.models.TD3_barrier_policy import SafeTD3Policy
from SRL.core.utils.buffers import SafeReplayBuffer

class TestTD3SafeAgent(unittest.TestCase):
    """Test suite for TD3 Safe Agent."""

    def setUp(self):
        """Setup test environment and agent."""
        # Create environment
        self.env = gym.make("Pendulum-v1")
        self.vec_env = DummyVecEnv([lambda: self.env])

        # Agent parameters
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        action_dim = self.env.action_space.shape[0]

        # Initialize action noise
        self.action_noise = NormalActionNoise(
            mean=np.zeros(action_dim),
            sigma=0.1 * np.ones(action_dim)
        )

        # Initialize agent
        self.agent = Safe_TD3(
            policy=SafeTD3Policy,
            env=self.vec_env,
            learning_rate=3e-4,
            buffer_size=1000,
            learning_starts=100,
            batch_size=16,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=self.action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            barrier_lambda=0.1,
            n_barrier_steps=5,
            gamma_barrier=0.99,
            safety_margin=0.1,
            verbose=0,
            device=self.device,
            policy_kwargs={
                "net_arch": dict(
                    pi=[256, 256],
                    qf=[256, 256],
                    br=[256, 256]
                )
            }
        )

    def test_agent_initialization(self):
        """Test agent initialization."""
        # Check policy components
        self.assertIsInstance(self.agent.policy, SafeTD3Policy)
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.critic)
        self.assertIsNotNone(self.agent.actor_target)
        self.assertIsNotNone(self.agent.critic_target)
        self.assertIsNotNone(self.agent.policy.barrier_net)

        # Check buffer
        self.assertIsInstance(self.agent.replay_buffer, SafeReplayBuffer)

        # Check parameters
        self.assertEqual(self.agent.learning_starts, 100)
        self.assertEqual(self.agent.batch_size, 16)
        self.assertEqual(self.agent.policy_delay, 2)

    def test_predict(self):
        """Test agent prediction."""
        obs, _ = self.env.reset()

        # Get prediction
        action, _states = self.agent.predict(obs, deterministic=True)

        # Check action properties
        self.assertEqual(action.shape, self.env.action_space.shape)
        self.assertTrue(np.all(action >= self.env.action_space.low))
        self.assertTrue(np.all(action <= self.env.action_space.high))

    def test_safe_action_selection(self):
        """Test safe action selection."""
        obs, _ = self.env.reset()
        obs_tensor = th.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Get safe action
        with th.no_grad():
            unsafe_action = self.agent.actor(obs_tensor)
            safe_action = self.agent._get_safe_action(obs_tensor, unsafe_action)

        # Check action properties
        self.assertEqual(safe_action.shape, unsafe_action.shape)
        self.assertTrue(th.all(safe_action >= -1))
        self.assertTrue(th.all(safe_action <= 1))

    def test_learn(self):
        """Test learning process."""
        # Train for a few steps
        total_timesteps = 100
        self.agent = self.agent.learn(
            total_timesteps=total_timesteps,
            log_interval=100
        )

        # Check training progress
        self.assertEqual(self.agent.num_timesteps, total_timesteps)

    def test_save_load(self):
        """Test model saving and loading."""
        # Train agent briefly
        self.agent.learn(total_timesteps=20)

        # Save agent
        path = "tmp_test_model"
        self.agent.save(path)

        # Create new agent
        loaded_agent = Safe_TD3.load(
            path,
            env=self.vec_env,
            device=self.device
        )

        # Compare predictions
        obs, _ = self.env.reset()
        original_action, _ = self.agent.predict(obs, deterministic=True)
        loaded_action, _ = loaded_agent.predict(obs, deterministic=True)

        np.testing.assert_array_almost_equal(original_action, loaded_action)

    def test_evaluation(self):
        """Test agent evaluation."""
        # Train agent briefly
        self.agent.learn(total_timesteps=200)

        # Evaluate agent
        mean_reward, std_reward = evaluate_policy(
            self.agent,
            self.vec_env,
            n_eval_episodes=5,
            deterministic=True
        )

        print(type(mean_reward), std_reward)

        self.assertTrue(isinstance(mean_reward, np.float32))
        self.assertTrue(isinstance(std_reward, np.float32))

    def test_barrier_constraint(self):
        """Test barrier certificate constraint."""
        obs, _ = self.env.reset()
        obs_tensor = th.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Get barrier value
        with th.no_grad():
            barrier_value = self.agent.policy.barrier_net(obs_tensor)

        # Check barrier properties
        self.assertEqual(barrier_value.shape, (1,))

    def test_batch_processing(self):
        """Test batch processing during training."""
        # Fill buffer
        obs, _ = self.env.reset()
        for _ in range(self.agent.batch_size + 10):
            action = self.env.action_space.sample()
            next_obs, reward, done, _, infos = self.env.step(action)

            # Add transition with safety data
            self.agent.replay_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                feasible=np.array([1.0]),
                infeasible=np.array([0.0]),
                infos=[infos]
            )

            obs = next_obs if not done else self.env.reset()

        # Get batch
        batch = self.agent.replay_buffer.sample(self.agent.batch_size)


        # Check batch properties
        self.assertEqual(batch.observations.shape[0], self.agent.batch_size)
        self.assertEqual(batch.next_n_observations.shape[0], self.agent.n_barrier_steps + 1)
        self.assertEqual(batch.n_dones.shape[0], self.agent.n_barrier_steps + 1)
        self.assertEqual(batch.actions.shape[0], self.agent.batch_size)
        self.assertEqual(batch.rewards.shape[0], self.agent.batch_size)
        self.assertEqual(batch.feasible_mask.shape[0], self.agent.batch_size)

    def tearDown(self):
        """Cleanup after tests."""
        self.env.close()


def evaluate_policy(
        agent: Safe_TD3,
        env: gym.Env,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate the agent for a number of episodes.

    Args:
        agent: Agent to evaluate
        env: Environment to evaluate in
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions

    Returns:
        mean_reward: Mean episode reward
        std_reward: Standard deviation of episode rewards
    """
    episode_rewards = []

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)

            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


if __name__ == '__main__':
    unittest.main()