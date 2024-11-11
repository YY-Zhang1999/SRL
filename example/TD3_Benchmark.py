import argparse
import numpy as np
import torch
import gym
import safety_gym
import safety_gymnasium
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import pandas as pd
import seaborn as sns
from datetime import datetime
import os

from SRL.core.agents.safe_TD3 import Safe_TD3
from SRL.core.utils.logger import Logger


class BenchmarkRunner:
    """
    Benchmark runner for comparing Safe TD3 against baseline TD3
    in both standard Gym and Safety Gym environments.
    """

    def __init__(
            self,
            save_dir: str = "benchmark_results",
            seed: int = 0,
    ):
        """
        Initialize benchmark runner.

        Args:
            save_dir: Directory to save results
            seed: Random seed
        """
        self.save_dir = save_dir
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        self.log_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize logger
        self.logger = Logger(
            log_dir=self.log_dir,
            experiment_name="td3_benchmarks"
        )

    def setup_gym_env(self, env_id: str) -> gym.Env:
        """
        Setup standard gym environment.

        Args:
            env_id: Environment ID

        Returns:
            Wrapped environment
        """
        env = gym.make(env_id)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env

    def setup_safety_gym_env(self, env_id: str) -> gym.Env:
        """
        Setup Safety Gym environment.

        Args:
            env_id: Environment ID

        Returns:
            Wrapped environment
        """
        env = safety_gymnasium.make(env_id)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env

    def setup_agent(
            self,
            env: gym.Env,
            is_safe: bool = True
    ) -> Union[Safe_TD3, TD3]:
        """
        Setup agent (either Safe TD3 or baseline TD3).

        Args:
            env: Environment
            is_safe: Whether to use Safe TD3

        Returns:
            Agent
        """
        action_dim = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(action_dim),
            sigma=0.1 * np.ones(action_dim)
        )

        if is_safe:
            return Safe_TD3(
                policy="SafeTD3Policy",
                env=env,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                barrier_lambda=0.1,
                n_barrier_steps=1,
                gamma_barrier=0.99,
                safety_margin=0.1,
                tensorboard_log=self.log_dir,
                device=self.device,
                seed=self.seed
            )
        else:
            return TD3(
                policy="MlpPolicy",
                env=env,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                tensorboard_log=self.log_dir,
                device=self.device,
                seed=self.seed
            )

    def evaluate(
            self,
            agent: Union[Safe_TD3, TD3],
            env: gym.Env,
            n_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            agent: Agent to evaluate
            env: Environment
            n_eval_episodes: Number of evaluation episodes

        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        costs = []
        lengths = []

        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_cost = 0
            episode_length = 0

            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                if "cost" in info:
                    episode_cost += info["cost"]
                episode_length += 1

            rewards.append(episode_reward)
            costs.append(episode_cost)
            lengths.append(episode_length)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_cost": np.mean(costs),
            "std_cost": np.std(costs),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths)
        }

    def train_and_evaluate(
            self,
            env_id: str,
            total_timesteps: int,
            eval_freq: int = 10000,
            is_safety_gym: bool = False
    ) -> Dict[str, List[float]]:
        """
        Train and evaluate both Safe TD3 and baseline TD3.

        Args:
            env_id: Environment ID
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            is_safety_gym: Whether environment is from Safety Gym

        Returns:
            Dictionary of training histories
        """
        # Setup environment
        env = self.setup_safety_gym_env(env_id) if is_safety_gym else self.setup_gym_env(env_id)

        # Training histories
        history = {
            "safe_rewards": [],
            "safe_costs": [],
            "baseline_rewards": [],
            "baseline_costs": [],
            "timesteps": []
        }

        # Train and evaluate Safe TD3
        safe_agent = self.setup_agent(env, is_safe=True)
        for timestep in range(0, total_timesteps, eval_freq):
            safe_agent.learn(eval_freq, progress_bar=True)
            metrics = self.evaluate(safe_agent, env)

            history["safe_rewards"].append(metrics["mean_reward"])
            history["safe_costs"].append(metrics["mean_cost"])
            history["timesteps"].append(timestep)

            self.logger.log_eval({
                "safe_td3/reward": metrics["mean_reward"],
                "safe_td3/cost": metrics["mean_cost"],
                "timestep": timestep
            })

        # Reset environment
        env = self.setup_safety_gym_env(env_id) if is_safety_gym else self.setup_gym_env(env_id)

        # Train and evaluate baseline TD3
        baseline_agent = self.setup_agent(env, is_safe=False)
        for timestep in range(0, total_timesteps, eval_freq):
            baseline_agent.learn(eval_freq, progress_bar=True)
            metrics = self.evaluate(baseline_agent, env)

            history["baseline_rewards"].append(metrics["mean_reward"])
            history["baseline_costs"].append(metrics["mean_cost"])

            self.logger.log_eval({
                "baseline_td3/reward": metrics["mean_reward"],
                "baseline_td3/cost": metrics["mean_cost"],
                "timestep": timestep
            })

        return history

    def plot_results(
            self,
            history: Dict[str, List[float]],
            env_id: str
    ) -> None:
        """
        Plot training results.

        Args:
            history: Training history
            env_id: Environment ID
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot rewards
        ax1.plot(history["timesteps"], history["safe_rewards"], label="Safe TD3")
        ax1.plot(history["timesteps"], history["baseline_rewards"], label="TD3")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Average Return")
        ax1.set_title(f"Training Rewards on {env_id}")
        ax1.legend()

        # Plot costs
        ax2.plot(history["timesteps"], history["safe_costs"], label="Safe TD3")
        ax2.plot(history["timesteps"], history["baseline_costs"], label="TD3")
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Average Cost")
        ax2.set_title(f"Safety Violations on {env_id}")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{env_id}_results.png"))
        plt.close()

    def run_benchmarks(
            self,
            gym_envs: List[str],
            safety_gym_envs: List[str],
            total_timesteps: int = 1_000_000,
            eval_freq: int = 10000
    ) -> None:
        """
        Run benchmarks on all environments.

        Args:
            gym_envs: List of standard gym environments
            safety_gym_envs: List of Safety Gym environments
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
        """
        results = {}

        # Standard Gym environments
        for env_id in gym_envs:
            print(f"\nEvaluating on {env_id}")
            history = self.train_and_evaluate(
                env_id=env_id,
                total_timesteps=total_timesteps,
                eval_freq=eval_freq,
                is_safety_gym=False
            )
            results[env_id] = history
            self.plot_results(history, env_id)

        # Safety Gym environments
        for env_id in safety_gym_envs:
            print(f"\nEvaluating on {env_id}")
            history = self.train_and_evaluate(
                env_id=env_id,
                total_timesteps=total_timesteps,
                eval_freq=eval_freq,
                is_safety_gym=True
            )
            results[env_id] = history
            self.plot_results(history, env_id)

        # Save results
        torch.save(results, os.path.join(self.log_dir, "benchmark_results.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval_freq", type=int, default=10000)
    args = parser.parse_args()

    # Define environments
    gym_envs = [
        "Pendulum-v1",
        "HalfCheetah-v4",
        "Hopper-v4"
    ]
    gym_envs = []

    safety_gym_envs = [
        "Safexp-PointGoal1-v0",
        "Safexp-CarGoal1-v0",
        "Safexp-DoggoGoal1-v0"
    ]

    # Run benchmarks
    benchmark_runner = BenchmarkRunner(seed=args.seed)
    benchmark_runner.run_benchmarks(
        gym_envs=gym_envs,
        safety_gym_envs=safety_gym_envs,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq
    )


if __name__ == "__main__":
    main()