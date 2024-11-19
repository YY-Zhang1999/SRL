import sys
import gymnasium as gym
sys.modules["gym"] = gym

import optuna
import numpy as np
from datetime import datetime
from typing import Dict, Any
import yaml
import json
from pathlib import Path

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from SRL.core.agents.safe_TD3 import Safe_TD3
from SRL.core.utils.safety_gym_env import make_safety_env
from SRL.core.utils.callbacks import SafetyMetricsCallback


class SafeTD3Optimizer:
    """Optimizer class for Safe TD3 hyperparameters using Optuna."""

    def __init__(
            self,
            env_id: str,
            n_trials: int = 100,
            n_timesteps: int = 2_000_00,
            study_name: str = None,
            storage: str = None,
            n_evaluations: int = 5
    ):
        """Initialize the optimizer.

        Args:
            env_id: Safety Gym environment ID
            n_trials: Number of optimization trials
            n_timesteps: Number of timesteps per trial
            study_name: Name of the study for storage
            storage: Optuna storage string
            n_evaluations: Number of evaluation runs per trial
        """
        self.env_id = env_id
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.study_name = study_name or f"safe_td3_{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        self.n_evaluations = n_evaluations

        # Create output directory
        self.output_dir = Path(f"optimization_results/{self.study_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def optimize(self) -> None:
        """Run the optimization process."""
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        study.optimize(
            func=self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        # Save results
        self.save_study_results(study)

    def sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for the trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled parameters
        """
        # Environment setup
        env = make_safety_env(self.env_id)
        n_actions = env.action_space.shape[-1]

        # Sample hyperparameters
        params = {
            "policy": "SafeTD3Policy",
            "env": env,
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": 20_0000, #trial.suggest_int("buffer_size", 100000, 1000000),
            "learning_starts": trial.suggest_int("learning_starts", 5000, 20000),
            "batch_size": 64 * trial.suggest_int("batch_size", 1, 8),
            #"tau": trial.suggest_float("tau", 0.001, 0.01),
            #"gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "train_freq": 1,
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
            "policy_delay": trial.suggest_int("policy_delay", 1, 4),
            "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.5),
            "target_noise_clip": trial.suggest_float("target_noise_clip", 0.3, 0.7),

            # Barrier certificate parameters
            "barrier_lambda": trial.suggest_float("barrier_lambda", 0.05, 0.2),
            "n_barrier_steps": trial.suggest_int("n_barrier_steps", 10, 20),
            #"gamma_barrier": trial.suggest_float("gamma_barrier", 0.95, 0.999),
            #"safety_margin": trial.suggest_float("safety_margin", 0.05, 0.2),
            "lambda_lr": trial.suggest_float("lambda_lr", 1e-4, 1e-2, log=True),
        }

        # Action noise
        noise_std = trial.suggest_float("action_noise_std", 0.05, 0.2)
        params["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions)
        )

        return params

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Mean reward across evaluations
        """
        # Sample parameters
        params = self.sample_params(trial)

        try:
            # Training
            model = Safe_TD3(**params)

            # Setup callback
            callback = SafetyMetricsCallback(
                params["env"],
                log_path=str(self.output_dir / f"trial_{trial.number}" / "logs"),
                best_model_save_path=str(self.output_dir / f"trial_{trial.number}" / "model"),
                best_model_save_criterion="combined",
                cost_weight=0.5
            )

            model.learn(
                total_timesteps=self.n_timesteps,
                callback=callback,
                progress_bar=True
            )

            return callback.best_combined_score

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")
            return float('-inf')

    def save_study_results(self, study: optuna.Study) -> None:
        """Save optimization results.

        Args:
            study: Completed Optuna study
        """
        # Save best parameters
        best_params = study.best_params
        best_value = study.best_value

        results = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": self.n_trials,
            "n_timesteps": self.n_timesteps,
            "env_id": self.env_id,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save as JSON
        with open(self.output_dir / "optimization_results.json", "w") as f:
            json.dump(results, f, indent=4)

        # Save study statistics
        df = study.trials_dataframe()
        df.to_csv(self.output_dir / "study_statistics.csv")

        # Create optimization visualization
        try:
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_html(str(self.output_dir / "optimization_history.html"))

            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_html(str(self.output_dir / "param_importances.html"))

            fig3 = optuna.visualization.plot_parallel_coordinate(study)
            fig3.write_html(str(self.output_dir / "parallel_coordinate.html"))
        except:
            print("Warning: Could not generate some visualization plots")


if __name__ == "__main__":
    # Example usage
    optimizer = SafeTD3Optimizer(
        env_id="SafetyCarCircle1-v0",
        n_trials=50,
        n_timesteps=100000,
        study_name="safe_td3_optimization",
        storage="sqlite:///safe_td3_study.db"
    )

    optimizer.optimize()