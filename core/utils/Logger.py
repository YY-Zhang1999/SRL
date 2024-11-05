import os
import time
from typing import Dict, Any, Optional
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    A logger class that handles both console output and TensorBoard logging.
    Supports episode statistics tracking, step-wise metrics logging, and
    experiment organization.
    """

    def __init__(
            self,
            log_dir: str,
            experiment_name: Optional[str] = None,
            enable_tensorboard: bool = True,
            console_output: bool = True
    ):
        """
        Initialize the logger.

        Args:
            log_dir: Base directory for storing logs
            experiment_name: Name of the experiment. If None, timestamp will be used
            enable_tensorboard: Whether to enable TensorBoard logging
            console_output: Whether to enable console output
        """
        self.console_output = console_output
        self.enable_tensorboard = enable_tensorboard

        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = time.strftime("%Y%m%d_%H%M%S")

        # Setup log directory
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        if enable_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # Initialize episode statistics tracking
        self.episode_stats = defaultdict(list)
        self.global_step = 0
        self.current_episode = 0

        print(f"Logger initialized. Logging to: {self.log_dir}")

    def log_step(
            self,
            metrics: Dict[str, Any],
            step: Optional[int] = None,
            prefix: str = ""
    ) -> None:
        """
        Log metrics for a single training step.

        Args:
            metrics: Dictionary of metrics to log
            step: Global step number (if None, internal counter is used)
            prefix: Prefix to add to metric names
        """
        if step is not None:
            self.global_step = step

        # Log to TensorBoard
        if self.enable_tensorboard:
            for name, value in metrics.items():
                metric_name = f"{prefix}{name}" if prefix else name
                if isinstance(value, (int, float, np.number)):
                    self.writer.add_scalar(metric_name, value, self.global_step)

        # Console output
        if self.console_output:
            metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float)
                                      else f"{k}: {v}" for k, v in metrics.items()])
            print(f"Step {self.global_step} | {metrics_str}")

    def log_episode(self, episode_metrics: Dict[str, Any]) -> None:
        """
        Log metrics for a complete episode.

        Args:
            episode_metrics: Dictionary of episode metrics to log
        """
        self.current_episode += 1

        # Store episode statistics
        for name, value in episode_metrics.items():
            self.episode_stats[name].append(value)

        # Calculate running statistics
        running_stats = {
            f"running_{k}": np.mean(v[-100:])
            for k, v in self.episode_stats.items()
        }

        # Log both episode metrics and running stats
        metrics_to_log = {
            **episode_metrics,
            **running_stats
        }

        # Add episode prefix and log
        self.log_step(metrics_to_log, prefix="episode/")

    def log_eval(self, eval_metrics: Dict[str, Any]) -> None:
        """
        Log evaluation metrics.

        Args:
            eval_metrics: Dictionary of evaluation metrics to log
        """
        # Log with eval prefix
        self.log_step(eval_metrics, prefix="eval/")

    def log_info(self, info: str) -> None:
        """
        Log general information message.

        Args:
            info: Information message to log
        """
        if self.console_output:
            print(f"[INFO] {info}")

    def log_barrier_stats(
            self,
            barrier_loss: float,
            feasible_loss: float,
            infeasible_loss: float,
            invariant_loss: float
    ) -> None:
        """
        Log barrier certificate related statistics.

        Args:
            barrier_loss: Total barrier loss
            feasible_loss: Feasible region loss
            infeasible_loss: Infeasible region loss
            invariant_loss: Invariant property loss
        """
        barrier_metrics = {
            "barrier_loss": barrier_loss,
            "feasible_loss": feasible_loss,
            "infeasible_loss": infeasible_loss,
            "invariant_loss": invariant_loss
        }
        self.log_step(barrier_metrics, prefix="barrier/")

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration.

        Args:
            config: Dictionary containing experiment configuration
        """
        # Save to tensorboard
        if self.enable_tensorboard:
            config_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
            self.writer.add_text("config", config_str, 0)

        # Save to file
        config_path = os.path.join(self.log_dir, "config.txt")
        with open(config_path, "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

    def close(self) -> None:
        """Close the logger and cleanup."""
        if self.enable_tensorboard:
            self.writer.close()

if __name__ == '__main__':
    # Initialize logger
    logger = Logger(
        log_dir="logs",
        experiment_name="srlnbc_experiment"
    )

    # Log training step
    logger.log_step({
        "policy_loss": 0.5,
        "value_loss": 0.3,
        "reward": 10
    })

    # Log episode stats
    logger.log_episode({
        "episode_return": 100,
        "episode_length": 200,
        "constraint_violations": 0
    })

    # Log barrier certificate stats
    logger.log_barrier_stats(
        barrier_loss=0.4,
        feasible_loss=0.1,
        infeasible_loss=0.2,
        invariant_loss=0.1
    )

    # Save configuration
    logger.save_config({
        "learning_rate": 0.001,
        "batch_size": 64
    })

    # Cleanup
    logger.close()