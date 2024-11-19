from typing import Optional, Union

import gym
import numpy as np
import json
import os

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

@dataclass
class EvaluationMetrics:
    """Class to store evaluation metrics"""
    rewards: List[float]
    costs: List[float]
    lengths: List[int]
    successes: List[bool]
    timestamps: List[int]
    mean_reward: float
    std_reward: float
    mean_cost: float
    std_cost: float
    mean_length: float
    std_length: float
    success_rate: float


class SafetyMetricsCallback(EvalCallback):
    """
    Callback for evaluating and saving a policy with safety metrics.
    """

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            callback_after_eval: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
            best_model_save_criterion: str = "reward",  # 'reward' or 'cost' or 'combined'
            cost_weight: float = 0.5,  # Weight for cost in combined criterion
    ):
        """
        Initialize SafetyMetricsCallback.

        :param eval_env: Environment used for evaluation
        :param callback_on_new_best: Callback called when there is a new best model
        :param callback_after_eval: Callback called after every evaluation
        :param n_eval_episodes: Number of episodes to evaluate
        :param eval_freq: Evaluate the agent every n steps
        :param log_path: Path to save evaluation logs
        :param best_model_save_path: Path to save best model
        :param deterministic: Whether to use deterministic actions
        :param render: Whether to render the environment during evaluation
        :param verbose: Verbosity level
        :param warn: Whether to show warnings
        :param best_model_save_criterion: Criterion for saving best model ('reward', 'cost', or 'combined')
        :param cost_weight: Weight of cost in combined criterion
        """
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )

        # Initialize metrics
        self.best_mean_reward = -np.inf
        self.best_mean_cost = np.inf
        self.best_combined_score = -np.inf
        self.cost_results = []
        self.best_model_save_criterion = best_model_save_criterion
        self.cost_weight = cost_weight

        # Create dirs if needed
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _evaluate_running_policy(self) -> EvaluationMetrics:
        """
        Evaluate the current policy and return metrics.
        """
        sync_envs_normalization(self.training_env, self.eval_env)

        episode_rewards, episode_costs, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            render=self.render,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
        )

        metrics = EvaluationMetrics(
            rewards=episode_rewards,
            costs=episode_costs,
            lengths=episode_lengths,
            successes=self._is_success_buffer,
            timestamps=self.evaluations_timesteps,
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            mean_cost=float(np.mean(episode_costs)),
            std_cost=float(np.std(episode_costs)),
            mean_length=float(np.mean(episode_lengths)),
            std_length=float(np.std(episode_lengths)),
            success_rate=float(np.mean(self._is_success_buffer)) if self._is_success_buffer else 0.0
        )

        return metrics

    def _update_logs(self, metrics: EvaluationMetrics) -> None:
        """
        Update evaluation logs.
        """
        if self.log_path is not None:
            self.evaluations_results.append(metrics.rewards)
            self.cost_results.append(metrics.costs)
            self.evaluations_length.append(metrics.lengths)
            self.evaluations_timesteps.append(self.num_timesteps)

            if len(metrics.successes) > 0:
                self.evaluations_successes.append(metrics.successes)

            # Save evaluation metrics
            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                costs=self.cost_results,
                ep_lengths=self.evaluations_length,
                successes=self.evaluations_successes if len(metrics.successes) > 0 else None,
            )

    def _log_metrics(self, metrics: EvaluationMetrics) -> None:
        """
        Log evaluation metrics.
        """
        # Console output
        if self.verbose >= 1:
            print(f"\nEvaluation at timestep {self.num_timesteps}:")
            print(f"┌ Mean reward: {metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}")
            print(f"├ Mean cost: {metrics.mean_cost:.2f} ± {metrics.std_cost:.2f}")
            print(f"├ Mean episode length: {metrics.mean_length:.2f} ± {metrics.std_length:.2f}")
            if metrics.successes:
                print(f"└ Success rate: {100 * metrics.success_rate:.2f}%")

        # Tensorboard logging
        self.logger.record("eval/mean_reward", metrics.mean_reward)
        self.logger.record("eval/mean_cost", metrics.mean_cost)
        self.logger.record("eval/mean_ep_length", metrics.mean_length)
        if metrics.successes:
            self.logger.record("eval/success_rate", metrics.success_rate)

        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

    def _check_and_save_best_model(self, metrics: EvaluationMetrics) -> bool:
        """
        Check if current model is the best and save if it is.
        Returns True if model was saved.
        """
        is_best = False
        combined_score = metrics.mean_reward - self.cost_weight * metrics.mean_cost

        if self.best_model_save_criterion == "reward" and metrics.mean_reward > self.best_mean_reward:
            is_best = True
            self.best_mean_reward = metrics.mean_reward
        elif self.best_model_save_criterion == "cost" and metrics.mean_cost < self.best_mean_cost:
            is_best = True
            self.best_mean_cost = metrics.mean_cost
        elif self.best_model_save_criterion == "combined" and combined_score > self.best_combined_score:
            is_best = True
            self.best_combined_score = combined_score

        if is_best and self.best_model_save_path is not None:
            model_path = os.path.join(self.best_model_save_path, "best_model")
            self.model.save(model_path)
            if self.verbose >= 1:
                print(f"\nSaving new best model to {model_path}")

            # Save additional metrics for the best model
            metrics_path = os.path.join(self.best_model_save_path, "best_model_metrics.json")
            metrics_dict = {
                "timestep": self.num_timesteps,
                "mean_reward": metrics.mean_reward,
                "mean_cost": metrics.mean_cost,
                "success_rate": metrics.success_rate,
                "combined_score": combined_score,
                "criterion": self.best_model_save_criterion
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics_dict, f, indent=4)

        return is_best

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []

            # Evaluate current policy
            metrics = self._evaluate_running_policy()

            # Update logs
            self._update_logs(metrics)

            # Log metrics
            self._log_metrics(metrics)

            # Check and save best model
            is_best = self._check_and_save_best_model(metrics)

            # Trigger callbacks if needed
            if is_best and self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()
            else:
                continue_training = True

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            return continue_training

        return True

class SafetyMetricsCallback_(EvalCallback):
    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            callback_after_eval: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn
        )

        self.best_mean_cost = -np.inf
        self.last_mean_cost = -np.inf

        self.cost_results = []


    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_costs, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.cost_results.append(episode_costs)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    costs=self.cost_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_cost = np.mean(episode_costs)
            std_cost = np.std(episode_costs)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward
            self.last_mean_cost = mean_cost

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " 
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}, ",
                    f"episode_cost={mean_cost:.2f} +/- {std_cost:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_cost", float(mean_cost))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
            if mean_cost < self.best_mean_cost:
                self.best_mean_reward = mean_reward

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_costs = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_costs = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_costs += np.array([info.get("cost", 0) for info in infos])
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_costs.append(current_costs[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_costs[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_cost = np.mean(episode_costs)
    std_cost = np.std(episode_costs)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_costs, episode_lengths
    return mean_reward, std_reward
