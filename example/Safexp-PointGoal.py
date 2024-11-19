from datetime import datetime
import sys
import time
import gymnasium as gym
sys.modules["gym"] = gym

import numpy as np

from SRL.core.agents.safe_TD3 import Safe_TD3
from SRL.core.utils.safety_gym_env import make_safety_env
from SRL.core.utils.callbacks import SafetyMetricsCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3 import TD3

# Initialize the training environment
env_id = 'SafetyCarCircle1-v0'
train = True
render_mode = "human" if not train else None

env = make_safety_env(env_id, render_mode=render_mode)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Hyperparameters
configs = {
    # TD3 parameters
    "policy": "SafeTD3Policy",
    "env": env,
    "learning_rate": 1e-4,
    "buffer_size": 5_000_00,
    "learning_starts": 10000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1, #(1, "episode"),
    "gradient_steps": 1,
    "action_noise": None,
    "replay_buffer_kwargs": None,
    "optimize_memory_usage": False,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "tensorboard_log": None,
    "policy_kwargs": None,
    "verbose": 1,
    "seed": None,

    # Barrier certificate parameters
    "barrier_lambda": 0.1,
    "n_barrier_steps": 19,
    "gamma_barrier": 0.99,
    "safety_margin": 0.1,
    "lambda_lr": 1e-3,
}




if train:
    model = Safe_TD3(**configs)
    time_label = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    callback = SafetyMetricsCallback(env, log_path=f"./{env_id}/logs_{time_label}", best_model_save_path=f"./{env_id}/best_model_{time_label}")
    model.learn(total_timesteps=2_000_00, callback=callback, log_interval=10, progress_bar=True)
else:
    model = Safe_TD3.load("./SafetyCarCircle1-v0/best_model_2024-11-16_01_49_50/best_model.zip", env)

env = model.get_env()
obs = env.reset()
costs = 0
while True:
    action, _states = model.predict(obs, safe_mode=True)

    obs, rewards, dones, info = env.step(action)
    costs += info[0]["cost"]

    if dones:
        print(costs)
        break
    env.render()