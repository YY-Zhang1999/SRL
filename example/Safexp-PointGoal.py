import sys
import gymnasium as gym
sys.modules["gym"] = gym

import numpy as np

from SRL.core.agents.safe_TD3 import Safe_TD3
from SRL.core.utils.safety_gym_env import make_safety_env
from SRL.core.utils.callbacks import SafetyMetricsCallback

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import TD3

env_id = 'SafetyCarCircle1-v0'
train = False
render_mode = "human" if not train else None

env = make_safety_env(env_id, render_mode=render_mode)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = Safe_TD3("SafeTD3Policy", env, action_noise=action_noise, verbose=1)

if train:
    callback = SafetyMetricsCallback(env, log_path=f"./{env_id}/logs", best_model_save_path=f"./{env_id}/best_SafeTD3_model")
    model.learn(total_timesteps=2_000_00, callback=callback, log_interval=10, progress_bar=True)



vec_env = model.get_env()

del model  # remove to demonstrate saving and loading

model = Safe_TD3.load("./SafetyCarCircle1-v0/best_SafeTD3_model/best_model.zip")

obs = vec_env.reset()
costs = 0
while True:
    action, _states = model.predict(obs, safe_mode=True)

    obs, rewards, dones, info = vec_env.step(action)


    if dones:
        break
    vec_env.render()