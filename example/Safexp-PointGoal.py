import sys
import gymnasium as gym
sys.modules["gym"] = gym

import numpy as np

from SRL.core.agents.safe_TD3 import Safe_TD3
from SRL.core.envs.safety_gym_env import make_safety_env

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import TD3

env_id = 'SafetyPointGoal1-v0'
env = make_safety_env(env_id, render_mode="human")

train = True
# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = Safe_TD3("SafeTD3Policy", env, action_noise=action_noise, verbose=1)

if False:
    model.learn(total_timesteps=1_000_000, log_interval=10, progress_bar=True)
    model.save("td3_SafetyPointGoal1")

vec_env = model.get_env()

del model  # remove to demonstrate saving and loading

model = Safe_TD3.load("td3_SafetyPointGoal1")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(info)
    if dones:
        break
    vec_env.render()