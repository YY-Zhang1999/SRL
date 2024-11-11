import sys
import gymnasium as gym
import numpy as np
sys.modules["gym"] = gym

from SRL.core.agents.safe_TD3 import Safe_TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import TD3

env = gym.make("Pendulum-v1", render_mode="rgb_array")
print(type(env))
train = False
# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


model = Safe_TD3("SafeTD3Policy", env, action_noise=action_noise, verbose=1)

if True:
    model.learn(total_timesteps=20000, callback=EvalCallback(env), log_interval=10, progress_bar=True)
    model.save("td3_pendulum")

vec_env = model.get_env()

del model  # remove to demonstrate saving and loading

model = Safe_TD3.load("td3_pendulum")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")


