##  Begin Local Imports
import model.rsaenv as rsaenv

##  Begin Standard Imports
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import DQN

##  Example environment preset
# env:gym.Env = gym.make("LunarLander-v3", render_mode="human")

##  Custom environment
env:gym.Env = rsaenv.RSAEnv()

model:DQN = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()