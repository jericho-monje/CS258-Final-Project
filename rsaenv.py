##  Begin Standard Imports
import numpy as np
import gymnasium as gym
from gymnasium import spaces

CONST_MODEL_SHAPE:int = 4

class RSAEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space:spaces.Box = spaces.Box(
            low=-1.0, high=1.0, shape=(CONST_MODEL_SHAPE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(CONST_MODEL_SHAPE, dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        self.state = np.random.uniform(-1, 1, size=CONST_MODEL_SHAPE).astype(np.float32)
        reward = 1.0
        terminated = False
        truncated = False
        return self.state, reward, truncated, terminated, {}