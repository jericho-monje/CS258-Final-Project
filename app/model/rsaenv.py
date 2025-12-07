##  Begin Standard Imports
import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum

CONST_MODEL_SHAPE:int = 4

class State(IntEnum):
    AVAILABLE = 0
    OCCUPIED = 1

def csv_lineReader(file_path:str):
    with open(file_path, 'r') as file:
        csv_reader:csv.DictReader = csv.DictReader(file)
        for line in csv_reader:
            yield line

def get_next_csv_line(lineReader) -> dict:
    try:
        return next(lineReader)
    except StopIteration:
        return None

class RSAEnv(gym.Env):
    def __init__(self, req_file:str, max_ht:int, num_links:int, link_capacity:int=20, debug:bool=False):
        super().__init__()
        self._debug:bool = debug
        self.num_links:int = num_links
        self.link_capacity:int = link_capacity
        self.max_ht:int = max_ht
        self.round:int = 0

        obs_tmp_dict:dict = {}
        for ia in range(num_links):
            obs_tmp_dict.update({
                f"LINK-{ia}" : spaces.MultiBinary(self.link_capacity)
            })
        obs_tmp_dict.update({
            # ####### THIS
            "REQUEST" : spaces.Discrete()
        })

        self.observation_space:spaces.Box = spaces.Dict(obs_tmp_dict)

        self._linkstates:list[np.ndarray] = RSAEnv.make_blank_linkstates(self.num_links, self.link_capacity)

        self.action_space = spaces.Discrete(2)

        self.req_file:str = self.req_file
        self.req_loader = csv_lineReader(self.req_file)
        self._req:dict = None

    def _get_obs(self):
        result:dict = {}
        for linkpath in range(self.num_links):
            result.update({
                f"LINK-{linkpath}" : (self._linkstates[linkpath] != 0).astype(np.int8)
            })

        result.update({
            "req" : self._req
        })

        return result

    def _get_info(self):
        return {}
        
    def _find_available_color(link_state, link_capacity:int):
        for color in range(link_capacity):
            if link_state[color] == State.AVAILABLE:
                return color
        return -1

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.round = 0
        self._linkstates = RSAEnv.make_blank_linkstates(self.num_links, self.link_capacity)
        self.req_loader = csv_lineReader(self.req_file)
        self._req = get_next_csv_line(self.req_loader)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _clock_forward(self):
        self.round += 1
        for linkpath in range(len(self._linkstates)):
            self._linkstates[linkpath] = np.where(self._linkstates[linkpath] > 0, self._linkstates[linkpath] - 1, 0)
    
    def step(self, action:int):
        self._clock_forward()
        truncated:bool = (self.round == self.max_ht - 1)
        assert action in range(self.num_links), "Invalid action"

        available_color = self._find_available_color(self._linkstates[action])
        if available_color == -1:
            reward = -1
        else:
            reward = 1
            curr_req_ht = self._req["holding_time"]
            self._linkstates[action][available_color] = curr_req_ht

        self._req = get_next_csv_line(self.req_loader)
        observation = self._get_obs()
        info = self._get_info()

        if self._debug:
            print(f"{self.round}, obs: {observation}")

        return observation, reward, False, truncated, info
    
    def make_blank_linkstates(num_links:int, link_capacity:int) -> list[np.ndarray]:
        return list((np.array([State.AVAILABLE] * link_capacity, dtype=np.int8)) for x in num_links)