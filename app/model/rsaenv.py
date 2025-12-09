##  Begin Local Imports
import model.nwutil as nwutil
import model.resource as resource

##  Begin Standard Imports
import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum
import networkx as nx

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
    def __init__(self, req_file:str, max_ht:int=100, link_capacity:int=20, debug:bool=False):
        super().__init__()
        self._debug:bool = debug
        self.topology:nx.Graph = nwutil.generate_sample_graph()
        self.num_links:int = len(self.topology.edges)
        self.num_nodes:int = len(self.topology.nodes)
        self.link_capacity:int = link_capacity
        self.max_ht:int = max_ht
        self.round:int = 0

        obs_tmp_dict:dict = {}
        for ia in range(self.num_links):
            obs_tmp_dict.update({
                f"LINK-{ia}" : spaces.MultiBinary(self.link_capacity)
            })
        obs_tmp_dict.update({
            "source": spaces.Discrete(self.num_nodes),
            "destination": spaces.Discrete(self.num_nodes),
            "holding_time": spaces.Discrete(self.max_ht)
        })

        self.observation_space:spaces.Box = spaces.Dict(obs_tmp_dict)

        self._linkstates:list[np.ndarray] = RSAEnv.make_blank_linkstates(self.topology)

        # Set action space to length of possible actions
        self.action_space = spaces.Discrete(len(nwutil.POSSIBLE_ACTIONS))

        self.req_file:str = req_file
        self.req_loader = csv_lineReader(self.req_file)
        self._req:dict = None

    def _get_obs(self):
        result:dict = {}
        for linkpath in range(self.num_links):
            result.update({
                f"LINK-{linkpath}" : (self._linkstates[linkpath] != 0).astype(np.int8)
            })

        result.update({
            "source": int(self._req["source"]),
            "destination": int(self._req["destination"]),
            "holding_time": int(self._req["holding_time"])
        })

        return result

    def _get_info(self):
        return {}
    
    def _find_available_color(self, action:int):
        for color in range(self.link_capacity):
            available_for_all = True
            for ia in nwutil.POSSIBLE_ACTIONS[action]["path"]:
                if self._linkstate[ia][color] != State.AVAILABLE:
                    available_for_all = False
                    break
            if available_for_all:
                return color
        return -1


    def reset(self, seed=None):
        super().reset(seed=seed)
        self.round = 0
        self._linkstates = RSAEnv.make_blank_linkstates(self.topology)
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
        # assert int(self._req["holding_time"]) < self.max_ht
        assert action in range(self.action_space.n), "Invalid action"

        ###     Need to be able to apply operation to all links on chosen path.  Right now forcing chosen path number on linkstates array.  Incorrect.
        curr_req_ht = int(self._req["holding_time"])
        available_color = self._find_available_color(action)
        if available_color == -1:
            reward = -1
        else:
            reward = 1
            for ia in nwutil.POSSIBLE_ACTIONS[action]["path"]:
                self._linkstates[ia][available_color] = curr_req_ht

        self._req = get_next_csv_line(self.req_loader)
        observation = self._get_obs()
        info = self._get_info()

        if self._debug == 2:
            print(f"{self.round}, obs: {observation}")

        return observation, reward, False, truncated, info

    def make_blank_linkstates(tgtGraph:nx.Graph) -> list[np.ndarray]:
        result = []
        for edge in tgtGraph.edges:
            result.append(np.array([State.AVAILABLE] * tgtGraph.edges[edge[0], edge[1]]["state"].capacity, dtype=np.int8))
        return result