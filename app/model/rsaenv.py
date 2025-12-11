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

def get_next_request_from_csv(lineReader) -> nwutil.Request:
    try:
        newLine:dict = next(lineReader)
        newReq:nwutil.Request = nwutil.Request(
            source=int(newLine["source"]),
            destination=int(newLine["destination"]),
            holding_time=int(newLine["holding_time"])
        )
        return newReq
    except StopIteration:
        return None

class RSAEnv(gym.Env):
    class RSAEnvInfo:
        def __init__(self) -> None:
            self.blocks:int = 0
            self.requests:int = 0

    def __init__(self, req_file:str, max_ht:int=100, link_capacity:int=10, debug:bool=False):
        super().__init__()
        self._debug:bool = debug
        self.topology:nx.Graph = nwutil.generate_sample_graph()
        self.routing_paths:list[dict] = nwutil.generate_routing_paths_by_edge(nwutil.SAMPLE_GRAPH_ROUTING_PATHS_BY_NODE, self.topology)
        self.num_links:int = len(self.topology.edges)
        self.num_nodes:int = len(self.topology.nodes)
        self.link_capacity:int = link_capacity
        self.max_ht:int = max_ht
        self.round:int = 0

        self._infostate:RSAEnv.RSAEnvInfo = RSAEnv.RSAEnvInfo()

        obs_tmp_dict:dict = {}
        for ia in range(self.num_links):
            obs_tmp_dict.update({
                f"LINK-{ia}" : spaces.MultiBinary(self.link_capacity)
            })
        obs_tmp_dict.update({
            "source": spaces.Discrete(self.num_nodes),
            "destination": spaces.Discrete(self.num_nodes),
            "holding_time": spaces.Discrete(self.max_ht+1)
        })

        self.observation_space:spaces.Box = spaces.Dict(obs_tmp_dict)

        self.__linkstates:list[np.ndarray] = RSAEnv.make_blank_linkstates(self.topology)

        # Set action space to length of possible actions
        self.action_space = spaces.Discrete(len(self.routing_paths))

        self.__req_file:str = req_file
        self.__req_loader = csv_lineReader(self.__req_file)
        self._req:nwutil.Request = None

    def make_blank_linkstates(tgtGraph:nx.Graph) -> list[np.ndarray]:
        result = []
        for edge in tgtGraph.edges:
            result.append(np.array([State.AVAILABLE] * tgtGraph.edges[edge[0], edge[1]]["state"].capacity, dtype=np.int8))
        return result

    def _get_obs(self):
        result:dict = {}
        for linkpath in range(self.num_links):
            result.update({
                f"LINK-{linkpath}" : (self.__linkstates[linkpath] != 0).astype(np.int8)
            })

        result.update({
            "source": self._req.source,
            "destination": self._req.destination,
            "holding_time": self._req.holding_time
        })

        return result

    def _get_info(self) -> dict:
        return vars(self._infostate)
    
    def _find_available_color(self, action:int) -> int:
        for color in range(self.link_capacity):
            available_for_all = True
            for ia in self.routing_paths[action]["path"]:
                if self.__linkstates[ia][color] != State.AVAILABLE:
                    available_for_all = False
                    break
            if available_for_all:
                return color
        return -1


    def reset(self, seed=None) -> tuple[dict,dict]:
        super().reset(seed=seed)
        self.round = 0
        self.__linkstates = RSAEnv.make_blank_linkstates(self.topology)
        self.__req_loader = csv_lineReader(self.__req_file)
        self._req = self._next_request()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _next_request(self) -> nwutil.Request:
        self._infostate.requests += 1
        return get_next_request_from_csv(self.__req_loader)
    
    def _clock_forward(self) -> None:
        self.round += 1
        for linkpath in range(len(self.__linkstates)):
            self.__linkstates[linkpath] = np.where(self.__linkstates[linkpath] > 0, self.__linkstates[linkpath] - 1, 0)
    
    def step(self, action:int):
        self._clock_forward()
        truncated:bool = (self.round == self.max_ht - 1)
        assert action in range(self.action_space.n), "Invalid action"

        if (self.routing_paths[action]["source"] != self._req.source) or \
            (self.routing_paths[action]["destination"] != self._req.destination):
            self._infostate.blocks += 1
            reward = -1
        else:
            curr_req_ht = self._req.holding_time
            available_color = self._find_available_color(action)
            if available_color == -1:
                self._infostate.blocks += 1
                reward = -1
            else:
                reward = 1
                for ia in self.routing_paths[action]["path"]:
                    self.__linkstates[ia][available_color] = curr_req_ht

        self._req = self._next_request()
        observation = self._get_obs()
        info = self._get_info()

        if self._debug >= 2:
            print(f"{self.round}, obs: {observation}")

        return observation, reward, False, truncated, info