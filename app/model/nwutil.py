import model.resource as resource

import networkx as nx

class Request:
    def __init__(self, source: int, destination: int, holding_time: int):
        self.source = source
        self.destination = destination
        self.holding_time = holding_time

    def __repr__(self):
        return f"Request(src={self.source}, dst={self.destination}, hold={self.holding_time})"


class BaseLinkState:
    def __init__(self, u, v, capacity=20, utilization=0.0):
        if u > v: # sort by the node ID
            u, v = v, u
        self.endpoints = (u, v)
        self.capacity = capacity
        self.utilization = utilization

    def __repr__(self):
        return f"LinkState(capacity={self.capacity}, util={self.utilization})"

class LinkState(BaseLinkState):
    """ 
    Data structure to store the link state.
    You can extend this class to add more attributes if needed.
    Do not change the BaseLinkState class.
    """
    def __init__(self, u, v, capacity=int(resource.config_values.get_option("LINK_CAPACITY")), utilization=0.0):
        super().__init__(u, v, capacity, utilization) 


def generate_sample_graph():
    # Create the sample graph
    G = nx.Graph()

    G.add_nodes_from(range(9))

    # Define links: ring links + extra links
    links = [(n, (n + 1) % 9) for n in range(9)] + [(1, 7), (1, 5), (3, 6)]

    # Add edges with link state objects
    for u, v in links:
        G.add_edge(u, v, state=LinkState(u, v))
    return G

SAMPLE_GRAPH_ROUTING_PATHS_BY_NODE = [
    {
        "source":0,
        "destination":3,
        "path":[0,1,2,3]
    },
    {
        "source":0,
        "destination":3,
        "path":[0,8,7,6,3]
    },
    {
        "source":0,
        "destination":4,
        "path":[0,1,5,4]
    },
    {
        "source":0,
        "destination":4,
        "path":[0,8,7,6,3,4]
    },
    {
        "source":7,
        "destination":3,
        "path":[7,1,2,3]
    },
    {
        "source":7,
        "destination":3,
        "path":[7,6,3]
    },
    {
        "source":7,
        "destination":4,
        "path":[7,1,5,4]
    },
    {
        "source":7,
        "destination":4,
        "path":[7,6,3,4]
    },
]

def generate_routing_paths_by_edge(routing_paths_by_node:list[dict[str:int,str:int,str:list[int]]], target_graph:nx.Graph) -> list[dict[str:int,str:int,str:list[int]]]:
    paths_by_edge:list[dict[str:int,str:int,str:list[int]]] = []
    for node_path in routing_paths_by_node:
        edge_path:dict[str:int,str:int,str:list[int]] = {
            "source":node_path["source"],
            "destination":node_path["destination"],
        }
        new_path:list[int] = []
        for ia in range(0, len(node_path["path"])-1, 1):
            u = node_path["path"][ia]
            v = node_path["path"][ia+1]
            if u > v:
                u,v = v,u
            new_path.append(list(target_graph.edges).index((u, v)))
        
        edge_path.update({
            "path":new_path
        })
        paths_by_edge.append(edge_path)

    return paths_by_edge