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
    def __init__(self, u, v, capacity=20, utilization=0.0):
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

ROUTING_PATHS = {
    (0, 3): [
        [0, 1, 2, 3],
        [0, 8, 7, 6, 3]
    ],
    (0, 4): [
        [0, 1, 5, 4],
        [0, 8, 7, 6, 3, 4]
    ],
    (7, 3): [
        [7, 1, 2, 3],
        [7, 6, 3]
    ],
    (7, 4): [
        [7, 1, 5, 4],
        [7, 6, 3, 4]
    ]
}

POSSIBLE_ACTIONS = []
for src_dst_tuple, routing_paths_list in ROUTING_PATHS.items():
    for path in routing_paths_list:
        POSSIBLE_ACTIONS.append({"src_dst": src_dst_tuple, "path": path})