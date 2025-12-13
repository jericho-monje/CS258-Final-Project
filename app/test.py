import model.nwutil as nwutil

import csv
import networkx


def csv_lineReader(file_path:str):
    with open(file_path, 'r') as file:
        csv_reader:csv.DictReader = csv.DictReader(file)
        for line in csv_reader:
            yield line

def _generate_req(t0):
    try:
        return next(t0)
    except StopIteration:
        return None
    
def test_csvLineReader():
    req_loader = csv_lineReader(r"C:\Users\phong\Desktop\work\CS 258\CS258-Final-Project\CS258-Final-Project\app\model\data\train\requests-0.csv")
    t1 = _generate_req(req_loader)
    while t1:
        print(str(t1))
        t1 = _generate_req(req_loader)

def test_nxGraphs():
    t0 = nwutil.generate_sample_graph()
    # print(t0.edges[1,2])
    idx:int = 0
    for ia in t0.edges:
        print(f"{idx}::{ia}")
        idx += 1
    print(list(ia for ia in t0.adjacency()))
    print(len(t0.edges))

def test_generate_paths_by_edge():
    result = nwutil.generate_routing_paths_by_edge(nwutil.SAMPLE_GRAPH_ROUTING_PATHS_BY_NODE, nwutil.generate_sample_graph())
    print(result)

def main() -> None:
    # test_csvLineReader()
    # test_nxGraphs()
    # test_generate_paths_by_edge()
    pass

if __name__ == "__main__":
    main()