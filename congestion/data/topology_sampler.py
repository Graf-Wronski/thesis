from dataclasses import dataclass
from typing import List, Tuple

from congestion.utils.random_tree import RandomTree


@dataclass
class Topology:
    bus_ids: List[str]
    edges: List[Tuple[str, str]]
    load_ids: List[str]

class TopologySampler(object):
    def __init__(self):
        self.topologies = []

    def sample(self, n_loads: int = 10, n_feeders: int = 4):
        """ At the moment only a single topology is implemented. """

        tree = RandomTree(n_grid_loads=n_loads, n_feeders=n_feeders)
        bus_list, edge_list, leaf_list = tree.nodes, tree.edges, tree.leaves

        bus_ids = [f"Bus {idx}" for idx in bus_list]
        edge_ids = [(f"Bus {idx1}", f"Bus {idx2}") for idx1, idx2 in edge_list]
        load_ids = [f"Bus {idx}" for idx in leaf_list]

        return Topology(bus_ids, edge_ids, load_ids)