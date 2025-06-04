import time
from typing import List

import numpy as np

from congestion.optimization.optimal_flow.flow_graph import FlowGraph
from congestion.utils.config import PushRelabelConfiguration

# ToDo: For large graphs, edge lists should be more efficient than matrices.


class Preflow:
    def __init__(self, graph: FlowGraph, verbose: bool = False):
        self.verbose = verbose
        self.graph = graph
        self.original_capacities = graph.capacities
        self.height = [0 for _ in range(self.num_nodes)]
        self.height[self.source] = self.num_nodes

        # Init load: Capacity of source edges to max. All further loads to 0.
        self.load = np.zeros((self.num_nodes, self.num_nodes))
        self.load[self.source, :] = self.original_capacities[self.source, :]

        # Initial excess is load on source edges.
        self.excess = self.load[self.source, :].flatten().copy()

        # Active nodes are nodes with positive excess.
        self.active_nodes = np.argwhere(self.excess > 0).flatten().tolist()

        # A valid push is a push to a lower height. Initially there are none.
        self.valid_pushes = np.zeros(
            (self.num_nodes, self.num_nodes)).astype(int)

        # Residual capacities: remove initial flow from original capacities.
        self.residual_capacities = self.original_capacities - self.load

        # Dictionaries allow us to access edges in constant time.
        self.edge_val_to_key = {x: [] for x in self.graph.nodes}
        self.edge_key_to_val = {x: [] for x in self.graph.nodes}
        for i in range(self.num_nodes):
            for j in range(i + 1):
                if self.original_capacities[i][j] > 0:
                    self.edge_val_to_key[j].append(i)
                    self.edge_key_to_val[i].append(j)
                if self.original_capacities[j][i] > 0:
                    self.edge_key_to_val[j].append(i)
                    self.edge_val_to_key[i].append(j)

        # Add backward flow possibilities.
        self.residual_capacities += self.load.T
        for x in list(*np.nonzero(self.load[self.source, :])):
            self.edge_val_to_key[self.source].append(x)
            self.edge_key_to_val[x].append(self.source)

    @property
    def source(self) -> int:
        return self.graph.source

    @property
    def sink(self) -> int:
        return self.graph.sink

    @property
    def net_load(self) -> np.ndarray:
        return self.load - self.load.T

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes

    @property
    def total_flow(self) -> float:
        """ The total flow is measured as the excess at the sink. """
        return self.excess[self.sink]

    def push_potential(self, node: int, target: int) -> float:
        maximal_capacity = min(
            self.residual_capacities[node, target],
            self.excess[node].item())

        return maximal_capacity

    def push(self, node: int, target: int, flow_size: float) -> None:
        # Determine how much flow can be pushed.
        if flow_size == 0:
            raise ValueError("Trying to push an empty flow.")

        if self.verbose:
            print(f"Pushing {flow_size} from {node} to {target}.")

        # Push flow.
        self.excess[node] -= flow_size
        self.excess[target] += flow_size

        self.load[node, target] += flow_size

        self.residual_capacities[node, target] -= flow_size
        self.residual_capacities[target, node] += flow_size

        # After pushing a flow, the backwards flow becomes available.
        if node not in self.edge_key_to_val[target]:
            self.edge_key_to_val[target].append(node)
            self.edge_val_to_key[node].append(target)

        # Adapt active nodes.
        if self.excess[node] == 0:
            self.active_nodes.remove(node)

        if target not in [self.source, self.sink]:
            if target not in self.active_nodes:
                self.active_nodes.append(target)

    def relabel(self, node: int):
        """ Increase the height of a node. """
        self.height[node] += 1
        self.update_valid_pushes(node)

    def update_valid_pushes(self, node: int):
        """ Update valid pushes according to capacity and height.
            Only push downstream!"""

        # Allow all pushes that now became downstream.
        for neighbour in self.edge_key_to_val[node]:
            if self.height[neighbour] < self.height[node]:
                if self.residual_capacities[node, neighbour] > 0:
                    self.valid_pushes[node, neighbour] = 1

        # Forbid all pushes that are now upstream.
        for neighbour in self.edge_val_to_key[node]:
            if self.height[neighbour] <= self.height[node]:
                self.valid_pushes[neighbour, node] = 0


class PushRelabel:
    def __init__(self, config: PushRelabelConfiguration):
        self.config = config
        self.start_time = None

    @staticmethod
    def choose_active_node(active_nodes: List[int], heights: List[int]) -> int:
        """ Choose the highest of several active nodes.
            Tie-breaker: lowest index. """

        if len(active_nodes) > 0:
            heights = [heights[x] for x in active_nodes]
            return active_nodes[np.argmax(heights).item()]
        else:
            raise RuntimeError('No active nodes.')

    @staticmethod
    def choose_push_target(targets: List[int]):
        """ Simply pick the first of possible targets. """
        return targets[0]

    @property
    def current_runtime(self) -> float:
        if self.start_time is None:
            raise RuntimeError('Current runtime only available after calling'
                               'calculate_maximal_flow().')
        return time.time() - self.start_time

    def iterate_flow(self, flow: Preflow) -> Preflow:
        node = self.choose_active_node(flow.active_nodes, flow.height)
        targets = np.argwhere(flow.valid_pushes[node] == 1).flatten().tolist()

        # Only allow targets where actually flow can be pushed.
        targets = [t for t in targets if flow.push_potential(node, t) > 0]

        if len(targets) > 0:
            target = self.choose_push_target(targets)
            flow_size = flow.push_potential(node, target)
            flow.push(node, target, flow_size)
        else:
            flow.relabel(node)
            if flow.height[node] > flow.num_nodes * 2:
                msg = f"Height of node {node} exceeded limit."
                raise ValueError(msg)

        return flow

    def calculate_maximal_flow(self, graph: FlowGraph) -> Preflow:
        self.start_time = time.time()

        flow = Preflow(graph, verbose=self.config.verbose)

        while True:
            if len(flow.active_nodes) == 0:
                return flow

            if self.current_runtime > self.config.max_runtime:
                raise RuntimeError("Maximum flow could not be calculated in"
                                   "time.")

            flow = self.iterate_flow(flow)
