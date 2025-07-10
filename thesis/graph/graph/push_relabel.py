import time
from typing import List

import numpy as np

from thesis.graph.graph.flow_graph import FlowNetwork
from thesis.graph.utils.config import PushRelabelConfiguration

# ToDo: For large graphs, edge lists should be more efficient than matrices.
class Preflow:
    def __init__(self, graph: FlowNetwork, verbose: bool = False):
        self.verbose = verbose
        self.graph = graph
        self.original_capacities = graph.capacities
        self.height = [0 for _ in range(self.num_vertices)]
        self.height[self.source] = self.num_vertices

        # Init load: Capacity of source edges to max. All further loads to 0.
        self.load = np.zeros((self.num_vertices, self.num_vertices))
        self.load[self.source, :] = self.original_capacities[self.source, :]

        # Initial excess is load on source edges.
        self.excess = self.load[self.source, :].flatten().copy()

        # Active vertices are vertices with positive excess.
        self.active_vertices = np.argwhere(self.excess > 0).flatten().tolist()

        # A valid push is a push to a lower height. Initially there are none.
        self.valid_pushes = np.zeros(
            (self.num_vertices, self.num_vertices)).astype(int)

        # Residual capacities: remove initial flow from original capacities.
        self.residual_capacities = self.original_capacities - self.load

        # Dictionaries allow us to access edges in constant time.
        self.edge_val_to_key = {x: [] for x in self.graph.vertices}
        self.edge_key_to_val = {x: [] for x in self.graph.vertices}
        for i in range(self.num_vertices):
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
    def is_flow(self) -> bool:
        """ Flow is a preflow where vertices except sink and source have 0
        excess. """
        return np.delete(self.excess, [self.sink, self.source]).sum() == 0.

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
    def num_vertices(self) -> int:
        return self.graph.num_vertices

    @property
    def value(self) -> float:
        """ The total flow is measured as the excess at the sink. """
        return self.excess[self.sink]

    def push_potential(self, vertex: int, target: int) -> float:
        maximal_capacity = min(
            self.residual_capacities[vertex, target],
            self.excess[vertex].item())

        return maximal_capacity

    def push(self, vertex: int, target: int, flow_size: float) -> None:
        # Determine how much flow can be pushed.
        if flow_size == 0:
            raise ValueError("Trying to push an empty flow.")

        if self.verbose:
            print(f"Pushing {flow_size} from {vertex} to {target}.")

        # Push flow.
        self.excess[vertex] -= flow_size
        self.excess[target] += flow_size

        self.load[vertex, target] += flow_size

        self.residual_capacities[vertex, target] -= flow_size
        self.residual_capacities[target, vertex] += flow_size

        # After pushing a flow, the backwards flow becomes available.
        if vertex not in self.edge_key_to_val[target]:
            self.edge_key_to_val[target].append(vertex)
            self.edge_val_to_key[vertex].append(target)

        # Adapt active vertices.
        if self.excess[vertex] == 0:
            self.active_vertices.remove(vertex)

        if target not in [self.source, self.sink]:
            if target not in self.active_vertices:
                self.active_vertices.append(target)

    def relabel(self, vertex: int):
        """ Increase the height of a vertex. """
        self.height[vertex] += 1
        self.update_valid_pushes(vertex)

    def update_valid_pushes(self, vertex: int):
        """ Update valid pushes according to capacity and height.
            Only push downstream!"""

        # Allow all pushes that now became downstream.
        for neighbour in self.edge_key_to_val[vertex]:
            if self.height[neighbour] < self.height[vertex]:
                if self.residual_capacities[vertex, neighbour] > 0:
                    self.valid_pushes[vertex, neighbour] = 1

        # Forbid all pushes that are now upstream.
        for neighbour in self.edge_val_to_key[vertex]:
            if self.height[neighbour] <= self.height[vertex]:
                self.valid_pushes[neighbour, vertex] = 0


class PushRelabel:
    def __init__(self, config: PushRelabelConfiguration):
        self.config = config
        self.start_time = None

    @staticmethod
    def choose_active_vertex(active_vertices: List[int], heights: List[int]) -> int:
        """ Choose the highest of several active vertices.
            Tie-breaker: lowest index. """

        if len(active_vertices) > 0:
            heights = [heights[x] for x in active_vertices]
            return active_vertices[np.argmax(heights).item()]
        else:
            raise RuntimeError('No active vertices.')

    @staticmethod
    def choose_push_target(targets: List[int]):
        """ Simply pick the first of possible targets. """
        return targets[0]

    @property
    def current_runtime(self) -> float:
        if self.start_time is None:
            msg = ('Current runtime only available after calling '
                   'calculate_maximal_flow().')
            raise RuntimeError(msg)
        return time.time() - self.start_time

    def iterate_flow(self, flow: Preflow) -> Preflow:
        vertex = self.choose_active_vertex(flow.active_vertices, flow.height)
        targets = np.argwhere(flow.valid_pushes[vertex] == 1).flatten().tolist()

        # Only allow targets where actually flow can be pushed.
        targets = [t for t in targets if flow.push_potential(vertex, t) > 0]

        if len(targets) > 0:
            target = self.choose_push_target(targets)
            flow_size = flow.push_potential(vertex, target)
            flow.push(vertex, target, flow_size)
        else:
            flow.relabel(vertex)
            if flow.height[vertex] > flow.num_vertices * 2:
                msg = f"Height of vertex {vertex} exceeded limit."
                raise ValueError(msg)

        return flow

    def calculate_maximal_flow(self, graph: FlowNetwork) -> Preflow:
        """ In every iteration of push-relabel, a pre-flow is calculated. If
        this pre-flow is a valid flow, by method it is a maximal flow.

        Args:
            graph: FlowNetwork of intererst, including information about
            source, sink and capacities.

        Returns:
            - Preflow: A maximal flow. """

        self.start_time = time.time()
        preflow = Preflow(graph, verbose=self.config.verbose)

        iteration = 0

        while True:
            if iteration % 50_000 == 0:
                print(f"Current iteration: {iteration}")
                print(f"Current highest label = {max(preflow.height)}")
                print(f"Current runtime is {self.current_runtime}.")
                print(f"Maximal label is {2 * preflow.num_vertices - 1}.")

            if preflow.is_flow:
                return preflow

            if self.current_runtime > self.config.max_runtime:
                msg = "Maximum flow could not be calculated in time."
                raise RuntimeError(msg)

            preflow = self.iterate_flow(preflow)
            iteration += 1
