import numpy as np
import pypsa
from tqdm.contrib import itertools

from thesis.graph.graph import graph_builder, flow_graph
from thesis.graph.graph.push_relabel import PushRelabel, Preflow
from thesis.graph.utils import config, graph


class ComplexNetworkAnalysis:
    """ Complex network analysis identifies the weaknesses of a lov-voltage
    grid using methods of complex network theory. """

    def __init__(
            self,
            network: pypsa.Network,
            builder_config: config.PushRelabelConfiguration):

        self.builder = graph_builder.GraphBuilder(builder_config)
        self.graph = self.builder.build(network)
        self.push_relabel = PushRelabel(builder_config)


    @staticmethod
    def get_edge_interbetweenness(self) -> None:
        return None

    def run(self):
        max_flow = self.push_relabel.calculate_maximal_flow(self.graph)
        min_cut = self.max_flow_to_min_cut(max_flow)

        if max_flow.value != min_cut.value:
            msg = "Maximal flow must match minimal cut."
            raise RuntimeError(msg)

        bottleneck = [(self.builder.get_grid_id(x), self.builder.get_grid_id(y))
                      for (x, y) in np.argwhere(min_cut.edges)]

        print(bottleneck)

    def get_residual_graph(self, flow: Preflow) -> flow_graph.FlowNetwork:
        """ The residual graph for a given flow is obtained by reducing the
        capacity of all edges by their respective flow capacity. All edges which
        result in zero capacity are removed.

        Args:
            n: The original FlowNetwork.
            flow: The "dual" of the residual graph.

        Returns:
            Preflow: The residual graph. """

        residual_graph = flow_graph.FlowNetwork(
            vertices=self.graph.vertices,
            source=self.graph.source,
            sink=self.graph.sink,
            capacities=flow.residual_capacities,
            meta=self.graph.meta)

        return residual_graph


    def max_flow_to_min_cut(self, max_flow: Preflow) -> flow_graph.Cut:
        """ Convert a maximal flow to a minimal cut. """

        residual_graph = self.get_residual_graph(max_flow)
        edges = residual_graph.edges

        v_source = graph.get_reachable_vertices(edges, max_flow.source)
        v_sink = [v for v in range(edges.shape[0]) if v not in v_source]

        # Minimal cut is defined by alle edges between v_source and v_sink.
        mask = np.zeros(self.graph.edges.shape)
        for x, y in itertools.product(v_source, v_sink):
            mask[x, y] = 1.
        edges = self.graph.edges.copy() * mask

        capacities = edges * self.graph.capacities

        minimal_cut = flow_graph.Cut(v_source, v_sink, capacities)

        return minimal_cut

