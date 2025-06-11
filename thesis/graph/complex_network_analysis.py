import pypsa

from thesis.graph.graph import graph_builder, flow_graph
from thesis.graph.graph.flow_graph import FlowGraph
from thesis.graph.utils import config, graph


class ComplexNetworkAnalysis:
    """ Complex network analysis identifies the weaknesses of a lov-voltage
    grid using methods of complex network theory. """

    def __init__(
            self,
            network: pypsa.Network,
            builder_config: config.PushRelabelConfiguration):

        builder = graph_builder.GraphBuilder(builder_config)
        self.graph = builder.build(network)

    @staticmethod
    def get_minimal_cut(g: FlowGraph) -> flow_graph.MinimalCut:
        maximal_flow = ...
        return graph.flow_to_cut(maximal_flow)

    @staticmethod
    def get_edge_interbetweenness(self) -> None:
        return None

    def run(self):
        minimal_cut = self.get_minimal_cut(self.graph)
