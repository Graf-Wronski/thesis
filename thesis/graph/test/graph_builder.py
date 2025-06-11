from thesis.graph.utils.graph import is_connected


def test_graph_builder(graph_builder, pypsa_network):
    """ Connectedness is only true for graphs where capacities are not 0. """
    graph = graph_builder.build(pypsa_network)
    assert(is_connected(graph.edges))


