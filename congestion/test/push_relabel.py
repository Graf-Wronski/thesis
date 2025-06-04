import numpy as np
import pytest

from congestion.optimization.optimal_flow.flow_graph import \
    FlowGraph
from congestion.optimization.optimal_flow.push_relabel import \
    PushRelabel
from congestion.utils.config import PushRelabelConfiguration
from congestion.utils.network import utilization_ratio


class TestGraphs:
    """ Create different test graphs."""

    @property
    def test_graph_1(self) -> FlowGraph:
        nodes = [x for x in range(5)]
        source = 0
        sink = 4
        edge_capacities = np.array([
            [0, 10, 3, 5, 0],
            [10, 0, 5, 0, 5],
            [3, 5, 0, 5, 5],
            [5, 0, 5, 0, 5],
            [0, 5, 5, 5, 0]
        ])

        graph = FlowGraph(nodes, source, sink, edge_capacities)

        return graph

    @property
    def test_graph_2(self) -> FlowGraph:
        return self.test_graph_4(10)

    @property
    def test_graph_3(self) -> FlowGraph:
        nodes = [x for x in range(7)]
        capacities = np.array([
            [0, 5, 10, 0, 0, 0, 0],
            [0, 0, 0, 3, 7, 0, 0],
            [0, 0, 0, 4, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0, 9],
            [0, 0, 0, 0, 0, 0, 0]
        ])

        source = 0
        sink = 6

        graph = FlowGraph(nodes, source, sink, capacities)

        return graph

    @staticmethod
    def test_graph_4(n_nodes) -> FlowGraph:
        if n_nodes > 10000:
            raise ValueError("Matrix exceeds memory.")

        nodes = [x for x in range(n_nodes)]
        capacities = np.zeros((n_nodes, n_nodes))
        source = 0
        sink = n_nodes - 1

        for i in nodes:
            for j in nodes:
                if i < j:
                    capacities[i, j] = i + 1
                    capacities[j, i] = i + 1

        graph = FlowGraph(nodes, source, sink, capacities)

        return graph

    @staticmethod
    def test_graph_5(n_nodes) -> FlowGraph:
        if n_nodes > 10000:
            raise ValueError("Matrix exceeds memory.")

        nodes = [x for x in range(n_nodes)]
        capacities = np.zeros((n_nodes, n_nodes))
        source = 0
        sink = n_nodes - 1

        for i in nodes:
            for j in nodes:
                if i < j and (j % 2 != i % 2):
                    capacities[i, j] = i + 1
                    capacities[j, i] = i + 1

        graph = FlowGraph(nodes, source, sink, capacities)

        return graph

x = TestGraphs()

@pytest.mark.parametrize("test_graph, expected",
                         [(x.test_graph_1, 15), (x.test_graph_2, 9),
                          (x.test_graph_3, 15)])
def test_push_relabel(test_graph, expected):
    config = PushRelabelConfiguration()
    pr = PushRelabel(config)
    flow = pr.calculate_maximal_flow(test_graph)
    assert flow.total_flow == expected

@pytest.mark.parametrize("n_nodes", [10 ** x for x in range(5)])
def test_push_relabel_runtime(n_nodes):
    test_graph = x.test_graph_4(n_nodes)
    config = PushRelabelConfiguration()
    pr = PushRelabel(config)
    flow = pr.calculate_maximal_flow(test_graph)
    expected = n_nodes - 1
    assert flow.total_flow == expected

@pytest.mark.parametrize("n_nodes", [10 ** x for x in range(5)])
def test_push_relabel_runtime_2(n_nodes):
    test_graph = x.test_graph_5(n_nodes)
    config = PushRelabelConfiguration()
    pr = PushRelabel(config)
    flow = pr.calculate_maximal_flow(test_graph)
    expected = np.floor(n_nodes / 2)
    assert flow.total_flow == expected


def test_flow_conservation(maximal_flow):
    """ The flows leaving the source and entering the sink are equal. """
    flow_leaving_source = (maximal_flow.net_load[maximal_flow.source, :]).sum()
    flow_entering_sink = (maximal_flow.net_load[:, maximal_flow.sink]).sum()
    assert flow_leaving_source == flow_entering_sink


def test_utilization_limit(network_optimized_by_pure):
    """ Maximal flow utilizes less than 100% capacity."""
    network_optimized_by_pure.lpf()
    network_optimized_by_pure.pf(use_seed=True)
    utilization = utilization_ratio(network_optimized_by_pure)

    assert np.all(utilization < 100)

def test_load_conservation(pypsa_network, network_optimized_by_pure):
    """ Optimized network should not have less load volume than original."""
    original_load_volume = pypsa_network.loads_t["p_set"].to_numpy().sum()
    optimized_load_volume = network_optimized_by_pure.loads_t["p_set"].to_numpy().sum()
    load_diff = pypsa_network.loads_t["p_set"] - network_optimized_by_pure.loads_t["p_set"]
    assert np.absolute(load_diff.to_numpy().sum()) < 0.001





