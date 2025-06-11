import warnings
from pathlib import Path
from typing import List

import pypsa

import pytest

from thesis.optimization.optimal_flow.flow_graph import FlowGraph
from thesis.optimization.optimal_flow.graph_builder import GraphBuilder
from thesis.optimization.optimal_flow.push_relabel import PushRelabel, \
    Preflow
from thesis.graph.utils.config import PushRelabelConfiguration
from thesis.graph.utils.format import Format
from thesis.graph.utils.results import extract_p_set


@pytest.fixture
def data_root() -> Path:
    return Format().data_root

@pytest.fixture(params=range(1))
def network_path(data_root: Path, request) -> List[Path]:
    paths = [data_root / "pypsa" / "trafo_violation" / "26-02-2025" /
             "25_grid_23"]

    return paths[request.param]

# [2, 4, 8, 16, 32, 64, 96]
@pytest.fixture(params=[2, 4, 8, 16, 32, 64, 96])
def pypsa_network(network_path, request) -> pypsa.Network:

    # Load network.
    n = pypsa.Network()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        n.import_from_csv_folder(network_path)

    # Set number of snapshots as parametrized.
    n.set_snapshots(n.snapshots[:request.param])

    return n.copy()

@pytest.fixture()
def push_relabel_config() -> PushRelabelConfiguration:
    return PushRelabelConfiguration()

@pytest.fixture
def graph_builder(push_relabel_config) -> GraphBuilder:
    return GraphBuilder(push_relabel_config)

@pytest.fixture
def pure_algorithm(push_relabel_config) -> PushRelabel:
    return PushRelabel(push_relabel_config)

@pytest.fixture
def flow_graph(pypsa_network, graph_builder) -> FlowGraph:
    flow_graph = graph_builder.build(pypsa_network)
    return flow_graph

@pytest.fixture
def maximal_flow(flow_graph, pure_algorithm) -> Preflow:
    maximal_flow = pure_algorithm.calculate_maximal_flow(flow_graph)
    return maximal_flow

@pytest.fixture
def network_optimized_by_pure(pypsa_network, flow_graph, maximal_flow, push_relabel_config
                              ) -> pypsa.Network:
    # Get loads corresponding to flow and convert them to megawatt.
    loads = extract_p_set(pypsa_network, flow_graph, maximal_flow)
    backwards_conversion = lambda x: x / (1000 ** push_relabel_config.conversion_order)
    mw_loads = loads.apply(backwards_conversion)

    # A network to store optimized load data.
    optimized_network = pypsa_network.copy()
    optimized_network.loads_t["p_set"] = mw_loads

    return optimized_network