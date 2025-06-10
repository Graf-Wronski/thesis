""" Test runtime for PURE-Optimization with different conversion factors. """
import warnings
import time

import numpy as np
import pypsa

from thesis.optimization.optimal_flow.graph_builder import GraphBuilder
from thesis.optimization.optimal_flow.push_relabel import PushRelabel
from thesis.complex_network.utils.config import PushRelabelConfiguration
from thesis.complex_network.utils.format import Format
from thesis.complex_network.utils.results import extract_p_set


def main(conversion_factor, n_snapshots):
    start_time = time.time()

    data_root = Format().data_root
    path = data_root / "pypsa" / "trafo_violation" / "26-02-2025" / "25_grid_23"

    n = pypsa.Network()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        n.import_from_csv_folder(path)

    # Set number of snapshots as parametrized.
    n.set_snapshots(n.snapshots[0:n_snapshots])

    push_relabel_config = PushRelabelConfiguration(
        conversion_order=conversion_factor)
    graph_builder = GraphBuilder(push_relabel_config)
    pure_algorithm = PushRelabel(push_relabel_config)
    flow_graph = graph_builder.build(n)
    maximal_flow = pure_algorithm.calculate_maximal_flow(flow_graph)

    loads = extract_p_set(n, flow_graph, maximal_flow)
    backwards_conversion = lambda x: x / (
                1000 ** push_relabel_config.conversion_order)
    mw_loads = loads.apply(backwards_conversion)

    # A network to store optimized load data.
    optimized_network = n.copy()
    optimized_network.loads_t["p_set"] = mw_loads

    load_error = n.loads_t["p_set"] - optimized_network.loads_t["p_set"]
    total_load_error = np.absolute(load_error.to_numpy().sum())

    runtime = time.time() - start_time

    return runtime, total_load_error

if __name__ == '__main__':
    conversion_factors = [1, 2]
    number_of_snapshots = [2, 4, 8, 16, 32]

    for nos in number_of_snapshots:
        for cf in conversion_factors:
            runtime, error = main(cf, nos)
            print(f"Number of snapshots: {nos}, Conversion factor: {cf}")
            print(f"Runtime: {runtime}, Error: {error}")
