import pandas as pd
import pypsa

from congestion.optimization.optimal_flow.flow_graph import FlowGraph
from congestion.optimization.optimal_flow.push_relabel import Preflow


def extract_p_set(
        network: pypsa.Network,
        g: FlowGraph,
        flow: Preflow) -> pd.DataFrame:

    """ Extract p_set for a grid unit as given by a flow.

    Args:
        network: The network on which the FlowGraph was built on.
        g: The graph equivalent of the grid.
        flow: The (maximal) flow determining p_set.

    Returns:
        pd.DataFrame: p_set for all loads and all timestamps. As graph uses
            kw, results will also be in kw.
    """

    loads = network.loads_t["p_set"].copy()

    for load in network.loads.itertuples():
        grid_id_load = load.Index
        grid_id_bus = load.bus

        for t in network.snapshots:
            # Identify load and bus node in graph.
            graph_idx_load = g.meta["grid_id_to_graph_id"][grid_id_load][t]
            graph_idx_bus = g.meta["grid_id_to_graph_id"][grid_id_bus][t]

            # P_set is the load flowing from bus to load.
            p_set = flow.net_load[graph_idx_bus, graph_idx_load]
            loads.loc[t, grid_id_load] = p_set

    return loads