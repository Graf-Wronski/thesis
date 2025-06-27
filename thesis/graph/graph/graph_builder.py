from typing import Optional, Callable

import numpy as np
import pandas as pd
import pypsa
from pandas import Timestamp

from thesis.graph.graph.flow_graph import FlowNetwork
from thesis.graph.graph.push_relabel import Preflow
from thesis.graph.utils.config import PushRelabelConfiguration
from thesis.graph.utils.network import get_p_capacity_mw, get_inflexible_loads, \
    get_heatpumps


class GraphBuilder:
    def __init__(self, config: PushRelabelConfiguration):
        """ Build a FlowGraph based on a PyPSA.Network with flexible and
        inflexible nodes. """
        self.config = config
        self.graph_id_to_grid_id = {}
        self.grid_id_to_graph_id = {}
        self.vertices = []

    @property
    def scale_to_kw(self) -> Callable[[float], int]:
        """ Avoid float calculations by scaling up. """
        conversion_factor = 1000 ** self.config.conversion_order
        return lambda x: int(x * conversion_factor)

    def _clear(self):
        """ Reset the GraphBuilder to process the next Network. """
        self.graph_id_to_grid_id = {}
        self.grid_id_to_graph_id = {}
        self.vertices = []

    def get_grid_id(self, graph_id: int) -> str:
        """ Each node represents a grid object at a certain time step. """
        return self.graph_id_to_grid_id[graph_id]

    def get_graph_id(self, grid_id: str, t: Optional[Timestamp] = None) -> int:
        """ For each time step, each grid object is represented by a node. """
        if t is not None:
            return self.grid_id_to_graph_id[grid_id][t]
        else:
            return self.grid_id_to_graph_id[grid_id]

    def add_vertex(self, grid_id: str, t: Optional[Timestamp] = None) -> int:
        """ Add a grid object (load, generation or bus) and assign a unique
        node index. If timestamp is attached, it is used for storing. """

        graph_idx = len(self.vertices)
        self.vertices.append(graph_idx)

        if t is not None:
            if grid_id not in self.grid_id_to_graph_id.keys():
                self.grid_id_to_graph_id[grid_id] = {}
            self.grid_id_to_graph_id[grid_id][t] = graph_idx
            self.graph_id_to_grid_id[graph_idx] = (grid_id, t)
        else:
            self.grid_id_to_graph_id[grid_id] = graph_idx
            self.graph_id_to_grid_id[graph_idx] = grid_id
        return graph_idx

    def build(self, n: pypsa.Network) -> FlowNetwork:
        """ Build the FlowGraph for a LV-grid.
        # ToDo: Interpretation of single units could be externalized.

        Args:
            n: A lv-grid that is translated to a flow graph. All
                snapshots of the grid are regarded.

        Returns:
            FlowNetwork: A directed graph where grid capacities as well as
                power demand and generation are expressed as edge weights.
                Meeting all demands while not violating capacities is
                equivalent to solving the maximal flow on the returned graph
                (Caveat: A maximal flow might not meet all demands if
                meeting all demands is not possible). """

        # Clear results from last build.
        self._clear()

        # Add meta source node. It will provide all energy produced.
        self.add_vertex('Source')
        source_idx = self.get_graph_id('Source')  # Should be 0.

        # Add meta sink node. It will suck in all energy consumed.
        self.add_vertex('Sink')
        sink_idx = self.get_graph_id('Sink') # Should be 1.

        node_description = {}

        edges_with_capacities = []
        # For graph algorithm we treat capacities as int.
        capacities_mw = get_p_capacity_mw(n)
        capacities = {x: self.scale_to_kw(y)
                      for x,y in capacities_mw.items()}

        # Base graph: Extract buses and lines.
        for bus in n.buses.index:
            for t in n.snapshots:
                self.add_vertex(bus, t)

        for line in n.lines.itertuples():
            capacity = capacities[line.Index]
            for t in n.snapshots:
                node0_idx = self.get_graph_id(line.bus0, t)
                node1_idx = self.get_graph_id(line.bus1, t)
                edges_with_capacities.append((node0_idx, node1_idx, capacity))
                edges_with_capacities.append((node1_idx, node0_idx, capacity))

        if len(n.transformers) != 1:
            msg = "Network is assumed to have exactly one transformer."
            raise NotImplementedError(msg)

        for trafo in n.transformers.itertuples():
            capacity = capacities[trafo.Index]
            if trafo.bus0 != "Slack":
                raise Warning(f"Slack node is not bus0 of {trafo.Index}.")
            for t in n.snapshots:
                # Add transformer nodes for each timestep.
                node0_idx = self.get_graph_id(trafo.bus0, t)
                node1_idx = self.get_graph_id(trafo.bus1, t)
                edges_with_capacities.append((node0_idx, node1_idx, capacity))
                edges_with_capacities.append((node1_idx, node0_idx, capacity))

                # Edge between source and slack with arbitrary high capacity.
                # ToDo: Slack can also serve as sink.
                if self.config.slack_as_source:
                    edge_source_slack = (source_idx, node0_idx, 1000*capacity)
                    edges_with_capacities.append(edge_source_slack)
                else:
                    msg = "Slack as sink not implemented, yet."
                    raise NotImplementedError(msg)

        # Add inflexible loads.
        inflexible_loads = get_inflexible_loads(n)
        if len(inflexible_loads) != len(inflexible_loads["bus"].unique()):
            msg = "Inflexible loads are assumed to have unique bus."
            raise NotImplementedError(msg)

        for load in inflexible_loads.itertuples():
            for t in n.snapshots:
                # Each load gets its own node that is attached to resp. bus.
                load_idx = self.add_vertex(load.Index, t)
                bus_idx = self.get_graph_id(load.bus, t)
                load_mw = n.loads_t["p_set"].loc[t, load.Index]

                # Load size is depicted as capacity to sink.
                load_kw = int(self.scale_to_kw(load_mw))
                edges_with_capacities.append((bus_idx, load_idx, load_kw))
                edges_with_capacities.append((load_idx, sink_idx, load_kw))

        # Add flexible loads.
        heatpumps = get_heatpumps(n)
        if len(heatpumps) != len(heatpumps["bus"].unique()):
            msg = "Heatpumps are assumed to have unique bus."
            raise NotImplementedError(msg)

        for hp in heatpumps.itertuples():
            max_kw_per_t = self.scale_to_kw(hp.p_set)
            total_demand = n.loads_t["p"].loc[:, hp.Index].apply(
                self.scale_to_kw).sum()

            for k, t  in enumerate(n.snapshots):
                # Each load gets its own node that is attached to resp. bus.
                hp_idx = self.add_vertex(hp.Index, t)
                bus_idx = self.get_graph_id(hp.bus, t)
                load_mw = n.loads_t["p_set"].loc[t, hp.Index]

                # Only max_kw_per_t can be applied per time step.
                edges_with_capacities.append((bus_idx, hp_idx, max_kw_per_t))
                # The load added at each timestep is accumulated at hp node.
                if k > 0:
                    t_minus_one = n.snapshots[k - 1]
                    if not t - t_minus_one == pd.Timedelta(minutes=15):
                        msg = "Delta t is assumed to be 15 minutes"
                        raise NotImplementedError(msg)
                    past_hp = self.get_graph_id(hp.Index, t_minus_one)
                    edges_with_capacities.append((past_hp, hp_idx, total_demand))

            # In the end, the flexible load has to meet the total power.
            edges_with_capacities.append((hp_idx, sink_idx, total_demand))

        n_nodes = len(self.vertices)
        capacities = np.zeros((n_nodes, n_nodes))
        for i, j, capacity in edges_with_capacities:
            capacities[i, j] = capacity

        graph = FlowNetwork(
            vertices=self.vertices,
            source=source_idx,
            sink=sink_idx,
            capacities=capacities,
            meta={"grid_id_to_graph_id": self.grid_id_to_graph_id})

        return graph

    def reconstruct(self, flow: Preflow, n: pypsa.Network) -> pypsa.Network:
        """ Apply loads from flow to network. """
        network = n.copy()
        inflexible_loads = get_inflexible_loads(network)
        heat_pumps = get_heatpumps(network)

        for name, load in pd.concat([inflexible_loads, heat_pumps]).iterrows():
            def get_load_mw(t: Timestamp) -> float:
                load_id = self.get_graph_id(name, t)
                bus_id = self.get_graph_id(load.bus, t)
                load_w = flow.load[bus_id, load_id] - flow.load[load_id, bus_id]
                load_mw = load_w / 1000**2
                return load_mw

            load_ts = [get_load_mw(t) for t in network.snapshots]
            network.loads_t["p_set"][name] = load_ts

        network.lpf()

        return network
