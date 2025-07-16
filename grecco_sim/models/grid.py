import functools
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pypsa

from pathlib import Path

from grecco_sim.util import configs, network_io, build
from grecco_sim.graph.utils import network

logging.getLogger("pypsa").setLevel(logging.WARNING)


class Grid:
    def __init__(self, simulation_config: configs.SimulationConfiguration):

        self.simulation_config = simulation_config
        self.dt_h = simulation_config.dt_h
        self.time_index = simulation_config.time_index

        self.n = pypsa.Network()
        
        print("Importing pypsa.Network ...")
        
        with warnings.catch_warnings(action="ignore"):
            p = simulation_config.grid_data_path
            self.n.import_from_csv_folder(p)

        # ToDo: The setting of snapshots is not very clean.
        n_index = self.time_index.tz_localize(None)
        self.n.set_snapshots(snapshots=n_index)

        print("... Done.")

        if self.simulation_config.transformer_lim:
            transformer_lim = self.simulation_config.transformer_lim
            self.n.transformers["capacity"] = transformer_lim
        if self.simulation_config.feeder_lim:
            feeder_lim = self.simulation_config.feeder_lim
            self.n.lines["capacity"] = feeder_lim

        system_buses = network_io.get_system_buses(self.n)
        self.sys_ids = [build.sys_id(b) for b in system_buses]

        params, p_set = network_io.get_baseload(self.n)
        # Rename data to carry system names.
        name_dict = build.id_mapping(params, unit="baseload")
        self.baseload_params = params.rename(index=name_dict)
        self.p_baseload_t = p_set.rename(columns=name_dict)

        if simulation_config.use_pv:
            params, p_set = network_io.get_pv(self.n)
            name_dict = build.id_mapping(params, unit="pv")
            self.pv_params = params.rename(index=name_dict)
            self.p_pv_t = p_set.rename(columns=name_dict)

        if simulation_config.use_batteries:
            params = network_io.get_bss(self.n)
            name_dict = build.id_mapping(params, unit="bat")
            self.bat_params = params.rename(index=name_dict)

        if simulation_config.use_heatpumps:
            params = network_io.get_hp(self.n)
            name_dict = build.id_mapping(params, unit="hp")
            self.hp_params = params.rename(index=name_dict)
            
        if simulation_config.use_ev:
            request_path = simulation_config.charging_request_path
            params, requests = network_io.get_ev(self.n, request_path)
            self.ev_params = params.rename(index=name_dict)
            self.requests = network_io.preprocess_charging_requests(
                requests, self.simulation_config.time_index)

        self.units_at = {sys_id: self._units_at(sys_id)
                         for sys_id in self.sys_ids}

        for load in self.n.loads.index:
            self.n.loads_t["p_set"][load] = np.zeros(len(self.time_index))

        # Map buses and lines to their respective feeder idx.
        self.feeder_map = self.determine_feeders()

        # Extract limits for optimization.
        self.capacities = network.get_p_capacity_mw(self.n)

    @property
    def feeder(self) -> set[int]:
        return set(self.feeder_map.values())

    @property
    def trafo_p_lim(self) -> float:
        return self.capacities[self.n.transformers.index[0]]

    @functools.cached_property
    def unit_dict(self) -> dict:
        """ A dict with all units for look up operations. Key is unit type. """

        # Default values are empty lists.
        units = {unit: [] for unit in ["baseload", "pv", "bat", "hp", "ev"]}

        units["baseload"] = self.baseload_params.index.to_list()

        if self.simulation_config.use_pv:
            units["pv"] = self.pv_params.index.to_list()

        if self.simulation_config.use_batteries:
            units["bat"] = self.bat_params.index.to_list()

        if self.simulation_config.use_heatpumps:
            units["hp"] = self.hp_params.index.to_list()

        if self.simulation_config.use_ev:
            units["ev"] = self.ev_params.index.to_list()

        return units

    def _units_at(self, sys_id: str) -> list[str]:
        """ Return the available units for a given system. """

        units = []

        for unit in ["baseload", "pv", "bat", "hp", "ev"]:
            if f"{sys_id}_{unit}" in self.unit_dict[unit]:
                units.append(unit)

        return units

    def determine_feeders(self) -> dict[str, int]:
        """ A feeeder is defined as subtree rooted in main bus bar (root bus).

        Returns:
            dict[str, int]: Map bus or line to feeder index. """

        feeder_map = {}

        # Copy network to avoid side effects.
        n = self.n.copy()

        # Remove slack and main bus.
        if len(n.transformers) != 1:
            raise ValueError("Multiple transformers not supported.")

        slack = n.transformers.iloc[0]["bus0"]
        root_bus = n.transformers.iloc[0]["bus1"]

        n.remove("Bus", slack)
        n.remove("Bus", root_bus)
        n.remove("Transformer", n.transformers.index[0])

        # Root segments have to be removed for topology determination.
        # Their feeder is determined later by the second bus.
        root_segments = []

        for line_idx, line in n.lines.iterrows():
            if line["bus0"] == root_bus:
                root_segments.append((line_idx, line["bus1"]))
                n.remove("Line", line_idx)
            if line["bus1"] == root_bus:
                root_segments.append((line_idx, line["bus0"]))
                n.remove("Line", line_idx)

        n.determine_network_topology()

        for bus_idx, bus_data in n.buses.iterrows():
            feeder_map[bus_idx] = int(bus_data["sub_network"])
            
        for line_idx, line_data in n.lines.iterrows():
            feeder_map[line_idx] = int(line_data["sub_network"])

        for line_idx, bus in root_segments:
            feeder_map[line_idx] = feeder_map[bus]

        return feeder_map

    @staticmethod
    def build_sys_id(load_index: int, bus_name: str) -> str:
        """ Unique identifier for energy management systems.

        Args:
            load_index: Index of baseload associated with EMS.
            bus_name: Name of bus the EMS is attached to.

        Returns:
            str: Unique system identifier. """

        return f"bus_{bus_name}_load_{load_index}"

    def get_system_ts_dict(self, sys_id: str) -> dict:
        """ Return (inflexible) timeseries input data.

        Keys are: {sys_id}_{unit}_{attr}, i.e. sys_at_bus_3_pv_p. """

        # ToDo: Maybe move weather data here.

        data = dict()

        unit_id = f"{sys_id}_baseload"
        data[f"{unit_id}_p"] = self.p_baseload_t.loc[:, unit_id]

        if self.simulation_config.use_pv:
            unit_id = f"{sys_id}_pv"

            if unit_id in self.p_pv_t.columns:  # Not every node has pv.
                data[f"{unit_id}_p"] = self.p_pv_t.loc[:, unit_id]

        return data

    def congestion(self, snapshots: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """ Amount of congestion for transmission gear at given time steps.

        Args:
            snapshots:

        Returns:
            pd.DataFrame: Columns are gear names, rows are time steps. Value is
                amount of congestion, 0. if capacities are respected.
        """


        n = self.n.copy()

        if snapshots is not None:
            n.set_snapshots(snapshots=snapshots)

        n.lpf()
        p_transmission = network.get_p_transmission_mw(n)
        congestion = (p_transmission.abs() - self.capacities).clip(lower=0)

        return congestion

    @property
    def ptdf_matrix(self) -> np.ndarray:
        """ Calculate the Power Transfer Distribution Factor matrix.

        Given a bus and a line the PTD-factor describes how much a change in
        load at the bus would affect the load at the line.

        Returns:
            np.ndarray: PTDF-matrix where rows correspond to lines and columns
             correspond to buses.

        Raises:
            NotImplementedError: Since PTDFs in pypsa are calculated on
                sub_network level, we assume whole network is connected.
        """

        self.n.determine_network_topology()

        if len(self.n.sub_networks) != 1:
            raise NotImplementedError("Network is assumed to be connected.")

        return self.n.sub_networks["obj"].iloc[0].calculate_PTDF()

    def write(self, p: Path):
        self.n.export_to_csv_folder(p / "network")

    def write_loads(self, state: dict[str, dict], t: int) -> None:
        """ Set grid state from simulation node state. """

        p_set_load = dict()
        p_set_gen = dict()
        p_set_bat = dict()

        for load in self.n.loads.index:
            bus = self.n.loads.loc[load, "bus"]
            data = state[build.sys_id(bus)]

            if "baseload" in load.lower():
                p_set_load[load] = data["baseload_p_model"]

            elif "heat" in load.lower():
                if not self.simulation_config.use_heatpumps:
                    continue
                else:
                    p_set_load[load] = data["hp_p_model"]

            else:
                raise NotImplementedError(f"Unknown load {load}.")

        if self.simulation_config.use_pv:
            for generator in self.n.generators.index:

                # Slack generators are not set as they are dependent variables.
                if self.n.generators.loc[generator, "control"] == "Slack":
                    continue

                bus = self.n.generators.loc[generator, "bus"]
                data = state[build.sys_id(bus)]

                if "pv" in generator.lower():
                    p_set_gen[generator] = data["pv_p_model"]
                else:
                    msg = f"Unknown generator {generator}."
                    raise NotImplementedError(msg)

        if self.simulation_config.use_batteries:
            storage_units = self.n.storage_units.query("type == 'h0_battery'")
            for storage in storage_units.index:
                bus = self.n.storage_units.loc[storage, "bus"]
                data = state[build.sys_id(bus)]
                p_set_bat[storage] = data["bat_p_model"]

        if self.simulation_config.use_ev:
            pass

        # PyPSA snapshots are not localized. PyPSA loads are in MW.
        time_index = [self.time_index[t].tz_localize(None)]
        p_set_load = pd.DataFrame(p_set_load , index=time_index) / 1000
        self.n.loads_t["p_set"].update(p_set_load)

        p_set_gen = pd.DataFrame(p_set_gen, index=time_index) / 1000
        self.n.generators_t["p_set"].update(p_set_gen)

        p_set_bat = pd.DataFrame(p_set_bat, index=time_index) / 1000
        self.n.storage_units_t["p_set"].update(p_set_bat)
