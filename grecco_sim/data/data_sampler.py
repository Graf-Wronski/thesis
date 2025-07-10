from pathlib import Path
import random
from typing import List

import numpy as np
import pandas as pd
import pypsa

from grecco_sim.data.simbench_data import SimBenchData, UnitProfiles
from grecco_sim.data.topology_sampler import TopologySampler
from grecco_sim.graph.utils.config import SamplerConfiguration
from grecco_sim.graph.utils.format import Format
from grecco_sim.graph.utils.units import CustomPyPSAObjects


class DataSampler:
    def __init__(self, node_data: Path,
                 config: SamplerConfiguration = SamplerConfiguration()):

        self.seed = config.seed

        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.config = config
        self.topology_sampler = TopologySampler()
        self.simbench = SimBenchData(node_data)

    def sample(self, len_timeseries: int = 96, n_nodes: int = 10) -> (
            pypsa.Network):

        # Create a Network 'Container' and set index.
        lv_grid = pypsa.Network()
        custom_transformers = CustomPyPSAObjects().transformers()
        lv_grid.transformer_types = pd.concat(
            [lv_grid.transformer_types, custom_transformers])

        # Start at a random point in time and extract the time window.
        timestamps =  self.simbench.load_t.index.values
        timestamps.sort()

        start_idx = random.randint(0, len(timestamps) - len_timeseries)
        end_idx = start_idx + len_timeseries
        start_time = timestamps[start_idx]
        snapshot_range = pd.date_range(start=start_time,
                                       periods=len_timeseries, freq="15min")

        if not (snapshot_range == timestamps[start_idx:end_idx]).all():
            msg = "Timestamps and snapshots index do not match."
            raise ValueError(msg)

        lv_grid.set_snapshots(snapshot_range.strftime(Format().date))

        # Sample topology.
        topology = self.topology_sampler.sample(n_loads=n_nodes)

        # Add buses.
        lv_grid.add(class_name="Bus", name="Slack", v_nom=10.0, type="Slack")
        lv_grid.add(class_name="Bus", name=topology.bus_ids, v_nom=0.4)

        # Add slack generator.
        lv_grid.add(
            class_name="Generator",
            name="ExternalGrid",
            bus="Slack",
            p_nom=10,
            control="Slack")

        # 0.4 MVA transformer between lv radial network and external mv grid.
        transformer_type = self.config.transformer_type
        base_data = lv_grid.transformer_types.loc[transformer_type].to_dict()

        lv_grid.add(
            class_name="Transformer",
            name="MV/LV Transformer",
            bus0="Slack",
            bus1="Bus 0",
            type=transformer_type)

        if self.config.s_nom:
            lv_grid.lpf()  # lpf is called to overtake transformer values.
            lv_grid.transformers.at["MV/LV Transformer", "s_nom"] = (
                self.config.s_nom)
            lv_grid.transformers.at["MV/LV Transformer", "type"] = ""

        # Sample units.
        unit_profiles = self.sample_units(topology.load_ids)
        lv_grid = self.add_units_to_network(unit_profiles, lv_grid)

        # Add lines of type NAYY 4x150 SE.
        line_names = []
        bus0_list, bus1_list = [], []

        for idx, edge in enumerate(topology.edges):
            line_names.append(f"Line {idx}")
            bus0_list.append(edge[0]), bus1_list.append(edge[1])

        lv_grid.add(
            "Line",
            name=line_names,
            bus0=bus0_list,
            bus1=bus1_list,
            type="NAYY 4x150 SE",
            length=0.1)

        return lv_grid

    def sample_units(self, bus_ids: List[int]) -> UnitProfiles:
        inflex = {"bus_id": [], "profile": []}
        heatpump = {"bus_id": [], "profile": []}
        generation = {"bus_id": [], "profile": []}
        storage = {"bus_id": [], "profile": []}

        for bus_id in bus_ids:
            inflex["bus_id"].append(bus_id)
            inflex["profile"].append(np.random.choice(self.simbench.household_tags))

            # Eventually add heatpump.
            if np.random.random() < self.config.p_heatpump:
                heatpump["bus_id"].append(bus_id)
                heatpump["profile"].append(
                    np.random.choice(self.simbench.heatpump_tags))

            # Eventually add PV (and storage).
            if np.random.random() < self.config.p_pv:
                generation["bus_id"].append(bus_id)
                generation["profile"].append(np.random.choice(self.simbench.pv_tags))
                storage_tag = (f"Storage_{generation['profile'][-1]}"
                               f"_{inflex['profile'][-1]}")
                if storage_tag in self.simbench.storage_tags:
                    storage["bus_id"].append(bus_id)
                    storage["profile"].append(storage_tag)

        return UnitProfiles(inflex, heatpump, generation, storage)

    def add_units_to_network(
            self,
            unit_profiles: UnitProfiles,
            lv_grid:pypsa.Network)-> pypsa.Network:

        t = lv_grid.snapshots

        # Add inflexible loads.
        if len(unit_profiles.inflex["bus_id"]) > 0:
            profiles = unit_profiles.inflex["profile"]
            buses = unit_profiles.inflex["bus_id"]
            gen_data, gen_t_data = self.simbench.load, self.simbench.load_t

            # Sample unit data for each profile.
            pools = [gen_data[gen_data["profile"] == p] for p in profiles]
            units  = pd.DataFrame([pool.sample().squeeze() for pool in pools])

            # Add bus and carrier information to units.
            units["bus"] = unit_profiles.inflex["bus_id"]
            units = units.to_dict()
            units["carrier"] = "inflex"

            # If adding multiple units network.add() requires consistent names.
            names = [f"{bus}_Inflex_{idx}" for idx, bus in enumerate(buses)]
            inflex_p_t = gen_t_data.loc[t, pd.Series(profiles) +  "_pload"]
            inflex_q_t = gen_t_data.loc[t, pd.Series(profiles) +  "_qload"]
            inflex_p_t.columns, inflex_q_t.columns = names, names
            units["p_set"] = inflex_p_t * units["pLoad"].values()
            units["q_set"] = inflex_q_t * units["qLoad"].values()

            lv_grid.add(class_name="Load", name=names, **units)
            
        # Add Heatpumps
        if len(unit_profiles.heatpump["bus_id"]) > 0:
            profiles = unit_profiles.heatpump["profile"]
            buses = unit_profiles.heatpump["bus_id"]
            gen_data, gen_t_data = self.simbench.load, self.simbench.load_t

            # Sample unit data for each profile.
            pools = [gen_data[gen_data["profile"] == p] for p in profiles]
            units = pd.DataFrame([pool.sample().squeeze() for pool in pools])

            # Add bus and carrier information to units.
            units["bus"] = unit_profiles.heatpump["bus_id"]
            # orient="list" is importans as otherwise dicts are created that
            # loose information
            units = units.to_dict(orient="list")
            units["carrier"] = "heat_pump"

            # If adding multiple units network.add() requires consistent names.
            names = [f"{bus}_Heatpump_{idx}" for idx, bus in enumerate(buses)]
            heatpump_p_t = gen_t_data.loc[t, pd.Series(profiles) +  "_pload"]
            heatpump_q_t = gen_t_data.loc[t, pd.Series(profiles) +  "_qload"]
            heatpump_p_t.columns, heatpump_q_t.columns = names, names
            units["p_set"] = heatpump_p_t * units["pLoad"]
            units["q_set"] = heatpump_q_t * units["qLoad"]

            lv_grid.add(class_name="Load", name=names, **units)

        # Add Storages
        if len(unit_profiles.generation["bus_id"]) > 0:
            profiles = unit_profiles.generation["profile"]
            buses = unit_profiles.generation["bus_id"]
            gen_data, gen_t_data = self.simbench.gen, self.simbench.gen_t

            # Sample unit data for each profile.
            pools = [gen_data[gen_data["profile"] == p] for p in profiles]
            units = pd.DataFrame([pool.sample().squeeze() for pool in pools])

            # Add bus and carrier information to units.
            units["bus"] = unit_profiles.generation["bus_id"]
            units = units.to_dict()
            units["carrier"] = "solar"
            units["control"] = "PQ"
            units["p_nom"] = units["pRES"]

            # If adding multiple units network.add() requires consistent names.
            names = [f"{bus}_PV_{idx}" for idx, bus in enumerate(buses)]
            generation_p_t = gen_t_data.loc[t, profiles]
            generation_p_t.columns = names
            units["p_set"] = generation_p_t * units["p_nom"].values()

            lv_grid.add(class_name="Generator", name=names, **units)

            # Add Storage
            if len(unit_profiles.storage["bus_id"]) > 0:
                profiles = unit_profiles.storage["profile"]
                buses = unit_profiles.storage["bus_id"]
                storage_data = self.simbench.storage
                storage_t_data = self.simbench.storage_t
                
                # Sample unit data for each profile.
                pools = [storage_data[storage_data["profile"] == p] for p in
                         profiles]
                units = pd.DataFrame(
                    [pool.sample().squeeze() for pool in pools])

                # Add bus and carrier information to units.
                units["bus"] = unit_profiles.storage["bus_id"]
                units["p_nom"] = units["sR"]
                units = units.to_dict()

                # If adding multiple units network.add() requires consistent names.
                names = [f"{bus}_Storage_{idx}" for idx, bus in enumerate(
                    buses)]
                storage_p_t = storage_t_data.loc[t, profiles]
                storage_p_t.columns = names
                units["p_set"] = storage_p_t * units["p_nom"].values()

                lv_grid.add(class_name="StorageUnit", name=names, **units)

        return lv_grid
