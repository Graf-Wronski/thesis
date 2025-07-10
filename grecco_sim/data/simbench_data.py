from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from pathlib import Path

from grecco_sim.graph.utils.format import Format


@dataclass
class UnitProfiles:
    inflex: Dict
    heatpump: Dict
    generation: Dict
    storage: Dict

class SimBenchData:
    """ Sample load (and generation) data from a SimBench csv grid."""

    def __init__(self, import_dir: Path):
        self.import_dir = import_dir

        # Load units and time series data and store it as attributes.
        def load_csv(csv_name: str) -> pd.DataFrame:
            csv_path = self.import_dir / csv_name
            df = pd.read_csv(csv_path, sep=";", on_bad_lines="warn")

            # For time-series: Remove duplicates by averaging and set index.
            if "time" in df.columns:
                old_len = len(df)
                df = df.groupby(["time"]).mean()
                print(f"{old_len - len(df)} duplicates in {csv_name} "
                      f"were averaged. ")
            return df

        time_format = Format().date
        self.gen_t = load_csv("RESProfile.csv")
        self.load_t = load_csv("LoadProfile.csv")
        self.storage_t = load_csv("StorageProfile.csv")

        for time_varying_data in [self.gen_t, self.load_t, self.storage_t]:
            time_varying_data.index = pd.DatetimeIndex(time_varying_data.index).strftime(time_format)

        self.load = load_csv("Load.csv").query('id.str.contains("LV")')
        self.gen = load_csv("RES.csv").query('id.str.contains("LV")')
        self.storage = load_csv("Storage.csv").query('id.str.contains("LV")')

    # The names of all SimBench profiles. For details see:
    # https://simbench.de/wp-content/uploads/2021/09/simbench_documentation_de_1.1.0.pdf
    @property
    def household_tags(self) -> List[str]:
        """ Simbench household profile tags indicated by 'H0'. """
        return [f"H0-{x}" for x in ["A", "B", "C", "G", "L"]]

    @property
    def heatpump_tags(self) -> List[str]:
        """ Simbench heatpump profile tags indicated by 'Air_' or 'Soil'."""
        return ["Soil_Alternative_1", "Soil_Alternative_2",
                "Air_Semi-Parallel_1", "Air_Semi-Parallel_2",
                "Air_Alternative_1", "Air_Alternative_2",
                "Air_Parallel_1", "Air_Parallel_2"]

    @property
    def home_ev_tags(self) -> List[str]:
        """ Simbench EV profile tags indicated by 'HLS' (= Heimladesäule)."""
        return ["HLS_A_3.7", "HLS_B_3.7", "HLS_C_3.7",
                "HLS_A_11.0", "HLS_B_11.0", "HLS_A_22.0",]

    @property
    def pv_tags(self) -> List[str]:
        """ Simbench PV profile tags PV1 - PV8. """
        return [f"PV{idx}" for idx in range(1, 9)]

    @property
    def storage_tags(self) -> List[str]:
        """ Simbench PV pv storage profile tags. """
        # ToDo: Find out why they are restricted like this.
        return ['Storage_PV1_H0-B', 'Storage_PV1_H0-C', 'Storage_PV2_H0-A',
                'Storage_PV2_H0-B', 'Storage_PV2_H0-G', 'Storage_PV3_H0-A',
                'Storage_PV3_H0-B', 'Storage_PV3_H0-G', 'Storage_PV4_H0-A',
                'Storage_PV4_H0-C', 'Storage_PV4_H0-G', 'Storage_PV4_H0-L',
                'Storage_PV5_H0-A', 'Storage_PV5_H0-B', 'Storage_PV5_H0-G',
                'Storage_PV5_H0-L', 'Storage_PV6_H0-B', 'Storage_PV6_H0-C',
                'Storage_PV7_H0-B', 'Storage_PV7_H0-C', 'Storage_PV7_H0-G',
                'Storage_PV7_H0-L', 'Storage_PV8_H0-B', 'Storage_PV8_H0-C',
                'Storage_PV8_H0-L']

    def sample_units(self, bus_ids: List[int]) -> UnitProfiles:
        inflex = {"bus_id": [], "profile": []}
        heatpump = {"bus_id": [], "profile": []}
        generation = {"bus_id": [], "profile": []}
        storage = {"bus_id": [], "profile": []}

        for bus_id in bus_ids:
            inflex["bus_id"].append(bus_id)
            inflex["profile"].append(np.random.choice(self.household_tags))

            # Eventually add heatpump.
            if np.random.random() < 0.2:
                heatpump["bus_id"].append(bus_id)
                heatpump["profile"].append(np.random.choice(self.heatpump_tags))
                
            # Eventually add PV (and storage).
            if np.random.random() < 0.5:
                generation["bus_id"].append(bus_id)
                generation["profile"].append(np.random.choice(self.pv_tags))
                storage_tag = (f"Storage_{generation['profile'][-1]}"
                               f"_{inflex['profile'][-1]}")
                if storage_tag in self.storage_tags:
                    storage["bus_id"].append(bus_id)
                    storage["profile"].append(storage_tag)

        return UnitProfiles(inflex, heatpump, generation, storage)
