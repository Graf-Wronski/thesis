import abc
import os
import pathlib
from typing import List, Dict
import datetime
import warnings
import pandas as pd
import pytz
import numpy as np

from grecco_sim.models import grid
from grecco_sim.util import type_defs
from grecco_sim.simulator import simulation_setup


class InputDataLoader(abc.ABC):
    """
    Interface definition for Input data loading.
    Use different implementations for different kinds of input data.
    E.g. dummy data, data with grid information and without.
    """
    @abc.abstractmethod
    def get_input_data(self, sys_id: str, scenario: Dict) -> pd.DataFrame:
        """Return dataframe with necessary input data columns."""

    @abc.abstractmethod
    def get_parameters(self, sys_id: str) -> type_defs.SysPars:
        """Return list of parameterizations of agents in scenario."""

    @abc.abstractmethod
    def get_sys_ids(self) -> List[str]:
        "Return a list of unique ids of agents in the scenario."


class DummyInputDataLoader(InputDataLoader):
    """
    This class creates dummy data by randomly drawing power profiles.

    Enhance or write an alternative to read actual simulation data.
    """
    def __init__(self, time_index, scenario: Dict, pv_kwp=5., load_kw=8.):
        self.time_index = time_index

        self.sys_ids = [f"dummy_{i:02d}" for i in range(scenario["n_agents"])]

        self.load_kw = load_kw
        self.pv_kwp = pv_kwp
    
    def _get_load(self):
        noon = datetime.datetime(1, 1, 1, 12, 0, 0)
        diff = np.array([(datetime.datetime.combine(noon.date(), idx.time()) - noon).total_seconds() / 3600.
                         for idx in self.time_index])
        diff_sq = (diff ** 2 - 36) ** 2
        data = np.exp(-diff_sq * np.random.random(len(self.time_index)) / 150)
        return self.load_kw * data

    def _get_pv(self):
        noon = datetime.datetime(1, 1, 1, 12, 0, 0)
        diff = np.array([(datetime.datetime.combine(noon.date(), idx.time()) - noon).total_seconds() / 3600.
                         for idx in self.time_index])
        pv_ts = self.pv_kwp * np.exp(-(diff / 6.) ** 2)
        pv_ts += np.random.random(len(pv_ts))
        return pv_ts

    def _get_ev(self):
        ts_cp = np.array([np.NaN] * len(self.time_index), dtype=object)
        ts_cp[::20] = {"parking_time_h": 3, "capacity": 40., "init_soc": 0.7, "target_soc": 0.8}
        return ts_cp
    
    def get_input_data(self, sys_id: str, scenario: Dict) -> pd.DataFrame:
        data = {
            f"{sys_id}_load_p_load": self._get_load(),
            f"{sys_id}_pv_p_ac": self._get_pv()
        }
        
        ret = pd.DataFrame(index=self.time_index, data=data)
        return ret

    def get_parameters(self, sys_id: str) -> type_defs.SysPars:        
        assert sys_id in self.sys_ids, f"Requested sys_id '{sys_id}' not in available Ids"
        
        if sys_id == "dummy_00":
            return type_defs.SysParsPVBat(sys_id, c_sup=0.3, c_feed=0.1, eff=0.9, capacity=10., init_soc=0., dt_h=0.25, p_inv=5.)
        else:
            return type_defs.SysParsLoad(sys_id, 0.25, 0.3, 0.1, "load")
        
        # ev_params = dict(system="ev", eff=0.9, init_soc=0.5, p_lim_dc=11., p_lim_ac=11.)

    def get_sys_ids(self):
        return self.sys_ids

class SampleInputDataLoader(InputDataLoader):
    """
    Loads Sample data from example data input.
    """

    def __init__(
        self,
        time_index: pd.DatetimeIndex,
        scenario: Dict
    ):
        self.time_index = time_index

        if "data_path" in scenario:
            self.data_path = pathlib.Path(scenario["data_path"])
        else:
            # get the path of current file
            path = pathlib.Path(__file__).parent.absolute()
            # go two levels up
            self.data_path = path.parent.parent / "data" / "sample_scenario"
        
        self.scenario = scenario
        self.sys_ids = []

        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise ValueError(f"Given data path for scenario input {self.data_path} is non-existant")
        
        self._data = {}

        for comp in ["load", "pv"]:
            data = pd.read_csv(self.data_path / f"{comp}_data.csv", index_col=0)

            # Compare data length with time_index length
            # If Data is longer, then takes the first len(time_index) elements
            # If Data is shorter, raise exception
            if len(data) > len(self.time_index):
                data = data[: len(self.time_index)]
            elif len(data) < len(self.time_index):
                raise ValueError(
                    f"Input data for {comp} in scenario '{self.scenario['name']}' "
                    f"has shorter length ({len(data)}) then requested simulation horizon ({len(self.time_index)})"
                )
            self.time_index = self.time_index[: len(data)]

            self._data[comp] = data.reset_index(drop=True).set_index(self.time_index)

        # =================================================================
        # This could be a little more usefully selecting. But works for now I guess
        self.sys_ids = self._data["pv"].columns[:self.scenario["n_agents"]]
        # =================================================================

        if self.scenario["focus"] == "hp":
            data = pd.read_csv(self.data_path / "weather_data.csv", index_col=0)

            if len(data) > len(self.time_index):
                data = data[: len(self.time_index)]
            elif len(data) < len(self.time_index):
                raise ValueError(
                    f"Input data for weather in scenario '{self.scenario['name']}' "
                    f"has shorter length ({len(data)}) then requested simulation horizon ({len(self.time_index)})"
                )
            
            self._data["weather"] = data.reset_index(drop=True).set_index(self.time_index)

        # TODO: add check that agents (= _data.columns) are identical over all data sets.

    def get_input_data(self, sys_id: str, scenario: Dict) -> pd.DataFrame:
        data = {
            f"{sys_id}_load_p_load": self._data["load"][sys_id].values
        }
        if sys_id in self._data["pv"]:
            data[f"{sys_id}_pv_p_ac"] = self._data["pv"][sys_id].values
        else:
            data[f"{sys_id}_pv_p_ac"] = [0.] * len(self.time_index)
        
        if self.scenario["focus"] == "hp":
            data.update({_col: self._data["weather"][_col].values for _col in self._data["weather"]})
        
        ret = pd.DataFrame(index=self.time_index, data=data)
        return ret

    def get_parameters(self, sys_id: str) -> type_defs.SysPars:        
        assert sys_id in self.sys_ids, f"Requested sys_id '{sys_id}' not in available Ids"

        if self.scenario["focus"] == "pv_bat":
            return type_defs.SysParsPVBat(sys_id, c_sup=0.3, c_feed=0.1, eff=0.9, capacity=10., init_soc=0., dt_h=0.25, p_inv=5.)
        elif self.scenario["focus"] == "hp":
            return type_defs.SysParsHeatPump(sys_id, dt_h=0.25, c_sup=0.3, c_feed=0.1)
        else:
            raise ValueError(f"Unknown focus of sample data {self.scenario['focus']}")

        # ev_params = dict(system="ev", eff=0.9, init_soc=0.5, p_lim_dc=11., p_lim_ac=11.)

    def get_sys_ids(self):
        return self.sys_ids


class PyPsaGridInputLoader(InputDataLoader):
    """
    This class is a data container for the input data of the simulation, to be used by the simulator,
    and avoid reloading the data for each agent.
    The data is loaded from the paths specified in the constructor, and stored in the class attributes.
    If no path is provided, the framework checks if the default path contains
    the required files. If not, dummy data is used.
    """

    MWH_TO_KWH = 1000.

    def __init__(
        self,
        time_index: pd.DatetimeIndex,
        scenario: Dict,
    ):

        self.time_index = time_index
        if "grid_data_path" and "weather_data_path" in scenario:
            self.network_data_path = pathlib.Path(scenario["grid_data_path"])
            weather_data_path = pathlib.Path(scenario["weather_data_path"])
        else:
            # get the path of current file
            path = pathlib.Path(__file__).parent.absolute()
            # go two levels up
            self.network_data_path = path.parent.parent / "data" / "opfingen" / "grid"
            weather_data_path = path.parent.parent / "data" / "opfingen" / "weather_data.csv"

        self.weather_data = pd.read_csv(weather_data_path, index_col=0, date_format="%Y-%m-%d %H:%M:%S")
        self.weather_data.index = self.weather_data.index.tz_localize("utc")

        self._model_hps  = scenario["hp"]

        self._load_network_data()

    def _load_network_data(self):

        if not os.path.exists(self.network_data_path):
            raise ValueError(
                f"Provided path for network data '{self.network_data_path}' doesn't exist."
            )

        self.grid = grid.Grid(self.network_data_path)

    def get_sys_ids(self):
        return self.grid.sys_ids

    def get_input_data(self, sys_id: str, scenario: Dict) -> pd.DataFrame:
        assert sys_id in self.grid.sys_ids, f"Agent {sys_id} not known in grid data"
        load_ts = self.grid.get_load_ts()
        data = {
            f"{sys_id}_load_p_load": load_ts.loc[self.time_index, sys_id],
        }
        if sys_id in self.grid.get_pv_ts():
            data[f"{sys_id}_pv_p_ac"] = self.grid.get_pv_ts().loc[self.time_index, sys_id]

        if sys_id in self.grid.heat_pumps.index and self._model_hps:
            for col_name in self.weather_data:
                data[col_name] = self.weather_data.loc[self.time_index, col_name]

        ret = pd.DataFrame(index=self.time_index, data=data)
        return ret

    def get_parameters(self, sys_id: str) -> type_defs.SysPars:        
        assert sys_id in self.grid.sys_ids, f"Requested sys_id '{sys_id}' not in available Ids"

        if sys_id in self.grid.pv_p:
            # Use PV size for sizing of battery though

            # TODO make the MWH <-> KWH scaling consistent
            bat_size = self.grid.generators.loc[sys_id, "p_set"]
            # 1/2 C inverter power
            bat_p_inv = bat_size / 2.
            return type_defs.SysParsPVBat(sys_id, c_sup=0.3, c_feed=0.1, eff=0.9,
                                          capacity=bat_size, init_soc=0., dt_h=0.25, p_inv=bat_p_inv)
        elif sys_id in self.grid.heat_pumps.index and self._model_hps:
            # HP initialization can only include on-off heatpumps since information from synthetic profiles is limited
            heat_pump_size = self.grid.heat_pumps.loc[sys_id, "p_set"]
            initial_temp = type_defs.SysParsHeatPump.temp_min_heat + self.grid.heat_pumps.index.get_loc(sys_id) / len(
                self.grid.heat_pumps.index
            ) * (type_defs.SysParsHeatPump.temp_max_heat - type_defs.SysParsHeatPump.temp_min_heat)
            return type_defs.SysParsHeatPump(
                sys_id, c_sup=0.3, c_feed=0.1, dt_h=0.25, p_max=heat_pump_size, initial_temp=initial_temp
            )
        else:
            return type_defs.SysParsLoad(sys_id, c_sup=0.3, c_feed=0.1, dt_h=0.25)

        # ev_params = dict(system="ev", eff=0.9, init_soc=0.5, p_lim_dc=11., p_lim_ac=11.)
