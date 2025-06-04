import dataclasses
from typing import Optional

import numpy as np
import pandas as pd

from grecco_sim.simulator import dataloader


@dataclasses.dataclass
class NodeForecast:
    baseload: pd.DataFrame
    residual_load: pd.DataFrame

    solar_irradiance: np.ndarray
    temp_outisde: np.ndarray

    pv: Optional[pd.DataFrame] = None

    def __len__(self):
        return len(self.residual_load)

@dataclasses.dataclass
class SimulationForecast:
    nodes: dict[str, NodeForecast]
    solar_irradiance: np.ndarray
    temp_outisde: np.ndarray

class Forecaster:
    def __init__(
            self,
            horizon: int,
            sim_dataloader: dataloader.Dataloader,
            time_index: pd.DatetimeIndex):

        self.index = time_index

        self.sys_ids = sim_dataloader.get_sys_ids()
        timeseries = {sys_id: sim_dataloader.get_input_data(sys_id)
                      for sys_id in self.sys_ids}

        pv_p = dict()
        baseload_p = dict()
        residual_load = dict()

        for sys_id in self.sys_ids:
            data = timeseries[sys_id].copy()
            baseload_p[sys_id] = data[f"{sys_id}_baseload_p"].values
            residual_load[sys_id] = baseload_p[sys_id].copy()

            # If pv is available: update it and update residual load.
            if f"{sys_id}_pv_p" in data.columns:
                pv_p[sys_id] = data[f"{sys_id}_pv_p"].values
                residual_load[sys_id] -= pv_p[sys_id]

        self.baseload_p = pd.DataFrame(baseload_p, index=self.index)
        self.residual_load = pd.DataFrame(residual_load, index=self.index)

        if len(pv_p) == 0:
            self.pv_p = None
        else:
            self.pv_p = pd.DataFrame(pv_p, index=self.index)

        self.solar_irradiance = sim_dataloader.weather_data["Solar Irradiance"]
        self.temp_outside = sim_dataloader.weather_data["Outside Temperature"]

        self.time_slice = self.index[0: horizon]

    def set_time_window(self, t: int, horizon: int):
        self.time_slice = self.index[t: t + horizon]

    def get_node_forecast(self, sys_id: str) -> NodeForecast:
        """ Contains baseload, residual load and pv where required."""


        if not self.pv_p is None and sys_id in self.pv_p.columns:
            pv = self.pv_p.loc[self.time_slice, sys_id]
        else:
            pv = None

        temp_outside = self.temp_outside[self.time_slice]
        solar_irradiance = self.solar_irradiance[self.time_slice]

        return NodeForecast(
                baseload=self.baseload_p.loc[self.time_slice, sys_id],
                residual_load=self.residual_load.loc[self.time_slice, sys_id],
                temp_outisde=temp_outside,
                solar_irradiance=solar_irradiance,
                pv=pv)

    def get_sim_forecast(self) -> SimulationForecast:
        """ Contains wheather data and node forecasts for each node. """

        temp_outside = self.temp_outside[self.time_slice]
        solar_irradiance = self.solar_irradiance[self.time_slice]

        load_forecast = {sys_id: self.get_node_forecast(sys_id)
                          for sys_id in self.sys_ids}

        sim_forecast = SimulationForecast(
            nodes=load_forecast,
            temp_outisde=temp_outside,
            solar_irradiance=solar_irradiance)

        return sim_forecast
