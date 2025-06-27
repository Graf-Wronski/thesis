from typing import Optional

import pandas as pd

from grecco_sim.models import grid
from grecco_sim.util import configs, data_io

class Dataloader:

    def __init__(self, simulation_config: configs.SimulationConfiguration):

        self.simulation_config = simulation_config
        self.dt_h = self.simulation_config.dt_h
        self.time_index = self.simulation_config.time_index

        weather_data = pd.read_csv(
            self.simulation_config.weather_data_path,
            index_col=0,
            date_format="%Y-%m-%d %H:%M:%S")
        self.weather_data = data_io.set_tz_index_to_utc(weather_data)

        self.grid = grid.Grid(simulation_config)

    def get_sys_ids(self):
        return self.grid.sys_ids

    @property
    def sys_ids(self):
        return self.grid.sys_ids

    def get_input_data(self, sys_id: str) -> pd.DataFrame:
        """ DataFrame that holds all timeseries relevant to a system."""

        data = dict()
        unit_data = self.grid.get_system_ts_dict(sys_id=sys_id)

        data.update(unit_data)

        for col in self.weather_data:
            data[col] = self.weather_data.loc[self.time_index, col]

        return pd.DataFrame(data)

    def get_ems_config(self, sys_id: str) -> configs.EMSConfiguration:

        """ Returns modelling parameters. """

        if sys_id not in self.grid.sys_ids:
            raise ValueError(f"Requested sys_id '{sys_id}' not in grid.")

        baseload_config = self.get_baseload_config(sys_id)
        pv_config = self.get_pv_config(sys_id)
        bat_config = self.get_bat_config(sys_id)
        hp_config = self.get_hp_config(sys_id)
        ev_config = self.get_ev_config(sys_id)

        return configs.EMSConfiguration(
            sys_id=sys_id,
            horizon=self.simulation_config.n_time_steps,
            dt_h=self.simulation_config.dt_h,
            market=self.simulation_config.market_config,
            baseload=baseload_config,
            pv=pv_config,
            bat=bat_config,
            hp=hp_config,
            ev=ev_config)

    def get_baseload_config(self, sys_id) -> configs.BaseloadConfig:
        """ We assume baseload at every system."""
        return configs.BaseloadConfig(
            name=sys_id,
            market_config=self.simulation_config.market_config,
            dt_h=self.simulation_config.dt_h,)

    def get_pv_config(self, sys_id) -> Optional[configs.PVConfig]:
        """ If system is associated with pv data, return config. """

        if not "pv" in self.grid.units_at[sys_id]:
            return None

        return configs.PVConfig(
            name=sys_id,
            market_config=self.simulation_config.market_config,
            dt_h=0.25)

    def get_hp_config(self, sys_id) -> Optional[configs.HeatPumpConfig]:
        """ If system is associated with heat pump data, build config. """

        if not "hp" in self.grid.units_at[sys_id]:
            return None

        heat_pump_size = self.grid.hp_params.loc[f"{sys_id}_hp", "p_set"]

        return configs.HeatPumpConfig(
            name=sys_id,
            heat_pump_model=self.simulation_config.heat_pump_model,
            market_config=self.simulation_config.market_config,
            dt_h=self.simulation_config.dt_h,
            p_max=heat_pump_size)

    def get_bat_config(self, sys_id) -> Optional[configs.StorageConfig]:
        """ If system is associated with battery data, build config. """

        if not "bat" in self.grid.units_at[sys_id]:
            return None

        p_nom = self.grid.bat_params.loc[f"{sys_id}_bat", "p_nom"]

        # ToDo: (ReDo) make battery sizes flexible.
        return configs.StorageConfig(
            name=sys_id,
            market_config=self.simulation_config.market_config,
            dt_h=self.dt_h,
            capacity=5.,
            init_soc=0.1,
            p_inv=p_nom)

    def get_ev_config(self, sys_id) -> Optional[configs.EVConfig]:
        """ If system is associated with ev data, build config. """

        if not "ev" in self.grid.units_at[sys_id]:
            return None

        p_nom = self.grid.ev_params.loc[f"{sys_id}_bat", "p_nom"]
        capacity = self.grid.ev_params.loc[f"{sys_id}_bat", "capacity"]

        return configs.EVConfig(
            name=sys_id,
            market_config=self.simulation_config.market_config ,
            dt_h=self.simulation_config.dt_h,
            capacity=capacity,
            init_soc=0.5,
            target_soc=1.,
            p_inv=p_nom)
