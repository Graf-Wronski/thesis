from typing import Optional, Any

import numpy as np
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
            date_format="%Y-%m-%d %H:%M:%S",
        )
        self.weather_data = weather_data

        self.grid = grid.Grid(simulation_config)

    def get_sys_ids(self):
        return self.grid.sys_ids

    @property
    def sys_ids(self):
        return self.grid.sys_ids

    def get_input_data(self, sys_id: str) -> pd.DataFrame:
        """DataFrame that holds all timeseries relevant to a system."""

        data = dict()
        unit_data = self.grid.get_system_ts_dict(sys_id=sys_id)

        data.update(unit_data)

        for col in self.weather_data:
            data[col] = self.weather_data.loc[self.time_index, col]

        return pd.DataFrame(data)

    def get_ems_config(self, sys_id: str) -> configs.EMSConfiguration:
        """Returns modelling parameters."""

        if sys_id not in self.grid.sys_ids:
            raise ValueError(f"Requested sys_id '{sys_id}' not in grid.")

        baseload_config = self.get_baseload_config(sys_id)
        pv_config = self.get_pv_config(sys_id)
        bat_config = self.get_bat_config(sys_id)
        hp_config = self.get_hp_config(sys_id)
        charger_config, ev_requests = self.get_ev_config(sys_id)

        return configs.EMSConfiguration(
            sys_id=sys_id,
            horizon=self.simulation_config.n_time_steps,
            dt_h=self.simulation_config.dt_h,
            market=self.simulation_config.market_config,
            baseload=baseload_config,
            pv=pv_config,
            bat=bat_config,
            hp=hp_config,
            ev_charger=charger_config,
            ev_requests=ev_requests,
        )

    def get_baseload_config(self, sys_id) -> configs.BaseloadConfig:
        """We assume baseload at every system."""
        return configs.BaseloadConfig(
            name=sys_id,
            market_config=self.simulation_config.market_config,
            dt_h=self.simulation_config.dt_h,
        )

    def get_pv_config(self, sys_id) -> Optional[configs.PVConfig]:
        """If system is associated with pv data, return config."""

        if not "pv" in self.grid.units_at[sys_id]:
            return None

        return configs.PVConfig(
            name=sys_id, market_config=self.simulation_config.market_config, dt_h=0.25
        )

    def get_hp_config(self, sys_id) -> Optional[configs.HeatPumpConfig]:
        """If system is associated with heat pump data, build config."""

        if not "hp" in self.grid.units_at[sys_id]:
            return None

        heat_pump_size = self.grid.hp_params.loc[f"{sys_id}_hp", "p_set"]
        # Choose initial temperature randomly in [19, 23]
        initial_temperature = 21 + 4 * (np.random.rand() - 0.5)

        return configs.HeatPumpConfig(
            name=sys_id,
            initial_temp=initial_temperature,
            heat_pump_model=self.simulation_config.heat_pump_model,
            market_config=self.simulation_config.market_config,
            dt_h=self.simulation_config.dt_h,
            p_max=heat_pump_size,
        )

    def get_bat_config(self, sys_id) -> Optional[configs.StorageConfig]:
        """If system is associated with battery data, build config."""

        if not "bat" in self.grid.units_at[sys_id]:
            return None

        p_nom = self.grid.bat_params.loc[f"{sys_id}_bat", "p_nom"]

        # ToDo: (ReDo) make battery sizes flexible.
        return configs.StorageConfig(
            name=sys_id,
            market_config=self.simulation_config.market_config,
            dt_h=self.dt_h,
            p_inv=p_nom,
        )

    def get_ev_config(
        self, sys_id
    ) -> Optional[tuple[configs.ChargerAndEVConfig, list[configs.ChargingRequest]]]:
        """If system is associated with ev data, build config."""

        if not "ev" in self.grid.units_at[sys_id]:
            return None, None

        charging_requests = []
        charger_id = self.grid.ev_params.loc[f"{sys_id}_ev", "charger_id"]
        ts_data = self.grid.requests.query("ChargerID == @charger_id")
        for _, row in ts_data.iterrows():
            capacity = row["TargetSoc"] - row["fictive_soc_start"]
            time_index = [x.tz_localize(None) for x in self.time_index]

            req = configs.ChargingRequest(
                start_step=time_index.index(row["relative_start"]),
                end_step=time_index.index(row["relative_end"]),
                capacity=capacity,
            )
            charging_requests.append(req)

        p_nom = self.grid.ev_params.loc[f"{sys_id}_ev", "p_nom"]
        capacity = self.grid.ev_params.loc[f"{sys_id}_ev", "capacity"]

        charger_config = configs.ChargerAndEVConfig(
            name=sys_id,
            ev_name=sys_id,
            market_config=self.simulation_config.market_config,
            dt_h=self.simulation_config.dt_h,
            capacity=capacity,
            p_inv=p_nom,
        )

        return charger_config, charging_requests
