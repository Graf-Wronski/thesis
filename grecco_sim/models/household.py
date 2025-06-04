import numpy as np
import pandas as pd

from grecco_sim.models import battery, ev_charger, heat_pump, inflexible
from grecco_sim.models import model
from grecco_sim.util import configs


class Household(model.Model):
    """
    A household is a grid connection point!

    That means, the household is the point the EMS can control.

    For each household, all unit models are initialized according to the given parameters.
    """

    def __init__(
            self,
            ems_config: configs.EMSConfiguration,
            input_ts: pd.DataFrame):

        model_pars = {"sys_id": ems_config.sys_id,
                      "horizon": ems_config.horizon,
                      "dt_h": ems_config.dt_h,}

        super().__init__(**model_pars)
        self.ts_data = input_ts

        baseload_ts = input_ts[f"{self.sys_id}_baseload_p"]
        config = ems_config.baseload
        self.baseload = inflexible.Baseload(
            config=config,
            ts_data=baseload_ts,
            **model_pars)

        # By default, a household only has baseload.
        self.pv, self.storage, self.hp, self.ev = None, None, None, None

        if ems_config.pv:
            pv_ts = input_ts[f"{self.sys_id}_pv_p"]
            self.pv = inflexible.PV(
                config=ems_config.pv,
                ts_data=pv_ts,
                **model_pars)

        if ems_config.bat:
            self.storage = battery.Storage(
                config=ems_config.bat,
                **model_pars)

        if ems_config.hp:
            hp_ts = input_ts[["Outside Temperature", "Solar Irradiance"]]
            self.hp = heat_pump.ThermalSystem(
                config=ems_config.hp,
                ts_data=hp_ts,
                **model_pars)

        if ems_config.ev:
            # ToDo: Key might be wrong.
            key = self.sys_id
            ev_cols = [f"{key}_cp", f"{key}_initial_soc",
                       f"{key}_target_soc", f"{key}_until_departure"]
            self.ev = ev_charger.EVCharger(
                ts_data=input_ts[ev_cols],
                config=ems_config.ev,
                **model_pars)

        self.market_config = ems_config.market

    def step(self):

        for unit in self.units.values():
            unit.step()

        self.t += 1

    @property
    def irradiance_ts(self) -> pd.Series:
        return self.ts_data["Solar Irradiance"]

    @property
    def temp_out_ts(self) -> pd.Series:
        return self.ts_data["Outside Temperature"]

    @property
    def units(self) -> dict[str, model.Model]:
        """ Return a dict with all available units. """

        units = dict()

        # Add units if available.
        if self.baseload:
            units["baseload"] = self.baseload
        if self.pv:
            units["pv"] = self.pv
        if self.storage:
            units["bat"] = self.storage
        if self.hp:
            units["hp"] = self.hp
        if self.ev:
            units["ev"] = self.ev

        return units

    @property
    def model_type(self) -> str:
        return "household"

    def apply_control(self, control):

        for unit in self.units.values():
            unit.apply_control(control)

    @property
    def p_model(self) -> np.ndarray:
        """ Grid power of household is sum of units. """
        p_units = np.array([self.units[unit].p_model for unit in self.units])
        p_units = np.nan_to_num(p_units, nan=0.)

        return p_units.sum(axis=0)


    @property
    def state_history(self) -> dict[str, np.ndarray]:
        """ unit_history[unit_type][param] = np.ndarray. """

        state_history = dict()
        state_history["household_p_model"] = self.p_model

        # For each unit, add its parameters and indicate unit_name.
        for unit_name, unit in self.units.items():
            for param_name, param_value in unit.state_history.items():
                state_history[f"{unit_name}_{param_name}"] = param_value
                state_history[f"{unit_name}_p_model"] = unit.p_model

        return state_history

    @property
    def p_grid_by_unit(self):
        return {name: unit.p_model for name, unit in self.units.items()}