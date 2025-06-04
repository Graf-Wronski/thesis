from typing import Any

import numpy as np
from pyproj.database import Unit

from grecco_sim.models import model_base


class LocalEMS(model_base.Model):
    """ An Energy Management System (EMS) controls potentially multiple units.
        Currently, can be thought of as building energy management system."""

    def __init__(
        self,
        system_id: str,
        horizon: int,
        dt_h: float,
        units: dict[list[Unit]],
        params: dict):

        super().__init__(system_id, horizon, dt_h)
        self._units = units
        self.c_sup = params[system_id].c_sup
        self.c_feed = params[system_id].c_feed

    @property
    def inflexible_loads(self) -> list[Load]:
        return self._units['load']#

    @property
    def pv_units(self) -> list[PV]:
        return self._units['pv']

    @property
    def hp_units(self) -> list[ThermalSystem]:
        return self._units['hp']

    @property
    def batteries(self) -> list[Storage]:
        return self._units['bat']

    @property
    def ev_units(self) -> list[EVCharger]:
        return self._units['ev']

    @property
    def units(self) -> list[Unit]:
        units = []
        for unit_type in ['load', 'pv', 'hp', 'bat', 'ev']:
            units.append(self._units[unit_type])

    def apply_control(self, control):
        for unit in self.units:
            unit.apply_control(control)
        self.t += 1

    def get_state(self) -> dict[str, Any]:
        # Base state: information about current timestep.
        state = {"t": self.t}

        # Update state information with unit-specific information.
        for unit in self.units:
            state.update(unit.get_state())

        return state

    # ToDo: Rename method.
    def get_grid_power_at(self, k: int) -> float:
        """ Total power demand by all units. """

        current_power_demand = 0.

        for inflexible_load in self.inflexible_loads:
            current_power_demand += inflexible_load.baseload[k]

        for pv in self.pv_units:
            current_power_demand -= pv.pvs[k]

        for battery in self.batteries:
            current_power_demand += battery.p_ac[k]

        for ev in self.ev_units:
            current_power_demand += ev.p_ac[k]

        for hp in self.hp_units:
            current_power_demand += hp.p[k]

        return current_power_demand

    # ToDo @Ramiz: Rename to get_results?
    def get_output(self):
        """ Compile results for EMS and all controlled units.

        Returns:
            dict:
                "<unit_name>": individual unit output
                "grid": total power demand for each timestep [nd.array]
                "c_supp": ... for each timestep [nd.array]
                "c_feed": ... for each timestep [nd.array]

        # ToDo: Documentation for c_supp and c_feed.
        """

        result = {}
        power_demand_over_time = np.zeros(self.horizon)

        # ToDo @Ramiz: get_output and get_state are not consistent.
        for unit in self._units:
            # ToDo: Make sure get_output is unique for units of same type.
            result[unit] = unit.state_history()

        for unit in self.inflexible_loads:
            power_demand_over_time += result[unit]["p_load"]

        for unit in self.pv_units:
            power_demand_over_time -= result[unit]["p_ac"]

        for unit in self.batteries:
            power_demand_over_time += result[unit]["p_ac"]

        for unit in self.ev_units:
            power_demand_over_time += result[unit]["p_ac"]

        for unit in self.hp_units:
            power_demand_over_time += result[unit]["p_in"]

        result["grid"] = power_demand_over_time
        result["c_supp"] = np.ones(self.horizon) * self.c_sup
        result["c_feed"] = np.ones(self.horizon) * self.c_feed

        return result