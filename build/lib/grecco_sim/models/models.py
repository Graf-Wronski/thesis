import numpy as np
import pandas as pd

from grecco_sim.models import battery, ev_charger, heat_pump
from grecco_sim.models import model_base
from grecco_sim.util import type_defs


class PV(model_base.Model):
    """Class for plain PV system."""
    def __init__(
        self,
        sys_id: str,
        horizon: int,
        dt_h: float,
        params: type_defs.SysPars,
        ts_in: pd.DataFrame,
    ):
        """
        Initialize system.

        Pass input time series with column named '{sys_id}_p_ac'
        """
        super().__init__(sys_id, horizon, dt_h)

        assert len(ts_in) == self.horizon, f"Length of given time series ({len(ts_in)})" \
                                           f" does not match horizon length ({self.horizon})"
        self.pvs = ts_in[f"{sys_id}_p_ac"].values

    def apply_control(self, control):
        self.k += 1

    def get_state(self):
        return {"pv_generation": self.pvs[self.k]}

    def get_output(self):
        return {
            "p_ac": self.pvs
        }


class Load(model_base.Model):
    def __init__(self, sys_id: str, horizon: int, dt_h: float, params: type_defs.SysPars, ts_in: pd.DataFrame):
        super().__init__(sys_id, horizon, dt_h)

        assert len(ts_in) == self.horizon, f"Length of given time series ({len(ts_in)})" \
                                           f" does not match horizon length ({self.horizon})"
        self.load = ts_in[f"{sys_id}_p_load"].values
        assert (self.load >= 0.).all(), "Sign convention is load >! 0"

    def apply_control(self, control):
        self.k += 1

    def get_state(self):
        return {"load_power": self.load[self.k]}

    def get_output(self):
        return {
            "p_load": self.load
        }


class Household(model_base.Model):
    """
    A household is a grid connection point!

    That means, the household is the point the EMS can control.

    """

    def __init__(self, sys_id: str, horizon: int, dt_h: float, params: type_defs.SysPars, ts_in: pd.DataFrame):
        super().__init__(sys_id, horizon, dt_h)
        # print(params)

        match params.system:
            case "load":
                self.subsystems = {
                    "load": Load(f"{sys_id}_load", horizon, dt_h, params, ts_in)
                }
            case "pv":
                self.subsystems = {
                    "load": Load(f"{sys_id}_load", horizon, dt_h, params, ts_in),
                    "pv": PV(f"{sys_id}_pv", horizon, dt_h, params, ts_in),
                }
            case "pv_bat":
                self.subsystems = {
                    "load": Load(f"{sys_id}_load", horizon, dt_h, params, ts_in),
                    "pv": PV(f"{sys_id}_pv", horizon, dt_h, params, ts_in),
                    "bat": battery.Storage(f"{sys_id}_bat", horizon, dt_h, params),
                }
            case "ev":
                self.subsystems = {
                    "load": Load(f"{sys_id}_load", horizon, dt_h, params, ts_in),
                    "pv": PV(f"{sys_id}_pv", horizon, dt_h, params, ts_in),
                    "ev": ev_charger.EVCharger(
                        f"{sys_id}_ev", horizon, dt_h, params, ts_in
                    ),
                }
            case "heatpump":
                self.subsystems = {
                    "load": Load(f"{sys_id}_load", horizon, dt_h, params, ts_in),
                    "hp": heat_pump.ThermalSystem(
                        f"{sys_id}_hp", horizon, dt_h, params, ts_in
                    ),
                }
                if f"{sys_id}_pv" in ts_in:
                    self.subsystems[f"{sys_id}_pv_p_ac"] = PV(f"{sys_id}_pv", horizon, dt_h, params, ts_in),

        self.c_sup = params.c_sup
        self.c_feed = params.c_feed

    def apply_control(self, control):
        for sub in self.subsystems.values():
            sub.apply_control(control)
        self.k += 1

    def get_state(self):
        ret = {"k": self.k}
        for sub in self.subsystems.values():
            ret.update(sub.get_state())
        return ret

    def get_grid_power_at(self, k:int):
        grid = 0.
        grid += self.subsystems["load"].load[k]
        if "pv" in self.subsystems:
            grid -= self.subsystems["pv"].pvs[k]
        if "bat" in self.subsystems:
            grid += self.subsystems["bat"].p_ac[k]
        elif "ev" in self.subsystems:
            grid += self.subsystems["ev"].p_ac[k]

        return grid

    def get_output(self):

        subs = {
            sub: self.subsystems[sub].get_output() for sub in self.subsystems
        }

        grid = np.zeros(self.horizon)

        grid += subs["load"]["p_load"]

        if "pv" in self.subsystems:
            grid -= subs["pv"]["p_ac"]

        if "bat" in subs:
            grid += subs["bat"]["p_ac"]

        elif "ev" in subs:
            grid += subs["ev"]["p_ac"]

        if "hp" in subs:
            grid += subs["hp"]["p_in"]

        res = {
            "grid": grid,
            "c_supp": np.ones(grid.shape) * self.c_sup,
            "c_feed": np.ones(grid.shape) * self.c_feed,
            **subs
        }

        return res
