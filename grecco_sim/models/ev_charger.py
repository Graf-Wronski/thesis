import numpy as np
from grecco_sim.models import model
from grecco_sim.models import battery
import pandas as pd
from grecco_sim.util import configs


class ChargingProcess(object):
    def __init__(
            self,
            dt_h: float,
            ts_data: pd.DataFrame,
            config: configs.EVConfig):

        self.ts_data = ts_data
        self.config = config
        self.dt_h = dt_h

        self.until_departure = self.ts_data["until_departure"]
        self.capacity = self.config.capacity
        self.target_soc = self.ts_data["target_soc"]
        self.soc = self.ts_data["initial_soc"]
        self.p_lim_ac = self.config.p_lim_ac
        self.p_lim_dc = self.config.p_lim_dc

        self.converter = battery.ConverterModelEff(self.config.eff)

    def apply_control(self, control):
        p_ac, p_dc = self._set_power(control)
        self._evolve(p_dc)
        return p_ac, p_dc

    def _set_power_continuous(self, p_ac_set):
        """
        This is the function copied from the battery system with continuous control space
        :param p_ac_set:
        :return:
        """
        # Correct for AC limit on charge power
        p_ac_set = min(self.p_lim_ac, p_ac_set)
        p_ac_set = max(-self.p_lim_ac, p_ac_set)

        # Get power limits imposed from DC side
        max_dc = (1. - self.soc) * self.capacity / self.dt_h
        max_dc = min(max_dc, self.p_lim_dc)

        min_dc = - self.soc * self.capacity / self.dt_h
        min_dc = max(-self.p_lim_dc, min_dc)

        p_ac_set = min(p_ac_set, self.converter.get_ac_power(max_dc))
        p_ac_set = max(p_ac_set, self.converter.get_ac_power(min_dc))
        # This p_ac set is now a valid AC power

        p_ac = p_ac_set
        p_dc = self.converter.get_dc_power(p_ac_set)

        return p_ac, p_dc

    def _set_power(self, p_ac_set):
        if p_ac_set > 0:
            p_ac_set = self.p_lim_ac

            # Get power limits imposed from DC side
            max_dc = (1. - self.soc) * self.capacity / self.dt_h
            max_dc = min(max_dc, self.p_lim_dc)

            p_ac_set = min(p_ac_set, self.converter.get_ac_power(max_dc))
            # This p_ac set is now a valid AC power

            p_ac = p_ac_set
            p_dc = self.converter.get_dc_power(p_ac_set)
            return p_ac, p_dc
        else:
            return 0., 0.

    def _evolve(self, p_dc):
        self.soc = self.soc + p_dc / self.capacity * self.dt_h
        self.until_departure -= 1

    def get_remaining_time_h(self):
        return self.until_departure * self.dt_h


class EVCharger(model.Model):
    def __init__(
            self,
            sys_id: str,
            horizon: int,
            dt_h: float,
            config: configs.EVConfig,
            ts_data: pd.DataFrame):

        super().__init__(sys_id, horizon, dt_h)

        self.sys_id = sys_id
        self.config = config

        self.soc = np.zeros(self.horizon + 1)
        self.soc.fill(np.NaN)
        self.p_ac_set = np.zeros(self.horizon)
        self.p_ac = np.zeros(self.horizon)
        self.p_dc = np.zeros(self.horizon)

        self.idx_current_cp = 0
        self.cp_indices = []
        self.active_cp = None

        self.initialize_charging_processes(ts_data, sys_id)

    @property
    def p_model(self) -> np.ndarray:
        return self.p_ac

    def initialize_charging_processes(self, ts_in, sys_id):
         #pointer to the current charging process

        #time steps in which a charging process is active
        non_nans = ts_in[[self.sys_id + "_until_departure", self.sys_id + "_initial_soc", self.sys_id + "_target_soc"]].dropna()
        #time steps in which a charging process starts, if any

        if len(non_nans)>0:
            self.cp_indices.append(ts_in.index.get_loc(non_nans.index[0]))
            zero_times = non_nans.index[(non_nans[sys_id + "_until_departure"] == 0) & (non_nans.index != non_nans.index[-1])]
            next_times = zero_times.map(lambda t: non_nans.index[non_nans.index.get_loc(t) + 1] if t in zero_times else None)
            next_indices = next_times.map(lambda t: ts_in.index.get_loc(t) if t is not None else None)
            #list of charging processes: index: time_step in which cp starts, columns: until_departure, initial_soc, target_soc
            self.cp_indices.extend(next_indices)
            self.cps = ts_in.iloc[self.cp_indices].copy()
            #self.cps["target_soc"] = 1
            self.cps.columns = self.cps.columns.str.replace(sys_id +"_", "", regex=True)
            # initialice charging process, if there is one at the beginning
            if (not self.cps.empty) & (self.cp_indices[0] == 0):
                self.active_cp = ChargingProcess(self.dt_h, self.cps.iloc[0], self.config)
                self.soc[0] = self.active_cp.soc

        print(self.sys_id)
        if hasattr(self, "cps"):
            print(self.cps)
        else:
            print("No charging processes for this EV")

    def apply_control(self, control):
        """
        Sign convention is always as a load perspective:
        positive value means charging the battery.

        :param control: intended AC battery power
        :return: no return
        """

        if control["p_ev"] is None:
            raise ValueError(f"{control} misses EV information.")

        if self.active_cp is not None:
            p_ac, p_dc = self.active_cp.apply_control(control)
            soc = self.active_cp.soc

            if self.active_cp.until_departure == 0:
                self.active_cp = None
                self.idx_current_cp += 1
        else:
            p_ac, p_dc = 0., np.NaN
            soc = np.NaN
            #soc = 0

            if not self.idx_current_cp == len(self.cp_indices) and self.cp_indices[self.idx_current_cp] == self.t + 1:
                self.active_cp = ChargingProcess(self.dt_h, self.cps.iloc[
                    self.idx_current_cp], self.config)
                soc = self.active_cp.soc

        self.p_ac_set[self.t] = control
        self.p_ac[self.t] = p_ac
        self.p_dc[self.t] = p_dc

        self.t += 1

        self.soc[self.t] = soc

    def state(self):
        state = {
            "soc": self.soc[self.t],
            "ev_connected": self.active_cp is not None
        }
        if self.active_cp is not None:
            state.update({
                "target_soc": self.active_cp.target_soc,
                "capacity": self.active_cp.capacity,
                "remaining_time_h": self.active_cp.get_remaining_time_h()
            })
        return state

    def state_history(self):
        return {
            "soc": self.soc,
            "p_ac": self.p_ac,
            "p_dc": self.p_dc
        }
    