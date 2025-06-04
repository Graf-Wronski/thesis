import datetime
import numpy as np
from grecco_sim.models import model_base
from grecco_sim.models import battery


class ChargingProcess(object):
    def __init__(self, dt_h: float, params_cp: dict, params_charger: dict):
        self.params_cp = params_cp
        self.dt_h = dt_h

        self._init_model(params_cp, params_charger)

    def _init_model(self, params_cp, params_charger):

        self.until_departure = np.ceil(params_cp["parking_time_h"] / self.dt_h)
        self.capacity = params_cp["capacity"]
        self.target_soc = params_cp["target_soc"]

        self.soc = params_cp["init_soc"]

        assert params_charger["p_lim_ac"] > 0. and params_charger["p_lim_dc"] > 0
        self.p_lim_ac = params_charger["p_lim_ac"]
        self.p_lim_dc = params_charger["p_lim_dc"]

        self.converter = battery.ConverterModelEff(params_charger["eff"])

    def apply_control(self, control):
        p_ac, p_net = self._set_power(control)
        self._evolve(p_net)
        return p_ac, p_net

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
        p_net = self.converter.get_dc_power(p_ac_set)

        return p_ac, p_net

    def _set_power(self, p_ac_set):
        if p_ac_set > 0:
            p_ac_set = self.p_lim_ac

            # Get power limits imposed from DC side
            max_dc = (1. - self.soc) * self.capacity / self.dt_h
            max_dc = min(max_dc, self.p_lim_dc)

            p_ac_set = min(p_ac_set, self.converter.get_ac_power(max_dc))
            # This p_ac set is now a valid AC power

            p_ac = p_ac_set
            p_net = self.converter.get_dc_power(p_ac_set)
            return p_ac, p_net
        else:
            return 0., 0.

    def _evolve(self, p_net):
        self.soc = self.soc + p_net / self.capacity * self.dt_h
        self.until_departure -= 1

    def get_remaining_time_h(self):
        return self.until_departure * self.dt_h


class EVCharger(model_base.Model):
    def __init__(self, sys_id: str, horizon: int, dt_h: float, params: dict, ts_in):

        super().__init__(sys_id, horizon, dt_h)

        self.params_charger = params

        self.soc = np.zeros(self.horizon + 1)
        self.p_ac_set = np.zeros(self.horizon)
        self.p_ac = np.zeros(self.horizon)
        self.p_net = np.zeros(self.horizon)

        self.active_cp = None

        self._init_cps(ts_in)

    def _init_cps(self, ts_in):
        self.idx_current_cp = 0

        non_nans = ts_in[f"{self.sys_id}_cp"].dropna()
        self.cps = non_nans.values
        self.cp_indices = [ts_in.index.get_loc(cp_start) for cp_start in non_nans.index]

        # msg = f"Initializing EV {self.sys_id}: \n"
        # for idx in non_nans.index:
        #     msg += f"{idx}: {non_nans[idx]}\n"
        # print(msg)

        for i in range(len(non_nans) - 1):
            end_time = non_nans.index[i] + datetime.timedelta(hours=self.cps[i]["parking_time_h"])
            assert end_time < non_nans.index[i+1], f"Departure time ({end_time}) of Charge process {i} " \
                                                   f"is after following arrival time ({non_nans.index[i+1]})"

        if self.cp_indices[0] == 0:
            self.active_cp = ChargingProcess(self.dt_h, self.cps[0], self.params_charger)
            self.soc[0] = self.active_cp.soc

    def apply_control(self, control):
        """
        Sign convention is always as a load perspective:
        positive value means charging the battery.

        :param control: intended AC battery power
        :return: no return
        """
        if self.active_cp is not None:
            p_ac, p_net = self.active_cp.apply_control(control)
            soc = self.active_cp.soc

            if self.active_cp.until_departure == 0:
                self.active_cp = None
                self.idx_current_cp += 1
        else:
            p_ac, p_net = 0., np.NaN
            soc = np.NaN

            if not self.idx_current_cp == len(self.cp_indices) and self.cp_indices[self.idx_current_cp] == self.k + 1:
                self.active_cp = ChargingProcess(self.dt_h, self.cps[self.idx_current_cp], self.params_charger)
                soc = self.active_cp.soc

        self.p_ac_set[self.k] = control
        self.p_ac[self.k] = p_ac
        self.p_net[self.k] = p_net

        self.k += 1

        self.soc[self.k] = soc

    def get_state(self):
        state = {
            "soc": self.soc[self.k],
            "ev_connected": self.active_cp is not None
        }
        if self.active_cp is not None:
            state.update({
                "target_soc": self.active_cp.target_soc,
                "capacity": self.active_cp.capacity,
                "remaining_time_h": self.active_cp.get_remaining_time_h()
            })
        # print(f"{self.sys_id}: {state}")
        return state

    def get_output(self):
        return {
            "soc": self.soc,
            "p_ac": self.p_ac,
            "p_net": self.p_net
        }


