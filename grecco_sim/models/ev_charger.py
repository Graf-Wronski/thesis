from typing import Optional

import numpy as np

from grecco_sim.models import model
from grecco_sim.util import configs

class EVCharger(model.Model):

    @property
    def model_type(self) -> str:
        return "ev"

    def __init__(
            self,
            sys_id: str,
            horizon: int,
            dt_h: float,
            config: configs.ChargerAndEVConfig,
            requests: list[configs.ChargingRequest]):

        super().__init__(sys_id, horizon, dt_h)

        self.sys_id = sys_id
        self.config = config

        self.p_ac_set = self.build_param()
        self.p_ac = self.build_param()
        self.p_dc = self.build_param()
        self.ev_connected = self.build_param()

        self.requests = requests
        self.remaining_capacity = np.zeros(self.horizon)

    @property
    def active_request(self) -> Optional[configs.ChargingRequest]:
        active_requests = [r for r in self.requests if r.active_at(self.t)]
        if len(active_requests) > 1:
            raise NotImplementedError
        elif len(active_requests) == 0:
            return None
        else:
            return active_requests[0]

    @property
    def p_model(self) -> np.ndarray:
        return self.p_ac

    def apply_control(self, control):
        """ """

        if control["p_ev"] is None:
            raise ValueError(f"{control} misses EV information.")

        if control["p_ev"] < 0.005:
            control["p_ev"] = 0.

        if self.active_request is None and control["p_ev"] != 0.:
            msg = (f"Attempting to charge {control['p_ev']} kw while no "
                   f"request active.")
            print(msg)
            control["p_ev"] = 0

        # Reduce active charging process capacity by effective p_ev.
        if self.active_request:
            remaining_capacity = self.active_request.capacity
        else:
            remaining_capacity = 0.
        max_ac_possible =  (remaining_capacity / self.dt_h) / self.config.eff
        p_ac = min(max_ac_possible, control["p_ev"], self.config.p_lim_ac)
        p_dc = self.config.eff * p_ac

        self.ev_connected[self.t] = float(self.active_request is not None)
        self.p_ac_set[self.t] = control["p_ev"]
        self.remaining_capacity[self.t] = remaining_capacity
        self.p_ac[self.t] = p_ac
        self.p_dc[self.t] = p_dc

    def _evolve(self, p_dc):
        if self.active_request is None:
            return
        else:
           self.active_request.capacity -= p_dc

    @property
    def state_history(self) -> dict[str, np.ndarray]:

        return {
            "p_ac_set": self.p_ac_set,
            "p_ac": self.p_ac,
            "p_dc": self.p_dc,
            "ev_connected": self.ev_connected,
            "remaining_capacity": self.remaining_capacity}
    