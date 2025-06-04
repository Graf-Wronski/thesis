import numpy as np
from grecco_sim.util import type_defs
from grecco_sim.models import model_base


class ConverterModelEff(object):
    """
    The gaol with this model is that the two provided functions are EXACTLY! inverse of each other.

    The equation is always:
    DC = AC - Loss (AC)

    Hence, Loss > 0 always. DC and AC > 0 is equivalent to charging the battery.

    """
    def __init__(self, eff):
        assert 0 < eff <= 1
        self.eff = eff

    def get_ac_power(self, p_dc):
        # No closed form to invert the absolute
        if p_dc < 0:
            return p_dc / (2-self.eff)
        else:
            return p_dc / self.eff

    def get_dc_power(self, p_ac):
        return p_ac - abs(p_ac) * (1 - self.eff)


class Storage(model_base.Model):
    def __init__(self, sys_id: str, horizon: int, dt_h: float, params: type_defs.SysParsPVBat):
        """
        Necessary parameters in params dictionary

        +----------------+-------+----------+------+----------------------------------------+
        | Parameter name |  Type | Range    | Unit | Meaning                                |
        +================+=======+==========+======+========================================+
        | soc_init       | float | [0, 1]   |      | Initial storage SoC                    |
        +----------------+-------+----------+------+----------------------------------------+
        | eff            | float | [0, 1]   |      | Efficiency of charging and discharging |
        +----------------+-------+----------+------+----------------------------------------+
        | p_lim_ac       | float | (0, inf) | kW   | Maximum AC power of converter          |
        +----------------+-------+----------+------+----------------------------------------+
        | p_lim_dc       | float | (0, inf) | kW   | Maximum DC power of converter          |
        +----------------+-------+----------+------+----------------------------------------+
        | capacity       | float | [0, 1]   | kWh  | Capacity of the storage                |
        +----------------+-------+----------+------+----------------------------------------+


        :param sys_id: unique ID of the system
        :param horizon: horizon of simulation
        :param params: parameters to govern storage behavior in simulation
        """
        super().__init__(sys_id, horizon, dt_h)

        self.soc = np.zeros(self.horizon + 1)
        self.p_ac_set = np.zeros(self.horizon)
        self.p_ac = np.zeros(self.horizon)
        self.p_net = np.zeros(self.horizon)

        self._init_model(params)

    def _init_model(self, params):

        self.soc[0] = params.init_soc
        self.capacity = params.capacity

        assert params.p_inv > 0. and params.p_lim_dc > 0
        self.p_lim_ac = params.p_inv
        self.p_lim_dc = params.p_lim_dc

        self.converter = ConverterModelEff(params.eff)

    def apply_control(self, control):
        """
        Sign convention is always as a load perspective:
        positive value means charging the battery.

        :param control: intended AC battery power
        :return: no return
        """
        self.k += 1

        self._set_power(control)
        self._evolve()

    def _set_power(self, p_ac_set):

        self.p_ac_set[self.k-1] = p_ac_set

        # Correct for AC limit on charge power
        p_ac_set = min(self.p_lim_ac, p_ac_set)
        p_ac_set = max(-self.p_lim_ac, p_ac_set)

        # Get power limits imposed from DC side
        max_dc = (1. - self.soc[self.k-1]) * self.capacity / self.dt_h
        max_dc = min(max_dc, self.p_lim_dc)

        min_dc = - self.soc[self.k-1] * self.capacity / self.dt_h
        min_dc = max(-self.p_lim_dc, min_dc)

        p_ac_set = min(p_ac_set, self.converter.get_ac_power(max_dc))
        p_ac_set = max(p_ac_set, self.converter.get_ac_power(min_dc))
        # This p_ac set is now a valid AC power

        self.p_ac[self.k-1] = p_ac_set
        self.p_net[self.k-1] = self.converter.get_dc_power(p_ac_set)

    def _evolve(self):

        self.soc[self.k] = self.soc[self.k-1] + self.p_net[self.k-1] / self.capacity * self.dt_h

    def get_state(self):
        return {
            "soc": self.soc[self.k],
        }

    def get_output(self):
        return {
            "soc": self.soc,
            "p_ac": self.p_ac,
            "p_net": self.p_net
        }

