import numpy as np
from grecco_sim.util import configs
from grecco_sim.models import model


class ConverterModelEff(object):
    """
    The gaol with this model is that the two provided functions are EXACTLY! inverse of each other.

    The equation is always:
    DC = AC - Loss (AC)

    Hence, Loss > 0 always. DC and AC > 0 is equivalent to charging the battery.

    """
    def __init__(self, eff):
        if not (0 < eff <= 1):
            raise ValueError("Charging efficiency must be between 0 and 1.")
        self.eff = eff

    def get_ac_power(self, p_dc):
        # No closed form to invert the absolute
        if p_dc < 0:
            return p_dc / (2-self.eff)
        else:
            return p_dc / self.eff

    def get_dc_power(self, p_ac):
        return p_ac - abs(p_ac) * (1 - self.eff)


class Storage(model.Model):
    def __init__(
            self,
            sys_id: str,
            horizon: int,
            dt_h: float,
            config: configs.StorageConfig):

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
        :param config: parameters to govern storage behavior in simulation
        """
        super().__init__(sys_id, horizon, dt_h)

        self.config = config

        self.soc = self.build_param(initial_value=config.init_soc)
        self.p_ac_set = self.build_param()
        self.p_ac = self.build_param(initial_value=0.)
        self.p_dc = self.build_param(initial_value=0.)

        self.capacity = config.capacity
        self.p_lim_ac = config.p_inv
        self.p_lim_dc = config.p_lim_dc
        self.eff = config.eff

        self.converter = ConverterModelEff(config.eff)

    @property
    def model_type(self) -> str:
        return "bat"

    @property
    def p_model(self) -> np.ndarray:
        return self.p_ac

    def apply_control(self, control):
        """ React to control signal and evolve battery state.

        Args:
            control: Positive control signals charging. Negative signal
                signals discharging.
        """

        if control["p_bat"] is None:
            msg = (f"Invalid control signal. {control} has no battery "
                   f"information.")
            raise ValueError(msg)

        if np.isnan(control["p_bat"]):
            msg = f"Control signal for battery is nan. Replacing with 0"
            print(msg)
            control["p_bat"] = 0.

        battery_control = control["p_bat"]

        self._set_power(battery_control)
        self._evolve()

    def _set_power(self, p_ac_set):

        self.p_ac_set[self.t] = p_ac_set

        """ # Correct for AC limit on charge power
        p_ac_set = min(self.p_lim_ac, p_ac_set)
        p_ac_set = max(-self.p_lim_ac, p_ac_set)

        # Get power limits imposed from DC side
        max_dc = (1. - self.soc[self.k-1]) * self.capacity / self.dt_h
        max_dc = min(max_dc, self.p_lim_dc)

        min_dc = - self.soc[self.k-1] * self.capacity / self.dt_h
        min_dc = max(-self.p_lim_dc, min_dc)

        p_ac_set = min(p_ac_set, self.converter.get_ac_power(max_dc))
        p_ac_set = max(p_ac_set, self.converter.get_ac_power(min_dc))
        # This p_ac set is now a valid AC power"""

        self.p_ac[self.t] = p_ac_set

        # Analogous to first-order optimization: regard charging efficiency.
        # ToDo: This is different from battery model above.

        """    if p_ac_set >= 0.:
            p_net = p_ac_set * self.eff
        else:
            p_net = """

        # Since the controller already regards losses for discharge, we only
        # have to regard losses for charges. ToDo: This is not pretty.
        self.p_dc[self.t] = p_ac_set * self.eff


    def _evolve(self):
        soc_gain = (self.p_dc[self.t] / self.capacity) * self.dt_h
        self.soc[self.t + 1] = self.soc[self.t] + soc_gain

    @property
    def state_history(self) -> dict[str, np.ndarray]:
        state_history = dict()
        state_history["soc"] = self.soc
        state_history["p_ac"] = self.p_ac
        state_history["p_dc"] = self.p_dc

        return state_history