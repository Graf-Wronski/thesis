import abc
from typing import Any, Dict, Optional
from grecco_sim.util import signals
from grecco_sim.util import type_defs
from grecco_sim.util.type_defs import Schedule


class LocalControllerBase(abc.ABC):
    """Interface for local controllers."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_schedule(self, signal: signals.Signal, state: dict[str, Any]) -> float:
        """Get control at current time for local system's flexibility."""

    def get_flex_schedule(
        self,
        forecast: type_defs.Forecast,
        state: dict[str, Any],
        signal: signals.Signal,
        previous_signals: Optional[Dict[str, signals.Signal]] = None,
    ) -> type_defs.Schedule:
        """Provide a local future for coordination."""
        future = type_defs.Schedule(forecast.residual_load)
        return future


class LocalControllerSelfSuff(LocalControllerBase):
    """Local controller that aims for self-sufficiency by matching generation to load."""

    def __init__(self, *args, model_params):
        super().__init__()

    def get_schedule(self, signal, state):
        return state["pv_generation"] - state["p_baseload"]


class LocalControllerEVBaseline(LocalControllerBase):
    """Local controller implementing baseline EV charging behavior."""

    def __init__(self, *args, model_params):
        super().__init__()
        self.charger_power = model_params.p_lim_ac

    def get_schedule(self, signal, state):
        if state["ev_connected"] and state["soc"] < state["target_soc"]:
            return self.charger_power
        else:
            return 0.0


class LocalControllerHeatPumpOnOff(LocalControllerBase):
    def __init__(self, *args, model_params):
        super().__init__()
        model_params = model_params

        # Define State based on inner temperature TK
        # e.g. cooling when Tk > 26°C
        # Stop cooling when Tk < 22°C
        # heating when Tk < 19°C
        # Stop heating when Tk > 23°
        self.temp_max_heat = model_params.temp_max_heat
        self.temp_min_heat = model_params.temp_min_heat
        self.temp_max_cold = model_params.temp_max_cold
        self.temp_min_cold = model_params.temp_min_cold

    def get_schedule(self, signal, state):
        # state should be cool, heat, off (-1, 1, 0)
        # defined in state['mode']
        # signal should be temperature
        temp_k = state["temp"]
        if state["mode"] == 1:
            if temp_k > self.temp_max_heat:
                return 0
            else:
                return 1
        elif state["mode"] == -1:
            if temp_k < self.temp_min_cold:
                return 0
            else:
                return -1
        elif state["mode"] == 0:
            if temp_k < self.temp_min_heat:
                return 1
            elif temp_k > self.temp_max_cold:
                return -1
            else:
                return 0
        else:
            raise ValueError(f"Mode '{state['mode']}' invalid")


class LocalControllerNoBat(LocalControllerBase):
    """Local controller for systems without battery storage."""
    
    def __init__(self, *args, **kwargs):
        pass

    def get_schedule(self, signal, state, forecast):
        # ToDo for after thesis: Why is residual load a series and not an
        #  array?
        return Schedule(p_grid=forecast.residual_load.values)

    def step(self):
        pass


class LocalControllerPassControl(LocalControllerBase):
    def __init__(self, *args):
        super().__init__()

    def get_schedule(self, signal: signals.DirectControlSignal, state):
        return signal.control[0] 

    def get_flex_schedule(
        self,
        forecast: type_defs.Forecast,
        state: dict[str, Any],
        signal: signals.Signal,
        previous_signals: Optional[Dict[str, signals.Signal]] = None,
    ) -> type_defs.Schedule:
        future = type_defs.Schedule(forecast.residual_load, flex_type="continuous")
        return future
