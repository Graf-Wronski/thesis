from grecco_sim.util import sig_types
from grecco_sim.models import models
from grecco_sim.sim_data_tools import forecast_provider
from grecco_sim import coordinators
from grecco_sim.controller import local_control
from grecco_sim.util import type_defs


class GridNode(object):
    """
    This is basically a local EMS on top of a household.

    I.e. a household providing an interface to the central coordinator.

    """

    def __init__(
        self, sys_id: str, run_pars: type_defs.RunParameters, model_input: dict, opt_params: type_defs.OptParameters
    ):
        # horizon: int, dt_h: float, params: dict, ts_in: pd.DataFrame):

        self.sys_id = sys_id
        self.model_input = model_input

        # Use this container to store the signals obtained in the horizon
        self.signals = {}

        self.phys_model = models.Household(
            sys_id, run_pars.sim_horizon, run_pars.dt_h, model_input["params"], model_input["ts"]
        )
        if opt_params.fc_type == "perfect":
            self.fc_access = forecast_provider.PerfectForesightForecast(self.phys_model)
        else:
            self.fc_access = forecast_provider.ForecastProviderNaive(self.phys_model, 96)
        self.run_pars = run_pars

        self._init_local_controller(opt_params, model_input["params"], run_pars)

    def _init_local_controller(self, opt_params, model_params, run_pars: type_defs.RunParameters):

        coordination_mechanism = run_pars.coordination_mechanism

        if coordination_mechanism == "central":
            self.local_controller = local_control.LocalControllerPassControl()
        else:
            for cm in coordinators.AVAILABLE_COORDINATORS:
                if cm.name == coordination_mechanism:
                    local_controller_class = cm.controller_local[model_params.system]
                    self.local_controller = local_controller_class(
                        self.sys_id, self.fc_access, opt_params, model_params
                    )
                    return
            raise ValueError(
                f"Coordination mechanism '{coordination_mechanism}' not in implemented mechanisms"
                f" {[cm.name for cm in coordinators.AVAILABLE_COORDINATORS]}"
            )

    def get_current_future(self, time_index, horizon, signal: sig_types.SignalType) -> type_defs.LocalFuture:
        """
        This interface is under construction.
        It might change in the future as it is part of the coordination mechanism...

        :param time_index: current time index as an index in the shared numpy array time series.
        :param horizon: horizon of the forecast in the future time series
        :param signal: Control signal from central controller

        :return:
        """
        fc = self.fc_access.get_fc(time_index, horizon)

        state = self.phys_model.get_state()
        # Update forecast with current measurement
        fc.fc_res_load[0] = state["load_power"]
        if "pv_generation" in state:
            fc.fc_res_load[0] -= state["pv_generation"]

        extra_arg = {"previous_signals": self.signals} if self.run_pars.use_prev_signals else {}
        future = self.local_controller.get_flex_schedule(fc, state, signal, **extra_arg)
        assert isinstance(future, type_defs.LocalFuture), f"Stick to Typing! for {type(self.local_controller)}"

        future._meta = {"fc": fc, "state": self.get_state(), "_model_pars": self.model_input["params"]}

        return future

    def get_state(self):
        """
        TODO this function can be restricted to only transmit the state parts which are necessary.

        :return:
        """
        return self.phys_model.get_state()

    def get_grid_power_at(self, k: int) -> float:
        return self.phys_model.get_grid_power_at(k)

    def apply_control(self, signal: sig_types.SignalType):

        sys_state = self.phys_model.get_state()
        self.signals[sys_state["k"]] = signal

        control = self.local_controller.get_control(signal, sys_state)

        self.phys_model.apply_control(control)

    def get_output(self):
        sys_output = self.phys_model.get_output()
        sys_output["signals"] = self.signals

        return sys_output
