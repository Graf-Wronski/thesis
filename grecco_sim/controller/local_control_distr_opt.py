from typing import Dict, Optional, Union
import numpy as np

from grecco_sim.util import signals
from grecco_sim.sim_data_tools import forecast_provider
from grecco_sim.util import type_defs
from grecco_sim.local_problems import local_admm, local_gradient_descent

from grecco_sim.controller import local_control


class LocalControllerFirstOrder(local_control.LocalControllerBase):
    """_summary_

    :param local_control: _description_
    :type local_control: _type_
    """
    def __init__(
            self,
            sys_id,
            forecast_access: forecast_provider.ForecastProvider,
            controller_pars: type_defs.OptParameters,
            model_params: type_defs.SysParsPVBat
        ):

        self.fc_access = forecast_access
        self.dt_h = 0.25
        self.sys_id = sys_id

        self.system_sim_pars = model_params
        self.opt_pars = controller_pars

    def get_schedule(
            self,
            signal: signals.Signal,
            state
        ) -> float:
        
        assert isinstance(signal, signals.FirstOrderSignal), "First order signal needed for this controller."

        fc = self.fc_access.get_fc(state["t"], signals.Signal_len)
        future = self.get_flex_schedule(fc, state, signal)
        return future.u[0]

    def get_flex_schedule(
        self,
        forecast: type_defs.Forecast,
        state,
        signal: signals.Signal,
        previous_signals: Optional[dict[str, signals.Signal]] = None):

        # ToDo What is the benefit of having a NoneSignal if it later has to
        #   be converted to another SignalType anyways?
        if isinstance(signal, signal.NoneSignal):
            signal = signals.FirstOrderSignal(np.zeros(forecast.fc_len))
        elif isinstance(signal, signals.FirstOrderSignal):
            # Signal type is already correct.
            pass
        else:
            raise ValueError("First order signal needed for this controller.")
        
        solver = local_gradient_descent.get_solver(forecast.fc_len, self.sys_id, self.system_sim_pars, self.opt_pars)
        solver.solve(state, forecast, signal)
        # if solver.get_u()[0] != 0.:
        #     # for debugging HP controller
        #     u_vec = solver.get_u()
        #     print(solver.get_u())

        return type_defs.Schedule(
            u = solver.get_u(),
            yg= solver.get_yg(),
        )


class LocalControllerSecondOrder(local_control.LocalControllerBase):
    """Class providing a controller for a local agent to take part in an ADMM scheme.
    """
    def __init__(
            self, sys_id, forecast_access,
            controller_pars: type_defs.OptParameters,
            system_sim_pars: type_defs.SysParsPVBat
        ):

        super().__init__()
        self.fc_access = forecast_access
        self.sys_id= sys_id

        self.system_sim_pars = system_sim_pars
        self.controller_pars = controller_pars

    def get_schedule(
            self,
            signal: signals.Signal,
            state
        ):

        assert isinstance(signal, signal.SecondOrderSignal), "Please pass a second order signal!"

        fc_len = len(signal.res_power_set)

        fc = self.fc_access.get_sim_forecast(state["t"], fc_len)
        future = self.get_flex_schedule(fc, state, signal)
        return future.u[0]

    def get_flex_schedule(
        self,
        forecast,
        state,
        signal: signal.SecondOrderSignal,
        previous_signals: Optional[Dict[int, signal.SecondOrderSignal]] = None,
    ):

        if isinstance(signal, signal.NoneSignal):
            signal = signal.SecondOrderSignal(np.zeros(forecast.fc_len), forecast.residual_load)
        elif isinstance(signal, signal.SecondOrderSignal):
            # Signal type is already correct.
            pass
        else:
            raise ValueError("Second order or None signal needed for this controller.")
        
        if previous_signals is None:
            previous_signals = {}

        time_index = state["t"]
        cropped_signals = [
            signal.SecondOrderSignal(signal.res_power_set[time_index-k:], signal.mul_lambda[time_index-k:])
            for k, signal in previous_signals.items()
            if k + signals.Signal_len > time_index
        ]

        signal_lengths = [signals.Signal_len for signal in cropped_signals]

        solver = local_admm.get_solver(signals.Signal_len, signal_lengths, self.sys_id, self.system_sim_pars, self.controller_pars)
        solver.solve(state, forecast, signal, cropped_signals)

        return type_defs.Schedule(
            u = solver.get_u(),
            yg= solver.get_yg(),
            # Returning the gradient is only necessary when using in a real second order
            # context (ALADIN). Can be set to arbitrary value in plain ADMM.
            grads=solver.get_local_gradients(),
            jacobian=solver.get_active_constraint_jacobian(),
            flex_type="continuous"
        )




if __name__ == '__main__':
    pass

