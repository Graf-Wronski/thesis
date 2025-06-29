import numpy as np
import pandas as pd

from grecco_sim.controller.multiunit import muc
from grecco_sim.simulator import forecaster
from grecco_sim.util import signals, configs
from grecco_sim.models import household
from grecco_sim.controller import local_control
from grecco_sim.util import type_defs


class SimulationNode:
    def __init__(
            self,
            simulation_config: configs.SimulationConfiguration,
            ems_config: configs.EMSConfiguration,
            timeseries: pd.DataFrame):

        """ GridNodes are passive grid elements that pass information. """

        self.sys_id = ems_config.sys_id
        self.config = simulation_config
        self.timeseries = timeseries
        self.ems_config = ems_config
        self.t = 0

        # Container to store control signals.
        shape = self.config.n_time_steps, self.config.optimizer_config.horizon
        self.signal_matrix = np.full(shape, np.nan)

        # ToDo: Distinction between physical model and Household might be
        #  deprecated
        self.phys_model = household.Household(ems_config, timeseries)

        # Init LocalController
        # ToDO: Make this code part more explicit.
        cm_name = self.config.coordinator_name
        if cm_name == "central":
            # self.controller = local_control.LocalControllerPassControl()
            self.controller = None
        else:
            # If no flexibility is given, use simple controller.
            if ems_config.is_inflexible:
                self.controller = local_control.LocalControllerNoBat()
            else:
                self.controller = muc.MultiUnitController(
                    self.config,
                    self.ems_config,
                    self.timeseries)

    def __str__(self) -> str:
        """ Nodes are identified by their sys_id. """
        return self.sys_id

    def step(self):
        self.t += 1
        self.phys_model.step()

    def get_schedule(
            self,
            forecast: forecaster.NodeForecast,
            coordinator_signal: signals.Signal) -> type_defs.Schedule:

        """ Apply control and get respective schedule states.

        Args:
            forecast: ...
            coordinator_signal: Node's LocalController reacts to signal.

        Returns:
            Future states in reaction to signal. """

        state = self.phys_model.state

        """# Update forecast with current measurement  # ToDo: Necessary?
        forecast.nodes[self.sys_id]
        forecast.residual_load[0] = state["baseload_p"]
        forecast.baseload[0] = state["baseload_p"]

        if "pv_p" in state:
            forecast.residual_load[0] -= state["pv_p"]"""

        # Determine if global (direct) control or local control is applied.
        if isinstance(coordinator_signal, signals.DirectControlSignal):
            schedule = coordinator_signal.schedule
        else:
            schedule = self.controller.get_schedule(
                forecast=forecast,
                state=state,
                signal=coordinator_signal)

        return schedule

    @property
    def state(self) -> dict[str, float]:
        return {"t": self.t, **self.phys_model.state}

    @property
    def p_node(self) -> np.array:
        return self.phys_model.p_model

    def realize_schedule(self, schedule: type_defs.Schedule):
        """ Converts coordination signal to local control and applies it.

        Args:
            schedule: As provided by central coordinator or local controller.
            
        ToDo: It is a little unintuitive to create a control in a method
            called 'apply_control'. """

        if not isinstance(schedule.p_grid, np.ndarray):
            print("")
        self.phys_model.apply_control(schedule.now)

    @property
    def state_history(self) -> dict[str, np.ndarray]:
        """ Pass output of physical model together with respective signals. """
        state_history = self.phys_model.state_history
        state_history["p_node"] = self.p_node
        return state_history

    @property
    def p_grid_by_unit(self) -> dict:
        return self.phys_model.p_grid_by_unit

    @property
    def has_pv(self) -> bool:
        return self.ems_config.pv is not None

    @property
    def has_bat(self) -> bool:
        return self.ems_config.bat is not None

    @property
    def has_hp(self) -> bool:
        return self.ems_config.hp is not None

    @property
    def has_ev(self) -> bool:
        return self.ems_config.ev is not None