from typing import Any

from grecco_sim.util import signals
from grecco_sim.controller.central import cc
from grecco_sim.coordinator import coordinator


class CentralCoordinator(coordinator.Coordinator):
    def __init__(self, grecco_sim: Any):
        super().__init__(grecco_sim)

        self.controler = cc.CentralController(
            opt_config=self.sim_config.optimizer_config,
            ems_configs=self.ems_configs)

    def get_signals(self, state: dict[str, dict]) -> dict[str, signals.Signal]:

        forecast = self.forecaster.get_sim_forecast()
        schedules = self.controler.get_schedules(state, forecast, self.horizon)
        signal_dict = {sys_id: signals.DirectControlSignal(schedule)
                       for sys_id, schedule in schedules.items()}


        return signal_dict
