import time
import pandas as pd

from grecco_sim.simulator import dataloader, result, forecaster
from grecco_sim.util import configs, build
from grecco_sim.models import sim_node


class Simulation:
    def __init__(self, sim_config: configs.SimulationConfiguration):
        """

        Args:
            sim_config: Data sources, simulation settings, optimization
                parameters, etc. See SimulationConfiguration class. """

        self.config = sim_config
        self.optimizer_config = sim_config.optimizer_config

        # Current simulation time step.
        self.t = 0

        self.dataloader = dataloader.Dataloader(self.config)
        self.grid = self.dataloader.grid
        # ToDo: Forecaster currently only used by central coordinator.
        self.forecaster = forecaster.Forecaster(
            horizon=self.opt_horizon,
            sim_dataloader=self.dataloader,
            time_index=self.index)

        # Nodes are controllable grid participants. As of now: private EMS.
        self.node_ids = self.dataloader.get_sys_ids()
        self.nodes = [self._build_simulation_node(node_id)
                      for node_id in self.node_ids]

        self.execution_time = 0.
        self.results = result.SimulationResult(self.config, self.nodes)

        self.coordinator = build.coordinator(self)

    def __str__(self):
        return self.config.sim_tag

    @property
    def index(self) -> pd.DatetimeIndex:
        """ Timesteps known to simulation. """
        return self.config.time_index

    @property
    def opt_horizon(self) -> int:
        """ We need to adapt opt horizon if remaining steps are small. """
        remaining_steps = self.config.n_time_steps - self.t
        return min(remaining_steps, self.config.optimizer_config.horizon)

    @property
    def ems_configs(self) -> dict[str, configs.EMSConfiguration]:
        return {sys_id: self.dataloader.get_ems_config(sys_id)
               for sys_id in self.node_ids}

    @property
    def state(self) -> dict[str, dict[str, float]]:
        return {str(node): node.state for node in self.nodes}

    def _build_simulation_node(self, node_id: str) -> sim_node.SimulationNode:

        ems_config = self.dataloader.get_ems_config(node_id)
        timeseries = self.dataloader.get_input_data(node_id)

        simulation_node = sim_node.SimulationNode(
            simulation_config=self.config,
            ems_config=ems_config,
            timeseries=timeseries,)

        return simulation_node

    def run(self):
        """ Run central and run local are not synchronized, yet. """

        print(f"Simulation: {self.config.sim_tag}.")

        while self.t < self.config.n_time_steps:

            start_time = time.time()

            msg = (f"\rSimulation iteration: {self.t} / "
                   f"{self.config.n_time_steps - 1}.")
            print(msg, end="", flush=True)

            signals = self.coordinator.get_signals(self.state)

            for node in self.nodes:
                node_forecast = self.forecaster.get_node_forecast(str(node))
                schedule = node.get_schedule(node_forecast, signals[str(node)])
                node.realize_schedule(schedule)

            self.results.log_iteration_time(time.time() - start_time)

            self.grid.write_loads(self.state, self.t)

            self.step()

        # Powerflow is called once with all values as it is more efficient.
        self.grid.n.lpf()

    def step(self) -> None:
        """ Progress simulation time. """

        self.coordinator.step()

        for node in self.nodes:
            node.step()

        self.t += 1
        self.forecaster.set_time_window(t=self.t, horizon=self.opt_horizon)
        
