from typing import Any


class Coordinator:
    def __init__(self, grecco_sim: Any):
        self.sim_config = grecco_sim.config
        self.sim_grid =  grecco_sim.grid
        self.sim_result =  grecco_sim.results
        self.sim_nodes =  grecco_sim.nodes
        self.ems_configs = grecco_sim.ems_configs
        self.forecaster = grecco_sim.forecaster

        self.t = 0

    def step(self):
        self.t += 1

    @property
    def horizon(self) -> int:
        remaining_steps = self.sim_config.n_time_steps - self.t
        return min(remaining_steps, self.sim_config.optimizer_config.horizon)