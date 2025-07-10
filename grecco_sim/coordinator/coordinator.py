from typing import Any

import numpy as np
import pandas as pd

from grecco_sim.util import type_defs
from grecco_sim.graph.utils import network


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

    def get_congestion(
            self,
            schedules: dict[str, type_defs.Schedule]) \
            -> tuple[pd.DataFrame, pd.DataFrame]:

        """ Project congestion for given schedules.

        Congestion is the load amount above transmission capacities.
        Signed congestion is congestion that regards direction of flow.
        +: Power flows downstream, -: Power flows upstream. """

        # Use a network copy to avoid side effects.
        n = self.sim_grid.n.copy()
        n.set_snapshots([x for x in range(self.horizon)])

        # Replace loads with aggregates (P grid) to simplify power flow.
        for load in n.loads.index:
            n.remove("Load", name=load)
        for sys_id, schedule in schedules.items():
            bus_id = sys_id.split('_')[-1]
            identifier = {"name": f"P grid at {bus_id}", "bus": bus_id}
            n.add(class_name="Load", p_set=schedule.p_grid, **identifier)

        n.lpf(snapshots=[x for x in range(self.horizon)])

        p_transmission = network.get_p_transmission_mw(n)
        congestion = (p_transmission - self.sim_grid.capacities).clip(0)
        signed_congestion = congestion * np.sign(n.lines_t["p0"])

        return congestion, signed_congestion