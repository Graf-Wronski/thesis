from pathlib import Path

import numpy as np

from grecco_sim.simulator import simulation
from grecco_sim.util import configs
from thesis.experiment import result_table


class Experiment:
    def __init__(
            self,
            sim_configs: list[configs.SimulationConfiguration],
            result_dir: Path):

        self.configs = sim_configs
        self.result_dir = result_dir
        self.result_table = result_table.ResultTable(
            path=result_dir / "kpis.pkl",
            overwrite=False)

    def run(self):
        for sim_config in self.configs:
            sim = simulation.Simulation(sim_config)
            sim.run()

            result = dict()
            result.update(self.extract_kpis(sim))
            result.update(sim.config.as_dict())

            self.result_table.add_result(result)

    @staticmethod
    def extract_kpis(sim: simulation.Simulation) -> dict:
        """ Calculate KPIs that monitor congestion.

        sim: A past simulation.

        We are interested in the following KPIs:
            - n_trafo_violations: Number of simulation steps with congested
                transformer.
            - n_feeder_violations (int): Number of simulation steps with
                congested feeder. If multiple feeder are congested the
                respective time step is counted multiple times.
                (Maxmial value thus is num_feeders x num_steps)
            - trafo_excess (float): Cummulative load excess of
                transformer (summed over time steps).
            - feeder_excess (float): Cummulative load excess of
                feeders (summed over time steps and feeders).

        """

        # Compare transformer limit to reached transformer loads.
        trafo_lim = sim.grid.feeder_p_lim

        # Compute p_trafo as the absolute maximum between MV and LV side.
        p0 = np.abs(sim.grid.n.transformers_t["p0"].values)
        p1 = np.abs(sim.grid.n.transformers_t["p1"].values)
        p_trafo = np.max(np.concatenate([p0, p1]), axis=1)
        p_trafo *= 1000  # mW -> kW
        trafo_violations = trafo_lim <= p_trafo
        n_trafo_violations = np.sum(trafo_violations)
        trafo_excess = ((np.abs(p_trafo) - trafo_lim) * trafo_violations).sum()

        # For each feeder: compare transformer limit to reached feeder loads.
        n_feeder_violations, feeder_excess = 0, 0.

        lines = sim.grid.n.lines.copy()
        lines["feeder"] = [sim.grid.feeder_map[x] for x in lines.index]

        for feeder_idx in sim.grid.feeder:
            feeder_segments = lines[lines["feeder"] == feeder_idx]
            p0 = np.abs(sim.grid.n.lines_t["p0"][feeder_segments.index].values)
            p1 = np.abs(sim.grid.n.lines_t["p1"][feeder_segments.index].values)

            # Max over p0, p1 and all segments.
            p_segment = np.max([p0, p1], axis=0)
            p_feeder = np.max(p_segment, axis=1)
            p_feeder *= 1000  # mW -> kW

            # ToDo: Actual limit should be stored in line s_nom.
            feeder_lim = sim.grid.feeder_p_lim
            feeder_violations = feeder_lim <= p_feeder
            n_feeder_violations += np.sum(feeder_violations)
            feeder_excess += ((np.abs(p_feeder) - feeder_lim) *
                              feeder_violations).sum()
            
        kpis = dict()
        kpis["n_trafo_violations"] = n_trafo_violations
        kpis["n_feeder_violations"] = n_feeder_violations
        kpis["trafo_excess"] = trafo_excess
        kpis["feeder_excess"] = feeder_excess

        return kpis
