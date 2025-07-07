from pathlib import Path

from grecco_sim.simulator import simulation
from grecco_sim.util import configs
from grecco_sim.experiment import result_table
from thesis.graph.utils import network


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
        self.ts_data = []

    def run(self):
        for sim_config in self.configs:
            print(sim_config.coordinator_name)
            sim = simulation.Simulation(sim_config)
            sim.run()
            self.ts_data.append(sim.grid)

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

        # Compare limits with transmitted power.
        capacities = network.get_p_capacity_mw(sim.grid.n)
        p_transmission = network.get_p_transmission_mw(sim.grid.n)
        congestion = (p_transmission.abs() - capacities).clip(lower=0)
        trafo_name = sim.grid.n.transformers.index[0]
        n_trafo_violations = (congestion[trafo_name] >= 0).sum()
        trafo_excess = congestion[trafo_name].sum()

        # For each feeder: compare transformer limit to reached feeder loads.

        n_feeder_violations, feeder_excess = 0, 0.
        lines = sim.grid.n.lines.copy()
        lines["feeder"] = [sim.grid.feeder_map[x] for x in lines.index]

        for feeder_idx in sim.grid.feeder:
            feeder_segments = lines[lines["feeder"] == feeder_idx]
            # For each feeder: regard the most congested segment for each t.
            feeder_congestion = congestion[feeder_segments].max(axis=1)
            n_feeder_violations += (feeder_congestion >= 0).sum()
            feeder_excess += feeder_congestion.sum()

        kpis = dict()
        kpis["n_trafo_violations"] = n_trafo_violations
        kpis["n_feeder_violations"] = n_feeder_violations
        kpis["trafo_excess"] = trafo_excess * 1000  # mW -> kW
        kpis["feeder_excess"] = feeder_excess * 1000  # mW -> kW

        return kpis
