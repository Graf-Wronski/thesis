from thesis.graph import complex_network_analysis
from pathlib import Path

from thesis.graph.utils import config

import datetime
import pytz

from grecco_sim.util import configs
from grecco_sim.simulator import simulation

import pandas as pd
from warnings import simplefilter


simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

data_root = Path("/home/carl-wanninger/data/")
grid_path = data_root / "samples" / "Opfingen" / "six-bus-test-grid_0"
weather_data_path =  data_root / "weather" / "2023_dwd.csv"

coordination_mechanism = "transformer_fee"
market_config = configs.MarketConfiguration(max_market_iterations=2)
optimizer_config = configs.OptimizerConfiguration(
    horizon=12,
    solver_name="osqp")

start = pd.Timestamp(year=2023, month=1, day=13, hour=0)
end = pd.Timestamp(year=2023, month=1, day=13, hour=23)
time_index = pd.date_range(start=start, end=end, freq="15min")

simulation_config = configs.SimulationConfiguration(
    time_index=time_index,
    coordinator_name=coordination_mechanism,
    sim_tag=f"{coordination_mechanism}",
    use_pv=False,
    use_heatpumps=True,
    use_ev=False,
    use_batteries=False,
    output_dir=Path("default") / timestamp,
    optimizer_config=optimizer_config,
    grid_data_path=grid_path,
    weather_data_path=weather_data_path,
    market_config=market_config,
    heat_pump_model="continous")

if __name__ == "__main__":
    sim = simulation.Simulation(simulation_config)
    sim.run()

    # Analyze graph structure.
    pr_config = config.PushRelabelConfiguration(
        sim_config=simulation_config)
    cna = complex_network_analysis.ComplexNetworkAnalysis(sim.grid.n, pr_config)
    graph_congestion_table = cna.run()
    print("Graph")
    print(graph_congestion_table.sum())