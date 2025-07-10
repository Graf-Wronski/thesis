import pathlib
import datetime
import pytz

from grecco_sim.analysis import plotter
from grecco_sim.util import configs
from grecco_sim.simulator import simulation

import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

data_root = pathlib.Path("/home/carl-wanninger/data/")
grid_path = data_root / "samples" / "Opfingen" / "Opfingen_20250705_184425_0"
weather_data_path =  data_root / "weather" / "2023_dwd.csv"

def main():

    # coordination_mechanism = "central"
    coordination_mechanism = "transformer_fee"
    # coordination_mechanism = "feeder_fee"
    # coordination_mechanism = "none"

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

    market_config = configs.MarketConfiguration(max_market_iterations=2)

    optimizer_config = configs.OptimizerConfiguration(
        horizon=12,
        solver_name="gurobi")

    start = pd.Timestamp(year=2023, month=1, day=13, hour=0, tzinfo=pytz.utc)
    end = pd.Timestamp(year=2023, month=1, day=13, hour=23, tzinfo=pytz.utc)
    time_index = pd.date_range(start=start, end=end, freq="15min")

    simulation_config = configs.SimulationConfiguration(
        time_index=time_index,
        coordinator_name=coordination_mechanism,
        sim_tag=f"{coordination_mechanism}",
        use_pv=False,
        use_heatpumps=True,
        use_ev=False,
        use_batteries=False,
        output_dir=pathlib.Path("default") / timestamp,
        optimizer_config=optimizer_config,
        grid_data_path=grid_path,
        weather_data_path=weather_data_path,
        market_config=market_config,
        heat_pump_model="continous")

    sim = simulation.Simulation(simulation_config)
    sim.run()

    simulation_plotter = plotter.Plotter(sim)
    simulation_plotter.plot()


if __name__ == "__main__":
    main()
