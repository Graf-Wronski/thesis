import pathlib
import datetime
import pytz

from grecco_sim.analysis import plotter
from grecco_sim.util import configs
from grecco_sim.simulator import simulation

# data_dir =  pathlib.Path(__file__).parents[2] / "data"
# grid_path = data_dir / "Opfingen_Profiles_2023" / "data_Rebecca_07_03_2025"
# weather_data_path = data_dir / "Opfingen_Profiles_2023" / "pvgis_2023_01.csv"

data_root = pathlib.Path("/home/carl-wanninger/data/")
grid_path = data_root / "sample_grids" / "lv_minimal_1"
weather_data_path =  data_root / "weather" / "test" / "pvgis_2016_00.csv"

def main():

    # coordination_mechanism = "central"
    # coordination_mechanism = "transformer_fee"
    coordination_mechanism = "feeder_fee"
    # coordination_mechanism = "none"

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

    market_config = configs.MarketConfiguration(max_market_iterations=2)

    optimizer_config = configs.OptimizerConfiguration(
        horizon=20,
        alpha=1.,  # Grid-fee scales with alpha (and congestion amount).
        solver_name="osqp",
        forecast_type="perfect",
        slack_penalty_thermal=1000.0)

    simulation_config = configs.SimulationConfiguration(
        n_time_steps=96,
        step_size=datetime.timedelta(minutes=15),
        start_time=datetime.datetime(2016, 10, 12, tzinfo=pytz.utc),
        coordinator_name=coordination_mechanism,
        sim_tag=f"{coordination_mechanism}",
        use_pv=True,
        use_heatpumps=True,
        use_ev=False,
        use_batteries=True,
        output_dir=pathlib.Path("default") / timestamp,
        optimizer_config=optimizer_config,
        grid_data_path=grid_path,
        weather_data_path=weather_data_path,
        market_config=market_config)

    sim = simulation.Simulation(simulation_config)
    sim.run()

    simulation_plotter = plotter.Plotter(sim)
    simulation_plotter.plot()


if __name__ == "__main__":
    main()
