import datetime
import pytz

from grecco_sim.util import configs
from pathlib import Path

from grecco_sim.experiment.experiment import Experiment

data_root = Path("/home/carl-wanninger/data")
grid_path = data_root / "sample_grids" / "lv_minimal_1"
weather_data_path =  data_root / "weather" / "test" / "pvgis_2016_00.csv"

def main():
    simulation_configs = []
    exp_dir = Path("experiments") / "results" / "mini"

    for mechanism in ["transformer_fee", "feeder_fee", "central"]:
        sim_tag = f"{mechanism}_test"
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
        sim_dir =  Path("experiments") / "results" / f"{sim_tag}"

        for max_market_iterations in [2]:

            market_config = configs.MarketConfiguration(
                max_market_iterations=max_market_iterations)

            for horizon in [10, 20]:

                optimizer_config = configs.OptimizerConfiguration(
                    horizon=horizon,
                    alpha=1.,  # Grid-fee scales with alpha (and congestion amount).
                    solver_name="gurobi",
                    forecast_type="perfect",
                    slack_penalty_thermal=1000.0)

                simulation_config = configs.SimulationConfiguration(
                    n_time_steps=16,
                    step_size=datetime.timedelta(minutes=15),
                    start_time=datetime.datetime(2016, 10, 12, tzinfo=pytz.utc),
                    coordinator_name=mechanism,
                    sim_tag=sim_tag,
                    use_pv=True,
                    use_heatpumps=False,
                    use_ev=True,
                    use_batteries=False,
                    output_dir=sim_dir,
                    optimizer_config=optimizer_config,
                    grid_data_path=grid_path,
                    weather_data_path=weather_data_path,
                    market_config=market_config)

                simulation_configs.append(simulation_config)

    experiment = Experiment(simulation_configs, exp_dir)
    experiment.run()

if __name__ == "__main__":
    main()
