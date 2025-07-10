import datetime
import pytz

from grecco_sim.util import configs
from pathlib import Path

from grecco_sim.experiment.experiment import Experiment


data_root = Path("/home/carl-wanninger/data")
grid_path = data_root / "sample_grids" / "lv_minimal_1"
weather_data_path =  data_root / "weather" / "test" / "pvgis_2016_00.csv"


def experiment_0() -> Experiment:
    simulation_configs = []
    exp_dir = Path("experiments") / "results" / "ex_00"

    for mechanism in ["transformer_fee"]:
        sim_tag = f"{mechanism}_test"
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
        sim_dir =  Path("experiments") / "results" / f"{sim_tag}"

        for max_market_iterations in [2]:

            market_config = configs.MarketConfiguration(
                max_market_iterations=max_market_iterations)

            for horizon in [2]:

                optimizer_config = configs.OptimizerConfiguration(
                    horizon=horizon,
                    alpha=1.,  # Grid-fee scales with alpha (and congestion amount).
                    solver_name="osqp",
                    forecast_type="perfect",
                    slack_penalty_thermal=1000.0)

                simulation_config = configs.SimulationConfiguration(
                    n_time_steps=15,
                    step_size=datetime.timedelta(minutes=15),
                    start_time=datetime.datetime(2016, 10, 12, tzinfo=pytz.utc),
                    coordinator_name=mechanism,
                    sim_tag=sim_tag,
                    use_pv=True,
                    use_heatpumps=True,
                    use_ev=False,
                    use_batteries=False,
                    output_dir=sim_dir,
                    optimizer_config=optimizer_config,
                    grid_data_path=grid_path,
                    weather_data_path=weather_data_path,
                    market_config=market_config)

                simulation_configs.append(simulation_config)

    return Experiment(simulation_configs, exp_dir)


def experiment_1() -> Experiment:

    simulation_configs = []
    exp_dir = Path("experiments") / "results" / "ex_00"

    coordination_mechanisms = ["feeder_fee", "central", "transformer_fee",
                               "uncoordinated"]
    for mechanism in coordination_mechanisms:
        sim_tag = f"{mechanism}_test"
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
        sim_dir = Path("experiments") / "results" / f"{sim_tag}"

        for max_market_iterations in [2]:

            market_config = configs.MarketConfiguration(
                max_market_iterations=max_market_iterations)

            for horizon in [5]:
                optimizer_config = configs.OptimizerConfiguration(
                    horizon=horizon,
                    alpha=1.,
                    # Grid-fee scales with alpha (and congestion amount).
                    solver_name="osqp",
                    forecast_type="perfect",
                    slack_penalty_thermal=1000.0)

                simulation_config = configs.SimulationConfiguration(
                    n_time_steps=50,
                    step_size=datetime.timedelta(minutes=15),
                    start_time=datetime.datetime(2016, 10, 12,
                                                 tzinfo=pytz.utc),
                    coordinator_name=mechanism,
                    sim_tag=sim_tag,
                    use_pv=True,
                    use_heatpumps=True,
                    heat_pump_model="continous",
                    use_ev=False,
                    use_batteries=False,
                    output_dir=sim_dir,
                    optimizer_config=optimizer_config,
                    grid_data_path=grid_path,
                    weather_data_path=weather_data_path,
                    market_config=market_config)

                simulation_configs.append(simulation_config)

    return Experiment(simulation_configs, exp_dir)