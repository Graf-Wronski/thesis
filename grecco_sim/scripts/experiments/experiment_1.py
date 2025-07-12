import sys
from copy import copy

from pathlib import Path
import datetime

import yaml

import timeit
from grecco_sim.graph.utils.format import Format
from grecco_sim.util import configs
from grecco_sim.simulator import simulation
from sklearn.model_selection import ParameterGrid

import pandas as pd
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

data_root = Format().data_root
output_root = Format().output_root / "experiment_1"

coordination_mechanism = "transformer_fee"

scenario = "Opfingen"
load_distribution = "shed_2050"

param_grid = {
    "solver": ["gurobi", "osqp"],
    "seed": [3, 5, 17, 257, 65537],
    "topology": ["simbench-LV-rural1--2"],
    "heat_pump_model": ["discrete", "continous"],
    "horizon": [5, 10, 15, 20],
    "date": [(2, 27), (8, 11), (8, 30), (10, 11)]}

param_grid = list(ParameterGrid(param_grid))


def main(param_index : int):
    params = param_grid[param_index]
    meta = copy(params)

    dir_name = f"{param_index}"
    for val in params.values:
        dir_name += val
    result_path = output_root / dir_name

    seed = params["seed"]
    topology = params["topology"]
    hp_model = params["heat_pump_model"]
    solver = params["solver"]
    horizon = params["horizon"]
    month, day = params["date"]

    grid = f"{topology}_{scenario}_{month}_{day}_{load_distribution}_{seed}"
    grid_path = data_root / "samples" / scenario / load_distribution / grid

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

    market_config = configs.MarketConfiguration(max_market_iterations=2)

    optimizer_config = configs.OptimizerConfiguration(
        horizon=horizon,
        solver_name=solver)

    dates = pd.read_csv(grid_path / "snapshots.csv", index_col=0)["snapshot"]
    start, end = dates.iloc[0], dates.iloc[-1]
    time_index = pd.date_range(start=start, end=end, freq="15min")
    weather_data_path = data_root / "weather" / f"{time_index[0].year}_dwd.csv"

    meta["start"], meta["end"] = start, end

    feeder_lim, transformer_lim = 0.017, 0.025
    meta["feeder_lim"] = feeder_lim, meta["transformer_lim"] = transformer_lim

    simulation_config = configs.SimulationConfiguration(
        feeder_lim = feeder_lim,
        transformer_lim = transformer_lim,
        time_index=time_index,
        coordinator_name=coordination_mechanism,
        sim_tag=f"{coordination_mechanism}",
        output_dir=result_path / timestamp,
        optimizer_config=optimizer_config,
        grid_data_path=grid_path,
        weather_data_path=weather_data_path,
        market_config=market_config,
        heat_pump_model=hp_model)

    sim = simulation.Simulation(simulation_config)
    meta["start"] = timeit.default_timer()
    sim.run()
    meta["end"] = timeit.default_timer()
    meta["runtime"] = meta["end"] - meta["start"]

    sim.write(result_path)
    with open('meta.yaml', 'w+') as ff:
        yaml.dump(meta, ff)

if __name__ == "__main__":
    index = int(sys.argv[1])
    main(index)