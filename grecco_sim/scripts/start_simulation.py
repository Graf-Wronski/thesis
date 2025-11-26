from pathlib import Path
import datetime
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from grecco_sim.graph.utils.format import Format
from grecco_sim.util import configs
from grecco_sim.simulator import simulation
from grecco_sim.analysis import plotter

import pandas as pd
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

scenario = "Opfingen"  # or "SimBench"
topology = "simbench-LV-rural1--2"
data_root = Format().data_root
month, day = 2, 27
seed = 3
is_energy_price_static: bool = True  # if False, uses dynamic energy prices

if scenario == "Opfingen":
    weather_data_path = data_root / "weather" / "2023_dwd.csv"
elif scenario == "SimBench":
    weather_data_path = data_root / "weather" / "2016_dwd.csv"

fname = f"{topology}_{scenario}_{month}_{day}_shed_2050_{seed}"
grid_path = data_root / "samples" / scenario / "shed_2050" / fname
energy_price_path: Path = data_root / "market" / "energy_prices_entsoe_2023.csv"


coordination_mechanisms = ["central", "mixed", "feeder_fee", "uncoordinated"]
coordination_mechanism = coordination_mechanisms[0]  # Change index to select mechanism

result_path = (
    Path(".")
    / "results"
    # / "test"
    / "central vs uncoordinated"
    / "ev_conservative"
    # / f"{'static' if is_energy_price_static else 'dynamic'}_{coordination_mechanism}_{fname}".format()
)
if not result_path.exists():
    result_path.mkdir()


def main():

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

    market_config = configs.MarketConfiguration(
        max_market_iterations=4,
        # is_energy_price_static=is_energy_price_static,
        # energy_price_path=energy_price_path,
    )

    optimizer_config = configs.OptimizerConfiguration(horizon=12, solver_name="osqp")

    # time_index = grid.snapshots
    # start = pd.Timestamp(year=2023, month=1, day=13, hour=0)
    # end = pd.Timestamp(year=2023, month=1, day=13, hour=23)
    # time_index = pd.date_range(start=start, end=end, freq="15min")
    dates = pd.read_csv(grid_path / "snapshots.csv", index_col=0)["snapshot"]
    time_index = pd.date_range(start=dates.iloc[0], end=dates.iloc[-1], freq="15min")

    simulation_config = configs.SimulationConfiguration(
        feeder_lim=0.015,
        transformer_lim=0.025,
        time_index=time_index,
        coordinator_name=coordination_mechanism,
        sim_tag=f"{coordination_mechanism}",
        use_pv=True,
        use_heatpumps=True,
        use_ev=True,
        use_batteries=True,
        output_dir=Path("default") / timestamp,
        optimizer_config=optimizer_config,
        grid_data_path=grid_path,
        weather_data_path=weather_data_path,
        market_config=market_config,
        heat_pump_model="continous",
        is_energy_price_static=is_energy_price_static,
        energy_price_path=energy_price_path,
    )

    sim = simulation.Simulation(simulation_config)
    sim.run()
    sim.write(result_path)

    simulation_plotter = plotter.Plotter(sim)
    simulation_plotter.plot()


if __name__ == "__main__":
    main()
